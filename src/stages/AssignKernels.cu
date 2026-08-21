#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>

#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr std::size_t MaximumAssignmentsPerLaunch = 8;
constexpr std::size_t MaximumInstructionsPerLaunch = 96;
constexpr std::size_t MaximumStackDepth = 32;

class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

struct DeviceInstruction
{
    const void* source = nullptr;
    double immediate = 0.0;
    double coordinateScale = 1.0;
    double coordinateOffset = 0.0;
    DimensionType sourceType = DimensionType::None;
    ExpressionOp op = ExpressionOp::PushConstant;
    int coordinateAxis = -1;
};

struct DeviceAssignment
{
    void* destination = nullptr;
    DimensionType destinationType = DimensionType::None;
    std::uint16_t valueBegin = 0;
    std::uint16_t valueCount = 0;
    std::uint16_t conditionBegin = 0;
    std::uint16_t conditionCount = 0;
};

struct DeviceProgram
{
    DeviceInstruction instructions[MaximumInstructionsPerLaunch];
    DeviceAssignment assignments[MaximumAssignmentsPerLaunch];
    std::uint16_t instructionCount = 0;
    std::uint16_t assignmentCount = 0;
};

__device__ double loadPhysical(const void* data, DimensionType type,
                               std::size_t index)
{
    switch (type)
    {
    case DimensionType::Signed8:
        return static_cast<const std::int8_t*>(data)[index];
    case DimensionType::Signed16:
        return static_cast<const std::int16_t*>(data)[index];
    case DimensionType::Signed32:
        return static_cast<const std::int32_t*>(data)[index];
    case DimensionType::Signed64:
        return static_cast<double>(
            static_cast<const std::int64_t*>(data)[index]);
    case DimensionType::Unsigned8:
        return static_cast<const std::uint8_t*>(data)[index];
    case DimensionType::Unsigned16:
        return static_cast<const std::uint16_t*>(data)[index];
    case DimensionType::Unsigned32:
        return static_cast<const std::uint32_t*>(data)[index];
    case DimensionType::Unsigned64:
        return static_cast<double>(
            static_cast<const std::uint64_t*>(data)[index]);
    case DimensionType::Float:
        return static_cast<const float*>(data)[index];
    case DimensionType::Double:
        return static_cast<const double*>(data)[index];
    case DimensionType::None:
        return 0.0;
    }
    return 0.0;
}

__device__ double symmetricRound(double value)
{
    return value > 0.0 ? floor(value + 0.5) : ceil(value - 0.5);
}

template <typename T>
__device__ void storeIntegral(void* data, std::size_t index, double value,
                              double lowest, double maximum)
{
    const double rounded = symmetricRound(value);
    if (rounded >= lowest && rounded <= maximum)
        static_cast<T*>(data)[index] = static_cast<T>(rounded);
}

__device__ void storePhysical(void* data, DimensionType type, std::size_t index,
                              double value)
{
    switch (type)
    {
    case DimensionType::Signed8:
        storeIntegral<std::int8_t>(data, index, value, -128.0, 127.0);
        return;
    case DimensionType::Signed16:
        storeIntegral<std::int16_t>(data, index, value, -32768.0, 32767.0);
        return;
    case DimensionType::Signed32:
        storeIntegral<std::int32_t>(data, index, value, -2147483648.0,
                                    2147483647.0);
        return;
    case DimensionType::Signed64:
        storeIntegral<std::int64_t>(data, index, value, -0x1p63, 0x1p63);
        return;
    case DimensionType::Unsigned8:
        storeIntegral<std::uint8_t>(data, index, value, 0.0, 255.0);
        return;
    case DimensionType::Unsigned16:
        storeIntegral<std::uint16_t>(data, index, value, 0.0, 65535.0);
        return;
    case DimensionType::Unsigned32:
        storeIntegral<std::uint32_t>(data, index, value, 0.0, 4294967295.0);
        return;
    case DimensionType::Unsigned64:
        storeIntegral<std::uint64_t>(data, index, value, 0.0, 0x1p64);
        return;
    case DimensionType::Float:
        if (isnan(value) ||
            (value >= -0x1.fffffep+127 && value <= 0x1.fffffep+127))
            static_cast<float*>(data)[index] = static_cast<float>(value);
        return;
    case DimensionType::Double:
        static_cast<double*>(data)[index] = value;
        return;
    case DimensionType::None:
        return;
    }
}

__device__ double quietNan()
{
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ double evaluate(const DeviceProgram& program, std::uint16_t begin,
                           std::uint16_t count, std::size_t point,
                           double (&stack)[MaximumStackDepth])
{
    std::size_t size = 0;
    const std::size_t end = static_cast<std::size_t>(begin) + count;
    for (std::size_t position = begin; position < end; ++position)
    {
        const DeviceInstruction& instruction = program.instructions[position];
        switch (instruction.op)
        {
        case ExpressionOp::PushConstant:
            stack[size++] = instruction.immediate;
            break;
        case ExpressionOp::PushFalse:
            stack[size++] = 0.0;
            break;
        case ExpressionOp::PushTrue:
            stack[size++] = 1.0;
            break;
        case ExpressionOp::LoadDimension:
        {
            double value =
                loadPhysical(instruction.source, instruction.sourceType, point);
            if (instruction.coordinateAxis >= 0)
                value = __dadd_rn(__dmul_rn(value, instruction.coordinateScale),
                                  instruction.coordinateOffset);
            stack[size++] = value;
            break;
        }
        case ExpressionOp::Add:
            stack[size - 2U] = __dadd_rn(stack[size - 2U], stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Subtract:
            stack[size - 2U] = __dsub_rn(stack[size - 2U], stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Multiply:
            stack[size - 2U] = __dmul_rn(stack[size - 2U], stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Divide:
            stack[size - 2U] =
                stack[size - 1U] == 0.0
                    ? quietNan()
                    : __ddiv_rn(stack[size - 2U], stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Negative:
        {
            const auto bits = static_cast<unsigned long long>(
                __double_as_longlong(stack[size - 1U]));
            stack[size - 1U] = __longlong_as_double(
                static_cast<long long>(bits ^ 0x8000000000000000ULL));
            break;
        }
        case ExpressionOp::Equal:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] == stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::NotEqual:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] != stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Greater:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] > stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::GreaterEqual:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] >= stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::Less:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] < stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::LessEqual:
            stack[size - 2U] =
                static_cast<double>(stack[size - 2U] <= stack[size - 1U]);
            --size;
            break;
        case ExpressionOp::LogicalNot:
            stack[size - 1U] = static_cast<double>(!stack[size - 1U]);
            break;
        case ExpressionOp::LogicalAnd:
            stack[size - 2U] = static_cast<double>(stack[size - 2U] != 0.0 &&
                                                   stack[size - 1U] != 0.0);
            --size;
            break;
        case ExpressionOp::LogicalOr:
            stack[size - 2U] = static_cast<double>(stack[size - 2U] != 0.0 ||
                                                   stack[size - 1U] != 0.0);
            --size;
            break;
        case ExpressionOp::IsNan:
            stack[size - 1U] = static_cast<double>(isnan(stack[size - 1U]));
            break;
        case ExpressionOp::IsMaximum:
            stack[size - 1U] = static_cast<double>(stack[size - 1U] ==
                                                   0x1.fffffffffffffp+1023);
            break;
        case ExpressionOp::IsMinimum:
            stack[size - 1U] = static_cast<double>(stack[size - 1U] ==
                                                   -0x1.fffffffffffffp+1023);
            break;
        case ExpressionOp::Floor:
        case ExpressionOp::Ceil:
        case ExpressionOp::Round:
        case ExpressionOp::Absolute:
        case ExpressionOp::SquareRoot:
        case ExpressionOp::Sine:
        case ExpressionOp::Cosine:
        case ExpressionOp::Tangent:
        case ExpressionOp::ArcSine:
        case ExpressionOp::ArcCosine:
        case ExpressionOp::ArcTangent:
        case ExpressionOp::HyperbolicSine:
        case ExpressionOp::HyperbolicCosine:
        case ExpressionOp::HyperbolicTangent:
        case ExpressionOp::InverseHyperbolicSine:
        case ExpressionOp::InverseHyperbolicCosine:
        case ExpressionOp::NaturalLog:
        case ExpressionOp::Log2:
        case ExpressionOp::Log10:
        case ExpressionOp::Exponential:
        case ExpressionOp::Exponential2:
            return quietNan();
        }
    }
    return size == 1 ? stack[0] : quietNan();
}

__global__ void assignKernel(std::size_t pointCount, DeviceProgram program)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        double stack[MaximumStackDepth];
        for (std::size_t index = 0; index < program.assignmentCount; ++index)
        {
            const DeviceAssignment& assignment = program.assignments[index];
            if (assignment.valueCount == 0)
                continue;
            if (assignment.conditionCount != 0 &&
                evaluate(program, assignment.conditionBegin,
                         assignment.conditionCount, point, stack) == 0.0)
                continue;
            const double value = evaluate(program, assignment.valueBegin,
                                          assignment.valueCount, point, stack);
            storePhysical(assignment.destination, assignment.destinationType,
                          point, value);
        }
    }
}

__global__ void predicateKernel(std::size_t pointCount, DeviceProgram program,
                                std::uint16_t begin, std::uint16_t count,
                                std::uint8_t* keep)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        double stack[MaximumStackDepth];
        keep[point] = static_cast<std::uint8_t>(
            evaluate(program, begin, count, point, stack) != 0.0);
    }
}

int gridSize(std::size_t count)
{
    const std::size_t blocks =
        count / BlockSize + static_cast<std::size_t>(count % BlockSize != 0);
    return static_cast<int>(
        std::min<std::size_t>(blocks, std::numeric_limits<int>::max()));
}

int coordinateAxis(DimensionId id) noexcept
{
    if (id == DimensionId(StandardDimension::X))
        return 0;
    if (id == DimensionId(StandardDimension::Y))
        return 1;
    if (id == DimensionId(StandardDimension::Z))
        return 2;
    return -1;
}

void appendExpression(DeviceProgram& result,
                      const CompiledExpression& expression,
                      const PointBatch& batch, std::uint16_t& begin,
                      std::uint16_t& count)
{
    begin = result.instructionCount;
    count = static_cast<std::uint16_t>(expression.instructions.size());
    for (const ExpressionInstruction& instruction : expression.instructions)
    {
        DeviceInstruction& target =
            result.instructions[result.instructionCount++];
        target.op = instruction.op;
        target.immediate = instruction.immediate;
        if (instruction.op == ExpressionOp::LoadDimension)
        {
            target.source = batch.rawData(instruction.dimension);
            target.sourceType =
                batch.columnInfo(instruction.dimension).physicalType;
            const int axis = coordinateAxis(instruction.dimension);
            if (axis >= 0 && target.sourceType == DimensionType::Signed32)
            {
                target.coordinateAxis = axis;
                target.coordinateScale =
                    batch.coordinateEncoding()
                        .scale()[static_cast<std::size_t>(axis)];
                target.coordinateOffset =
                    batch.coordinateEncoding()
                        .offset()[static_cast<std::size_t>(axis)];
            }
        }
    }
}
} // unnamed namespace

void executeAssignDevice(PointBatch& batch, const AssignProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA assignment requires a device "
                                    "PointBatch");
    if (!assignSupportsExactDevice(batch, program))
        throw std::invalid_argument(
            "assignment program is outside the exact CUDA expression "
            "envelope");

    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    for (const PointAssignment& assignment : program.assignments)
    {
        const bool destinationPresent = batch.has(assignment.destination);
        if (!destinationPresent)
            batch.materialize(assignment.destination);
        if (assignment.destinationCreated || !destinationPresent)
        {
            const ColumnInfo& destination =
                batch.columnInfo(assignment.destination);
            PDG_CUDA_CHECK(cudaMemsetAsync(
                batch.rawData(assignment.destination), 0,
                batch.size() * dimensionTypeSize(destination.physicalType),
                stream));
        }
    }
    for (DimensionId source : program.reads)
        if (!batch.has(source))
            throw std::invalid_argument(
                "assignment source column is not materialized");
    if (batch.size() == 0 || program.assignments.empty())
        return;

    NvtxRange range("pdg::filters.assign");
    std::size_t begin = 0;
    while (begin < program.assignments.size())
    {
        DeviceProgram deviceProgram;
        while (begin < program.assignments.size() &&
               deviceProgram.assignmentCount < MaximumAssignmentsPerLaunch)
        {
            const PointAssignment& assignment = program.assignments[begin];
            const std::size_t instructionCount =
                assignment.value.instructions.size() +
                assignment.condition.instructions.size();
            if (static_cast<std::size_t>(deviceProgram.instructionCount) +
                    instructionCount >
                MaximumInstructionsPerLaunch)
                break;

            DeviceAssignment& target =
                deviceProgram.assignments[deviceProgram.assignmentCount++];
            target.destination = batch.rawData(assignment.destination);
            target.destinationType =
                batch.columnInfo(assignment.destination).physicalType;
            appendExpression(deviceProgram, assignment.value, batch,
                             target.valueBegin, target.valueCount);
            appendExpression(deviceProgram, assignment.condition, batch,
                             target.conditionBegin, target.conditionCount);
            ++begin;
        }
        if (deviceProgram.assignmentCount == 0)
            throw std::invalid_argument(
                "assignment exceeds the exact CUDA launch limits");
        assignKernel<<<gridSize(batch.size()), BlockSize, 0, stream>>>(
            batch.size(), deviceProgram);
        PDG_CUDA_CHECK(cudaGetLastError());
    }
}

void evaluatePredicateDevice(PointBatch& batch, const PredicateProgram& program,
                             std::uint8_t* keep)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA predicate evaluation requires a device PointBatch");
    if (!predicateSupportsExactDevice(batch, program))
        throw std::invalid_argument(
            "predicate is outside the exact CUDA expression envelope");
    for (DimensionId source : program.reads)
        if (!batch.has(source))
            throw std::invalid_argument(
                "predicate source column is not materialized");
    if (!keep && batch.size())
        throw std::invalid_argument("predicate output is null");
    if (batch.size() == 0)
        return;

    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    DeviceProgram deviceProgram;
    std::uint16_t begin = 0;
    std::uint16_t count = 0;
    appendExpression(deviceProgram, program.expression, batch, begin, count);
    NvtxRange range("pdg::filters.expression::predicate");
    predicateKernel<<<gridSize(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), deviceProgram, begin, count, keep);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
