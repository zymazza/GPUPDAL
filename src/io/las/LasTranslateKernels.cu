#include <pdg/Compaction.hpp>
#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/io/LasCuda.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslateCuda.hpp>

#include <cuda.h>
#include <nvrtc.h>

#include <dlfcn.h>

#include <cstdio>
#include <map>
#include <mutex>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace pdg::las
{

namespace
{
static_assert(std::endian::native == std::endian::little,
              "PDG v1 supports little-endian x86_64 hosts only");
static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));

constexpr std::size_t OutputHeaderBytes = 375;
constexpr std::size_t OutputPointBytes = 36;
constexpr double OutputScale = 0.01;
constexpr int BlockSize = 256;
// The staged translate kernels use wider blocks: per-point work is short, so
// amortizing the stage barriers over more points wins as long as the staging
// spans stay inside the default dynamic shared budget.
constexpr int StagedBlockSize = 256;
constexpr std::size_t MaximumFusedDimensions = 8;
constexpr std::size_t MaximumFusedAssignments = 8;
constexpr std::size_t MaximumFusedInstructions = 96;
constexpr std::size_t MaximumFusedStackDepth = 32;

__host__ __device__ constexpr std::uint32_t
standardId(StandardDimension dimension) noexcept
{
    return static_cast<std::uint16_t>(dimension);
}

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

struct RawSummary
{
    std::int32_t minimum[3];
    std::int32_t maximum[3];
    unsigned long long returns[15];
};

struct FusedDimension
{
    std::uint32_t id = 0;
    DimensionType type = DimensionType::None;
    bool decode = false;
    bool pack = false;
    bool initializeZero = false;
};

struct FusedInstruction
{
    double immediate = 0.0;
    ExpressionOp op = ExpressionOp::PushConstant;
    std::uint16_t dimension = 0;
};

struct FusedAssignment
{
    std::uint16_t destination = 0;
    std::uint16_t valueBegin = 0;
    std::uint16_t valueCount = 0;
    std::uint16_t conditionBegin = 0;
    std::uint16_t conditionCount = 0;
};

struct FusedPointProgram
{
    FusedDimension dimensions[MaximumFusedDimensions]{};
    FusedInstruction instructions[MaximumFusedInstructions]{};
    FusedAssignment assignments[MaximumFusedAssignments]{};
    std::uint16_t dimensionCount = 0;
    std::uint16_t instructionCount = 0;
    std::uint16_t assignmentCount = 0;
};

struct HostSummary
{
    bool populated = false;
    std::array<double, 3> minimum{};
    std::array<double, 3> maximum{};
    std::array<std::uint64_t, 15> returns{};

    void merge(const RawSummary& other, bool hasPoints)
    {
        if (hasPoints && !populated)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                minimum[axis] =
                    static_cast<double>(other.minimum[axis]) * OutputScale;
                maximum[axis] =
                    static_cast<double>(other.maximum[axis]) * OutputScale;
            }
            populated = true;
        }
        else if (hasPoints)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                minimum[axis] = std::min(
                    minimum[axis],
                    static_cast<double>(other.minimum[axis]) * OutputScale);
                maximum[axis] = std::max(
                    maximum[axis],
                    static_cast<double>(other.maximum[axis]) * OutputScale);
            }
        }
        for (std::size_t index = 0; index < returns.size(); ++index)
            returns[index] += static_cast<std::uint64_t>(other.returns[index]);
    }
};

template <typename T>
void write(std::span<std::byte> bytes, std::size_t offset, T value)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw Error("CUDA LAS translation exceeded its output buffer");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void writeString(std::span<std::byte> bytes, std::size_t offset,
                 std::size_t length, std::string_view value)
{
    if (offset > bytes.size() || bytes.size() - offset < length)
        throw Error("CUDA LAS translation exceeded its output header");
    const std::size_t copied = std::min(length, value.size());
    std::memcpy(bytes.data() + offset, value.data(), copied);
}

void writeHeader(std::span<std::byte> output, std::uint64_t pointCount,
                 const HostSummary& summary,
                 const DefaultTranslationMetadata& metadata)
{
    writeString(output, 0, 4, "LASF");
    write<std::uint16_t>(output, 6, 1U << 4U);
    write<std::uint8_t>(output, 24, 1);
    write<std::uint8_t>(output, 25, 4);
    writeString(output, 26, 32, "PDAL");
    writeString(output, 58, 32, metadata.softwareId);
    write(output, 90, metadata.creationDayOfYear);
    write(output, 92, metadata.creationYear);
    write<std::uint16_t>(output, 94, OutputHeaderBytes);
    write<std::uint32_t>(output, 96, OutputHeaderBytes);
    write<std::uint8_t>(output, 104, 7);
    write<std::uint16_t>(output, 105, OutputPointBytes);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        write(output, 131 + axis * sizeof(double), OutputScale);
        write<double>(output, 155 + axis * sizeof(double), 0.0);
        const double minimum = summary.populated ? summary.minimum[axis] : 0.0;
        const double maximum = summary.populated ? summary.maximum[axis] : 0.0;
        write(output, 179 + axis * 16U, maximum);
        write(output, 187 + axis * 16U, minimum);
    }
    write(output, 247, pointCount);
    for (std::size_t index = 0; index < summary.returns.size(); ++index)
        write(output, 255 + index * sizeof(std::uint64_t),
              summary.returns[index]);
}

__device__ std::uint16_t loadLe16(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1])
                                      << 8U);
}

__device__ std::int16_t loadLeSigned16(const std::uint8_t* bytes)
{
    const std::uint16_t value = loadLe16(bytes);
    if (value <= 0x7fffU)
        return static_cast<std::int16_t>(value);
    return static_cast<std::int16_t>(
        -1 - static_cast<std::int32_t>(0xffffU - value));
}

__device__ std::int32_t loadLe32(const std::uint8_t* bytes)
{
    const std::uint32_t value = static_cast<std::uint32_t>(bytes[0]) |
                                (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                (static_cast<std::uint32_t>(bytes[3]) << 24U);
    if (value <= 0x7fffffffU)
        return static_cast<std::int32_t>(value);
    return -1 - static_cast<std::int32_t>(0xffffffffU - value);
}

__device__ double loadLeDouble(const std::uint8_t* bytes)
{
    unsigned long long bits = 0;
#pragma unroll
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        bits |= static_cast<unsigned long long>(bytes[shift / 8U]) << shift;
    return __longlong_as_double(static_cast<long long>(bits));
}

__device__ void storeLe16(std::uint8_t* bytes, std::uint16_t value)
{
    bytes[0] = static_cast<std::uint8_t>(value);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

__device__ void storeLeSigned16(std::uint8_t* bytes, std::int16_t value)
{
    storeLe16(bytes, static_cast<std::uint16_t>(value));
}

__device__ void storeLe32(std::uint8_t* bytes, std::int32_t signedValue)
{
    const std::uint32_t value = static_cast<std::uint32_t>(signedValue);
    bytes[0] = static_cast<std::uint8_t>(value);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
    bytes[2] = static_cast<std::uint8_t>(value >> 16U);
    bytes[3] = static_cast<std::uint8_t>(value >> 24U);
}

__device__ void storeLeDouble(std::uint8_t* bytes, double value)
{
    const unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value));
#pragma unroll
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        bytes[shift / 8U] = static_cast<std::uint8_t>(bits >> shift);
}

__device__ double decodeCanonicalValue(const std::uint8_t* record,
                                       std::uint32_t dimension)
{
    if (dimension >= standardId(StandardDimension::X) &&
        dimension <= standardId(StandardDimension::Z))
    {
        const std::size_t axis = dimension - standardId(StandardDimension::X);
        return __dmul_rn(
            static_cast<double>(loadLe32(record + axis * sizeof(std::int32_t))),
            OutputScale);
    }
    if (dimension == standardId(StandardDimension::Intensity) ||
        dimension == standardId(StandardDimension::PointSourceId) ||
        dimension == standardId(StandardDimension::Red) ||
        dimension == standardId(StandardDimension::Green) ||
        dimension == standardId(StandardDimension::Blue))
    {
        std::size_t offset = 12;
        if (dimension == standardId(StandardDimension::PointSourceId))
            offset = 20;
        else if (dimension == standardId(StandardDimension::Red))
            offset = 30;
        else if (dimension == standardId(StandardDimension::Green))
            offset = 32;
        else if (dimension == standardId(StandardDimension::Blue))
            offset = 34;
        return static_cast<double>(loadLe16(record + offset));
    }
    if (dimension == standardId(StandardDimension::GpsTime))
        return loadLeDouble(record + 22);
    if (dimension == standardId(StandardDimension::ScanAngleRank))
    {
        const double scaled =
            __dmul_rn(static_cast<double>(loadLeSigned16(record + 18)), 0.006);
        return static_cast<double>(__double2float_rn(scaled));
    }

    const std::uint8_t returns = record[14];
    const std::uint8_t flags = record[15];
    if (dimension == standardId(StandardDimension::ReturnNumber))
        return static_cast<double>(returns & 0x0fU);
    if (dimension == standardId(StandardDimension::NumberOfReturns))
        return static_cast<double>((returns >> 4U) & 0x0fU);
    if (dimension == standardId(StandardDimension::Synthetic))
        return static_cast<double>(flags & 0x01U);
    if (dimension == standardId(StandardDimension::KeyPoint))
        return static_cast<double>((flags >> 1U) & 0x01U);
    if (dimension == standardId(StandardDimension::Withheld))
        return static_cast<double>((flags >> 2U) & 0x01U);
    if (dimension == standardId(StandardDimension::Overlap))
        return static_cast<double>((flags >> 3U) & 0x01U);
    if (dimension == standardId(StandardDimension::ScanChannel))
        return static_cast<double>((flags >> 4U) & 0x03U);
    if (dimension == standardId(StandardDimension::ScanDirectionFlag))
        return static_cast<double>((flags >> 6U) & 0x01U);
    if (dimension == standardId(StandardDimension::EdgeOfFlightLine))
        return static_cast<double>((flags >> 7U) & 0x01U);
    if (dimension == standardId(StandardDimension::Classification))
        return static_cast<double>(record[16]);
    if (dimension == standardId(StandardDimension::UserData))
        return static_cast<double>(record[17]);
    return 0.0;
}

__device__ void packCanonicalValue(std::uint8_t* record,
                                   std::uint32_t dimension, double value)
{
    if (dimension == standardId(StandardDimension::Intensity) ||
        dimension == standardId(StandardDimension::PointSourceId) ||
        dimension == standardId(StandardDimension::Red) ||
        dimension == standardId(StandardDimension::Green) ||
        dimension == standardId(StandardDimension::Blue))
    {
        std::size_t offset = 12;
        if (dimension == standardId(StandardDimension::PointSourceId))
            offset = 20;
        else if (dimension == standardId(StandardDimension::Red))
            offset = 30;
        else if (dimension == standardId(StandardDimension::Green))
            offset = 32;
        else if (dimension == standardId(StandardDimension::Blue))
            offset = 34;
        storeLe16(record + offset, static_cast<std::uint16_t>(value));
        return;
    }
    if (dimension == standardId(StandardDimension::GpsTime))
    {
        storeLeDouble(record + 22, value);
        return;
    }

    const std::uint8_t byte = static_cast<std::uint8_t>(value);
    if (dimension == standardId(StandardDimension::ReturnNumber))
        record[14] =
            static_cast<std::uint8_t>((record[14] & 0xf0U) | (byte & 0x0fU));
    else if (dimension == standardId(StandardDimension::NumberOfReturns))
        record[14] = static_cast<std::uint8_t>((record[14] & 0x0fU) |
                                               ((byte & 0x0fU) << 4U));
    else if (dimension == standardId(StandardDimension::Classification))
        record[16] = byte;
    else if (dimension == standardId(StandardDimension::UserData))
        record[17] = byte;
    else
    {
        unsigned shift = 0;
        std::uint8_t mask = 0x01U;
        if (dimension == standardId(StandardDimension::KeyPoint))
            shift = 1;
        else if (dimension == standardId(StandardDimension::Withheld))
            shift = 2;
        else if (dimension == standardId(StandardDimension::Overlap))
            shift = 3;
        else if (dimension == standardId(StandardDimension::ScanChannel))
        {
            shift = 4;
            mask = 0x03U;
        }
        else if (dimension == standardId(StandardDimension::ScanDirectionFlag))
            shift = 6;
        else if (dimension == standardId(StandardDimension::EdgeOfFlightLine))
            shift = 7;
        const std::uint8_t shiftedMask =
            static_cast<std::uint8_t>(mask << shift);
        record[15] = static_cast<std::uint8_t>(
            (record[15] & static_cast<std::uint8_t>(~shiftedMask)) |
            static_cast<std::uint8_t>((byte & mask) << shift));
    }
}

__device__ double symmetricRound(double value)
{
    return value > 0.0 ? floor(value + 0.5) : ceil(value - 0.5);
}

template <typename T>
__device__ void storeFusedIntegral(double& destination, double value,
                                   double lowest, double maximum)
{
    const double rounded = symmetricRound(value);
    if (rounded >= lowest && rounded <= maximum)
        destination = static_cast<double>(static_cast<T>(rounded));
}

__device__ void storeFusedValue(double& destination, DimensionType type,
                                double value)
{
    switch (type)
    {
    case DimensionType::Signed8:
        storeFusedIntegral<std::int8_t>(destination, value, -128.0, 127.0);
        return;
    case DimensionType::Signed16:
        storeFusedIntegral<std::int16_t>(destination, value, -32768.0, 32767.0);
        return;
    case DimensionType::Signed32:
        storeFusedIntegral<std::int32_t>(destination, value, -2147483648.0,
                                         2147483647.0);
        return;
    case DimensionType::Signed64:
        storeFusedIntegral<std::int64_t>(destination, value, -0x1p63, 0x1p63);
        return;
    case DimensionType::Unsigned8:
        storeFusedIntegral<std::uint8_t>(destination, value, 0.0, 255.0);
        return;
    case DimensionType::Unsigned16:
        storeFusedIntegral<std::uint16_t>(destination, value, 0.0, 65535.0);
        return;
    case DimensionType::Unsigned32:
        storeFusedIntegral<std::uint32_t>(destination, value, 0.0,
                                          4294967295.0);
        return;
    case DimensionType::Unsigned64:
        storeFusedIntegral<std::uint64_t>(destination, value, 0.0, 0x1p64);
        return;
    case DimensionType::Float:
        if (isnan(value) ||
            (value >= -0x1.fffffep+127 && value <= 0x1.fffffep+127))
            destination = static_cast<double>(__double2float_rn(value));
        return;
    case DimensionType::Double:
        destination = value;
        return;
    case DimensionType::None:
        return;
    }
}

__device__ double quietNan()
{
    return __longlong_as_double(0x7ff8000000000000LL);
}

// Register-backed value file for the eight fused dimensions. Dynamic
// indexing through the switch keeps every slot in registers instead of
// spilled local memory, which the profiler showed dominating L2 traffic.
struct FusedValueFile
{
    double v0, v1, v2, v3, v4, v5, v6, v7;

    __device__ double get(std::size_t index) const
    {
        switch (index)
        {
        case 0:
            return v0;
        case 1:
            return v1;
        case 2:
            return v2;
        case 3:
            return v3;
        case 4:
            return v4;
        case 5:
            return v5;
        case 6:
            return v6;
        default:
            return v7;
        }
    }

    __device__ void set(std::size_t index, double value)
    {
        switch (index)
        {
        case 0:
            v0 = value;
            break;
        case 1:
            v1 = value;
            break;
        case 2:
            v2 = value;
            break;
        case 3:
            v3 = value;
            break;
        case 4:
            v4 = value;
            break;
        case 5:
            v5 = value;
            break;
        case 6:
            v6 = value;
            break;
        default:
            v7 = value;
            break;
        }
    }
};

// Stack interpreter with the top two elements cached in registers, so an
// expression whose depth never exceeds two touches no local memory at all;
// deeper expressions spill only their third-and-below elements.
__device__ double evaluateFused(const FusedPointProgram& program,
                                std::uint16_t begin, std::uint16_t count,
                                const FusedValueFile& values)
{
    // Two scratch slots in front of the live spill area make the spill store
    // and reload unconditional: at logical depths below three they land in
    // (or read from) the scratch slots, whose values are never consumed.
    double spillStorage[MaximumFusedStackDepth + 2U];
    double* spill = spillStorage + 2U;
    double top = 0.0;
    double next = 0.0;
    std::uint32_t size = 0;
    const auto push = [&](double value)
    {
        spill[static_cast<std::int32_t>(size) - 2] = next;
        next = top;
        top = value;
        ++size;
    };
    const auto reduce = [&](double value)
    {
        top = value;
        --size;
        next = spill[static_cast<std::int32_t>(size) - 2];
    };
    const FusedInstruction* instructionCursor = program.instructions + begin;
    for (std::uint32_t position = 0; position < count;
         ++position, ++instructionCursor)
    {
        const FusedInstruction& instruction = *instructionCursor;
        switch (instruction.op)
        {
        case ExpressionOp::PushConstant:
            push(instruction.immediate);
            break;
        case ExpressionOp::PushFalse:
            push(0.0);
            break;
        case ExpressionOp::PushTrue:
            push(1.0);
            break;
        case ExpressionOp::LoadDimension:
            push(values.get(instruction.dimension));
            break;
        case ExpressionOp::Add:
            reduce(__dadd_rn(next, top));
            break;
        case ExpressionOp::Subtract:
            reduce(__dsub_rn(next, top));
            break;
        case ExpressionOp::Multiply:
            reduce(__dmul_rn(next, top));
            break;
        case ExpressionOp::Divide:
            reduce(top == 0.0 ? quietNan() : __ddiv_rn(next, top));
            break;
        case ExpressionOp::Negative:
        {
            const auto bits =
                static_cast<unsigned long long>(__double_as_longlong(top));
            top = __longlong_as_double(
                static_cast<long long>(bits ^ 0x8000000000000000ULL));
            break;
        }
        case ExpressionOp::Equal:
            reduce(static_cast<double>(next == top));
            break;
        case ExpressionOp::NotEqual:
            reduce(static_cast<double>(next != top));
            break;
        case ExpressionOp::Greater:
            reduce(static_cast<double>(next > top));
            break;
        case ExpressionOp::GreaterEqual:
            reduce(static_cast<double>(next >= top));
            break;
        case ExpressionOp::Less:
            reduce(static_cast<double>(next < top));
            break;
        case ExpressionOp::LessEqual:
            reduce(static_cast<double>(next <= top));
            break;
        case ExpressionOp::LogicalNot:
            top = static_cast<double>(!top);
            break;
        case ExpressionOp::LogicalAnd:
            reduce(static_cast<double>(next != 0.0 && top != 0.0));
            break;
        case ExpressionOp::LogicalOr:
            reduce(static_cast<double>(next != 0.0 || top != 0.0));
            break;
        case ExpressionOp::IsNan:
            top = static_cast<double>(isnan(top));
            break;
        case ExpressionOp::IsMaximum:
            top = static_cast<double>(top == 0x1.fffffffffffffp+1023);
            break;
        case ExpressionOp::IsMinimum:
            top = static_cast<double>(top == -0x1.fffffffffffffp+1023);
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
    return size == 1U ? top : quietNan();
}

template <std::size_t Length>
__device__ void copyBytes(std::uint8_t* output, const std::uint8_t* input)
{
#pragma unroll
    for (std::size_t index = 0; index < Length; ++index)
        output[index] = input[index];
}

template <std::size_t Length> __device__ void zeroBytes(std::uint8_t* output)
{
#pragma unroll
    for (std::size_t index = 0; index < Length; ++index)
        output[index] = 0;
}

__device__ std::int16_t legacyScanAngle(std::uint8_t encoded)
{
    const int rank = encoded <= 127U ? static_cast<int>(encoded)
                                     : static_cast<int>(encoded) - 256;
    const int scaled = rank * 500;
    return static_cast<std::int16_t>(scaled >= 0 ? (scaled + 1) / 3
                                                 : (scaled - 1) / 3);
}

__device__ void translatePointFields(const std::uint8_t* input,
                                     std::uint8_t inputFormat,
                                     std::uint8_t* output)
{
    const std::int32_t x = loadLe32(input);
    const std::int32_t y = loadLe32(input + 4);
    const std::int32_t z = loadLe32(input + 8);
    storeLe32(output, x);
    storeLe32(output + 4, y);
    storeLe32(output + 8, z);
    copyBytes<2>(output + 12, input + 12);

    std::uint8_t returnNumber;
    if (inputFormat <= 3)
    {
        const std::uint8_t returns = input[14];
        const std::uint8_t classification = input[15];
        returnNumber = returns & 0x07U;
        const std::uint8_t numberOfReturns = (returns >> 3U) & 0x07U;
        output[14] = static_cast<std::uint8_t>(
            returnNumber | static_cast<std::uint8_t>(numberOfReturns << 4U));
        std::uint8_t outputClassification = classification & 0x1fU;
        std::uint8_t overlap = 0;
        if (outputClassification == 12U)
        {
            outputClassification = 0;
            overlap = 1;
        }
        output[15] = static_cast<std::uint8_t>(
            ((classification >> 5U) & 0x01U) |
            (((classification >> 6U) & 0x01U) << 1U) |
            (((classification >> 7U) & 0x01U) << 2U) | (overlap << 3U) |
            (((returns >> 6U) & 0x01U) << 6U) |
            (((returns >> 7U) & 0x01U) << 7U));
        output[16] = outputClassification;
        output[17] = input[17];
        storeLeSigned16(output + 18, legacyScanAngle(input[16]));
        copyBytes<2>(output + 20, input + 18);
        if (inputFormat == 1 || inputFormat == 3)
            copyBytes<8>(output + 22, input + 20);
        else
            zeroBytes<8>(output + 22);
        const std::size_t colorOffset = inputFormat == 2   ? 20
                                        : inputFormat == 3 ? 28
                                                           : 0;
        if (colorOffset)
            copyBytes<6>(output + 30, input + colorOffset);
        else
            zeroBytes<6>(output + 30);
    }
    else
    {
        returnNumber = input[14] & 0x0fU;
        output[14] = input[14];
        output[15] = input[15];
        output[16] = input[16];
        output[17] = input[17];
        storeLeSigned16(output + 18, loadLeSigned16(input + 18));
        copyBytes<2>(output + 20, input + 20);
        copyBytes<8>(output + 22, input + 22);
        if (inputFormat == 7 || inputFormat == 8)
            copyBytes<6>(output + 30, input + 30);
        else
            zeroBytes<6>(output + 30);
    }
}

__device__ void summarizeCanonicalPoint(const std::uint8_t* record,
                                        RawSummary* summary)
{
    const std::int32_t x = loadLe32(record);
    const std::int32_t y = loadLe32(record + 4);
    const std::int32_t z = loadLe32(record + 8);
    atomicMin(summary->minimum, x);
    atomicMin(summary->minimum + 1, y);
    atomicMin(summary->minimum + 2, z);
    atomicMax(summary->maximum, x);
    atomicMax(summary->maximum + 1, y);
    atomicMax(summary->maximum + 2, z);
    const std::uint8_t returnNumber = record[14] & 0x0fU;
    if (returnNumber >= 1U && returnNumber <= 15U)
        atomicAdd(summary->returns + returnNumber - 1U, 1ULL);
}

__device__ void applyFusedPointProgram(std::uint8_t* record,
                                       FusedPointProgram& program)
{
    FusedValueFile values{};
    for (std::size_t index = 0; index < program.dimensionCount; ++index)
    {
        const FusedDimension& dimension = program.dimensions[index];
        if (dimension.decode)
            values.set(index, decodeCanonicalValue(record, dimension.id));
        else if (dimension.initializeZero)
            values.set(index, 0.0);
    }

    for (std::size_t index = 0; index < program.assignmentCount; ++index)
    {
        const FusedAssignment& assignment = program.assignments[index];
        if (!assignment.valueCount)
            continue;
        // Evaluate the condition and the value unconditionally and predicate
        // only the store: the evaluator is pure, so discarded values cannot
        // change results, and the uniform instruction stream avoids
        // divergent serialization between passing and failing lanes.
        const bool apply =
            !assignment.conditionCount ||
            evaluateFused(program, assignment.conditionBegin,
                          assignment.conditionCount, values) != 0.0;
        const double value = evaluateFused(program, assignment.valueBegin,
                                           assignment.valueCount, values);
        if (apply)
        {
            FusedDimension& destination =
                program.dimensions[assignment.destination];
            double slot = values.get(assignment.destination);
            storeFusedValue(slot, destination.type, value);
            values.set(assignment.destination, slot);
        }
    }

    for (std::size_t index = 0; index < program.dimensionCount; ++index)
        if (program.dimensions[index].pack)
            packCanonicalValue(record, program.dimensions[index].id,
                               values.get(index));
}

// Cooperative block copy of one contiguous byte span. Threads move
// consecutive 4-byte words when both pointers and the length share 4-byte
// alignment and consecutive bytes otherwise; either shape keeps every warp
// on one contiguous cache-line run, so global traffic stays coalesced.
__device__ void blockCopyBytes(std::uint8_t* destination,
                               const std::uint8_t* source, std::size_t bytes)
{
    const std::uintptr_t addresses =
        reinterpret_cast<std::uintptr_t>(destination) |
        reinterpret_cast<std::uintptr_t>(source) |
        static_cast<std::uintptr_t>(bytes);
    if ((addresses & 15U) == 0U)
    {
        const uint4* in = reinterpret_cast<const uint4*>(source);
        uint4* out = reinterpret_cast<uint4*>(destination);
        const std::size_t quads = bytes / 16U;
#pragma unroll 4
        for (std::size_t quad = threadIdx.x; quad < quads; quad += blockDim.x)
            out[quad] = in[quad];
        return;
    }
    if ((addresses & 3U) == 0U)
    {
        const std::uint32_t* in =
            reinterpret_cast<const std::uint32_t*>(source);
        std::uint32_t* out = reinterpret_cast<std::uint32_t*>(destination);
        const std::size_t words = bytes / 4U;
#pragma unroll 4
        for (std::size_t word = threadIdx.x; word < words; word += blockDim.x)
            out[word] = in[word];
        return;
    }
    for (std::size_t byte = threadIdx.x; byte < bytes; byte += blockDim.x)
        destination[byte] = source[byte];
}

// The packed input and output records of one block are contiguous global
// spans, so the block stages them through shared memory with coalesced
// copies and performs the byte-granular record translation in shared
// storage. The legacy unstaged kernels remain the fallback for record
// strides whose staging would exceed the shared-memory budget.
// Block-local twin of summarizeCanonicalPoint. Integer minimum/maximum and
// counter addition are commutative and associative, so aggregating per block
// in shared memory and merging once per block into the global summary is
// exactly the value the per-point global atomics would produce.
__device__ void summarizeCanonicalPointShared(const std::uint8_t* record,
                                              std::int32_t* sharedMinimum,
                                              std::int32_t* sharedMaximum,
                                              unsigned long long* sharedReturns)
{
    const std::int32_t x = loadLe32(record);
    const std::int32_t y = loadLe32(record + 4);
    const std::int32_t z = loadLe32(record + 8);
    atomicMin(sharedMinimum, x);
    atomicMin(sharedMinimum + 1, y);
    atomicMin(sharedMinimum + 2, z);
    atomicMax(sharedMaximum, x);
    atomicMax(sharedMaximum + 1, y);
    atomicMax(sharedMaximum + 2, z);
    const std::uint8_t returnNumber = record[14] & 0x0fU;
    if (returnNumber >= 1U && returnNumber <= 15U)
        atomicAdd(sharedReturns + returnNumber - 1U, 1ULL);
}

template <bool Fused>
__device__ void
stagedTranslateBody(const std::uint8_t* input, std::size_t inputStride,
                    std::uint8_t inputFormat, std::size_t pointCount,
                    std::uint8_t* output, RawSummary* summary,
                    FusedPointProgram* program)
{
    __shared__ std::int32_t sharedMinimum[3];
    __shared__ std::int32_t sharedMaximum[3];
    __shared__ unsigned long long sharedReturns[15];
    extern __shared__ std::uint8_t staged[];
    // Formats 7 and 8 with a packed 36-byte stride translate to an identical
    // record, so the block mutates its staged input in place and skips both
    // the output staging span and the per-point record copy.
    const bool inPlace = (inputFormat == 7U || inputFormat == 8U) &&
                         inputStride == OutputPointBytes;
    std::uint8_t* stagedInput = staged;
    std::uint8_t* stagedOutput =
        inPlace ? staged
                : staged + static_cast<std::size_t>(blockDim.x) * inputStride;
    constexpr std::int32_t SentinelMinimum = 0x7fffffff;
    constexpr std::int32_t SentinelMaximum = -0x7fffffff - 1;
    if (threadIdx.x < 3U)
    {
        sharedMinimum[threadIdx.x] = SentinelMinimum;
        sharedMaximum[threadIdx.x] = SentinelMaximum;
    }
    if (threadIdx.x < 15U)
        sharedReturns[threadIdx.x] = 0ULL;
    const std::size_t blockPoints = blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockPoints;
         base < pointCount;
         base += static_cast<std::size_t>(gridDim.x) * blockPoints)
    {
        const std::size_t count =
            pointCount - base < blockPoints ? pointCount - base : blockPoints;
        blockCopyBytes(stagedInput, input + base * inputStride,
                       count * inputStride);
        __syncthreads();
        if (threadIdx.x < count)
        {
            std::uint8_t* record =
                stagedOutput +
                static_cast<std::size_t>(threadIdx.x) * OutputPointBytes;
            const std::uint8_t* source =
                stagedInput +
                static_cast<std::size_t>(threadIdx.x) * inputStride;
            // Formats 7 and 8 translate to a byte-identical 36-byte prefix
            // (see translatePointFields): in place there is nothing to move,
            // a wider stride moves word-wise, and other formats keep the
            // field-wise translation.
            if (inPlace)
                ;
            else if ((inputFormat == 7U || inputFormat == 8U) &&
                     (inputStride & 3U) == 0U)
            {
                const std::uint32_t* in =
                    reinterpret_cast<const std::uint32_t*>(source);
                std::uint32_t* out = reinterpret_cast<std::uint32_t*>(record);
#pragma unroll
                for (int word = 0; word < 9; ++word)
                    out[word] = in[word];
            }
            else
                translatePointFields(source, inputFormat, record);
            if constexpr (Fused)
                applyFusedPointProgram(record, *program);
            summarizeCanonicalPointShared(record, sharedMinimum, sharedMaximum,
                                          sharedReturns);
        }
        __syncthreads();
        blockCopyBytes(output + base * OutputPointBytes, stagedOutput,
                       count * OutputPointBytes);
        __syncthreads();
    }
    if (threadIdx.x < 3U && sharedMinimum[threadIdx.x] != SentinelMinimum)
        atomicMin(summary->minimum + threadIdx.x, sharedMinimum[threadIdx.x]);
    if (threadIdx.x < 3U && sharedMaximum[threadIdx.x] != SentinelMaximum)
        atomicMax(summary->maximum + threadIdx.x, sharedMaximum[threadIdx.x]);
    if (threadIdx.x < 15U && sharedReturns[threadIdx.x])
        atomicAdd(summary->returns + threadIdx.x, sharedReturns[threadIdx.x]);
}

__global__ void __launch_bounds__(StagedBlockSize, 4)
    translateStagedKernel(const std::uint8_t* input, std::size_t inputStride,
                          std::uint8_t inputFormat, std::size_t pointCount,
                          std::uint8_t* output, RawSummary* summary)
{
    stagedTranslateBody<false>(input, inputStride, inputFormat, pointCount,
                               output, summary, nullptr);
}

__global__ void __launch_bounds__(StagedBlockSize, 5)
    fusedTranslatePointProgramStagedKernel(
        const std::uint8_t* input, std::size_t inputStride,
        std::uint8_t inputFormat, std::size_t pointCount, std::uint8_t* output,
        RawSummary* summary, FusedPointProgram program)
{
    stagedTranslateBody<true>(input, inputStride, inputFormat, pointCount,
                              output, summary, &program);
}

__global__ void translateKernel(const std::uint8_t* input,
                                std::size_t inputStride,
                                std::uint8_t inputFormat,
                                std::size_t pointCount, std::uint8_t* output,
                                RawSummary* summary)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < pointCount; index += grid)
    {
        std::uint8_t* record = output + index * OutputPointBytes;
        translatePointFields(input + index * inputStride, inputFormat, record);
        summarizeCanonicalPoint(record, summary);
    }
}

__global__ void fusedTranslatePointProgramKernel(
    const std::uint8_t* input, std::size_t inputStride,
    std::uint8_t inputFormat, std::size_t pointCount, std::uint8_t* output,
    RawSummary* summary, FusedPointProgram program)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        std::uint8_t* record = output + point * OutputPointBytes;
        translatePointFields(input + point * inputStride, inputFormat, record);
        applyFusedPointProgram(record, program);
        summarizeCanonicalPoint(record, summary);
    }
}

__global__ void summarizeReturnNumbersKernel(const std::uint8_t* values,
                                             std::size_t pointCount,
                                             RawSummary* summary)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < pointCount; index += grid)
    {
        const std::uint8_t value = values[index];
        if (value >= 1U && value <= 15U)
            atomicAdd(summary->returns + value - 1U, 1ULL);
    }
}

__global__ void initializeIndexesKernel(std::uint64_t* indexes,
                                        std::size_t pointCount)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < pointCount; index += grid)
        indexes[index] = index;
}

__global__ void gatherCanonicalRecordsKernel(const std::uint8_t* input,
                                             const std::uint64_t* indexes,
                                             std::size_t pointCount,
                                             std::uint8_t* output)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        const std::uint8_t* source =
            input + static_cast<std::size_t>(indexes[point]) * OutputPointBytes;
        std::uint8_t* destination = output + point * OutputPointBytes;
        copyBytes<OutputPointBytes>(destination, source);
    }
}

__global__ void summarizeCanonicalRecordsKernel(const std::uint8_t* records,
                                                std::size_t pointCount,
                                                RawSummary* summary)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        const std::uint8_t* record = records + point * OutputPointBytes;
        const std::int32_t x = loadLe32(record);
        const std::int32_t y = loadLe32(record + 4);
        const std::int32_t z = loadLe32(record + 8);
        atomicMin(summary->minimum, x);
        atomicMin(summary->minimum + 1, y);
        atomicMin(summary->minimum + 2, z);
        atomicMax(summary->maximum, x);
        atomicMax(summary->maximum + 1, y);
        atomicMax(summary->maximum + 2, z);
        const std::uint8_t returnNumber = record[14] & 0x0fU;
        if (returnNumber >= 1U && returnNumber <= 15U)
            atomicAdd(summary->returns + returnNumber - 1U, 1ULL);
    }
}

int gridSize(std::size_t count)
{
    const std::size_t blocks =
        count / BlockSize + static_cast<std::size_t>(count % BlockSize != 0);
    return static_cast<int>(
        std::min<std::size_t>(blocks, std::numeric_limits<int>::max()));
}

std::size_t checkedProduct(std::size_t count, std::size_t stride,
                           const char* description)
{
    if (count > std::numeric_limits<std::size_t>::max() / stride)
        throw Error(description);
    return count * stride;
}

std::size_t checkedAdd(std::size_t left, std::size_t right,
                       const char* description)
{
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw Error(description);
    return left + right;
}

DimensionType canonicalPhysicalType(DimensionId dimension) noexcept
{
    if (dimension == DimensionId(StandardDimension::X) ||
        dimension == DimensionId(StandardDimension::Y) ||
        dimension == DimensionId(StandardDimension::Z))
        return DimensionType::Signed32;
    if (dimension == DimensionId(StandardDimension::Intensity) ||
        dimension == DimensionId(StandardDimension::PointSourceId) ||
        dimension == DimensionId(StandardDimension::Red) ||
        dimension == DimensionId(StandardDimension::Green) ||
        dimension == DimensionId(StandardDimension::Blue))
        return DimensionType::Unsigned16;
    if (dimension == DimensionId(StandardDimension::GpsTime))
        return DimensionType::Double;
    if (dimension == DimensionId(StandardDimension::ScanAngleRank))
        return DimensionType::Float;
    if (dimension == DimensionId(StandardDimension::ReturnNumber) ||
        dimension == DimensionId(StandardDimension::NumberOfReturns) ||
        dimension == DimensionId(StandardDimension::ScanDirectionFlag) ||
        dimension == DimensionId(StandardDimension::EdgeOfFlightLine) ||
        dimension == DimensionId(StandardDimension::Classification) ||
        dimension == DimensionId(StandardDimension::UserData) ||
        dimension == DimensionId(StandardDimension::ScanChannel) ||
        dimension == DimensionId(StandardDimension::Synthetic) ||
        dimension == DimensionId(StandardDimension::KeyPoint) ||
        dimension == DimensionId(StandardDimension::Withheld) ||
        dimension == DimensionId(StandardDimension::Overlap))
        return DimensionType::Unsigned8;
    return DimensionType::None;
}

bool fusedExpressionOperation(ExpressionOp op) noexcept
{
    switch (op)
    {
    case ExpressionOp::PushConstant:
    case ExpressionOp::PushFalse:
    case ExpressionOp::PushTrue:
    case ExpressionOp::LoadDimension:
    case ExpressionOp::Add:
    case ExpressionOp::Subtract:
    case ExpressionOp::Multiply:
    case ExpressionOp::Divide:
    case ExpressionOp::Negative:
    case ExpressionOp::Equal:
    case ExpressionOp::NotEqual:
    case ExpressionOp::Greater:
    case ExpressionOp::GreaterEqual:
    case ExpressionOp::Less:
    case ExpressionOp::LessEqual:
    case ExpressionOp::LogicalNot:
    case ExpressionOp::LogicalAnd:
    case ExpressionOp::LogicalOr:
    case ExpressionOp::IsNan:
    case ExpressionOp::IsMaximum:
    case ExpressionOp::IsMinimum:
        return true;
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
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Fused-program specialization (D0071): each admitted FusedPointProgram is
// compiled once with NVRTC into a straight-line kernel with the program,
// record format, and stride baked in as constants. The generated arithmetic
// walks the same instruction stream with the same exactly-rounded intrinsics
// in the same order as the interpreter, so results are byte-identical; every
// existing differential and process gate runs through this path when it is
// active. Any generation or compilation failure falls back to the staged
// interpreter kernel. Specialization currently covers the identity record
// formats (7 and 8) at the packed 36-byte stride, which is the measured E1
// class; other shapes keep the interpreter.
namespace
{

std::string hexDouble(double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%a", value);
    return buffer;
}

const char* jitCType(DimensionType type)
{
    switch (type)
    {
    case DimensionType::Signed8:
        return "signed char";
    case DimensionType::Signed16:
        return "short";
    case DimensionType::Signed32:
        return "int";
    case DimensionType::Signed64:
        return "long long";
    case DimensionType::Unsigned8:
        return "unsigned char";
    case DimensionType::Unsigned16:
        return "unsigned short";
    case DimensionType::Unsigned32:
        return "unsigned int";
    case DimensionType::Unsigned64:
        return "unsigned long long";
    default:
        return nullptr;
    }
}

bool jitIntegralRange(DimensionType type, double& lowest, double& maximum)
{
    switch (type)
    {
    case DimensionType::Signed8:
        lowest = -128.0;
        maximum = 127.0;
        return true;
    case DimensionType::Signed16:
        lowest = -32768.0;
        maximum = 32767.0;
        return true;
    case DimensionType::Signed32:
        lowest = -2147483648.0;
        maximum = 2147483647.0;
        return true;
    case DimensionType::Signed64:
        lowest = -0x1p63;
        maximum = 0x1p63;
        return true;
    case DimensionType::Unsigned8:
        lowest = 0.0;
        maximum = 255.0;
        return true;
    case DimensionType::Unsigned16:
        lowest = 0.0;
        maximum = 65535.0;
        return true;
    case DimensionType::Unsigned32:
        lowest = 0.0;
        maximum = 4294967295.0;
        return true;
    case DimensionType::Unsigned64:
        lowest = 0.0;
        maximum = 0x1p64;
        return true;
    default:
        return false;
    }
}

// Mirrors decodeCanonicalValue for one dimension with literal offsets; the
// differential tests pin this against the interpreter bit-for-bit.
bool jitDecodeExpression(std::uint32_t id, std::string& expression)
{
    const auto standard = [&](StandardDimension dimension)
    { return id == standardId(dimension); };
    if (id >= standardId(StandardDimension::X) &&
        id <= standardId(StandardDimension::Z))
    {
        const std::uint32_t axis = id - standardId(StandardDimension::X);
        expression = "__dmul_rn((double)jitLoadLe32(record + " +
                     std::to_string(axis * 4U) + "), " +
                     hexDouble(OutputScale) + ")";
        return true;
    }
    const auto le16 = [&](unsigned offset)
    {
        expression =
            "(double)jitLoadLe16(record + " + std::to_string(offset) + ")";
        return true;
    };
    if (standard(StandardDimension::Intensity))
        return le16(12U);
    if (standard(StandardDimension::PointSourceId))
        return le16(20U);
    if (standard(StandardDimension::Red))
        return le16(30U);
    if (standard(StandardDimension::Green))
        return le16(32U);
    if (standard(StandardDimension::Blue))
        return le16(34U);
    if (standard(StandardDimension::GpsTime))
    {
        expression = "jitLoadLeDouble(record + 22)";
        return true;
    }
    if (standard(StandardDimension::ScanAngleRank))
    {
        expression = "(double)__double2float_rn(__dmul_rn("
                     "(double)jitLoadLeSigned16(record + 18), " +
                     hexDouble(0.006) + "))";
        return true;
    }
    if (standard(StandardDimension::ReturnNumber))
    {
        expression = "(double)(record[14] & 0x0fU)";
        return true;
    }
    if (standard(StandardDimension::NumberOfReturns))
    {
        expression = "(double)((record[14] >> 4U) & 0x0fU)";
        return true;
    }
    if (standard(StandardDimension::Synthetic))
    {
        expression = "(double)(record[15] & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::KeyPoint))
    {
        expression = "(double)((record[15] >> 1U) & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::Withheld))
    {
        expression = "(double)((record[15] >> 2U) & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::Overlap))
    {
        expression = "(double)((record[15] >> 3U) & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::ScanChannel))
    {
        expression = "(double)((record[15] >> 4U) & 0x03U)";
        return true;
    }
    if (standard(StandardDimension::ScanDirectionFlag))
    {
        expression = "(double)((record[15] >> 6U) & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::EdgeOfFlightLine))
    {
        expression = "(double)((record[15] >> 7U) & 0x01U)";
        return true;
    }
    if (standard(StandardDimension::Classification))
    {
        expression = "(double)record[16]";
        return true;
    }
    if (standard(StandardDimension::UserData))
    {
        expression = "(double)record[17]";
        return true;
    }
    return false;
}

// Mirrors packCanonicalValue for one dimension with literal offsets.
bool jitPackStatement(std::uint32_t id, const std::string& value,
                      std::string& statement)
{
    const auto standard = [&](StandardDimension dimension)
    { return id == standardId(dimension); };
    const auto le16 = [&](unsigned offset)
    {
        statement = "jitStoreLe16(record + " + std::to_string(offset) +
                    ", (unsigned short)" + value + ");";
        return true;
    };
    if (standard(StandardDimension::Intensity))
        return le16(12U);
    if (standard(StandardDimension::PointSourceId))
        return le16(20U);
    if (standard(StandardDimension::Red))
        return le16(30U);
    if (standard(StandardDimension::Green))
        return le16(32U);
    if (standard(StandardDimension::Blue))
        return le16(34U);
    if (standard(StandardDimension::GpsTime))
    {
        statement = "jitStoreLeDouble(record + 22, " + value + ");";
        return true;
    }
    const std::string byte = "((unsigned char)" + value + ")";
    if (standard(StandardDimension::ReturnNumber))
    {
        statement = "record[14] = (unsigned char)((record[14] & 0xf0U) | (" +
                    byte + " & 0x0fU));";
        return true;
    }
    if (standard(StandardDimension::NumberOfReturns))
    {
        statement = "record[14] = (unsigned char)((record[14] & 0x0fU) | ((" +
                    byte + " & 0x0fU) << 4U));";
        return true;
    }
    if (standard(StandardDimension::Classification))
    {
        statement = "record[16] = " + byte + ";";
        return true;
    }
    if (standard(StandardDimension::UserData))
    {
        statement = "record[17] = " + byte + ";";
        return true;
    }
    unsigned shift = 0;
    unsigned mask = 0x01U;
    if (standard(StandardDimension::Synthetic))
        shift = 0;
    else if (standard(StandardDimension::KeyPoint))
        shift = 1;
    else if (standard(StandardDimension::Withheld))
        shift = 2;
    else if (standard(StandardDimension::Overlap))
        shift = 3;
    else if (standard(StandardDimension::ScanChannel))
    {
        shift = 4;
        mask = 3;
    }
    else if (standard(StandardDimension::ScanDirectionFlag))
        shift = 6;
    else if (standard(StandardDimension::EdgeOfFlightLine))
        shift = 7;
    else
        return false;
    const unsigned shifted = mask << shift;
    statement = "record[15] = (unsigned char)((record[15] & " +
                std::to_string(0xffU & ~shifted) + "U) | ((" + byte + " & " +
                std::to_string(mask) + "U) << " + std::to_string(shift) +
                "U));";
    return true;
}

bool jitExpression(const FusedPointProgram& program, std::uint16_t begin,
                   std::uint16_t count, std::string& body, int& temporary,
                   std::string& result)
{
    std::vector<std::string> stack;
    const auto temp = [&](const std::string& expression)
    {
        const std::string name = "t" + std::to_string(temporary++);
        body += "        const double " + name + " = " + expression + ";\n";
        stack.push_back(name);
    };
    const auto binary = [&](const std::string& expression)
    {
        if (stack.size() < 2U)
            return false;
        const std::string b = stack.back();
        stack.pop_back();
        const std::string a = stack.back();
        stack.pop_back();
        std::string text = expression;
        std::size_t position;
        while ((position = text.find("$A")) != std::string::npos)
            text.replace(position, 2, a);
        while ((position = text.find("$B")) != std::string::npos)
            text.replace(position, 2, b);
        temp(text);
        return true;
    };
    const auto unary = [&](const std::string& expression)
    {
        if (stack.empty())
            return false;
        const std::string a = stack.back();
        stack.pop_back();
        std::string text = expression;
        std::size_t position;
        while ((position = text.find("$A")) != std::string::npos)
            text.replace(position, 2, a);
        temp(text);
        return true;
    };
    for (std::uint16_t index = 0; index < count; ++index)
    {
        const FusedInstruction& instruction =
            program.instructions[begin + index];
        bool ok = true;
        switch (instruction.op)
        {
        case ExpressionOp::PushConstant:
            temp(hexDouble(instruction.immediate));
            break;
        case ExpressionOp::PushFalse:
            temp("0.0");
            break;
        case ExpressionOp::PushTrue:
            temp("1.0");
            break;
        case ExpressionOp::LoadDimension:
            stack.push_back("v" + std::to_string(instruction.dimension));
            break;
        case ExpressionOp::Add:
            ok = binary("__dadd_rn($A, $B)");
            break;
        case ExpressionOp::Subtract:
            ok = binary("__dsub_rn($A, $B)");
            break;
        case ExpressionOp::Multiply:
            ok = binary("__dmul_rn($A, $B)");
            break;
        case ExpressionOp::Divide:
            ok = binary("($B == 0.0 ? jitQuietNan() : __ddiv_rn($A, $B))");
            break;
        case ExpressionOp::Negative:
            ok = unary("__longlong_as_double((long long)(((unsigned long "
                       "long)__double_as_longlong($A)) ^ "
                       "0x8000000000000000ULL))");
            break;
        case ExpressionOp::Equal:
            ok = binary("(double)($A == $B)");
            break;
        case ExpressionOp::NotEqual:
            ok = binary("(double)($A != $B)");
            break;
        case ExpressionOp::Greater:
            ok = binary("(double)($A > $B)");
            break;
        case ExpressionOp::GreaterEqual:
            ok = binary("(double)($A >= $B)");
            break;
        case ExpressionOp::Less:
            ok = binary("(double)($A < $B)");
            break;
        case ExpressionOp::LessEqual:
            ok = binary("(double)($A <= $B)");
            break;
        case ExpressionOp::LogicalNot:
            ok = unary("(double)(!$A)");
            break;
        case ExpressionOp::LogicalAnd:
            ok = binary("(double)($A != 0.0 && $B != 0.0)");
            break;
        case ExpressionOp::LogicalOr:
            ok = binary("(double)($A != 0.0 || $B != 0.0)");
            break;
        case ExpressionOp::IsNan:
            ok = unary("(double)(isnan($A))");
            break;
        case ExpressionOp::IsMaximum:
            ok = unary("(double)($A == " + hexDouble(0x1.fffffffffffffp+1023) +
                       ")");
            break;
        case ExpressionOp::IsMinimum:
            ok = unary("(double)($A == " + hexDouble(-0x1.fffffffffffffp+1023) +
                       ")");
            break;
        default:
            return false;
        }
        if (!ok)
            return false;
    }
    if (stack.size() != 1U)
        return false;
    result = stack.back();
    return true;
}

// Mandatory record bytes per LAS point format; a header may declare a
// larger stride with extra bytes, which stage untouched and unread.
constexpr std::size_t minimumRecordLength(std::uint8_t format) noexcept
{
    constexpr std::size_t lengths[] = {20U, 28U, 26U, 34U, 57U,
                                       63U, 30U, 36U, 38U};
    return format <= 8U ? lengths[format] : ~std::size_t{0};
}

// Emits the straight-line specialization of translatePointFields for one
// constant record format: the canonical 36-byte record is built from the
// staged source record with the same operations, in the same order, as
// the interpreter. Identity formats (7/8 at the canonical stride) emit
// nothing; the record stages in place.
std::string jitTranslateStatements(std::uint8_t inputFormat)
{
    std::string text;
    text += "            const u8* source = stagedInput + (u64)threadIdx.x * "
            "%STRIDE%U;\n";
    text += "            jitStoreLe32(record, jitLoadLe32(source));\n";
    text += "            jitStoreLe32(record + 4, jitLoadLe32(source + 4));\n";
    text += "            jitStoreLe32(record + 8, jitLoadLe32(source + 8));\n";
    text += "            record[12] = source[12]; record[13] = source[13];\n";
    if (inputFormat <= 3U)
    {
        text += "            { const u8 returns = source[14];\n";
        text += "              const u8 classification = source[15];\n";
        text += "              const u8 returnNumber = returns & 0x07U;\n";
        text += "              const u8 numberOfReturns = (returns >> 3U) & "
                "0x07U;\n";
        text += "              record[14] = (u8)(returnNumber | "
                "(u8)(numberOfReturns << 4U));\n";
        text += "              u8 outputClassification = classification & "
                "0x1fU;\n";
        text += "              u8 overlap = 0;\n";
        text += "              if (outputClassification == 12U)\n";
        text += "              { outputClassification = 0; overlap = 1; }\n";
        text += "              record[15] = (u8)(((classification >> 5U) & "
                "0x01U) | (((classification >> 6U) & 0x01U) << 1U) | "
                "(((classification >> 7U) & 0x01U) << 2U) | (overlap << 3U) | "
                "(((returns >> 6U) & 0x01U) << 6U) | (((returns >> 7U) & "
                "0x01U) << 7U));\n";
        text += "              record[16] = outputClassification;\n";
        text += "              record[17] = source[17];\n";
        text += "              jitStoreLeSigned16(record + 18, "
                "jitLegacyScanAngle(source[16]));\n";
        text += "              record[20] = source[18]; record[21] = "
                "source[19];\n";
        if (inputFormat == 1U || inputFormat == 3U)
            text += "              for (unsigned i = 0; i < 8U; ++i) "
                    "record[22 + i] = source[20 + i];\n";
        else
            text += "              for (unsigned i = 0; i < 8U; ++i) "
                    "record[22 + i] = 0U;\n";
        if (inputFormat == 2U || inputFormat == 3U)
        {
            const unsigned colorOffset = inputFormat == 2U ? 20U : 28U;
            text += "              for (unsigned i = 0; i < 6U; ++i) "
                    "record[30 + i] = source[" +
                    std::to_string(colorOffset) + "U + i];\n";
        }
        else
            text += "              for (unsigned i = 0; i < 6U; ++i) "
                    "record[30 + i] = 0U;\n";
        text += "            }\n";
    }
    else
    {
        text += "            for (unsigned i = 14U; i < 30U; ++i) record[i] = "
                "source[i];\n";
        if (inputFormat == 7U || inputFormat == 8U)
            text += "            for (unsigned i = 30U; i < 36U; ++i) "
                    "record[i] = source[i];\n";
        else
            text += "            for (unsigned i = 30U; i < 36U; ++i) "
                    "record[i] = 0U;\n";
    }
    return text;
}

std::optional<std::string>
generateFusedKernelSource(const FusedPointProgram& program,
                          std::uint8_t inputFormat, std::size_t inputStride)
{
    std::string body;
    int temporary = 0;
    for (std::uint16_t index = 0; index < program.dimensionCount; ++index)
    {
        const FusedDimension& dimension = program.dimensions[index];
        std::string initializer = "0.0";
        if (dimension.decode && !jitDecodeExpression(dimension.id, initializer))
            return std::nullopt;
        body += "        double v" + std::to_string(index) + " = " +
                initializer + ";\n";
    }
    for (std::uint16_t index = 0; index < program.assignmentCount; ++index)
    {
        const FusedAssignment& assignment = program.assignments[index];
        if (!assignment.valueCount)
            continue;
        std::string apply = "true";
        if (assignment.conditionCount)
        {
            std::string condition;
            if (!jitExpression(program, assignment.conditionBegin,
                               assignment.conditionCount, body, temporary,
                               condition))
                return std::nullopt;
            apply = condition + " != 0.0";
        }
        std::string value;
        if (!jitExpression(program, assignment.valueBegin,
                           assignment.valueCount, body, temporary, value))
            return std::nullopt;
        const FusedDimension& destination =
            program.dimensions[assignment.destination];
        const std::string slot = "v" + std::to_string(assignment.destination);
        double lowest = 0.0;
        double maximum = 0.0;
        if (const char* ctype = jitCType(destination.type);
            ctype && jitIntegralRange(destination.type, lowest, maximum))
        {
            body += "        { const double r = jitSymmetricRound(" + value +
                    "); if ((" + apply + ") && r >= " + hexDouble(lowest) +
                    " && r <= " + hexDouble(maximum) + ") " + slot +
                    " = (double)(" + ctype + ")r; }\n";
        }
        else if (destination.type == DimensionType::Float)
            body += "        if ((" + apply + ") && (isnan(" + value +
                    ") || (" + value + " >= " + hexDouble(-0x1.fffffep+127) +
                    " && " + value + " <= " + hexDouble(0x1.fffffep+127) +
                    "))) " + slot + " = (double)__double2float_rn(" + value +
                    ");\n";
        else if (destination.type == DimensionType::Double)
            body +=
                "        if (" + apply + ") " + slot + " = " + value + ";\n";
        else
            return std::nullopt;
    }
    for (std::uint16_t index = 0; index < program.dimensionCount; ++index)
    {
        const FusedDimension& dimension = program.dimensions[index];
        if (!dimension.pack)
            continue;
        std::string statement;
        if (!jitPackStatement(dimension.id, "v" + std::to_string(index),
                              statement))
            return std::nullopt;
        body += "        " + statement + "\n";
    }

    std::string source = R"(
typedef unsigned char u8; typedef unsigned short u16; typedef unsigned int u32;
typedef unsigned long long u64; typedef long long s64;
struct JitSummary { int minimum[3]; int maximum[3]; u64 returns[15]; };
__device__ __forceinline__ u16 jitLoadLe16(const u8* b)
{ return (u16)((u32)b[0] | ((u32)b[1] << 8U)); }
__device__ __forceinline__ short jitLoadLeSigned16(const u8* b)
{ u16 v = jitLoadLe16(b); return v <= 0x7fffU ? (short)v
      : (short)(-1 - (int)(0xffffU - v)); }
__device__ __forceinline__ int jitLoadLe32(const u8* b)
{ u32 v = (u32)b[0] | ((u32)b[1] << 8U) | ((u32)b[2] << 16U)
      | ((u32)b[3] << 24U);
  return v <= 0x7fffffffU ? (int)v : -1 - (int)(0xffffffffU - v); }
__device__ __forceinline__ double jitLoadLeDouble(const u8* b)
{ u64 bits = 0;
  for (unsigned s = 0; s < 64U; s += 8U) bits |= (u64)b[s / 8U] << s;
  return __longlong_as_double((s64)bits); }
__device__ __forceinline__ void jitStoreLe16(u8* b, u16 v)
{ b[0] = (u8)v; b[1] = (u8)(v >> 8U); }
%HELPERS%__device__ __forceinline__ void jitStoreLeDouble(u8* b, double v)
{ u64 bits = (u64)__double_as_longlong(v);
  for (unsigned s = 0; s < 64U; s += 8U) b[s / 8U] = (u8)(bits >> s); }
__device__ __forceinline__ double jitQuietNan()
{ return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ double jitSymmetricRound(double v)
{ return v > 0.0 ? floor(v + 0.5) : ceil(v - 0.5); }
__device__ void jitBlockCopy(u8* d, const u8* s, u64 bytes)
{
    unsigned long long a = ((unsigned long long)(size_t)d) |
                           ((unsigned long long)(size_t)s) | bytes;
    if ((a & 15U) == 0U)
    {
        const uint4* in = (const uint4*)s; uint4* out = (uint4*)d;
        u64 quads = bytes / 16U;
        for (u64 q = threadIdx.x; q < quads; q += blockDim.x) out[q] = in[q];
        return;
    }
    if ((a & 3U) == 0U)
    {
        const u32* in = (const u32*)s; u32* out = (u32*)d;
        u64 words = bytes / 4U;
        for (u64 w = threadIdx.x; w < words; w += blockDim.x) out[w] = in[w];
        return;
    }
    for (u64 i = threadIdx.x; i < bytes; i += blockDim.x) d[i] = s[i];
}
extern "C" __global__ void __launch_bounds__(256, 5) pdgJitFusedKernel(
    const u8* input, u64 pointCount, u8* output, JitSummary* summary)
{
    __shared__ int sharedMinimum[3];
    __shared__ int sharedMaximum[3];
    __shared__ u64 sharedReturns[15];
    extern __shared__ u8 staged[];
%STAGES%
    if (threadIdx.x < 3U)
    { sharedMinimum[threadIdx.x] = 0x7fffffff;
      sharedMaximum[threadIdx.x] = -0x7fffffff - 1; }
    if (threadIdx.x < 15U) sharedReturns[threadIdx.x] = 0ULL;
    const u64 blockPoints = blockDim.x;
    for (u64 base = (u64)blockIdx.x * blockPoints; base < pointCount;
         base += (u64)gridDim.x * blockPoints)
    {
        const u64 count = pointCount - base < blockPoints
                              ? pointCount - base : blockPoints;
        jitBlockCopy(stagedInput, input + base * %STRIDE%U,
                     count * %STRIDE%U);
        __syncthreads();
        if (threadIdx.x < count)
        {
            u8* record = stagedOutput + (u64)threadIdx.x * 36U;
%TRANSLATE%
%BODY%
            const int px = jitLoadLe32(record);
            const int py = jitLoadLe32(record + 4);
            const int pz = jitLoadLe32(record + 8);
            atomicMin(sharedMinimum, px);
            atomicMin(sharedMinimum + 1, py);
            atomicMin(sharedMinimum + 2, pz);
            atomicMax(sharedMaximum, px);
            atomicMax(sharedMaximum + 1, py);
            atomicMax(sharedMaximum + 2, pz);
            const u8 returnNumber = record[14] & 0x0fU;
            if (returnNumber >= 1U && returnNumber <= 15U)
                atomicAdd(sharedReturns + returnNumber - 1U, 1ULL);
        }
        __syncthreads();
        jitBlockCopy(output + base * 36U, stagedOutput, count * 36U);
        __syncthreads();
    }
    if (threadIdx.x < 3U && sharedMinimum[threadIdx.x] != 0x7fffffff)
        atomicMin(summary->minimum + threadIdx.x, sharedMinimum[threadIdx.x]);
    if (threadIdx.x < 3U && sharedMaximum[threadIdx.x] != -0x7fffffff - 1)
        atomicMax(summary->maximum + threadIdx.x, sharedMaximum[threadIdx.x]);
    if (threadIdx.x < 15U && sharedReturns[threadIdx.x])
        atomicAdd(summary->returns + threadIdx.x, sharedReturns[threadIdx.x]);
}
)";
    const bool identity = (inputFormat == 7U || inputFormat == 8U) &&
                          inputStride == OutputPointBytes;
    // The identity layout stages one canonical span in place; every other
    // format stages the source records ahead of the canonical output span
    // and rebuilds each record exactly as the interpreter's
    // translatePointFields does.
    const std::string stages =
        identity ? "    u8* stagedInput = staged;\n"
                   "    u8* stagedOutput = staged;\n"
                 : "    u8* stagedInput = staged;\n"
                   "    u8* stagedOutput = staged + (u64)blockDim.x * " +
                       std::to_string(inputStride) + "U;\n";
    const std::string translate =
        identity ? std::string() : jitTranslateStatements(inputFormat);
    const auto substitute =
        [&source](const char* token, const std::string& value)
    {
        const std::size_t position = source.find(token);
        source.replace(position, std::string_view(token).size(), value);
    };
    // The specialization helpers exist only when the translate block needs
    // them so the identity source (and its compile time) stays minimal.
    const std::string helpers =
        identity ? std::string()
                 : "__device__ __forceinline__ void jitStoreLe32(u8* b, int "
                   "v)\n"
                   "{ u32 bits = (u32)v;\n"
                   "  b[0] = (u8)bits; b[1] = (u8)(bits >> 8U);\n"
                   "  b[2] = (u8)(bits >> 16U); b[3] = (u8)(bits >> 24U); }\n"
                   "__device__ __forceinline__ void jitStoreLeSigned16(u8* b, "
                   "short v)\n"
                   "{ jitStoreLe16(b, (u16)v); }\n"
                   "__device__ __forceinline__ short jitLegacyScanAngle(u8 "
                   "encoded)\n"
                   "{ const int rank = encoded <= 127U ? (int)encoded : "
                   "(int)encoded - 256;\n"
                   "  const int scaled = rank * 500;\n"
                   "  return (short)(scaled >= 0 ? (scaled + 1) / 3 : (scaled "
                   "- 1) / 3); }\n";
    substitute("%HELPERS%", helpers);
    substitute("%STAGES%", stages);
    substitute("%TRANSLATE%", translate);
    const std::string stride =
        std::to_string(identity ? OutputPointBytes : inputStride);
    while (source.find("%STRIDE%") != std::string::npos)
        substitute("%STRIDE%", stride);
    substitute("%BODY%", body);
    return source;
}

// B0155: NVRTC is 114.5 MB and costs 5.4 ms to load in every process, yet it is
// only needed when D0076's fused JIT actually compiles, which most pipelines
// never do. Resolving it through dlopen on first use keeps that cost off every
// other invocation. The symbol set is small and fixed; a missing library or
// symbol simply means no JIT, which the caller already treats as a compile
// failure and falls back from.
struct NvrtcApi
{
    nvrtcResult (*createProgram)(nvrtcProgram*, const char*, const char*, int,
                                 const char* const*, const char* const*) =
        nullptr;
    nvrtcResult (*compileProgram)(nvrtcProgram, int, const char* const*) =
        nullptr;
    nvrtcResult (*getProgramLogSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*getProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*getPtxSize)(nvrtcProgram, std::size_t*) = nullptr;
    nvrtcResult (*getPtx)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*destroyProgram)(nvrtcProgram*) = nullptr;

    [[nodiscard]] bool valid() const noexcept
    {
        return createProgram && compileProgram && getProgramLogSize &&
               getProgramLog && getPtxSize && getPtx && destroyProgram;
    }
};

const NvrtcApi* loadNvrtc()
{
    static const NvrtcApi* const resolved = []() -> const NvrtcApi*
    {
        static NvrtcApi api;
        // Versioned SONAME first, then the development symlink, so a
        // runtime-only CUDA install resolves without the full toolkit.
        void* handle = dlopen("libnvrtc.so.13", RTLD_LAZY | RTLD_LOCAL);
        if (!handle)
            handle = dlopen("libnvrtc.so", RTLD_LAZY | RTLD_LOCAL);
        if (!handle)
            return nullptr;
        api.createProgram = reinterpret_cast<decltype(api.createProgram)>(
            dlsym(handle, "nvrtcCreateProgram"));
        api.compileProgram = reinterpret_cast<decltype(api.compileProgram)>(
            dlsym(handle, "nvrtcCompileProgram"));
        api.getProgramLogSize =
            reinterpret_cast<decltype(api.getProgramLogSize)>(
                dlsym(handle, "nvrtcGetProgramLogSize"));
        api.getProgramLog = reinterpret_cast<decltype(api.getProgramLog)>(
            dlsym(handle, "nvrtcGetProgramLog"));
        api.getPtxSize = reinterpret_cast<decltype(api.getPtxSize)>(
            dlsym(handle, "nvrtcGetPTXSize"));
        api.getPtx =
            reinterpret_cast<decltype(api.getPtx)>(dlsym(handle, "nvrtcGetPTX"));
        api.destroyProgram = reinterpret_cast<decltype(api.destroyProgram)>(
            dlsym(handle, "nvrtcDestroyProgram"));
        return api.valid() ? &api : nullptr;
    }();
    return resolved;
}

// One compiled specialization per generated source; a failed compilation is
// cached as null so the interpreter fallback is chosen without retrying.
CUfunction jitFusedKernel(const FusedPointProgram& program,
                          std::uint8_t inputFormat, std::size_t inputStride)
{
    // Every host-admitted LAS record format specializes (the waveform
    // formats 4/5 stay outside the native envelope); the stride must
    // cover the format's mandatory fields and one block's input and
    // output spans must fit the staged shared budget the launch sites
    // verify.
    if (inputFormat > 8U || inputFormat == 4U || inputFormat == 5U ||
        inputStride < minimumRecordLength(inputFormat) ||
        static_cast<std::size_t>(StagedBlockSize) *
                (inputStride + OutputPointBytes) >
            48U * 1024U ||
        std::getenv("PDG_DISABLE_FUSED_JIT"))
        return nullptr;
    const std::optional<std::string> source =
        generateFusedKernelSource(program, inputFormat, inputStride);
    if (!source)
        return nullptr;
    static std::mutex mutex;
    static std::map<std::size_t, CUfunction> cache;
    const std::size_t key = std::hash<std::string>{}(*source);
    std::lock_guard<std::mutex> lock(mutex);
    const auto existing = cache.find(key);
    if (existing != cache.end())
        return existing->second;
    const auto fail = [&]() -> CUfunction
    {
        cache.emplace(key, nullptr);
        return nullptr;
    };
    int device = 0;
    int major = 0;
    int minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess)
        return fail();
    const NvrtcApi* nvrtc = loadNvrtc();
    if (!nvrtc)
        return fail();
    nvrtcProgram compiled = nullptr;
    if (nvrtc->createProgram(&compiled, source->c_str(), "pdg_jit_fused.cu", 0,
                             nullptr, nullptr) != NVRTC_SUCCESS)
        return fail();
    const std::string architecture = "--gpu-architecture=compute_" +
                                     std::to_string(major) +
                                     std::to_string(minor);
    const char* options[] = {architecture.c_str(), "--std=c++17"};
    const bool built =
        nvrtc->compileProgram(compiled, 2, options) == NVRTC_SUCCESS;
    if (!built && std::getenv("PDG_DEBUG_FUSED_JIT"))
    {
        std::size_t logBytes = 0;
        std::string log;
        if (nvrtc->getProgramLogSize(compiled, &logBytes) ==
                NVRTC_SUCCESS &&
            logBytes)
        {
            log.resize(logBytes);
            nvrtc->getProgramLog(compiled, log.data());
        }
        std::fprintf(stderr,
                     "pdg fused JIT compile failed:\n%s\n--- source\n%s\n",
                     log.c_str(), source->c_str());
    }
    std::string ptx;
    if (built)
    {
        std::size_t bytes = 0;
        if (nvrtc->getPtxSize(compiled, &bytes) == NVRTC_SUCCESS && bytes)
        {
            ptx.resize(bytes);
            if (nvrtc->getPtx(compiled, ptx.data()) != NVRTC_SUCCESS)
                ptx.clear();
        }
    }
    nvrtc->destroyProgram(&compiled);
    if (ptx.empty())
        return fail();
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (cuModuleLoadData(&module, ptx.c_str()) != CUDA_SUCCESS)
        return fail();
    if (cuModuleGetFunction(&function, module, "pdgJitFusedKernel") !=
        CUDA_SUCCESS)
    {
        cuModuleUnload(module);
        return fail();
    }
    (void)cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
    cache.emplace(key, function);
    return function;
}

} // unnamed namespace

std::optional<FusedPointProgram>
makeFusedPointProgram(const AssignProgram& source,
                      const DimensionRegistry& dimensions)
{
    if (source.assignments.empty() ||
        source.assignments.size() > MaximumFusedAssignments)
        return std::nullopt;

    std::vector<DimensionId> created;
    for (const PointAssignment& assignment : source.assignments)
        if (assignment.destinationCreated &&
            std::find(created.begin(), created.end(), assignment.destination) ==
                created.end())
            created.push_back(assignment.destination);

    FusedPointProgram result;
    const auto bindDimension =
        [&](DimensionId id) -> std::optional<std::uint16_t>
    {
        for (std::uint16_t index = 0; index < result.dimensionCount; ++index)
            if (result.dimensions[index].id == id.value())
                return index;
        if (result.dimensionCount >= MaximumFusedDimensions)
            return std::nullopt;
        const DimensionDefinition& definition = dimensions.require(id);
        FusedDimension binding;
        binding.id = id.value();
        if (definition.standard)
        {
            binding.type = canonicalPhysicalType(id);
            binding.decode = true;
            if (binding.type == DimensionType::None)
                return std::nullopt;
        }
        else
        {
            if (std::find(created.begin(), created.end(), id) == created.end())
                return std::nullopt;
            binding.type = definition.type;
            binding.initializeZero = true;
            if (binding.type == DimensionType::None)
                return std::nullopt;
        }
        const std::uint16_t index = result.dimensionCount++;
        result.dimensions[index] = binding;
        return index;
    };
    const auto appendExpression = [&](const CompiledExpression& expression,
                                      std::uint16_t& begin,
                                      std::uint16_t& count)
    {
        if (expression.maximumStackDepth > MaximumFusedStackDepth ||
            expression.instructions.size() >
                MaximumFusedInstructions - result.instructionCount)
            return false;
        begin = result.instructionCount;
        count = static_cast<std::uint16_t>(expression.instructions.size());
        for (const ExpressionInstruction& instruction : expression.instructions)
        {
            if (!fusedExpressionOperation(instruction.op))
                return false;
            FusedInstruction& target =
                result.instructions[result.instructionCount++];
            target.op = instruction.op;
            target.immediate = instruction.immediate;
            if (instruction.op == ExpressionOp::LoadDimension)
            {
                const std::optional<std::uint16_t> binding =
                    bindDimension(instruction.dimension);
                if (!binding)
                    return false;
                target.dimension = *binding;
            }
        }
        return true;
    };

    for (const PointAssignment& assignment : source.assignments)
    {
        const std::optional<std::uint16_t> destination =
            bindDimension(assignment.destination);
        if (!destination)
            return std::nullopt;
        const DimensionDefinition& definition =
            dimensions.require(assignment.destination);
        if (definition.standard)
        {
            if (assignment.destination == DimensionId(StandardDimension::X) ||
                assignment.destination == DimensionId(StandardDimension::Y) ||
                assignment.destination == DimensionId(StandardDimension::Z) ||
                assignment.destination ==
                    DimensionId(StandardDimension::ScanAngleRank))
                return std::nullopt;
            result.dimensions[*destination].pack = true;
        }
        FusedAssignment& target = result.assignments[result.assignmentCount++];
        target.destination = *destination;
        if (!appendExpression(assignment.value, target.valueBegin,
                              target.valueCount) ||
            !appendExpression(assignment.condition, target.conditionBegin,
                              target.conditionCount))
            return std::nullopt;
    }
    return result;
}

RawSummary initialSummary()
{
    RawSummary summary{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        summary.minimum[axis] = std::numeric_limits<std::int32_t>::max();
        summary.maximum[axis] = std::numeric_limits<std::int32_t>::lowest();
    }
    return summary;
}
} // unnamed namespace

bool supportsDefaultCudaTranslation(const FileView& input) noexcept
{
    if (!supportsDefaultTranslation(input))
        return false;
    const Header& header = input.header();
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (header.scale[axis] != OutputScale || header.offset[axis] != 0.0)
            return false;
    return true;
}

bool supportsDefaultCudaPointProgram(
    const FileView& input, const AssignProgram& program,
    const DimensionRegistry& dimensions) noexcept
{
    OrderedPointProgram ordered;
    ordered.operations.emplace_back(program);
    ordered.reads = program.reads;
    ordered.writes = program.writes;
    return supportsDefaultCudaPointProgram(input, ordered, dimensions);
}

bool supportsDefaultCudaPointProgram(
    const FileView& input, const OrderedPointProgram& program,
    const DimensionRegistry& dimensions) noexcept
{
    try
    {
        if (!supportsDefaultCudaTranslation(input) ||
            !supportsDefaultPointProgram(input, program, dimensions))
            return false;

        DimensionRegistry& mutableDimensions =
            const_cast<DimensionRegistry&>(dimensions);
        HostMemoryResource memory;
        PointBatch layout(
            0,
            CoordinateEncoding{{OutputScale, OutputScale, OutputScale},
                               {0.0, 0.0, 0.0}},
            mutableDimensions, memory);
        std::vector<DimensionId> touched = program.reads;
        const auto appendTouched = [&](DimensionId id)
        {
            if (std::find(touched.begin(), touched.end(), id) == touched.end())
                touched.push_back(id);
        };
        for (DimensionId id : program.writes)
            appendTouched(id);
        for (const PointOperation& operation : program.operations)
        {
            if (const auto* assignments =
                    std::get_if<AssignProgram>(&operation))
            {
                for (DimensionId id : assignments->reads)
                    appendTouched(id);
                for (DimensionId id : assignments->writes)
                    appendTouched(id);
            }
            else if (const auto* predicate =
                         std::get_if<PredicateProgram>(&operation))
                for (DimensionId id : predicate->reads)
                    appendTouched(id);
        }
        for (DimensionId id : touched)
            layout.materialize(id);

        for (const PointOperation& operation : program.operations)
        {
            if (const auto* assignments =
                    std::get_if<AssignProgram>(&operation))
            {
                if (!assignSupportsExactDevice(layout, *assignments))
                    return false;
            }
            else if (const auto* predicate =
                         std::get_if<PredicateProgram>(&operation))
            {
                if (!predicateSupportsExactDevice(layout, *predicate))
                    return false;
            }
            else if (!ordinalSupportsMode(std::get<OrdinalProgram>(operation),
                                          program.ordinalMode))
                return false;
        }

        std::uint64_t ordinalInputTotal = input.header().pointCount;
        bool dataDependentCount = false;
        for (const PointOperation& operation : program.operations)
        {
            if (std::holds_alternative<PredicateProgram>(operation))
            {
                dataDependentCount = true;
                continue;
            }
            const auto* ordinal = std::get_if<OrdinalProgram>(&operation);
            if (!ordinal)
                continue;
            if (program.ordinalMode == OrdinalMode::Standard)
            {
                if (dataDependentCount || (ordinal->kind == OrdinalKind::Tail &&
                                           ordinal->count > ordinalInputTotal))
                    return false;
                static_cast<void>(makeOrdinalState(
                    *ordinal, program.ordinalMode, ordinalInputTotal));
                ordinalInputTotal =
                    ordinalStandardOutputCount(*ordinal, ordinalInputTotal);
            }
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool supportsDefaultFusedCudaPointProgram(
    const FileView& input, const AssignProgram& program,
    const DimensionRegistry& dimensions) noexcept
{
    try
    {
        return supportsDefaultCudaPointProgram(input, program, dimensions) &&
               makeFusedPointProgram(program, dimensions).has_value();
    }
    catch (...)
    {
        return false;
    }
}

bool preferDefaultCudaPointProgram(std::uint64_t pointCount,
                                   const AssignProgram& program) noexcept
{
    // The RTX 4090 crossover matrix showed a 19.5% win at 16M points for the
    // smallest measured shape below. Keep automatic selection inside that
    // measured envelope; explicit experimental/required modes remain wider.
    constexpr std::uint64_t MinimumPoints = 16'000'000;
    constexpr std::size_t MinimumAssignments = 5;
    constexpr std::size_t MinimumInstructions = 28;
    constexpr std::size_t MaximumWrites = 5;
    constexpr std::size_t MaximumTouchedDimensions = 6;
    if (pointCount < MinimumPoints ||
        program.assignments.size() < MinimumAssignments ||
        program.writes.size() > MaximumWrites)
        return false;

    std::size_t instructionCount = 0;
    for (const PointAssignment& assignment : program.assignments)
    {
        const std::size_t count = assignment.value.instructions.size() +
                                  assignment.condition.instructions.size();
        instructionCount =
            std::min(MinimumInstructions, instructionCount + count);
    }
    if (instructionCount < MinimumInstructions)
        return false;

    std::size_t touchedDimensions = program.reads.size();
    for (auto write = program.writes.begin(); write != program.writes.end();
         ++write)
    {
        if (std::find(program.reads.begin(), program.reads.end(), *write) ==
                program.reads.end() &&
            std::find(program.writes.begin(), write, *write) == write)
            ++touchedDimensions;
    }
    return touchedDimensions <= MaximumTouchedDimensions;
}

namespace
{
void appendUnique(std::vector<DimensionId>& dimensions, DimensionId dimension)
{
    if (std::find(dimensions.begin(), dimensions.end(), dimension) ==
        dimensions.end())
        dimensions.push_back(dimension);
}

struct CudaLane
{
    std::unique_ptr<MemoryResource> pinnedMemory;
    std::unique_ptr<Allocation> pinnedInput;
    std::unique_ptr<Allocation> pinnedOutput;
    std::unique_ptr<Allocation> pinnedSummary;
    std::unique_ptr<MemoryResource> deviceMemory;
    std::unique_ptr<Allocation> deviceInput;
    std::unique_ptr<Allocation> deviceOutput;
    std::unique_ptr<Allocation> deviceSummary;
    std::unique_ptr<PointBatch> batch;
    cudaStream_t stream = nullptr;
    std::size_t first = 0;
    std::size_t count = 0;
    bool pending = false;
};

TiledSchedule translateDefaultCudaIntoImpl(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram* program, DimensionRegistry* dimensions,
    std::span<std::byte> output, const CudaTranslationSink* sink,
    std::size_t chunkPoints, std::size_t schedulerLanes,
    std::size_t memoryBudgetBytes, CudaTranslationMetrics* metrics)
{
    if (metrics)
        *metrics = {};
    if (program)
    {
        if (!dimensions ||
            !supportsDefaultCudaPointProgram(input, *program, *dimensions))
            throw Error("LAS point program is outside the exact CUDA envelope");
    }
    else if (!supportsDefaultCudaTranslation(input))
        throw Error("LAS input is outside the exact CUDA translation envelope");
    if (!chunkPoints)
        throw std::invalid_argument("CUDA LAS translation chunk size is zero");

    const std::optional<FusedPointProgram> fusedProgram =
        program ? makeFusedPointProgram(*program, *dimensions) : std::nullopt;

    const std::uint64_t count = input.header().pointCount;
    if (count > (std::numeric_limits<std::size_t>::max() - OutputHeaderBytes) /
                    OutputPointBytes)
        throw Error("CUDA LAS translation output size overflows size_t");
    const std::size_t expectedOutputBytes =
        OutputHeaderBytes + static_cast<std::size_t>(count) * OutputPointBytes;
    if (sink && !*sink)
        throw std::invalid_argument("CUDA LAS translation sink is empty");
    if (!sink && output.size() != expectedOutputBytes)
        throw Error("CUDA LAS translation output has the wrong size");
    std::array<std::byte, OutputHeaderBytes> streamedHeader{};
    std::span<std::byte> header = sink ? std::span<std::byte>(streamedHeader)
                                       : output.first(OutputHeaderBytes);
    std::fill(header.begin(), header.end(), std::byte{});
    if (!count)
    {
        writeHeader(header, count, {}, metadata);
        if (sink)
            (*sink)(0, header);
        return makeTiledSchedule(
            {.pipelineClass = program ? PipelineClass::FusedPointProgram
                                      : PipelineClass::LasTranslation,
             .itemCount = 0U,
             .tileItems = chunkPoints,
             .memoryBudgetBytes = memoryBudgetBytes,
             .requestedLanes = schedulerLanes});
    }

    const std::size_t pointCount = static_cast<std::size_t>(count);
    chunkPoints = std::min(chunkPoints, pointCount);
    const std::size_t inputStride = input.header().pointRecordLength;
    const std::size_t inputCapacity = checkedProduct(
        chunkPoints, inputStride,
        "CUDA LAS translation input staging size overflows size_t");
    const std::size_t outputCapacity = checkedProduct(
        chunkPoints, OutputPointBytes,
        "CUDA LAS translation output staging size overflows size_t");

    std::vector<DimensionId> touched;
    std::vector<DimensionId> decoded;
    std::vector<DimensionId> packed;
    std::size_t columnCapacity = 0;
    bool writesReturnNumber = false;
    if (program && !fusedProgram)
    {
        touched = program->reads;
        for (const DimensionId dimension : program->writes)
            appendUnique(touched, dimension);
        for (const DimensionId dimension : touched)
        {
            const DimensionDefinition& definition =
                dimensions->require(dimension);
            if (definition.standard)
                decoded.push_back(dimension);
            const DimensionType physicalType =
                dimension == DimensionId(StandardDimension::X) ||
                        dimension == DimensionId(StandardDimension::Y) ||
                        dimension == DimensionId(StandardDimension::Z)
                    ? DimensionType::Signed32
                    : definition.type;
            columnCapacity = checkedAdd(
                columnCapacity,
                checkedProduct(
                    chunkPoints, dimensionTypeSize(physicalType),
                    "CUDA point-program column size overflows size_t"),
                "CUDA point-program columns overflow size_t");
        }
        for (const DimensionId dimension : program->writes)
        {
            if (dimensions->require(dimension).standard)
                appendUnique(packed, dimension);
            writesReturnNumber =
                writesReturnNumber ||
                dimension == DimensionId(StandardDimension::ReturnNumber);
        }
    }

    std::size_t deviceCapacity = checkedAdd(
        checkedAdd(inputCapacity, outputCapacity,
                   "CUDA LAS translation device size overflows size_t"),
        sizeof(RawSummary),
        "CUDA LAS translation device size overflows size_t");
    deviceCapacity =
        checkedAdd(deviceCapacity, columnCapacity,
                   "CUDA LAS point-program device size overflows size_t");

    const auto makeLane = [&]()
    {
        auto lane = std::make_unique<CudaLane>();
        lane->pinnedMemory = makeCudaPinnedMemoryResource();
        lane->pinnedInput = lane->pinnedMemory->allocate(
            inputCapacity, alignof(std::max_align_t));
        lane->pinnedOutput = lane->pinnedMemory->allocate(
            outputCapacity, alignof(std::max_align_t));
        lane->pinnedSummary = lane->pinnedMemory->allocate(sizeof(RawSummary),
                                                           alignof(RawSummary));
        lane->deviceMemory = makeCudaMemoryResource(deviceCapacity);
        lane->deviceInput = lane->deviceMemory->allocate(
            inputCapacity, alignof(std::max_align_t));
        lane->deviceOutput = lane->deviceMemory->allocate(
            outputCapacity, alignof(std::max_align_t));
        lane->deviceSummary = lane->deviceMemory->allocate(sizeof(RawSummary),
                                                           alignof(RawSummary));
        lane->stream =
            static_cast<cudaStream_t>(lane->deviceMemory->nativeStreamHandle());
        if (program && !fusedProgram)
        {
            lane->batch = std::make_unique<PointBatch>(
                chunkPoints,
                CoordinateEncoding{{OutputScale, OutputScale, OutputScale},
                                   {0.0, 0.0, 0.0}},
                *dimensions, *lane->deviceMemory);
            for (const DimensionId dimension : touched)
                lane->batch->materialize(dimension);
        }
        return lane;
    };

    const TiledSchedule schedule = makeTiledSchedule(
        {.pipelineClass = program ? PipelineClass::FusedPointProgram
                                  : PipelineClass::LasTranslation,
         .itemCount = pointCount,
         .tileItems = chunkPoints,
         .bytesPerLane = deviceCapacity,
         .memoryBudgetBytes = memoryBudgetBytes,
         .requestedLanes = schedulerLanes});
    const std::size_t laneCount = schedule.activeLaneCount;
    std::vector<std::unique_ptr<CudaLane>> lanes;
    lanes.reserve(laneCount);
    for (std::size_t lane = 0; lane < laneCount; ++lane)
        lanes.push_back(makeLane());

    HostSummary summary;
    const auto finish = [&](CudaLane& lane)
    {
        if (!lane.pending)
            return;
        PDG_CUDA_CHECK(cudaStreamSynchronize(lane.stream));
        const std::size_t offset =
            OutputHeaderBytes + lane.first * OutputPointBytes;
        const std::span<const std::byte> chunk(
            static_cast<const std::byte*>(lane.pinnedOutput->data()),
            lane.count * OutputPointBytes);
        if (sink)
            (*sink)(offset, chunk);
        else
            std::memcpy(output.data() + offset, chunk.data(), chunk.size());
        summary.merge(
            *static_cast<const RawSummary*>(lane.pinnedSummary->data()),
            lane.count != 0);
        lane.pending = false;
    };

    NvtxRange range(fusedProgram
                        ? "pdg::las::fusedTranslateDefaultPointProgramCuda"
                    : program ? "pdg::las::translateDefaultPointProgramCuda"
                              : "pdg::las::translateDefaultCuda");
    // Ask for the full shared-memory carveout so four staged blocks fit per
    // SM; the attribute is a hint and a failure keeps the default carveout.
    static const bool carveoutConfigured = []
    {
        (void)cudaFuncSetAttribute(
            reinterpret_cast<const void*>(
                &fusedTranslatePointProgramStagedKernel),
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        (void)cudaFuncSetAttribute(
            reinterpret_cast<const void*>(&translateStagedKernel),
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        (void)cudaGetLastError();
        return true;
    }();
    (void)carveoutConfigured;
    std::size_t sequence = 0;
    for (std::size_t first = 0; first < pointCount; first += chunkPoints)
    {
        CudaLane& lane = *lanes[sequence % laneCount];
        finish(lane);
        const std::size_t chunkCount =
            std::min(chunkPoints, pointCount - first);
        const std::size_t inputBytes = chunkCount * inputStride;
        const std::size_t outputBytes = chunkCount * OutputPointBytes;
        const std::byte* source = input.bytes().data() +
                                  input.header().pointDataOffset +
                                  first * inputStride;
        std::memcpy(lane.pinnedInput->data(), source, inputBytes);
        *static_cast<RawSummary*>(lane.pinnedSummary->data()) =
            initialSummary();

        PDG_CUDA_CHECK(cudaMemcpyAsync(lane.deviceInput->data(),
                                       lane.pinnedInput->data(), inputBytes,
                                       cudaMemcpyHostToDevice, lane.stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            lane.deviceSummary->data(), lane.pinnedSummary->data(),
            sizeof(RawSummary), cudaMemcpyHostToDevice, lane.stream));
        // Records stage through shared memory when one block's input and
        // output spans fit the default dynamic budget; oversized strides
        // keep the unstaged compatibility kernels.
        const bool stagedInPlace = (input.header().pointFormat == 7U ||
                                    input.header().pointFormat == 8U) &&
                                   inputStride == OutputPointBytes;
        const std::size_t stagedSharedBytes =
            static_cast<std::size_t>(StagedBlockSize) *
            (stagedInPlace ? OutputPointBytes : inputStride + OutputPointBytes);
        const bool staged = stagedSharedBytes <= 48U * 1024U;
        if (fusedProgram)
        {
            const CUfunction jit = jitFusedKernel(
                *fusedProgram, input.header().pointFormat, inputStride);
            if (std::getenv("PDG_REQUIRE_FUSED_JIT") && !jit)
                throw std::runtime_error(
                    "required fused-program specialization was unavailable");
            if (jit)
            {
                const void* jitInput = lane.deviceInput->data();
                void* jitOutput = lane.deviceOutput->data();
                void* jitSummary = lane.deviceSummary->data();
                unsigned long long jitCount = chunkCount;
                void* jitParameters[] = {&jitInput, &jitCount, &jitOutput,
                                         &jitSummary};
                if (cuLaunchKernel(jit,
                                   static_cast<unsigned>(gridSize(chunkCount)),
                                   1U, 1U, StagedBlockSize, 1U, 1U,
                                   static_cast<unsigned>(stagedSharedBytes),
                                   reinterpret_cast<CUstream>(lane.stream),
                                   jitParameters, nullptr) != CUDA_SUCCESS)
                    throw std::runtime_error(
                        "fused-program specialization launch failed");
            }
            else if (staged)
                fusedTranslatePointProgramStagedKernel<<<
                    gridSize(chunkCount), StagedBlockSize, stagedSharedBytes,
                    lane.stream>>>(
                    static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                    inputStride, input.header().pointFormat, chunkCount,
                    static_cast<std::uint8_t*>(lane.deviceOutput->data()),
                    static_cast<RawSummary*>(lane.deviceSummary->data()),
                    *fusedProgram);
            else
                fusedTranslatePointProgramKernel<<<gridSize(chunkCount),
                                                   BlockSize, 0, lane.stream>>>(
                    static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                    inputStride, input.header().pointFormat, chunkCount,
                    static_cast<std::uint8_t*>(lane.deviceOutput->data()),
                    static_cast<RawSummary*>(lane.deviceSummary->data()),
                    *fusedProgram);
            PDG_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            if (staged)
                translateStagedKernel<<<gridSize(chunkCount), StagedBlockSize,
                                        stagedSharedBytes, lane.stream>>>(
                    static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                    inputStride, input.header().pointFormat, chunkCount,
                    static_cast<std::uint8_t*>(lane.deviceOutput->data()),
                    static_cast<RawSummary*>(lane.deviceSummary->data()));
            else
                translateKernel<<<gridSize(chunkCount), BlockSize, 0,
                                  lane.stream>>>(
                    static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                    inputStride, input.header().pointFormat, chunkCount,
                    static_cast<std::uint8_t*>(lane.deviceOutput->data()),
                    static_cast<RawSummary*>(lane.deviceSummary->data()));
            PDG_CUDA_CHECK(cudaGetLastError());
        }
        if (program && !fusedProgram)
        {
            lane.batch->setSize(chunkCount);
            decodeCanonicalColumnsAsync(lane.deviceOutput->data(), chunkCount,
                                        *lane.batch, decoded, lane.stream);
            executeAssign(*lane.batch, *program);
            if (writesReturnNumber)
            {
                constexpr std::size_t ReturnsOffset =
                    offsetof(RawSummary, returns);
                PDG_CUDA_CHECK(cudaMemsetAsync(
                    static_cast<std::byte*>(lane.deviceSummary->data()) +
                        ReturnsOffset,
                    0, sizeof(RawSummary::returns), lane.stream));
                summarizeReturnNumbersKernel<<<gridSize(chunkCount), BlockSize,
                                               0, lane.stream>>>(
                    lane.batch->data<std::uint8_t>(
                        DimensionId(StandardDimension::ReturnNumber)),
                    chunkCount,
                    static_cast<RawSummary*>(lane.deviceSummary->data()));
                PDG_CUDA_CHECK(cudaGetLastError());
            }
            packCanonicalColumnsAsync(*lane.batch, packed, chunkCount,
                                      lane.deviceOutput->data(), lane.stream);
        }
        PDG_CUDA_CHECK(cudaMemcpyAsync(lane.pinnedOutput->data(),
                                       lane.deviceOutput->data(), outputBytes,
                                       cudaMemcpyDeviceToHost, lane.stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            lane.pinnedSummary->data(), lane.deviceSummary->data(),
            sizeof(RawSummary), cudaMemcpyDeviceToHost, lane.stream));
        lane.first = first;
        lane.count = chunkCount;
        lane.pending = true;
        ++sequence;
    }
    for (const auto& lane : lanes)
        finish(*lane);
    writeHeader(header, count, summary, metadata);
    if (sink)
        (*sink)(0, header);
    if (metrics)
    {
        const std::size_t summaryBytes =
            checkedProduct(schedule.tileCount, sizeof(RawSummary),
                           "CUDA LAS summary transfer size overflows size_t");
        metrics->hostToDeviceBytes = checkedAdd(
            checkedProduct(pointCount, inputStride,
                           "CUDA LAS input transfer size overflows size_t"),
            summaryBytes, "CUDA LAS input transfers overflow size_t");
        metrics->deviceToHostBytes = checkedAdd(
            checkedProduct(pointCount, OutputPointBytes,
                           "CUDA LAS output transfer size overflows size_t"),
            summaryBytes, "CUDA LAS output transfers overflow size_t");
    }
    return schedule;
}

struct OrderedCudaLane
{
    std::unique_ptr<MemoryResource> pinnedMemory;
    std::unique_ptr<Allocation> pinnedInput;
    std::unique_ptr<Allocation> pinnedOutput;
    std::unique_ptr<Allocation> pinnedSummary;
    std::unique_ptr<MemoryResource> deviceMemory;
    std::unique_ptr<Allocation> deviceInput;
    std::unique_ptr<Allocation> deviceCanonical;
    std::unique_ptr<Allocation> deviceFiltered;
    std::unique_ptr<Allocation> deviceSummary;
    std::unique_ptr<Allocation> deviceKeep;
    std::unique_ptr<PointBatch> firstBatch;
    std::unique_ptr<PointBatch> secondBatch;
    cudaStream_t stream = nullptr;
    std::size_t sequence = 0;
    std::size_t count = 0;
    bool pending = false;
};

std::uint64_t translateDefaultOrderedCudaToSinkImpl(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const OrderedPointProgram& program, DimensionRegistry& dimensions,
    const CudaTranslationSink& sink, std::size_t chunkPoints,
    std::size_t schedulerLanes, std::size_t memoryBudgetBytes,
    CudaTranslationMetrics* metrics, TiledSchedule* scheduleOut)
{
    if (metrics)
        *metrics = {};
    if (scheduleOut)
        *scheduleOut = {};
    if (!supportsDefaultCudaPointProgram(input, program, dimensions))
        throw Error("ordered LAS point program is outside the exact CUDA "
                    "envelope");
    if (!sink)
        throw std::invalid_argument("CUDA LAS translation sink is empty");
    if (!chunkPoints)
        throw std::invalid_argument("CUDA LAS translation chunk size is zero");

    const std::uint64_t inputCount = input.header().pointCount;
    std::array<std::byte, OutputHeaderBytes> header{};
    if (!inputCount)
    {
        writeHeader(header, 0, {}, metadata);
        sink(0, header);
        return 0;
    }
    if (inputCount > std::numeric_limits<std::size_t>::max())
        throw Error("CUDA LAS point count exceeds size_t");

    const std::size_t pointCount = static_cast<std::size_t>(inputCount);
    chunkPoints = std::min(chunkPoints, pointCount);
    const std::size_t inputStride = input.header().pointRecordLength;
    const std::size_t inputCapacity =
        checkedProduct(chunkPoints, inputStride,
                       "ordered CUDA LAS input staging size overflows size_t");
    const std::size_t outputCapacity =
        checkedProduct(chunkPoints, OutputPointBytes,
                       "ordered CUDA LAS output staging size overflows size_t");

    std::vector<DimensionId> touched = program.reads;
    std::vector<DimensionId> written = program.writes;
    for (DimensionId id : written)
        appendUnique(touched, id);
    bool hasPredicate = false;
    bool hasOrdinal = false;
    for (const PointOperation& operation : program.operations)
    {
        if (const auto* assignments = std::get_if<AssignProgram>(&operation))
        {
            for (DimensionId id : assignments->reads)
                appendUnique(touched, id);
            for (DimensionId id : assignments->writes)
            {
                appendUnique(touched, id);
                appendUnique(written, id);
            }
        }
        else if (const auto* predicate =
                     std::get_if<PredicateProgram>(&operation))
        {
            hasPredicate = true;
            for (DimensionId id : predicate->reads)
                appendUnique(touched, id);
        }
        else
        {
            hasPredicate = true;
            hasOrdinal = true;
        }
    }

    std::vector<DimensionId> decoded;
    std::vector<DimensionId> packed;
    for (DimensionId id : touched)
        if (dimensions.require(id).standard)
            decoded.push_back(id);
    for (DimensionId id : written)
        if (dimensions.require(id).standard)
            appendUnique(packed, id);

    std::string indexName = "PdgInternalOriginalIndex";
    std::size_t indexSuffix = 0;
    while (dimensions.find(indexName))
        indexName = "PdgInternalOriginalIndex" + std::to_string(++indexSuffix);
    const DimensionId indexId =
        dimensions.registerCustom(indexName, DimensionType::Unsigned64).id;
    std::vector<DimensionId> compacted = touched;
    appendUnique(compacted, indexId);

    std::size_t columnCapacity = 0;
    for (DimensionId id : compacted)
    {
        const DimensionDefinition& definition = dimensions.require(id);
        const DimensionType physicalType =
            id == DimensionId(StandardDimension::X) ||
                    id == DimensionId(StandardDimension::Y) ||
                    id == DimensionId(StandardDimension::Z)
                ? DimensionType::Signed32
                : definition.type;
        columnCapacity = checkedAdd(
            columnCapacity,
            checkedProduct(chunkPoints, dimensionTypeSize(physicalType),
                           "ordered CUDA point-program column size overflows "
                           "size_t"),
            "ordered CUDA point-program columns overflow size_t");
    }
    if (hasPredicate)
        columnCapacity = checkedProduct(
            columnCapacity, 2,
            "ordered CUDA point-program double-buffer size overflows size_t");

    std::size_t deviceCapacity = checkedAdd(
        checkedAdd(inputCapacity, outputCapacity,
                   "ordered CUDA LAS device size overflows size_t"),
        outputCapacity, "ordered CUDA LAS device size overflows size_t");
    deviceCapacity =
        checkedAdd(deviceCapacity, sizeof(RawSummary),
                   "ordered CUDA LAS device size overflows size_t");
    deviceCapacity =
        checkedAdd(deviceCapacity, columnCapacity,
                   "ordered CUDA point-program device size overflows size_t");
    if (hasPredicate)
        deviceCapacity =
            checkedAdd(deviceCapacity, chunkPoints,
                       "ordered CUDA predicate storage overflows size_t");

    const auto makeLane = [&]()
    {
        auto lane = std::make_unique<OrderedCudaLane>();
        lane->pinnedMemory = makeCudaPinnedMemoryResource();
        lane->pinnedInput = lane->pinnedMemory->allocate(
            inputCapacity, alignof(std::max_align_t));
        lane->pinnedOutput = lane->pinnedMemory->allocate(
            outputCapacity, alignof(std::max_align_t));
        lane->pinnedSummary = lane->pinnedMemory->allocate(sizeof(RawSummary),
                                                           alignof(RawSummary));
        lane->deviceMemory = makeCudaMemoryResource(deviceCapacity);
        lane->deviceInput = lane->deviceMemory->allocate(
            inputCapacity, alignof(std::max_align_t));
        lane->deviceCanonical = lane->deviceMemory->allocate(
            outputCapacity, alignof(std::max_align_t));
        lane->deviceFiltered = lane->deviceMemory->allocate(
            outputCapacity, alignof(std::max_align_t));
        lane->deviceSummary = lane->deviceMemory->allocate(sizeof(RawSummary),
                                                           alignof(RawSummary));
        if (hasPredicate)
            lane->deviceKeep = lane->deviceMemory->allocate(
                chunkPoints, alignof(std::uint8_t));
        lane->stream =
            static_cast<cudaStream_t>(lane->deviceMemory->nativeStreamHandle());
        lane->firstBatch = std::make_unique<PointBatch>(
            chunkPoints,
            CoordinateEncoding{{OutputScale, OutputScale, OutputScale},
                               {0.0, 0.0, 0.0}},
            dimensions, *lane->deviceMemory);
        for (DimensionId id : compacted)
            lane->firstBatch->materialize(id);
        if (hasPredicate)
        {
            lane->secondBatch = std::make_unique<PointBatch>(
                chunkPoints,
                CoordinateEncoding{{OutputScale, OutputScale, OutputScale},
                                   {0.0, 0.0, 0.0}},
                dimensions, *lane->deviceMemory);
            for (DimensionId id : compacted)
                lane->secondBatch->materialize(id);
        }
        return lane;
    };

    const TiledSchedule schedule =
        makeTiledSchedule({.pipelineClass = PipelineClass::OrderedPointProgram,
                           .itemCount = pointCount,
                           .tileItems = chunkPoints,
                           .bytesPerLane = deviceCapacity,
                           .memoryBudgetBytes = memoryBudgetBytes,
                           .requestedLanes = schedulerLanes,
                           .serialDependency = hasOrdinal});
    if (scheduleOut)
        *scheduleOut = schedule;
    const std::size_t laneCount = schedule.activeLaneCount;
    std::vector<std::unique_ptr<OrderedCudaLane>> lanes;
    lanes.reserve(laneCount);
    for (std::size_t lane = 0; lane < laneCount; ++lane)
        lanes.push_back(makeLane());

    HostSummary summary;
    std::size_t hostToDeviceBytes = 0;
    std::size_t deviceToHostBytes = 0;
    std::vector<std::optional<OrdinalState>> ordinalStates(
        program.operations.size());
    std::uint64_t ordinalInputTotal = inputCount;
    bool dataDependentCount = false;
    for (std::size_t index = 0; index < program.operations.size(); ++index)
    {
        const PointOperation& operation = program.operations[index];
        if (std::holds_alternative<PredicateProgram>(operation))
        {
            dataDependentCount = true;
            continue;
        }
        const auto* ordinal = std::get_if<OrdinalProgram>(&operation);
        if (!ordinal)
            continue;
        if (program.ordinalMode == OrdinalMode::Standard && dataDependentCount)
            throw Error("standard ordinal stage follows a data-dependent "
                        "predicate inside the direct CUDA region");
        ordinalStates[index] =
            makeOrdinalState(*ordinal, program.ordinalMode, ordinalInputTotal);
        if (program.ordinalMode == OrdinalMode::Standard)
            ordinalInputTotal =
                ordinalStandardOutputCount(*ordinal, ordinalInputTotal);
    }
    std::uint64_t emittedPointCount = 0;
    const auto finish = [&](OrderedCudaLane& lane)
    {
        if (!lane.pending)
            return;
        PDG_CUDA_CHECK(cudaStreamSynchronize(lane.stream));
        const std::size_t emitted = static_cast<std::size_t>(emittedPointCount);
        const std::size_t offset =
            OutputHeaderBytes + emitted * OutputPointBytes;
        const std::span<const std::byte> chunk(
            static_cast<const std::byte*>(lane.pinnedOutput->data()),
            lane.count * OutputPointBytes);
        sink(offset, chunk);
        summary.merge(
            *static_cast<const RawSummary*>(lane.pinnedSummary->data()),
            lane.count != 0);
        emittedPointCount += lane.count;
        lane.pending = false;
    };

    NvtxRange range("pdg::las::translateDefaultOrderedPointProgramCuda");
    std::size_t sequence = 0;
    for (std::size_t first = 0; first < pointCount; first += chunkPoints)
    {
        OrderedCudaLane& lane = *lanes[sequence % laneCount];
        finish(lane);
        const std::size_t chunkCount =
            std::min(chunkPoints, pointCount - first);
        const std::size_t inputBytes = chunkCount * inputStride;
        const std::byte* source = input.bytes().data() +
                                  input.header().pointDataOffset +
                                  first * inputStride;
        std::memcpy(lane.pinnedInput->data(), source, inputBytes);
        *static_cast<RawSummary*>(lane.pinnedSummary->data()) =
            initialSummary();

        PDG_CUDA_CHECK(cudaMemcpyAsync(lane.deviceInput->data(),
                                       lane.pinnedInput->data(), inputBytes,
                                       cudaMemcpyHostToDevice, lane.stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            lane.deviceSummary->data(), lane.pinnedSummary->data(),
            sizeof(RawSummary), cudaMemcpyHostToDevice, lane.stream));
        hostToDeviceBytes = checkedAdd(
            hostToDeviceBytes, checkedAdd(inputBytes, sizeof(RawSummary), ""),
            "ordered CUDA H2D metric overflows size_t");
        const bool stagedInPlace = (input.header().pointFormat == 7U ||
                                    input.header().pointFormat == 8U) &&
                                   inputStride == OutputPointBytes;
        const std::size_t stagedSharedBytes =
            static_cast<std::size_t>(StagedBlockSize) *
            (stagedInPlace ? OutputPointBytes : inputStride + OutputPointBytes);
        if (stagedSharedBytes <= 48U * 1024U)
            translateStagedKernel<<<gridSize(chunkCount), StagedBlockSize,
                                    stagedSharedBytes, lane.stream>>>(
                static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                inputStride, input.header().pointFormat, chunkCount,
                static_cast<std::uint8_t*>(lane.deviceCanonical->data()),
                static_cast<RawSummary*>(lane.deviceSummary->data()));
        else
            translateKernel<<<gridSize(chunkCount), BlockSize, 0,
                              lane.stream>>>(
                static_cast<const std::uint8_t*>(lane.deviceInput->data()),
                inputStride, input.header().pointFormat, chunkCount,
                static_cast<std::uint8_t*>(lane.deviceCanonical->data()),
                static_cast<RawSummary*>(lane.deviceSummary->data()));
        PDG_CUDA_CHECK(cudaGetLastError());

        PointBatch* active = lane.firstBatch.get();
        PointBatch* spare = lane.secondBatch.get();
        active->setSize(chunkCount);
        decodeCanonicalColumnsAsync(lane.deviceCanonical->data(), chunkCount,
                                    *active, decoded, lane.stream);
        initializeIndexesKernel<<<gridSize(chunkCount), BlockSize, 0,
                                  lane.stream>>>(
            active->data<std::uint64_t>(indexId), chunkCount);
        PDG_CUDA_CHECK(cudaGetLastError());
        for (std::size_t operationIndex = 0;
             operationIndex < program.operations.size(); ++operationIndex)
        {
            const PointOperation& operation =
                program.operations[operationIndex];
            if (const auto* assignments =
                    std::get_if<AssignProgram>(&operation))
                executeAssign(*active, *assignments);
            else if (const auto* predicate =
                         std::get_if<PredicateProgram>(&operation))
            {
                auto* keep =
                    static_cast<std::uint8_t*>(lane.deviceKeep->data());
                evaluatePredicate(*active, *predicate, keep);
                static_cast<void>(
                    compactPointBatch(*active, *spare, compacted, keep));
                std::swap(active, spare);
            }
            else
            {
                if (!ordinalStates[operationIndex])
                    throw Error("ordered CUDA ordinal state is missing");
                auto* keep =
                    static_cast<std::uint8_t*>(lane.deviceKeep->data());
                evaluateOrdinal(*active, std::get<OrdinalProgram>(operation),
                                *ordinalStates[operationIndex], keep);
                static_cast<void>(
                    compactPointBatch(*active, *spare, compacted, keep));
                std::swap(active, spare);
            }
        }

        if (active->size())
        {
            gatherCanonicalRecordsKernel<<<gridSize(active->size()), BlockSize,
                                           0, lane.stream>>>(
                static_cast<const std::uint8_t*>(lane.deviceCanonical->data()),
                active->data<std::uint64_t>(indexId), active->size(),
                static_cast<std::uint8_t*>(lane.deviceFiltered->data()));
            PDG_CUDA_CHECK(cudaGetLastError());
            packCanonicalColumnsAsync(*active, packed, active->size(),
                                      lane.deviceFiltered->data(), lane.stream);
        }

        *static_cast<RawSummary*>(lane.pinnedSummary->data()) =
            initialSummary();
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            lane.deviceSummary->data(), lane.pinnedSummary->data(),
            sizeof(RawSummary), cudaMemcpyHostToDevice, lane.stream));
        hostToDeviceBytes =
            checkedAdd(hostToDeviceBytes, sizeof(RawSummary),
                       "ordered CUDA H2D metric overflows size_t");
        if (active->size())
        {
            summarizeCanonicalRecordsKernel<<<gridSize(active->size()),
                                              BlockSize, 0, lane.stream>>>(
                static_cast<const std::uint8_t*>(lane.deviceFiltered->data()),
                active->size(),
                static_cast<RawSummary*>(lane.deviceSummary->data()));
            PDG_CUDA_CHECK(cudaGetLastError());
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                lane.pinnedOutput->data(), lane.deviceFiltered->data(),
                active->size() * OutputPointBytes, cudaMemcpyDeviceToHost,
                lane.stream));
            deviceToHostBytes =
                checkedAdd(deviceToHostBytes, active->size() * OutputPointBytes,
                           "ordered CUDA D2H metric overflows size_t");
        }
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            lane.pinnedSummary->data(), lane.deviceSummary->data(),
            sizeof(RawSummary), cudaMemcpyDeviceToHost, lane.stream));
        deviceToHostBytes =
            checkedAdd(deviceToHostBytes, sizeof(RawSummary),
                       "ordered CUDA D2H metric overflows size_t");
        lane.sequence = sequence;
        lane.count = active->size();
        lane.pending = true;
        ++sequence;
    }

    std::vector<OrderedCudaLane*> pending;
    for (const auto& lane : lanes)
        if (lane->pending)
            pending.push_back(lane.get());
    std::sort(pending.begin(), pending.end(),
              [](const OrderedCudaLane* left, const OrderedCudaLane* right)
              { return left->sequence < right->sequence; });
    for (OrderedCudaLane* lane : pending)
        finish(*lane);

    writeHeader(header, emittedPointCount, summary, metadata);
    sink(0, header);
    if (metrics)
    {
        metrics->hostToDeviceBytes = hostToDeviceBytes;
        metrics->deviceToHostBytes = deviceToHostBytes;
    }
    return emittedPointCount;
}
} // unnamed namespace

std::vector<std::byte>
translateDefaultCuda(const FileView& input,
                     const DefaultTranslationMetadata& metadata,
                     std::size_t chunkPoints, std::size_t schedulerLanes)
{
    std::vector<std::byte> output(defaultTranslationSize(input));
    translateDefaultCudaIntoImpl(input, metadata, nullptr, nullptr, output,
                                 nullptr, chunkPoints, schedulerLanes, 0U,
                                 nullptr);
    return output;
}

void translateDefaultCudaInto(const FileView& input,
                              const DefaultTranslationMetadata& metadata,
                              std::span<std::byte> output,
                              std::size_t chunkPoints,
                              std::size_t schedulerLanes)
{
    translateDefaultCudaIntoImpl(input, metadata, nullptr, nullptr, output,
                                 nullptr, chunkPoints, schedulerLanes, 0U,
                                 nullptr);
}

std::vector<std::byte> translateDefaultPointProgramCuda(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    std::size_t chunkPoints, std::size_t schedulerLanes)
{
    std::vector<std::byte> output(defaultTranslationSize(input));
    translateDefaultCudaIntoImpl(input, metadata, &program, &dimensions, output,
                                 nullptr, chunkPoints, schedulerLanes, 0U,
                                 nullptr);
    return output;
}

void translateDefaultPointProgramCudaInto(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    std::span<std::byte> output, std::size_t chunkPoints,
    std::size_t schedulerLanes)
{
    translateDefaultCudaIntoImpl(input, metadata, &program, &dimensions, output,
                                 nullptr, chunkPoints, schedulerLanes, 0U,
                                 nullptr);
}

TiledSchedule translateDefaultPointProgramCudaToSink(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    const CudaTranslationSink& sink, std::size_t chunkPoints,
    std::size_t schedulerLanes, std::size_t memoryBudgetBytes,
    CudaTranslationMetrics* metrics)
{
    return translateDefaultCudaIntoImpl(input, metadata, &program, &dimensions,
                                        {}, &sink, chunkPoints, schedulerLanes,
                                        memoryBudgetBytes, metrics);
}

std::uint64_t translateDefaultOrderedPointProgramCudaToSink(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const OrderedPointProgram& program, DimensionRegistry& dimensions,
    const CudaTranslationSink& sink, std::size_t chunkPoints,
    std::size_t schedulerLanes, std::size_t memoryBudgetBytes,
    CudaTranslationMetrics* metrics, TiledSchedule* scheduleOut)
{
    return translateDefaultOrderedCudaToSinkImpl(
        input, metadata, program, dimensions, sink, chunkPoints, schedulerLanes,
        memoryBudgetBytes, metrics, scheduleOut);
}

} // namespace pdg::las
