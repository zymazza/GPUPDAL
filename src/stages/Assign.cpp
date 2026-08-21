#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pdg
{

void executeAssignDevice(PointBatch& batch, const AssignProgram& program);
void evaluatePredicateDevice(PointBatch& batch, const PredicateProgram& program,
                             std::uint8_t* keep);

namespace
{
constexpr std::size_t MaximumDeviceInstructionsPerAssignment = 96;
constexpr std::size_t MaximumDeviceStackDepth = 32;

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

bool exactDeviceOperation(ExpressionOp op) noexcept
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

bool exactDeviceExpression(const CompiledExpression& expression,
                           bool allowCoordinateLoads) noexcept
{
    if (expression.instructions.size() >
            MaximumDeviceInstructionsPerAssignment ||
        expression.maximumStackDepth > MaximumDeviceStackDepth)
        return false;
    for (const ExpressionInstruction& instruction : expression.instructions)
    {
        if (!exactDeviceOperation(instruction.op) ||
            (instruction.op == ExpressionOp::LoadDimension &&
             coordinateAxis(instruction.dimension) >= 0 &&
             !allowCoordinateLoads))
            return false;
    }
    return true;
}

double loadPhysical(const void* data, DimensionType type, std::size_t index)
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
        break;
    }
    throw std::invalid_argument("assignment source has no physical type");
}

double symmetricRound(double value)
{
    return value > 0.0 ? std::floor(value + 0.5) : std::ceil(value - 0.5);
}

double isMaximumValue(double value)
{
    return static_cast<double>(value == std::numeric_limits<double>::max());
}

template <typename T>
void storeIntegral(void* data, std::size_t index, double value)
{
    const double rounded = symmetricRound(value);
    if (rounded >= static_cast<double>(std::numeric_limits<T>::lowest()) &&
        rounded <= static_cast<double>(std::numeric_limits<T>::max()))
        static_cast<T*>(data)[index] = static_cast<T>(rounded);
}

void storePhysical(void* data, DimensionType type, std::size_t index,
                   double value)
{
    switch (type)
    {
    case DimensionType::Signed8:
        storeIntegral<std::int8_t>(data, index, value);
        return;
    case DimensionType::Signed16:
        storeIntegral<std::int16_t>(data, index, value);
        return;
    case DimensionType::Signed32:
        storeIntegral<std::int32_t>(data, index, value);
        return;
    case DimensionType::Signed64:
        storeIntegral<std::int64_t>(data, index, value);
        return;
    case DimensionType::Unsigned8:
        storeIntegral<std::uint8_t>(data, index, value);
        return;
    case DimensionType::Unsigned16:
        storeIntegral<std::uint16_t>(data, index, value);
        return;
    case DimensionType::Unsigned32:
        storeIntegral<std::uint32_t>(data, index, value);
        return;
    case DimensionType::Unsigned64:
        storeIntegral<std::uint64_t>(data, index, value);
        return;
    case DimensionType::Float:
        if (std::isnan(value) ||
            (value >=
                 static_cast<double>(std::numeric_limits<float>::lowest()) &&
             value <= static_cast<double>(std::numeric_limits<float>::max())))
            static_cast<float*>(data)[index] = static_cast<float>(value);
        return;
    case DimensionType::Double:
        static_cast<double*>(data)[index] = value;
        return;
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument("assignment destination has no physical type");
}

struct ColumnBinding
{
    void* data = nullptr;
    DimensionType type = DimensionType::None;
    int coordinateAxis = -1;
};

struct BoundInstruction
{
    ExpressionOp op = ExpressionOp::PushConstant;
    ColumnBinding source;
    double immediate = 0.0;
};

struct BoundExpression
{
    std::vector<BoundInstruction> instructions;
    std::size_t maximumStackDepth = 0;
};

struct BoundAssignment
{
    ColumnBinding destination;
    BoundExpression value;
    BoundExpression condition;
};

using BindingMap = std::unordered_map<std::uint32_t, ColumnBinding>;

ColumnBinding binding(PointBatch& batch, DimensionId id)
{
    const ColumnInfo& column = batch.columnInfo(id);
    return {batch.rawData(id), column.physicalType, coordinateAxis(id)};
}

BoundExpression bindExpression(const CompiledExpression& expression,
                               const BindingMap& columns)
{
    BoundExpression bound;
    bound.maximumStackDepth = expression.maximumStackDepth;
    bound.instructions.reserve(expression.instructions.size());
    for (const ExpressionInstruction& instruction : expression.instructions)
    {
        BoundInstruction result;
        result.op = instruction.op;
        result.immediate = instruction.immediate;
        if (instruction.op == ExpressionOp::LoadDimension)
        {
            const auto position = columns.find(instruction.dimension.value());
            if (position == columns.end())
                throw std::invalid_argument(
                    "assignment source column is not materialized");
            result.source = position->second;
        }
        bound.instructions.push_back(result);
    }
    return bound;
}

std::vector<BoundAssignment> bindHost(PointBatch& batch,
                                      const AssignProgram& program)
{
    for (const PointAssignment& assignment : program.assignments)
    {
        const bool present = batch.has(assignment.destination);
        if (!present)
            batch.materialize(assignment.destination);
        if (assignment.destinationCreated || !present)
        {
            const ColumnInfo& destination =
                batch.columnInfo(assignment.destination);
            std::memset(batch.rawData(assignment.destination), 0,
                        batch.size() *
                            dimensionTypeSize(destination.physicalType));
        }
    }

    BindingMap columns;
    for (const PointAssignment& assignment : program.assignments)
        columns.emplace(assignment.destination.value(),
                        binding(batch, assignment.destination));
    for (DimensionId id : program.reads)
    {
        if (!batch.has(id))
            throw std::invalid_argument(
                "assignment source column is not materialized");
        columns.emplace(id.value(), binding(batch, id));
    }

    std::vector<BoundAssignment> assignments;
    assignments.reserve(program.assignments.size());
    for (const PointAssignment& assignment : program.assignments)
        assignments.push_back({binding(batch, assignment.destination),
                               bindExpression(assignment.value, columns),
                               bindExpression(assignment.condition, columns)});
    return assignments;
}

double loadLogical(const ColumnBinding& column, std::size_t index,
                   const CoordinateEncoding& coordinates)
{
    if (column.coordinateAxis >= 0 && column.type == DimensionType::Signed32)
        return coordinates.decode(
            static_cast<std::size_t>(column.coordinateAxis),
            static_cast<const std::int32_t*>(column.data)[index]);
    return loadPhysical(column.data, column.type, index);
}

void storeLogical(const ColumnBinding& column, std::size_t index, double value,
                  const CoordinateEncoding& coordinates)
{
    if (column.coordinateAxis >= 0 && column.type == DimensionType::Signed32)
    {
        static_cast<std::int32_t*>(column.data)[index] = coordinates.encode(
            static_cast<std::size_t>(column.coordinateAxis), value);
        return;
    }
    storePhysical(column.data, column.type, index, value);
}

double evaluate(const BoundExpression& expression, std::size_t point,
                const CoordinateEncoding& coordinates,
                std::vector<double>& stack)
{
    std::size_t size = 0;
    const auto unary = [&](auto function)
    { stack[size - 1U] = function(stack[size - 1U]); };
    const auto binary = [&](auto function)
    {
        stack[size - 2U] = function(stack[size - 2U], stack[size - 1U]);
        --size;
    };

    for (const BoundInstruction& instruction : expression.instructions)
    {
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
            stack[size++] = loadLogical(instruction.source, point, coordinates);
            break;
        case ExpressionOp::Add:
            binary([](double left, double right) { return left + right; });
            break;
        case ExpressionOp::Subtract:
            binary([](double left, double right) { return left - right; });
            break;
        case ExpressionOp::Multiply:
            binary([](double left, double right) { return left * right; });
            break;
        case ExpressionOp::Divide:
            binary(
                [](double left, double right)
                {
                    return right == 0.0
                               ? std::numeric_limits<double>::quiet_NaN()
                               : left / right;
                });
            break;
        case ExpressionOp::Negative:
            unary([](double value) { return -value; });
            break;
        case ExpressionOp::Floor:
            unary([](double value) { return std::floor(value); });
            break;
        case ExpressionOp::Ceil:
            unary([](double value) { return std::ceil(value); });
            break;
        case ExpressionOp::Round:
            unary([](double value) { return std::round(value); });
            break;
        case ExpressionOp::Absolute:
            unary([](double value) { return std::fabs(value); });
            break;
        case ExpressionOp::SquareRoot:
            unary([](double value) { return std::sqrt(value); });
            break;
        case ExpressionOp::Sine:
            unary([](double value) { return std::sin(value); });
            break;
        case ExpressionOp::Cosine:
            unary([](double value) { return std::cos(value); });
            break;
        case ExpressionOp::Tangent:
            unary([](double value) { return std::tan(value); });
            break;
        case ExpressionOp::ArcSine:
            unary([](double value) { return std::asin(value); });
            break;
        case ExpressionOp::ArcCosine:
            unary([](double value) { return std::acos(value); });
            break;
        case ExpressionOp::ArcTangent:
            unary([](double value) { return std::atan(value); });
            break;
        case ExpressionOp::HyperbolicSine:
            unary([](double value) { return std::sinh(value); });
            break;
        case ExpressionOp::HyperbolicCosine:
            unary([](double value) { return std::cosh(value); });
            break;
        case ExpressionOp::HyperbolicTangent:
            unary([](double value) { return std::tanh(value); });
            break;
        case ExpressionOp::InverseHyperbolicSine:
            unary([](double value) { return std::asinh(value); });
            break;
        case ExpressionOp::InverseHyperbolicCosine:
            unary([](double value) { return std::acosh(value); });
            break;
        case ExpressionOp::NaturalLog:
            unary([](double value) { return std::log(value); });
            break;
        case ExpressionOp::Log2:
            unary([](double value) { return std::log2(value); });
            break;
        case ExpressionOp::Log10:
            unary([](double value) { return std::log10(value); });
            break;
        case ExpressionOp::Exponential:
            unary([](double value) { return std::exp(value); });
            break;
        case ExpressionOp::Exponential2:
            unary([](double value) { return std::exp2(value); });
            break;
        case ExpressionOp::Equal:
            binary([](double left, double right)
                   { return static_cast<double>(left == right); });
            break;
        case ExpressionOp::NotEqual:
            binary([](double left, double right)
                   { return static_cast<double>(left != right); });
            break;
        case ExpressionOp::Greater:
            binary([](double left, double right)
                   { return static_cast<double>(left > right); });
            break;
        case ExpressionOp::GreaterEqual:
            binary([](double left, double right)
                   { return static_cast<double>(left >= right); });
            break;
        case ExpressionOp::Less:
            binary([](double left, double right)
                   { return static_cast<double>(left < right); });
            break;
        case ExpressionOp::LessEqual:
            binary([](double left, double right)
                   { return static_cast<double>(left <= right); });
            break;
        case ExpressionOp::LogicalNot:
            unary([](double value) { return static_cast<double>(!value); });
            break;
        case ExpressionOp::LogicalAnd:
            binary(
                [](double left, double right)
                { return static_cast<double>(left != 0.0 && right != 0.0); });
            break;
        case ExpressionOp::LogicalOr:
            binary(
                [](double left, double right)
                { return static_cast<double>(left != 0.0 || right != 0.0); });
            break;
        case ExpressionOp::IsNan:
            unary([](double value)
                  { return static_cast<double>(std::isnan(value)); });
            break;
        case ExpressionOp::IsMaximum:
            unary(isMaximumValue);
            break;
        case ExpressionOp::IsMinimum:
            unary(
                [](double value)
                {
                    return static_cast<double>(
                        value == std::numeric_limits<double>::lowest());
                });
            break;
        }
    }
    if (size != 1)
        throw std::logic_error("assignment bytecode stack is unbalanced");
    return stack[0];
}

void executeRange(const std::vector<BoundAssignment>& assignments,
                  const CoordinateEncoding& coordinates, std::size_t begin,
                  std::size_t end, std::size_t maximumStackDepth)
{
    std::vector<double> stack(std::max<std::size_t>(1, maximumStackDepth));
    for (std::size_t point = begin; point < end; ++point)
        for (const BoundAssignment& assignment : assignments)
        {
            if (assignment.value.instructions.empty())
                continue;
            if (!assignment.condition.instructions.empty() &&
                evaluate(assignment.condition, point, coordinates, stack) ==
                    0.0)
                continue;
            const double value =
                evaluate(assignment.value, point, coordinates, stack);
            storeLogical(assignment.destination, point, value, coordinates);
        }
}

void evaluatePredicateRange(const BoundExpression& expression,
                            const CoordinateEncoding& coordinates,
                            std::size_t begin, std::size_t end,
                            std::uint8_t* keep)
{
    std::vector<double> stack(
        std::max<std::size_t>(1, expression.maximumStackDepth));
    for (std::size_t point = begin; point < end; ++point)
        keep[point] = static_cast<std::uint8_t>(
            evaluate(expression, point, coordinates, stack) != 0.0);
}
} // unnamed namespace

bool assignSupportsExactDevice(const AssignProgram& program) noexcept
{
    for (const PointAssignment& assignment : program.assignments)
        if (coordinateAxis(assignment.destination) >= 0 ||
            !exactDeviceExpression(assignment.value, false) ||
            !exactDeviceExpression(assignment.condition, false) ||
            assignment.value.instructions.size() +
                    assignment.condition.instructions.size() >
                MaximumDeviceInstructionsPerAssignment)
            return false;
    return true;
}

bool assignSupportsExactDevice(const PointBatch& batch,
                               const AssignProgram& program) noexcept
{
    const auto coordinateDestinationIsDouble = [&](DimensionId id)
    {
        return coordinateAxis(id) < 0 ||
               (batch.has(id) &&
                batch.columnInfo(id).physicalType == DimensionType::Double);
    };
    const auto coordinateSourceIsExact = [&](DimensionId id)
    {
        const int axis = coordinateAxis(id);
        if (axis < 0)
            return true;
        if (!batch.has(id))
            return false;
        const DimensionType type = batch.columnInfo(id).physicalType;
        return type == DimensionType::Double ||
               (type == DimensionType::Signed32 &&
                batch.coordinateEncoding()
                        .offset()[static_cast<std::size_t>(axis)] == 0.0);
    };
    const auto expressionCoordinatesAreDouble =
        [&](const CompiledExpression& expression)
    {
        for (const ExpressionInstruction& instruction : expression.instructions)
            if (instruction.op == ExpressionOp::LoadDimension &&
                !coordinateSourceIsExact(instruction.dimension))
                return false;
        return true;
    };

    for (const PointAssignment& assignment : program.assignments)
        if (!coordinateDestinationIsDouble(assignment.destination) ||
            !exactDeviceExpression(assignment.value, true) ||
            !exactDeviceExpression(assignment.condition, true) ||
            !expressionCoordinatesAreDouble(assignment.value) ||
            !expressionCoordinatesAreDouble(assignment.condition) ||
            assignment.value.instructions.size() +
                    assignment.condition.instructions.size() >
                MaximumDeviceInstructionsPerAssignment)
            return false;
    return true;
}

PredicateProgram compilePredicate(std::string_view specification,
                                  DimensionRegistry& dimensions)
{
    PredicateProgram program;
    program.expression =
        compileConditionalExpression(specification, dimensions);
    program.reads = program.expression.reads;
    return program;
}

bool predicateSupportsExactDevice(const PredicateProgram& program) noexcept
{
    return exactDeviceExpression(program.expression, false);
}

bool predicateMaySupportExactDevice(const PredicateProgram& program) noexcept
{
    return exactDeviceExpression(program.expression, true);
}

bool predicateSupportsExactDevice(const PointBatch& batch,
                                  const PredicateProgram& program) noexcept
{
    if (!exactDeviceExpression(program.expression, true))
        return false;
    for (const ExpressionInstruction& instruction :
         program.expression.instructions)
        if (instruction.op == ExpressionOp::LoadDimension)
        {
            const int axis = coordinateAxis(instruction.dimension);
            if (axis < 0)
                continue;
            if (!batch.has(instruction.dimension))
                return false;
            const DimensionType type =
                batch.columnInfo(instruction.dimension).physicalType;
            if (type != DimensionType::Double &&
                (type != DimensionType::Signed32 ||
                 batch.coordinateEncoding()
                         .offset()[static_cast<std::size_t>(axis)] != 0.0))
                return false;
        }
    return true;
}

void evaluatePredicate(PointBatch& batch, const PredicateProgram& program,
                       std::uint8_t* keep, std::size_t maximumHostWorkers)
{
    if (!keep && batch.size())
        throw std::invalid_argument("predicate output is null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        evaluatePredicateDevice(batch, program, keep);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported predicate batch memory kind");
    for (DimensionId id : program.reads)
        if (!batch.has(id))
            throw std::invalid_argument(
                "predicate source column is not materialized");
    if (batch.size() == 0)
        return;

    BindingMap columns;
    for (DimensionId id : program.reads)
        columns.emplace(id.value(), binding(batch, id));
    const BoundExpression expression =
        bindExpression(program.expression, columns);
    constexpr std::size_t MinimumPointsPerWorker = 65536;
    const std::size_t usefulWorkers =
        batch.size() / MinimumPointsPerWorker +
        static_cast<std::size_t>(batch.size() % MinimumPointsPerWorker != 0);
    const std::size_t availableWorkers =
        maximumHostWorkers
            ? maximumHostWorkers
            : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t workerCount =
        std::max<std::size_t>(1, std::min(usefulWorkers, availableWorkers));
    if (workerCount == 1)
    {
        evaluatePredicateRange(expression, batch.coordinateEncoding(), 0,
                               batch.size(), keep);
        return;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(workerCount);
    workers.reserve(workerCount);
    const std::size_t pointsPerWorker = batch.size() / workerCount;
    const std::size_t remainder = batch.size() % workerCount;
    for (std::size_t worker = 0; worker < workerCount; ++worker)
    {
        const std::size_t begin =
            worker * pointsPerWorker + std::min(worker, remainder);
        const std::size_t end = begin + pointsPerWorker +
                                static_cast<std::size_t>(worker < remainder);
        workers.emplace_back(
            [&, worker, begin, end]
            {
                try
                {
                    evaluatePredicateRange(expression,
                                           batch.coordinateEncoding(), begin,
                                           end, keep);
                }
                catch (...)
                {
                    errors[worker] = std::current_exception();
                }
            });
    }
    for (std::thread& worker : workers)
        worker.join();
    for (const std::exception_ptr& error : errors)
        if (error)
            std::rethrow_exception(error);
}

void executeAssign(PointBatch& batch, const AssignProgram& program,
                   std::size_t maximumHostWorkers)
{
    if (batch.memoryKind() == MemoryKind::Device)
    {
        executeAssignDevice(batch, program);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported assignment batch memory kind");

    const std::vector<BoundAssignment> assignments = bindHost(batch, program);
    if (batch.size() == 0 || assignments.empty())
        return;
    std::size_t maximumStackDepth = 1;
    for (const BoundAssignment& assignment : assignments)
        maximumStackDepth =
            std::max({maximumStackDepth, assignment.value.maximumStackDepth,
                      assignment.condition.maximumStackDepth});

    constexpr std::size_t MinimumPointsPerWorker = 65536;
    const std::size_t usefulWorkers =
        batch.size() / MinimumPointsPerWorker +
        static_cast<std::size_t>(batch.size() % MinimumPointsPerWorker != 0);
    const std::size_t availableWorkers =
        maximumHostWorkers
            ? maximumHostWorkers
            : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t workerCount =
        std::max<std::size_t>(1, std::min(usefulWorkers, availableWorkers));
    if (workerCount == 1)
    {
        executeRange(assignments, batch.coordinateEncoding(), 0, batch.size(),
                     maximumStackDepth);
        return;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(workerCount);
    workers.reserve(workerCount);
    const std::size_t pointsPerWorker = batch.size() / workerCount;
    const std::size_t remainder = batch.size() % workerCount;
    for (std::size_t worker = 0; worker < workerCount; ++worker)
    {
        const std::size_t begin =
            worker * pointsPerWorker + std::min(worker, remainder);
        const std::size_t end = begin + pointsPerWorker +
                                static_cast<std::size_t>(worker < remainder);
        workers.emplace_back(
            [&, worker, begin, end]
            {
                try
                {
                    executeRange(assignments, batch.coordinateEncoding(), begin,
                                 end, maximumStackDepth);
                }
                catch (...)
                {
                    errors[worker] = std::current_exception();
                }
            });
    }
    for (std::thread& worker : workers)
        worker.join();
    for (const std::exception_ptr& error : errors)
        if (error)
            std::rethrow_exception(error);
}

} // namespace pdg
