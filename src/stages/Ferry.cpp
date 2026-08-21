#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ferry.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pdg
{

void executeFerryDevice(PointBatch& batch, const FerryProgram& program);

namespace
{
DimensionType resolveDimensionType(DimensionType first,
                                   DimensionType second) noexcept
{
    if (first == DimensionType::None)
        return second;
    if (second == DimensionType::None || first == second)
        return first;

    const auto category = [](DimensionType type)
    { return static_cast<std::uint16_t>(type) & 0xf00U; };
    if (category(first) == category(second))
        return dimensionTypeSize(first) >= dimensionTypeSize(second) ? first
                                                                     : second;
    if (isFloating(first))
        return first;
    if (isFloating(second))
        return second;

    constexpr std::uint16_t UnsignedCategory = 0x200U;
    if (category(first) == UnsignedCategory &&
        dimensionTypeSize(first) < dimensionTypeSize(second))
        return second;
    if (category(second) == UnsignedCategory &&
        dimensionTypeSize(second) < dimensionTypeSize(first))
        return first;

    switch ((std::max)(dimensionTypeSize(first), dimensionTypeSize(second)))
    {
    case 1:
        return DimensionType::Signed16;
    case 2:
        return DimensionType::Signed32;
    case 4:
        return DimensionType::Signed64;
    default:
        return DimensionType::Double;
    }
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
    throw std::invalid_argument("ferry source has no physical type");
}

double symmetricRound(double value)
{
    return value > 0.0 ? std::floor(value + 0.5) : std::ceil(value - 0.5);
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
    throw std::invalid_argument("ferry destination has no physical type");
}

struct Binding
{
    bool hasSource = false;
    const void* source = nullptr;
    void* destination = nullptr;
    DimensionType sourceType = DimensionType::None;
    DimensionType destinationType = DimensionType::None;
    int sourceAxis = -1;
    int destinationAxis = -1;
};

std::vector<Binding> bindHost(PointBatch& batch, const FerryProgram& program)
{
    std::vector<Binding> bindings;
    bindings.reserve(program.copies.size());
    for (const FerryCopy& copy : program.copies)
    {
        const bool destinationPresent = batch.has(copy.destination);
        if (!destinationPresent)
            batch.materialize(copy.destination);
        const ColumnInfo& destination = batch.columnInfo(copy.destination);
        if (copy.destinationCreated || !destinationPresent)
            std::memset(batch.rawData(copy.destination), 0,
                        batch.size() *
                            dimensionTypeSize(destination.physicalType));

        Binding binding;
        binding.hasSource = copy.hasSource;
        binding.destination = batch.rawData(copy.destination);
        binding.destinationType = destination.physicalType;
        binding.destinationAxis = coordinateAxis(copy.destination);
        if (copy.hasSource)
        {
            if (!batch.has(copy.source))
                throw std::invalid_argument(
                    "ferry source column is not materialized");
            binding.source = batch.rawData(copy.source);
            binding.sourceType = batch.columnInfo(copy.source).physicalType;
            binding.sourceAxis = coordinateAxis(copy.source);
        }
        bindings.push_back(binding);
    }
    return bindings;
}

void executeHostRange(PointBatch& batch, const std::vector<Binding>& bindings,
                      std::size_t begin, std::size_t end)
{
    const CoordinateEncoding& coordinates = batch.coordinateEncoding();
    for (std::size_t index = begin; index < end; ++index)
    {
        for (const Binding& binding : bindings)
        {
            if (!binding.hasSource)
                continue;
            double value =
                loadPhysical(binding.source, binding.sourceType, index);
            if (binding.sourceAxis >= 0)
                value = coordinates.decode(
                    static_cast<std::size_t>(binding.sourceAxis),
                    static_cast<const std::int32_t*>(binding.source)[index]);
            if (binding.destinationAxis >= 0)
                static_cast<std::int32_t*>(binding.destination)[index] =
                    coordinates.encode(
                        static_cast<std::size_t>(binding.destinationAxis),
                        value);
            else
                storePhysical(binding.destination, binding.destinationType,
                              index, value);
        }
    }
}
} // unnamed namespace

bool ferrySupportsExactPointProgram(
    const FerryProgram& program, const DimensionRegistry& dimensions) noexcept
{
    for (const FerryCopy& copy : program.copies)
    {
        if (!copy.hasSource)
            return false;
        const DimensionDefinition* source = dimensions.find(copy.source);
        const DimensionDefinition* destination =
            dimensions.find(copy.destination);
        if (!source || !destination)
            return false;
        if (!copy.destinationCreated &&
            resolveDimensionType(source->type, destination->type) !=
                destination->type)
            return false;
    }
    return true;
}

void executeFerry(PointBatch& batch, const FerryProgram& program,
                  std::size_t maximumHostWorkers)
{
    if (batch.memoryKind() == MemoryKind::Device)
    {
        executeFerryDevice(batch, program);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported ferry batch memory kind");

    const std::vector<Binding> bindings = bindHost(batch, program);
    if (batch.size() == 0 || bindings.empty())
        return;

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
        executeHostRange(batch, bindings, 0, batch.size());
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
                    executeHostRange(batch, bindings, begin, end);
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
