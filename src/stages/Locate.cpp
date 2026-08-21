#include <pdg/PointBatch.hpp>
#include <pdg/stages/Locate.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

namespace
{
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
    throw std::invalid_argument("locate source has no physical type");
}

double initialValue(LocateKind kind)
{
    return kind == LocateKind::Minimum ? (std::numeric_limits<double>::max)()
                                       : std::numeric_limits<double>::lowest();
}

bool improves(LocateKind kind, double candidate, double current) noexcept
{
    return kind == LocateKind::Minimum ? candidate < current
                                       : candidate > current;
}

LocateResult locateExtremeHost(PointBatch& batch, const LocateProgram& program,
                               std::uint64_t indexOffset)
{
    LocateResult result;
    result.value = initialValue(program.kind);
    result.index = indexOffset;
    result.hasPoints = static_cast<std::uint32_t>(batch.size() != 0);
    if (!batch.size() || program.kind == LocateKind::None)
        return result;

    const ColumnInfo& column = batch.columnInfo(program.dimension);
    const void* data = batch.rawData(program.dimension);
    const int axis = coordinateAxis(program.dimension);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        double value = loadPhysical(data, column.physicalType, point);
        if (axis >= 0 && column.physicalType == DimensionType::Signed32)
            value = batch.coordinateEncoding().decode(
                static_cast<std::size_t>(axis),
                static_cast<const std::int32_t*>(data)[point]);
        if (improves(program.kind, value, result.value))
        {
            result.value = value;
            result.index = indexOffset + static_cast<std::uint64_t>(point);
            result.comparable = 1;
        }
    }
    return result;
}
} // unnamed namespace

LocateResult locateExtremeDevice(PointBatch& batch,
                                 const LocateProgram& program,
                                 std::uint64_t indexOffset);

LocateResult mergeLocateResults(const LocateProgram& program,
                                const LocateResult& first,
                                const LocateResult& second)
{
    if (!first.hasPoints)
        return second;
    if (!second.hasPoints)
        return first;
    if (first.comparable != second.comparable)
        return first.comparable ? first : second;
    if (!first.comparable)
        return first.index <= second.index ? first : second;
    if (improves(program.kind, second.value, first.value))
        return second;
    if (improves(program.kind, first.value, second.value))
        return first;
    return first.index <= second.index ? first : second;
}

LocateResult locateExtreme(PointBatch& batch, const LocateProgram& program,
                           std::uint64_t indexOffset)
{
    if (!batch.has(program.dimension))
        throw std::invalid_argument("locate source column is not materialized");
    if (batch.size() >
        (std::numeric_limits<std::uint64_t>::max)() - indexOffset)
        throw std::overflow_error("locate point index overflow");
    if (program.kind == LocateKind::None)
        return {};
    if (batch.memoryKind() == MemoryKind::Device)
        return locateExtremeDevice(batch, program, indexOffset);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported locate memory kind");
    return locateExtremeHost(batch, program, indexOffset);
}

} // namespace pdg
