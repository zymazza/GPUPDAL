#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

BoundsResult summarizeBoundsDevice(PointBatch& batch,
                                   std::uint64_t indexOffset);

namespace
{
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
    throw std::invalid_argument("bounds coordinate has no physical type");
}

void consider(LocateResult& result, LocateKind kind, double value,
              std::uint64_t index)
{
    if (!result.hasPoints)
    {
        result.hasPoints = 1;
        result.index = index;
    }
    const bool improves = kind == LocateKind::Minimum ? value < result.value
                                                      : value > result.value;
    if (improves)
    {
        result.value = value;
        result.index = index;
        result.comparable = 1;
    }
}

BoundsResult summarizeBoundsHost(PointBatch& batch, std::uint64_t indexOffset)
{
    BoundsResult result;
    result.count = static_cast<std::uint64_t>(batch.size());
    const std::array<DimensionId, 3> dimensions = {
        DimensionId(StandardDimension::X), DimensionId(StandardDimension::Y),
        DimensionId(StandardDimension::Z)};
    std::array<const void*, 3> columns{};
    std::array<DimensionType, 3> types{};
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
    {
        if (!batch.has(dimensions[axis]))
            throw std::invalid_argument(
                "bounds coordinate column is not materialized");
        columns[axis] = batch.rawData(dimensions[axis]);
        types[axis] = batch.columnInfo(dimensions[axis]).physicalType;
    }

    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::uint64_t index =
            indexOffset + static_cast<std::uint64_t>(point);
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
        {
            double value = loadPhysical(columns[axis], types[axis], point);
            if (types[axis] == DimensionType::Signed32)
                value = batch.coordinateEncoding().decode(
                    axis,
                    static_cast<const std::int32_t*>(columns[axis])[point]);
            consider(result.minimum[axis], LocateKind::Minimum, value, index);
            consider(result.maximum[axis], LocateKind::Maximum, value, index);
        }
    }
    return result;
}
} // unnamed namespace

BoundsResult summarizeBounds(PointBatch& batch, std::uint64_t indexOffset)
{
    if (batch.size() >
        (std::numeric_limits<std::uint64_t>::max)() - indexOffset)
        throw std::overflow_error("bounds point index overflow");
    if (batch.memoryKind() == MemoryKind::Device)
        return summarizeBoundsDevice(batch, indexOffset);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported bounds memory kind");
    return summarizeBoundsHost(batch, indexOffset);
}

BoundsResult mergeBoundsResults(const BoundsResult& first,
                                const BoundsResult& second)
{
    if (second.count >
        (std::numeric_limits<std::uint64_t>::max)() - first.count)
        throw std::overflow_error("bounds point count overflow");
    BoundsResult result;
    result.count = first.count + second.count;
    const std::array<DimensionId, 3> dimensions = {
        DimensionId(StandardDimension::X), DimensionId(StandardDimension::Y),
        DimensionId(StandardDimension::Z)};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.minimum[axis] =
            mergeLocateResults({dimensions[axis], LocateKind::Minimum},
                               first.minimum[axis], second.minimum[axis]);
        result.maximum[axis] =
            mergeLocateResults({dimensions[axis], LocateKind::Maximum},
                               first.maximum[axis], second.maximum[axis]);
    }
    return result;
}

} // namespace pdg
