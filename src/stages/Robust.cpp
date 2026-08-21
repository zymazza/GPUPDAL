#include <pdg/PointBatch.hpp>
#include <pdg/stages/Robust.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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
    throw std::invalid_argument("robust source has no physical type");
}

double logicalValue(const PointBatch& batch, DimensionId id, std::size_t index)
{
    const ColumnInfo& column = batch.columnInfo(id);
    const void* data = batch.rawData(id);
    const int axis = coordinateAxis(id);
    if (axis >= 0 && column.physicalType == DimensionType::Signed32)
        return batch.coordinateEncoding().decode(
            static_cast<std::size_t>(axis),
            static_cast<const std::int32_t*>(data)[index]);
    return loadPhysical(data, column.physicalType, index);
}

double percentile(std::vector<double> values, double fraction)
{
    const int index =
        static_cast<int>(static_cast<double>(values.size()) * fraction);
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[static_cast<std::size_t>(index)];
}

double median(std::vector<double> values)
{
    const std::size_t index = values.size() / 2U;
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

RobustResult evaluateRobustHost(PointBatch& batch, const RobustProgram& program,
                                std::uint8_t* keep)
{
    std::vector<double> values(batch.size());
    for (std::size_t point = 0; point < batch.size(); ++point)
        values[point] = logicalValue(batch, program.dimension, point);

    RobustResult result;
    if (program.kind == RobustKind::Iqr)
    {
        result.first = percentile(values, 0.25);
        result.second = percentile(values, 0.75);
        result.scale = result.second - result.first;
        result.lowFence = result.first - program.multiplier * result.scale;
        result.highFence = result.second + program.multiplier * result.scale;
        for (std::size_t point = 0; point < values.size(); ++point)
            keep[point] =
                static_cast<std::uint8_t>(values[point] > result.lowFence &&
                                          values[point] < result.highFence);
    }
    else
    {
        result.first = median(values);
        std::transform(values.begin(), values.end(), values.begin(),
                       [center = result.first](double value)
                       { return std::fabs(value - center); });
        result.scale = median(values) * program.madMultiplier;
        for (std::size_t point = 0; point < values.size(); ++point)
            keep[point] = static_cast<std::uint8_t>(
                values[point] / result.scale < program.multiplier);
        result.lowFence = result.first - program.multiplier * result.scale;
        result.highFence = result.first + program.multiplier * result.scale;
        result.second = result.scale;
    }
    return result;
}
} // unnamed namespace

RobustResult evaluateRobustDevice(PointBatch& batch,
                                  const RobustProgram& program,
                                  std::uint8_t* keep);

bool robustSupportsExactDevice(const PointBatch& hostBatch,
                               const RobustProgram& program) noexcept
{
    if (hostBatch.memoryKind() != MemoryKind::Host &&
        hostBatch.memoryKind() != MemoryKind::PinnedHost)
        return false;
    if (!hostBatch.has(program.dimension) || !hostBatch.size() ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;
    try
    {
        for (std::size_t point = 0; point < hostBatch.size(); ++point)
        {
            const double value =
                logicalValue(hostBatch, program.dimension, point);
            if (!std::isfinite(value) || (value == 0.0 && std::signbit(value)))
                return false;
        }
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

RobustResult evaluateRobust(PointBatch& batch, const RobustProgram& program,
                            std::uint8_t* keep)
{
    if (!batch.has(program.dimension))
        throw std::invalid_argument("robust source column is not materialized");
    if (!batch.size())
        throw std::invalid_argument(
            "robust statistics require a nonempty batch");
    if (!keep)
        throw std::invalid_argument("robust selection mask is null");
    if (batch.memoryKind() == MemoryKind::Device)
        return evaluateRobustDevice(batch, program, keep);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported robust memory kind");
    return evaluateRobustHost(batch, program, keep);
}

} // namespace pdg
