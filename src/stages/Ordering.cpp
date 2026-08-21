#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
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

template <typename T>
void sortPass(const T* values, std::vector<std::uint64_t>& permutation,
              OrderingDirection direction, bool stable)
{
    const auto less =
        [values, direction](std::uint64_t left, std::uint64_t right)
    {
        if (direction == OrderingDirection::Ascending)
            return values[left] < values[right];
        return values[right] < values[left];
    };
    if (stable)
        std::stable_sort(permutation.begin(), permutation.end(), less);
    else
        std::sort(permutation.begin(), permutation.end(), less);
}

void sortCoordinatePass(const PointBatch& batch, DimensionId dimension,
                        const std::int32_t* values,
                        std::vector<std::uint64_t>& permutation,
                        OrderingDirection direction, bool stable)
{
    const std::size_t axis =
        static_cast<std::size_t>(coordinateAxis(dimension));
    const auto less = [&batch, values, axis, direction](std::uint64_t left,
                                                        std::uint64_t right)
    {
        const double leftValue =
            batch.coordinateEncoding().decode(axis, values[left]);
        const double rightValue =
            batch.coordinateEncoding().decode(axis, values[right]);
        if (direction == OrderingDirection::Ascending)
            return leftValue < rightValue;
        return rightValue < leftValue;
    };
    if (stable)
        std::stable_sort(permutation.begin(), permutation.end(), less);
    else
        std::sort(permutation.begin(), permutation.end(), less);
}

void sortDimension(const PointBatch& batch, DimensionId dimension,
                   std::vector<std::uint64_t>& permutation,
                   OrderingDirection direction, bool stable)
{
    const ColumnInfo& column = batch.columnInfo(dimension);
    const void* data = batch.rawData(dimension);
    if (coordinateAxis(dimension) >= 0 &&
        column.physicalType == DimensionType::Signed32)
    {
        sortCoordinatePass(batch, dimension,
                           static_cast<const std::int32_t*>(data), permutation,
                           direction, stable);
        return;
    }
    switch (column.physicalType)
    {
    case DimensionType::Signed8:
        sortPass(static_cast<const std::int8_t*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Signed16:
        sortPass(static_cast<const std::int16_t*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Signed32:
        sortPass(static_cast<const std::int32_t*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Signed64:
        sortPass(static_cast<const std::int64_t*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Unsigned8:
        sortPass(static_cast<const std::uint8_t*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Unsigned16:
        sortPass(static_cast<const std::uint16_t*>(data), permutation,
                 direction, stable);
        return;
    case DimensionType::Unsigned32:
        sortPass(static_cast<const std::uint32_t*>(data), permutation,
                 direction, stable);
        return;
    case DimensionType::Unsigned64:
        sortPass(static_cast<const std::uint64_t*>(data), permutation,
                 direction, stable);
        return;
    case DimensionType::Float:
        sortPass(static_cast<const float*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::Double:
        sortPass(static_cast<const double*>(data), permutation, direction,
                 stable);
        return;
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument("ordering key has no physical type");
}

template <typename T> bool containsNan(const T*, std::size_t) noexcept
{
    return false;
}

template <>
bool containsNan<float>(const float* values, std::size_t size) noexcept
{
    for (std::size_t point = 0; point < size; ++point)
        if (std::isnan(values[point]))
            return true;
    return false;
}

template <>
bool containsNan<double>(const double* values, std::size_t size) noexcept
{
    for (std::size_t point = 0; point < size; ++point)
        if (std::isnan(values[point]))
            return true;
    return false;
}

bool dimensionHasNan(const PointBatch& batch, DimensionId dimension) noexcept
{
    try
    {
        const ColumnInfo& column = batch.columnInfo(dimension);
        const void* data = batch.rawData(dimension);
        switch (column.physicalType)
        {
        case DimensionType::Float:
            return containsNan(static_cast<const float*>(data), batch.size());
        case DimensionType::Double:
            return containsNan(static_cast<const double*>(data), batch.size());
        default:
            return false;
        }
    }
    catch (const std::exception&)
    {
        return true;
    }
}

bool deviceDimensionSupported(const PointBatch& batch,
                              DimensionId dimension) noexcept
{
    try
    {
        const ColumnInfo& column = batch.columnInfo(dimension);
        if (column.physicalType == DimensionType::None)
            return false;
        const int axis = coordinateAxis(dimension);
        if (axis >= 0 && column.physicalType == DimensionType::Signed32)
        {
            const std::size_t coordinate = static_cast<std::size_t>(axis);
            return std::isfinite(
                       batch.coordinateEncoding().scale()[coordinate]) &&
                   std::isfinite(
                       batch.coordinateEncoding().offset()[coordinate]);
        }
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}
} // unnamed namespace

OrderingResult orderPointsDevice(PointBatch& batch,
                                 const OrderingProgram& program,
                                 std::uint64_t* permutation);

bool orderingMaySupportExactDevice(const PointBatch& hostBatch,
                                   const OrderingProgram& program) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        program.dimensions.empty() ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;
    for (DimensionId dimension : program.dimensions)
        if (!hostBatch.has(dimension) ||
            !deviceDimensionSupported(hostBatch, dimension) ||
            dimensionHasNan(hostBatch, dimension))
            return false;
    return true;
}

OrderingResult orderPoints(PointBatch& batch, const OrderingProgram& program,
                           std::uint64_t* permutation)
{
    if (program.dimensions.empty())
        throw std::invalid_argument("ordering requires at least one dimension");
    for (DimensionId dimension : program.dimensions)
        if (!batch.has(dimension))
            throw std::invalid_argument(
                "ordering key column is not materialized");
    if (batch.size() && !permutation)
        throw std::invalid_argument("ordering permutation is null");
    if (!batch.size())
        return {};
    if (batch.memoryKind() == MemoryKind::Device)
        return orderPointsDevice(batch, program, permutation);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported ordering memory kind");

    std::vector<std::uint64_t> result(batch.size());
    std::iota(result.begin(), result.end(), std::uint64_t{0});
    for (std::size_t pass = 0; pass < program.dimensions.size(); ++pass)
    {
        const bool stable =
            program.dimensions.size() > 1U
                ? pass != 0U
                : program.algorithm == OrderingAlgorithm::Stable;
        sortDimension(batch, program.dimensions[pass], result,
                      program.direction, stable);
    }
    std::copy(result.begin(), result.end(), permutation);
    return {};
}

} // namespace pdg
