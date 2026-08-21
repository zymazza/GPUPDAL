#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/index/SpatialIndex.hpp>
#include <pdg/index/SpatialTile.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pdg
{
namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr std::size_t ConservativeDeviceBytesPerPoint = 128U;

bool hostVisible(MemoryKind kind) noexcept
{
    return kind == MemoryKind::Host || kind == MemoryKind::PinnedHost;
}

std::size_t executionLaneCount(const SpatialTileSet& tiles,
                               const MemoryResource& stagingMemory,
                               const MemoryResource& executionMemory)
{
    if (tiles.tiles().empty())
        return 0U;
    if (executionMemory.kind() == MemoryKind::Device &&
        stagingMemory.kind() == MemoryKind::PinnedHost &&
        tiles.tiles().size() > 1U)
    {
        if (tiles.peakPointCount() > (std::numeric_limits<std::size_t>::max)() /
                                         ConservativeDeviceBytesPerPoint)
            throw std::overflow_error(
                "spatial tile scheduler byte estimate overflows");
        return makeTiledSchedule(
                   {.pipelineClass = PipelineClass::RadiusNeighborhood,
                    .itemCount = tiles.tiles().size(),
                    .tileItems = 1U,
                    .bytesPerLane = tiles.peakPointCount() *
                                    ConservativeDeviceBytesPerPoint,
                    .memoryBudgetBytes = tiles.config().deviceMemoryBudgetBytes,
                    .requestedLanes = tiles.config().schedulerLanes})
            .activeLaneCount;
    }
    return 1U;
}

std::size_t laneReuseCount(const SpatialTileSet& tiles, std::size_t laneCount,
                           const MemoryResource& executionMemory) noexcept
{
    if (executionMemory.kind() != MemoryKind::Device ||
        tiles.tiles().size() <= laneCount)
        return 0U;
    return tiles.tiles().size() - laneCount;
}

void validateConfig(const SpatialTileConfig& config)
{
    if (config.dimensions != 2U && config.dimensions != 3U)
        throw std::invalid_argument(
            "spatial tile dimensions must be two or three");
    if (!std::isfinite(config.edgeLength) || config.edgeLength <= 0.0)
        throw std::invalid_argument(
            "spatial tile edge length must be finite and positive");
    if (!std::isfinite(config.halo) || config.halo < 0.0)
        throw std::invalid_argument(
            "spatial tile halo must be finite and nonnegative");
    if (config.maximumPoints == 0U)
        throw std::invalid_argument(
            "spatial tile point capacity must be positive");
    if (config.schedulerLanes &&
        (config.schedulerLanes < MinimumSweptLaneCount ||
         config.schedulerLanes > MaximumSweptLaneCount))
        throw std::invalid_argument(
            "spatial tile scheduler lanes must be zero or in [2, 6]");
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
        if (!std::isfinite(config.origin[axis]))
            throw std::invalid_argument("spatial tile origin must be finite");
}

void validateSource(const PointBatch& source, const SpatialTileConfig& config)
{
    if (!hostVisible(source.memoryKind()))
        throw std::invalid_argument(
            "spatial tiling requires host-visible coordinates");
    for (DimensionId dimension : {X, Y})
        if (!source.has(dimension) ||
            source.columnInfo(dimension).physicalType != DimensionType::Double)
            throw std::invalid_argument(
                "spatial tiling requires logical-double coordinate columns");
    if (config.dimensions == 3U &&
        (!source.has(Z) ||
         source.columnInfo(Z).physicalType != DimensionType::Double))
        throw std::invalid_argument(
            "three-dimensional tiling requires logical-double Z");
}

std::int64_t floorCell(long double quotient)
{
    constexpr long double Minimum =
        static_cast<long double>((std::numeric_limits<std::int64_t>::min)());
    constexpr long double Maximum =
        static_cast<long double>((std::numeric_limits<std::int64_t>::max)());
    if (!std::isfinite(quotient) || quotient < Minimum || quotient > Maximum)
        throw std::out_of_range(
            "coordinate is outside the signed spatial tile frame");
    const long double cell = std::floor(quotient);
    if (cell < Minimum || cell > Maximum)
        throw std::out_of_range(
            "coordinate is outside the signed spatial tile frame");
    return static_cast<std::int64_t>(cell);
}

std::int64_t pointCell(double value, double origin, double edge)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(
            "spatial tiling requires finite coordinates");
    return floorCell(
        (static_cast<long double>(value) - static_cast<long double>(origin)) /
        static_cast<long double>(edge));
}

SpatialTileCoordinate
ownerCoordinate(const std::array<const double*, 3>& coordinates,
                std::size_t point, const SpatialTileConfig& config)
{
    SpatialTileCoordinate result;
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
        result.cell[axis] = pointCell(coordinates[axis][point],
                                      config.origin[axis], config.edgeLength);
    return result;
}

bool insideExpanded(const SpatialTileCoordinate& coordinate,
                    const std::array<const double*, 3>& coordinates,
                    std::size_t point, const SpatialTileConfig& config)
{
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        const long double origin =
            static_cast<long double>(config.origin[axis]);
        const long double edge = static_cast<long double>(config.edgeLength);
        const long double halo = static_cast<long double>(config.halo);
        const long double cell =
            static_cast<long double>(coordinate.cell[axis]);
        const long double lower = origin + cell * edge - halo;
        const long double upper = origin + (cell + 1.0L) * edge + halo;
        const long double value =
            static_cast<long double>(coordinates[axis][point]);
        if (value < lower || value > upper)
            return false;
    }
    return true;
}

std::array<std::int64_t, 3>
candidateEdge(const std::array<const double*, 3>& coordinates,
              std::size_t point, const SpatialTileConfig& config,
              double haloSign)
{
    std::array<std::int64_t, 3> result{0, 0, 0};
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        const long double value =
            static_cast<long double>(coordinates[axis][point]);
        const long double origin =
            static_cast<long double>(config.origin[axis]);
        const long double halo = static_cast<long double>(config.halo);
        const long double edge = static_cast<long double>(config.edgeLength);
        result[axis] = floorCell(
            (value - origin + static_cast<long double>(haloSign) * halo) /
            edge);
    }
    return result;
}

template <typename Callback>
void forEachCoordinate(const std::array<std::int64_t, 3>& minimum,
                       const std::array<std::int64_t, 3>& maximum,
                       std::uint8_t dimensions, Callback&& callback)
{
    for (std::int64_t z = minimum[2];; ++z)
    {
        for (std::int64_t y = minimum[1];; ++y)
        {
            for (std::int64_t x = minimum[0];; ++x)
            {
                callback(SpatialTileCoordinate{{x, y, z}});
                if (x == maximum[0])
                    break;
            }
            if (y == maximum[1])
                break;
        }
        if (z == maximum[2] || dimensions == 2U)
            break;
    }
}
} // unnamed namespace

const SpatialTileConfig& SpatialTileSet::config() const noexcept
{
    return m_config;
}

const std::vector<SpatialTile>& SpatialTileSet::tiles() const noexcept
{
    return m_tiles;
}

std::size_t SpatialTileSet::sourcePointCount() const noexcept
{
    return m_sourcePointCount;
}

std::size_t SpatialTileSet::ghostPointCount() const noexcept
{
    return m_ghostPointCount;
}

std::size_t SpatialTileSet::peakPointCount() const noexcept
{
    return m_peakPointCount;
}

SpatialTileSet makeSpatialTiles(const PointBatch& source,
                                SpatialTileConfig config)
{
    validateConfig(config);
    validateSource(source, config);

    SpatialTileSet result;
    result.m_config = config;
    result.m_sourcePointCount = source.size();
    if (source.size() == 0U)
        return result;

    const std::array<const double*, 3> coordinates{
        source.data<double>(X), source.data<double>(Y),
        config.dimensions == 3U ? source.data<double>(Z) : nullptr};

    std::vector<SpatialTileCoordinate> owners(source.size());
    std::map<SpatialTileCoordinate, SpatialTile> tiles;
    std::array<std::int64_t, 3> minimumCell{
        (std::numeric_limits<std::int64_t>::max)(),
        (std::numeric_limits<std::int64_t>::max)(),
        (std::numeric_limits<std::int64_t>::max)()};
    std::array<std::int64_t, 3> maximumCell{
        (std::numeric_limits<std::int64_t>::min)(),
        (std::numeric_limits<std::int64_t>::min)(),
        (std::numeric_limits<std::int64_t>::min)()};
    if (config.dimensions == 2U)
        minimumCell[2] = maximumCell[2] = 0;
    for (std::size_t point = 0; point < source.size(); ++point)
    {
        owners[point] = ownerCoordinate(coordinates, point, config);
        SpatialTile& tile = tiles[owners[point]];
        tile.coordinate = owners[point];
        for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
        {
            minimumCell[axis] =
                (std::min)(minimumCell[axis], owners[point].cell[axis]);
            maximumCell[axis] =
                (std::max)(maximumCell[axis], owners[point].cell[axis]);
        }
    }

    for (std::size_t point = 0; point < source.size(); ++point)
    {
        std::array<std::int64_t, 3> minimum =
            candidateEdge(coordinates, point, config, -1.0);
        std::array<std::int64_t, 3> maximum =
            candidateEdge(coordinates, point, config, 1.0);
        for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
        {
            // One adjacent cell on each side covers closed halo faces even
            // when a binary64 quotient lands exactly on a tile boundary.
            if (minimum[axis] > minimumCell[axis])
                --minimum[axis];
            if (maximum[axis] < maximumCell[axis])
                ++maximum[axis];
            minimum[axis] = (std::max)(minimum[axis], minimumCell[axis]);
            maximum[axis] = (std::min)(maximum[axis], maximumCell[axis]);
        }
        forEachCoordinate(
            minimum, maximum, config.dimensions,
            [&](const SpatialTileCoordinate& coordinate)
            {
                auto position = tiles.find(coordinate);
                if (position == tiles.end() ||
                    !insideExpanded(coordinate, coordinates, point, config))
                    return;
                SpatialTile& tile = position->second;
                tile.sourcePointIds.push_back(point);
                const bool ghost = coordinate != owners[point];
                tile.ghost.push_back(static_cast<std::uint8_t>(ghost));
                if (ghost)
                    ++result.m_ghostPointCount;
                else
                    ++tile.ownedCount;
            });
    }

    result.m_tiles.reserve(tiles.size());
    for (auto& [coordinate, tile] : tiles)
    {
        (void)coordinate;
        if (tile.sourcePointIds.size() > config.maximumPoints)
            throw std::length_error(
                "spatial tile including ghosts exceeds point capacity");
        result.m_peakPointCount =
            (std::max)(result.m_peakPointCount, tile.sourcePointIds.size());
        result.m_tiles.push_back(std::move(tile));
    }
    return result;
}

PointBatch gatherSpatialTile(const PointBatch& source, const SpatialTile& tile,
                             std::span<const DimensionId> dimensions,
                             DimensionRegistry& registry,
                             MemoryResource& destinationMemory)
{
    PointBatch result(tile.sourcePointIds.size(), source.coordinateEncoding(),
                      registry, destinationMemory);
    gatherSpatialTileInto(source, tile, dimensions, result);
    return result;
}

void gatherSpatialTileInto(const PointBatch& source, const SpatialTile& tile,
                           std::span<const DimensionId> dimensions,
                           PointBatch& destination)
{
    if (&source == &destination)
        throw std::invalid_argument(
            "spatial tile source and destination must be distinct");
    if (!hostVisible(source.memoryKind()) ||
        !hostVisible(destination.memoryKind()))
        throw std::invalid_argument(
            "spatial tile gathering requires host-visible memory");
    if (tile.sourcePointIds.size() != tile.ghost.size())
        throw std::invalid_argument("spatial tile mapping size mismatch");
    if (tile.sourcePointIds.size() > destination.capacity())
        throw std::length_error(
            "spatial tile exceeds reusable destination capacity");
    const CoordinateEncoding& sourceEncoding = source.coordinateEncoding();
    const CoordinateEncoding& destinationEncoding =
        destination.coordinateEncoding();
    if (std::memcmp(sourceEncoding.scale().data(),
                    destinationEncoding.scale().data(),
                    sizeof(std::array<double, 3>)) != 0 ||
        std::memcmp(sourceEncoding.offset().data(),
                    destinationEncoding.offset().data(),
                    sizeof(std::array<double, 3>)) != 0)
        throw std::invalid_argument(
            "spatial tile source and destination encodings differ");
    for (const std::size_t sourcePoint : tile.sourcePointIds)
        if (sourcePoint >= source.size())
            throw std::out_of_range(
                "spatial tile source id exceeds source batch");

    for (DimensionId dimension : dimensions)
    {
        if (!source.has(dimension))
            throw std::invalid_argument(
                "spatial tile source column is not materialized");
        const DimensionType physical =
            source.columnInfo(dimension).physicalType;
        destination.materialize(dimension, physical);
        const std::size_t stride = dimensionTypeSize(physical);
        const std::byte* input =
            static_cast<const std::byte*>(source.rawData(dimension));
        std::byte* output =
            static_cast<std::byte*>(destination.rawData(dimension));
        for (std::size_t local = 0; local < tile.sourcePointIds.size(); ++local)
        {
            const std::size_t sourcePoint = tile.sourcePointIds[local];
            std::memcpy(output + local * stride, input + sourcePoint * stride,
                        stride);
        }
    }
    destination.materializeGhostMask();
    if (!tile.ghost.empty())
        std::memcpy(destination.ghostData(), tile.ghost.data(),
                    tile.ghost.size());
    destination.setSize(tile.sourcePointIds.size());
}

PointBatch gatherSpatialTile(const PointBatch& source, const SpatialTile& tile,
                             std::initializer_list<DimensionId> dimensions,
                             DimensionRegistry& registry,
                             MemoryResource& destinationMemory)
{
    return gatherSpatialTile(
        source, tile,
        std::span<const DimensionId>(dimensions.begin(), dimensions.size()),
        registry, destinationMemory);
}

void gatherSpatialTileInto(const PointBatch& source, const SpatialTile& tile,
                           std::initializer_list<DimensionId> dimensions,
                           PointBatch& destination)
{
    gatherSpatialTileInto(
        source, tile,
        std::span<const DimensionId>(dimensions.begin(), dimensions.size()),
        destination);
}

SpatialTileExecutionStats
tiledRadiusCounts(const PointBatch& source, const SpatialTileSet& tiles,
                  std::uint8_t queryDimensions, double radius,
                  DimensionRegistry& registry, MemoryResource& stagingMemory,
                  MemoryResource& executionMemory,
                  std::span<std::uint32_t> counts)
{
    if (queryDimensions != 2U && queryDimensions != 3U)
        throw std::invalid_argument(
            "tiled radius query dimensions must be two or three");
    if (queryDimensions < tiles.config().dimensions)
        throw std::invalid_argument(
            "tiled radius query cannot ignore a partition axis");
    if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument(
            "tiled radius query must be finite and positive");
    if (tiles.config().halo < radius)
        throw std::invalid_argument(
            "spatial tile halo is smaller than the query radius");
    if (tiles.sourcePointCount() != source.size() ||
        counts.size() != source.size())
        throw std::invalid_argument("tiled radius source/result size mismatch");

    const std::uint32_t missing = (std::numeric_limits<std::uint32_t>::max)();
    std::fill(counts.begin(), counts.end(), missing);
    if (executionMemory.kind() == MemoryKind::Device)
        detail::tiledRadiusCountsDevice(source, tiles, queryDimensions, radius,
                                        registry, stagingMemory,
                                        executionMemory, counts);
    else
    {
        if (!hostVisible(executionMemory.kind()))
            throw std::invalid_argument(
                "unsupported tiled radius execution memory");
        const std::array<DimensionId, 3> coordinateIds{X, Y, Z};
        const std::span<const DimensionId> selected(coordinateIds.data(),
                                                    queryDimensions);
        for (const SpatialTile& tile : tiles.tiles())
        {
            PointBatch tileBatch = gatherSpatialTile(source, tile, selected,
                                                     registry, executionMemory);
            const UniformGridConfig config =
                makeUniformGridConfig(tileBatch, queryDimensions, radius);
            SpatialIndex index(tileBatch, config);
            index.build();
            std::vector<std::uint32_t> local(tileBatch.size());
            radiusCounts(index, radius, local.data());
            scatterSpatialTileOwned(tile, std::span<const std::uint32_t>(local),
                                    counts);
        }
    }
    if (std::find(counts.begin(), counts.end(), missing) != counts.end())
        throw std::logic_error(
            "spatial tile mosaic did not publish every core owner");

    const std::size_t laneCount =
        executionLaneCount(tiles, stagingMemory, executionMemory);
    return {tiles.tiles().size(),
            tiles.tiles().size(),
            tiles.ghostPointCount(),
            tiles.peakPointCount(),
            laneCount,
            laneReuseCount(tiles, laneCount, executionMemory)};
}

SpatialTileExecutionStats tiledRadiusScaledValues(
    const PointBatch& source, const SpatialTileSet& tiles,
    std::uint8_t queryDimensions, double radius, double factor,
    DimensionRegistry& registry, MemoryResource& stagingMemory,
    MemoryResource& executionMemory, std::span<double> values)
{
    if (queryDimensions != 2U && queryDimensions != 3U)
        throw std::invalid_argument(
            "tiled radius query dimensions must be two or three");
    if (queryDimensions < tiles.config().dimensions)
        throw std::invalid_argument(
            "tiled radius query cannot ignore a partition axis");
    if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument(
            "tiled radius query must be finite and positive");
    if (tiles.config().halo < radius)
        throw std::invalid_argument(
            "spatial tile halo is smaller than the query radius");
    if (tiles.sourcePointCount() != source.size() ||
        values.size() != source.size())
        throw std::invalid_argument("tiled radius source/result size mismatch");

    if (executionMemory.kind() == MemoryKind::Device)
        detail::tiledRadiusScaledValuesDevice(
            source, tiles, queryDimensions, radius, factor, registry,
            stagingMemory, executionMemory, values);
    else
    {
        if (!hostVisible(executionMemory.kind()))
            throw std::invalid_argument(
                "unsupported tiled radius execution memory");
        const std::array<DimensionId, 3> coordinateIds{X, Y, Z};
        const std::span<const DimensionId> selected(coordinateIds.data(),
                                                    queryDimensions);
        for (const SpatialTile& tile : tiles.tiles())
        {
            PointBatch tileBatch = gatherSpatialTile(source, tile, selected,
                                                     registry, executionMemory);
            const UniformGridConfig config =
                makeUniformGridConfig(tileBatch, queryDimensions, radius);
            SpatialIndex index(tileBatch, config);
            index.build();
            std::vector<double> local(tileBatch.size());
            radiusScaledValues(index, radius, factor, local.data());
            scatterSpatialTileOwned(tile, std::span<const double>(local),
                                    values);
        }
    }

    std::vector<std::uint8_t> published(source.size(), 0U);
    for (const SpatialTile& tile : tiles.tiles())
        for (std::size_t local = 0; local < tile.sourcePointIds.size(); ++local)
            if (!tile.ghost[local])
                published.at(tile.sourcePointIds[local]) = 1U;
    if (std::find(published.begin(), published.end(), std::uint8_t{0}) !=
        published.end())
        throw std::logic_error(
            "spatial tile mosaic did not publish every core owner");

    const std::size_t laneCount =
        executionLaneCount(tiles, stagingMemory, executionMemory);
    return {tiles.tiles().size(),
            tiles.tiles().size(),
            tiles.ghostPointCount(),
            tiles.peakPointCount(),
            laneCount,
            laneReuseCount(tiles, laneCount, executionMemory)};
}

} // namespace pdg
