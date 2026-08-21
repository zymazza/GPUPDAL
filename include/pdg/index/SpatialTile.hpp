#pragma once

#include <pdg/Dimension.hpp>

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace pdg
{

class MemoryResource;
class PointBatch;

struct SpatialTileCoordinate
{
    std::array<std::int64_t, 3> cell{0, 0, 0};

    auto operator<=>(const SpatialTileCoordinate&) const = default;
};

// Core tiles are half-open [origin + cell * edge,
// origin + (cell + 1) * edge). The halo is conservative and closed on both
// faces: an exact-boundary candidate may be carried as a ghost, but the query
// primitive still applies the stage's strict distance predicate.
struct SpatialTileConfig
{
    std::uint8_t dimensions = 2;
    double edgeLength = 0.0;
    double halo = 0.0;
    std::array<double, 3> origin{0.0, 0.0, 0.0};
    std::size_t maximumPoints = (std::numeric_limits<std::size_t>::max)();
    // Internal P1.5 scheduler controls. Zero lanes selects the benchmark-fixed
    // radius-neighborhood default; zero bytes means D4 supplied no VRAM cap.
    std::size_t schedulerLanes = 0;
    std::size_t deviceMemoryBudgetBytes = 0;
};

struct SpatialTile
{
    SpatialTileCoordinate coordinate;
    // Source ids and ghost flags retain source order. A zero flag identifies
    // the point's sole owner; nonzero entries participate in neighborhood
    // queries but must never be emitted from this tile.
    std::vector<std::size_t> sourcePointIds;
    std::vector<std::uint8_t> ghost;
    std::size_t ownedCount = 0;
};

class SpatialTileSet
{
public:
    SpatialTileSet() = default;

    [[nodiscard]] const SpatialTileConfig& config() const noexcept;
    [[nodiscard]] const std::vector<SpatialTile>& tiles() const noexcept;
    [[nodiscard]] std::size_t sourcePointCount() const noexcept;
    [[nodiscard]] std::size_t ghostPointCount() const noexcept;
    [[nodiscard]] std::size_t peakPointCount() const noexcept;

private:
    friend SpatialTileSet makeSpatialTiles(const PointBatch& source,
                                           SpatialTileConfig config);

    SpatialTileConfig m_config;
    std::vector<SpatialTile> m_tiles;
    std::size_t m_sourcePointCount = 0;
    std::size_t m_ghostPointCount = 0;
    std::size_t m_peakPointCount = 0;
};

// Builds deterministic core ownership and conservative ghosts from
// host-visible logical-double XYZ. Tiling may be 2D while the eventual query
// remains 3D: every 3D neighbor within the halo is necessarily within the XY
// halo, so this only admits harmless extra Z candidates.
[[nodiscard]] SpatialTileSet makeSpatialTiles(const PointBatch& source,
                                              SpatialTileConfig config);

// Materializes selected columns into source order for one tile and attaches
// its ghost mask. Both source and destination resources must be host-visible;
// the resulting contiguous tile can then be transferred asynchronously.
[[nodiscard]] PointBatch
gatherSpatialTile(const PointBatch& source, const SpatialTile& tile,
                  std::span<const DimensionId> dimensions,
                  DimensionRegistry& registry,
                  MemoryResource& destinationMemory);
[[nodiscard]] PointBatch
gatherSpatialTile(const PointBatch& source, const SpatialTile& tile,
                  std::initializer_list<DimensionId> dimensions,
                  DimensionRegistry& registry,
                  MemoryResource& destinationMemory);

// Reuses a caller-owned host-visible batch. The destination capacity must fit
// the complete tile including ghosts and its coordinate encoding must match
// the source exactly. Columns and the ghost mask are materialized once and
// retained across calls, allowing a pinned tile lane to avoid reallocation.
void gatherSpatialTileInto(const PointBatch& source, const SpatialTile& tile,
                           std::span<const DimensionId> dimensions,
                           PointBatch& destination);
void gatherSpatialTileInto(const PointBatch& source, const SpatialTile& tile,
                           std::initializer_list<DimensionId> dimensions,
                           PointBatch& destination);

struct SpatialTileExecutionStats
{
    std::size_t tileCount = 0;
    std::size_t indexBuilds = 0;
    std::size_t ghostPointCount = 0;
    std::size_t peakPointCount = 0;
    // Device execution reports the bounded scheduler's active lane count.
    // Host execution reports its one serial path but no lane reuse.
    std::size_t executionLaneCount = 0;
    std::size_t laneReuseCount = 0;
};

// Executes an exact radius query tile by tile, then mosaics only core-owner
// rows into source order. `tiles.config().halo` must be at least `radius`.
// A 2D tiling may feed a 3D query; the reverse is rejected because Z-separated
// tiles cannot prove completeness for a query that ignores Z.
[[nodiscard]] SpatialTileExecutionStats
tiledRadiusCounts(const PointBatch& source, const SpatialTileSet& tiles,
                  std::uint8_t queryDimensions, double radius,
                  DimensionRegistry& registry, MemoryResource& stagingMemory,
                  MemoryResource& executionMemory,
                  std::span<std::uint32_t> counts);

[[nodiscard]] SpatialTileExecutionStats tiledRadiusScaledValues(
    const PointBatch& source, const SpatialTileSet& tiles,
    std::uint8_t queryDimensions, double radius, double factor,
    DimensionRegistry& registry, MemoryResource& stagingMemory,
    MemoryResource& executionMemory, std::span<double> values);

namespace detail
{
void tiledRadiusCountsDevice(const PointBatch& source,
                             const SpatialTileSet& tiles,
                             std::uint8_t queryDimensions, double radius,
                             DimensionRegistry& registry,
                             MemoryResource& stagingMemory,
                             MemoryResource& deviceMemory,
                             std::span<std::uint32_t> counts);
void tiledRadiusScaledValuesDevice(const PointBatch& source,
                                   const SpatialTileSet& tiles,
                                   std::uint8_t queryDimensions, double radius,
                                   double factor, DimensionRegistry& registry,
                                   MemoryResource& stagingMemory,
                                   MemoryResource& deviceMemory,
                                   std::span<double> values);
} // namespace detail

template <typename T>
void scatterSpatialTileOwned(const SpatialTile& tile,
                             std::span<const T> tileValues,
                             std::span<T> destination)
{
    if (tileValues.size() != tile.sourcePointIds.size() ||
        tile.ghost.size() != tile.sourcePointIds.size())
        throw std::invalid_argument("spatial tile value/mapping size mismatch");
    for (std::size_t local = 0; local < tile.sourcePointIds.size(); ++local)
    {
        const std::size_t source = tile.sourcePointIds[local];
        if (source >= destination.size())
            throw std::out_of_range(
                "spatial tile source id exceeds scatter destination");
        if (!tile.ghost[local])
            destination[source] = tileValues[local];
    }
}

} // namespace pdg
