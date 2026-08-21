#pragma once

#include <pdg/Scheduler.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace pdg
{

class Allocation;
class MemoryResource;

enum class RasterGridFramePolicy : std::uint8_t
{
    None = 0U,
    PmfV1 = 1U
};

struct RasterGridFrame
{
    double minimumX = 0.0;
    double minimumY = 0.0;
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    double cellSize = 1.0;
    RasterGridFramePolicy policy = RasterGridFramePolicy::None;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return rows * columns;
    }
};

struct RasterGridTile
{
    std::size_t coreRow = 0U;
    std::size_t coreColumn = 0U;
    std::size_t coreRows = 0U;
    std::size_t coreColumns = 0U;

    std::size_t expandedRow = 0U;
    std::size_t expandedColumn = 0U;
    std::size_t expandedRows = 0U;
    std::size_t expandedColumns = 0U;

    std::size_t frameRows = 0U;
    std::size_t frameColumns = 0U;

    [[nodiscard]] std::size_t expandedCellCount() const noexcept
    {
        return expandedRows * expandedColumns;
    }
};

struct RasterGridTileOptions
{
    std::size_t haloCells = 0U;
    std::size_t maximumExpandedCells = 0U;
};

struct RasterGridTileSet
{
    [[nodiscard]] const RasterGridFrame& frame() const noexcept;
    [[nodiscard]] const std::vector<RasterGridTile>& tiles() const noexcept;
    [[nodiscard]] std::size_t ownedCellCount() const noexcept;
    [[nodiscard]] std::size_t haloCellCount() const noexcept;
    [[nodiscard]] std::size_t peakExpandedCellCount() const noexcept;

private:
    friend RasterGridTileSet
    makeRasterGridTiles(const RasterGridFrame& frame,
                        const RasterGridTileOptions& options);

    RasterGridFrame m_frame;
    std::vector<RasterGridTile> m_tiles;
    std::size_t m_ownedCellCount = 0U;
    std::size_t m_haloCellCount = 0U;
    std::size_t m_peakExpandedCellCount = 0U;
};

[[nodiscard]] RasterGridTileSet
makeRasterGridTiles(const RasterGridFrame& frame,
                    const RasterGridTileOptions& options);

struct RasterGridProductConfig
{
    std::size_t haloCells = 0U;
    std::size_t deviceBytesPerExpandedCell = 0U;
    std::size_t deviceBackingCount = 0U;
    std::size_t deviceProofBytesPerCell = 0U;
    std::size_t hostBytesPerCell = 0U;
    std::size_t hostTileBytesPerExpandedCell = 0U;
    std::size_t hostBackingCount = 0U;
    std::size_t baseDeviceBytes = 0U;
    std::size_t baseHostBytes = 0U;
    std::size_t deviceMemoryBudgetBytes = 0U;
    std::size_t hostMemoryBudgetBytes = 0U;
};

// Planner-scoped raster storage. The complete global backings live in the
// staging resource, while the tile schedule bounds all execution-resource
// allocations. A serial dependency is explicit because every owner mosaic
// completes before the next stencil phase can begin.
class RasterGridProduct
{
public:
    RasterGridProduct(const RasterGridFrame& frame,
                      const RasterGridProductConfig& config,
                      MemoryResource& stagingMemory,
                      MemoryResource& executionMemory);
    ~RasterGridProduct();

    RasterGridProduct(const RasterGridProduct&) = delete;
    RasterGridProduct& operator=(const RasterGridProduct&) = delete;
    RasterGridProduct(RasterGridProduct&&) noexcept;
    RasterGridProduct& operator=(RasterGridProduct&&) noexcept;

    [[nodiscard]] const RasterGridFrame& frame() const noexcept;
    [[nodiscard]] const RasterGridTileSet& tiles() const noexcept;
    [[nodiscard]] const TiledSchedule& schedule() const noexcept;
    [[nodiscard]] std::size_t backingBytes() const noexcept;
    [[nodiscard]] void* currentBacking() noexcept;
    [[nodiscard]] const void* currentBacking() const noexcept;
    [[nodiscard]] void* nextBacking() noexcept;
    [[nodiscard]] const void* nextBacking() const noexcept;
    [[nodiscard]] void* tileScratch() noexcept;
    [[nodiscard]] const void* tileScratch() const noexcept;
    [[nodiscard]] std::size_t tileScratchBytes() const noexcept;
    [[nodiscard]] std::size_t totalHostBytes() const noexcept;
    // A provisional complete-frame proof allocation is visible only to its
    // producer. It must either be discarded without publication or promoted
    // into the next phase-pair generation by publishDeviceRasterBuild(). A
    // fully consumed generation may reuse the same allocation.
    [[nodiscard]] bool canMaterializeDeviceProofWorkspace() const noexcept;
    [[nodiscard]] bool materializeDeviceProofWorkspace();
    [[nodiscard]] bool hasDeviceProofWorkspace() const noexcept;
    [[nodiscard]] std::size_t deviceProofWorkspaceBytes() const noexcept;
    [[nodiscard]] void* deviceProofWorkspace() noexcept;
    [[nodiscard]] const void* deviceProofWorkspace() const noexcept;
    void discardDeviceProofWorkspace();
    void publishDeviceRasterBuild();
    [[nodiscard]] bool deviceRasterBuild() const noexcept;
    // A full-frame device phase pair is materialized only after a producer has
    // published a raster generation and when the complete declared bytes fit
    // the runtime Grid lane. Larger frames retain the exact host-tiled phase
    // path.
    [[nodiscard]] bool canMaterializeResidentDeviceBackings() const noexcept;
    [[nodiscard]] bool materializeResidentDeviceBackings();
    [[nodiscard]] bool hasResidentDeviceBackings() const noexcept;
    [[nodiscard]] std::size_t deviceBackingBytes() const noexcept;
    [[nodiscard]] std::size_t deviceBackingAllocationCount() const noexcept;
    [[nodiscard]] void* currentDeviceBacking() noexcept;
    [[nodiscard]] const void* currentDeviceBacking() const noexcept;
    [[nodiscard]] void* nextDeviceBacking() noexcept;
    [[nodiscard]] const void* nextDeviceBacking() const noexcept;
    void swapDeviceBackings() noexcept;
    // A host producer resets the phase orientation before writing a new
    // generation. At most one complete generation may be pending; consumers
    // claim it before either producer can reuse the backings.
    void prepareHostRasterBuild();
    void publishRasterBuild();
    [[nodiscard]] bool hasPendingRasterBuild() const noexcept;
    void consumeRasterBuild();
    [[nodiscard]] std::size_t rasterBuildCount() const noexcept;
    void swapBackings() noexcept;
    [[nodiscard]] MemoryResource& stagingMemory() noexcept;
    [[nodiscard]] MemoryResource& executionMemory() noexcept;

private:
    RasterGridFrame m_frame;
    RasterGridTileSet m_tiles;
    TiledSchedule m_schedule;
    MemoryResource* m_stagingMemory = nullptr;
    MemoryResource* m_executionMemory = nullptr;
    std::unique_ptr<Allocation> m_backing;
    std::unique_ptr<Allocation> m_tileScratch;
    std::unique_ptr<Allocation> m_deviceBacking;
    std::size_t m_backingBytes = 0U;
    std::size_t m_tileScratchBytes = 0U;
    std::size_t m_totalHostBytes = 0U;
    std::size_t m_currentBacking = 0U;
    std::size_t m_deviceBackingBytes = 0U;
    std::size_t m_deviceBackingAllocationCount = 0U;
    std::size_t m_currentDeviceBacking = 0U;
    std::size_t m_deviceProofWorkspaceBytes = 0U;
    bool m_canMaterializeDeviceBackings = false;
    bool m_canMaterializeDeviceProofWorkspace = false;
    bool m_deviceProofWorkspaceActive = false;
    bool m_deviceBackingsPromoted = false;
    bool m_deviceRasterBuild = false;
    std::size_t m_rasterBuildCount = 0U;
    std::size_t m_rasterConsumeCount = 0U;
};

template <typename T>
void gatherRasterGridExpanded(const RasterGridTile& tile,
                              std::span<const T> source,
                              std::span<T> destination)
{
    if (tile.frameRows == 0U || tile.frameColumns == 0U)
        throw std::invalid_argument("raster grid tile frame is invalid");
    const std::size_t expectedSourceSize = tile.frameRows * tile.frameColumns;
    if (source.size() != expectedSourceSize)
        throw std::invalid_argument(
            "raster grid source size does not match raster frame");
    const std::size_t expandedCount = tile.expandedRows * tile.expandedColumns;
    if (destination.size() < expandedCount)
        throw std::invalid_argument(
            "raster grid destination is too small for expanded tile");

    for (std::size_t column = 0U; column < tile.expandedColumns; ++column)
        for (std::size_t row = 0U; row < tile.expandedRows; ++row)
        {
            const std::size_t sourceColumn = tile.expandedColumn + column;
            const std::size_t sourceRow = tile.expandedRow + row;
            const std::size_t sourceCell =
                sourceColumn * tile.frameRows + sourceRow;
            const std::size_t localCell = column * tile.expandedRows + row;
            destination[localCell] = source[sourceCell];
        }
}

template <typename T>
void mosaicRasterGridOwned(const RasterGridTile& tile,
                           std::span<const T> source, std::span<T> destination)
{
    if (tile.frameRows == 0U || tile.frameColumns == 0U)
        throw std::invalid_argument("raster grid tile frame is invalid");
    const std::size_t expandedCount = tile.expandedRows * tile.expandedColumns;
    if (source.size() < expandedCount)
        throw std::invalid_argument(
            "raster grid source size does not match expanded tile");
    const std::size_t destinationSize = tile.frameRows * tile.frameColumns;
    if (destination.size() < destinationSize)
        throw std::invalid_argument(
            "raster grid destination is smaller than full raster frame");

    const std::size_t rowOffset = tile.coreRow - tile.expandedRow;
    const std::size_t columnOffset = tile.coreColumn - tile.expandedColumn;
    for (std::size_t column = 0U; column < tile.coreColumns; ++column)
        for (std::size_t row = 0U; row < tile.coreRows; ++row)
        {
            const std::size_t destinationCell =
                (tile.coreColumn + column) * tile.frameRows +
                (tile.coreRow + row);
            const std::size_t sourceCell =
                (column + columnOffset) * tile.expandedRows + (row + rowOffset);
            destination[destinationCell] = source[sourceCell];
        }
}

} // namespace pdg
