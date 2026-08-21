#include <pdg/index/RasterGrid.hpp>

#include <pdg/Memory.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pdg
{
namespace
{
struct CoreShape
{
    std::size_t rows = 0U;
    std::size_t columns = 0U;
};

std::size_t checkedAdd(std::size_t left, std::size_t right)
{
    constexpr std::size_t Max = (std::numeric_limits<std::size_t>::max)();
    if (Max - left < right)
        return Max;
    return left + right;
}

std::size_t checkedMul(std::size_t left, std::size_t right)
{
    if (left == 0U || right == 0U)
        return 0U;
    if (right > (std::numeric_limits<std::size_t>::max)() / left)
        throw std::overflow_error(
            "raster grid geometry exceeds addressable index range");
    return left * right;
}

std::size_t checkedAddOrThrow(std::size_t left, std::size_t right,
                              const char* message)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
        throw std::overflow_error(message);
    return left + right;
}

void validateFrame(const RasterGridFrame& frame)
{
    if (frame.policy != RasterGridFramePolicy::PmfV1)
        throw std::invalid_argument(
            "raster grid frame policy is not supported");
    if (!std::isfinite(frame.minimumX) || !std::isfinite(frame.minimumY))
        throw std::invalid_argument("raster grid frame origin must be finite");
    if (!std::isfinite(frame.cellSize) || frame.cellSize <= 0.0)
        throw std::invalid_argument(
            "raster grid frame cell size must be finite and positive");
    if (frame.rows == 0U || frame.columns == 0U)
        throw std::invalid_argument(
            "raster grid frame dimensions must be positive");
    checkedMul(frame.rows, frame.columns);
}

void validateOptions(const RasterGridTileOptions& options)
{
    if (options.maximumExpandedCells == 0U)
        throw std::invalid_argument(
            "raster grid maximum expanded cell capacity must be positive");
}

CoreShape chooseCoreShape(std::size_t frameRows, std::size_t frameColumns,
                          std::size_t halo, std::size_t maximumExpanded)
{
    if (checkedMul(frameRows, frameColumns) <= maximumExpanded)
        return {frameRows, frameColumns};

    const std::size_t doubleHalo = checkedAdd(halo, halo);
    const std::size_t minimumExpandedRows =
        (std::min)(frameRows, checkedAdd(1U, doubleHalo));
    const std::size_t minimumExpandedColumns =
        (std::min)(frameColumns, checkedAdd(1U, doubleHalo));
    if (checkedMul(minimumExpandedRows, minimumExpandedColumns) >
        maximumExpanded)
        throw std::invalid_argument(
            "raster grid maximum expanded tile capacity is too small");

    // Choose a near-square expanded tile in constant time. If one frame axis
    // is narrow, spend the remaining budget along the other axis. Interior
    // tiles reserve both halo faces; clipped boundary tiles can only shrink.
    const auto squareRoot = static_cast<std::size_t>(
        std::floor(std::sqrt(static_cast<long double>(maximumExpanded))));
    std::size_t expandedRows =
        (std::min)(frameRows, (std::max)(squareRoot, minimumExpandedRows));
    std::size_t expandedColumns =
        (std::min)(frameColumns, maximumExpanded / expandedRows);
    if (expandedColumns < minimumExpandedColumns)
    {
        expandedColumns = minimumExpandedColumns;
        expandedRows = (std::min)(frameRows, maximumExpanded / expandedColumns);
    }
    if (expandedRows < minimumExpandedRows ||
        expandedColumns < minimumExpandedColumns)
        throw std::invalid_argument(
            "raster grid maximum expanded tile capacity is too small");

    const std::size_t coreRows =
        frameRows <= expandedRows ? frameRows : expandedRows - doubleHalo;
    const std::size_t coreColumns = frameColumns <= expandedColumns
                                        ? frameColumns
                                        : expandedColumns - doubleHalo;
    if (!coreRows || !coreColumns)
        throw std::invalid_argument(
            "raster grid maximum expanded tile capacity is too small");
    return {coreRows, coreColumns};
}

void addCount(std::size_t& total, std::size_t increment)
{
    if (increment > (std::numeric_limits<std::size_t>::max)() - total)
        throw std::overflow_error("raster grid statistics overflow");
    total += increment;
}

std::size_t clipLower(std::size_t start, std::size_t halo)
{
    return (start > halo) ? (start - halo) : 0U;
}

std::size_t clipUpper(std::size_t start, std::size_t span, std::size_t limit,
                      std::size_t halo)
{
    const std::size_t stop = checkedAdd(start, span);
    const std::size_t upper = checkedAdd(stop, halo);
    return (std::min)(upper, limit);
}
} // namespace

const RasterGridFrame& RasterGridTileSet::frame() const noexcept
{
    return m_frame;
}

const std::vector<RasterGridTile>& RasterGridTileSet::tiles() const noexcept
{
    return m_tiles;
}

std::size_t RasterGridTileSet::ownedCellCount() const noexcept
{
    return m_ownedCellCount;
}

std::size_t RasterGridTileSet::haloCellCount() const noexcept
{
    return m_haloCellCount;
}

std::size_t RasterGridTileSet::peakExpandedCellCount() const noexcept
{
    return m_peakExpandedCellCount;
}

RasterGridTileSet makeRasterGridTiles(const RasterGridFrame& frame,
                                      const RasterGridTileOptions& options)
{
    validateFrame(frame);
    validateOptions(options);

    const CoreShape core =
        chooseCoreShape(frame.rows, frame.columns, options.haloCells,
                        options.maximumExpandedCells);

    RasterGridTileSet result;
    result.m_frame = frame;
    const std::size_t expected = checkedMul(frame.rows, frame.columns);

    for (std::size_t coreColumn = 0U; coreColumn < frame.columns;)
    {
        const std::size_t coreColumns =
            (std::min)(core.columns, frame.columns - coreColumn);
        for (std::size_t coreRow = 0U; coreRow < frame.rows;)
        {
            RasterGridTile tile;
            tile.frameRows = frame.rows;
            tile.frameColumns = frame.columns;
            tile.coreColumn = coreColumn;
            tile.coreRow = coreRow;
            tile.coreColumns = coreColumns;
            tile.coreRows = (std::min)(core.rows, frame.rows - coreRow);

            tile.expandedColumn = clipLower(coreColumn, options.haloCells);
            const std::size_t expandedColumnStop =
                (std::min)(clipUpper(coreColumn, tile.coreColumns,
                                     frame.columns, options.haloCells),
                           frame.columns);
            tile.expandedColumns = expandedColumnStop - tile.expandedColumn;

            tile.expandedRow = clipLower(coreRow, options.haloCells);
            const std::size_t expandedRowStop =
                (std::min)(clipUpper(coreRow, tile.coreRows, frame.rows,
                                     options.haloCells),
                           frame.rows);
            tile.expandedRows = expandedRowStop - tile.expandedRow;

            const std::size_t expandedCount =
                checkedMul(tile.expandedRows, tile.expandedColumns);
            if (expandedCount > options.maximumExpandedCells)
                throw std::invalid_argument(
                    "raster grid tile exceeds maximum expanded capacity");

            const std::size_t coreCount =
                checkedMul(tile.coreRows, tile.coreColumns);
            addCount(result.m_ownedCellCount, coreCount);
            addCount(
                result.m_haloCellCount,
                (expandedCount >= coreCount ? expandedCount - coreCount : 0U));
            result.m_peakExpandedCellCount =
                (std::max)(result.m_peakExpandedCellCount, expandedCount);

            result.m_tiles.push_back(std::move(tile));
            coreRow += result.m_tiles.back().coreRows;
        }
        coreColumn += coreColumns;
    }

    if (result.m_tiles.empty() || result.m_ownedCellCount != expected)
        throw std::logic_error("raster grid core tiles do not cover the frame");
    return result;
}

RasterGridProduct::RasterGridProduct(const RasterGridFrame& frame,
                                     const RasterGridProductConfig& config,
                                     MemoryResource& stagingMemory,
                                     MemoryResource& executionMemory)
    : m_frame(frame), m_stagingMemory(&stagingMemory),
      m_executionMemory(&executionMemory)
{
    validateFrame(frame);
    if (!config.deviceBytesPerExpandedCell || !config.hostBytesPerCell ||
        !config.hostTileBytesPerExpandedCell || config.hostBackingCount != 2U ||
        !config.deviceMemoryBudgetBytes || !config.hostMemoryBudgetBytes ||
        config.baseDeviceBytes >= config.deviceMemoryBudgetBytes ||
        config.baseHostBytes >= config.hostMemoryBudgetBytes)
        throw std::invalid_argument(
            "raster grid product has an invalid memory contract");
    if (config.hostBytesPerCell % config.hostBackingCount != 0U)
        throw std::invalid_argument(
            "raster grid host bytes do not divide its backings");
    if (config.deviceBackingCount != 0U &&
        (config.deviceBackingCount != 2U ||
         config.deviceBytesPerExpandedCell % config.deviceBackingCount != 0U))
        throw std::invalid_argument(
            "raster grid device bytes do not divide its phase backings");
    if (config.deviceProofBytesPerCell != 0U &&
        (config.deviceBackingCount != 2U ||
         config.deviceProofBytesPerCell != config.deviceBytesPerExpandedCell))
        throw std::invalid_argument(
            "raster grid proof workspace cannot promote into its phase "
            "backings");

    const std::size_t availableDeviceBytes =
        config.deviceMemoryBudgetBytes - config.baseDeviceBytes;
    const std::size_t maximumExpandedCells =
        availableDeviceBytes / config.deviceBytesPerExpandedCell;
    m_tiles = makeRasterGridTiles(
        frame, {.haloCells = config.haloCells,
                .maximumExpandedCells = maximumExpandedCells});

    const std::size_t bytesPerBacking =
        config.hostBytesPerCell / config.hostBackingCount;
    if (config.deviceBackingCount != 0U &&
        config.deviceBytesPerExpandedCell / config.deviceBackingCount !=
            bytesPerBacking)
        throw std::invalid_argument(
            "raster grid host and device phase backing widths differ");
    m_backingBytes = checkedMul(frame.size(), bytesPerBacking);
    const std::size_t totalBackingBytes =
        checkedMul(m_backingBytes, config.hostBackingCount);
    m_tileScratchBytes = checkedMul(m_tiles.peakExpandedCellCount(),
                                    config.hostTileBytesPerExpandedCell);
    const std::size_t productHostBytes =
        checkedAddOrThrow(totalBackingBytes, m_tileScratchBytes,
                          "raster grid host working set overflows");
    m_totalHostBytes =
        checkedAddOrThrow(config.baseHostBytes, productHostBytes,
                          "raster grid cumulative host working set overflows");
    if (m_totalHostBytes > config.hostMemoryBudgetBytes)
        throw std::invalid_argument(
            "raster grid host working set exceeds its budget");
    m_backing =
        stagingMemory.allocate(totalBackingBytes, alignof(std::max_align_t));
    m_tileScratch =
        stagingMemory.allocate(m_tileScratchBytes, alignof(std::max_align_t));

    const std::size_t tileDeviceBytes = checkedMul(
        m_tiles.peakExpandedCellCount(), config.deviceBytesPerExpandedCell);
    m_schedule.itemCount = frame.size();
    for (const RasterGridTile& tile : m_tiles.tiles())
        m_schedule.tileItemCapacity =
            (std::max)(m_schedule.tileItemCapacity,
                       checkedMul(tile.coreRows, tile.coreColumns));
    m_schedule.tileCount = m_tiles.tiles().size();
    m_schedule.configuredLaneCount = 1U;
    m_schedule.activeLaneCount = 1U;
    m_schedule.laneReuseCount = m_schedule.tileCount - 1U;
    m_schedule.peakLaneBytes =
        checkedAddOrThrow(config.baseDeviceBytes, tileDeviceBytes,
                          "raster grid device working set overflows");
    if (m_schedule.peakLaneBytes > config.deviceMemoryBudgetBytes)
        throw std::invalid_argument(
            "raster grid device working set exceeds its budget");
    m_schedule.memoryLimited = m_schedule.tileCount > 1U;
    m_schedule.serialDependency = true;

    if (config.deviceBackingCount == 2U && m_schedule.tileCount == 1U &&
        executionMemory.kind() == MemoryKind::Device)
    {
        m_deviceBackingBytes = tileDeviceBytes / config.deviceBackingCount;
        m_canMaterializeDeviceBackings = true;
    }
    if (config.deviceProofBytesPerCell != 0U &&
        executionMemory.kind() == MemoryKind::Device)
    {
        m_deviceProofWorkspaceBytes =
            checkedMul(frame.size(), config.deviceProofBytesPerCell);
        const std::size_t proofPeak = checkedAddOrThrow(
            config.baseDeviceBytes, m_deviceProofWorkspaceBytes,
            "raster grid proof working set overflows");
        m_canMaterializeDeviceProofWorkspace =
            proofPeak <= config.deviceMemoryBudgetBytes;
    }
}

RasterGridProduct::~RasterGridProduct() = default;
RasterGridProduct::RasterGridProduct(RasterGridProduct&&) noexcept = default;
RasterGridProduct&
RasterGridProduct::operator=(RasterGridProduct&&) noexcept = default;

const RasterGridFrame& RasterGridProduct::frame() const noexcept
{
    return m_frame;
}

const RasterGridTileSet& RasterGridProduct::tiles() const noexcept
{
    return m_tiles;
}

const TiledSchedule& RasterGridProduct::schedule() const noexcept
{
    return m_schedule;
}

std::size_t RasterGridProduct::backingBytes() const noexcept
{
    return m_backingBytes;
}

void* RasterGridProduct::currentBacking() noexcept
{
    return static_cast<std::byte*>(m_backing->data()) +
           m_currentBacking * m_backingBytes;
}

const void* RasterGridProduct::currentBacking() const noexcept
{
    return static_cast<const std::byte*>(m_backing->data()) +
           m_currentBacking * m_backingBytes;
}

void* RasterGridProduct::nextBacking() noexcept
{
    return static_cast<std::byte*>(m_backing->data()) +
           (1U - m_currentBacking) * m_backingBytes;
}

const void* RasterGridProduct::nextBacking() const noexcept
{
    return static_cast<const std::byte*>(m_backing->data()) +
           (1U - m_currentBacking) * m_backingBytes;
}

void* RasterGridProduct::tileScratch() noexcept
{
    return m_tileScratch->data();
}

const void* RasterGridProduct::tileScratch() const noexcept
{
    return m_tileScratch->data();
}

std::size_t RasterGridProduct::tileScratchBytes() const noexcept
{
    return m_tileScratchBytes;
}

std::size_t RasterGridProduct::totalHostBytes() const noexcept
{
    return m_totalHostBytes;
}

bool RasterGridProduct::canMaterializeDeviceProofWorkspace() const noexcept
{
    return m_canMaterializeDeviceProofWorkspace;
}

bool RasterGridProduct::materializeDeviceProofWorkspace()
{
    if (!m_canMaterializeDeviceProofWorkspace)
        return false;
    if (hasPendingRasterBuild() || m_rasterBuildCount != m_rasterConsumeCount)
        throw std::logic_error(
            "raster grid proof workspace requires a consumed generation");
    if (hasDeviceProofWorkspace())
        throw std::logic_error(
            "raster grid proof workspace is already materialized");
    if (!m_deviceBacking)
    {
        m_deviceBacking = m_executionMemory->allocate(
            m_deviceProofWorkspaceBytes, alignof(std::max_align_t));
        ++m_deviceBackingAllocationCount;
    }
    m_currentDeviceBacking = 0U;
    m_deviceBackingsPromoted = false;
    m_deviceRasterBuild = false;
    m_deviceProofWorkspaceActive = true;
    return true;
}

bool RasterGridProduct::hasDeviceProofWorkspace() const noexcept
{
    return m_deviceProofWorkspaceActive && m_deviceBacking != nullptr;
}

std::size_t RasterGridProduct::deviceProofWorkspaceBytes() const noexcept
{
    return m_deviceProofWorkspaceBytes;
}

void* RasterGridProduct::deviceProofWorkspace() noexcept
{
    return hasDeviceProofWorkspace() ? m_deviceBacking->data() : nullptr;
}

const void* RasterGridProduct::deviceProofWorkspace() const noexcept
{
    return hasDeviceProofWorkspace() ? m_deviceBacking->data() : nullptr;
}

void RasterGridProduct::discardDeviceProofWorkspace()
{
    if (!hasDeviceProofWorkspace() || m_deviceBackingsPromoted ||
        hasPendingRasterBuild() || m_rasterBuildCount != m_rasterConsumeCount)
        throw std::logic_error(
            "raster grid product has no discardable proof workspace");
    m_deviceBacking.reset();
    m_deviceProofWorkspaceActive = false;
    m_currentDeviceBacking = 0U;
    m_deviceRasterBuild = false;
}

void RasterGridProduct::publishDeviceRasterBuild()
{
    if (!hasDeviceProofWorkspace() || m_deviceBackingsPromoted ||
        hasPendingRasterBuild() || m_rasterBuildCount != m_rasterConsumeCount ||
        m_rasterBuildCount == (std::numeric_limits<std::size_t>::max)() ||
        m_deviceProofWorkspaceBytes != 2U * m_deviceBackingBytes)
        throw std::logic_error(
            "raster grid device proof cannot publish its phase backings");
    m_deviceProofWorkspaceActive = false;
    m_deviceBackingsPromoted = true;
    m_deviceRasterBuild = true;
    ++m_rasterBuildCount;
}

bool RasterGridProduct::deviceRasterBuild() const noexcept
{
    return m_deviceRasterBuild;
}

bool RasterGridProduct::hasResidentDeviceBackings() const noexcept
{
    return m_deviceBacking != nullptr && m_deviceBackingsPromoted;
}

bool RasterGridProduct::canMaterializeResidentDeviceBackings() const noexcept
{
    return m_canMaterializeDeviceBackings;
}

bool RasterGridProduct::materializeResidentDeviceBackings()
{
    if (!m_canMaterializeDeviceBackings)
        return false;
    if (!hasPendingRasterBuild())
        throw std::logic_error(
            "raster grid device backings require one pending raster build");
    if (hasDeviceProofWorkspace())
        throw std::logic_error(
            "raster grid proof workspace is not a published phase pair");
    if (hasResidentDeviceBackings())
        return true;
    if (!m_deviceBacking)
    {
        m_deviceBacking = m_executionMemory->allocate(
            2U * m_deviceBackingBytes, alignof(std::max_align_t));
        ++m_deviceBackingAllocationCount;
    }
    m_deviceBackingsPromoted = true;
    return true;
}

std::size_t RasterGridProduct::deviceBackingBytes() const noexcept
{
    return m_deviceBackingBytes;
}

std::size_t RasterGridProduct::deviceBackingAllocationCount() const noexcept
{
    return m_deviceBackingAllocationCount;
}

void* RasterGridProduct::currentDeviceBacking() noexcept
{
    if (!hasResidentDeviceBackings())
        return nullptr;
    return static_cast<std::byte*>(m_deviceBacking->data()) +
           m_currentDeviceBacking * m_deviceBackingBytes;
}

const void* RasterGridProduct::currentDeviceBacking() const noexcept
{
    if (!hasResidentDeviceBackings())
        return nullptr;
    return static_cast<const std::byte*>(m_deviceBacking->data()) +
           m_currentDeviceBacking * m_deviceBackingBytes;
}

void* RasterGridProduct::nextDeviceBacking() noexcept
{
    if (!hasResidentDeviceBackings())
        return nullptr;
    return static_cast<std::byte*>(m_deviceBacking->data()) +
           (1U - m_currentDeviceBacking) * m_deviceBackingBytes;
}

const void* RasterGridProduct::nextDeviceBacking() const noexcept
{
    if (!hasResidentDeviceBackings())
        return nullptr;
    return static_cast<const std::byte*>(m_deviceBacking->data()) +
           (1U - m_currentDeviceBacking) * m_deviceBackingBytes;
}

void RasterGridProduct::swapDeviceBackings() noexcept
{
    if (hasResidentDeviceBackings())
        m_currentDeviceBacking = 1U - m_currentDeviceBacking;
}

void RasterGridProduct::prepareHostRasterBuild()
{
    if (hasPendingRasterBuild() || m_rasterBuildCount != m_rasterConsumeCount ||
        hasDeviceProofWorkspace())
        throw std::logic_error(
            "raster grid host build requires a consumed generation");
    m_currentBacking = 0U;
    m_currentDeviceBacking = 0U;
    m_deviceBackingsPromoted = false;
    m_deviceRasterBuild = false;
}

void RasterGridProduct::publishRasterBuild()
{
    if (hasPendingRasterBuild() || m_rasterBuildCount != m_rasterConsumeCount ||
        hasDeviceProofWorkspace() ||
        m_rasterBuildCount == (std::numeric_limits<std::size_t>::max)())
        throw std::logic_error(
            "raster grid product already has a pending build");
    m_currentDeviceBacking = 0U;
    m_deviceBackingsPromoted = false;
    m_deviceRasterBuild = false;
    ++m_rasterBuildCount;
}

bool RasterGridProduct::hasPendingRasterBuild() const noexcept
{
    return m_rasterConsumeCount < m_rasterBuildCount;
}

void RasterGridProduct::consumeRasterBuild()
{
    if (!hasPendingRasterBuild())
        throw std::logic_error(
            "raster grid consumer has no published raster build");
    ++m_rasterConsumeCount;
}

std::size_t RasterGridProduct::rasterBuildCount() const noexcept
{
    return m_rasterBuildCount;
}

void RasterGridProduct::swapBackings() noexcept
{
    m_currentBacking = 1U - m_currentBacking;
}

MemoryResource& RasterGridProduct::stagingMemory() noexcept
{
    return *m_stagingMemory;
}

MemoryResource& RasterGridProduct::executionMemory() noexcept
{
    return *m_executionMemory;
}

} // namespace pdg
