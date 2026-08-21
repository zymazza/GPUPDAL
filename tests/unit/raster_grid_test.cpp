#include <pdg/Memory.hpp>
#include <pdg/index/RasterGrid.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
class TrackingDeviceAllocation final : public pdg::Allocation
{
public:
    explicit TrackingDeviceAllocation(std::size_t bytes) : m_bytes(bytes) {}

    void* data() noexcept override
    {
        return m_bytes.data();
    }

    const void* data() const noexcept override
    {
        return m_bytes.data();
    }

    std::size_t size() const noexcept override
    {
        return m_bytes.size();
    }

    pdg::MemoryKind kind() const noexcept override
    {
        return pdg::MemoryKind::Device;
    }

private:
    std::vector<std::byte> m_bytes;
};

class TrackingDeviceResource final : public pdg::MemoryResource
{
public:
    std::unique_ptr<pdg::Allocation> allocate(std::size_t bytes,
                                              std::size_t) override
    {
        ++allocations;
        allocatedBytes += bytes;
        return std::make_unique<TrackingDeviceAllocation>(bytes);
    }

    pdg::MemoryKind kind() const noexcept override
    {
        return pdg::MemoryKind::Device;
    }

    void* nativeStreamHandle() const noexcept override
    {
        return nullptr;
    }

    std::size_t allocations = 0U;
    std::size_t allocatedBytes = 0U;
};

std::vector<std::uint64_t> stencilStep(const std::vector<std::uint64_t>& input,
                                       std::size_t rows, std::size_t columns)
{
    std::vector<std::uint64_t> output(input.size());
    for (std::size_t column = 0U; column < columns; ++column)
        for (std::size_t row = 0U; row < rows; ++row)
        {
            const std::size_t cell = column * rows + row;
            std::uint64_t value = input[cell];
            if (row > 0U)
                value += input[cell - 1U];
            if (row + 1U < rows)
                value += input[cell + 1U];
            if (column > 0U)
                value += input[cell - rows];
            if (column + 1U < columns)
                value += input[cell + rows];
            output[cell] = value;
        }
    return output;
}
} // unnamed namespace

TEST(RasterGrid, BuildsStableCoreHaloTilesAndOwnerOnlyMosaic)
{
    const pdg::RasterGridFrame frame{
        -4.5, 2.25, 7U, 11U, 0.5, pdg::RasterGridFramePolicy::PmfV1};
    const pdg::RasterGridTileSet tiles = pdg::makeRasterGridTiles(
        frame, {.haloCells = 1U, .maximumExpandedCells = 20U});

    ASSERT_GT(tiles.tiles().size(), 1U);
    EXPECT_EQ(tiles.frame().minimumX, -4.5);
    EXPECT_EQ(tiles.frame().minimumY, 2.25);
    EXPECT_EQ(tiles.frame().rows, 7U);
    EXPECT_EQ(tiles.frame().columns, 11U);
    EXPECT_EQ(tiles.frame().cellSize, 0.5);
    EXPECT_EQ(tiles.ownedCellCount(), 77U);
    EXPECT_LE(tiles.peakExpandedCellCount(), 20U);
    EXPECT_GT(tiles.haloCellCount(), 0U);

    std::vector<std::size_t> ownerVisits(frame.size(), 0U);
    std::vector<std::uint64_t> source(frame.size());
    for (std::size_t cell = 0U; cell < source.size(); ++cell)
        source[cell] = static_cast<std::uint64_t>(cell * 17U + 3U);
    std::vector<std::uint64_t> mosaic(
        frame.size(), (std::numeric_limits<std::uint64_t>::max)());

    for (const pdg::RasterGridTile& tile : tiles.tiles())
    {
        EXPECT_GT(tile.coreRows, 0U);
        EXPECT_GT(tile.coreColumns, 0U);
        EXPECT_LE(tile.expandedCellCount(), 20U);
        EXPECT_LE(tile.expandedRow, tile.coreRow);
        EXPECT_LE(tile.expandedColumn, tile.coreColumn);
        EXPECT_GE(tile.expandedRow + tile.expandedRows,
                  tile.coreRow + tile.coreRows);
        EXPECT_GE(tile.expandedColumn + tile.expandedColumns,
                  tile.coreColumn + tile.coreColumns);

        std::vector<std::uint64_t> local(tile.expandedCellCount());
        pdg::gatherRasterGridExpanded(tile,
                                      std::span<const std::uint64_t>(source),
                                      std::span<std::uint64_t>(local));
        pdg::mosaicRasterGridOwned(tile, std::span<const std::uint64_t>(local),
                                   std::span<std::uint64_t>(mosaic));
        for (std::size_t column = 0U; column < tile.coreColumns; ++column)
            for (std::size_t row = 0U; row < tile.coreRows; ++row)
                ++ownerVisits[(tile.coreColumn + column) * frame.rows +
                              tile.coreRow + row];
    }

    EXPECT_EQ(mosaic, source);
    EXPECT_TRUE(std::all_of(ownerVisits.begin(), ownerVisits.end(),
                            [](std::size_t visits) { return visits == 1U; }));
}

TEST(RasterGrid, HaloMosaicMatchesWholeGridAtEveryCoreEdgeAndCorner)
{
    for (const auto [rows, columns, capacity] :
         {std::array<std::size_t, 3>{5U, 17U, 20U},
          std::array<std::size_t, 3>{17U, 5U, 24U},
          std::array<std::size_t, 3>{13U, 14U, 30U},
          std::array<std::size_t, 3>{2U, 19U, 12U}})
    {
        SCOPED_TRACE(rows);
        SCOPED_TRACE(columns);
        SCOPED_TRACE(capacity);
        const pdg::RasterGridFrame frame{
            -3.0, -7.0, rows, columns, 0.6, pdg::RasterGridFramePolicy::PmfV1};
        const pdg::RasterGridTileSet tiles = pdg::makeRasterGridTiles(
            frame, {.haloCells = 1U, .maximumExpandedCells = capacity});
        ASSERT_GT(tiles.tiles().size(), 1U);

        std::vector<std::uint64_t> input(frame.size());
        for (std::size_t cell = 0U; cell < input.size(); ++cell)
            input[cell] = static_cast<std::uint64_t>(
                (cell * 0x9e3779b97f4a7c15ULL) ^ (cell >> 3U));
        const std::vector<std::uint64_t> expected =
            stencilStep(input, rows, columns);
        std::vector<std::uint64_t> actual(frame.size(), 0U);

        for (const pdg::RasterGridTile& tile : tiles.tiles())
        {
            std::vector<std::uint64_t> gathered(tile.expandedCellCount());
            pdg::gatherRasterGridExpanded(tile,
                                          std::span<const std::uint64_t>(input),
                                          std::span<std::uint64_t>(gathered));
            const std::vector<std::uint64_t> local =
                stencilStep(gathered, tile.expandedRows, tile.expandedColumns);
            pdg::mosaicRasterGridOwned(tile,
                                       std::span<const std::uint64_t>(local),
                                       std::span<std::uint64_t>(actual));
        }
        EXPECT_EQ(actual, expected);
    }
}

TEST(RasterGrid, RejectsInvalidFramesAndInsufficientTileBudgets)
{
    using Policy = pdg::RasterGridFramePolicy;
    EXPECT_THROW(static_cast<void>(pdg::makeRasterGridTiles(
                     {0.0, 0.0, 0U, 2U, 1.0, Policy::PmfV1},
                     {.haloCells = 1U, .maximumExpandedCells = 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeRasterGridTiles(
                     {0.0, 0.0, 2U, 2U, 0.0, Policy::PmfV1},
                     {.haloCells = 1U, .maximumExpandedCells = 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeRasterGridTiles(
                     {0.0, 0.0, 2U, 2U, 1.0, Policy::None},
                     {.haloCells = 1U, .maximumExpandedCells = 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeRasterGridTiles(
                     {0.0, 0.0, 3U, 3U, 1.0, Policy::PmfV1},
                     {.haloCells = 1U, .maximumExpandedCells = 3U})),
                 std::invalid_argument);

    const pdg::RasterGridTileSet whole = pdg::makeRasterGridTiles(
        {0.0, 0.0, 3U, 4U, 1.0, Policy::PmfV1},
        {.haloCells = 1U, .maximumExpandedCells = 12U});
    ASSERT_EQ(whole.tiles().size(), 1U);
    EXPECT_EQ(whole.haloCellCount(), 0U);

    const pdg::RasterGridTile& tile = whole.tiles().front();
    std::vector<std::uint32_t> source(12U, 1U);
    std::vector<std::uint32_t> tooSmall(11U, 0U);
    EXPECT_THROW(pdg::gatherRasterGridExpanded(
                     tile, std::span<const std::uint32_t>(source),
                     std::span<std::uint32_t>(tooSmall)),
                 std::invalid_argument);
    EXPECT_THROW(pdg::mosaicRasterGridOwned(
                     tile, std::span<const std::uint32_t>(tooSmall),
                     std::span<std::uint32_t>(source)),
                 std::invalid_argument);
}

TEST(RasterGrid, ProductOwnsBackingAndMemoryDrivenSerialSchedule)
{
    pdg::HostMemoryResource staging;
    pdg::HostMemoryResource execution;
    const pdg::RasterGridFrame frame{
        -1.0, 4.0, 17U, 19U, 0.6, pdg::RasterGridFramePolicy::PmfV1};
    pdg::RasterGridProduct product(
        frame,
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = 2U * sizeof(double),
         .hostBytesPerCell = 2U * sizeof(double),
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 4096U,
         .deviceMemoryBudgetBytes = 8192U,
         .hostMemoryBudgetBytes = 8192U},
        staging, execution);

    EXPECT_EQ(product.frame().size(), 17U * 19U);
    EXPECT_GT(product.tiles().tiles().size(), 1U);
    EXPECT_EQ(product.backingBytes(), frame.size() * sizeof(double));
    ASSERT_NE(product.currentBacking(), nullptr);
    ASSERT_NE(product.nextBacking(), nullptr);
    ASSERT_NE(product.tileScratch(), nullptr);
    EXPECT_GE(product.tileScratchBytes(), sizeof(double));
    EXPECT_EQ(product.totalHostBytes(),
              2U * product.backingBytes() + product.tileScratchBytes());
    EXPECT_NE(product.currentBacking(), product.nextBacking());
    const void* first = product.currentBacking();
    const void* second = product.nextBacking();
    product.swapBackings();
    EXPECT_EQ(product.currentBacking(), second);
    EXPECT_EQ(product.nextBacking(), first);

    const pdg::TiledSchedule& schedule = product.schedule();
    EXPECT_EQ(schedule.itemCount, frame.size());
    EXPECT_EQ(schedule.tileCount, product.tiles().tiles().size());
    EXPECT_EQ(schedule.activeLaneCount, 1U);
    EXPECT_TRUE(schedule.serialDependency);
    EXPECT_TRUE(schedule.memoryLimited);
    EXPECT_LE(schedule.peakLaneBytes, 8192U);
    EXPECT_EQ(&product.stagingMemory(), &staging);
    EXPECT_EQ(&product.executionMemory(), &execution);
}

TEST(RasterGrid, HostBackingBudgetRejectsBeforeAllocation)
{
    pdg::HostMemoryResource staging;
    pdg::HostMemoryResource execution;
    const pdg::RasterGridFrame frame{
        0.0, 0.0, 4U, 4U, 1.0, pdg::RasterGridFramePolicy::PmfV1};
    const pdg::RasterGridProductConfig config{
        .haloCells = 1U,
        .deviceBytesPerExpandedCell = 2U * sizeof(double),
        .hostBytesPerCell = 2U * sizeof(double),
        .hostTileBytesPerExpandedCell = sizeof(double),
        .hostBackingCount = 2U,
        .baseDeviceBytes = 1024U,
        .baseHostBytes = 100U,
        .deviceMemoryBudgetBytes = 4096U,
        .hostMemoryBudgetBytes = 483U};
    EXPECT_THROW(static_cast<void>(
                     pdg::RasterGridProduct(frame, config, staging, execution)),
                 std::invalid_argument);

    pdg::RasterGridProductConfig exact = config;
    exact.hostMemoryBudgetBytes = 484U;
    pdg::RasterGridProduct product(frame, exact, staging, execution);
    EXPECT_EQ(product.totalHostBytes(), 484U);

    pdg::RasterGridProductConfig mismatchedProof = exact;
    mismatchedProof.deviceBackingCount = 2U;
    mismatchedProof.deviceProofBytesPerCell = sizeof(double);
    EXPECT_THROW(static_cast<void>(pdg::RasterGridProduct(
                     frame, mismatchedProof, staging, execution)),
                 std::invalid_argument);
}

TEST(RasterGrid, FullDevicePhaseBackingsUseTheExactRuntimeBudgetBoundary)
{
    constexpr std::size_t BaseDeviceBytes = 1024U;
    constexpr std::size_t FrameCells = 16U;
    constexpr std::size_t DeviceBytesPerCell = 2U * sizeof(double);
    constexpr std::size_t ExactDeviceBytes =
        BaseDeviceBytes + FrameCells * DeviceBytesPerCell;
    const pdg::RasterGridFrame frame{
        0.0, 0.0, 4U, 4U, 1.0, pdg::RasterGridFramePolicy::PmfV1};
    const auto config = [&](std::size_t budget)
    {
        return pdg::RasterGridProductConfig{
            .haloCells = 1U,
            .deviceBytesPerExpandedCell = DeviceBytesPerCell,
            .deviceBackingCount = 2U,
            .hostBytesPerCell = 2U * sizeof(double),
            .hostTileBytesPerExpandedCell = sizeof(double),
            .hostBackingCount = 2U,
            .baseDeviceBytes = BaseDeviceBytes,
            .deviceMemoryBudgetBytes = budget,
            .hostMemoryBudgetBytes = 4096U};
    };

    pdg::HostMemoryResource staging;
    TrackingDeviceResource belowExecution;
    pdg::RasterGridProduct below(frame, config(ExactDeviceBytes - 1U), staging,
                                 belowExecution);
    EXPECT_GT(below.schedule().tileCount, 1U);
    EXPECT_FALSE(below.canMaterializeResidentDeviceBackings());
    EXPECT_FALSE(below.materializeResidentDeviceBackings());
    EXPECT_FALSE(below.hasResidentDeviceBackings());
    EXPECT_EQ(belowExecution.allocations, 0U);

    TrackingDeviceResource exactExecution;
    pdg::RasterGridProduct exact(frame, config(ExactDeviceBytes), staging,
                                 exactExecution);
    EXPECT_EQ(exact.schedule().tileCount, 1U);
    EXPECT_TRUE(exact.canMaterializeResidentDeviceBackings());
    EXPECT_EQ(exactExecution.allocations, 0U);
    EXPECT_THROW(static_cast<void>(exact.materializeResidentDeviceBackings()),
                 std::logic_error);
    EXPECT_EQ(exactExecution.allocations, 0U);
    EXPECT_THROW(exact.consumeRasterBuild(), std::logic_error);
    exact.publishRasterBuild();
    EXPECT_TRUE(exact.hasPendingRasterBuild());
    EXPECT_THROW(exact.publishRasterBuild(), std::logic_error);
    EXPECT_EQ(exact.rasterBuildCount(), 1U);
    ASSERT_TRUE(exact.materializeResidentDeviceBackings());
    ASSERT_TRUE(exact.hasResidentDeviceBackings());
    EXPECT_EQ(exact.deviceBackingBytes(), FrameCells * sizeof(double));
    EXPECT_EQ(exactExecution.allocations, 1U);
    EXPECT_EQ(exactExecution.allocatedBytes, 2U * exact.deviceBackingBytes());
    ASSERT_NE(exact.currentDeviceBacking(), nullptr);
    ASSERT_NE(exact.nextDeviceBacking(), nullptr);
    EXPECT_NE(exact.currentDeviceBacking(), exact.nextDeviceBacking());
    const void* first = exact.currentDeviceBacking();
    const void* second = exact.nextDeviceBacking();
    exact.swapDeviceBackings();
    EXPECT_EQ(exact.currentDeviceBacking(), second);
    EXPECT_EQ(exact.nextDeviceBacking(), first);
    exact.consumeRasterBuild();
    EXPECT_FALSE(exact.hasPendingRasterBuild());
    EXPECT_THROW(static_cast<void>(exact.materializeResidentDeviceBackings()),
                 std::logic_error);
    EXPECT_THROW(exact.consumeRasterBuild(), std::logic_error);
}

TEST(RasterGrid, DeviceProofWorkspacePromotesOnlyAfterPublication)
{
    constexpr std::size_t BaseDeviceBytes = 1024U;
    constexpr std::size_t FrameCells = 16U;
    constexpr std::size_t ProofBytesPerCell = 2U * sizeof(double);
    constexpr std::size_t ExactDeviceBytes =
        BaseDeviceBytes + FrameCells * ProofBytesPerCell;
    const pdg::RasterGridFrame frame{
        0.0, 0.0, 4U, 4U, 1.0, pdg::RasterGridFramePolicy::PmfV1};
    const auto config = [&](std::size_t budget)
    {
        return pdg::RasterGridProductConfig{
            .haloCells = 1U,
            .deviceBytesPerExpandedCell = 2U * sizeof(double),
            .deviceBackingCount = 2U,
            .deviceProofBytesPerCell = ProofBytesPerCell,
            .hostBytesPerCell = 2U * sizeof(double),
            .hostTileBytesPerExpandedCell = sizeof(double),
            .hostBackingCount = 2U,
            .baseDeviceBytes = BaseDeviceBytes,
            .deviceMemoryBudgetBytes = budget,
            .hostMemoryBudgetBytes = 4096U};
    };

    pdg::HostMemoryResource staging;
    TrackingDeviceResource belowExecution;
    pdg::RasterGridProduct below(frame, config(ExactDeviceBytes - 1U), staging,
                                 belowExecution);
    EXPECT_FALSE(below.canMaterializeDeviceProofWorkspace());
    EXPECT_FALSE(below.materializeDeviceProofWorkspace());
    EXPECT_EQ(belowExecution.allocations, 0U);

    TrackingDeviceResource exactExecution;
    pdg::RasterGridProduct exact(frame, config(ExactDeviceBytes), staging,
                                 exactExecution);
    EXPECT_TRUE(exact.canMaterializeDeviceProofWorkspace());
    ASSERT_TRUE(exact.materializeDeviceProofWorkspace());
    EXPECT_TRUE(exact.hasDeviceProofWorkspace());
    EXPECT_FALSE(exact.hasResidentDeviceBackings());
    EXPECT_FALSE(exact.deviceRasterBuild());
    EXPECT_EQ(exact.deviceProofWorkspaceBytes(),
              FrameCells * ProofBytesPerCell);
    EXPECT_NE(exact.deviceProofWorkspace(), nullptr);
    EXPECT_EQ(exactExecution.allocations, 1U);
    EXPECT_EQ(exactExecution.allocatedBytes, FrameCells * ProofBytesPerCell);
    EXPECT_THROW(static_cast<void>(exact.materializeDeviceProofWorkspace()),
                 std::logic_error);
    EXPECT_THROW(exact.publishRasterBuild(), std::logic_error);
    EXPECT_THROW(static_cast<void>(exact.materializeResidentDeviceBackings()),
                 std::logic_error);

    const void* proof = exact.deviceProofWorkspace();
    exact.publishDeviceRasterBuild();
    EXPECT_FALSE(exact.hasDeviceProofWorkspace());
    EXPECT_TRUE(exact.hasResidentDeviceBackings());
    EXPECT_TRUE(exact.deviceRasterBuild());
    EXPECT_TRUE(exact.hasPendingRasterBuild());
    EXPECT_EQ(exact.rasterBuildCount(), 1U);
    EXPECT_EQ(exact.currentDeviceBacking(), proof);
    EXPECT_TRUE(exact.materializeResidentDeviceBackings());
    EXPECT_EQ(exactExecution.allocations, 1U);
    EXPECT_THROW(exact.publishDeviceRasterBuild(), std::logic_error);
    EXPECT_THROW(exact.discardDeviceProofWorkspace(), std::logic_error);
    exact.consumeRasterBuild();
    EXPECT_FALSE(exact.hasPendingRasterBuild());
}

TEST(RasterGrid, ReusesPromotedDeviceAllocationForNextRasterGeneration)
{
    constexpr std::size_t FrameCells = 16U;
    constexpr std::size_t DeviceBytesPerCell = 2U * sizeof(double);
    pdg::HostMemoryResource staging;
    TrackingDeviceResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 4U, 4U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = DeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .deviceProofBytesPerCell = DeviceBytesPerCell,
         .hostBytesPerCell = 2U * sizeof(double),
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .deviceMemoryBudgetBytes = 1U + FrameCells * DeviceBytesPerCell,
         .hostMemoryBudgetBytes = 4096U},
        staging, execution);

    ASSERT_TRUE(product.materializeDeviceProofWorkspace());
    const void* allocation = product.deviceProofWorkspace();
    EXPECT_EQ(product.deviceBackingAllocationCount(), 1U);
    product.publishDeviceRasterBuild();
    product.consumeRasterBuild();
    ASSERT_FALSE(product.hasPendingRasterBuild());

    ASSERT_TRUE(product.materializeDeviceProofWorkspace());
    EXPECT_EQ(product.deviceProofWorkspace(), allocation);
    EXPECT_EQ(execution.allocations, 1U);
    EXPECT_EQ(product.deviceBackingAllocationCount(), 1U);
    EXPECT_FALSE(product.hasResidentDeviceBackings());
    product.publishDeviceRasterBuild();
    EXPECT_EQ(product.rasterBuildCount(), 2U);
    EXPECT_TRUE(product.hasPendingRasterBuild());
    product.consumeRasterBuild();
    EXPECT_FALSE(product.hasPendingRasterBuild());
}

TEST(RasterGrid, FailedDeviceProofCanBeDiscardedAndRetried)
{
    constexpr std::size_t FrameCells = 9U;
    constexpr std::size_t DeviceBytesPerCell = 2U * sizeof(double);
    pdg::HostMemoryResource staging;
    TrackingDeviceResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 3U, 3U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = DeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .deviceProofBytesPerCell = DeviceBytesPerCell,
         .hostBytesPerCell = 2U * sizeof(double),
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .deviceMemoryBudgetBytes = 1U + FrameCells * DeviceBytesPerCell,
         .hostMemoryBudgetBytes = 4096U},
        staging, execution);

    ASSERT_TRUE(product.materializeDeviceProofWorkspace());
    product.discardDeviceProofWorkspace();
    EXPECT_FALSE(product.hasDeviceProofWorkspace());
    EXPECT_FALSE(product.hasResidentDeviceBackings());
    EXPECT_EQ(product.rasterBuildCount(), 0U);
    ASSERT_TRUE(product.materializeDeviceProofWorkspace());
    EXPECT_EQ(execution.allocations, 2U);
    product.publishDeviceRasterBuild();
    EXPECT_TRUE(product.hasResidentDeviceBackings());
    EXPECT_TRUE(product.deviceRasterBuild());
}
