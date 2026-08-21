#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/RasterGrid.hpp>
#include <pdg/stages/Pmf.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
struct BatchFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch;

    explicit BatchFixture(std::size_t capacity)
        : batch(capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::X),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Y),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Z),
                          pdg::DimensionType::Double);
        batch.materialize(
            pdg::DimensionId(pdg::StandardDimension::Classification),
            pdg::DimensionType::Unsigned8);
        batch.setSize(capacity);
    }
};
} // unnamed namespace

TEST(Pmf, ClassifiesACompactObjectWithoutChangingPointOrder)
{
    BatchFixture fixture(25U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const std::size_t point = column * 5U + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            z[point] = point == 12U ? 5.0 : 0.0;
            classification[point] = 7U;
        }

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    const pdg::PmfResult result = pdg::classifyPmf(fixture.batch, program);

    EXPECT_EQ(result.rows, 5U);
    EXPECT_EQ(result.columns, 5U);
    EXPECT_EQ(result.groundPoints, 24U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 12U ? 1U : 2U);
}

TEST(Pmf, OnlyGroundPreservesRejectedClassifications)
{
    BatchFixture fixture(9U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < 3U; ++column)
        for (std::size_t row = 0U; row < 3U; ++row)
        {
            const std::size_t point = column * 3U + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            z[point] = point == 4U ? 6.0 : 0.0;
            classification[point] = 12U;
        }

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    program.groundClass = 9U;
    program.otherClass = 9U;
    program.onlyGround = true;
    const pdg::PmfResult result = pdg::classifyPmf(fixture.batch, program);

    EXPECT_EQ(result.groundPoints, 8U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 4U ? 12U : 9U);
}

TEST(Pmf, ExactDeviceEnvelopeRejectsInvalidOrUnboundedPrograms)
{
    pdg::PmfProgram program;
    EXPECT_TRUE(pdg::pmfProgramWithinExactDeviceEnvelope(program));

    program.cellSize = 0.0;
    EXPECT_FALSE(pdg::pmfProgramWithinExactDeviceEnvelope(program));

    program = {};
    program.maxWindowSize = 1000.0;
    EXPECT_FALSE(pdg::pmfProgramWithinExactDeviceEnvelope(program));

    program = {};
    program.groundClass = program.otherClass;
    EXPECT_FALSE(pdg::pmfProgramWithinExactDeviceEnvelope(program));
}

TEST(Pmf, TiledEnvelopePreservesFractionalFrameBeyondTheGlobalCanvasCap)
{
    constexpr std::size_t Width = 65U;
    BatchFixture fixture(Width * Width);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < Width; ++column)
        for (std::size_t row = 0U; row < Width; ++row)
        {
            const std::size_t point = column * Width + row;
            x[point] = -12.25 + static_cast<double>(column) * 0.6;
            y[point] = 3.125 + static_cast<double>(row) * 0.6;
            z[point] = static_cast<double>((column * 13U + row * 7U) % 19U);
            classification[point] = 8U;
        }

    pdg::PmfProgram program;
    program.cellSize = 0.6;
    program.maxWindowSize = 3.0;
    const pdg::PmfRasterFrame frame =
        pdg::pmfRasterFrame(fixture.batch, program);
    EXPECT_EQ(frame.rows, Width);
    EXPECT_EQ(frame.columns, Width);
    EXPECT_GT(frame.size(), pdg::PmfExactDeviceMaximumRasterCells);
    EXPECT_FALSE(pdg::pmfSupportsExactDevice(fixture.batch, program));
    EXPECT_TRUE(pdg::pmfSupportsExactTiledDevice(fixture.batch, program));
}

TEST(Pmf, TiledEnvelopeRejectsFiniteInputsWithNonfiniteRasterCenters)
{
    BatchFixture fixture(2U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    x[0] = 0.0;
    x[1] = (std::numeric_limits<double>::max)();
    y[0] = y[1] = 0.0;
    z[0] = z[1] = 0.0;

    pdg::PmfProgram program;
    program.cellSize = (std::numeric_limits<double>::max)() / 2.0;
    program.maxWindowSize = 0.0;
    EXPECT_FALSE(pdg::pmfSupportsExactDevice(fixture.batch, program));
    EXPECT_FALSE(pdg::pmfSupportsExactTiledDevice(fixture.batch, program));

    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, 3U, program.cellSize, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 1024U,
         .hostMemoryBudgetBytes = 1024U},
        fixture.memory, execution);
    EXPECT_THROW(pdg::buildPmfTiledRaster(fixture.batch, program, product),
                 std::invalid_argument);
    EXPECT_EQ(product.rasterBuildCount(), 0U);
    EXPECT_FALSE(product.hasPendingRasterBuild());
}

TEST(Pmf, DistinctNearestFillTieFailsBeforeClassificationMutation)
{
    BatchFixture fixture(3U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    x[0] = 0.0;
    y[0] = 0.0;
    z[0] = 0.0;
    x[1] = 2.0;
    y[1] = 0.0;
    z[1] = 1.0;
    x[2] = 64.0;
    y[2] = 64.0;
    z[2] = 0.0;
    std::fill_n(classification, fixture.batch.size(), std::uint8_t{7U});

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    EXPECT_THROW(static_cast<void>(pdg::classifyPmf(fixture.batch, program)),
                 std::invalid_argument);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], 7U);
}

TEST(Pmf, RebuildsPlannerOwnedRasterAfterEachConsumedGeneration)
{
    BatchFixture fixture(5U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
    {
        x[point] = static_cast<double>(point);
        y[point] = 0.0;
        z[point] = static_cast<double>(point) + 0.25;
    }

    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, 5U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 1U + 3U * pdg::PmfTiledDeviceBytesPerCell,
         .hostMemoryBudgetBytes = 1024U},
        fixture.memory, execution);
    ASSERT_GT(product.tiles().tiles().size(), 1U);

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    pdg::buildPmfTiledRaster(fixture.batch, program, product);

    EXPECT_EQ(product.rasterBuildCount(), 1U);
    EXPECT_TRUE(product.hasPendingRasterBuild());
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), 5U);
    for (std::size_t cell = 0U; cell < raster.size(); ++cell)
        EXPECT_EQ(raster[cell], z[cell]);
    EXPECT_THROW(pdg::buildPmfTiledRaster(fixture.batch, program, product),
                 std::logic_error);
    EXPECT_EQ(product.rasterBuildCount(), 1U);

    product.consumeRasterBuild();
    EXPECT_FALSE(product.hasPendingRasterBuild());
    z[2] = -4.5;
    EXPECT_NO_THROW(pdg::buildPmfTiledRaster(fixture.batch, program, product));
    EXPECT_EQ(product.rasterBuildCount(), 2U);
    EXPECT_TRUE(product.hasPendingRasterBuild());
    for (std::size_t cell = 0U; cell < raster.size(); ++cell)
        EXPECT_EQ(raster[cell], z[cell]);
    product.consumeRasterBuild();
    EXPECT_FALSE(product.hasPendingRasterBuild());
}

TEST(Pmf, PlannerRasterPreservesFirstSignedZeroMinimum)
{
    BatchFixture fixture(2U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    x[0] = 0.0;
    x[1] = 0.25;
    y[0] = y[1] = 0.0;
    z[0] = 0.0;
    z[1] = -0.0;

    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, 1U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 1024U,
         .hostMemoryBudgetBytes = 1024U},
        fixture.memory, execution);
    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    pdg::buildPmfTiledRaster(fixture.batch, program, product);

    const double value = *static_cast<const double*>(product.currentBacking());
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value),
              std::bit_cast<std::uint64_t>(0.0));
}

TEST(Pmf, PlannerRasterRejectsDistinctEquidistantBitsBeforePublishing)
{
    BatchFixture fixture(2U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    x[0] = 0.0;
    x[1] = 4.0;
    y[0] = y[1] = 0.0;
    z[0] = 0.0;
    z[1] = 1.0;

    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, 5U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 1024U,
         .hostMemoryBudgetBytes = 1024U},
        fixture.memory, execution);
    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;

    EXPECT_THROW(pdg::buildPmfTiledRaster(fixture.batch, program, product),
                 std::invalid_argument);
    EXPECT_EQ(product.rasterBuildCount(), 0U);
    EXPECT_FALSE(product.hasPendingRasterBuild());
}

TEST(Pmf, PlannerRasterAcceptsSameBitEquidistantSources)
{
    BatchFixture fixture(2U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    x[0] = 0.0;
    x[1] = 4.0;
    y[0] = y[1] = 0.0;
    z[0] = z[1] = -0.0;

    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 1U, 5U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 1024U,
         .hostMemoryBudgetBytes = 1024U},
        fixture.memory, execution);
    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;

    pdg::PmfRasterBuildFacts facts;
    EXPECT_NO_THROW(
        pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts));
    ASSERT_TRUE(product.hasPendingRasterBuild());
    EXPECT_TRUE(facts.usedBoundedSourceScan);
    EXPECT_FALSE(facts.usedOccupancyHierarchy);
    EXPECT_EQ(facts.populatedCells, 2U);
    EXPECT_EQ(facts.sourceSlotsVisited, 6U);
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), 5U);
    for (double value : raster)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(value),
                  std::bit_cast<std::uint64_t>(-0.0));
}

TEST(Pmf, PlannerRasterFillMatchesLiteralLargeOriginFractionalDistances)
{
    BatchFixture fixture(2U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    constexpr double OriginX = 1.0e12;
    constexpr double OriginY = -1.0e12;
    constexpr double Cell = 0.6;
    x[0] = OriginX;
    y[0] = OriginY;
    z[0] = 17.0;
    x[1] = OriginX + 4.2;
    y[1] = OriginY + 1.2;
    z[1] = -23.0;

    pdg::PmfProgram program;
    program.cellSize = Cell;
    program.maxWindowSize = 3.0;
    const pdg::PmfRasterFrame frame =
        pdg::pmfRasterFrame(fixture.batch, program);
    ASSERT_GT(frame.size(), 2U);
    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {frame.minimumX, frame.minimumY, frame.rows, frame.columns, Cell,
         pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes =
             1U + frame.size() * pdg::PmfTiledDeviceBytesPerCell,
         .hostMemoryBudgetBytes = 1U + frame.size() * 32U},
        fixture.memory, execution);

    pdg::PmfRasterBuildFacts facts;
    pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts);
    ASSERT_TRUE(product.hasPendingRasterBuild());
    EXPECT_TRUE(facts.usedBoundedSourceScan);
    EXPECT_FALSE(facts.usedOccupancyHierarchy);
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), frame.size());

    const std::size_t secondColumn =
        static_cast<std::size_t>(std::floor(x[1] - frame.minimumX) / Cell);
    const std::size_t secondRow =
        static_cast<std::size_t>(std::floor(y[1] - frame.minimumY) / Cell);
    const std::size_t secondCell = secondColumn * frame.rows + secondRow;
    ASSERT_LT(secondCell, frame.size());
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
    {
        const std::size_t column = cell / frame.rows;
        const std::size_t row = cell % frame.rows;
        const double targetX =
            frame.minimumX + (static_cast<double>(column) + 0.5) * Cell;
        const double targetY =
            frame.minimumY + (static_cast<double>(row) + 0.5) * Cell;
        const double firstX = frame.minimumX + 0.5 * Cell;
        const double firstY = frame.minimumY + 0.5 * Cell;
        const double secondX =
            frame.minimumX + (static_cast<double>(secondColumn) + 0.5) * Cell;
        const double secondY =
            frame.minimumY + (static_cast<double>(secondRow) + 0.5) * Cell;
        const double firstDeltaX = targetX - firstX;
        const double firstDeltaY = targetY - firstY;
        const double secondDeltaX = targetX - secondX;
        const double secondDeltaY = targetY - secondY;
        const double firstDistance =
            firstDeltaX * firstDeltaX + firstDeltaY * firstDeltaY;
        const double secondDistance =
            secondDeltaX * secondDeltaX + secondDeltaY * secondDeltaY;
        ASSERT_NE(firstDistance, secondDistance);
        EXPECT_EQ(raster[cell], firstDistance < secondDistance ? z[0] : z[1]);
    }
}

TEST(Pmf, PlannerRasterHierarchyMatchesHeterogeneousLiteralScan)
{
    constexpr std::size_t SourceCount = 256U;
    constexpr double OriginX = 1099511627776.0;
    constexpr double OriginY = -1099511627776.0;
    constexpr double Cell = 1.25;
    BatchFixture fixture(SourceCount);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    x[0] = OriginX;
    y[0] = OriginY;
    z[0] = -17.0;
    for (std::size_t source = 1U; source < SourceCount; ++source)
    {
        const std::size_t desiredCell = source * 3U;
        const double integerDelta =
            std::ceil(static_cast<double>(desiredCell) * Cell);
        x[source] = OriginX;
        y[source] = OriginY + integerDelta + 0.25;
        z[source] = -17.0 + static_cast<double>(source) * 0.125;
    }

    pdg::PmfProgram program;
    program.cellSize = Cell;
    program.maxWindowSize = 3.0;
    const pdg::PmfRasterFrame frame =
        pdg::pmfRasterFrame(fixture.batch, program);
    ASSERT_EQ(frame.columns, 1U);
    ASSERT_EQ(frame.rows, 766U);
    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {frame.minimumX, frame.minimumY, frame.rows, frame.columns, Cell,
         pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes =
             1U + frame.size() * pdg::PmfTiledDeviceBytesPerCell,
         .hostMemoryBudgetBytes = 1U + frame.size() * 32U},
        fixture.memory, execution);

    std::vector<std::size_t> sourceCells(SourceCount);
    for (std::size_t source = 0U; source < SourceCount; ++source)
    {
        sourceCells[source] = static_cast<std::size_t>(
            std::floor(y[source] - frame.minimumY) / Cell);
        ASSERT_LT(sourceCells[source], frame.size());
        if (source != 0U)
            ASSERT_GT(sourceCells[source], sourceCells[source - 1U]);
    }

    pdg::PmfRasterBuildFacts facts;
    pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts);
    EXPECT_FALSE(facts.usedBoundedSourceScan);
    EXPECT_TRUE(facts.usedOccupancyHierarchy);
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), frame.size());
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
    {
        const double target =
            frame.minimumY + (static_cast<double>(cell) + 0.5) * Cell;
        std::size_t bestSource = 0U;
        double bestDistance = (std::numeric_limits<double>::max)();
        for (std::size_t source = 0U; source < SourceCount; ++source)
        {
            const double sourceCenter =
                frame.minimumY +
                (static_cast<double>(sourceCells[source]) + 0.5) * Cell;
            const double delta = target - sourceCenter;
            const double distance = delta * delta;
            ASSERT_NE(distance, bestDistance);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestSource = source;
            }
        }
        EXPECT_EQ(std::bit_cast<std::uint64_t>(raster[cell]),
                  std::bit_cast<std::uint64_t>(z[bestSource]));
    }
}

TEST(Pmf, PlannerRasterHierarchyImprovesMeasuredSparseDistribution)
{
    constexpr std::size_t Width = 257U;
    constexpr std::size_t SourceCount = 256U;
    BatchFixture fixture(SourceCount);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    for (std::size_t source = 0U; source < SourceCount; ++source)
    {
        std::size_t column = source;
        std::size_t row = (source * 29U) % Width;
        if (source == 0U)
            column = row = 0U;
        else if (source == 1U)
            column = row = Width - 1U;
        else if (source == 2U)
        {
            column = 0U;
            row = Width - 1U;
        }
        else if (source == 3U)
        {
            column = Width - 1U;
            row = 0U;
        }
        x[source] = static_cast<double>(column);
        y[source] = static_cast<double>(row);
        z[source] = -0.0;
    }

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    const pdg::PmfRasterFrame frame =
        pdg::pmfRasterFrame(fixture.batch, program);
    ASSERT_EQ(frame.rows, Width);
    ASSERT_EQ(frame.columns, Width);
    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {frame.minimumX, frame.minimumY, frame.rows, frame.columns,
         program.cellSize, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes =
             1U + frame.size() * pdg::PmfTiledDeviceBytesPerCell,
         .hostMemoryBudgetBytes = 1U + frame.size() * 32U},
        fixture.memory, execution);

    pdg::PmfRasterBuildFacts facts;
    pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts);
    EXPECT_EQ(facts.populatedCells, SourceCount);
    EXPECT_FALSE(facts.usedBoundedSourceScan);
    EXPECT_TRUE(facts.usedOccupancyHierarchy);
    const std::size_t voidCells = frame.size() - SourceCount;
    EXPECT_LT(facts.sourceSlotsVisited, voidCells * SourceCount / 2U);
    EXPECT_GT(facts.hierarchyNodesVisited, facts.sourceSlotsVisited);
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), frame.size());
    for (double value : raster)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(value),
                  std::bit_cast<std::uint64_t>(-0.0));
}

TEST(Pmf, PlannerRasterHierarchyProvesFourWayTieAndAllowsExactReattempt)
{
    constexpr std::size_t SourceCount = 256U;
    constexpr std::array<std::array<double, 2U>, 4U> TieCoordinates{{
        {{0.0, 1.0}},
        {{2.0, 1.0}},
        {{1.0, 0.0}},
        {{1.0, 2.0}},
    }};
    BatchFixture fixture(SourceCount);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    for (std::size_t point = 0U; point < TieCoordinates.size(); ++point)
    {
        x[point] = TieCoordinates[point][0];
        y[point] = TieCoordinates[point][1];
        z[point] = point == 1U ? 1.0 : -0.0;
    }
    x[4] = 32.0;
    y[4] = 32.0;
    z[4] = -0.0;
    std::size_t point = 5U;
    for (std::size_t column = 0U; point < SourceCount && column < 33U; ++column)
        for (std::size_t row = 0U; point < SourceCount && row < 33U; ++row)
        {
            const bool reserved =
                (column == 1U && row == 1U) || (column == 32U && row == 32U) ||
                std::any_of(
                    TieCoordinates.begin(), TieCoordinates.end(),
                    [&](const auto& coordinate)
                    {
                        return coordinate[0] == static_cast<double>(column) &&
                               coordinate[1] == static_cast<double>(row);
                    });
            if (!reserved)
            {
                x[point] = static_cast<double>(column);
                y[point] = static_cast<double>(row);
                z[point] = -0.0;
                ++point;
            }
        }
    ASSERT_EQ(point, SourceCount);

    pdg::PmfProgram program;
    program.maxWindowSize = 3.0;
    pdg::HostMemoryResource execution;
    pdg::RasterGridProduct product(
        {0.0, 0.0, 33U, 33U, 1.0, pdg::RasterGridFramePolicy::PmfV1},
        {.haloCells = 1U,
         .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
         .deviceBackingCount = 2U,
         .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
         .hostTileBytesPerExpandedCell = sizeof(double),
         .hostBackingCount = 2U,
         .baseDeviceBytes = 1U,
         .baseHostBytes = 1U,
         .deviceMemoryBudgetBytes = 65536U,
         .hostMemoryBudgetBytes = 65536U},
        fixture.memory, execution);
    pdg::PmfRasterBuildFacts facts;
    EXPECT_THROW(
        pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts),
        std::invalid_argument);
    EXPECT_TRUE(facts.usedOccupancyHierarchy);
    EXPECT_EQ(product.rasterBuildCount(), 0U);
    EXPECT_FALSE(product.hasPendingRasterBuild());

    z[1] = -0.0;
    facts = {};
    EXPECT_NO_THROW(
        pdg::buildPmfTiledRaster(fixture.batch, program, product, &facts));
    EXPECT_TRUE(facts.usedOccupancyHierarchy);
    EXPECT_EQ(product.rasterBuildCount(), 1U);
    ASSERT_TRUE(product.hasPendingRasterBuild());
    const auto raster = std::span<const double>(
        static_cast<const double*>(product.currentBacking()), 33U * 33U);
    for (double value : raster)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(value),
                  std::bit_cast<std::uint64_t>(-0.0));
}
