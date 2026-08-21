#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>
#include <pdg/index/SpatialTile.hpp>

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
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

struct TileFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;

    pdg::PointBatch makeBatch(const std::vector<std::array<double, 3>>& points)
    {
        pdg::PointBatch batch(
            points.size(),
            pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            dimensions, memory);
        for (pdg::DimensionId dimension : {X, Y, Z})
            batch.materialize(dimension, pdg::DimensionType::Double);
        batch.setSize(points.size());
        for (std::size_t point = 0; point < points.size(); ++point)
        {
            batch.data<double>(X)[point] = points[point][0];
            batch.data<double>(Y)[point] = points[point][1];
            batch.data<double>(Z)[point] = points[point][2];
        }
        return batch;
    }
};

const pdg::SpatialTile& requireTile(const pdg::SpatialTileSet& tiles,
                                    std::int64_t x, std::int64_t y)
{
    const auto position = std::find_if(
        tiles.tiles().begin(), tiles.tiles().end(),
        [=](const pdg::SpatialTile& tile)
        {
            return tile.coordinate.cell == std::array<std::int64_t, 3>{x, y, 0};
        });
    if (position == tiles.tiles().end())
        throw std::runtime_error("required test tile is missing");
    return *position;
}
} // unnamed namespace

TEST(SpatialTile, AssignsOneStableOwnerAndConservativeBoundaryGhosts)
{
    TileFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{-1.0, 0.0, 8.0},
                                               {0.0, 0.0, -8.0},
                                               {0.25, 0.0, 1.0},
                                               {1.99, 0.0, 2.0},
                                               {2.0, 0.0, 3.0},
                                               {3.0, 0.0, 4.0},
                                               {4.01, 0.0, 5.0}});
    const pdg::SpatialTileConfig config{2, 2.0, 1.0, {0.0, 0.0, 0.0}, 64U};
    const pdg::SpatialTileSet tiles = pdg::makeSpatialTiles(batch, config);

    ASSERT_EQ(tiles.sourcePointCount(), batch.size());
    ASSERT_EQ(tiles.tiles().size(), 4U);
    EXPECT_GT(tiles.ghostPointCount(), 0U);
    EXPECT_LE(tiles.peakPointCount(), config.maximumPoints);
    EXPECT_TRUE(std::is_sorted(
        tiles.tiles().begin(), tiles.tiles().end(),
        [](const pdg::SpatialTile& left, const pdg::SpatialTile& right)
        { return left.coordinate < right.coordinate; }));

    std::vector<std::size_t> ownerVisits(batch.size(), 0U);
    for (const pdg::SpatialTile& tile : tiles.tiles())
    {
        ASSERT_EQ(tile.sourcePointIds.size(), tile.ghost.size());
        EXPECT_TRUE(std::is_sorted(tile.sourcePointIds.begin(),
                                   tile.sourcePointIds.end()));
        EXPECT_EQ(tile.ownedCount,
                  static_cast<std::size_t>(std::count(
                      tile.ghost.begin(), tile.ghost.end(), std::uint8_t{0})));
        for (std::size_t local = 0; local < tile.sourcePointIds.size(); ++local)
            if (!tile.ghost[local])
                ++ownerVisits.at(tile.sourcePointIds[local]);
    }
    EXPECT_EQ(ownerVisits,
              std::vector<std::size_t>(batch.size(), std::size_t{1}));

    const pdg::SpatialTile& center = requireTile(tiles, 0, 0);
    EXPECT_EQ(center.sourcePointIds,
              (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(center.ghost, (std::vector<std::uint8_t>{1, 0, 0, 0, 1, 1}));

    pdg::PointBatch reusable(tiles.peakPointCount(), batch.coordinateEncoding(),
                             fixture.dimensions, fixture.memory);
    pdg::gatherSpatialTileInto(batch, center, {X, Y, Z}, reusable);
    ASSERT_TRUE(reusable.hasGhostMask());
    const void* const xStorage = reusable.rawData(X);
    const void* const yStorage = reusable.rawData(Y);
    const void* const zStorage = reusable.rawData(Z);
    const std::uint8_t* const ghostStorage = reusable.ghostData();
    EXPECT_EQ(reusable.size(), center.sourcePointIds.size());
    EXPECT_EQ(std::vector<std::uint8_t>(reusable.ghostData(),
                                        reusable.ghostData() + reusable.size()),
              center.ghost);

    const pdg::SpatialTile& positive = requireTile(tiles, 1, 0);
    pdg::gatherSpatialTileInto(batch, positive, {X, Y, Z}, reusable);
    EXPECT_EQ(reusable.rawData(X), xStorage);
    EXPECT_EQ(reusable.rawData(Y), yStorage);
    EXPECT_EQ(reusable.rawData(Z), zStorage);
    EXPECT_EQ(reusable.ghostData(), ghostStorage);
    EXPECT_EQ(reusable.size(), positive.sourcePointIds.size());
    for (std::size_t local = 0; local < reusable.size(); ++local)
    {
        const std::size_t source = positive.sourcePointIds[local];
        EXPECT_EQ(reusable.data<double>(X)[local],
                  batch.data<double>(X)[source]);
        EXPECT_EQ(reusable.data<double>(Y)[local],
                  batch.data<double>(Y)[source]);
        EXPECT_EQ(reusable.data<double>(Z)[local],
                  batch.data<double>(Z)[source]);
        EXPECT_EQ(reusable.ghostData()[local], positive.ghost[local]);
    }

    pdg::PointBatch undersized(1U, batch.coordinateEncoding(),
                               fixture.dimensions, fixture.memory);
    EXPECT_THROW(
        pdg::gatherSpatialTileInto(batch, center, {X, Y, Z}, undersized),
        std::length_error);

    std::vector<std::uint32_t> mosaic(
        batch.size(), (std::numeric_limits<std::uint32_t>::max)());
    for (const pdg::SpatialTile& tile : tiles.tiles())
    {
        std::vector<std::uint32_t> local(tile.sourcePointIds.size(), 9999U);
        for (std::size_t point = 0; point < local.size(); ++point)
            if (!tile.ghost[point])
                local[point] = static_cast<std::uint32_t>(
                    tile.sourcePointIds[point] * 10U);
        pdg::scatterSpatialTileOwned(tile,
                                     std::span<const std::uint32_t>(local),
                                     std::span<std::uint32_t>(mosaic));
    }
    EXPECT_EQ(mosaic, (std::vector<std::uint32_t>{0, 10, 20, 30, 40, 50, 60}));
}

TEST(SpatialTile, MosaickedRadiusCountsMatchWholeViewAcrossSeams)
{
    TileFixture fixture;
    std::vector<std::array<double, 3>> points;
    for (int row = -5; row <= 5; ++row)
        for (int column = -7; column <= 7; ++column)
            points.push_back(
                {static_cast<double>(column) * 0.5,
                 static_cast<double>(row) * 0.5,
                 static_cast<double>((row * 17 + column * 13) % 7) * 0.05});
    points.push_back(points[17]);
    points.push_back({2.0, 0.0, 0.0});
    points.push_back({3.0, 0.0, 0.0});
    pdg::PointBatch batch = fixture.makeBatch(points);

    constexpr double Radius = 1.0;
    const pdg::UniformGridConfig wholeConfig =
        pdg::makeUniformGridConfig(batch, 3, Radius);
    pdg::SpatialIndex wholeIndex(batch, wholeConfig);
    wholeIndex.build();
    std::vector<std::uint32_t> expected(batch.size());
    pdg::radiusCounts(wholeIndex, Radius, expected.data());

    for (double edge : {1.25, 2.0, 3.5})
    {
        SCOPED_TRACE(edge);
        const pdg::SpatialTileConfig config{
            2, edge, Radius, {-4.0, -3.0, 0.0}, 4096U};
        const pdg::SpatialTileSet tiles = pdg::makeSpatialTiles(batch, config);
        ASSERT_GT(tiles.tiles().size(), 1U);

        std::vector<std::uint32_t> actual(
            batch.size(), (std::numeric_limits<std::uint32_t>::max)());
        for (const pdg::SpatialTile& tile : tiles.tiles())
        {
            pdg::PointBatch tileBatch = pdg::gatherSpatialTile(
                batch, tile, {X, Y, Z}, fixture.dimensions, fixture.memory);
            ASSERT_TRUE(tileBatch.hasGhostMask());
            EXPECT_EQ(std::vector<std::uint8_t>(tileBatch.ghostData(),
                                                tileBatch.ghostData() +
                                                    tileBatch.size()),
                      tile.ghost);
            const pdg::UniformGridConfig tileConfig =
                pdg::makeUniformGridConfig(tileBatch, 3, Radius);
            pdg::SpatialIndex tileIndex(tileBatch, tileConfig);
            tileIndex.build();
            std::vector<std::uint32_t> local(tileBatch.size());
            pdg::radiusCounts(tileIndex, Radius, local.data());
            pdg::scatterSpatialTileOwned(tile,
                                         std::span<const std::uint32_t>(local),
                                         std::span<std::uint32_t>(actual));
        }
        EXPECT_EQ(actual, expected);

        std::vector<std::uint32_t> dispatched(batch.size());
        const pdg::SpatialTileExecutionStats stats =
            pdg::tiledRadiusCounts(batch, tiles, 3, Radius, fixture.dimensions,
                                   fixture.memory, fixture.memory, dispatched);
        EXPECT_EQ(dispatched, expected);
        EXPECT_EQ(stats.tileCount, tiles.tiles().size());
        EXPECT_EQ(stats.indexBuilds, tiles.tiles().size());
        EXPECT_EQ(stats.ghostPointCount, tiles.ghostPointCount());
        EXPECT_EQ(stats.peakPointCount, tiles.peakPointCount());
        EXPECT_EQ(stats.executionLaneCount, 1U);
        EXPECT_EQ(stats.laneReuseCount, 0U);

        const double factor =
            1.0 / ((4.0 / 3.0) * 3.14159 * (Radius * Radius * Radius));
        std::vector<double> expectedScaled(batch.size());
        for (std::size_t point = 0; point < batch.size(); ++point)
            expectedScaled[point] =
                static_cast<double>(expected[point]) * factor;
        std::vector<double> scaled(batch.size());
        const pdg::SpatialTileExecutionStats scaledStats =
            pdg::tiledRadiusScaledValues(batch, tiles, 3, Radius, factor,
                                         fixture.dimensions, fixture.memory,
                                         fixture.memory, scaled);
        EXPECT_EQ(scaled, expectedScaled);
        EXPECT_EQ(scaledStats.tileCount, tiles.tiles().size());
        EXPECT_EQ(scaledStats.executionLaneCount, 1U);
        EXPECT_EQ(scaledStats.laneReuseCount, 0U);
    }
}

TEST(SpatialTile, RejectsInvalidFramesNonfiniteInputAndCapacityOverflow)
{
    TileFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}});
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {1, 1.0, 0.0, {0.0, 0.0, 0.0}, 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {3, 0.0, 0.0, {0.0, 0.0, 0.0}, 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {3, 1.0, -1.0, {0.0, 0.0, 0.0}, 16U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {3, 1.0, 0.5, {0.0, 0.0, 0.0}, 1U})),
                 std::length_error);
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {2, 1.0, 0.5, {0.0, 0.0, 0.0}, 16U, 1U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {2, 1.0, 0.5, {0.0, 0.0, 0.0}, 16U, 7U})),
                 std::invalid_argument);

    batch.data<double>(X)[1] = (std::numeric_limits<double>::quiet_NaN)();
    EXPECT_THROW(static_cast<void>(pdg::makeSpatialTiles(
                     batch, {2, 1.0, 0.5, {0.0, 0.0, 0.0}, 16U})),
                 std::invalid_argument);

    pdg::PointBatch empty = fixture.makeBatch({});
    const pdg::SpatialTileSet emptyTiles =
        pdg::makeSpatialTiles(empty, {2, 1.0, 0.5, {0.0, 0.0, 0.0}, 16U});
    EXPECT_TRUE(emptyTiles.tiles().empty());
    EXPECT_EQ(emptyTiles.peakPointCount(), 0U);
}
