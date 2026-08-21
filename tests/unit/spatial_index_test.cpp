#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

struct SpatialFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::CoordinateEncoding coordinates{{0.01, 0.01, 0.01},
                                        {500000.0, 4800000.0, 100.0}};

    pdg::PointBatch makeBatch(const std::vector<std::array<double, 3>>& points)
    {
        pdg::PointBatch batch(points.size(), coordinates, dimensions, memory);
        batch.materialize(X, pdg::DimensionType::Double);
        batch.materialize(Y, pdg::DimensionType::Double);
        batch.materialize(Z, pdg::DimensionType::Double);
        batch.setSize(points.size());
        for (std::size_t point = 0; point < points.size(); ++point)
        {
            batch.hostSpan<double>(X)[point] = points[point][0];
            batch.hostSpan<double>(Y)[point] = points[point][1];
            batch.hostSpan<double>(Z)[point] = points[point][2];
        }
        return batch;
    }
};

pdg::KnnConfigSummary summarizeBatch(const pdg::PointBatch& batch,
                                     std::uint8_t dimensions)
{
    pdg::KnnConfigSummary summary;
    summary.pointCount = batch.size();
    summary.dimensions = dimensions;
    if (batch.size() == 0U)
        return summary;
    const std::array<const double*, 3> values{
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z)};
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
        summary.minimum[axis] = summary.maximum[axis] = values[axis][0];
    for (std::size_t point = 1U; point < batch.size(); ++point)
        for (std::uint8_t axis = 0; axis < dimensions; ++axis)
        {
            summary.minimum[axis] =
                (std::min)(summary.minimum[axis], values[axis][point]);
            summary.maximum[axis] =
                (std::max)(summary.maximum[axis], values[axis][point]);
        }
    const std::size_t probeCount =
        (std::min)(batch.size(), pdg::KnnConfigMaximumProbePoints);
    summary.probes.resize(probeCount);
    for (std::size_t probe = 0U; probe < probeCount; ++probe)
    {
        const std::size_t point =
            pdg::knnConfigProbePoint(batch.size(), probe);
        for (std::uint8_t axis = 0; axis < dimensions; ++axis)
            summary.probes[probe][axis] = values[axis][point];
    }
    return summary;
}

void expectSameConfig(const pdg::UniformGridConfig& left,
                      const pdg::UniformGridConfig& right)
{
    EXPECT_EQ(left.dimensions, right.dimensions);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(left.cellSize),
              std::bit_cast<std::uint64_t>(right.cellSize));
    EXPECT_EQ(left.origin, right.origin);
    EXPECT_EQ(left.maximumCell, right.maximumCell);
    EXPECT_EQ(left.backend, right.backend);
    EXPECT_EQ(left.knnCandidateArrays, right.knnCandidateArrays);
}

std::vector<std::uint32_t> bruteCounts(const pdg::PointBatch& batch,
                                       std::uint8_t dimensions, double radius)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    const double radiusSquared = radius * radius;
    std::vector<std::uint32_t> counts(batch.size(), 0);
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        for (std::size_t candidate = 0; candidate < batch.size(); ++candidate)
        {
            const double dx = x[query] - x[candidate];
            const double dy = y[query] - y[candidate];
            double distance = dx * dx + dy * dy;
            if (dimensions == 3)
            {
                const double dz = z[query] - z[candidate];
                distance += dz * dz;
            }
            if (distance < radiusSquared)
                ++counts[query];
        }
    }
    return counts;
}

struct BruteNeighbor
{
    std::uint32_t point;
    double distance;
};

std::vector<BruteNeighbor> bruteNeighbors(const pdg::PointBatch& batch,
                                          std::size_t query,
                                          std::uint8_t dimensions = 3)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    std::vector<BruteNeighbor> neighbors(batch.size());
    for (std::size_t candidate = 0; candidate < batch.size(); ++candidate)
    {
        const double dx = x[query] - x[candidate];
        const double dy = y[query] - y[candidate];
        double distance = dx * dx + dy * dy;
        if (dimensions == 3)
        {
            const double dz = z[query] - z[candidate];
            distance += dz * dz;
        }
        neighbors[candidate] = {static_cast<std::uint32_t>(candidate),
                                distance};
    }
    std::stable_sort(neighbors.begin(), neighbors.end(),
                     [](const BruteNeighbor& left, const BruteNeighbor& right)
                     { return left.distance < right.distance; });
    return neighbors;
}

TEST(SpatialIndex, RadiusCountsMatchStrictPdalDistanceContract)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{500000.0, 4800000.0, 100.0},
                                               {500001.0, 4800000.0, 100.0},
                                               {499999.0, 4800000.0, 100.0},
                                               {500000.0, 4800001.0, 100.0},
                                               {500000.0, 4800000.0, 101.0},
                                               {500002.0, 4800000.0, 100.0},
                                               {500000.0, 4800000.0, 100.0}});

    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, 1.01);
    ASSERT_TRUE(pdg::uniformGridMaySupportExactDevice(batch, config));
    pdg::SpatialIndex index(batch, config);
    index.build();
    EXPECT_TRUE(index.valid());
    EXPECT_EQ(index.buildCount(), 1U);
    EXPECT_GT(index.cellCount(), 0U);

    std::vector<std::uint32_t> actual(batch.size());
    pdg::radiusCounts(index, 1.01, actual.data());
    EXPECT_EQ(actual, bruteCounts(batch, 3, 1.01));

    // nanoflann's RadiusResultSet accepts dist < radius, not <= radius.
    pdg::radiusCounts(index, 1.0, actual.data());
    EXPECT_EQ(actual, bruteCounts(batch, 3, 1.0));
    EXPECT_EQ(actual[0], 2U);
    EXPECT_EQ(actual[6], 2U);
}

TEST(SpatialIndex, RadiusScaledValuesPreservePdalLiteralArithmetic)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.75, 0.0, 0.0}, {2.0, 0.0, 0.0}});
    constexpr double Radius = 1.0;
    const double factor =
        1.0 / ((4.0 / 3.0) * 3.14159 * (Radius * Radius * Radius));
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, Radius);
    pdg::SpatialIndex index(batch, config);
    index.build();

    std::vector<std::uint32_t> counts(batch.size());
    std::vector<double> values(batch.size());
    pdg::radiusCounts(index, Radius, counts.data());
    pdg::radiusScaledValues(index, Radius, factor, values.data());
    for (std::size_t point = 0; point < batch.size(); ++point)
        EXPECT_DOUBLE_EQ(values[point],
                         static_cast<double>(counts[point]) * factor);
}

TEST(SpatialIndex, TwoDimensionalQueriesIgnoreZ)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(
        {{0.0, 0.0, -1000.0}, {0.5, 0.0, 1000.0}, {3.0, 0.0, 0.0}});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 2, 1.0);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> actual(batch.size());
    pdg::radiusCounts(index, 1.0, actual.data());
    EXPECT_EQ(actual, bruteCounts(batch, 2, 1.0));
    EXPECT_EQ(actual, (std::vector<std::uint32_t>{2, 2, 1}));
}

TEST(SpatialIndex, RadiusAnyHonorsDomainsStrictBoundaryAndTwoDimensionalCaps)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, 0.0},
                                               {0.5, 0.0, 2.0},
                                               {10.0, 0.0, 0.0},
                                               {10.5, 0.0, 2.01},
                                               {20.0, 0.0, 0.0},
                                               {21.0, 0.0, 0.0}});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 2, 1.0);
    pdg::SpatialIndex index(batch, config);
    index.build();

    const std::vector<std::uint8_t> source{1, 0, 1, 0, 1, 0};
    const std::vector<std::uint8_t> reference{0, 1, 0, 1, 0, 1};
    std::vector<std::uint8_t> actual(batch.size(), 0xffU);
    pdg::radiusAny(index, 1.0, source.data(), reference.data(), 2.0, -1.0,
                   actual.data());

    // The first candidate is exactly at the allowed height and passes.  The
    // second is above the cap, and the third lies exactly on the strict XY
    // radius boundary.  Non-source rows are always false.
    EXPECT_EQ(actual, (std::vector<std::uint8_t>{1, 0, 0, 0, 0, 0}));
}

TEST(SpatialIndex, InvalidationReusesStorageAndAccountsForRebuild)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, 1.0);
    pdg::SpatialIndex index(batch, config);
    index.build();
    index.invalidate();
    EXPECT_FALSE(index.valid());
    EXPECT_THROW(pdg::radiusCounts(index, 1.0, nullptr), std::logic_error);
    index.build();
    EXPECT_TRUE(index.valid());
    EXPECT_EQ(index.buildCount(), 2U);
}

TEST(SpatialIndex, EmptyInputHasAnEmptyValidCellTable)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, 2.0);
    pdg::SpatialIndex index(batch, config);
    index.build();
    EXPECT_TRUE(index.valid());
    EXPECT_EQ(index.cellCount(), 0U);
    EXPECT_NO_THROW(pdg::radiusCounts(index, 2.0, nullptr));
    EXPECT_NO_THROW(pdg::knnGather(index, 8, nullptr, nullptr, nullptr));
    EXPECT_NO_THROW(pdg::knnMeanDistances(index, 8, nullptr, nullptr));
    EXPECT_NO_THROW(pdg::knnDistanceValues(index, 8, pdg::KnnDistanceMode::Kth,
                                           nullptr, nullptr));
}

TEST(SpatialIndex, ExactEnvelopeRejectsInvalidGeometryAndConfiguration)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}});
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(batch, 1, 1.0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(batch, 3, 0.0)),
                 std::invalid_argument);

    pdg::UniformGridConfig config = pdg::makeUniformGridConfig(batch, 3, 1.0);
    EXPECT_TRUE(pdg::uniformGridMaySupportExactDevice(batch, config));
    batch.hostSpan<double>(X)[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::uniformGridMaySupportExactDevice(batch, config));
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(batch, 3, 1.0)),
                 std::invalid_argument);
}

TEST(SpatialIndex, RadiusCannotExceedConstructionCellEdge)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, 1.0);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> counts(batch.size());
    EXPECT_THROW(pdg::radiusCounts(index, 1.01, counts.data()),
                 std::invalid_argument);
}

TEST(SpatialIndex, KnnGatherMatchesUniqueDistanceOrder)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, 0.0},
                                               {0.31, 0.07, 0.02},
                                               {1.13, 0.21, 0.05},
                                               {2.71, 0.11, 0.41},
                                               {5.33, 0.29, 0.17},
                                               {-1.79, 0.37, 0.23},
                                               {8.17, 0.53, 0.31}});
    constexpr std::uint32_t Neighbors = 4;
    const pdg::UniformGridConfig config =
        pdg::makeKnnGridConfig(batch, 3, Neighbors);
    pdg::SpatialIndex index(batch, config);
    index.build();

    std::vector<std::uint32_t> actualIds(batch.size() * Neighbors);
    std::vector<double> actualDistances(batch.size() * Neighbors);
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnGather(index, Neighbors, actualIds.data(), actualDistances.data(),
                   status.data());
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        EXPECT_EQ(status[query], pdg::KnnExact) << query;
        const std::vector<BruteNeighbor> expected =
            bruteNeighbors(batch, query);
        for (std::size_t neighbor = 0; neighbor < Neighbors; ++neighbor)
        {
            const std::size_t output = query * Neighbors + neighbor;
            EXPECT_EQ(actualIds[output], expected[neighbor].point)
                << query << ':' << neighbor;
            EXPECT_DOUBLE_EQ(actualDistances[output],
                             expected[neighbor].distance)
                << query << ':' << neighbor;
        }
    }

    std::vector<double> means(batch.size());
    pdg::knnMeanDistances(index, Neighbors, means.data(), status.data());
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        EXPECT_EQ(status[query], pdg::KnnExact) << query;
        const std::vector<BruteNeighbor> expected =
            bruteNeighbors(batch, query);
        double expectedMean = 0.0;
        for (std::size_t neighbor = 1; neighbor < Neighbors; ++neighbor)
        {
            const double delta =
                std::sqrt(expected[neighbor].distance) - expectedMean;
            expectedMean += delta / static_cast<double>(neighbor);
        }
        EXPECT_DOUBLE_EQ(means[query], expectedMean) << query;
    }

    for (const pdg::KnnDistanceMode mode :
         {pdg::KnnDistanceMode::Kth, pdg::KnnDistanceMode::Average})
    {
        std::vector<double> values(batch.size());
        pdg::knnDistanceValues(index, Neighbors, mode, values.data(),
                               status.data());
        for (std::size_t query = 0; query < batch.size(); ++query)
        {
            EXPECT_EQ(status[query], pdg::KnnExact) << query;
            const std::vector<BruteNeighbor> expected =
                bruteNeighbors(batch, query);
            double expectedValue = 0.0;
            if (mode == pdg::KnnDistanceMode::Kth)
                expectedValue = std::sqrt(expected[Neighbors - 1U].distance);
            else
            {
                for (std::size_t neighbor = 1U; neighbor < Neighbors;
                     ++neighbor)
                    expectedValue += std::sqrt(expected[neighbor].distance);
                expectedValue /= static_cast<double>(Neighbors - 1U);
            }
            EXPECT_DOUBLE_EQ(values[query], expectedValue) << query;
        }
    }
}

TEST(SpatialIndex, MaskedKnnGatherUsesOnlyTheReferenceDomain)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({
        {0.0, 0.0, 100.0},
        {2.0, 0.0, 10.0},
        {5.0, 0.0, 200.0},
        {8.0, 0.0, 20.0},
        {10.0, 0.0, 300.0},
    });
    constexpr std::uint32_t Neighbors = 1U;
    constexpr std::uint32_t References = 2U;
    const std::array<std::uint8_t, 5> sources{1U, 0U, 1U, 0U, 1U};
    const std::array<std::uint8_t, 5> references{0U, 1U, 0U, 1U, 0U};
    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(batch, 2U, Neighbors),
        pdg::makeMortonBvhConfig(batch, 2U)};

    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex index(batch, config);
        index.build();
        std::array<std::uint32_t, 5> ids{};
        std::array<double, 5> distances{};
        std::array<std::uint8_t, 5> status{};
        pdg::knnGatherMasked(index, Neighbors, References, sources.data(),
                             references.data(), ids.data(), distances.data(),
                             status.data());

        EXPECT_EQ(ids[0], 1U);
        EXPECT_DOUBLE_EQ(distances[0], 4.0);
        EXPECT_EQ(status[0], pdg::KnnExact);
        EXPECT_EQ(ids[2], 1U);
        EXPECT_DOUBLE_EQ(distances[2], 9.0);
        EXPECT_EQ(status[2], pdg::KnnDistanceTie);
        EXPECT_EQ(ids[4], 3U);
        EXPECT_DOUBLE_EQ(distances[4], 4.0);
        EXPECT_EQ(status[4], pdg::KnnExact);
        for (const std::size_t point : {1U, 3U})
        {
            EXPECT_EQ(ids[point],
                      (std::numeric_limits<std::uint32_t>::max)());
            EXPECT_TRUE(std::isinf(distances[point]));
            EXPECT_EQ(status[point], pdg::KnnExact);
        }
    }
}

TEST(SpatialIndex, KnnDistanceValuesValidateNativeEnvelope)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {0.3, 0.7, 1.1}, {1.9, 2.3, 2.9}});
    const pdg::UniformGridConfig config = pdg::makeKnnGridConfig(batch, 3, 3);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::array<double, 3> values{};
    std::array<std::uint8_t, 3> status{};

    EXPECT_THROW(pdg::knnDistanceValues(index, 1, pdg::KnnDistanceMode::Kth,
                                        values.data(), status.data()),
                 std::invalid_argument);
    EXPECT_THROW(pdg::knnDistanceValues(index, 3, pdg::KnnDistanceMode::Kth,
                                        nullptr, status.data()),
                 std::invalid_argument);
    EXPECT_THROW(pdg::knnDistanceValues(index, 3, pdg::KnnDistanceMode::Kth,
                                        values.data(), nullptr),
                 std::invalid_argument);
}

TEST(SpatialIndex, KnnGatherFlagsEveryObservableDistanceTie)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0},
                                               {-1.0, 0.0, 0.0},
                                               {1.0, 0.0, 0.0},
                                               {4.0, 0.0, 0.0}});
    const pdg::UniformGridConfig config = pdg::makeKnnGridConfig(batch, 3, 3);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> ids(batch.size() * 3U);
    std::vector<double> distances(batch.size() * 3U);
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnGather(index, 3, ids.data(), distances.data(), status.data());
    EXPECT_EQ(status[0] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
    EXPECT_EQ(status[1] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
    EXPECT_EQ(status[4] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
}

TEST(SpatialIndex, KnnLofValuesMatchUpstreamThreePassAlgorithm)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, 0.0},
                                               {0.31, 0.07, 0.02},
                                               {1.13, 0.21, 0.05},
                                               {2.71, 0.11, 0.41},
                                               {5.33, 0.29, 0.17},
                                               {-1.79, 0.37, 0.23},
                                               {8.17, 0.53, 0.31}});
    constexpr std::uint32_t Neighbors = 4;
    const pdg::UniformGridConfig config =
        pdg::makeKnnGridConfig(batch, 3, Neighbors);
    pdg::SpatialIndex index(batch, config);
    index.build();

    std::vector<double> kDistances(batch.size());
    std::vector<double> densities(batch.size());
    std::vector<double> factors(batch.size());
    std::vector<std::uint8_t> status(batch.size());
    std::vector<std::uint8_t> neighborStatus(batch.size());
    std::vector<std::uint8_t> closureStatus(batch.size());
    pdg::knnLofValues(index, Neighbors, kDistances.data(), densities.data(),
                      factors.data(), status.data(), neighborStatus.data(),
                      closureStatus.data());

    // Reference: filters.lof's exact three passes over the brute-force
    // adjacency, including the ordered online means.
    std::vector<double> expectedKDistances(batch.size());
    for (std::size_t query = 0; query < batch.size(); ++query)
        expectedKDistances[query] =
            std::sqrt(bruteNeighbors(batch, query)[Neighbors - 1U].distance);
    std::vector<double> expectedDensities(batch.size());
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        const std::vector<BruteNeighbor> expected =
            bruteNeighbors(batch, query);
        double mean = 0.0;
        for (std::size_t item = 0; item < Neighbors; ++item)
        {
            const double distance = std::sqrt(expected[item].distance);
            const double reachability =
                (std::max)(expectedKDistances[expected[item].point], distance);
            mean += (reachability - mean) / static_cast<double>(item + 1U);
        }
        expectedDensities[query] = 1.0 / mean;
    }
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        EXPECT_EQ(status[query], pdg::KnnExact) << query;
        EXPECT_EQ(neighborStatus[query], pdg::KnnExact) << query;
        EXPECT_DOUBLE_EQ(kDistances[query], expectedKDistances[query]) << query;
        EXPECT_DOUBLE_EQ(densities[query], expectedDensities[query]) << query;
        const std::vector<BruteNeighbor> expected =
            bruteNeighbors(batch, query);
        double mean = 0.0;
        for (std::size_t item = 0; item < Neighbors; ++item)
        {
            const double ratio = expectedDensities[expected[item].point] /
                                 expectedDensities[query];
            mean += (ratio - mean) / static_cast<double>(item + 1U);
        }
        EXPECT_DOUBLE_EQ(factors[query], mean) << query;
    }
}

TEST(SpatialIndex, KnnLofValuesReportAmbiguousNeighborRows)
{
    SpatialFixture fixture;
    // Point 3 sits exactly between points 2 and 4, so its row carries a
    // distance tie. Point 4's own row (itself, 3, 2) is strictly ordered, but
    // it reaches point 3's ambiguous row, so only its factor needs repair.
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, 0.0},
                                               {0.0, 0.0, 0.0},
                                               {2.0, 0.0, 0.0},
                                               {5.0, 0.0, 0.0},
                                               {8.0, 0.0, 0.0}});
    const pdg::UniformGridConfig config = pdg::makeKnnGridConfig(batch, 3, 3);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<double> kDistances(batch.size());
    std::vector<double> densities(batch.size());
    std::vector<double> factors(batch.size());
    std::vector<std::uint8_t> status(batch.size());
    std::vector<std::uint8_t> neighborStatus(batch.size());
    std::vector<std::uint8_t> closureStatus(batch.size());
    pdg::knnLofValues(index, 3, kDistances.data(), densities.data(),
                      factors.data(), status.data(), neighborStatus.data(),
                      closureStatus.data());
    EXPECT_EQ(status[4] & pdg::KnnDistanceTie, 0U);
    EXPECT_EQ(neighborStatus[4] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
    // The equidistant midpoint and the duplicate pair are ambiguous outright.
    EXPECT_EQ(status[3] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
    EXPECT_EQ(status[0] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
    EXPECT_EQ(status[1] & pdg::KnnDistanceTie, pdg::KnnDistanceTie);
}

TEST(SpatialIndex, KnnEnvelopeRejectsUnsupportedNeighborCounts)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}});
    EXPECT_THROW(static_cast<void>(pdg::makeKnnGridConfig(batch, 3, 0)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeKnnGridConfig(batch, 3, 65)),
                 std::invalid_argument);
    const pdg::UniformGridConfig config = pdg::makeKnnGridConfig(batch, 3, 2);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> ids(batch.size() * 2U);
    std::vector<double> distances(batch.size() * 2U);
    std::vector<std::uint8_t> status(batch.size());
    EXPECT_THROW(
        pdg::knnGather(index, 65, ids.data(), distances.data(), status.data()),
        std::invalid_argument);
}

TEST(SpatialIndex, KnnGridMatchesBruteForceAcrossNeighborCounts)
{
    SpatialFixture fixture;
    std::vector<std::array<double, 3>> points;
    points.reserve(257);
    for (std::size_t point = 0; point < 257; ++point)
    {
        const double ordinal = static_cast<double>(point);
        points.push_back(
            {500000.0 + ordinal * 0.173 +
                 static_cast<double>((point * 17U) % 13U) * 0.0011,
             4800000.0 + ordinal * ordinal * 0.00037 +
                 static_cast<double>((point * 29U) % 19U) * 0.0013,
             100.0 + static_cast<double>((point * 53U) % 257U) * 0.019 +
                 ordinal * 0.000001});
    }
    pdg::PointBatch batch = fixture.makeBatch(points);
    for (const std::uint32_t neighbors :
         std::array<std::uint32_t, 4>{1U, 8U, 32U, 64U})
    {
        const pdg::UniformGridConfig config =
            pdg::makeKnnGridConfig(batch, 3, neighbors);
        pdg::SpatialIndex index(batch, config);
        index.build();
        std::vector<std::uint32_t> ids(batch.size() * neighbors);
        std::vector<double> distances(batch.size() * neighbors);
        std::vector<std::uint8_t> status(batch.size());
        pdg::knnGather(index, neighbors, ids.data(), distances.data(),
                       status.data());
        for (std::size_t query = 0; query < batch.size(); ++query)
        {
            EXPECT_EQ(status[query] & pdg::KnnSearchIncomplete, 0U)
                << neighbors << ':' << query;
            const std::vector<BruteNeighbor> expected =
                bruteNeighbors(batch, query);
            for (std::size_t neighbor = 0; neighbor < neighbors; ++neighbor)
            {
                const std::size_t output = query * neighbors + neighbor;
                EXPECT_EQ(ids[output], expected[neighbor].point)
                    << neighbors << ':' << query << ':' << neighbor;
                EXPECT_DOUBLE_EQ(distances[output], expected[neighbor].distance)
                    << neighbors << ':' << query << ':' << neighbor;
            }
        }
    }
}

TEST(SpatialIndex, KnnTwoDimensionalQueriesIgnoreZ)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch({{0.0, 0.0, -1000.0},
                                               {0.17, 0.03, 1000.0},
                                               {0.61, 0.09, -3000.0},
                                               {1.43, 0.22, 7000.0}});
    constexpr std::uint32_t Neighbors = 3U;
    const pdg::UniformGridConfig config =
        pdg::makeKnnGridConfig(batch, 2, Neighbors);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> ids(batch.size() * Neighbors);
    std::vector<double> distances(batch.size() * Neighbors);
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnGather(index, Neighbors, ids.data(), distances.data(),
                   status.data());
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        EXPECT_EQ(status[query], pdg::KnnExact);
        const std::vector<BruteNeighbor> expected =
            bruteNeighbors(batch, query, 2);
        for (std::size_t neighbor = 0; neighbor < Neighbors; ++neighbor)
        {
            const std::size_t output = query * Neighbors + neighbor;
            EXPECT_EQ(ids[output], expected[neighbor].point);
            EXPECT_DOUBLE_EQ(distances[output], expected[neighbor].distance);
        }
    }
}

TEST(SpatialIndex, KnnFixedRowsUseSentinelsWhenKExceedsPointCount)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {1.0, 2.0, 3.0}});
    constexpr std::uint32_t Neighbors = 4U;
    const pdg::UniformGridConfig config =
        pdg::makeKnnGridConfig(batch, 3, Neighbors);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::vector<std::uint32_t> ids(batch.size() * Neighbors);
    std::vector<double> distances(batch.size() * Neighbors);
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnGather(index, Neighbors, ids.data(), distances.data(),
                   status.data());
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        EXPECT_EQ(status[query], pdg::KnnExact);
        EXPECT_EQ(ids[query * Neighbors + 2U],
                  (std::numeric_limits<std::uint32_t>::max)());
        EXPECT_TRUE(std::isinf(distances[query * Neighbors + 2U]));
        EXPECT_EQ(ids[query * Neighbors + 3U],
                  (std::numeric_limits<std::uint32_t>::max)());
        EXPECT_TRUE(std::isinf(distances[query * Neighbors + 3U]));
    }
}

TEST(SpatialIndex, KnnBoundedSearchReportsIncompleteInsteadOfGuessing)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {5001.0, 0.0, 0.0}});
    const pdg::UniformGridConfig config =
        pdg::makeUniformGridConfig(batch, 3, 1.0);
    pdg::SpatialIndex index(batch, config);
    index.build();
    std::array<std::uint32_t, 4> ids{};
    std::array<double, 4> distances{};
    std::array<std::uint8_t, 2> status{};
    pdg::knnGather(index, 2, ids.data(), distances.data(), status.data());
    EXPECT_EQ(status[0] & pdg::KnnSearchIncomplete, pdg::KnnSearchIncomplete);
    EXPECT_EQ(status[1] & pdg::KnnSearchIncomplete, pdg::KnnSearchIncomplete);
}

TEST(SpatialIndex, MortonBvhCompletesKnnAcrossSparseGridGaps)
{
    SpatialFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {5001.0, 0.0, 0.0}});
    pdg::UniformGridConfig config = pdg::makeUniformGridConfig(batch, 3, 1.0);
    config.backend = pdg::SpatialIndexBackend::MortonBvh;
    pdg::SpatialIndex index(batch, config);
    index.build();

    std::array<std::uint32_t, 4> ids{};
    std::array<double, 4> distances{};
    std::array<std::uint8_t, 2> status{};
    pdg::knnGather(index, 2, ids.data(), distances.data(), status.data());

    EXPECT_EQ(status[0], pdg::KnnExact);
    EXPECT_EQ(status[1], pdg::KnnExact);
    EXPECT_EQ(ids, (std::array<std::uint32_t, 4>{0U, 1U, 1U, 0U}));
    EXPECT_EQ(distances[0], 0.0);
    EXPECT_EQ(distances[1], 5001.0 * 5001.0);
    EXPECT_EQ(distances[2], 0.0);
    EXPECT_EQ(distances[3], 5001.0 * 5001.0);

    std::array<std::uint32_t, 2> counts{};
    EXPECT_NO_THROW(pdg::radiusCounts(index, 5001.5, counts.data()));
    EXPECT_EQ(counts, (std::array<std::uint32_t, 2>{2U, 2U}));
}

TEST(SpatialIndex,
     MortonBvhMatchesMixedDensityBruteForceInTwoAndThreeDimensions)
{
    SpatialFixture fixture;
    std::vector<std::array<double, 3>> points;
    points.reserve(193);
    for (std::size_t point = 0; point < 181; ++point)
    {
        const double ordinal = static_cast<double>(point);
        points.push_back(
            {500000.0 + ordinal * 0.000013,
             4800000.0 + static_cast<double>((point * 17U) % 31U) * 0.000019,
             100.0 + static_cast<double>((point * 29U) % 47U) * 0.000023});
    }
    for (std::size_t point = 0; point < 12; ++point)
    {
        const double ordinal = static_cast<double>(point + 1U);
        points.push_back({500000.0 + ordinal * 100000.0,
                          4800000.0 - ordinal * 70000.0,
                          100.0 + ordinal * 30000.0});
    }
    pdg::PointBatch batch = fixture.makeBatch(points);

    for (const std::uint8_t dimensions : {std::uint8_t{2}, std::uint8_t{3}})
    {
        // A tiny adversarial fixture proves BVH exactness but is below the
        // measured device crossover, so automatic selection stays on grid.
        EXPECT_EQ(pdg::makeAdaptiveKnnConfig(batch, dimensions, 64).backend,
                  pdg::SpatialIndexBackend::UniformGrid);
        const pdg::UniformGridConfig config =
            pdg::makeMortonBvhConfig(batch, dimensions);
        EXPECT_EQ(config.backend, pdg::SpatialIndexBackend::MortonBvh);
        pdg::SpatialIndex index(batch, config);
        index.build();
        for (const std::uint32_t neighbors :
             std::array<std::uint32_t, 3>{1U, 8U, 64U})
        {
            std::vector<std::uint32_t> ids(batch.size() * neighbors);
            std::vector<double> distances(batch.size() * neighbors);
            std::vector<std::uint8_t> status(batch.size());
            pdg::knnGather(index, neighbors, ids.data(), distances.data(),
                           status.data());
            for (std::size_t query = 0; query < batch.size(); ++query)
            {
                EXPECT_EQ(status[query] & pdg::KnnSearchIncomplete, 0U)
                    << static_cast<unsigned int>(dimensions) << ':' << neighbors
                    << ':' << query;
                const std::vector<BruteNeighbor> expected =
                    bruteNeighbors(batch, query, dimensions);
                for (std::size_t neighbor = 0; neighbor < neighbors; ++neighbor)
                {
                    const std::size_t output = query * neighbors + neighbor;
                    EXPECT_EQ(ids[output], expected[neighbor].point)
                        << static_cast<unsigned int>(dimensions) << ':'
                        << neighbors << ':' << query << ':' << neighbor;
                    EXPECT_DOUBLE_EQ(distances[output],
                                     expected[neighbor].distance)
                        << static_cast<unsigned int>(dimensions) << ':'
                        << neighbors << ':' << query << ':' << neighbor;
                }
            }
        }
    }
}

TEST(SpatialIndex, AdaptiveKnnKeepsUniformGridForOrdinaryDensity)
{
    SpatialFixture fixture;
    std::vector<std::array<double, 3>> points;
    points.reserve(512);
    for (std::size_t z = 0; z < 8; ++z)
        for (std::size_t y = 0; y < 8; ++y)
            for (std::size_t x = 0; x < 8; ++x)
                points.push_back({static_cast<double>(x),
                                  static_cast<double>(y),
                                  static_cast<double>(z)});
    pdg::PointBatch batch = fixture.makeBatch(points);
    EXPECT_EQ(pdg::makeAdaptiveKnnConfig(batch, 3, 8).backend,
              pdg::SpatialIndexBackend::UniformGrid);
}

TEST(SpatialIndex, AdaptiveKnnUsesBvhOnlyForBroadExpensiveClustering)
{
    SpatialFixture fixture;
    constexpr std::size_t Count = 8192U;
    std::vector<std::array<double, 3>> points;
    points.reserve(Count);
    for (std::size_t point = 0U; point < Count; ++point)
    {
        const double ordinal = static_cast<double>(point);
        if (point + 12U < Count)
            points.push_back(
                {500000.0 + ordinal * 0.0000001,
                 4800000.0 +
                     static_cast<double>((point * 17U) % Count) * 0.00000011,
                 100.0 +
                     static_cast<double>((point * 29U) % Count) * 0.00000013});
        else
        {
            const double outlier = static_cast<double>(point + 13U - Count);
            points.push_back({500000.0 + outlier * 1000.0,
                              4800000.0 - outlier * 700.0,
                              100.0 + outlier * 300.0});
        }
    }
    pdg::PointBatch batch = fixture.makeBatch(points);
    EXPECT_EQ(pdg::makeAdaptiveKnnConfig(batch, 3, 16).backend,
              pdg::SpatialIndexBackend::MortonBvh);
}

TEST(SpatialIndex, KnnConfigurationBuildersReturnExactHostEnvelopes)
{
    SpatialFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(
        {{-0.0, 4800000.0, -1000.0},
         {0.0, 4800000.0000001, 1000.0},
         {500000.0, -4800000.0, 0.0},
         {900000.0, 8100000.0, 1.0}});

    for (const std::uint8_t dimensions : {std::uint8_t{2}, std::uint8_t{3}})
    {
        for (const pdg::UniformGridConfig& config :
             {pdg::makeKnnGridConfig(batch, dimensions, 3),
              pdg::makeMortonBvhConfig(batch, dimensions),
              pdg::makeAdaptiveKnnConfig(batch, dimensions, 3)})
            EXPECT_TRUE(
                pdg::uniformGridMaySupportExactDevice(batch, config));
    }

    batch.hostSpan<double>(Y)[2] =
        (std::numeric_limits<double>::quiet_NaN)();
    EXPECT_THROW(static_cast<void>(pdg::makeKnnGridConfig(batch, 3, 3)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeMortonBvhConfig(batch, 3)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeAdaptiveKnnConfig(batch, 3, 3)),
                 std::invalid_argument);
}

TEST(SpatialIndex, KnnCoordinateSummaryMatchesEveryPointBatchBuilder)
{
    SpatialFixture fixture;
    std::vector<std::array<double, 3>> points;
    constexpr std::size_t Count = 20000U;
    points.reserve(Count);
    for (std::size_t point = 0U; point < Count; ++point)
    {
        if (point < Count - 12U)
            points.push_back(
                {500000.0 +
                     static_cast<double>((point * 11U) % Count) * 0.00000009,
                 4800000.0 +
                     static_cast<double>((point * 17U) % Count) * 0.00000011,
                 100.0 +
                     static_cast<double>((point * 29U) % Count) * 0.00000013});
        else
        {
            const double outlier = static_cast<double>(point + 13U - Count);
            points.push_back({500000.0 + outlier * 1000.0,
                              4800000.0 - outlier * 700.0,
                              100.0 + outlier * 300.0});
        }
    }
    pdg::PointBatch batch = fixture.makeBatch(points);

    for (const std::uint8_t dimensions : {std::uint8_t{2}, std::uint8_t{3}})
    {
        const pdg::KnnConfigSummary summary =
            summarizeBatch(batch, dimensions);
        expectSameConfig(pdg::makeUniformGridConfig(summary, 1.25),
                         pdg::makeUniformGridConfig(batch, dimensions, 1.25));
        expectSameConfig(pdg::makeKnnGridConfig(summary, 16U),
                         pdg::makeKnnGridConfig(batch, dimensions, 16U));
        expectSameConfig(pdg::makeMortonBvhConfig(summary),
                         pdg::makeMortonBvhConfig(batch, dimensions));
        expectSameConfig(pdg::makeAdaptiveKnnConfig(summary, 16U),
                         pdg::makeAdaptiveKnnConfig(batch, dimensions, 16U));
    }

    pdg::KnnConfigSummary invalid = summarizeBatch(batch, 3U);
    invalid.probes.pop_back();
    EXPECT_THROW(static_cast<void>(pdg::makeAdaptiveKnnConfig(invalid, 16U)),
                 std::invalid_argument);

    invalid = summarizeBatch(batch, 3U);
    invalid.minimum[0] =
        (std::numeric_limits<double>::quiet_NaN)();
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(invalid, 1.25)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeKnnGridConfig(invalid, 16U)),
                 std::invalid_argument);
    invalid = summarizeBatch(batch, 3U);
    invalid.maximum[1] = invalid.minimum[1] - 1.0;
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(invalid, 1.25)),
                 std::invalid_argument);
    invalid = summarizeBatch(batch, 3U);
    invalid.probes[0][2] =
        (std::numeric_limits<double>::infinity)();
    EXPECT_THROW(static_cast<void>(pdg::makeMortonBvhConfig(invalid)),
                 std::invalid_argument);
    invalid = summarizeBatch(batch, 3U);
    EXPECT_THROW(static_cast<void>(pdg::makeUniformGridConfig(invalid, 0.0)),
                 std::invalid_argument);
    invalid.pointCount = 0U;
    expectSameConfig(pdg::makeUniformGridConfig(invalid, 1.25),
                     pdg::makeUniformGridConfig(
                         fixture.makeBatch({}), 3U, 1.25));
    EXPECT_THROW(static_cast<void>(pdg::knnConfigProbePoint(0U, 0U)),
                 std::out_of_range);
}
} // unnamed namespace
