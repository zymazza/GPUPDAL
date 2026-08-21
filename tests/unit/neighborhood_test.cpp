#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

struct NeighborhoodFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::CoordinateEncoding encoding{{0.001, 0.001, 0.001},
                                     {500000.0, 4800000.0, 100.0}};

    pdg::PointBatch makeBatch(const std::vector<std::array<double, 3>>& points)
    {
        pdg::PointBatch batch(points.size(), encoding, dimensions, memory);
        for (pdg::DimensionId dimension : {X, Y, Z})
            batch.materialize(dimension, pdg::DimensionType::Double);
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

Eigen::Matrix3d
pinnedCovariance(const std::vector<std::array<double, 3>>& points,
                 const std::vector<std::uint32_t>& ids, std::size_t row,
                 std::size_t neighbors)
{
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    std::size_t count = 0;
    for (std::size_t item = 0; item < neighbors; ++item)
    {
        ++count;
        const auto& point = points[ids[row + item]];
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const double delta = point[axis] - centroid[axis];
            centroid[axis] += delta / static_cast<double>(count);
        }
    }
    Eigen::MatrixXd demeaned(3, static_cast<Eigen::Index>(neighbors));
    for (std::size_t item = 0; item < neighbors; ++item)
    {
        const auto& point = points[ids[row + item]];
        for (std::size_t axis = 0; axis < 3; ++axis)
            demeaned(static_cast<Eigen::Index>(axis),
                     static_cast<Eigen::Index>(item)) =
                static_cast<float>(point[axis] - centroid[axis]);
    }
    return demeaned * demeaned.transpose() /
           static_cast<double>(neighbors - 1U);
}

void expectCovariance(const pdg::Covariance3d& actual,
                      const Eigen::Matrix3d& expected, std::size_t query)
{
    EXPECT_EQ(actual.xx, expected(0, 0)) << query << ":xx";
    EXPECT_EQ(actual.xy, expected(0, 1)) << query << ":xy";
    EXPECT_EQ(actual.xz, expected(0, 2)) << query << ":xz";
    EXPECT_EQ(actual.yy, expected(1, 1)) << query << ":yy";
    EXPECT_EQ(actual.yz, expected(1, 2)) << query << ":yz";
    EXPECT_EQ(actual.zz, expected(2, 2)) << query << ":zz";
}

TEST(NeighborhoodCovariance, MatchesPinnedPdalForEveryKnnRowBitForBit)
{
    std::vector<std::array<double, 3>> points;
    for (std::size_t point = 0; point < 37; ++point)
    {
        points.push_back(
            {500000.0 + static_cast<double>((point * 7919U) % 1009U) * 0.013,
             4800000.0 + static_cast<double>((point * 1543U) % 701U) * 0.017,
             100.0 + static_cast<double>((point * 313U) % 101U) * 0.019});
    }
    constexpr std::uint32_t Neighbors = 9;
    NeighborhoodFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(points);
    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(batch, 3, Neighbors),
        pdg::makeMortonBvhConfig(batch, 3)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex index(batch, config);
        index.build();

        std::vector<std::uint32_t> ids(points.size() * Neighbors);
        std::vector<double> distances(points.size() * Neighbors);
        std::vector<std::uint8_t> gatherStatus(points.size());
        pdg::knnGather(index, Neighbors, ids.data(), distances.data(),
                       gatherStatus.data());
        std::vector<pdg::Covariance3d> actual(points.size());
        std::vector<std::uint8_t> status(points.size());
        pdg::knnCovariances(index, Neighbors, actual.data(), status.data());
        EXPECT_EQ(status, gatherStatus);

        for (std::size_t query = 0; query < points.size(); ++query)
        {
            ASSERT_EQ(status[query] & pdg::KnnSearchIncomplete, 0U) << query;
            const std::size_t row = query * Neighbors;
            expectCovariance(actual[query],
                             pinnedCovariance(points, ids, row, Neighbors),
                             query);
        }
    }
}

TEST(NeighborhoodCovariance, PreservesTieStatusForIdSensitiveConsumers)
{
    NeighborhoodFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}});
    constexpr std::uint32_t Neighbors = 3;
    pdg::SpatialIndex index(batch, pdg::makeKnnGridConfig(batch, 3, Neighbors));
    index.build();
    std::vector<pdg::Covariance3d> covariances(batch.size());
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnCovariances(index, Neighbors, covariances.data(), status.data());
    EXPECT_NE(status[0] & pdg::KnnDistanceTie, 0U);
}

TEST(NeighborhoodEigenSystem, MatchesPinnedSolverBitForBit)
{
    std::vector<std::array<double, 3>> points;
    for (std::size_t point = 0; point < 31; ++point)
    {
        points.push_back(
            {400000.0 + static_cast<double>((point * 3571U) % 997U) * 0.021,
             4700000.0 + static_cast<double>((point * 1741U) % 683U) * 0.023,
             60.0 + static_cast<double>((point * 281U) % 109U) * 0.029});
    }
    constexpr std::uint32_t Neighbors = 11;
    NeighborhoodFixture fixture;
    pdg::PointBatch batch = fixture.makeBatch(points);
    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(batch, 3, Neighbors),
        pdg::makeMortonBvhConfig(batch, 3)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex index(batch, config);
        index.build();
        std::vector<std::uint32_t> ids(points.size() * Neighbors);
        std::vector<double> distances(points.size() * Neighbors);
        std::vector<std::uint8_t> gatherStatus(points.size());
        pdg::knnGather(index, Neighbors, ids.data(), distances.data(),
                       gatherStatus.data());
        std::vector<pdg::EigenSystem3d> actual(points.size());
        std::vector<std::uint8_t> status(points.size());
        pdg::knnEigenSystems(index, Neighbors, actual.data(), status.data());
        for (std::size_t query = 0; query < points.size(); ++query)
        {
            ASSERT_EQ(status[query] &
                          (pdg::KnnSearchIncomplete | pdg::KnnCovarianceZero |
                           pdg::KnnEigenFailure),
                      0U)
                << query;
            const Eigen::Matrix3d covariance =
                pinnedCovariance(points, ids, query * Neighbors, Neighbors);
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> expected(
                covariance);
            ASSERT_EQ(expected.info(), Eigen::Success);
            for (std::size_t eigen = 0; eigen < 3; ++eigen)
            {
                EXPECT_EQ(
                    actual[query].values[eigen],
                    expected.eigenvalues()[static_cast<Eigen::Index>(eigen)])
                    << query << ':' << eigen;
                for (std::size_t axis = 0; axis < 3; ++axis)
                    EXPECT_EQ(actual[query].vectors[axis * 3U + eigen],
                              expected.eigenvectors()(
                                  static_cast<Eigen::Index>(axis),
                                  static_cast<Eigen::Index>(eigen)))
                        << query << ':' << axis << ':' << eigen;
            }
        }
    }
}

TEST(NeighborhoodEigenSystem, ReportsPdalApproximateZeroCovariance)
{
    NeighborhoodFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}});
    pdg::SpatialIndex index(batch, pdg::makeKnnGridConfig(batch, 3, 3));
    index.build();
    std::vector<pdg::EigenSystem3d> systems(batch.size());
    std::vector<std::uint8_t> status(batch.size());
    pdg::knnEigenSystems(index, 3, systems.data(), status.data());
    for (std::uint8_t value : status)
        EXPECT_NE(value & pdg::KnnCovarianceZero, 0U);
}

TEST(NeighborhoodCovariance, RejectsUnsafeShapesBeforeWriting)
{
    NeighborhoodFixture fixture;
    pdg::PointBatch batch =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});
    pdg::SpatialIndex index(batch, pdg::makeKnnGridConfig(batch, 3, 3));
    index.build();
    std::vector<pdg::Covariance3d> covariances(batch.size());
    std::vector<std::uint8_t> status(batch.size());
    EXPECT_THROW(
        pdg::knnCovariances(index, 2, covariances.data(), status.data()),
        std::invalid_argument);
    EXPECT_THROW(
        pdg::knnCovariances(index, 65, covariances.data(), status.data()),
        std::invalid_argument);
    EXPECT_THROW(pdg::knnCovariances(index, 3, nullptr, status.data()),
                 std::invalid_argument);
    EXPECT_THROW(pdg::knnCovariances(index, 3, covariances.data(), nullptr),
                 std::invalid_argument);

    pdg::PointBatch tooSmall =
        fixture.makeBatch({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
    pdg::SpatialIndex smallIndex(tooSmall,
                                 pdg::makeKnnGridConfig(tooSmall, 3, 3));
    smallIndex.build();
    EXPECT_THROW(
        pdg::knnCovariances(smallIndex, 3, covariances.data(), status.data()),
        std::invalid_argument);
}
} // unnamed namespace
