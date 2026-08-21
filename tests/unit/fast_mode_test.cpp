// D0271: under `gpupdal --fast` (internal marker PDG_INTERNAL_FAST_MODE=1) the
// spatial index does not report kNN distance ties, so the device/host-index
// tie choice stands and no exact tie repair runs. Everything but the tie bit
// — neighbor ids, distances, incomplete flags — is unchanged.

#include <pdg/FastMode.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
    }
    ~ScopedEnvironment()
    {
        if (m_prior)
            ::setenv(m_name.c_str(), m_prior->c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
    }

private:
    std::string m_name;
    std::optional<std::string> m_prior;
};

struct KnnRun
{
    std::vector<std::uint32_t> ids;
    std::vector<double> distances;
    std::vector<std::uint8_t> status;
};

KnnRun hostKnn(bool bvh)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::CoordinateEncoding coordinates{{0.01, 0.01, 0.01},
                                        {500000.0, 4800000.0, 100.0}};
    const std::vector<std::array<double, 3>> points{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
        {0.0, -1.0, 0.0}, {9.0, 9.0, 9.0}};
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
    const pdg::UniformGridConfig config =
        bvh ? pdg::makeMortonBvhConfig(batch, 3)
            : pdg::makeKnnGridConfig(batch, 3, 3);
    pdg::SpatialIndex index(batch, config);
    index.build();
    KnnRun run;
    run.ids.resize(batch.size() * 3U);
    run.distances.resize(batch.size() * 3U);
    run.status.resize(batch.size());
    pdg::knnGather(index, 3, run.ids.data(), run.distances.data(),
                   run.status.data());
    return run;
}
} // unnamed namespace

TEST(FastMode, MarkerControlsTheStatusMask)
{
    {
        ScopedEnvironment off("PDG_INTERNAL_FAST_MODE", nullptr);
        EXPECT_FALSE(pdg::fastModeEnabled());
        EXPECT_FALSE(pdg::relaxedTieOrder());
        EXPECT_EQ(pdg::knnStatusMask(), 0xFFU);
    }
    {
        ScopedEnvironment other("PDG_INTERNAL_FAST_MODE", "yes");
        EXPECT_FALSE(pdg::fastModeEnabled());
        EXPECT_EQ(pdg::knnStatusMask(), 0xFFU);
    }
    {
        ScopedEnvironment on("PDG_INTERNAL_FAST_MODE", "1");
        EXPECT_TRUE(pdg::fastModeEnabled());
        EXPECT_TRUE(pdg::relaxedTieOrder());
        EXPECT_EQ(pdg::knnStatusMask(),
                  static_cast<std::uint8_t>(0xFFU & ~pdg::KnnDistanceTie));
        EXPECT_EQ(pdg::knnStatusMask() & pdg::KnnSearchIncomplete,
                  pdg::KnnSearchIncomplete);
    }
}

TEST(FastMode, HostIndexReportsTiesOnlyUnderTheDefaultContract)
{
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        KnnRun exact;
        {
            ScopedEnvironment off("PDG_INTERNAL_FAST_MODE", nullptr);
            exact = hostKnn(bvh);
        }
        std::size_t ties = 0;
        for (const std::uint8_t status : exact.status)
            ties += (status & pdg::KnnDistanceTie) != 0U;
        ASSERT_GT(ties, 0U) << "fixture must expose distance ties";

        KnnRun relaxed;
        {
            ScopedEnvironment on("PDG_INTERNAL_FAST_MODE", "1");
            relaxed = hostKnn(bvh);
        }
        for (const std::uint8_t status : relaxed.status)
            EXPECT_EQ(status & pdg::KnnDistanceTie, 0U);
        // Only the tie bit changes: ids, distances, every other bit equal.
        EXPECT_EQ(relaxed.ids, exact.ids);
        EXPECT_EQ(relaxed.distances, exact.distances);
        for (std::size_t point = 0; point < exact.status.size(); ++point)
            EXPECT_EQ(relaxed.status[point],
                      exact.status[point] & ~pdg::KnnDistanceTie);
    }
}
