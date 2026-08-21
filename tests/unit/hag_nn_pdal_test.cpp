#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>

#include "src/pdal/PdgNeighborhood.hpp"
#include "src/pdal/PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
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

struct HagPoint
{
    double x;
    double y;
    double z;
    std::uint8_t classification;
};

struct NeighborhoodColumns
{
    std::vector<double> hag;
    std::vector<double> nnDistance;
};

std::vector<double>
runHag(const std::string& stageName, const std::vector<HagPoint>& points,
       std::uint64_t count = 1U, std::uint8_t groundClass = 2U,
       std::optional<double> maximumDistance = std::nullopt,
       std::optional<bool> allowExtrapolation = std::nullopt)
{
    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::X, Id::Y, Id::Z, Id::Classification, Id::HeightAboveGround});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const HagPoint& value = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, value.x);
        view->setField(Id::Y, point, value.y);
        view->setField(Id::Z, point, value.z);
        view->setField(Id::Classification, point, value.classification);
    }
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("HAG NN test stage is unavailable");
    pdal::Options options;
    options.add("count", count);
    if (groundClass != 2U)
        options.add("class", static_cast<unsigned int>(groundClass));
    if (maximumDistance)
        options.add("max_distance", *maximumDistance);
    if (allowExtrapolation)
        options.add("allow_extrapolation", *allowExtrapolation);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));

    std::vector<double> result;
    result.reserve(points.size());
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        result.push_back(
            view->getFieldAs<double>(Id::HeightAboveGround, point));
    return result;
}

NeighborhoodColumns
runHostNeighborhoodChain(const std::vector<HagPoint>& points, bool hagFirst)
{
    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::HeightAboveGround, Id::NNDistance});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const HagPoint& value = points.at(static_cast<std::size_t>(point));
        view->setField(Id::X, point, value.x);
        view->setField(Id::Y, point, value.y);
        view->setField(Id::Z, point, value.z);
        view->setField(Id::Classification, point, value.classification);
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* hag = factory.createStage("filters.hag_nn");
    pdal::Stage* nn = factory.createStage("filters.nndistance");
    if (!hag || !nn)
        throw std::runtime_error(
            "neighborhood chain test stage is unavailable");
    pdal::Options hagOptions;
    hagOptions.add("count", 1U);
    hag->setOptions(hagOptions);
    pdal::Options nnOptions;
    nnOptions.add("k", 3U);
    nn->setOptions(nnOptions);
    pdal::Stage* first = hagFirst ? hag : nn;
    pdal::Stage* second = hagFirst ? nn : hag;
    first->setInput(reader);
    second->setInput(*first);
    second->prepare(table);
    static_cast<void>(second->execute(table));

    NeighborhoodColumns result;
    result.hag.reserve(points.size());
    result.nnDistance.reserve(points.size());
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        result.hag.push_back(
            view->getFieldAs<double>(Id::HeightAboveGround, point));
        result.nnDistance.push_back(
            view->getFieldAs<double>(Id::NNDistance, point));
    }
    return result;
}

bool cudaDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

std::size_t boundaryId(const pdg::Plan& plan, pdg::ResidencyBoundaryKind kind)
{
    const auto position =
        std::find_if(plan.summary().residencyBoundaries.begin(),
                     plan.summary().residencyBoundaries.end(),
                     [&](const pdg::ResidencyBoundary& boundary)
                     { return boundary.kind == kind; });
    EXPECT_NE(position, plan.summary().residencyBoundaries.end());
    return static_cast<std::size_t>(
        std::distance(plan.summary().residencyBoundaries.begin(), position));
}
} // unnamed namespace

TEST(PdgHagNnFilter, CountTwoCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.0, 0.0, 100.0, 1U},
                                       {4.0, 0.0, 14.0, 2U},
                                       {8.0, 0.0, 200.0, 1U},
                                       {10.0, 0.0, 20.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 2U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 2U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountTwoCudaPreservesDistanceAndExtrapolationOptions)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {4.0, 0.0, 18.0, 2U},
                                       {1.0, 0.0, 100.0, 1U},
                                       {9.0, 0.0, 200.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), points, 2U, 2U, 2.0, false),
        runHag("filters.hag_nn", points, 2U, 2U, 2.0, false));
}

TEST(PdgHagNnFilter, CountTwoPreservesStrictCutoffAndInclusiveBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> cutoff{
        {0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 30.0, 2U}, {2.0, 0.0, 100.0, 1U}};
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), cutoff, 2U, 2U, 2.0, true),
        runHag("filters.hag_nn", cutoff, 2U, 2U, 2.0, true));

    const std::vector<HagPoint> boundsEdge{
        {0.0, 0.0, 10.0, 2U}, {5.0, 4.0, 30.0, 2U}, {0.0, 2.0, 100.0, 1U}};
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 2U, 2U,
               std::nullopt, false),
        runHag("filters.hag_nn", boundsEdge, 2U, 2U, std::nullopt, false));
}

TEST(PdgHagNnFilter, CountTwoInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}, {100.0, 100.0, 21.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback(
        "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 2U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 2U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountTwoNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {1.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                      "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 2U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 2U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountTwoDistanceTieFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {2.0, 0.0, 20.0, 2U}, {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 2U),
                  runHag("filters.hag_nn", points, 2U));
    }
}

TEST(PdgHagNnFilter, CountTwoBoundaryTieFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {-1.0, 0.0, 20.0, 2U},
                                       {3.0, 0.0, 30.0, 2U},
                                       {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 2U),
                  runHag("filters.hag_nn", points, 2U));
    }
}

TEST(PdgHagNnFilter, CountTwoIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 20.0, 1U}, {5001.0, 0.0, 10.0, 2U}, {5002.0, 0.0, 11.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 2U),
              runHag("filters.hag_nn", points, 2U));
}

TEST(PdgHagNnFilter, CountThreeCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},  {1.0, 0.0, 100.0, 1U}, {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, 200.0, 1U}, {10.0, 0.0, 20.0, 2U}, {12.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 3U),
              runHag("filters.hag_nn", points, 3U));
}

TEST(PdgHagNnFilter, CountThreeCudaPreservesDistanceAndExtrapolationOptions)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},  {4.0, 0.0, 18.0, 2U},  {8.0, 0.0, 12.0, 2U},
        {1.0, 0.0, 100.0, 1U}, {6.5, 0.0, 180.0, 1U}, {9.0, 0.0, 200.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), points, 3U, 2U, 2.0, false),
        runHag("filters.hag_nn", points, 3U, 2U, 2.0, false));
}

TEST(PdgHagNnFilter, CountThreePreservesStrictCutoffAndInclusiveBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> maxDistance{{0.0, 0.0, 10.0, 2U},
                                            {5.0, 0.0, 30.0, 2U},
                                            {0.0, 1.0, 18.0, 2U},
                                            {0.0, 3.0, 100.0, 1U}};
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 3U, 2U,
                     2.0, true),
              runHag("filters.hag_nn", maxDistance, 3U, 2U, 2.0, true));
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 3U, 2U,
                     -2.0, true),
              runHag("filters.hag_nn", maxDistance, 3U, 2U, -2.0, true));

    const std::vector<HagPoint> boundsEdge{{0.0, 0.0, 10.0, 2U},
                                           {5.0, 4.0, 30.0, 2U},
                                           {0.0, 2.0, 100.0, 2U},
                                           {1.0, 0.0, 10.0, 1U}};
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 3U, 2U,
               std::nullopt, false),
        runHag("filters.hag_nn", boundsEdge, 3U, 2U, std::nullopt, false));
}

TEST(PdgHagNnFilter, CountThreeInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> oneGround{{0.0, 0.0, 10.0, 2U},
                                          {1.0, 0.0, 17.0, 1U},
                                          {100.0, 100.0, 21.0, 1U},
                                          {0.0, 0.0, 100.0, 1U}};
    const std::vector<HagPoint> twoGround{{0.0, 0.0, 10.0, 2U},
                                          {4.0, 0.0, 14.0, 2U},
                                          {1.0, 0.0, 17.0, 1U},
                                          {0.0, 0.0, 20.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    for (const std::vector<HagPoint>& points : {oneGround, twoGround})
    {
        ScopedEnvironment requireFallback(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 3U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 3U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountThreeNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 2U},
        {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                      "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 3U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 3U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountThreeDistanceTieFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.0, 0.0, 20.0, 2U},
                                       {-1.0, 0.0, 18.0, 2U},
                                       {0.0, 1.0, 22.0, 2U},
                                       {0.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 3U),
                  runHag("filters.hag_nn", points, 3U));
    }
}

TEST(PdgHagNnFilter, CountThreeBoundaryTieFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {-1.0, 0.0, 20.0, 2U},
                                       {3.0, 0.0, 30.0, 2U},
                                       {1.0, 0.0, 25.0, 2U},
                                       {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 3U),
                  runHag("filters.hag_nn", points, 3U));
    }
}

TEST(PdgHagNnFilter, CountThreeDistanceOverflowUsesPinnedTieRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{1.4e154, 0.0, 10.0, 2U},
                                       {1.5e154, 0.0, 14.0, 2U},
                                       {1.6e154, 0.0, 18.0, 2U},
                                       {1.7e154, 0.0, 22.0, 2U},
                                       {0.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 3U);
    const std::vector<double> expected =
        runHag("filters.hag_nn", points, 3U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountThreeArithmeticCasesMatchPinnedUpstream)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
    };
    const std::array fixtures{
        Fixture{"ordered three-term accumulation",
                {{-0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 1.0, 2U},
                 {0.0, 5.0, 30.0, 2U},
                 {1.0, 1.5, 20.0, 1U}}},
        Fixture{"large finite coordinates and signed zero",
                {{8.0e153, 8.0e153, 10.0, 2U},
                 {1.0e154, 8.0e153, 14.0, 2U},
                 {8.0e153, 1.0e154, 18.0, 2U},
                 {8.4e153, 8.6e153, 20.0, 1U}}},
        Fixture{"finite underflow and overflow envelope",
                {{1.0e-160, 0.0, 10.0, 2U},
                 {2.0e-160, 0.0, 14.0, 2U},
                 {4.0e-160, 0.0, 18.0, 2U},
                 {0.0, 0.0, 30.0, 1U}}},
    };

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), fixture.points, 3U, 2U,
                   std::nullopt, true);
        const std::vector<double> expected = runHag(
            "filters.hag_nn", fixture.points, 3U, 2U, std::nullopt, true);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountThreeIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 20.0, 1U},
                                       {5001.0, 0.0, 10.0, 2U},
                                       {5002.0, 0.0, 11.0, 2U},
                                       {5003.0, 0.0, 12.0, 2U},
                                       {1.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 3U),
              runHag("filters.hag_nn", points, 3U));
}

TEST(PdgHagNnFilter, CountFourCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},  {1.4, 0.0, 100.0, 1U}, {4.0, 0.0, 14.0, 2U},
        {6.5, 0.0, 200.0, 1U}, {10.0, 0.0, 20.0, 2U}, {12.0, 0.0, 24.0, 2U},
        {14.0, 0.0, 30.0, 1U}, {20.0, 0.0, 28.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 4U),
                  runHag("filters.hag_nn", points, 4U));
    }
}

TEST(PdgHagNnFilter, CountFourPreservesOptionsCutoffAndBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> maxDistance{{1.0, 0.0, 10.0, 2U},
                                            {2.0, 0.0, 18.0, 2U},
                                            {3.0, 0.0, 14.0, 2U},
                                            {4.0, 0.0, 22.0, 2U},
                                            {0.0, 0.0, 30.0, 1U}};
    for (const double limit : {4.0, 2.5, -4.0})
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 4U,
                         2U, limit, true),
                  runHag("filters.hag_nn", maxDistance, 4U, 2U, limit, true));

    const std::vector<HagPoint> boundsEdge{{0.0, 0.0, 10.0, 2U},
                                           {5.0, 4.0, 30.0, 2U},
                                           {0.0, 4.0, 18.0, 2U},
                                           {2.0, 0.0, 20.0, 2U},
                                           {0.0, 1.0, 100.0, 1U}};
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 4U, 2U,
               std::nullopt, false),
        runHag("filters.hag_nn", boundsEdge, 4U, 2U, std::nullopt, false));
}

TEST(PdgHagNnFilter, CountFourInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U}, {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U},
                              {4.0, 0.0, 14.0, 2U},
                              {8.0, 0.0, 18.0, 2U},
                              {1.0, 0.0, 17.0, 1U}},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    for (const std::vector<HagPoint>& points : fixtures)
    {
        ScopedEnvironment requireFallback(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 4U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 4U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountFourNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, 18.0, 2U},
        {12.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 2U},
        {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                      "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 4U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 4U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountFourBoundaryTieAndOverflowUsePinnedRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> boundary{
        {1.0, 0.0, 10.0, 2U},  {2.0, 0.0, 14.0, 2U},
        {3.0, 0.0, 18.0, 2U},  {4.0, 0.0, 22.0, 2U},
        {-4.0, 0.0, 26.0, 2U}, {0.0, 0.0, 30.0, 1U}};
    const std::vector<HagPoint> overflow{
        {1.4e154, 0.0, 10.0, 2U}, {1.5e154, 0.0, 14.0, 2U},
        {1.6e154, 0.0, 18.0, 2U}, {1.7e154, 0.0, 22.0, 2U},
        {1.8e154, 0.0, 26.0, 2U}, {0.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundary, 4U),
                  runHag("filters.hag_nn", boundary, 4U));
    }

    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), overflow, 4U);
    const std::vector<double> expected =
        runHag("filters.hag_nn", overflow, 4U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountFourArithmeticCasesMatchPinnedUpstream)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{{-0.0, 0.0, 10.0, 2U},
                              {5.0, 0.0, 1.0, 2U},
                              {0.0, 5.0, 30.0, 2U},
                              {7.0, 2.0, 22.0, 2U},
                              {1.0, 1.5, 20.0, 1U}},
        std::vector<HagPoint>{{1.0e-160, 0.0, 10.0, 2U},
                              {2.0e-160, 0.0, 14.0, 2U},
                              {4.0e-160, 0.0, 18.0, 2U},
                              {8.0e-160, 0.0, 22.0, 2U},
                              {0.0, 0.0, 30.0, 1U}},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const std::vector<HagPoint>& points : fixtures)
    {
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 4U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 4U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountFourIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 20.0, 1U},
                                       {5001.0, 0.0, 10.0, 2U},
                                       {5002.0, 0.0, 11.0, 2U},
                                       {5003.0, 0.0, 12.0, 2U},
                                       {5004.0, 0.0, 13.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 4U),
              runHag("filters.hag_nn", points, 4U));
}

TEST(PdgHagNnFilter, CountFiveCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.4, 0.0, 100.0, 1U},
                                       {4.0, 0.0, 14.0, 2U},
                                       {6.5, 0.0, 200.0, 1U},
                                       {10.0, 0.0, 20.0, 2U},
                                       {12.0, 0.0, 24.0, 2U},
                                       {14.0, 0.0, 30.0, 2U},
                                       {20.0, 0.0, 28.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 5U),
                  runHag("filters.hag_nn", points, 5U));
    }
}

TEST(PdgHagNnFilter, CountFivePreservesOptionsCutoffAndBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> maxDistance{{1.0, 0.0, 10.0, 2U},
                                            {2.0, 0.0, 18.0, 2U},
                                            {3.0, 0.0, 14.0, 2U},
                                            {4.0, 0.0, 22.0, 2U},
                                            {5.0, 0.0, 26.0, 2U},
                                            {0.0, 0.0, 30.0, 1U}};
    for (const double limit : {5.0, 2.5, -5.0})
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 5U,
                         2U, limit, true),
                  runHag("filters.hag_nn", maxDistance, 5U, 2U,
                         limit, true));

    const std::vector<HagPoint> boundsEdge{{0.0, 0.0, 10.0, 2U},
                                          {5.0, 4.0, 30.0, 2U},
                                          {0.0, 4.0, 18.0, 2U},
                                          {2.0, 0.0, 20.0, 2U},
                                          {6.0, 1.0, 24.0, 2U},
                                          {0.0, 1.0, 100.0, 1U}};
    EXPECT_EQ(
        runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 5U,
               2U, std::nullopt, false),
        runHag("filters.hag_nn", boundsEdge, 5U, 2U, std::nullopt, false));
}

TEST(PdgHagNnFilter, CountFiveInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
            {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
                             {8.0, 0.0, 18.0, 2U}, {1.0, 0.0, 17.0, 1U}},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    for (const std::vector<HagPoint>& points : fixtures)
    {
        ScopedEnvironment requireFallback(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 5U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 5U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountFiveNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, 18.0, 2U},
        {12.0, 0.0, 20.0, 2U},
        {16.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 2U},
        {1.0, 0.0, 30.0, 1U},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                    "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 5U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 5U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountFiveBoundaryTieAndOverflowUsePinnedRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> boundary{
        {1.0, 0.0, 10.0, 2U}, {2.0, 0.0, 14.0, 2U},
        {3.0, 0.0, 18.0, 2U}, {4.0, 0.0, 22.0, 2U},
        {5.0, 0.0, 26.0, 2U}, {-5.0, 0.0, 28.0, 2U},
        {0.0, 0.0, 30.0, 1U}
    };
    const std::vector<HagPoint> overflow{
        {1.4e154, 0.0, 10.0, 2U}, {1.5e154, 0.0, 14.0, 2U},
        {1.6e154, 0.0, 18.0, 2U}, {1.7e154, 0.0, 22.0, 2U},
        {1.8e154, 0.0, 26.0, 2U}, {1.9e154, 0.0, 28.0, 2U},
        {0.0, 0.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundary, 5U),
                  runHag("filters.hag_nn", boundary, 5U));
    }

    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), overflow, 5U);
    const std::vector<double> expected = runHag("filters.hag_nn", overflow, 5U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountFiveArithmeticCasesMatchPinnedUpstream)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{
            {-0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 1.0, 2U},
            {0.0, 5.0, 30.0, 2U}, {8.0e153, 8.0e153, 18.0, 2U},
            {1.0e154, 8.0e153, 22.0, 2U}, {1.0, 1.5, 20.0, 1U}},
        std::vector<HagPoint>{
            {1.0e-160, 0.0, 10.0, 2U}, {2.0e-160, 0.0, 14.0, 2U},
            {4.0e-160, 0.0, 18.0, 2U}, {8.0e-160, 0.0, 22.0, 2U},
            {1.6e-159, 0.0, 26.0, 2U}, {0.0, 0.0, 30.0, 1U}},
    };

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    for (const std::vector<HagPoint>& points : fixtures)
    {
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 5U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 5U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountFiveIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 20.0, 1U},
                                       {5001.0, 0.0, 10.0, 2U},
                                       {5002.0, 0.0, 11.0, 2U},
                                       {5003.0, 0.0, 12.0, 2U},
                                       {5004.0, 0.0, 13.0, 2U},
                                       {5005.0, 0.0, 14.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 5U),
              runHag("filters.hag_nn", points, 5U));
}

TEST(PdgHagNnFilter, CountSixCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {1.4, 0.0, 100.0, 1U}, {4.0, 0.0, 14.0, 2U},
        {6.5, 0.0, 200.0, 1U}, {10.0, 0.0, 20.0, 2U},
        {12.0, 0.0, 24.0, 2U}, {14.0, 0.0, 28.0, 2U},
        {16.0, 0.0, 30.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 6U),
                  runHag("filters.hag_nn", points, 6U));
    }
}

TEST(PdgHagNnFilter, CountSixPreservesOptionsCutoffAndBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> maxDistance{
        {1.0, 0.0, 10.0, 2U}, {2.0, 0.0, 18.0, 2U}, {3.0, 0.0, 14.0, 2U},
        {4.0, 0.0, 22.0, 2U}, {5.0, 0.0, 26.0, 2U}, {6.0, 0.0, 30.0, 2U},
        {0.0, 0.0, 34.0, 1U}};
    for (const double limit : {6.0, 3.0, -6.0})
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 6U,
                         2U, limit, true),
                  runHag("filters.hag_nn", maxDistance, 6U, 2U, limit, true));

    const std::vector<HagPoint> boundsEdge{
        {0.0, 0.0, 10.0, 2U}, {5.0, 4.0, 30.0, 2U}, {0.0, 4.0, 18.0, 2U},
        {2.0, 0.0, 20.0, 2U}, {6.0, 1.0, 24.0, 2U},
        {8.0, 0.0, 28.0, 2U}, {0.0, 1.0, 100.0, 1U}};
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 6U, 2U,
                     std::nullopt, false),
              runHag("filters.hag_nn", boundsEdge, 6U, 2U, std::nullopt,
                     false));
}

TEST(PdgHagNnFilter, CountSixInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
            {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
            {8.0, 0.0, 18.0, 2U}, {12.0, 0.0, 22.0, 2U},
            {16.0, 0.0, 26.0, 2U}, {1.0, 0.0, 17.0, 1U}},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    for (const std::vector<HagPoint>& points : fixtures)
    {
        ScopedEnvironment requireFallback(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 6U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 6U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountSixNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, 18.0, 2U},
        {12.0, 0.0, 20.0, 2U},
        {16.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 2U},
        {20.0, 0.0, 24.0, 2U},
        {1.0, 0.0, 30.0, 1U},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                     "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 6U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 6U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountSixBoundaryTieAndOverflowUsePinnedRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> boundary{
        {1.0, 0.0, 10.0, 2U}, {2.0, 0.0, 14.0, 2U},
        {3.0, 0.0, 18.0, 2U}, {4.0, 0.0, 22.0, 2U},
        {5.0, 0.0, 26.0, 2U}, {6.0, 0.0, 30.0, 2U},
        {-6.0, 0.0, 30.0, 2U}, {0.0, 0.0, 34.0, 1U}};
    const std::vector<HagPoint> overflow{
        {1.4e154, 0.0, 10.0, 2U}, {1.5e154, 0.0, 14.0, 2U},
        {1.6e154, 0.0, 18.0, 2U}, {1.7e154, 0.0, 22.0, 2U},
        {1.8e154, 0.0, 26.0, 2U}, {1.9e154, 0.0, 30.0, 2U},
        {0.0, 0.0, 34.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundary, 6U),
                  runHag("filters.hag_nn", boundary, 6U));
    }

    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), overflow, 6U);
    const std::vector<double> expected = runHag("filters.hag_nn", overflow, 6U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountSixArithmeticCasesMatchPinnedUpstream)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{
            {-0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 1.0, 2U},
            {0.0, 5.0, 30.0, 2U}, {8.0e153, 8.0e153, 18.0, 2U},
            {1.0e154, 8.0e153, 22.0, 2U}, {1.0e154, 1.0e154, 26.0, 2U},
            {1.0, 1.5, 20.0, 1U}},
        std::vector<HagPoint>{
            {1.0e-160, 0.0, 10.0, 2U}, {2.0e-160, 0.0, 14.0, 2U},
            {4.0e-160, 0.0, 18.0, 2U}, {8.0e-160, 0.0, 22.0, 2U},
            {1.6e-159, 0.0, 26.0, 2U}, {3.2e-159, 0.0, 28.0, 2U},
            {0.0, 0.0, 30.0, 1U}},
    };

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    for (const std::vector<HagPoint>& points : fixtures)
    {
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 6U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 6U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountSixIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 20.0, 1U}, {5001.0, 0.0, 10.0, 2U},
        {5002.0, 0.0, 11.0, 2U}, {5003.0, 0.0, 12.0, 2U},
        {5004.0, 0.0, 13.0, 2U}, {5005.0, 0.0, 14.0, 2U},
        {5006.0, 0.0, 15.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 6U),
              runHag("filters.hag_nn", points, 6U));
}

TEST(PdgHagNnFilter, CountSevenCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {1.4, 0.0, 100.0, 1U}, {4.0, 0.0, 14.0, 2U},
        {6.5, 0.0, 200.0, 1U}, {10.0, 0.0, 20.0, 2U},
        {12.0, 0.0, 24.0, 2U}, {14.0, 0.0, 28.0, 2U},
        {16.0, 0.0, 30.0, 2U}, {20.0, 0.0, 32.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 7U),
                  runHag("filters.hag_nn", points, 7U));
    }
}

TEST(PdgHagNnFilter, CountSevenPreservesOptionsCutoffAndBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    const std::vector<HagPoint> maxDistance{
        {1.0, 0.0, 10.0, 2U}, {2.0, 0.0, 18.0, 2U}, {3.0, 0.0, 14.0, 2U},
        {4.0, 0.0, 22.0, 2U}, {5.0, 0.0, 26.0, 2U}, {6.0, 0.0, 30.0, 2U},
        {7.0, 0.0, 32.0, 2U}, {0.0, 0.0, 36.0, 1U}};
    for (const double limit : {7.0, 3.0, -7.0})
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), maxDistance, 7U,
                         2U, limit, true),
                  runHag("filters.hag_nn", maxDistance, 7U, 2U, limit, true));

    const std::vector<HagPoint> boundsEdge{
        {0.0, 0.0, 10.0, 2U}, {5.0, 4.0, 30.0, 2U}, {0.0, 4.0, 18.0, 2U},
        {2.0, 0.0, 20.0, 2U}, {6.0, 1.0, 24.0, 2U},
        {8.0, 0.0, 28.0, 2U}, {4.0, 3.0, 26.0, 2U},
        {0.0, 1.0, 100.0, 1U}};
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundsEdge, 7U, 2U,
                     std::nullopt, false),
              runHag("filters.hag_nn", boundsEdge, 7U, 2U, std::nullopt,
                     false));
}

TEST(PdgHagNnFilter, CountSevenInsufficientGroundUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
            {1.0, 0.0, 17.0, 1U}},
        std::vector<HagPoint>{
            {0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U},
            {8.0, 0.0, 18.0, 2U}, {12.0, 0.0, 22.0, 2U},
            {16.0, 0.0, 26.0, 2U}, {20.0, 0.0, 28.0, 2U},
            {1.0, 0.0, 17.0, 1U}},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    for (const std::vector<HagPoint>& points : fixtures)
    {
        ScopedEnvironment requireFallback(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 7U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 7U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountSevenNonfiniteZUsesPinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U},
        {4.0, 0.0, 14.0, 2U},
        {8.0, 0.0, 18.0, 2U},
        {12.0, 0.0, 20.0, 2U},
        {16.0, 0.0, 22.0, 2U},
        {20.0, 0.0, (std::numeric_limits<double>::quiet_NaN)(), 2U},
        {24.0, 0.0, 24.0, 2U},
        {1.0, 0.0, 30.0, 1U},
    };
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK",
                                     "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), points, 7U);
    const std::vector<double> expected = runHag("filters.hag_nn", points, 7U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountSevenBoundaryTieAndOverflowUsePinnedRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> boundary{
        {1.0, 0.0, 10.0, 2U}, {2.0, 0.0, 14.0, 2U},
        {3.0, 0.0, 18.0, 2U}, {4.0, 0.0, 22.0, 2U},
        {5.0, 0.0, 26.0, 2U}, {6.0, 0.0, 30.0, 2U},
        {7.0, 0.0, 32.0, 2U}, {-7.0, 0.0, 32.0, 2U},
        {0.0, 0.0, 36.0, 1U}};
    const std::vector<HagPoint> overflow{
        {1.4e154, 0.0, 10.0, 2U}, {1.5e154, 0.0, 14.0, 2U},
        {1.6e154, 0.0, 18.0, 2U}, {1.7e154, 0.0, 22.0, 2U},
        {1.8e154, 0.0, 26.0, 2U}, {1.9e154, 0.0, 30.0, 2U},
        {2.0e154, 0.0, 32.0, 2U}, {0.0, 0.0, 36.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                   bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                   bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), boundary, 7U),
                  runHag("filters.hag_nn", boundary, 7U));
    }

    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
    const std::vector<double> actual =
        runHag(std::string(pdg::HybridHagNnStage), overflow, 7U);
    const std::vector<double> expected = runHag("filters.hag_nn", overflow, 7U);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

TEST(PdgHagNnFilter, CountSevenArithmeticCasesMatchPinnedUpstream)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::array fixtures{
        std::vector<HagPoint>{
            {-0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 1.0, 2U},
            {0.0, 5.0, 30.0, 2U}, {8.0e153, 8.0e153, 18.0, 2U},
            {1.0e154, 8.0e153, 22.0, 2U}, {1.0e154, 1.0e154, 26.0, 2U},
            {9.0e153, 9.0e153, 28.0, 2U}, {1.0, 1.5, 20.0, 1U}},
        std::vector<HagPoint>{
            {1.0e-160, 0.0, 10.0, 2U}, {2.0e-160, 0.0, 14.0, 2U},
            {4.0e-160, 0.0, 18.0, 2U}, {8.0e-160, 0.0, 22.0, 2U},
            {1.6e-159, 0.0, 26.0, 2U}, {3.2e-159, 0.0, 28.0, 2U},
            {6.4e-159, 0.0, 30.0, 2U}, {0.0, 0.0, 32.0, 1U}},
    };

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    for (const std::vector<HagPoint>& points : fixtures)
    {
        const std::vector<double> actual =
            runHag(std::string(pdg::HybridHagNnStage), points, 7U);
        const std::vector<double> expected =
            runHag("filters.hag_nn", points, 7U);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t point = 0U; point < actual.size(); ++point)
            if (std::isnan(expected[point]))
                EXPECT_TRUE(std::isnan(actual[point]));
            else
                EXPECT_EQ(actual[point], expected[point]);
    }
}

TEST(PdgHagNnFilter, CountSevenIncompleteGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 20.0, 1U}, {5001.0, 0.0, 10.0, 2U},
        {5002.0, 0.0, 11.0, 2U}, {5003.0, 0.0, 12.0, 2U},
        {5004.0, 0.0, 13.0, 2U}, {5005.0, 0.0, 14.0, 2U},
        {5006.0, 0.0, 15.0, 2U}, {5007.0, 0.0, 16.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 7U),
              runHag("filters.hag_nn", points, 7U));
}

// D0203: counts eight through 64 are proved by generating each obligation over
// count rather than transcribing it per count. The shapes mirror
// tests/differential/hag_nn_matrix.py's generator, and each keeps every
// retained candidate distance distinct unless it is deliberately proving a tie.
namespace
{

std::vector<HagPoint> generatedUniqueGrounds(std::uint32_t count)
{
    // count + 1 grounds at even X with the query at 0.5 gives strictly
    // increasing distances 0.5, 1.5, 3.5, ..., so the extra candidate proving
    // the boundary is distinct.
    std::vector<HagPoint> points;
    for (std::uint32_t item = 0U; item <= count; ++item)
        points.push_back({2.0 * item, 0.0, 10.0 + item, 2U});
    points.push_back({0.5, 0.0, 500.0, 1U});
    return points;
}

std::vector<HagPoint> generatedExactGrounds(std::uint32_t count)
{
    std::vector<HagPoint> points;
    for (std::uint32_t item = 0U; item < count; ++item)
        points.push_back({2.0 * item, 0.0, 10.0 + item, 2U});
    points.push_back({0.5, 0.0, 500.0, 1U});
    return points;
}

std::vector<HagPoint> generatedBoundaryGrounds(std::uint32_t count)
{
    // Distances are exactly 1..count, so max_distance == count is an equality
    // boundary and count / 2 is a partial cutoff.
    std::vector<HagPoint> points;
    for (std::uint32_t item = 1U; item <= count; ++item)
        points.push_back({static_cast<double>(item), 0.0, 10.0 + item, 2U});
    points.push_back({0.0, 0.0, 500.0, 1U});
    return points;
}

std::vector<HagPoint> generatedTiedGrounds(std::uint32_t count)
{
    // A mirrored ground makes the count-th and (count + 1)-th distances equal.
    std::vector<HagPoint> points = generatedBoundaryGrounds(count);
    points.insert(points.end() - 1,
                  {-static_cast<double>(count), 0.0, 300.0, 2U});
    return points;
}

std::vector<HagPoint> generatedIncompleteGrounds(std::uint32_t count)
{
    std::vector<HagPoint> points{{0.0, 0.0, 500.0, 1U}};
    for (std::uint32_t item = 0U; item < count; ++item)
        points.push_back({5001.0 + item, 0.0, 10.0 + item, 2U});
    return points;
}

std::vector<HagPoint> generatedNonfiniteZGrounds(std::uint32_t count)
{
    std::vector<HagPoint> points = generatedExactGrounds(count);
    points[static_cast<std::size_t>(count) - 1U].z =
        (std::numeric_limits<double>::quiet_NaN)();
    return points;
}

void expectMatchesPinned(const std::vector<double>& actual,
                         const std::vector<double>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected[point]))
            EXPECT_TRUE(std::isnan(actual[point]));
        else
            EXPECT_EQ(actual[point], expected[point]);
}

constexpr std::uint32_t GeneratedCounts[] = {8U, 16U, 32U, 64U};

} // unnamed namespace

TEST(PdgHagNnFilter, GeneratedWideCountsMatchPinnedUpstreamOnBothBackends)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const std::uint32_t count : GeneratedCounts)
    {
        SCOPED_TRACE(count);
        const std::vector<HagPoint> points = generatedUniqueGrounds(count);
        for (const bool bvh : {false, true})
        {
            SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
            ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                        bvh ? nullptr : "1");
            ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                        bvh ? "1" : nullptr);
            EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, count),
                      runHag("filters.hag_nn", points, count));
        }
    }
}

TEST(PdgHagNnFilter, GeneratedWideCountsPreserveCutoffAndBounds)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const std::uint32_t count : GeneratedCounts)
    {
        SCOPED_TRACE(count);
        const std::vector<HagPoint> points = generatedBoundaryGrounds(count);
        const double exact = static_cast<double>(count);
        for (const double limit : {exact, exact / 2.0, -exact})
        {
            SCOPED_TRACE(limit);
            expectMatchesPinned(
                runHag(std::string(pdg::HybridHagNnStage), points, count, 2U,
                       limit, true),
                runHag("filters.hag_nn", points, count, 2U, limit, true));
        }
        expectMatchesPinned(
            runHag(std::string(pdg::HybridHagNnStage), points, count, 2U,
                   std::nullopt, false),
            runHag("filters.hag_nn", points, count, 2U, std::nullopt, false));
    }
}

TEST(PdgHagNnFilter, GeneratedWideCountsRepairExceptionalRowsExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const std::uint32_t count : GeneratedCounts)
    {
        SCOPED_TRACE(count);

        {
            SCOPED_TRACE("insufficient ground");
            const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                               {1.0, 0.0, 17.0, 1U}};
            ScopedEnvironment require(
                "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK", "1");
            expectMatchesPinned(
                runHag(std::string(pdg::HybridHagNnStage), points, count),
                runHag("filters.hag_nn", points, count));
        }
        {
            SCOPED_TRACE("nonfinite Z");
            const std::vector<HagPoint> points =
                generatedNonfiniteZGrounds(count);
            ScopedEnvironment require(
                "PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK", "1");
            expectMatchesPinned(
                runHag(std::string(pdg::HybridHagNnStage), points, count),
                runHag("filters.hag_nn", points, count));
        }
        {
            SCOPED_TRACE("candidate boundary tie");
            const std::vector<HagPoint> points = generatedTiedGrounds(count);
            for (const bool bvh : {false, true})
            {
                SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
                ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                            bvh ? nullptr : "1");
                ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH",
                                            bvh ? "1" : nullptr);
                ScopedEnvironment require("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                          "1");
                expectMatchesPinned(
                    runHag(std::string(pdg::HybridHagNnStage), points, count),
                    runHag("filters.hag_nn", points, count));
            }
        }
        {
            SCOPED_TRACE("incomplete bounded grid search");
            const std::vector<HagPoint> points =
                generatedIncompleteGrounds(count);
            ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
            ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
            ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
            ScopedEnvironment require("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
            expectMatchesPinned(
                runHag(std::string(pdg::HybridHagNnStage), points, count),
                runHag("filters.hag_nn", points, count));
        }
    }
}

TEST(PdgHagNnFilter, RejectsCountAboveTheSharedIndexCap)
{
    // 64 is the shared spatial index's own maximumNeighbors cap, so 65 must
    // retain pinned behaviour rather than widening the envelope.
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
        dimensions);
    EXPECT_FALSE(plan.stages()[1].native);
    EXPECT_EQ(plan.stages()[1].preferredResidency, pdg::MemoryKind::Host);

    pdg::DimensionRegistry capDimensions;
    const pdg::Plan cap = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":64}, "out.las"])",
        capDimensions);
    EXPECT_TRUE(cap.stages()[1].native);
    EXPECT_EQ(cap.stages()[1].descriptor.index.neighbors, 64U);
}

TEST(PdgHagNnFilter, CountOneCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.0, 0.0, 100.0, 1U},
                                       {4.0, 0.0, 14.0, 2U},
                                       {8.0, 0.0, 200.0, 1U},
                                       {10.0, 0.0, 20.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points),
              runHag("filters.hag_nn", points));
}

TEST(PdgHagNnFilter, SelectiveRepairProofRejectsAnExactGpuResult)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.0, 0.0, 100.0, 1U},
                                       {4.0, 0.0, 14.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireRepair("PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
                                    "1");
    EXPECT_THROW(runHag(std::string(pdg::HybridHagNnStage), points),
                 std::runtime_error);
}

TEST(PdgHagNnFilter, SelectiveRepairProofRejectsCudaDecline)
{
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {2.0, 0.0, 20.0, 2U},
        {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireRepair("PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
                                    "1");
    EXPECT_THROW(runHag(std::string(pdg::HybridHagNnStage), points, 2U),
                 std::runtime_error);
}

TEST(PdgHagNnFilter, SelectivelyRepairsEqualDistanceGroundTieExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {2.0, 0.0, 20.0, 2U},
        {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireRepair("PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
                                    "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points, 2U),
              runHag("filters.hag_nn", points, 2U));
}

TEST(PdgHagNnFilter, EqualDistanceGroundTieFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{
        {0.0, 0.0, 10.0, 2U}, {2.0, 0.0, 20.0, 2U}, {1.0, 0.0, 100.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool bvh : {false, true})
    {
        SCOPED_TRACE(bvh ? "MortonBvh" : "UniformGrid");
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    bvh ? nullptr : "1");
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", bvh ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK", "1");
        EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points),
                  runHag("filters.hag_nn", points));
    }
}

TEST(PdgHagNnFilter, IncompleteUniformGridSearchFallsBackBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 20.0, 1U},
                                       {5001.0, 0.0, 10.0, 2U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment requireFallback("PDG_REQUIRE_HAG_NN_HOST_FALLBACK", "1");
    EXPECT_EQ(runHag(std::string(pdg::HybridHagNnStage), points),
              runHag("filters.hag_nn", points));
}

TEST(PdgHagNnFilter, CountsAcrossTheEnvelopeExecuteThroughResidentLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {1.0, 0.0, 100.0, 1U},
                                       {4.0, 0.0, 14.0, 2U},
                                       {8.0, 0.0, 200.0, 1U},
                                       {10.0, 0.0, 20.0, 2U}};
    for (const std::uint32_t count : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 16U, 64U})
    {
        SCOPED_TRACE(count);
        const std::string pipeline =
            R"([{"type":"readers.las","filename":"in.las"},)" +
            std::string{"{\"type\":\"filters.hag_nn\",\"count\":"} +
            std::to_string(count) +
            R"(},{"type":"writers.las","filename":"out.las"}])";
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        ASSERT_EQ(plan.summary().residentRegions, 1U);
        const std::size_t region = plan.stages().at(1U).residentRegion;
        ASSERT_NE(region, pdg::NoResidentRegion);
        const std::size_t upload =
            boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
        const std::size_t spill =
            boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

        const std::vector<double> expected =
            runHag("filters.hag_nn", points, count);
        using pdal::Dimension::Id;
        pdal::PointTable table;
        table.layout()->registerDims(
            {Id::X, Id::Y, Id::Z, Id::Classification, Id::HeightAboveGround});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const HagPoint& value = points[static_cast<std::size_t>(point)];
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        constexpr std::size_t Budget = 64U * 1024U * 1024U;
        pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                       64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", count);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", count);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", true);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        static_cast<void>(filter->execute(table));
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const double actual =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double oracle =
                expected.at(static_cast<std::size_t>(point));
            if (std::isnan(oracle))
                EXPECT_TRUE(std::isnan(actual));
            else
                EXPECT_EQ(actual, oracle);
        }
    }
}

TEST(PdgHagNnFilter, ResidentProductFeedsDownstreamAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":1},
      {"type":"filters.assign","value":"UserData = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);
    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool requireTieFallback;
    };
    const std::array fixtures{
        Fixture{
            "unique",
            {{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 15.0, 1U}, {4.0, 0.0, 14.0, 2U}},
            false},
        Fixture{
            "equal-distance tie",
            {{0.0, 0.0, 10.0, 2U}, {2.0, 0.0, 20.0, 2U}, {1.0, 0.0, 100.0, 1U}},
            true},
        Fixture{
            "no ground", {{0.0, 0.0, 10.0, 1U}, {1.0, 0.0, 15.0, 1U}}, false},
        Fixture{"nonfinite query",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {(std::numeric_limits<double>::quiet_NaN)(), 0.0, 20.0, 1U}},
                false},
        Fixture{"empty", {}, false},
    };
    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::UserData});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
            view->setField(Id::UserData, point, std::uint8_t{0U});
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 1U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.requireTieFallback ? "1"
                                                                : nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       fixture.points.empty() ? nullptr : "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            EXPECT_EQ(view->getFieldAs<double>(Id::HeightAboveGround, point),
                      hag);
            EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::UserData, point),
                      static_cast<std::uint8_t>(hag));
        }
    }
}

TEST(PdgHagNnFilter, CountTwoNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":2},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
    };
    const std::array fixtures{
        Fixture{
            "native",
            {{0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U}, {1.0, 0.0, 100.0, 1U}},
            false},
        Fixture{"insufficient-ground host repair",
                {{0.0, 0.0, 10.0, 2U}, {1.0, 0.0, 17.0, 1U}},
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 2U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 2U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 2U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, CountThreeNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":3},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
        bool tiedBoundary;
        bool incompleteSearch;
    };
    const std::array fixtures{
        Fixture{"native",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {10.0, 0.0, 18.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                false,
                false,
                false},
        Fixture{
            "insufficient-ground host repair",
            {{0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U}, {1.0, 0.0, 100.0, 1U}},
            true,
            false,
            false},
        Fixture{"tied fourth-candidate host repair",
                {{1.0, 0.0, 10.0, 2U},
                 {0.0, 1.0, 14.0, 2U},
                 {-1.0, 0.0, 18.0, 2U},
                 {0.0, -1.0, 22.0, 2U},
                 {0.0, 0.0, 30.0, 1U}},
                false,
                true,
                false},
        Fixture{"incomplete-grid host repair",
                {{0.0, 0.0, 30.0, 1U},
                 {5001.0, 0.0, 10.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {5003.0, 0.0, 12.0, 2U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 3U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 3U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 3U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, CountFourNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":4},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
        bool tiedBoundary;
        bool incompleteSearch;
    };
    const std::array fixtures{
        Fixture{"native",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {10.0, 0.0, 18.0, 2U},
                 {14.0, 0.0, 22.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                false,
                false,
                false},
        Fixture{"insufficient-ground host repair",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {8.0, 0.0, 18.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                true,
                false,
                false},
        Fixture{"tied fifth-candidate host repair",
                {{1.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 14.0, 2U},
                 {3.0, 0.0, 18.0, 2U},
                 {4.0, 0.0, 22.0, 2U},
                 {-4.0, 0.0, 26.0, 2U},
                 {0.0, 0.0, 30.0, 1U}},
                false,
                true,
                false},
        Fixture{"incomplete-grid host repair",
                {{0.0, 0.0, 30.0, 1U},
                 {5001.0, 0.0, 10.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {5003.0, 0.0, 12.0, 2U},
                 {5004.0, 0.0, 13.0, 2U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 4U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 4U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 4U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, CountFiveNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":5},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
        bool tiedBoundary;
        bool incompleteSearch;
    };
    const std::array fixtures{
        Fixture{"native",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {10.0, 0.0, 18.0, 2U},
                 {14.0, 0.0, 22.0, 2U},
                 {18.0, 0.0, 26.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                false,
                false,
                false},
        Fixture{"insufficient-ground host repair",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {8.0, 0.0, 18.0, 2U},
                 {12.0, 0.0, 20.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                true,
                false,
                false},
        Fixture{"tied sixth-candidate host repair",
                {{1.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 14.0, 2U},
                 {3.0, 0.0, 18.0, 2U},
                 {4.0, 0.0, 22.0, 2U},
                 {5.0, 0.0, 26.0, 2U},
                 {-5.0, 0.0, 28.0, 2U},
                 {0.0, 0.0, 30.0, 1U}},
                false,
                true,
                false},
        Fixture{"incomplete-grid host repair",
                {{0.0, 0.0, 30.0, 1U},
                 {5001.0, 0.0, 10.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {5003.0, 0.0, 12.0, 2U},
                 {5004.0, 0.0, 13.0, 2U},
                 {5005.0, 0.0, 14.0, 2U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 5U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 5U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 5U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, CountSixNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":6},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
        bool tiedBoundary;
        bool incompleteSearch;
    };
    const std::array fixtures{
        Fixture{"native",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {10.0, 0.0, 18.0, 2U},
                 {14.0, 0.0, 22.0, 2U},
                 {18.0, 0.0, 26.0, 2U},
                 {20.0, 0.0, 30.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                false,
                false,
                false},
        Fixture{"insufficient-ground host repair",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {8.0, 0.0, 18.0, 2U},
                 {12.0, 0.0, 20.0, 2U},
                 {16.0, 0.0, 22.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                true,
                false,
                false},
        Fixture{"tied seventh-candidate host repair",
                {{1.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 14.0, 2U},
                 {3.0, 0.0, 18.0, 2U},
                 {4.0, 0.0, 22.0, 2U},
                 {5.0, 0.0, 26.0, 2U},
                 {6.0, 0.0, 30.0, 2U},
                 {-6.0, 0.0, 34.0, 2U},
                 {0.0, 0.0, 38.0, 1U}},
                false,
                true,
                false},
        Fixture{"incomplete-grid host repair",
                {{0.0, 0.0, 30.0, 1U},
                 {5001.0, 0.0, 10.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {5003.0, 0.0, 12.0, 2U},
                 {5004.0, 0.0, 13.0, 2U},
                 {5005.0, 0.0, 14.0, 2U},
                 {5006.0, 0.0, 15.0, 2U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 6U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 6U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 6U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, CountSevenNativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":7},
      {"type":"filters.assign","value":"NNDistance = HeightAboveGround"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const auto& assignment =
        std::get<pdg::AssignProgram>(plan.stages().at(2U).payload);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool insufficientGround;
        bool tiedBoundary;
        bool incompleteSearch;
    };
    const std::array fixtures{
        Fixture{"native",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {10.0, 0.0, 18.0, 2U},
                 {14.0, 0.0, 22.0, 2U},
                 {18.0, 0.0, 26.0, 2U},
                 {20.0, 0.0, 30.0, 2U},
                 {26.0, 0.0, 32.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                false,
                false,
                false},
        Fixture{"insufficient-ground host repair",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {8.0, 0.0, 18.0, 2U},
                 {12.0, 0.0, 20.0, 2U},
                 {16.0, 0.0, 22.0, 2U},
                 {20.0, 0.0, 24.0, 2U},
                 {1.0, 0.0, 100.0, 1U}},
                true,
                false,
                false},
        Fixture{"tied eighth-candidate host repair",
                {{1.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 14.0, 2U},
                 {3.0, 0.0, 18.0, 2U},
                 {4.0, 0.0, 22.0, 2U},
                 {5.0, 0.0, 26.0, 2U},
                 {6.0, 0.0, 30.0, 2U},
                 {7.0, 0.0, 34.0, 2U},
                 {-7.0, 0.0, 36.0, 2U},
                 {0.0, 0.0, 38.0, 1U}},
                false,
                true,
                false},
        Fixture{"incomplete-grid host repair",
                {{0.0, 0.0, 30.0, 1U},
                 {5001.0, 0.0, 10.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {5003.0, 0.0, 12.0, 2U},
                 {5004.0, 0.0, 13.0, 2U},
                 {5005.0, 0.0, 14.0, 2U},
                 {5006.0, 0.0, 15.0, 2U},
                 {5007.0, 0.0, 16.0, 2U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHag("filters.hag_nn", fixture.points, 7U);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* filter =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        ASSERT_NE(filter, nullptr);
        pdal::Options options;
        options.add("count", 7U);
        options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        options.add("pdg_region_neighbors", 7U);
        options.add("pdg_region_dimensions", 2U);
        options.add("pdg_region_last", false);
        options.add("pdg_resident_context", true);
        options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        ScopedEnvironment requireInsufficient(
            "PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_NN_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
        static_cast<void>(filter->execute(table));

        pdal::pdg_detail::CudaNeighborhoodRegion bridge;
        bridge.id = static_cast<std::uint64_t>(region + 1U);
        bridge.dimensions = 2U;
        bridge.last = true;
        ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                       "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
            *view, bridge, assignment, /*requireCuda=*/true));
        context.endDelegatedRegion(*view, region);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const double hag = expected.at(static_cast<std::size_t>(point));
            const double actualHag =
                view->getFieldAs<double>(Id::HeightAboveGround, point);
            const double actualBridge =
                view->getFieldAs<double>(Id::NNDistance, point);
            if (std::isnan(hag))
            {
                EXPECT_TRUE(std::isnan(actualHag));
                EXPECT_TRUE(std::isnan(actualBridge));
            }
            else
            {
                EXPECT_EQ(actualHag, hag);
                EXPECT_EQ(actualBridge, hag);
            }
        }
    }
}

TEST(PdgHagNnFilter, ResidentLifecycleRebuildsAcrossTwoAndThreeDimensions)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    struct Fixture
    {
        const char* name;
        bool hagFirst;
        bool requireTieFallback;
        std::vector<HagPoint> points;
    };
    const std::array fixtures{
        Fixture{"HAG tie then 3D nndistance",
                true,
                true,
                {{0.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 20.0, 2U},
                 {1.0, 0.0, 100.0, 1U},
                 {0.0, 3.0, 30.0, 1U},
                 {5.0, 4.0, 40.0, 1U}}},
        Fixture{"HAG no-ground repair then 3D nndistance",
                true,
                false,
                {{0.0, 0.0, 10.0, 1U},
                 {2.0, 0.0, 20.0, 1U},
                 {1.0, 0.0, 100.0, 1U},
                 {0.0, 3.0, 30.0, 1U},
                 {5.0, 4.0, 40.0, 1U}}},
        Fixture{"3D nndistance then HAG tie",
                false,
                true,
                {{0.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 20.0, 2U},
                 {1.0, 0.0, 100.0, 1U},
                 {0.0, 3.0, 30.0, 1U},
                 {5.0, 4.0, 40.0, 1U}}},
    };
    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const NeighborhoodColumns expected =
            runHostNeighborhoodChain(fixture.points, fixture.hagFirst);
        const std::string pipeline =
            fixture.hagFirst
                ? R"(["in.las",{"type":"filters.hag_nn","count":1},{"type":"filters.nndistance","k":3},"out.las"] )"
                : R"(["in.las",{"type":"filters.nndistance","k":3},{"type":"filters.hag_nn","count":1},"out.las"] )";
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        ASSERT_EQ(plan.summary().residentRegions, 1U);
        ASSERT_EQ(plan.summary().indexBuilds, 2U);
        const std::size_t region = plan.stages().at(1U).residentRegion;
        ASSERT_EQ(region, plan.stages().at(2U).residentRegion);
        const std::size_t upload =
            boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
        const std::size_t spill =
            boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::HeightAboveGround, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const HagPoint& value =
                fixture.points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value.x);
            view->setField(Id::Y, point, value.y);
            view->setField(Id::Z, point, value.z);
            view->setField(Id::Classification, point, value.classification);
        }

        pdg::ExecutionObservationScope observation;
        pdal::pdg_detail::ResidentExecutionScope scope(
            plan, dimensions, 64U * 1024U * 1024U, 64U);
        const std::array selectedRegions{region};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            region,
            plan.summary()
                .residencyBoundaries.at(upload)
                .requiresFullPointRecord);

        pdal::BufferReader reader;
        reader.addView(view);
        pdal::StageFactory factory;
        pdal::Stage* hag =
            factory.createStage(std::string(pdg::HybridHagNnStage));
        pdal::Stage* nn =
            factory.createStage(std::string(pdg::HybridNnDistanceStage));
        ASSERT_NE(hag, nullptr);
        ASSERT_NE(nn, nullptr);
        pdal::Options hagOptions;
        hagOptions.add("count", 1U);
        hagOptions.add("pdg_region_id",
                       static_cast<std::uint64_t>(region + 1U));
        hagOptions.add("pdg_region_neighbors", 4U);
        hagOptions.add("pdg_region_dimensions", 2U);
        hagOptions.add("pdg_region_last", !fixture.hagFirst);
        hagOptions.add("pdg_resident_context", true);
        hagOptions.add("pdg_execution_region",
                       static_cast<std::uint64_t>(region));
        hag->setOptions(hagOptions);
        pdal::Options nnOptions;
        nnOptions.add("k", 3U);
        nnOptions.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
        nnOptions.add("pdg_region_neighbors", 4U);
        nnOptions.add("pdg_region_last", fixture.hagFirst);
        nnOptions.add("pdg_resident_context", true);
        nnOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
        nn->setOptions(nnOptions);
        pdal::Stage* first = fixture.hagFirst ? hag : nn;
        pdal::Stage* second = fixture.hagFirst ? nn : hag;
        first->setInput(reader);
        second->setInput(*first);
        second->prepare(table);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_NN_TIE_FALLBACK",
                                     fixture.requireTieFallback ? "1"
                                                                : nullptr);
        static_cast<void>(second->execute(table));
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            region,
            plan.summary()
                .residencyBoundaries.at(spill)
                .requiresFullPointRecord);

        const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
        EXPECT_EQ(stats
                      .totals[static_cast<std::size_t>(
                          pdg::ExecutionEventKind::IndexBuild)]
                      .count,
                  2U);
        for (pdal::PointId point = 0U; point < fixture.points.size(); ++point)
        {
            const std::size_t row = static_cast<std::size_t>(point);
            EXPECT_EQ(view->getFieldAs<double>(Id::HeightAboveGround, point),
                      expected.hag.at(row));
            EXPECT_EQ(view->getFieldAs<double>(Id::NNDistance, point),
                      expected.nnDistance.at(row));
        }
    }
}
