#include <pdg/Cuda.hpp>
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

std::vector<double> runHagDelaunay(const std::string& stageName,
                                   const std::vector<HagPoint>& points,
                                   std::uint64_t count = 3U,
                                   std::uint8_t groundClass = 2U,
                                   bool allowExtrapolation = true)
{
    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::X, Id::Y, Id::Z, Id::Classification, Id::HeightAboveGround});
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
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("HAG Delaunay test stage is unavailable");
    pdal::Options options;
    options.add("count", count);
    options.add("allow_extrapolation", allowExtrapolation);
    options.add("class", static_cast<unsigned int>(groundClass));
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

void expectExact(const std::vector<double>& actual,
                 const std::vector<double>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t point = 0U; point < actual.size(); ++point)
        if (std::isnan(expected.at(point)))
            EXPECT_TRUE(std::isnan(actual.at(point))) << point;
        else
            EXPECT_EQ(actual.at(point), expected.at(point)) << point;
}
} // unnamed namespace

TEST(PdgHagDelaunayFilter, CountThreeCudaMatchesPinnedUpstreamExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        bool allowExtrapolation;
        std::uint8_t groundClass;
    };
    const std::array fixtures{
        Fixture{"inside triangle",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {1.0, 0.5, 30.0, 1U}},
                true,
                2U},
        Fixture{"Delaunator seed reordering",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {1.0, 3.0, 18.0, 2U},
                 {1.5, 0.5, 30.0, 1U}},
                true,
                2U},
        Fixture{"same XY",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {0.0, 0.0, 30.0, 1U}},
                true,
                2U},
        Fixture{"outside local triangle",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {4.0, 2.0, 30.0, 1U}},
                true,
                2U},
        Fixture{"outside global bounds without extrapolation",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {6.0, 4.0, 30.0, 1U}},
                false,
                2U},
        Fixture{"barycentric outside-edge tolerance",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {1.5, -1e-14, 30.0, 1U}},
                true,
                2U},
        Fixture{"barycentric outside-edge rejection",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {1.5, -1e-12, 30.0, 1U}},
                true,
                2U},
        Fixture{"negative-infinity interpolation is not outside",
                {{0.0, 0.0, -(std::numeric_limits<double>::max)(), 2U},
                 {2.0, 0.0, -(std::numeric_limits<double>::max)(), 2U},
                 {0.0, 2.0, -(std::numeric_limits<double>::max)(), 2U},
                 {0.3, 0.5, 0.0, 1U}},
                true,
                2U},
        Fixture{"positive-infinity interpolation falls back to nearest",
                {{0.0, 0.0, (std::numeric_limits<double>::max)(), 2U},
                 {2.0, 0.0, (std::numeric_limits<double>::max)(), 2U},
                 {0.0, 2.0, (std::numeric_limits<double>::max)(), 2U},
                 {0.3, 0.5, 0.0, 1U}},
                true,
                2U},
        Fixture{"signed-zero coordinates",
                {{-0.0, 0.0, 10.0, 2U},
                 {5.0, -0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {1.0, 0.5, 30.0, 1U}},
                true,
                2U},
        Fixture{"large finite coordinate products",
                {{8e153, 8e153, 10.0, 2U},
                 {1e154, 8e153, 14.0, 2U},
                 {8e153, 1e154, 18.0, 2U},
                 {8.4e153, 8.6e153, 30.0, 1U}},
                true,
                2U},
        Fixture{"global bounds edge outside selected triangle",
                {{0.0, 0.0, 10.0, 2U},
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {10.0, 10.0, 22.0, 2U},
                 {4.0, 3.0, 30.0, 1U}},
                false,
                2U},
        Fixture{"custom ground class",
                {{0.0, 0.0, 10.0, 9U},
                 {5.0, 0.0, 14.0, 9U},
                 {0.0, 3.0, 18.0, 9U},
                 {1.0, 0.5, 30.0, 1U}},
                true,
                9U},
        Fixture{"collinear ground",
                {{0.0, 0.0, 10.0, 2U},
                 {2.0, 0.0, 14.0, 2U},
                 {5.0, 0.0, 18.0, 2U},
                 {0.5, 1.0, 30.0, 1U}},
                true,
                2U},
        Fixture{
            "all ground",
            {{0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 14.0, 2U}, {1.0, 3.0, 18.0, 2U}},
            true,
            2U},
    };

    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        expectExact(runHagDelaunay(std::string(pdg::HybridHagDelaunayStage),
                                   fixture.points, 3U, fixture.groundClass,
                                   fixture.allowExtrapolation),
                    runHagDelaunay("filters.hag_delaunay", fixture.points, 3U,
                                   fixture.groundClass,
                                   fixture.allowExtrapolation));
    }
}

TEST(PdgHagDelaunayFilter, DataDependentAmbiguitiesUsePinnedHostRepair)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);

    struct Fixture
    {
        const char* name;
        std::vector<HagPoint> points;
        const char* proof;
    };
    const std::array fixtures{
        Fixture{"third/fourth distance tie",
                {{1.0, 0.0, 10.0, 2U},
                 {0.0, 1.0, 14.0, 2U},
                 {-1.0, 0.0, 18.0, 2U},
                 {0.0, -1.0, 22.0, 2U},
                 {0.0, 0.0, 30.0, 1U}},
                "PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK"},
        Fixture{"duplicate-ground tie",
                {{0.0, 0.0, 10.0, 2U},
                 {0.0, 0.0, 14.0, 2U},
                 {4.0, 0.0, 18.0, 2U},
                 {1.0, 1.0, 30.0, 1U}},
                "PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK"},
        Fixture{"subnormal-distance tie",
                {{0.0, 0.0, 10.0, 2U},
                 {1e-300, 0.0, 14.0, 2U},
                 {0.0, 2e-300, 18.0, 2U},
                 {3e-300, 0.0, 22.0, 2U},
                 {5e-301, 5e-301, 30.0, 1U}},
                "PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK"},
        Fixture{
            "insufficient ground",
            {{0.0, 0.0, 10.0, 2U}, {4.0, 0.0, 14.0, 2U}, {1.0, 1.0, 30.0, 1U}},
            "PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK"},
        Fixture{"nonfinite Z",
                {{0.0, 0.0, 10.0, 2U},
                 {4.0, 0.0, 14.0, 2U},
                 {0.0, 4.0, 18.0, 2U},
                 {1.0, 1.0, (std::numeric_limits<double>::quiet_NaN)(), 1U}},
                "PDG_REQUIRE_HAG_DELAUNAY_NONFINITE_FALLBACK"},
    };

    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        ScopedEnvironment requireFallback(fixture.proof, "1");
        expectExact(runHagDelaunay(std::string(pdg::HybridHagDelaunayStage),
                                   fixture.points),
                    runHagDelaunay("filters.hag_delaunay", fixture.points));
    }
}

TEST(PdgHagDelaunayFilter, UnsupportedCountsRemainPinnedHostOwned)
{
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {4.0, 0.0, 14.0, 2U},
                                       {0.0, 4.0, 18.0, 2U},
                                       {4.0, 4.0, 22.0, 2U},
                                       {1.0, 1.0, 30.0, 1U}};
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const std::uint64_t count : {4U, 10U})
    {
        SCOPED_TRACE(count);
        EXPECT_EQ(runHagDelaunay(std::string(pdg::HybridHagDelaunayStage),
                                 points, count),
                  runHagDelaunay("filters.hag_delaunay", points, count));
    }
}

TEST(PdgHagDelaunayFilter, CountThreeExecutesThroughResidentLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_delaunay","count":3},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);
    const std::vector<HagPoint> points{{0.0, 0.0, 10.0, 2U},
                                       {5.0, 0.0, 14.0, 2U},
                                       {0.0, 3.0, 18.0, 2U},
                                       {1.0, 0.5, 30.0, 1U}};
    const std::vector<double> expected =
        runHagDelaunay("filters.hag_delaunay", points);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::X, Id::Y, Id::Z, Id::Classification, Id::HeightAboveGround});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const HagPoint& value = points.at(static_cast<std::size_t>(point));
        view->setField(Id::X, point, value.x);
        view->setField(Id::Y, point, value.y);
        view->setField(Id::Z, point, value.z);
        view->setField(Id::Classification, point, value.classification);
    }

    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions,
                                                   64U * 1024U * 1024U, 64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter =
        factory.createStage(std::string(pdg::HybridHagDelaunayStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("count", 3U);
    options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
    options.add("pdg_region_neighbors", 3U);
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
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    std::vector<double> actual;
    for (pdal::PointId point = 0U; point < view->size(); ++point)
        actual.push_back(
            view->getFieldAs<double>(Id::HeightAboveGround, point));
    expectExact(actual, expected);
}

TEST(PdgHagDelaunayFilter, NativeAndHostRepairFeedAssignmentBridge)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_delaunay","count":3},
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
                 {5.0, 0.0, 14.0, 2U},
                 {0.0, 3.0, 18.0, 2U},
                 {1.0, 0.5, 30.0, 1U}},
                false,
                false,
                false},
        Fixture{
            "insufficient-ground host repair",
            {{0.0, 0.0, 10.0, 2U}, {5.0, 0.0, 14.0, 2U}, {1.0, 0.5, 30.0, 1U}},
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
                {{0.0, 0.0, 10.0, 2U},
                 {5001.0, 0.0, 20.0, 2U},
                 {5002.0, 0.0, 11.0, 2U},
                 {1.0, 1.0, 30.0, 1U}},
                false,
                false,
                true},
    };

    using pdal::Dimension::Id;
    for (const Fixture& fixture : fixtures)
    {
        SCOPED_TRACE(fixture.name);
        const std::vector<double> expected =
            runHagDelaunay("filters.hag_delaunay", fixture.points);
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
            factory.createStage(std::string(pdg::HybridHagDelaunayStage));
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
            "PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK",
            fixture.insufficientGround ? "1" : nullptr);
        ScopedEnvironment requireTie("PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK",
                                     fixture.tiedBoundary ? "1" : nullptr);
        ScopedEnvironment requireHost("PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET",
                                      fixture.incompleteSearch ? "1" : nullptr);
        ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID",
                                    fixture.incompleteSearch ? "1" : nullptr);
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

        for (pdal::PointId point = 0U; point < view->size(); ++point)
        {
            const double expectedHag =
                expected.at(static_cast<std::size_t>(point));
            EXPECT_EQ(view->getFieldAs<double>(Id::HeightAboveGround, point),
                      expectedHag);
            EXPECT_EQ(view->getFieldAs<double>(Id::NNDistance, point),
                      expectedHag);
        }
    }
}
