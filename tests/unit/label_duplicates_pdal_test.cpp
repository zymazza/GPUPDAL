#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>
#include <pdg/stages/Assign.hpp>

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
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
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

std::vector<std::uint8_t> runLabelDuplicates(const std::string& stageName,
                                             const char* dimensions)
{
    using pdal::Dimension::Id;
    constexpr std::array<double, 8> X{
        {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, -0.0, 0.0}};
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::GpsTime, Id::Duplicate});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < X.size(); ++point)
    {
        view->setField(Id::X, point, X[static_cast<std::size_t>(point)]);
        const double gps = point >= 4U && point <= 5U
                               ? std::numeric_limits<double>::quiet_NaN()
                               : X[static_cast<std::size_t>(point)];
        view->setField(Id::GpsTime, point, gps);
        view->setField(Id::Duplicate, point,
                       static_cast<std::uint8_t>(point == 0U ? 7U : 3U));
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("label_duplicates test stage is missing");
    pdal::Options options;
    if (dimensions)
        options.add("dimensions", dimensions);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));

    std::vector<std::uint8_t> result;
    result.reserve(X.size());
    for (pdal::PointId point = 0; point < X.size(); ++point)
        result.push_back(view->getFieldAs<std::uint8_t>(Id::Duplicate, point));
    return result;
}
} // unnamed namespace

TEST(PdgLabelDuplicatesFilter, HostFallbackMatchesUpstreamExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const char* dimensions :
         std::array<const char*, 2>{"X,GpsTime", nullptr})
        EXPECT_EQ(runLabelDuplicates(
                      std::string(pdg::HybridLabelDuplicatesStage), dimensions),
                  runLabelDuplicates("filters.label_duplicates", dimensions));
}

TEST(PdgLabelDuplicatesFilter, CudaMatchesUpstreamSpecialValuesExactly)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const char* dimensions :
         std::array<const char*, 2>{"X,GpsTime", nullptr})
        EXPECT_EQ(runLabelDuplicates(
                      std::string(pdg::HybridLabelDuplicatesStage), dimensions),
                  runLabelDuplicates("filters.label_duplicates", dimensions));
}

TEST(PdgLabelDuplicatesFilter, SelfReferentialOutputUsesExactHostFallback)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", "1");
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    const std::vector<std::uint8_t> expected{7U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    EXPECT_EQ(runLabelDuplicates(std::string(pdg::HybridLabelDuplicatesStage),
                                 "Duplicate"),
              expected);
    EXPECT_EQ(runLabelDuplicates("filters.label_duplicates", "Duplicate"),
              expected);
}

TEST(PdgLabelDuplicatesFilter,
     ResidentNoIndexProductFeedsDownstreamPointProgram)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::Classification, Id::Duplicate, Id::UserData});
    pdal::PointViewPtr view(new pdal::PointView(table));
    constexpr std::array<std::uint8_t, 5> Classification{{1U, 1U, 2U, 2U, 2U}};
    for (pdal::PointId point = 0; point < Classification.size(); ++point)
    {
        view->setField(Id::Classification, point,
                       Classification[static_cast<std::size_t>(point)]);
        view->setField(Id::Duplicate, point,
                       static_cast<std::uint8_t>(point == 0U ? 9U : 3U));
        view->setField(Id::UserData, point, std::uint8_t{0U});
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([{"type":"readers.las","filename":"in.las"},
             {"type":"filters.label_duplicates","dimensions":"Classification"},
             {"type":"filters.assign","value":"UserData = Duplicate"},
             {"type":"writers.las","filename":"out.las"}])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const auto boundaryId = [&](pdg::ResidencyBoundaryKind kind)
    {
        const auto position =
            std::find_if(plan.summary().residencyBoundaries.begin(),
                         plan.summary().residencyBoundaries.end(),
                         [&](const pdg::ResidencyBoundary& boundary)
                         { return boundary.kind == kind; });
        EXPECT_NE(position, plan.summary().residencyBoundaries.end());
        return static_cast<std::size_t>(std::distance(
            plan.summary().residencyBoundaries.begin(), position));
    };
    const std::size_t upload = boundaryId(pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill = boundaryId(pdg::ResidencyBoundaryKind::Spill);
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, 1U << 30U,
                                                   16U);
    const std::array<std::size_t, 1> selectedRegions{0U};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    context.enterBoundary(*view, upload,
                          pdal::pdg_detail::ResidentBoundaryDirection::Upload,
                          0U, false);
    context.beginDelegatedRegion(*view, 0U);

    pdg::LabelDuplicatesProgram label;
    label.dimensions = {
        pdg::DimensionId(pdg::StandardDimension::Classification)};
    pdal::pdg_detail::CudaNeighborhoodRegion region;
    region.id = 1U;
    region.indexRequired = false;
    region.last = false;
    ASSERT_TRUE(pdal::pdg_detail::tryCudaLabelDuplicatesColumn(
        *view, label, dimensions, region, /*requireCuda=*/true));

    const std::array<std::string, 1> specifications{"UserData = Duplicate"};
    const pdg::AssignProgram assignment =
        pdg::compileAssignments(specifications, dimensions);
    pdal::pdg_detail::CudaNeighborhoodRegion bridge;
    bridge.id = region.id;
    bridge.indexRequired = false;
    bridge.last = true;
    ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
                                   "1");
    ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
        *view, bridge, assignment, /*requireCuda=*/true));
    context.endDelegatedRegion(*view, 0U);
    context.enterBoundary(*view, spill,
                          pdal::pdg_detail::ResidentBoundaryDirection::Spill,
                          0U, false);

    const std::array<std::uint8_t, 5> expected{{9U, 1U, 0U, 1U, 1U}};
    for (pdal::PointId point = 0; point < expected.size(); ++point)
    {
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Duplicate, point),
                  expected[static_cast<std::size_t>(point)]);
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::UserData, point),
                  expected[static_cast<std::size_t>(point)]);
    }
}

TEST(PdgLabelDuplicatesFilter, EmptyResidentCompositionsRemainExactNoOps)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }

    enum class Consumer
    {
        Assignment,
        NnDistance,
        Normal
    };
    using pdal::Dimension::Id;
    for (const Consumer consumer :
         {Consumer::Assignment, Consumer::NnDistance, Consumer::Normal})
    {
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                      Id::Duplicate, Id::UserData,
                                      Id::NNDistance, Id::NormalX, Id::NormalY,
                                      Id::NormalZ, Id::Curvature});
        pdal::PointViewPtr view(new pdal::PointView(table));
        if (consumer == Consumer::Normal)
        {
            pdal::pdg_detail::CudaNeighborhoodRegion standalone;
            std::shared_ptr<const pdal::pdg_detail::CudaNeighborhoodResults>
                results;
            EXPECT_FALSE(pdal::pdg_detail::tryCudaNormalColumns(
                *view, 4U, standalone, /*alwaysUp=*/true, results,
                /*requireCuda=*/true));
            EXPECT_FALSE(results);
        }
        pdg::DimensionRegistry dimensions;
        const std::string pipeline =
            consumer == Consumer::NnDistance
                ? R"([{"type":"readers.las","filename":"in.las"},
                       {"type":"filters.label_duplicates","dimensions":"Classification"},
                       {"type":"filters.nndistance","k":3},
                       {"type":"writers.las","filename":"out.las"}])"
            : consumer == Consumer::Normal
                ? R"([{"type":"readers.las","filename":"in.las"},
                           {"type":"filters.label_duplicates","dimensions":"Classification"},
                           {"type":"filters.normal","knn":3},
                           {"type":"writers.las","filename":"out.las"}])"
                : R"([{"type":"readers.las","filename":"in.las"},
                       {"type":"filters.label_duplicates","dimensions":"Classification"},
                       {"type":"filters.assign","value":"UserData = Duplicate"},
                       {"type":"writers.las","filename":"out.las"}])";
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        ASSERT_EQ(plan.summary().residentRegions, 1U);
        ASSERT_EQ(plan.summary().indexBuilds,
                  consumer == Consumer::Assignment ? 0U : 1U);
        const auto boundaryId = [&](pdg::ResidencyBoundaryKind kind)
        {
            const auto position =
                std::find_if(plan.summary().residencyBoundaries.begin(),
                             plan.summary().residencyBoundaries.end(),
                             [&](const pdg::ResidencyBoundary& boundary)
                             { return boundary.kind == kind; });
            EXPECT_NE(position, plan.summary().residencyBoundaries.end());
            return static_cast<std::size_t>(std::distance(
                plan.summary().residencyBoundaries.begin(), position));
        };
        const std::size_t upload =
            boundaryId(pdg::ResidencyBoundaryKind::Upload);
        const std::size_t spill = boundaryId(pdg::ResidencyBoundaryKind::Spill);
        pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions,
                                                       1U << 30U, 16U);
        const std::array<std::size_t, 1> selectedRegions{0U};
        scope.preflight(*table.layout(), view->size(), selectedRegions);
        pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
        EXPECT_EQ(context.schedule().itemCount, 0U);
        EXPECT_EQ(context.schedule().tileCount, 0U);
        EXPECT_EQ(context.schedule().activeLaneCount, 0U);
        context.enterBoundary(
            *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
            0U, false);
        context.beginDelegatedRegion(*view, 0U);

        pdg::LabelDuplicatesProgram label;
        label.dimensions = {
            pdg::DimensionId(pdg::StandardDimension::Classification)};
        pdal::pdg_detail::CudaNeighborhoodRegion labelRegion;
        labelRegion.id = 1U;
        labelRegion.maximumNeighbors =
            consumer == Consumer::Assignment ? 0U : 4U;
        labelRegion.indexRequired = consumer != Consumer::Assignment;
        labelRegion.last = false;
        ASSERT_TRUE(pdal::pdg_detail::tryCudaLabelDuplicatesColumn(
            *view, label, dimensions, labelRegion, /*requireCuda=*/true));

        pdal::pdg_detail::CudaNeighborhoodRegion terminal = labelRegion;
        terminal.last = true;
        if (consumer == Consumer::NnDistance)
        {
            ASSERT_TRUE(pdal::pdg_detail::tryCudaNnDistanceColumns(
                *view, 4U, terminal, pdg::KnnDistanceMode::Kth,
                /*requireCuda=*/true));
        }
        else if (consumer == Consumer::Normal)
        {
            std::shared_ptr<const pdal::pdg_detail::CudaNeighborhoodResults>
                results;
            ASSERT_TRUE(pdal::pdg_detail::tryCudaNormalColumns(
                *view, 4U, terminal, /*alwaysUp=*/true, results,
                /*requireCuda=*/true));
            ASSERT_TRUE(results);
            EXPECT_TRUE(results->status.empty());
        }
        else
        {
            const std::array<std::string, 1> specifications{
                "UserData = Duplicate"};
            const pdg::AssignProgram assignment =
                pdg::compileAssignments(specifications, dimensions);
            ASSERT_TRUE(pdal::pdg_detail::tryCudaResidentAssignments(
                *view, terminal, assignment, /*requireCuda=*/true));
        }
        context.endDelegatedRegion(*view, 0U);
        context.enterBoundary(
            *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
            0U, false);
        EXPECT_TRUE(view->empty());
    }
}
