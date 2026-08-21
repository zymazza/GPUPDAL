#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include "src/pdal/PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>

namespace
{
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

TEST(PdgElmFilter, ExecutesThroughThePlannerSelectedStandaloneGridLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.elm","cell":10.0,"class":18,
       "threshold":1.0},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification});
    pdal::PointViewPtr view(new pdal::PointView(table));
    const std::array<double, 3> z{0.0, 2.0, 2.5};
    for (pdal::PointId point = 0U; point < z.size(); ++point)
    {
        view->setField(Id::X, point, static_cast<double>(point) * 0.1);
        view->setField(Id::Y, point, 0.0);
        view->setField(Id::Z, point, z[point]);
        view->setField(Id::Classification, point, std::uint8_t{5U});
    }

    constexpr std::size_t Budget = 64U * 1024U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    EXPECT_EQ(context.schedule().tileCount, 1U);
    EXPECT_EQ(context.schedule().configuredLaneCount, 1U);
    EXPECT_GE(context.schedule().peakLaneBytes,
              pdg::elmExactDeviceScratchBytes(view->size()));

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridElmStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("cell", 10.0);
    options.add("class", 18);
    options.add("threshold", 1.0);
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

    EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, 0U), 18U);
    EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, 1U), 5U);
    EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, 2U), 5U);
}

TEST(PdgElmFilter, CompletesAnEmptyPlannerSelectedGridLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.elm"},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification});
    pdal::PointViewPtr view(new pdal::PointView(table));
    constexpr std::size_t Budget = 64U * 1024U * 1024U;
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, Budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    EXPECT_EQ(context.schedule().tileCount, 0U);

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridElmStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));
    EXPECT_NO_THROW(context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord));
}

TEST(PdgElmFilter, PreflightAcceptsTheDerivedScratchBudgetExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.elm","cell":25.0},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::array selectedRegions{region};
    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification});
    constexpr std::size_t Points = 1000000U;

    std::size_t exactBudget = 0U;
    {
        pdal::pdg_detail::ResidentExecutionScope probe(
            plan, dimensions, 1024U * 1024U * 1024U, Points);
        probe.preflight(*table.layout(), Points, selectedRegions);
        exactBudget = probe.context().schedule().peakLaneBytes;
    }
    EXPECT_EQ(exactBudget, plan.estimatedDeviceBytes(Points) +
                               pdg::elmExactDeviceScratchBytes(Points));

    {
        pdal::pdg_detail::ResidentExecutionScope accepted(plan, dimensions,
                                                          exactBudget, Points);
        EXPECT_NO_THROW(
            accepted.preflight(*table.layout(), Points, selectedRegions));
    }
    ASSERT_GT(exactBudget, 0U);
    {
        pdal::pdg_detail::ResidentExecutionScope rejected(
            plan, dimensions, exactBudget - 1U, Points);
        EXPECT_THROW(
            rejected.preflight(*table.layout(), Points, selectedRegions),
            std::invalid_argument);
    }
}

TEST(PdgElmFilter, RequiredCudaRejectsAnOversizedFrameBeforePublication)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.elm","cell":1.0},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::array selectedRegions{region};

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification});
    pdal::PointViewPtr view(new pdal::PointView(table));
    view->setField(Id::X, 0U, 0.0);
    view->setField(Id::Y, 0U, 0.0);
    view->setField(Id::Z, 0U, 0.0);
    view->setField(Id::Classification, 0U, std::uint8_t{31U});
    view->setField(Id::X, 1U,
                   static_cast<double>(pdg::ElmExactDeviceMaximumGridCells));
    view->setField(Id::Y, 1U, 0.0);
    view->setField(Id::Z, 1U, 10.0);
    view->setField(Id::Classification, 1U, std::uint8_t{32U});

    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions,
                                                   64U * 1024U * 1024U, 64U);
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridElmStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("cell", 1.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    try
    {
        static_cast<void>(filter->execute(table));
        FAIL() << "oversized required-CUDA ELM unexpectedly executed";
    }
    catch (const std::exception& error)
    {
        EXPECT_STREQ(error.what(),
                     "filters.elm: planner-selected resident elm path was not "
                     "used");
    }
    EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, 0U), 31U);
    EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, 1U), 32U);
}

#if PDG_HAS_CUDA
TEST(ElmDevice, RejectsNonfiniteInputBeforeMutatingClassifications)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> memory = pdg::makeCudaMemoryResource();
    pdg::PointBatch batch(
        1U, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *memory);
    const pdg::DimensionId xId(pdg::StandardDimension::X);
    const pdg::DimensionId yId(pdg::StandardDimension::Y);
    const pdg::DimensionId zId(pdg::StandardDimension::Z);
    const pdg::DimensionId classificationId(
        pdg::StandardDimension::Classification);
    batch.materialize(xId, pdg::DimensionType::Double);
    batch.materialize(yId, pdg::DimensionType::Double);
    batch.materialize(zId, pdg::DimensionType::Double);
    batch.materialize(classificationId, pdg::DimensionType::Unsigned8);
    batch.setSize(1U);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double zero = 0.0;
    const std::uint8_t initialClass = 77U;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(batch.rawData(xId), &nan, sizeof(nan),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(batch.rawData(yId), &zero, sizeof(zero),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(batch.rawData(zId), &zero, sizeof(zero),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(batch.rawData(classificationId),
                                   &initialClass, sizeof(initialClass),
                                   cudaMemcpyHostToDevice, stream));

    EXPECT_THROW(static_cast<void>(pdg::classifyElm(batch, {})),
                 std::invalid_argument);
    std::uint8_t actualClass = 0U;
    PDG_CUDA_CHECK(
        cudaMemcpyAsync(&actualClass, batch.rawData(classificationId),
                        sizeof(actualClass), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actualClass, initialClass);
}
#endif
