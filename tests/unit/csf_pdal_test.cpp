#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
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

TEST(PdgCsfFilter, ExecutesThroughThePlannerSelectedStandaloneGridLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.csf","smooth":false,"step":1.0,
       "iterations":3,"returns":[]},
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
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const pdal::PointId point =
                static_cast<pdal::PointId>(column * 5U + row);
            view->setField(Id::X, point, static_cast<double>(column));
            view->setField(Id::Y, point, static_cast<double>(row));
            view->setField(Id::Z, point, point == 12U ? 5.0 : 0.0);
            view->setField(Id::Classification, point, std::uint8_t{7U});
            view->setField(Id::ReturnNumber, point, std::uint8_t{1U});
            view->setField(Id::NumberOfReturns, point, std::uint8_t{1U});
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
              pdg::CsfExactDeviceMaximumFixedScratchBytes);

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridCsfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("smooth", false);
    options.add("step", 1.0);
    options.add("iterations", 3);
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

    for (pdal::PointId point = 0U; point < view->size(); ++point)
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, point),
                  point == 12U ? 1U : 2U)
            << point;
}

TEST(PdgCsfFilter, CompletesAnEmptyPlannerSelectedGridLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.csf","smooth":false,"iterations":3},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
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
    pdal::Stage* filter = factory.createStage(std::string(pdg::HybridCsfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("smooth", false);
    options.add("iterations", 3);
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

TEST(CsfDevice, RejectsANonfiniteFirstPointBeforeMutatingClassifications)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> memory = pdg::makeCudaMemoryResource();
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch batch(1U, coordinates, dimensions, *memory);
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

    pdg::CsfProgram program;
    program.smooth = false;
    program.iterations = 1;
    EXPECT_THROW(static_cast<void>(pdg::classifyCsf(batch, program)),
                 std::invalid_argument);

    std::uint8_t actualClass = 0U;
    PDG_CUDA_CHECK(
        cudaMemcpyAsync(&actualClass, batch.rawData(classificationId),
                        sizeof(actualClass), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actualClass, initialClass);
}
