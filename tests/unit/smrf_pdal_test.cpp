#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Plan.hpp>

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

TEST(PdgSmrfFilter, RejectsPlannerSelectedUnqualifiedDeviceLifecycle)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.smrf","window":2.0},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
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
              pdg::SmrfExactDeviceMaximumFixedScratchBytes);

    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter =
        factory.createStage(std::string(pdg::HybridSmrfStage));
    ASSERT_NE(filter, nullptr);
    pdal::Options options;
    options.add("window", 2.0);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    EXPECT_THROW(static_cast<void>(filter->execute(table)), std::exception);
}
