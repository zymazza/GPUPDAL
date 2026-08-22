#include "src/pdal/PdgResidentContext.hpp"

#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Placement.hpp>
#include <pdg/Plan.hpp>
#include <pdg/stages/Assign.hpp>

#include <pdal/Dimension.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

constexpr std::size_t PointCount = 11U;
constexpr std::size_t TileCapacity = 4U;
constexpr std::size_t ResidentBudgetBytes = 64U * 1024U * 1024U;

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

pdg::DimensionType toPdgType(pdal::Dimension::Type type)
{
    using PdalType = pdal::Dimension::Type;
    switch (type)
    {
    case PdalType::Signed8:
        return pdg::DimensionType::Signed8;
    case PdalType::Signed16:
        return pdg::DimensionType::Signed16;
    case PdalType::Signed32:
        return pdg::DimensionType::Signed32;
    case PdalType::Signed64:
        return pdg::DimensionType::Signed64;
    case PdalType::Unsigned8:
        return pdg::DimensionType::Unsigned8;
    case PdalType::Unsigned16:
        return pdg::DimensionType::Unsigned16;
    case PdalType::Unsigned32:
        return pdg::DimensionType::Unsigned32;
    case PdalType::Unsigned64:
        return pdg::DimensionType::Unsigned64;
    case PdalType::Float:
        return pdg::DimensionType::Float;
    case PdalType::Double:
        return pdg::DimensionType::Double;
    case PdalType::None:
        return pdg::DimensionType::None;
    }
    return pdg::DimensionType::None;
}

bool stageWrites(const pdg::PlannedStage& stage, pdg::DimensionId id)
{
    return std::find(stage.descriptor.writes.begin(),
                     stage.descriptor.writes.end(),
                     id) != stage.descriptor.writes.end();
}

bool stageReads(const pdg::PlannedStage& stage, pdg::DimensionId id)
{
    return std::find(stage.descriptor.reads.begin(),
                     stage.descriptor.reads.end(),
                     id) != stage.descriptor.reads.end();
}

std::vector<pdg::PackedPointColumn>
packedColumnsForRegion(const pdg::Plan& plan,
                       const pdg::DimensionRegistry& dimensions,
                       const pdal::PointView& view, std::size_t residentRegion)
{
    std::vector<pdg::PackedPointColumn> columns;
    columns.reserve(view.dimTypes().size());
    for (const pdal::DimType& dim : view.dimTypes())
    {
        const std::string name = view.layout()->dimName(dim.m_id);
        const pdg::DimensionDefinition& definition = dimensions.require(name);
        const bool used =
            std::any_of(plan.stages().begin(), plan.stages().end(),
                        [&](const pdg::PlannedStage& stage)
                        {
                            return stage.residentRegion == residentRegion &&
                                   (stageReads(stage, definition.id) ||
                                    stageWrites(stage, definition.id));
                        });
        if (!used)
            continue;
        const bool written =
            std::any_of(plan.stages().begin(), plan.stages().end(),
                        [&](const pdg::PlannedStage& stage)
                        {
                            return stage.residentRegion == residentRegion &&
                                   stageWrites(stage, definition.id);
                        });
        columns.push_back({definition.id, toPdgType(dim.m_type),
                           view.layout()->dimOffset(dim.m_id), written});
    }
    return columns;
}

std::size_t boundaryId(const pdg::Plan& plan, pdg::ResidencyBoundaryKind kind,
                       std::size_t producer, std::size_t consumer)
{
    const auto position =
        std::find_if(plan.summary().residencyBoundaries.begin(),
                     plan.summary().residencyBoundaries.end(),
                     [&](const pdg::ResidencyBoundary& boundary)
                     {
                         return boundary.kind == kind &&
                                boundary.producer == producer &&
                                boundary.consumer == consumer;
                     });
    EXPECT_NE(position, plan.summary().residencyBoundaries.end());
    return static_cast<std::size_t>(
        std::distance(plan.summary().residencyBoundaries.begin(), position));
}

const pdg::ExecutionEvent&
requireEvent(const pdg::ExecutionStatsSnapshot& stats,
             pdg::ExecutionEventKind kind, std::size_t id)
{
    const auto position =
        std::find_if(stats.events.begin(), stats.events.end(),
                     [&](const pdg::ExecutionEvent& event)
                     { return event.kind == kind && event.regionId == id; });
    EXPECT_NE(position, stats.events.end());
    return *position;
}

std::size_t eventIndex(pdg::ExecutionEventKind kind)
{
    return static_cast<std::size_t>(kind);
}

std::size_t
physicalColumnUnionBytes(std::span<const pdg::PackedPointColumn> columns)
{
    std::vector<pdg::DimensionId> seen;
    std::size_t bytes = 0U;
    for (const pdg::PackedPointColumn& column : columns)
    {
        if (std::find(seen.begin(), seen.end(), column.id) != seen.end())
            continue;
        seen.push_back(column.id);
        bytes += pdg::dimensionTypeSize(column.physicalType);
    }
    return bytes;
}

std::size_t
writtenPhysicalColumnBytes(std::span<const pdg::PackedPointColumn> columns)
{
    std::vector<pdg::DimensionId> seen;
    std::size_t bytes = 0U;
    for (const pdg::PackedPointColumn& column : columns)
    {
        if (!column.written ||
            std::find(seen.begin(), seen.end(), column.id) != seen.end())
            continue;
        seen.push_back(column.id);
        bytes += pdg::dimensionTypeSize(column.physicalType);
    }
    return bytes;
}

} // unnamed namespace

TEST(ResidentExecutionContextCuda,
     PreservesSeededHostBoundaryAndAccountsForFullPhysicalRecords)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Scratch = Intensity"},
      {"type":"filters.randomize","seed":17},
      {"type":"filters.assign",
       "value":["Classification = Scratch", "UserData = Scratch + 1"]},
      {"type":"writers.las","filename":"out.las"}
    ]})";

    pdg::DimensionRegistry dimensions;
    // PDAL's filters.assign creates its destination dimensions as doubles.
    // The plan registry must match that physical layout exactly.
    dimensions.registerCustom("Scratch", pdg::DimensionType::Double);
    dimensions.registerCustom("SignedScratch", pdg::DimensionType::Signed32);
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    ASSERT_EQ(plan.stages().size(), 5U);
    ASSERT_TRUE(std::holds_alternative<pdg::AssignProgram>(
        plan.stages().at(1U).payload));
    ASSERT_TRUE(std::holds_alternative<pdg::AssignProgram>(
        plan.stages().at(3U).payload));

    const std::size_t initialUpload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload, 0U, 1U);
    const std::size_t fallbackSpill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill, 1U, 2U);
    const std::size_t fallbackUpload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload, 2U, 3U);
    const std::size_t finalSpill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill, 3U, 4U);
    const auto& boundaries = plan.summary().residencyBoundaries;
    ASSERT_TRUE(boundaries.at(fallbackSpill).fallback);
    ASSERT_TRUE(boundaries.at(fallbackUpload).fallback);
    ASSERT_TRUE(boundaries.at(fallbackSpill).requiresFullPointRecord);
    ASSERT_TRUE(boundaries.at(fallbackUpload).requiresFullPointRecord);

    const std::size_t firstRegion = plan.stages().at(1U).residentRegion;
    const std::size_t secondRegion = plan.stages().at(3U).residentRegion;
    ASSERT_NE(firstRegion, pdg::NoResidentRegion);
    ASSERT_NE(secondRegion, pdg::NoResidentRegion);
    ASSERT_NE(firstRegion, secondRegion);

    pdal::PointTable table;
    const auto layout = table.layout();
    layout->registerDim(pdal::Dimension::Id::Intensity,
                        pdal::Dimension::Type::Unsigned16);
    layout->registerDim(pdal::Dimension::Id::Classification,
                        pdal::Dimension::Type::Unsigned8);
    layout->registerDim(pdal::Dimension::Id::UserData,
                        pdal::Dimension::Type::Unsigned8);
    layout->registerDim(pdal::Dimension::Id::PointSourceId,
                        pdal::Dimension::Type::Unsigned16);
    const pdal::Dimension::Id scratch =
        layout->registerOrAssignDim("Scratch", pdal::Dimension::Type::Double);
    // This column is intentionally absent from both point programs. Its
    // signed physical type ensures the full-record boundary path preserves
    // columns outside the programs' declared read/write sets.
    const pdal::Dimension::Id signedScratch = layout->registerOrAssignDim(
        "SignedScratch", pdal::Dimension::Type::Signed32);
    table.finalize();
    pdal::PointView view(table);
    for (pdal::PointId point = 0; point < PointCount; ++point)
    {
        const auto value = static_cast<std::uint16_t>(10U + point);
        view.setField(pdal::Dimension::Id::Intensity, point, value);
        view.setField(pdal::Dimension::Id::Classification, point,
                      std::uint8_t{0});
        view.setField(pdal::Dimension::Id::UserData, point, std::uint8_t{0});
        view.setField(pdal::Dimension::Id::PointSourceId, point,
                      static_cast<std::uint16_t>(1000U + point));
        view.setField(scratch, point, -1.0);
        view.setField(signedScratch, point,
                      static_cast<std::int32_t>(-100 - point));
    }
    ASSERT_EQ(view.size(), PointCount);

    const std::vector<pdg::PackedPointColumn> firstColumns =
        packedColumnsForRegion(plan, dimensions, view, firstRegion);
    const std::vector<pdg::PackedPointColumn> secondColumns =
        packedColumnsForRegion(plan, dimensions, view, secondRegion);
    ASSERT_LT(firstColumns.size(), view.dimTypes().size());
    ASSERT_LT(secondColumns.size(), view.dimTypes().size());
    EXPECT_EQ(boundaries.at(fallbackSpill).repackBytesPerPoint,
              writtenPhysicalColumnBytes(firstColumns));
    EXPECT_EQ(boundaries.at(finalSpill).repackBytesPerPoint,
              writtenPhysicalColumnBytes(secondColumns));
    EXPECT_GT(boundaries.at(fallbackSpill).bytesPerPoint,
              boundaries.at(fallbackSpill).repackBytesPerPoint);

    const auto& firstProgram =
        std::get<pdg::AssignProgram>(plan.stages().at(1U).payload);
    const auto& secondProgram =
        std::get<pdg::AssignProgram>(plan.stages().at(3U).payload);
    pdg::ExecutionObservationScope observation;
    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, ResidentBudgetBytes, TileCapacity);
    auto& context = scope.context();
    EXPECT_EQ(&pdal::pdg_detail::requireResidentExecutionContext(), &context);
    const std::array selectedRegions{firstRegion, secondRegion};
    scope.preflight(*layout, PointCount, selectedRegions);

    // Preflight creates the same bounded two-lane plan that the marker path
    // will consume.  This assertion is intentionally before the first upload
    // marker: an allocation-probe failure must be a host-fallback decision,
    // not a partially executed resident region.
    const pdg::TiledSchedule& preflightSchedule = context.schedule();
    const std::size_t expectedBytesPerLane =
        TileCapacity * view.pointSize() +
        plan.estimatedDeviceBytes(TileCapacity);
    EXPECT_EQ(preflightSchedule.tileCount, 3U);
    EXPECT_EQ(preflightSchedule.configuredLaneCount, 2U);
    EXPECT_EQ(preflightSchedule.activeLaneCount, 2U);
    EXPECT_EQ(preflightSchedule.laneReuseCount, 1U);
    EXPECT_EQ(preflightSchedule.peakLaneBytes,
              preflightSchedule.activeLaneCount * expectedBytesPerLane);

    pdg::PlacementRequest placementRequest;
    placementRequest.stageInputPointCounts.assign(plan.stages().size(),
                                                  PointCount);
    placementRequest.stageOutputPointCounts.assign(plan.stages().size(),
                                                   PointCount);
    placementRequest.stagePointCapacities.assign(plan.stages().size(),
                                                 TileCapacity);
    placementRequest.stageCosts.resize(plan.stages().size());
    for (std::size_t stageId : {1U, 3U})
        placementRequest.stageCosts[stageId] = {
            .hostNanosecondsPerPoint = 1000.0,
            .deviceNanosecondsPerPoint = 1.0,
            .calibrated = true};
    placementRequest.executorLaneCount = 2U;
    placementRequest.deviceMemoryBudgetBytes = ResidentBudgetBytes;
    for (std::size_t boundaryId = 0; boundaryId < boundaries.size();
         ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary = boundaries[boundaryId];
        placementRequest.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = view.pointSize(),
             .packingBytesPerPoint =
                 boundary.kind == pdg::ResidencyBoundaryKind::Upload
                     ? view.pointSize()
                     : boundary.repackBytesPerPoint,
             .deviceStagingBytesPerPoint = view.pointSize()});
    }
    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, placementRequest, {});
    ASSERT_EQ(placement.selectedRegionCount, 2U);
    EXPECT_EQ(placement.configuredDeviceLaneCount,
              preflightSchedule.configuredLaneCount);
    EXPECT_EQ(placement.activeDeviceLaneCount,
              preflightSchedule.activeLaneCount);
    EXPECT_EQ(placement.peakDeviceBytes, preflightSchedule.peakLaneBytes);

    const auto executeRegion =
        [&](std::size_t region, std::span<const pdg::PackedPointColumn> columns,
            const pdg::AssignProgram& program)
    {
        context.beginRegion(view, region, columns);
        ASSERT_EQ(context.tileCount(), 3U);
        for (std::size_t tile = 0; tile < context.tileCount(); ++tile)
        {
            pdg::PointBatch& batch = context.acquireTile(view, tile);
            context.beginStage(tile, 0U);
            pdg::executeAssign(batch, program);
            context.endStage(tile, 0U);
            context.submitTile(view, tile, batch);
        }
        context.endRegion(view, region);
    };

    context.enterBoundary(view, initialUpload,
                          pdal::pdg_detail::ResidentBoundaryDirection::Upload,
                          firstRegion,
                          boundaries.at(initialUpload).requiresFullPointRecord);
    executeRegion(firstRegion, firstColumns, firstProgram);
    context.enterBoundary(
        view, fallbackSpill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        firstRegion, boundaries.at(fallbackSpill).requiresFullPointRecord);

    std::vector<std::size_t> expectedOrder(PointCount);
    std::iota(expectedOrder.begin(), expectedOrder.end(), 0U);
    std::mt19937 expectedRandom(17U);
    std::shuffle(expectedOrder.begin(), expectedOrder.end(), expectedRandom);
    std::mt19937 upstreamRandom(17U);
    std::shuffle(view.begin(), view.end(), upstreamRandom);

    context.enterBoundary(
        view, fallbackUpload,
        pdal::pdg_detail::ResidentBoundaryDirection::Upload, secondRegion,
        boundaries.at(fallbackUpload).requiresFullPointRecord);
    executeRegion(secondRegion, secondColumns, secondProgram);
    context.enterBoundary(
        view, finalSpill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        secondRegion, boundaries.at(finalSpill).requiresFullPointRecord);

    for (pdal::PointId point = 0; point < PointCount; ++point)
    {
        const std::size_t source = expectedOrder.at(point);
        const auto intensity = static_cast<std::uint16_t>(10U + source);
        EXPECT_EQ(view.getFieldAs<std::uint16_t>(pdal::Dimension::Id::Intensity,
                                                 point),
                  intensity)
            << point;
        EXPECT_EQ(view.getFieldAs<std::uint16_t>(
                      pdal::Dimension::Id::PointSourceId, point),
                  static_cast<std::uint16_t>(1000U + source))
            << point;
        EXPECT_EQ(view.getFieldAs<double>(scratch, point),
                  static_cast<double>(intensity))
            << point;
        EXPECT_EQ(view.getFieldAs<std::int32_t>(signedScratch, point),
                  static_cast<std::int32_t>(-100 - source))
            << point;
        EXPECT_EQ(view.getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point),
                  static_cast<std::uint8_t>(intensity))
            << point;
        EXPECT_EQ(
            view.getFieldAs<std::uint8_t>(pdal::Dimension::Id::UserData, point),
            static_cast<std::uint8_t>(intensity + 1U))
            << point;
    }

    const pdg::TiledSchedule& schedule = context.schedule();
    EXPECT_EQ(schedule.configuredLaneCount, 2U);
    EXPECT_EQ(schedule.activeLaneCount, 2U);
    EXPECT_EQ(schedule.laneReuseCount, 1U);

    EXPECT_GT(schedule.observedPeakLaneBytes, 0U);
    EXPECT_LE(schedule.observedPeakLaneBytes, schedule.peakLaneBytes);

    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    const std::size_t fullRecordBytes = view.pointSize() * PointCount;
    const auto expectTotal =
        [&](pdg::ExecutionEventKind kind, std::size_t count, std::size_t bytes)
    {
        const pdg::ExecutionEventTotals& total =
            stats.totals.at(eventIndex(kind));
        EXPECT_EQ(total.count, count);
        EXPECT_EQ(total.bytes, bytes);
    };
    expectTotal(pdg::ExecutionEventKind::BoundaryUpload, 2U,
                2U * fullRecordBytes);
    expectTotal(pdg::ExecutionEventKind::BoundarySpill, 2U,
                2U * fullRecordBytes);
    expectTotal(pdg::ExecutionEventKind::FallbackBoundary, 2U,
                2U * fullRecordBytes);
    expectTotal(pdg::ExecutionEventKind::HostToDevice, 2U,
                2U * fullRecordBytes);
    expectTotal(pdg::ExecutionEventKind::DeviceToHost, 2U,
                2U * fullRecordBytes);
    expectTotal(pdg::ExecutionEventKind::DeviceRegionBegin, 2U, 0U);
    expectTotal(pdg::ExecutionEventKind::DeviceRegionEnd, 2U, 0U);
    const std::size_t firstSpillPackingBytes =
        PointCount * writtenPhysicalColumnBytes(firstColumns);
    const std::size_t secondSpillPackingBytes =
        PointCount * writtenPhysicalColumnBytes(secondColumns);
    EXPECT_EQ(stats.totals.at(eventIndex(pdg::ExecutionEventKind::HostToDevice))
                  .packingBytes,
              2U * fullRecordBytes);
    EXPECT_EQ(stats.totals.at(eventIndex(pdg::ExecutionEventKind::DeviceToHost))
                  .packingBytes,
              firstSpillPackingBytes + secondSpillPackingBytes);

    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::BoundaryUpload,
                           initialUpload)
                  .bytes,
              fullRecordBytes);
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::BoundarySpill,
                           fallbackSpill)
                  .bytes,
              fullRecordBytes);
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::BoundaryUpload,
                           fallbackUpload)
                  .bytes,
              fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::BoundarySpill, finalSpill)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::HostToDevice, firstRegion)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::HostToDevice, firstRegion)
            .packingBytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, firstRegion)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, firstRegion)
            .packingBytes,
        firstSpillPackingBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::HostToDevice, secondRegion)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::HostToDevice, secondRegion)
            .packingBytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, secondRegion)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, secondRegion)
            .packingBytes,
        secondSpillPackingBytes);
}

TEST(ResidentExecutionContextCuda,
     ExecutesADeclaredCardinalityChangingExpressionRegion)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Scratch = Intensity"},
      {"type":"filters.expression","expression":"Intensity <= 16"},
      {"type":"filters.assign",
       "value":["Classification = Scratch", "UserData = Scratch + 1"]},
      {"type":"writers.las","filename":"out.las"}
    ]})";

    pdg::DimensionRegistry dimensions;
    dimensions.registerCustom("Scratch", pdg::DimensionType::Double);
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.stages().size(), 5U);
    ASSERT_TRUE(std::holds_alternative<pdg::PredicateProgram>(
        plan.stages().at(2U).payload));
    ASSERT_FALSE(plan.stages().at(2U).descriptor.fusion.cardinalityPreserving);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    ASSERT_EQ(plan.stages().at(2U).residentRegion, region);
    ASSERT_EQ(plan.stages().at(3U).residentRegion, region);

    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload, 0U, 1U);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill, 3U, 4U);
    const auto& boundaries = plan.summary().residencyBoundaries;

    pdal::PointTable table;
    const auto layout = table.layout();
    layout->registerDim(pdal::Dimension::Id::Intensity,
                        pdal::Dimension::Type::Unsigned16);
    layout->registerDim(pdal::Dimension::Id::Classification,
                        pdal::Dimension::Type::Unsigned8);
    layout->registerDim(pdal::Dimension::Id::UserData,
                        pdal::Dimension::Type::Unsigned8);
    const pdal::Dimension::Id scratch =
        layout->registerOrAssignDim("Scratch", pdal::Dimension::Type::Double);
    table.finalize();
    pdal::PointView view(table);
    for (pdal::PointId point = 0; point < PointCount; ++point)
    {
        view.setField(pdal::Dimension::Id::Intensity, point,
                      static_cast<std::uint16_t>(10U + point));
        view.setField(pdal::Dimension::Id::Classification, point,
                      std::uint8_t{0});
        view.setField(pdal::Dimension::Id::UserData, point, std::uint8_t{0});
        view.setField(scratch, point, -1.0);
    }
    ASSERT_EQ(view.size(), PointCount);
    // Intensity <= 16 keeps source points 0..6: the final tile of the
    // three-tile schedule drops every point, covering the empty-survivor
    // case.
    constexpr std::size_t SurvivorCount = 7U;

    const std::vector<pdg::PackedPointColumn> columns =
        packedColumnsForRegion(plan, dimensions, view, region);
    const auto& firstProgram =
        std::get<pdg::AssignProgram>(plan.stages().at(1U).payload);
    const auto& predicate =
        std::get<pdg::PredicateProgram>(plan.stages().at(2U).payload);
    const auto& secondProgram =
        std::get<pdg::AssignProgram>(plan.stages().at(3U).payload);

    pdg::ExecutionObservationScope observation;
    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, ResidentBudgetBytes, TileCapacity);
    auto& context = scope.context();
    const std::array selectedRegions{region};
    scope.preflight(*layout, PointCount, selectedRegions);

    // The keep mask is planner-owned lane storage: one byte per tile point on
    // top of the packed tile and resident columns.
    const pdg::TiledSchedule& preflightSchedule = context.schedule();
    const std::size_t expectedBytesPerLane =
        TileCapacity * view.pointSize() +
        plan.estimatedDeviceBytes(TileCapacity) + TileCapacity;
    EXPECT_EQ(preflightSchedule.tileCount, 3U);
    EXPECT_EQ(preflightSchedule.configuredLaneCount, 2U);
    EXPECT_EQ(preflightSchedule.activeLaneCount, 2U);
    EXPECT_EQ(preflightSchedule.peakLaneBytes,
              preflightSchedule.activeLaneCount * expectedBytesPerLane);

    pdg::PlacementRequest placementRequest;
    placementRequest.stageInputPointCounts.assign(plan.stages().size(),
                                                  PointCount);
    placementRequest.stageOutputPointCounts.assign(plan.stages().size(),
                                                   PointCount);
    placementRequest.stagePointCapacities.assign(plan.stages().size(),
                                                 TileCapacity);
    placementRequest.stageCosts.resize(plan.stages().size());
    for (std::size_t stageId : {1U, 2U, 3U})
        placementRequest.stageCosts[stageId] = {
            .hostNanosecondsPerPoint = 1000.0,
            .deviceNanosecondsPerPoint = 1.0,
            .calibrated = true};
    placementRequest.executorLaneCount = 2U;
    placementRequest.deviceMemoryBudgetBytes = ResidentBudgetBytes;
    for (std::size_t id = 0; id < boundaries.size(); ++id)
    {
        const bool spillBoundary =
            boundaries[id].kind == pdg::ResidencyBoundaryKind::Spill;
        placementRequest.boundaryExecutionFacts.push_back(
            {.boundaryId = id,
             .transferBytesPerPoint =
                 view.pointSize() + (spillBoundary ? 1U : 0U),
             .packingBytesPerPoint = spillBoundary
                                         ? boundaries[id].repackBytesPerPoint
                                         : view.pointSize(),
             .deviceStagingBytesPerPoint =
                 view.pointSize() + (spillBoundary ? 1U : 0U)});
    }
    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, placementRequest, {});
    ASSERT_EQ(placement.selectedRegionCount, 1U);
    EXPECT_EQ(placement.configuredDeviceLaneCount,
              preflightSchedule.configuredLaneCount);
    EXPECT_EQ(placement.activeDeviceLaneCount,
              preflightSchedule.activeLaneCount);
    EXPECT_EQ(placement.peakDeviceBytes, preflightSchedule.peakLaneBytes);

    context.enterBoundary(
        view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region, boundaries.at(upload).requiresFullPointRecord);
    // A declared cardinality change requires its survivor output view; the
    // cardinality-preserving call shape must fail closed before execution.
    EXPECT_THROW(context.beginRegion(view, region, columns),
                 std::invalid_argument);
    const pdal::PointViewPtr output = view.makeNew();
    context.beginRegion(view, region, columns, output.get());
    ASSERT_EQ(context.tileCount(), 3U);
    for (std::size_t tile = 0; tile < context.tileCount(); ++tile)
    {
        pdg::PointBatch& batch = context.acquireTile(view, tile);
        context.beginStage(tile, 0U);
        pdg::executeAssign(batch, firstProgram);
        context.endStage(tile, 0U);
        context.beginStage(tile, 1U);
        pdg::evaluatePredicate(batch, predicate, context.tileKeepMask(tile));
        context.endStage(tile, 1U);
        context.beginStage(tile, 2U);
        pdg::executeAssign(batch, secondProgram);
        context.endStage(tile, 2U);
        context.submitTile(view, tile, batch);
    }
    context.endRegion(view, region);
    EXPECT_EQ(context.observedOutputPointCount(), SurvivorCount);
    ASSERT_EQ(output->size(), SurvivorCount);
    context.enterBoundary(*output, spill,
                          pdal::pdg_detail::ResidentBoundaryDirection::Spill,
                          region, boundaries.at(spill).requiresFullPointRecord);

    for (pdal::PointId point = 0; point < SurvivorCount; ++point)
    {
        const auto intensity = static_cast<std::uint16_t>(10U + point);
        EXPECT_EQ(output->getFieldAs<std::uint16_t>(
                      pdal::Dimension::Id::Intensity, point),
                  intensity)
            << point;
        // Scratch is a dead intermediate: written in-region, consumed
        // in-region, and ignored by the writer. The planner releases it
        // before the spill, so it is deliberately never published.
        EXPECT_EQ(output->getFieldAs<double>(scratch, point), -1.0) << point;
        EXPECT_EQ(output->getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point),
                  static_cast<std::uint8_t>(intensity))
            << point;
        EXPECT_EQ(output->getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::UserData, point),
                  static_cast<std::uint8_t>(intensity + 1U))
            << point;
    }
    // Dropped points leave the pipeline at the declared cardinality change,
    // so their source rows publish nothing.
    for (pdal::PointId point = SurvivorCount; point < PointCount; ++point)
    {
        EXPECT_EQ(view.getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point),
                  std::uint8_t{0})
            << point;
        EXPECT_EQ(
            view.getFieldAs<std::uint8_t>(pdal::Dimension::Id::UserData, point),
            std::uint8_t{0})
            << point;
        EXPECT_EQ(view.getFieldAs<double>(scratch, point), -1.0) << point;
    }

    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    const std::size_t fullRecordBytes = view.pointSize() * PointCount;
    const std::size_t maskBytes = PointCount;
    EXPECT_EQ(
        requireEvent(stats, pdg::ExecutionEventKind::BoundaryUpload, upload)
            .bytes,
        fullRecordBytes);
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::BoundarySpill, spill)
                  .bytes,
              fullRecordBytes + maskBytes);
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::HostToDevice, region)
                  .bytes,
              fullRecordBytes);
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, region)
                  .bytes,
              fullRecordBytes + maskBytes);
    // Spill repacking is planner-owned: only the released-and-written spill
    // set (Classification and UserData) is packed, never the dead Scratch
    // intermediate.
    EXPECT_EQ(boundaries.at(spill).repackBytesPerPoint, 2U);
    EXPECT_LT(boundaries.at(spill).repackBytesPerPoint,
              writtenPhysicalColumnBytes(columns));
    EXPECT_EQ(requireEvent(stats, pdg::ExecutionEventKind::DeviceToHost, region)
                  .packingBytes,
              PointCount * boundaries.at(spill).repackBytesPerPoint);

    const pdg::TiledSchedule& schedule = context.schedule();
    EXPECT_GT(schedule.observedPeakLaneBytes, 0U);
    EXPECT_LE(schedule.observedPeakLaneBytes, schedule.peakLaneBytes);
}

TEST(ResidentExecutionContextCuda, RejectsPhysicalLayoutBeforeBoundaryExecution)
{
    // The type mismatch is rejected by bindLayout(), before lane allocation,
    // so this test is deterministic even on a CUDA build with no visible GPU.
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign","value":"Classification = Intensity"},
             "output.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);

    pdal::PointTable table;
    const auto layout = table.layout();
    // The planner's standard Intensity type is Unsigned16.  A Float layout is
    // invalid for exact resident packing and must fail in preflight.
    layout->registerDim(pdal::Dimension::Id::Intensity,
                        pdal::Dimension::Type::Float);
    layout->registerDim(pdal::Dimension::Id::Classification,
                        pdal::Dimension::Type::Unsigned8);
    table.finalize();

    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, ResidentBudgetBytes, TileCapacity);
    const std::array selectedRegions{region};
    EXPECT_THROW(scope.preflight(*layout, PointCount, selectedRegions),
                 std::invalid_argument);

    // No marker was entered and no tiled execution schedule was committed.
    const pdg::TiledSchedule& schedule = scope.context().schedule();
    EXPECT_EQ(schedule.tileCount, 0U);
    EXPECT_EQ(schedule.activeLaneCount, 0U);
}

TEST(ResidentExecutionContextCuda, ExecutesThePlannedLiveColumnPeak)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    // This is the D0051 three-assignment liveness pattern.  TmpA dies before
    // TmpB is materialized, so the planner's peak is 16 B/point even though
    // the union of region columns is 19 B/point.  A resident lane must obey
    // that lifetime plan: materializing the entire region up front silently
    // exceeds the scheduler's claimed VRAM peak.
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign","value":"TmpA = Intensity * 2"},
             {"type":"filters.assign","value":"TmpB = TmpA + 1"},
             {"type":"filters.assign","value":"Classification = TmpB"},
             "output.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().bytesPerPoint, 19U);
    ASSERT_EQ(plan.summary().peakDeviceColumnBytesPerPoint, 16U);

    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    ASSERT_EQ(plan.stages().at(2U).residentRegion, region);
    ASSERT_EQ(plan.stages().at(3U).residentRegion, region);
    const std::size_t initialUpload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload, 0U, 1U);
    const std::size_t finalSpill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill, 3U, 4U);

    pdal::PointTable table;
    const auto layout = table.layout();
    layout->registerDim(pdal::Dimension::Id::Intensity,
                        pdal::Dimension::Type::Unsigned16);
    layout->registerDim(pdal::Dimension::Id::Classification,
                        pdal::Dimension::Type::Unsigned8);
    const pdal::Dimension::Id tmpA =
        layout->registerOrAssignDim("TmpA", pdal::Dimension::Type::Double);
    const pdal::Dimension::Id tmpB =
        layout->registerOrAssignDim("TmpB", pdal::Dimension::Type::Double);
    table.finalize();
    pdal::PointView view(table);
    for (pdal::PointId point = 0; point < PointCount; ++point)
    {
        view.setField(pdal::Dimension::Id::Intensity, point,
                      static_cast<std::uint16_t>(point));
        view.setField(pdal::Dimension::Id::Classification, point,
                      std::uint8_t{0});
        view.setField(tmpA, point, 0.0);
        view.setField(tmpB, point, 0.0);
    }

    const std::vector<pdg::PackedPointColumn> columns =
        packedColumnsForRegion(plan, dimensions, view, region);
    ASSERT_EQ(physicalColumnUnionBytes(columns), 19U);

    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, ResidentBudgetBytes, TileCapacity);
    auto& context = scope.context();
    context.enterBoundary(view, initialUpload,
                          pdal::pdg_detail::ResidentBoundaryDirection::Upload,
                          region,
                          plan.summary()
                              .residencyBoundaries.at(initialUpload)
                              .requiresFullPointRecord);
    context.beginRegion(view, region, columns);

    for (std::size_t tile = 0; tile < context.tileCount(); ++tile)
    {
        pdg::PointBatch& batch = context.acquireTile(view, tile);
        for (std::size_t stage = 0; stage < 3U; ++stage)
        {
            context.beginStage(tile, stage);
            const auto& program = std::get<pdg::AssignProgram>(
                plan.stages().at(stage + 1U).payload);
            pdg::executeAssign(batch, program);
            context.endStage(tile, stage);
        }
        context.submitTile(view, tile, batch);
    }
    context.endRegion(view, region);
    context.enterBoundary(view, finalSpill,
                          pdal::pdg_detail::ResidentBoundaryDirection::Spill,
                          region,
                          plan.summary()
                              .residencyBoundaries.at(finalSpill)
                              .requiresFullPointRecord);

    const pdg::TiledSchedule& schedule = context.schedule();
    ASSERT_EQ(schedule.activeLaneCount, 2U);
    const std::size_t plannedPeakBytes =
        schedule.activeLaneCount * TileCapacity *
        (view.pointSize() + plan.summary().peakDeviceColumnBytesPerPoint);
    const std::size_t eagerUnionBytes =
        schedule.activeLaneCount * TileCapacity *
        (view.pointSize() + physicalColumnUnionBytes(columns));
    EXPECT_EQ(schedule.peakLaneBytes, plannedPeakBytes);
    EXPECT_LT(plannedPeakBytes, eagerUnionBytes);
    EXPECT_GT(schedule.observedPeakLaneBytes, 0U);
    EXPECT_LE(schedule.observedPeakLaneBytes, plannedPeakBytes);
    EXPECT_LT(schedule.observedPeakLaneBytes, eagerUnionBytes);
    for (pdal::PointId point = 0; point < PointCount; ++point)
        EXPECT_EQ(view.getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point),
                  static_cast<std::uint8_t>(point * 2U + 1U));
}
