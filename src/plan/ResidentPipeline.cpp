#include <pdg/ResidentPipeline.hpp>

#include <pdg/Hybrid.hpp>
#include <pdg/Placement.hpp>
#include <pdg/Plan.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

namespace pdg
{
namespace
{
using Json = nlohmann::json;

bool pointProgramPayload(const StagePayload& payload) noexcept
{
    return std::holds_alternative<FerryProgram>(payload) ||
           std::holds_alternative<AssignProgram>(payload) ||
           std::holds_alternative<PredicateProgram>(payload);
}

Json* pipelineArray(Json& root)
{
    if (root.is_array())
        return &root;
    if (!root.is_object())
        return nullptr;
    const auto position = root.find("pipeline");
    return position != root.end() && position->is_array() ? &*position
                                                          : nullptr;
}

ResidentPipelineRewrite unavailable(std::string reason)
{
    ResidentPipelineRewrite result;
    result.reason = std::move(reason);
    return result;
}

bool isNeighborhoodProducer(const StagePayload& payload)
{
    return std::holds_alternative<LabelDuplicatesProgram>(payload) ||
           std::holds_alternative<ApproximateCoplanarProgram>(payload) ||
           std::holds_alternative<LofProgram>(payload) ||
           std::holds_alternative<NnDistanceProgram>(payload) ||
           std::holds_alternative<OutlierProgram>(payload) ||
           std::holds_alternative<HagNnProgram>(payload) ||
           std::holds_alternative<HagDelaunayProgram>(payload) ||
           std::holds_alternative<NormalProgram>(payload) ||
           std::holds_alternative<EigenvaluesProgram>(payload) ||
           std::holds_alternative<CovarianceFeaturesProgram>(payload) ||
           std::holds_alternative<EstimateRankProgram>(payload) ||
           std::holds_alternative<OptimalNeighborhoodProgram>(payload) ||
           std::holds_alternative<NeighborClassifierProgram>(payload) ||
           std::holds_alternative<RadialDensityProgram>(payload) ||
           std::holds_alternative<RadiusAssignProgram>(payload);
}

bool isSmrfProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<SmrfProgram>(payload);
}

bool isPmfProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<PmfProgram>(payload);
}

bool isCsfProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<CsfProgram>(payload);
}

bool isElmProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<ElmProgram>(payload);
}

bool isSkewnessProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<SkewnessProgram>(payload);
}

bool isOrderingProducer(const StagePayload& payload) noexcept
{
    return std::holds_alternative<OrderingProgram>(payload);
}

bool sameOptionalJson(const Json& left, const Json& right, const char* name)
{
    const auto leftValue = left.find(name);
    const auto rightValue = right.find(name);
    if ((leftValue == left.end()) != (rightValue == right.end()))
        return false;
    return leftValue == left.end() || *leftValue == *rightValue;
}

bool samePmfRasterSource(const PlannedStage& left, const Json& leftNode,
                         const PlannedStage& right, const Json& rightNode)
{
    if (!isPmfProducer(left.payload) || !isPmfProducer(right.payload))
        return false;
    const GridRequest& leftGrid = left.descriptor.grid;
    const GridRequest& rightGrid = right.descriptor.grid;
    return leftGrid.framePolicy == rightGrid.framePolicy &&
           leftGrid.cellSize == rightGrid.cellSize &&
           leftGrid.deviceBytesPerCell == rightGrid.deviceBytesPerCell &&
           leftGrid.deviceBackingCount == rightGrid.deviceBackingCount &&
           leftGrid.deviceProofBytesPerCell ==
               rightGrid.deviceProofBytesPerCell &&
           leftGrid.deviceFixedBytes == rightGrid.deviceFixedBytes &&
           leftGrid.hostBytesPerPoint == rightGrid.hostBytesPerPoint &&
           leftGrid.hostBytesPerCell == rightGrid.hostBytesPerCell &&
           leftGrid.hostTileBytesPerExpandedCell ==
               rightGrid.hostTileBytesPerExpandedCell &&
           leftGrid.maximumHaloCells == rightGrid.maximumHaloCells &&
           leftGrid.phaseSynchronized == rightGrid.phaseSynchronized &&
           sameOptionalJson(leftNode, rightNode, "returns");
}

std::string_view neighborhoodWrapperType(const StagePayload& payload)
{
    if (std::holds_alternative<LabelDuplicatesProgram>(payload))
        return HybridLabelDuplicatesStage;
    if (std::holds_alternative<LofProgram>(payload))
        return HybridLofStage;
    if (std::holds_alternative<NnDistanceProgram>(payload))
        return HybridNnDistanceStage;
    if (std::holds_alternative<OutlierProgram>(payload))
        return HybridOutlierStage;
    if (std::holds_alternative<HagNnProgram>(payload))
        return HybridHagNnStage;
    if (std::holds_alternative<HagDelaunayProgram>(payload))
        return HybridHagDelaunayStage;
    if (std::holds_alternative<NormalProgram>(payload))
        return HybridNormalStage;
    if (std::holds_alternative<EigenvaluesProgram>(payload))
        return HybridEigenvaluesStage;
    if (std::holds_alternative<CovarianceFeaturesProgram>(payload))
        return HybridCovarianceFeaturesStage;
    if (std::holds_alternative<EstimateRankProgram>(payload))
        return HybridEstimateRankStage;
    if (std::holds_alternative<OptimalNeighborhoodProgram>(payload))
        return HybridOptimalNeighborhoodStage;
    if (std::holds_alternative<NeighborClassifierProgram>(payload))
        return HybridNeighborClassifierStage;
    if (std::holds_alternative<RadialDensityProgram>(payload))
        return HybridRadialDensityStage;
    if (std::holds_alternative<RadiusAssignProgram>(payload))
        return HybridRadiusAssignStage;
    return HybridApproximateCoplanarStage;
}

// Emits one shared-index neighborhood region: wrapper nodes for declared
// kNN payloads and point-program bridge nodes for assign/ferry consumers,
// all bound to one neighborhood region identifier so resident columns are
// consumed without a host round trip. Admission is descriptor-declared.
ResidentPipelineRewrite
emitNeighborhoodRegion(const Plan& plan, const Json& pipeline,
                       const std::vector<bool>& selectedStage,
                       std::size_t region, std::size_t& index, Json& rewritten,
                       ResidentPipelineRewrite& result)
{
    const auto unavailableRegion = [](std::string reason)
    {
        ResidentPipelineRewrite failure;
        failure.reason = std::move(reason);
        return failure;
    };
    std::size_t regionEnd = index;
    std::uint32_t maximumNeighbors = 0U;
    double maximumRadius = 0.0;
    std::uint32_t spatialDimensions = 0U;
    IndexKind regionIndexKind = IndexKind::None;
    while (regionEnd < plan.stages().size() && selectedStage[regionEnd] &&
           plan.stages()[regionEnd].residentRegion == region)
    {
        const PlannedStage& stage = plan.stages()[regionEnd];
        if (!stage.native || stage.preferredResidency != MemoryKind::Device ||
            !stage.descriptor.fusion.pure ||
            !stage.descriptor.fusion.cardinalityPreserving ||
            !stage.descriptor.fusion.deterministicSafe ||
            !stage.descriptor.preservesOrder ||
            stage.descriptor.mutatesCoordinates ||
            stage.descriptor.fusion.hasWhere ||
            stage.descriptor.fusion.whereMerge != WhereMergeMode::NotApplicable)
            return unavailableRegion(
                "selected resident neighborhood region is outside the "
                "declared exact envelope");
        if (isNeighborhoodProducer(stage.payload))
        {
            if (std::holds_alternative<LabelDuplicatesProgram>(stage.payload))
            {
                if (stage.descriptor.index.kind != IndexKind::None)
                    return unavailableRegion(
                        "selected resident label_duplicates stage declares "
                        "an unexpected index request");
                if (!pipeline.at(regionEnd).is_object())
                    return unavailableRegion(
                        "selected resident neighborhood stage is not an "
                        "object");
                ++regionEnd;
                continue;
            }
            const auto* outlier = std::get_if<OutlierProgram>(&stage.payload);
            const bool radiusProducer =
                std::holds_alternative<RadiusAssignProgram>(stage.payload) ||
                std::holds_alternative<RadialDensityProgram>(stage.payload) ||
                (outlier && outlier->method == OutlierMethod::Radius);
            const IndexKind expected =
                radiusProducer ? IndexKind::Radius : IndexKind::Knn;
            if (stage.descriptor.index.kind != expected ||
                (expected == IndexKind::Knn &&
                 (stage.descriptor.index.neighbors == 0U ||
                  (stage.descriptor.index.dimensions != 2U &&
                   stage.descriptor.index.dimensions != 3U))) ||
                (expected == IndexKind::Radius &&
                 (!(stage.descriptor.index.radius > 0.0) ||
                  !std::isfinite(stage.descriptor.index.radius))))
                return unavailableRegion(
                    "selected resident neighborhood stage declares an "
                    "invalid shared-index request");
            if (regionIndexKind != IndexKind::None &&
                regionIndexKind != expected)
                return unavailableRegion(
                    "selected resident neighborhood region mixes radius and "
                    "kNN index requests");
            regionIndexKind = expected;
            if (expected == IndexKind::Knn)
                maximumNeighbors = (std::max)(maximumNeighbors,
                                              stage.descriptor.index.neighbors);
            else
            {
                maximumRadius =
                    (std::max)(maximumRadius, stage.descriptor.index.radius);
                const std::uint32_t dimensions =
                    stage.descriptor.index.dimensions;
                if (dimensions != 2U && dimensions != 3U)
                    return unavailableRegion(
                        "selected resident radius stage declares invalid "
                        "spatial dimensions");
                if (spatialDimensions != 0U && spatialDimensions != dimensions)
                    return unavailableRegion(
                        "selected resident radius region mixes 2D and 3D "
                        "index requests");
                spatialDimensions = dimensions;
            }
        }
        else if (!std::holds_alternative<FerryProgram>(stage.payload) &&
                 !std::holds_alternative<AssignProgram>(stage.payload))
            return unavailableRegion(
                "selected resident neighborhood region contains an "
                "unsupported bridge");
        if (!pipeline.at(regionEnd).is_object())
            return unavailableRegion(
                "selected resident neighborhood stage is not an object");
        ++regionEnd;
    }
    if (region + 1U > (std::numeric_limits<std::uint64_t>::max)())
        return unavailableRegion("resident region identifier exceeds uint64");
    // Zero is reserved for an unplanned one-stage invocation.
    const std::uint64_t neighborhoodId =
        static_cast<std::uint64_t>(region) + 1U;
    while (index < regionEnd)
    {
        if (isNeighborhoodProducer(plan.stages()[index].payload))
        {
            Json replacement = pipeline.at(index);
            replacement["type"] =
                neighborhoodWrapperType(plan.stages()[index].payload);
            replacement["pdg_region_id"] = neighborhoodId;
            if (std::holds_alternative<LabelDuplicatesProgram>(
                    plan.stages()[index].payload))
                replacement["pdg_region_index_required"] =
                    regionIndexKind != IndexKind::None;
            if (regionIndexKind == IndexKind::Radius)
            {
                replacement["pdg_region_radius"] = maximumRadius;
                replacement["pdg_region_dimensions"] = spatialDimensions;
            }
            else
                replacement["pdg_region_neighbors"] = maximumNeighbors;
            if (plan.stages()[index].deviceKnnGatherNeighbors != 0U)
                replacement["pdg_region_gather_neighbors"] =
                    plan.stages()[index].deviceKnnGatherNeighbors;
            if (regionIndexKind == IndexKind::Knn &&
                plan.stages()[index].descriptor.index.dimensions != 3U)
                replacement["pdg_region_dimensions"] =
                    plan.stages()[index].descriptor.index.dimensions;
            replacement["pdg_region_reuse"] = false;
            replacement["pdg_region_last"] = index + 1U == regionEnd;
            // Only the terminal writer follows this region: no later stage can
            // read a PointView KD3 product, so the wrapper's exact tie repair
            // may keep its compatibility tree private (D0262).
            replacement["pdg_region_terminal_sink"] =
                regionEnd + 1U == plan.stages().size() &&
                plan.stages()[regionEnd].role == StageRole::Writer;
            replacement["pdg_resident_context"] = true;
            replacement["pdg_execution_region"] =
                static_cast<std::uint64_t>(region);
            rewritten.push_back(std::move(replacement));
            result.selectedStageIds.push_back(index);
            ++index;
            continue;
        }
        Json bridge = Json::array();
        while (index < regionEnd &&
               !isNeighborhoodProducer(plan.stages()[index].payload))
        {
            bridge.push_back(pipeline.at(index));
            result.selectedStageIds.push_back(index);
            ++index;
        }
        rewritten.push_back(
            {{"type", HybridPointProgramStage},
             {"program", bridge.dump()},
             {"pdg_plan_cuda", true},
             {"pdg_resident_context", true},
             {"pdg_execution_region", static_cast<std::uint64_t>(region)},
             {"pdg_neighborhood_region_id", neighborhoodId},
             {"pdg_neighborhood_region_last", index == regionEnd}});
    }
    ++result.pointProgramRegions;
    ResidentPipelineRewrite emitted;
    emitted.executable = true;
    return emitted;
}

ResidentPipelineRewrite emitGridRegion(const Plan& plan, const Json& pipeline,
                                       const std::vector<bool>& selectedStage,
                                       std::size_t region, std::size_t& index,
                                       Json& rewritten,
                                       ResidentPipelineRewrite& result)
{
    const PlannedStage& stage = plan.stages()[index];
    const bool smrf = isSmrfProducer(stage.payload);
    const bool pmf = isPmfProducer(stage.payload);
    const bool csf = isCsfProducer(stage.payload);
    const bool elm = isElmProducer(stage.payload);
    const std::string_view name = elm   ? "elm"
                                  : csf ? "csf"
                                  : pmf ? "pmf"
                                        : "smrf";
    if ((!smrf && !pmf && !csf && !elm) || !selectedStage[index] ||
        stage.residentRegion != region || !stage.native ||
        stage.preferredResidency != MemoryKind::Device ||
        stage.descriptor.kind != StageKind::Grid ||
        stage.descriptor.index.kind != IndexKind::None ||
        !stage.descriptor.fusion.pure ||
        !stage.descriptor.fusion.cardinalityPreserving ||
        !stage.descriptor.fusion.deterministicSafe ||
        !stage.descriptor.preservesOrder ||
        stage.descriptor.mutatesCoordinates ||
        stage.descriptor.fusion.hasWhere ||
        stage.descriptor.fusion.whereMerge != WhereMergeMode::NotApplicable)
        return unavailable("selected resident " + std::string(name) +
                           " region is outside the declared exact envelope");
    if (!pipeline.at(index).is_object())
        return unavailable("selected resident " + std::string(name) +
                           " stage is not an object");
    if (region > (std::numeric_limits<std::uint64_t>::max)())
        return unavailable("resident region identifier exceeds uint64");

    std::size_t regionEnd = index + 1U;
    while (regionEnd < plan.stages().size() && selectedStage[regionEnd] &&
           plan.stages()[regionEnd].residentRegion == region)
        ++regionEnd;
    if (regionEnd != index + 1U && !pmf)
        return unavailable("selected resident " + std::string(name) +
                           " region has no composable grid bridge");
    if (pmf)
        for (std::size_t candidate = index + 1U; candidate < regionEnd;
             ++candidate)
        {
            const PlannedStage& next = plan.stages()[candidate];
            if (!isPmfProducer(next.payload))
                return unavailable(
                    "selected resident pmf region has no composable grid "
                    "bridge");
            if (!next.native || next.preferredResidency != MemoryKind::Device ||
                next.descriptor.kind != StageKind::Grid ||
                next.descriptor.index.kind != IndexKind::None ||
                !next.descriptor.fusion.pure ||
                !next.descriptor.fusion.cardinalityPreserving ||
                !next.descriptor.fusion.deterministicSafe ||
                !next.descriptor.preservesOrder ||
                next.descriptor.mutatesCoordinates ||
                next.descriptor.fusion.hasWhere ||
                next.descriptor.fusion.whereMerge !=
                    WhereMergeMode::NotApplicable ||
                !pipeline.at(candidate).is_object())
                return unavailable(
                    "selected resident pmf region is outside the declared "
                    "exact envelope");
            if (!samePmfRasterSource(stage, pipeline.at(index), next,
                                     pipeline.at(candidate)))
                return unavailable(
                    "adjacent resident pmf stages have different raster "
                    "sources");
        }

    for (std::size_t candidate = index; candidate < regionEnd; ++candidate)
    {
        Json replacement = pipeline.at(candidate);
        replacement["type"] = elm   ? HybridElmStage
                              : csf ? HybridCsfStage
                              : pmf ? HybridPmfStage
                                    : HybridSmrfStage;
        replacement["pdg_resident_context"] = true;
        replacement["pdg_execution_region"] =
            static_cast<std::uint64_t>(region);
        if (pmf)
        {
            replacement["pdg_grid_reuse"] = candidate != index;
            replacement["pdg_grid_region_last"] = candidate + 1U == regionEnd;
        }
        rewritten.push_back(std::move(replacement));
        result.selectedStageIds.push_back(candidate);
    }
    index = regionEnd;
    ResidentPipelineRewrite emitted;
    emitted.executable = true;
    return emitted;
}

ResidentPipelineRewrite
emitSkewnessRegion(const Plan& plan, const Json& pipeline,
                   const std::vector<bool>& selectedStage, std::size_t region,
                   std::size_t& index, Json& rewritten,
                   ResidentPipelineRewrite& result)
{
    const PlannedStage& stage = plan.stages()[index];
    const auto* program = std::get_if<SkewnessProgram>(&stage.payload);
    if (!program || !selectedStage[index] || stage.residentRegion != region ||
        !stage.native || stage.preferredResidency != MemoryKind::Device ||
        stage.descriptor.kind != StageKind::Global ||
        stage.descriptor.index.kind != IndexKind::None ||
        !stage.descriptor.fusion.pure ||
        !stage.descriptor.fusion.cardinalityPreserving ||
        !stage.descriptor.fusion.deterministicSafe ||
        stage.descriptor.preservesOrder ||
        stage.descriptor.mutatesCoordinates ||
        stage.descriptor.fusion.hasWhere ||
        stage.descriptor.fusion.whereMerge != WhereMergeMode::NotApplicable ||
        !skewnessProgramValid(*program))
        return unavailable(
            "selected resident skewness region is outside the declared exact "
            "envelope");
    if (!pipeline.at(index).is_object())
        return unavailable("selected resident skewness stage is not an object");
    if (region > (std::numeric_limits<std::uint64_t>::max)())
        return unavailable("resident region identifier exceeds uint64");
    if (index + 1U < plan.stages().size() && selectedStage[index + 1U] &&
        plan.stages()[index + 1U].residentRegion == region)
        return unavailable(
            "selected resident skewness region has no composable bridge");

    Json replacement = pipeline.at(index);
    replacement["type"] = HybridSkewnessStage;
    replacement["pdg_resident_context"] = true;
    replacement["pdg_execution_region"] = static_cast<std::uint64_t>(region);
    rewritten.push_back(std::move(replacement));
    result.selectedStageIds.push_back(index);
    ++index;
    ResidentPipelineRewrite emitted;
    emitted.executable = true;
    return emitted;
}

ResidentPipelineRewrite
emitOrderingRegion(const Plan& plan, const Json& pipeline,
                   const std::vector<bool>& selectedStage, std::size_t region,
                   std::size_t& index, Json& rewritten,
                   ResidentPipelineRewrite& result)
{
    const PlannedStage& stage = plan.stages()[index];
    const auto* program = std::get_if<OrderingProgram>(&stage.payload);
    const DimensionId z(StandardDimension::Z);
    if (!program || !selectedStage[index] || stage.residentRegion != region ||
        !stage.native || stage.preferredResidency != MemoryKind::Device ||
        stage.descriptor.kind != StageKind::Global ||
        stage.descriptor.index.kind != IndexKind::None ||
        !stage.descriptor.fusion.pure ||
        !stage.descriptor.fusion.cardinalityPreserving ||
        !stage.descriptor.fusion.deterministicSafe ||
        stage.descriptor.preservesOrder ||
        stage.descriptor.mutatesCoordinates ||
        stage.descriptor.fusion.hasWhere ||
        stage.descriptor.fusion.whereMerge != WhereMergeMode::NotApplicable ||
        program->dimensions != std::vector<DimensionId>{z} ||
        program->direction != OrderingDirection::Ascending ||
        program->algorithm != OrderingAlgorithm::Normal)
        return unavailable(
            "selected resident ordering region is outside the declared exact "
            "envelope");
    if (!pipeline.at(index).is_object())
        return unavailable("selected resident ordering stage is not an object");
    if (region > (std::numeric_limits<std::uint64_t>::max)())
        return unavailable("resident region identifier exceeds uint64");
    if (index + 1U < plan.stages().size() && selectedStage[index + 1U] &&
        plan.stages()[index + 1U].residentRegion == region)
        return unavailable(
            "selected resident ordering region has no composable bridge");

    Json replacement = pipeline.at(index);
    replacement["type"] = HybridOrderStage;
    replacement["pdg_resident_context"] = true;
    replacement["pdg_execution_region"] = static_cast<std::uint64_t>(region);
    rewritten.push_back(std::move(replacement));
    result.selectedStageIds.push_back(index);
    ++index;
    ResidentPipelineRewrite emitted;
    emitted.executable = true;
    return emitted;
}

Json boundaryMarker(std::size_t id, const ResidencyBoundary& boundary,
                    std::size_t residentRegion)
{
    if (id > (std::numeric_limits<std::uint64_t>::max)() ||
        residentRegion > (std::numeric_limits<std::uint64_t>::max)())
        throw std::overflow_error(
            "resident boundary identifier exceeds uint64");
    return {
        {"type", HybridResidentBoundaryStage},
        {"pdg_boundary_kind",
         boundary.kind == ResidencyBoundaryKind::Upload ? "upload" : "spill"},
        {"pdg_boundary_id", static_cast<std::uint64_t>(id)},
        {"pdg_execution_region", static_cast<std::uint64_t>(residentRegion)},
        {"pdg_requires_full_point_record", boundary.requiresFullPointRecord}};
}
} // unnamed namespace

std::size_t selectedGridBuildCount(const Plan& plan,
                                   const PlanPlacementEstimate& placement)
{
    std::vector<bool> selectedRegion(plan.summary().residentRegions, false);
    for (const PlacementRegionEstimate& region : placement.regions)
        if (region.selected && region.residentRegion < selectedRegion.size())
            selectedRegion[region.residentRegion] = true;

    std::vector<bool> counted(plan.summary().residentRegions, false);
    std::size_t builds = 0U;
    for (const PlannedStage& stage : plan.stages())
        if (stage.residentRegion != NoResidentRegion &&
            selectedRegion.at(stage.residentRegion) &&
            stage.deviceGridBuildBytesPerCell != 0U &&
            !counted.at(stage.residentRegion))
        {
            counted[stage.residentRegion] = true;
            ++builds;
        }
    return builds;
}

ResidentPipelineRewrite
rewriteResidentPlacement(std::string_view pipelineJson, const Plan& plan,
                         const PlanPlacementEstimate& placement)
{
    Json root;
    try
    {
        root = Json::parse(pipelineJson, nullptr, true, true);
    }
    catch (const Json::parse_error& error)
    {
        return unavailable(std::string("invalid pipeline JSON: ") +
                           error.what());
    }
    Json* pipeline = pipelineArray(root);
    if (!pipeline)
        return unavailable(
            "pipeline root must be an array or contain a pipeline array");
    if (pipeline->size() != plan.stages().size())
        return unavailable("pipeline stage count differs from placement plan");

    std::vector<bool> selectedStage(plan.stages().size(), false);
    std::vector<bool> selectedRegion(plan.summary().residentRegions, false);
    std::vector<std::size_t> selectedRegionIds;
    std::size_t selectedRegionCount = 0;
    for (const PlacementRegionEstimate& region : placement.regions)
    {
        if (!region.selected)
            continue;
        ++selectedRegionCount;
        if (region.residentRegion >= plan.summary().residentRegions ||
            selectedRegion[region.residentRegion])
            return unavailable(
                "placement selected an unknown or duplicate resident region");
        selectedRegion[region.residentRegion] = true;
        selectedRegionIds.push_back(region.residentRegion);
        for (std::size_t stageId : region.stageIds)
        {
            if (stageId >= selectedStage.size() || selectedStage[stageId])
                return unavailable(
                    "placement selected an invalid or duplicate stage");
            const PlannedStage& stage = plan.stages()[stageId];
            if (!stage.native ||
                stage.preferredResidency != MemoryKind::Device ||
                stage.residentRegion != region.residentRegion)
                return unavailable(
                    "placement selected a stage outside its resident region");
            selectedStage[stageId] = true;
        }
        if (region.stageIds.empty())
            return unavailable("placement selected an empty resident region");
    }
    if (placement.selectedRegionCount != selectedRegionCount)
        return unavailable("placement selected-region count is inconsistent");
    for (const PlannedStage& stage : plan.stages())
        if (stage.residentRegion != NoResidentRegion &&
            stage.residentRegion < selectedRegion.size() &&
            selectedRegion[stage.residentRegion] && stage.native &&
            stage.preferredResidency == MemoryKind::Device &&
            !selectedStage[stage.id])
            return unavailable(
                "placement selected only part of a resident region");

    Json rewritten = Json::array();
    ResidentPipelineRewrite result;
    result.selectedRegions = std::move(selectedRegionIds);
    for (std::size_t index = 0; index < plan.stages().size();)
    {
        for (std::size_t boundaryId = 0;
             boundaryId < plan.summary().residencyBoundaries.size();
             ++boundaryId)
        {
            const ResidencyBoundary& boundary =
                plan.summary().residencyBoundaries[boundaryId];
            if (boundary.consumer != index)
                continue;
            const std::size_t deviceStage =
                boundary.kind == ResidencyBoundaryKind::Upload
                    ? boundary.consumer
                    : boundary.producer;
            if (deviceStage >= selectedStage.size() ||
                !selectedStage[deviceStage])
                continue;
            const PlannedStage& stage = plan.stages()[deviceStage];
            if (stage.residentRegion == NoResidentRegion)
                return unavailable(
                    "selected boundary has no resident execution region");
            if (boundary.consumers.size() != 1U ||
                boundary.consumers.front() != boundary.consumer)
                return unavailable(
                    "selected boundary has unsupported branching topology");
            rewritten.push_back(
                boundaryMarker(boundaryId, boundary, stage.residentRegion));
        }

        if (!selectedStage[index])
        {
            rewritten.push_back(pipeline->at(index));
            ++index;
            continue;
        }

        const std::size_t region = plan.stages()[index].residentRegion;
        const std::size_t begin = index;
        const bool neighborhoodRegion = [&]
        {
            for (std::size_t stageId = index;
                 stageId < plan.stages().size() && selectedStage[stageId] &&
                 plan.stages()[stageId].residentRegion == region;
                 ++stageId)
                if (isNeighborhoodProducer(plan.stages()[stageId].payload))
                    return true;
            return false;
        }();
        if (neighborhoodRegion)
        {
            const ResidentPipelineRewrite emitted =
                emitNeighborhoodRegion(plan, *pipeline, selectedStage, region,
                                       index, rewritten, result);
            if (!emitted.executable)
                return emitted;
            continue;
        }
        if (isSmrfProducer(plan.stages()[index].payload) ||
            isPmfProducer(plan.stages()[index].payload) ||
            isCsfProducer(plan.stages()[index].payload) ||
            isElmProducer(plan.stages()[index].payload))
        {
            const ResidentPipelineRewrite emitted =
                emitGridRegion(plan, *pipeline, selectedStage, region, index,
                               rewritten, result);
            if (!emitted.executable)
                return emitted;
            continue;
        }
        if (isSkewnessProducer(plan.stages()[index].payload))
        {
            const ResidentPipelineRewrite emitted =
                emitSkewnessRegion(plan, *pipeline, selectedStage, region,
                                   index, rewritten, result);
            if (!emitted.executable)
                return emitted;
            continue;
        }
        if (isOrderingProducer(plan.stages()[index].payload))
        {
            const ResidentPipelineRewrite emitted =
                emitOrderingRegion(plan, *pipeline, selectedStage, region,
                                   index, rewritten, result);
            if (!emitted.executable)
                return emitted;
            continue;
        }
        std::size_t declaredCardinalityChanges = 0U;
        Json program = Json::array();
        while (index < plan.stages().size() && selectedStage[index] &&
               plan.stages()[index].residentRegion == region)
        {
            const PlannedStage& stage = plan.stages()[index];
            if (!pointProgramPayload(stage.payload))
                return unavailable(
                    "selected resident region has no point-program executor");
            if (stage.descriptor.fusion.hasWhere ||
                stage.descriptor.fusion.whereMerge !=
                    WhereMergeMode::NotApplicable)
                return unavailable("selected resident stage declares "
                                   "conditional where semantics");
            if (!stage.descriptor.fusion.pure ||
                !stage.descriptor.fusion.deterministicSafe ||
                stage.descriptor.mutatesCoordinates)
                return unavailable(
                    "selected resident point program is outside the exact "
                    "cardinality-preserving context envelope");
            // A cardinality change is legal only when it is declared by the
            // descriptor contract, executes a pure order-preserving predicate
            // payload, and appears at most once in the region. Descriptors,
            // not stage names, own this decision.
            if (stage.descriptor.fusion.cardinalityPreserving ==
                std::holds_alternative<PredicateProgram>(stage.payload))
                return unavailable(
                    "selected resident point program is outside the exact "
                    "cardinality-preserving context envelope");
            if (!stage.descriptor.fusion.cardinalityPreserving)
            {
                if (!stage.descriptor.preservesOrder)
                    return unavailable(
                        "selected resident cardinality change does not "
                        "declare stable order");
                if (++declaredCardinalityChanges > 1U)
                    return unavailable(
                        "selected resident region declares more than one "
                        "cardinality change");
            }
            if (!pipeline->at(index).is_object())
                return unavailable(
                    "selected point-program stage is not an object");
            program.push_back(pipeline->at(index));
            result.selectedStageIds.push_back(index);
            ++index;
        }
        if (index < plan.stages().size() && selectedStage[index] &&
            plan.stages()[index].residentRegion != region)
            return unavailable("selected resident regions overlap");
        if (region > (std::numeric_limits<std::uint64_t>::max)())
            return unavailable("resident region identifier exceeds uint64");
        rewritten.push_back(
            {{"type", HybridPointProgramStage},
             {"program", program.dump()},
             {"pdg_plan_cuda", true},
             {"pdg_resident_context", true},
             {"pdg_execution_region", static_cast<std::uint64_t>(region)}});
        ++result.pointProgramRegions;

        // Maximal regions in the currently executable linear topology are
        // contiguous. A selected stage from this same region appearing later
        // would imply a graph shape this materializer cannot preserve.
        for (std::size_t remaining = index; remaining < plan.stages().size();
             ++remaining)
            if (selectedStage[remaining] &&
                plan.stages()[remaining].residentRegion == region)
                return unavailable(
                    "selected resident region is not contiguous");
        if (index == begin)
            throw std::logic_error("empty selected resident region");
    }

    *pipeline = std::move(rewritten);
    result.json = root.dump();
    result.executable = true;
    return result;
}

} // namespace pdg
