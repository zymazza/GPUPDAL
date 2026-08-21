#include <pdg/Placement.hpp>

#include <pdg/Plan.hpp>
#include <pdg/Scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace pdg
{
namespace
{
std::size_t checkedAdd(std::size_t left, std::size_t right, const char* message)
{
    if (left > (std::numeric_limits<std::size_t>::max)() - right)
        throw std::overflow_error(message);
    return left + right;
}

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* message)
{
    if (left && right > (std::numeric_limits<std::size_t>::max)() / left)
        throw std::overflow_error(message);
    return left * right;
}

void validateCoefficient(double value)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(
            "placement coefficients must be finite and nonnegative");
}

double checkedCost(double units, double coefficient)
{
    const double result = units * coefficient;
    if (!std::isfinite(result))
        throw std::overflow_error("placement cost estimate overflows");
    return result;
}

void addCost(double& destination, double value)
{
    destination += value;
    if (!std::isfinite(destination))
        throw std::overflow_error("placement cost estimate overflows");
}

bool isDeviceStage(const PlannedStage& stage,
                   std::optional<std::size_t> residentRegion) noexcept
{
    return stage.native && stage.preferredResidency == MemoryKind::Device &&
           (!residentRegion || stage.residentRegion == *residentRegion);
}

bool hasLinearExecutionTopology(const Plan& plan)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    std::vector<std::size_t> consumers(stages.size(), 0U);
    std::size_t readers = 0U;
    std::size_t writers = 0U;
    for (const PlannedStage& stage : stages)
    {
        readers += stage.role == StageRole::Reader ? 1U : 0U;
        writers += stage.role == StageRole::Writer ? 1U : 0U;
        if (stage.role == StageRole::Reader)
        {
            if (!stage.inputs.empty())
                return false;
            continue;
        }
        if (stage.inputs.size() != 1U || stage.inputs.front() >= stages.size())
            return false;
        const std::size_t input = stage.inputs.front();
        consumers[input] = checkedAdd(
            consumers[input], 1U, "placement plan consumer count overflows");
        if (consumers[input] > 1U)
            return false;
    }
    return readers == 1U && writers <= 1U;
}

void addBreakdown(PlacementCostBreakdown& destination,
                  const PlacementCostBreakdown& source)
{
    addCost(destination.stageNanoseconds, source.stageNanoseconds);
    addCost(destination.startupNanoseconds, source.startupNanoseconds);
    addCost(destination.transferNanoseconds, source.transferNanoseconds);
    addCost(destination.packingNanoseconds, source.packingNanoseconds);
    addCost(destination.indexBuildNanoseconds, source.indexBuildNanoseconds);
    addCost(destination.synchronizationNanoseconds,
            source.synchronizationNanoseconds);
    addCost(destination.totalNanoseconds, source.totalNanoseconds);
}
} // unnamed namespace

static PlacementEstimate
evaluatePlacementImpl(const Plan& plan, const PlacementRequest& request,
                      const PlacementModelCoefficients& coefficients,
                      std::optional<std::size_t> residentRegion,
                      bool includeStartup, bool includePlanWideSynchronization)
{
    const std::size_t stageCount = plan.stages().size();
    if (request.stageInputPointCounts.size() != stageCount ||
        request.stageOutputPointCounts.size() != stageCount ||
        request.stageCosts.size() != stageCount ||
        (!request.stagePointCapacities.empty() &&
         request.stagePointCapacities.size() != stageCount) ||
        (!request.stageScratchBytes.empty() &&
         request.stageScratchBytes.size() != stageCount) ||
        (!request.stageAdditionalSynchronizations.empty() &&
         request.stageAdditionalSynchronizations.size() != stageCount))
        throw std::invalid_argument(
            "placement request vectors must match the plan stage count");

    std::vector<const PlacementBoundaryExecutionFact*> boundaryFacts(
        plan.summary().residencyBoundaries.size(), nullptr);
    if (request.intrinsicSingleLaneExecutor &&
        request.boundaryExecutionFacts.empty())
        throw std::invalid_argument(
            "single-lane placement executor has no boundary facts");
    if ((request.executorUntiledDeviceBytes ||
         request.executorPeakDeviceBytes) &&
        (!request.intrinsicSingleLaneExecutor ||
         request.boundaryExecutionFacts.empty() ||
         !request.executorUntiledDeviceBytes ||
         !request.executorPeakDeviceBytes ||
         request.executorUntiledDeviceBytes > request.executorPeakDeviceBytes))
        throw std::invalid_argument(
            "placement executor memory override is incomplete or invalid");
    if (!request.boundaryExecutionFacts.empty())
    {
        if (request.intrinsicSingleLaneExecutor)
        {
            if (request.executorLaneCount != 1U)
                throw std::invalid_argument(
                    "intrinsic single-lane executor width is not one");
        }
        else if (request.executorLaneCount < MinimumSweptLaneCount ||
                 request.executorLaneCount > MaximumSweptLaneCount)
            throw std::invalid_argument(
                "placement executor lane count is outside [2, 6]");
        if (request.boundaryExecutionFacts.size() != boundaryFacts.size())
            throw std::invalid_argument(
                "placement boundary facts must cover the plan exactly");
        for (const PlacementBoundaryExecutionFact& fact :
             request.boundaryExecutionFacts)
        {
            if (fact.boundaryId >= boundaryFacts.size() ||
                boundaryFacts[fact.boundaryId] || !fact.transferBytesPerPoint ||
                !fact.deviceStagingBytesPerPoint)
                throw std::invalid_argument(
                    "placement boundary facts are invalid or duplicated");
            boundaryFacts[fact.boundaryId] = &fact;
        }
    }

    for (double value : {
             coefficients.cudaStartupNanoseconds,
             coefficients.hostToDeviceNanosecondsPerByte,
             coefficients.deviceToHostNanosecondsPerByte,
             coefficients.packingNanosecondsPerByte,
             coefficients.indexBuildNanosecondsPerByte,
             coefficients.synchronizationNanoseconds,
         })
        validateCoefficient(value);

    PlacementEstimate estimate;
    estimate.calibrationComplete = true;
    estimate.calibrationEnvelopeSatisfied = true;
    estimate.layoutComplete = true;
    std::vector<std::size_t> packingStagingBytes(stageCount, 0U);
    std::size_t executorStagingBytes = 0U;
    std::size_t executorStagingBytesPerPoint = 0U;
    std::size_t executorItemCount = 0U;
    std::size_t executorTileItems = 0U;
    bool hasDeviceStage = false;

    for (std::size_t index = 0; index < stageCount; ++index)
    {
        const PlannedStage& stage = plan.stages()[index];
        const StagePlacementCost& cost = request.stageCosts[index];
        for (double value : {
                 cost.hostFixedNanoseconds,
                 cost.deviceFixedNanoseconds,
                 cost.hostNanosecondsPerPoint,
                 cost.deviceNanosecondsPerPoint,
             })
            validateCoefficient(value);
        if (!isDeviceStage(stage, residentRegion))
            continue;

        hasDeviceStage = true;
        estimate.calibrationComplete =
            estimate.calibrationComplete && cost.calibrated;
        estimate.calibrationEnvelopeSatisfied =
            estimate.calibrationEnvelopeSatisfied &&
            request.stageInputPointCounts[index] >=
                cost.minimumDevicePointCount &&
            request.stageInputPointCounts[index] <=
                cost.maximumDevicePointCount;
        addCost(estimate.host.stageNanoseconds,
                cost.hostFixedNanoseconds +
                    checkedCost(static_cast<double>(
                                    request.stageInputPointCounts[index]),
                                cost.hostNanosecondsPerPoint));
        addCost(estimate.device.stageNanoseconds,
                cost.deviceFixedNanoseconds +
                    checkedCost(static_cast<double>(
                                    request.stageInputPointCounts[index]),
                                cost.deviceNanosecondsPerPoint));

        const std::size_t indexBytes =
            checkedProduct(stage.deviceIndexBuildBytesPerPoint,
                           request.stageInputPointCounts[index],
                           "placement index-build byte estimate overflows");
        estimate.indexBuildBytes =
            checkedAdd(estimate.indexBuildBytes, indexBytes,
                       "placement index-build byte estimate overflows");
        const std::size_t pointResultBytes =
            checkedProduct(stage.descriptor.deviceToHostBytesPerInputPoint,
                           request.stageInputPointCounts[index],
                           "placement stage-result byte estimate overflows");
        const std::size_t resultBytes = checkedAdd(
            pointResultBytes, stage.descriptor.deviceToHostFixedBytes,
            "placement stage-result byte estimate overflows");
        estimate.stageResultBytes =
            checkedAdd(estimate.stageResultBytes, resultBytes,
                       "placement stage-result byte estimate overflows");
    }

    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const bool selectedBoundary =
            boundary.kind == ResidencyBoundaryKind::Upload
                ? isDeviceStage(plan.stages()[boundary.consumer],
                                residentRegion)
                : isDeviceStage(plan.stages()[boundary.producer],
                                residentRegion);
        if (!selectedBoundary)
            continue;
        estimate.synchronizationCount =
            checkedAdd(estimate.synchronizationCount, 1U,
                       "placement synchronization count overflows");
        const std::size_t points =
            request.stageOutputPointCounts.at(boundary.producer);
        const std::size_t logicalColumnBytes =
            checkedProduct(boundary.bytesPerPoint, points,
                           "placement transfer byte estimate overflows");
        const PlacementBoundaryExecutionFact* executionFact =
            boundaryFacts[boundaryId];
        std::size_t transferBytes =
            executionFact
                ? checkedProduct(executionFact->transferBytesPerPoint, points,
                                 "placement executor transfer estimate "
                                 "overflows")
                : logicalColumnBytes;
        const std::size_t fullRecordBytes = checkedProduct(
            executionFact ? executionFact->deviceStagingBytesPerPoint
                          : request.fallbackRecordBytes,
            points, "placement runtime full-record byte estimate overflows");
        std::size_t recordBytes = 0U;
        std::size_t stagingStage =
            boundary.kind == ResidencyBoundaryKind::Upload ? boundary.consumer
                                                           : boundary.producer;
        if (executionFact)
        {
            const std::size_t capacity =
                request.stagePointCapacities.empty()
                    ? (std::max)(request.stageInputPointCounts[stagingStage],
                                 request.stageOutputPointCounts[stagingStage])
                    : request.stagePointCapacities[stagingStage];
            if (executorTileItems && executorTileItems != capacity)
                throw std::invalid_argument(
                    "placement executor tile capacities must match");
            executorTileItems = capacity;
            executorItemCount = (std::max)(executorItemCount, points);
            executorStagingBytesPerPoint =
                (std::max)(executorStagingBytesPerPoint,
                           executionFact->deviceStagingBytesPerPoint);
            executorStagingBytes =
                (std::max)(executorStagingBytes,
                           checkedProduct(
                               executionFact->deviceStagingBytesPerPoint,
                               capacity,
                               "placement executor staging estimate "
                               "overflows"));
        }
        else if (boundary.requiresFullPointRecord)
        {
            recordBytes = request.fallbackRecordBytes;
            estimate.layoutComplete =
                estimate.layoutComplete && recordBytes != 0U;
            transferBytes = checkedAdd(
                transferBytes,
                checkedProduct(
                    recordBytes, points,
                    "placement fallback transfer estimate overflows"),
                "placement fallback transfer estimate overflows");
        }
        else if (boundary.kind == ResidencyBoundaryKind::Upload &&
                 plan.stages()[boundary.producer].role == StageRole::Reader)
        {
            recordBytes = request.inputRecordBytes;
            estimate.layoutComplete =
                estimate.layoutComplete && recordBytes != 0U;
            stagingStage = boundary.consumer;
        }
        else if (boundary.kind == ResidencyBoundaryKind::Spill &&
                 std::any_of(boundary.consumers.begin(),
                             boundary.consumers.end(),
                             [&](std::size_t consumer)
                             {
                                 return plan.stages()[consumer].role ==
                                        StageRole::Writer;
                             }))
        {
            recordBytes = request.outputRecordBytes;
            estimate.layoutComplete =
                estimate.layoutComplete && recordBytes != 0U;
        }

        if (boundary.kind == ResidencyBoundaryKind::Upload)
            estimate.hostToDeviceBytes =
                checkedAdd(estimate.hostToDeviceBytes, transferBytes,
                           "placement host-to-device byte estimate overflows");
        else
        {
            const PlannedStage& producer = plan.stages()[boundary.producer];
            const std::size_t pointResultBytes = checkedProduct(
                producer.descriptor.deviceToHostBytesPerInputPoint,
                request.stageInputPointCounts[boundary.producer],
                "placement stage-result byte estimate overflows");
            const std::size_t resultBytes = checkedAdd(
                pointResultBytes, producer.descriptor.deviceToHostFixedBytes,
                "placement stage-result byte estimate overflows");
            if (!executionFact)
                transferBytes = checkedAdd(
                    transferBytes, resultBytes,
                    "placement device-to-host byte estimate overflows");
            estimate.deviceToHostBytes =
                checkedAdd(estimate.deviceToHostBytes, transferBytes,
                           "placement device-to-host byte estimate overflows");
        }

        std::size_t packingBytes =
            executionFact
                ? checkedProduct(executionFact->packingBytesPerPoint, points,
                                 "placement executor packing estimate "
                                 "overflows")
                : 0U;
        if (executionFact)
            estimate.packingBytes =
                checkedAdd(estimate.packingBytes, packingBytes,
                           "placement packing byte estimate overflows");
        else if (recordBytes)
        {
            packingBytes =
                checkedProduct(recordBytes, points,
                               "placement packing byte estimate overflows");
            estimate.packingBytes =
                checkedAdd(estimate.packingBytes, packingBytes,
                           "placement packing byte estimate overflows");
            const std::size_t capacity =
                request.stagePointCapacities.empty()
                    ? (std::max)(request.stageInputPointCounts[stagingStage],
                                 request.stageOutputPointCounts[stagingStage])
                    : request.stagePointCapacities[stagingStage];
            packingStagingBytes[stagingStage] = checkedAdd(
                packingStagingBytes[stagingStage],
                checkedProduct(recordBytes, capacity,
                               "placement packing staging estimate overflows"),
                "placement packing staging estimate overflows");
        }
        estimate.boundaries.push_back({.boundaryId = boundaryId,
                                       .pointCount = points,
                                       .logicalColumnBytes = logicalColumnBytes,
                                       .fullRecordBytes = fullRecordBytes,
                                       .predictedTransferBytes = transferBytes,
                                       .predictedPackingBytes = packingBytes});
    }

    for (std::size_t index = 0; index < stageCount; ++index)
    {
        const PlannedStage& stage = plan.stages()[index];
        if (!isDeviceStage(stage, residentRegion))
            continue;
        if (!request.stageAdditionalSynchronizations.empty())
            estimate.synchronizationCount =
                checkedAdd(estimate.synchronizationCount,
                           request.stageAdditionalSynchronizations[index],
                           "placement synchronization count overflows");
        const std::size_t capacity =
            request.stagePointCapacities.empty()
                ? (std::max)(request.stageInputPointCounts[index],
                             request.stageOutputPointCounts[index])
                : request.stagePointCapacities[index];
        // The current multi-region PointView executor allocates one reusable
        // lane pool at the plan-wide liveness/index high-water mark. Keep the
        // executor-declared estimate identical to that allocation contract;
        // calibration-default requests retain stage-local accounting.
        const std::size_t residentBytesPerPoint =
            request.boundaryExecutionFacts.empty()
                ? checkedAdd(checkedAdd(stage.deviceColumnBytesPerPoint,
                                        stage.deviceIndexBytesPerPoint,
                                        "placement resident byte estimate "
                                        "overflows"),
                             stage.deviceQueryBytesPerPoint,
                             "placement resident byte estimate overflows")
                : plan.summary().peakDeviceBytesPerPoint;
        std::size_t stageBytes =
            checkedProduct(residentBytesPerPoint, capacity,
                           "placement resident byte estimate overflows");
        stageBytes = checkedAdd(
            stageBytes,
            checkedProduct(stage.descriptor.deviceToHostBytesPerInputPoint,
                           capacity,
                           "placement stage-result memory estimate overflows"),
            "placement stage-result memory estimate overflows");
        stageBytes =
            checkedAdd(stageBytes, stage.descriptor.deviceToHostFixedBytes,
                       "placement stage-result memory estimate overflows");
        if (!request.stageScratchBytes.empty())
            stageBytes =
                checkedAdd(stageBytes, request.stageScratchBytes[index],
                           "placement scratch byte estimate overflows");
        stageBytes = checkedAdd(stageBytes, packingStagingBytes[index],
                                "placement staging byte estimate overflows");
        stageBytes = checkedAdd(stageBytes, executorStagingBytes,
                                "placement executor staging estimate "
                                "overflows");
        estimate.peakDeviceBytes =
            (std::max)(estimate.peakDeviceBytes, stageBytes);
    }
    if (!request.boundaryExecutionFacts.empty())
    {
        if (!executorTileItems)
            throw std::invalid_argument(
                "placement executor tile capacity is zero");
        if (request.intrinsicSingleLaneExecutor)
        {
            if (executorTileItems != executorItemCount)
                throw std::invalid_argument(
                    "intrinsic single-lane executor is not whole-view");
            estimate.configuredDeviceLaneCount = 1U;
            estimate.activeDeviceLaneCount = executorItemCount ? 1U : 0U;
            if (!executorItemCount)
                estimate.peakDeviceBytes = 0U;
        }
        else
        {
            const TiledSchedule schedule = makeTiledSchedule(
                {.itemCount = executorItemCount,
                 .tileItems = executorTileItems,
                 .bytesPerLane = estimate.peakDeviceBytes,
                 .requestedLanes = request.executorLaneCount});
            estimate.configuredDeviceLaneCount =
                schedule.configuredLaneCount;
            estimate.activeDeviceLaneCount = schedule.activeLaneCount;
            estimate.peakDeviceBytes = schedule.peakLaneBytes;
        }
        estimate.untiledDeviceBytes = checkedProduct(
            checkedAdd(plan.summary().peakDeviceBytesPerPoint,
                       executorStagingBytesPerPoint,
                       "placement untiled byte estimate overflows"),
            executorItemCount, "placement untiled byte estimate overflows");
        if (request.executorPeakDeviceBytes)
        {
            estimate.untiledDeviceBytes =
                request.executorUntiledDeviceBytes;
            estimate.peakDeviceBytes = request.executorPeakDeviceBytes;
        }
    }
    estimate.deviceMemoryFeasible =
        hasDeviceStage &&
        estimate.peakDeviceBytes <= request.deviceMemoryBudgetBytes;

    if (includePlanWideSynchronization)
        estimate.synchronizationCount = checkedAdd(
            estimate.synchronizationCount, request.additionalSynchronizations,
            "placement synchronization count overflows");

    if (hasDeviceStage && includeStartup && !request.cudaContextWarm)
        estimate.device.startupNanoseconds =
            coefficients.cudaStartupNanoseconds;
    estimate.device.transferNanoseconds =
        checkedCost(static_cast<double>(estimate.hostToDeviceBytes),
                    coefficients.hostToDeviceNanosecondsPerByte) +
        checkedCost(static_cast<double>(estimate.deviceToHostBytes),
                    coefficients.deviceToHostNanosecondsPerByte);
    estimate.device.packingNanoseconds =
        checkedCost(static_cast<double>(estimate.packingBytes),
                    coefficients.packingNanosecondsPerByte);
    estimate.device.indexBuildNanoseconds =
        checkedCost(static_cast<double>(estimate.indexBuildBytes),
                    coefficients.indexBuildNanosecondsPerByte);
    estimate.device.synchronizationNanoseconds =
        checkedCost(static_cast<double>(estimate.synchronizationCount),
                    coefficients.synchronizationNanoseconds);

    estimate.host.totalNanoseconds = estimate.host.stageNanoseconds;
    for (double term : {
             estimate.device.stageNanoseconds,
             estimate.device.startupNanoseconds,
             estimate.device.transferNanoseconds,
             estimate.device.packingNanoseconds,
             estimate.device.indexBuildNanoseconds,
             estimate.device.synchronizationNanoseconds,
         })
        addCost(estimate.device.totalNanoseconds, term);

    if (!hasDeviceStage)
    {
        estimate.calibrationComplete = false;
        estimate.calibrationEnvelopeSatisfied = false;
        estimate.deviceMemoryFeasible = false;
        estimate.reason = PlacementReason::NoDeviceStages;
    }
    else if (!estimate.calibrationComplete)
        estimate.reason = PlacementReason::UncalibratedStage;
    else if (!estimate.calibrationEnvelopeSatisfied)
        estimate.reason = PlacementReason::OutsideCalibrationEnvelope;
    else if (!estimate.layoutComplete)
        estimate.reason = PlacementReason::MissingRecordLayout;
    else if (!estimate.deviceMemoryFeasible)
        estimate.reason = PlacementReason::DeviceMemoryBudgetExceeded;
    else if (estimate.device.totalNanoseconds < estimate.host.totalNanoseconds)
    {
        estimate.choice = PlacementChoice::Device;
        estimate.reason = PlacementReason::DeviceFaster;
    }
    else
        estimate.reason = PlacementReason::HostFasterOrEqual;
    return estimate;
}

PlacementEstimate
evaluatePlacement(const Plan& plan, const PlacementRequest& request,
                  const PlacementModelCoefficients& coefficients)
{
    return evaluatePlacementImpl(plan, request, coefficients, std::nullopt,
                                 true, true);
}

PlanPlacementEstimate
evaluatePlanPlacement(const Plan& plan, const PlacementRequest& request,
                      const PlacementModelCoefficients& coefficients)
{
    // Validate the shared request and coefficients even when the plan has no
    // candidate resident region.
    static_cast<void>(evaluatePlacementImpl(plan, request, coefficients,
                                            std::nullopt, false, false));
    PlanPlacementEstimate result;
    std::vector<double> benefits(plan.summary().residentRegions, 0.0);
    double combinedBenefit = 0.0;

    for (std::size_t region = 0; region < plan.summary().residentRegions;
         ++region)
    {
        PlacementRegionEstimate regionEstimate;
        regionEstimate.residentRegion = region;
        for (const PlannedStage& stage : plan.stages())
            if (isDeviceStage(stage, region))
                regionEstimate.stageIds.push_back(stage.id);
        regionEstimate.estimate = evaluatePlacementImpl(
            plan, request, coefficients, region, false, false);
        addBreakdown(result.allHostPlacement, regionEstimate.estimate.host);
        if (regionEstimate.estimate.choice == PlacementChoice::Device)
        {
            const double benefit =
                regionEstimate.estimate.host.totalNanoseconds -
                regionEstimate.estimate.device.totalNanoseconds;
            benefits[region] = benefit;
            addCost(combinedBenefit, benefit);
        }
        result.regions.push_back(std::move(regionEstimate));
    }

    result.selectedPlacement = result.allHostPlacement;
    if (result.regions.empty())
        return result;

    if (!hasLinearExecutionTopology(plan))
    {
        result.reason = PlacementReason::UnsupportedPlanTopology;
        for (PlacementRegionEstimate& region : result.regions)
        {
            region.estimate.choice = PlacementChoice::Host;
            region.estimate.reason = PlacementReason::UnsupportedPlanTopology;
        }
        return result;
    }

    double sharedStartupNanoseconds = 0.0;
    double sharedSynchronizationNanoseconds = 0.0;
    if (combinedBenefit > 0.0)
    {
        if (!request.cudaContextWarm)
            sharedStartupNanoseconds = coefficients.cudaStartupNanoseconds;
        sharedSynchronizationNanoseconds =
            checkedCost(static_cast<double>(request.additionalSynchronizations),
                        coefficients.synchronizationNanoseconds);
    }
    const double sharedDeviceToll =
        sharedStartupNanoseconds + sharedSynchronizationNanoseconds;
    if (combinedBenefit > sharedDeviceToll)
    {
        result.selectedPlacement = {};
        result.choice = PlacementChoice::Device;
        result.reason = PlacementReason::DeviceFaster;
        for (std::size_t region = 0; region < result.regions.size(); ++region)
        {
            PlacementRegionEstimate& selectedRegion = result.regions[region];
            if (benefits[region] <= 0.0)
            {
                addBreakdown(result.selectedPlacement,
                             selectedRegion.estimate.host);
                continue;
            }
            selectedRegion.selected = true;
            ++result.selectedRegionCount;
            addBreakdown(result.selectedPlacement,
                         selectedRegion.estimate.device);
            result.hostToDeviceBytes =
                checkedAdd(result.hostToDeviceBytes,
                           selectedRegion.estimate.hostToDeviceBytes,
                           "plan placement host-to-device bytes overflow");
            result.deviceToHostBytes =
                checkedAdd(result.deviceToHostBytes,
                           selectedRegion.estimate.deviceToHostBytes,
                           "plan placement device-to-host bytes overflow");
            result.packingBytes = checkedAdd(
                result.packingBytes, selectedRegion.estimate.packingBytes,
                "plan placement packing bytes overflow");
            result.indexBuildBytes = checkedAdd(
                result.indexBuildBytes, selectedRegion.estimate.indexBuildBytes,
                "plan placement index-build bytes overflow");
            result.stageResultBytes =
                checkedAdd(result.stageResultBytes,
                           selectedRegion.estimate.stageResultBytes,
                           "plan placement stage-result bytes overflow");
            result.synchronizationCount =
                checkedAdd(result.synchronizationCount,
                           selectedRegion.estimate.synchronizationCount,
                           "plan placement synchronization count overflows");
            result.peakDeviceBytes =
                (std::max)(result.peakDeviceBytes,
                           selectedRegion.estimate.peakDeviceBytes);
            result.untiledDeviceBytes =
                (std::max)(result.untiledDeviceBytes,
                           selectedRegion.estimate.untiledDeviceBytes);
            result.configuredDeviceLaneCount =
                (std::max)(result.configuredDeviceLaneCount,
                           selectedRegion.estimate.configuredDeviceLaneCount);
            result.activeDeviceLaneCount =
                (std::max)(result.activeDeviceLaneCount,
                           selectedRegion.estimate.activeDeviceLaneCount);
            result.boundaries.insert(result.boundaries.end(),
                                     selectedRegion.estimate.boundaries.begin(),
                                     selectedRegion.estimate.boundaries.end());
        }
        std::sort(result.boundaries.begin(), result.boundaries.end(),
                  [](const PlacementBoundaryEstimate& left,
                     const PlacementBoundaryEstimate& right)
                  { return left.boundaryId < right.boundaryId; });
        result.synchronizationCount = checkedAdd(
            result.synchronizationCount, request.additionalSynchronizations,
            "plan placement synchronization count overflows");
        result.selectedPlacement.startupNanoseconds = sharedStartupNanoseconds;
        addCost(result.selectedPlacement.synchronizationNanoseconds,
                sharedSynchronizationNanoseconds);
        addCost(result.selectedPlacement.totalNanoseconds, sharedDeviceToll);
    }
    else
    {
        result.reason = combinedBenefit > 0.0
                            ? PlacementReason::SharedDeviceTollNotAmortized
                            : result.regions.front().estimate.reason;
        for (PlacementRegionEstimate& region : result.regions)
            if (region.estimate.choice == PlacementChoice::Device)
            {
                region.estimate.choice = PlacementChoice::Host;
                region.estimate.reason =
                    PlacementReason::SharedDeviceTollNotAmortized;
            }
    }
    return result;
}

} // namespace pdg
