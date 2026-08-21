#include <pdg/PlacementCalibration.hpp>

#include <pdg/Plan.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pdg
{
namespace
{
[[nodiscard]] bool deviceStage(const PlannedStage& stage)
{
    return stage.native && stage.preferredResidency == MemoryKind::Device;
}
} // unnamed namespace

PlacementRequest
makePlacementCalibrationRequest(const Plan& plan,
                                const PlacementCalibrationShape& shape)
{
    if (shape.inputPoints == 0U || shape.pointCapacity == 0U)
        throw std::invalid_argument(
            "placement calibration counts must be positive");
    PlacementRequest request;
    const std::size_t count = plan.stages().size();
    request.stageInputPointCounts.resize(count);
    request.stageOutputPointCounts.resize(count);
    request.stagePointCapacities.resize(count);
    request.stageScratchBytes.assign(count, 0U);
    request.stageCosts.resize(count);

    std::size_t currentPoints = shape.inputPoints;
    bool outputCardinalityApplied = false;
    for (std::size_t index = 0; index < count; ++index)
    {
        const PlannedStage& stage = plan.stages()[index];
        request.stageInputPointCounts[index] =
            stage.role == StageRole::Reader ? 0U : currentPoints;
        if (stage.role == StageRole::Reader)
            currentPoints = shape.inputPoints;
        else if (stage.role == StageRole::Filter &&
                 !stage.descriptor.fusion.cardinalityPreserving &&
                 !outputCardinalityApplied)
        {
            currentPoints = shape.outputPoints;
            outputCardinalityApplied = true;
        }
        request.stageOutputPointCounts[index] = currentPoints;
        request.stagePointCapacities[index] = shape.pointCapacity;
        if (deviceStage(stage))
            request.stageCosts[index].calibrated = true;
    }
    if (std::none_of(plan.stages().begin(), plan.stages().end(), deviceStage))
        throw std::invalid_argument("calibration case has no device stage: " +
                                    shape.id);

    request.inputRecordBytes = shape.inputRecordBytes;
    request.outputRecordBytes = shape.outputRecordBytes;
    request.fallbackRecordBytes = shape.fallbackRecordBytes;
    request.deviceMemoryBudgetBytes = shape.deviceMemoryBudgetBytes;
    request.additionalSynchronizations = shape.additionalSynchronizations;
    request.cudaContextWarm = shape.cudaContextWarm;
    if (shape.intrinsicSingleLaneExecutor)
    {
        if ((shape.model != "radiusassign-direct" &&
             shape.model != "outlier-nndistance-direct-compose" &&
             shape.model != "radius-outlier-radialdensity-direct-compose" &&
             shape.model != "skewness-direct-compose" &&
             shape.model != "hag-nn-count1-direct-compose" &&
             shape.model != "hag-delaunay-count3-direct-compose" &&
             shape.model != "sort-direct-compose") ||
            shape.pointCapacity != shape.inputPoints)
            throw std::invalid_argument(
                "intrinsic single-lane calibration case has wrong shape: " +
                shape.id);
        request.executorLaneCount = 1U;
        request.intrinsicSingleLaneExecutor = true;
        const bool directSort = shape.model == "sort-direct-compose";
        const bool directSkewness = shape.model == "skewness-direct-compose";
        const bool directHagNn = shape.model == "hag-nn-count1-direct-compose";
        const bool directHagDelaunay =
            shape.model == "hag-delaunay-count3-direct-compose";
        const bool directPermutation = directSort || directSkewness;
        if (directPermutation || directHagNn || directHagDelaunay)
        {
            if (count != 3U ||
                shape.inputPoints >
                    (std::numeric_limits<std::size_t>::max)() /
                        (directHagNn ? HagNnCountOneExactDevicePeakBytesPerPoint
                         : directHagDelaunay
                             ? HagDelaunayCountThreeExactDevicePeakBytesPerPoint
                         : directSkewness ? SkewnessExactDevicePeakBytesPerPoint
                                          : OrderingExactDevicePeakBytesPerPoint))
                throw std::invalid_argument(
                    "direct whole-view calibration memory overflows: " +
                    shape.id);
            request.stageScratchBytes[1U] =
                shape.inputPoints *
                (directHagNn ? HagNnCountOneExactDeviceScratchBytesPerPoint
                 : directHagDelaunay
                     ? HagDelaunayCountThreeExactDeviceScratchBytesPerPoint
                 : directSkewness ? SkewnessExactDeviceScratchBytesPerPoint
                                  : OrderingExactDeviceScratchBytesPerPoint);
            request.executorUntiledDeviceBytes =
                shape.inputPoints *
                (directHagNn ? HagNnCountOneExactDevicePeakBytesPerPoint
                 : directHagDelaunay
                     ? HagDelaunayCountThreeExactDevicePeakBytesPerPoint
                 : directSkewness ? SkewnessExactDevicePeakBytesPerPoint
                                  : OrderingExactDevicePeakBytesPerPoint);
            request.executorPeakDeviceBytes =
                request.executorUntiledDeviceBytes;
        }
        for (std::size_t boundaryId = 0U;
             boundaryId < plan.summary().residencyBoundaries.size();
             ++boundaryId)
        {
            const ResidencyBoundary& boundary =
                plan.summary().residencyBoundaries[boundaryId];
            std::size_t transferBytesPerPoint = boundary.bytesPerPoint;
            if (directSkewness)
                transferBytesPerPoint = sizeof(double);
            else if (directSort &&
                     boundary.kind == ResidencyBoundaryKind::Spill)
                transferBytesPerPoint = sizeof(std::uint64_t);
            request.boundaryExecutionFacts.push_back(
                {.boundaryId = boundaryId,
                 .transferBytesPerPoint = transferBytesPerPoint,
                 .packingBytesPerPoint = 0U,
                 .deviceStagingBytesPerPoint = shape.inputRecordBytes});
        }
    }
    return request;
}

StagePlacementCost
fitPlacementResidualModel(std::span<const PlacementResidualSample> samples)
{
    if (samples.empty())
        throw std::invalid_argument("placement model fit has no samples");
    double intercept = 0.0;
    double slope = 0.0;
    if (samples.size() == 1U)
        slope = samples.front().residualNanoseconds / samples.front().points;
    else
    {
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXX = 0.0;
        double sumXY = 0.0;
        for (const PlacementResidualSample& sample : samples)
        {
            sumX += sample.points;
            sumY += sample.residualNanoseconds;
            sumXX += sample.points * sample.points;
            sumXY += sample.points * sample.residualNanoseconds;
        }
        const double count = static_cast<double>(samples.size());
        const double denominator = count * sumXX - sumX * sumX;
        if (denominator == 0.0)
            throw std::invalid_argument(
                "placement model calibration has duplicate counts");
        slope = (count * sumXY - sumX * sumY) / denominator;
        intercept = (sumY - slope * sumX) / count;
    }
    return StagePlacementCost{
        .hostFixedNanoseconds = intercept < 0.0 ? -intercept : 0.0,
        .deviceFixedNanoseconds = intercept > 0.0 ? intercept : 0.0,
        .hostNanosecondsPerPoint = slope < 0.0 ? -slope : 0.0,
        .deviceNanosecondsPerPoint = slope > 0.0 ? slope : 0.0,
        .calibrated = true};
}

} // namespace pdg
