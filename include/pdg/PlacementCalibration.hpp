#pragma once

// Shared placement-calibration case machinery (D0277).  The bench audit tool
// (`pdg_placement_audit`) and the engine's `gpupal calibrate` command build the
// same PlacementRequest for a calibration case and fit the same residual
// model, so a locally calibrated profile is produced by exactly the procedure
// that produced the embedded SM-89 profile.  Nothing here measures anything.

#include <pdg/Placement.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace pdg
{
class Plan;

struct PlacementCalibrationShape
{
    std::string id;
    std::string model;
    std::string pipelineJson;
    std::size_t inputPoints = 0;
    std::size_t outputPoints = 0;
    std::size_t pointCapacity = 0;
    std::size_t inputRecordBytes = 0;
    std::size_t outputRecordBytes = 0;
    std::size_t fallbackRecordBytes = 0;
    std::size_t additionalSynchronizations = 0;
    std::size_t deviceMemoryBudgetBytes =
        (std::numeric_limits<std::size_t>::max)();
    bool cudaContextWarm = false;
    bool intrinsicSingleLaneExecutor = false;
};

// Builds the placement request the planner would evaluate for this case.
// Throws std::invalid_argument for a shape without a device stage or with an
// inconsistent direct-executor declaration.
[[nodiscard]] PlacementRequest
makePlacementCalibrationRequest(const Plan& plan,
                                const PlacementCalibrationShape& shape);

// One measured sample: input cardinality and the residual (measured device
// minus host nanoseconds, minus the planner-owned device terms).
struct PlacementResidualSample
{
    double points = 0.0;
    double residualNanoseconds = 0.0;
};

// Ordinary least squares over the samples, then split by sign exactly as the
// audit prints it: a negative intercept/slope is host work, a positive one is
// device work.  A single sample yields a slope-only model.  Throws
// std::invalid_argument on duplicate cardinalities or no samples.
[[nodiscard]] StagePlacementCost
fitPlacementResidualModel(std::span<const PlacementResidualSample> samples);

} // namespace pdg
