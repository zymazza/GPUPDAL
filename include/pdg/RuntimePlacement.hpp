#pragma once

#include <pdg/Placement.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pdg
{

class Plan;

// Reasons the runtime adapter refuses to build a placement request.  This is
// intentionally separate from PlacementReason: these failures mean the
// runtime facts cannot faithfully be represented by the calibrated linear
// model, so evaluation is not attempted.
enum class RuntimePlacementUnavailableReason
{
    None,
    ProfileNotExact,
    InvalidRuntimeFacts,
    UnsupportedTopology,
    // A stage the engine does not implement natively. Its properties are
    // unknown rather than known-bad, and `fusion.cardinalityPreserving`
    // defaults to false, so before B0163 every such stage was reported as a
    // cardinality problem it does not have (B0158). `filters.reprojection`,
    // which cannot change point count, reported exactly that.
    UnsupportedStage,
    // A stage that genuinely declares a cardinality change and does not
    // qualify as a declared predicate.
    NonCardinalityPreservingStage,
    MissingCalibrationModel,
    MixedCalibrationModels,
    OutsideCalibrationEnvelope,
    UnknownCalibrationModel,
    NoDeviceCandidate,
    EvaluationFailed
};

// These facts come from the runtime reader/layout and scheduler, never from
// heuristic inspection of pipeline JSON.  inputPointCount is the exact total
// cardinality.  tilePointCapacity is the bounded per-tile capacity used for
// every cardinality-preserving stage's VRAM estimate.
struct RuntimePlacementFacts
{
    std::size_t inputPointCount = 0;
    std::uint8_t inputPointFormat = 0xffU;
    bool inputCompressed = false;
    bool outputCompressed = false;
    std::size_t inputRecordBytes = 0;
    std::size_t outputRecordBytes = 0;
    std::size_t fallbackRecordBytes = 0;
    std::size_t tilePointCapacity = 0;
    std::vector<std::size_t> stageScratchBytes;
    std::vector<std::size_t> stageAdditionalSynchronizations;
    std::vector<PlacementBoundaryExecutionFact> boundaryExecutionFacts;
    std::size_t executorLaneCount = 0;
    std::size_t additionalSynchronizations = 0;
    std::size_t deviceMemoryBudgetBytes = 0;
    bool cudaContextWarm = false;
    // The caller has proved the strict serialized direct-radiusassign shape
    // and required direct LAS source/output boundary. This admits the fixed
    // whole-view lane and zero-pack boundary facts for exact layouts that do
    // not have a timing calibration of their own.
    bool directRadiusAssignBoundaryExecutor = false;
    // The caller has proved the strict B0083 serialized-pipeline envelope,
    // required direct LAS source/output/summary/no-host-XYZ executor, and the
    // calibrated 36-byte LAS record layout. The compiled plan deliberately
    // does not retain radius-domain bounds, so this may never be inferred
    // from descriptor reads alone.
    bool exactDirectRadiusAssignExecutor = false;
    // The caller has proved the strict B0231 neighborclassifier(k=7)
    // mapped-source/direct-Classification publisher, its one planner-owned
    // kNN index, and the calibrated uncompressed format-7/36-byte layout.
    // The measured whole-view executor is single-lane and bounded to its
    // explicit point-count ladder; it must not be inferred from the ordinary
    // neighborclassifier descriptor alone.
    bool directNeighborClassifierBoundaryExecutor = false;
    // The caller has proved the strict B0234 option-free skewnessbalancing
    // mapped-source/permutation-publisher shape, the exact ordering scratch
    // reservation, zero indexes, and calibrated uncompressed format-7/36-byte
    // layout. Comparator ties and non-finite Z remain execution-time exactness
    // refusals and must fall back before publication.
    bool directSkewnessBoundaryExecutor = false;
    // The caller has proved the strict B0235 literal HAG-NN(count=1)
    // mapped-source/direct-extra-double publisher, one planner-owned 2-D kNN
    // index, and the calibrated uncompressed format-7 40->48-byte layout.
    // This is an intrinsic whole-view single-lane executor and must not be
    // inferred from the ordinary HAG-NN descriptor.
    bool directHagNnBoundaryExecutor = false;
    // The caller has proved the strict B0236 literal HAG-Delaunay(count=3)
    // mapped-source/direct-extra-double publisher, one planner-owned 2-D kNN
    // index, and the calibrated uncompressed format-7 40->48-byte layout.
    // This is an intrinsic whole-view single-lane executor and must not be
    // inferred from the ordinary HAG-Delaunay descriptor.
    bool directHagDelaunayBoundaryExecutor = false;
    // The caller has proved the strict B0232 sort(Z,ASC,NORMAL) whole-view
    // direct-LAS source/permutation-publisher shape, the exact binary64 key
    // scratch allocation, zero indexes, and the calibrated uncompressed
    // format-7/36-byte layout. Data-dependent comparator ties are still an
    // execution-time exactness refusal and must fall back before publication.
    bool directSortBoundaryExecutor = false;
    // The caller has proved the strict B0089 direct-LAS statistical-outlier
    // -> NNDistance composition, including its one planner-owned max-k gather,
    // direct source/output boundary, and 36-byte exact record layout. This is
    // an intrinsic whole-view single-lane executor with its own calibration;
    // it must never be inferred for a standalone outlier or NNDistance stage.
    bool directOutlierNnDistanceBoundaryExecutor = false;
    // The caller has proved the strict B0096 approximate-coplanar(knn=8) ->
    // ferry(Coplanar=>UserData) shape and required the direct LAS output
    // boundary. This route deliberately retains the PointView source because
    // exact ambiguous-row repair is owned by pinned upstream KD3; it is not a
    // direct-source or automatic-admission capability.
    bool directApproximateCoplanarOutputExecutor = false;
    // The caller has proved the strict B0126 radius-outlier(radius=1.01,
    // min_k=2,class=7) -> radialdensity(radius=1.01) ->
    // assign(UserData) composition, one shared radius index, the direct LAS
    // source/output boundary, and the calibrated format-7/36-byte layout.
    // This is an intrinsic whole-view single-lane executor with its own
    // bounded calibration and must not be inferred for either stage alone.
    bool directRadiusOutlierRadialDensityBoundaryExecutor = false;
};

struct RuntimePlacementResult
{
    RuntimePlacementUnavailableReason unavailableReason =
        RuntimePlacementUnavailableReason::NoDeviceCandidate;
    PlacementRequest request;
    std::vector<PlacementRegionCalibration> regionCalibrations;
    PlanPlacementEstimate estimate;

    [[nodiscard]] bool available() const noexcept
    {
        return unavailableReason == RuntimePlacementUnavailableReason::None;
    }
};

// Builds the first bounded runtime request shape: one ordered reader-to-writer
// chain with exact unchanged cardinality through every filter. profile must be
// the exact core profile returned by placementCalibrationFor(), not a caller
// copy or a partial hardware match. This adapter evaluates the calibrated
// model but does not certify that a caller's executor matches that calibration.
// All failures are typed and imply host execution; this function never launches
// CUDA work.
[[nodiscard]] RuntimePlacementResult
buildRuntimePlacement(const Plan& plan, const RuntimePlacementFacts& facts,
                      const PlacementCalibrationProfile& profile);

// B0156/D0214: the subset of `buildRuntimePlacement`'s refusals that depend on
// the compiled plan alone — topology and cardinality — decided without touching
// CUDA. Callers run this before device and profile discovery so a pipeline that
// can never be placed declines for free instead of paying ~0.175 s of runtime
// initialisation first. Returns `None` when the plan is structurally
// admissible; a non-`None` reason is exactly what `buildRuntimePlacement`
// would have returned for the same plan.
[[nodiscard]] RuntimePlacementUnavailableReason
planStructureRefusal(const Plan& plan) noexcept;

} // namespace pdg
