#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace pdg
{

class Plan;
struct PlacementRequest;

enum class PlacementChoice
{
    Host,
    Device
};

enum class PlacementReason
{
    DeviceFaster,
    HostFasterOrEqual,
    DeviceMemoryBudgetExceeded,
    UncalibratedStage,
    OutsideCalibrationEnvelope,
    MissingRecordLayout,
    SharedDeviceTollNotAmortized,
    UnsupportedPlanTopology,
    NoDeviceStages
};

// Infrastructure coefficients are supplied from an append-only calibration
// record. The model deliberately has no fitting or autotuning behavior.
struct PlacementModelCoefficients
{
    double cudaStartupNanoseconds = 0.0;
    double hostToDeviceNanosecondsPerByte = 0.0;
    double deviceToHostNanosecondsPerByte = 0.0;
    double packingNanosecondsPerByte = 0.0;
    double indexBuildNanosecondsPerByte = 0.0;
    double synchronizationNanoseconds = 0.0;
};

// Stage coefficients remain descriptor/catalog calibration data rather than
// planner case analysis. Fixed terms are per stage invocation; point terms
// are charged against the declared input cardinality.
struct StagePlacementCost
{
    double hostFixedNanoseconds = 0.0;
    double deviceFixedNanoseconds = 0.0;
    double hostNanosecondsPerPoint = 0.0;
    double deviceNanosecondsPerPoint = 0.0;
    std::size_t minimumDevicePointCount = 0;
    std::size_t maximumDevicePointCount =
        (std::numeric_limits<std::size_t>::max)();
    bool calibrated = false;
};

// A calibration profile is deliberately keyed to the hardware and compiler
// environment that produced its coefficients. Unknown or partially matching
// devices fail closed; adding a GPU means appending a physically measured
// profile, not extrapolating the SM-89 fit.
struct PlacementDeviceKey
{
    std::string_view name;
    std::string_view computeCapability;
    std::string_view driverVersion;
    std::string_view cudaToolkitVersion;
};

struct PlacementStageCalibration
{
    std::string_view name;
    StagePlacementCost cost;
};

// A concrete executor may replace the calibration-default boundary formulas
// with measured physical work. One fact describes the per-point terms for a
// stable planner boundary ID. An empty fact table preserves the frozen B3
// calibration model; a nonempty table must cover every planner boundary
// exactly once.
struct PlacementBoundaryExecutionFact
{
    std::size_t boundaryId = 0;
    std::size_t transferBytesPerPoint = 0;
    std::size_t packingBytesPerPoint = 0;
    std::size_t deviceStagingBytesPerPoint = 0;
};

struct PlacementCalibrationProfile
{
    std::string_view id;
    PlacementDeviceKey device;
    PlacementModelCoefficients coefficients;
    std::span<const PlacementStageCalibration> stageModels;
};

struct PlacementRegionCalibration
{
    std::size_t residentRegion = 0;
    std::string_view model;
};

[[nodiscard]] const PlacementCalibrationProfile*
placementCalibrationFor(const PlacementDeviceKey& device) noexcept;

[[nodiscard]] const StagePlacementCost*
placementStageCalibration(const PlacementCalibrationProfile& profile,
                          std::string_view name) noexcept;

// Applies exactly one measured residual model to each resident region. The
// residual is charged once at the region's first device stage; later stages
// are marked calibrated with zero incremental residual so normal linear terms
// (transfer, packing, index, synchronization, and memory) remain planner-owned.
// Any missing/duplicate region or unknown model leaves every device stage
// uncalibrated and returns false, making evaluation select host.
[[nodiscard]] bool applyPlacementRegionCalibrations(
    const Plan& plan, const PlacementCalibrationProfile& profile,
    std::span<const PlacementRegionCalibration> calibrations,
    PlacementRequest& request);

struct PlacementRequest
{
    // One entry per planned stage. Inputs are point visits; outputs drive
    // downstream transfer/cardinality. Capacity bounds device working sets
    // and may be smaller than total cardinality for a tiled stage.
    std::vector<std::size_t> stageInputPointCounts;
    std::vector<std::size_t> stageOutputPointCounts;
    std::vector<std::size_t> stagePointCapacities;
    std::vector<std::size_t> stageScratchBytes;
    // Additional stage-owned synchronization points that are not already
    // represented by an upload or spill. Plan-region placement uses this
    // vector so costs remain attributable across host boundaries.
    std::vector<std::size_t> stageAdditionalSynchronizations;
    std::vector<StagePlacementCost> stageCosts;
    std::vector<PlacementBoundaryExecutionFact> boundaryExecutionFacts;
    // A concrete whole-view executor may own a proven allocation topology
    // that aliases a stage result with scratch and maps boundary records on
    // the host. In that case the generic column/result/staging sum would
    // double-count its memory. Nonzero values replace only the corresponding
    // memory diagnostics and feasibility term; transfer and stage costs stay
    // planner-derived. These overrides are valid only for an intrinsic
    // single-lane executor with a complete boundary fact table.
    std::size_t executorUntiledDeviceBytes = 0;
    std::size_t executorPeakDeviceBytes = 0;
    // Configured width of the concrete executor represented by a nonempty
    // boundary fact table. The model derives the active width from tile
    // count, but never silently downshifts below that calibrated width merely
    // to fit VRAM. Zero is valid only for the calibration-default model.
    std::size_t executorLaneCount = 0;
    // A fixed whole-view executor may declare its measured intrinsic width of
    // one without weakening the generic scheduler's swept [2, 6] contract.
    bool intrinsicSingleLaneExecutor = false;
    std::size_t inputRecordBytes = 0;
    std::size_t outputRecordBytes = 0;
    std::size_t fallbackRecordBytes = 0;
    std::size_t deviceMemoryBudgetBytes =
        (std::numeric_limits<std::size_t>::max)();
    std::size_t additionalSynchronizations = 0;
    bool cudaContextWarm = false;
};

struct PlacementCostBreakdown
{
    double stageNanoseconds = 0.0;
    double startupNanoseconds = 0.0;
    double transferNanoseconds = 0.0;
    double packingNanoseconds = 0.0;
    double indexBuildNanoseconds = 0.0;
    double synchronizationNanoseconds = 0.0;
    double totalNanoseconds = 0.0;
};

struct PlacementBoundaryEstimate
{
    std::size_t boundaryId = 0;
    std::size_t pointCount = 0;
    std::size_t logicalColumnBytes = 0;
    // The complete runtime PointView record represented by this boundary.
    // This is executor observability and may differ from the calibrated raw
    // reader/writer record term included in predictedTransferBytes.
    std::size_t fullRecordBytes = 0;
    std::size_t predictedTransferBytes = 0;
    std::size_t predictedPackingBytes = 0;
};

struct PlacementEstimate
{
    PlacementChoice choice = PlacementChoice::Host;
    PlacementReason reason = PlacementReason::NoDeviceStages;
    PlacementCostBreakdown host;
    PlacementCostBreakdown device;
    std::size_t hostToDeviceBytes = 0;
    std::size_t deviceToHostBytes = 0;
    std::size_t packingBytes = 0;
    std::size_t indexBuildBytes = 0;
    std::size_t stageResultBytes = 0;
    std::size_t synchronizationCount = 0;
    std::size_t peakDeviceBytes = 0;
    // One-lane whole-view allocation for the concrete executor. This remains
    // zero for the calibration-default model and is diagnostic evidence that
    // a selected bounded schedule is genuinely required by its VRAM budget.
    std::size_t untiledDeviceBytes = 0;
    std::size_t configuredDeviceLaneCount = 0;
    std::size_t activeDeviceLaneCount = 0;
    std::vector<PlacementBoundaryEstimate> boundaries;
    bool calibrationComplete = false;
    bool calibrationEnvelopeSatisfied = false;
    bool layoutComplete = false;
    bool deviceMemoryFeasible = false;
};

struct PlacementRegionEstimate
{
    std::size_t residentRegion = 0;
    std::vector<std::size_t> stageIds;
    PlacementEstimate estimate;
    bool selected = false;
};

struct PlanPlacementEstimate
{
    PlacementChoice choice = PlacementChoice::Host;
    PlacementReason reason = PlacementReason::NoDeviceStages;
    std::vector<PlacementRegionEstimate> regions;
    std::size_t selectedRegionCount = 0;
    // These totals contain only placement-dependent candidate-region work.
    // Reader, writer, and unchanged host-stage costs are common to both
    // choices and cancel from the bounded linear decision.
    PlacementCostBreakdown allHostPlacement;
    PlacementCostBreakdown selectedPlacement;
    std::size_t hostToDeviceBytes = 0;
    std::size_t deviceToHostBytes = 0;
    std::size_t packingBytes = 0;
    std::size_t indexBuildBytes = 0;
    std::size_t stageResultBytes = 0;
    std::size_t synchronizationCount = 0;
    std::size_t peakDeviceBytes = 0;
    std::size_t untiledDeviceBytes = 0;
    std::size_t configuredDeviceLaneCount = 0;
    std::size_t activeDeviceLaneCount = 0;
    std::vector<PlacementBoundaryEstimate> boundaries;
};

[[nodiscard]] PlacementEstimate
evaluatePlacement(const Plan& plan, const PlacementRequest& request,
                  const PlacementModelCoefficients& coefficients);

// Maximal resident regions are independent after their explicit host
// boundaries. The bounded plan rule selects every region with a positive warm
// device benefit only when their combined benefit repays one shared cold-start
// and plan synchronization toll. This is a linear placement rule, not a
// search, optimizer, or autotuner.
[[nodiscard]] PlanPlacementEstimate
evaluatePlanPlacement(const Plan& plan, const PlacementRequest& request,
                      const PlacementModelCoefficients& coefficients);

} // namespace pdg
