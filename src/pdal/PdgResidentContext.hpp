#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/PackedPointBatch.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/index/RasterGrid.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace pdal
{
class PointView;
class PointLayout;

namespace pdg_detail
{

enum class ResidentBoundaryDirection
{
    Upload,
    Spill
};

// Aggregate wall time the executor spent on each host-side boundary phase.
// This is diagnostic observability for the D4 boundary surface; it is never
// compared against oracle output.
struct ResidentPhaseSeconds
{
    // Serializing PointView records into pinned upload tiles, and gathering
    // whole-view attach columns out of the PointView.
    double uploadPack = 0.0;
    // Blocking on lane completion events or attach result transfers before
    // a drain may read.
    double spillWait = 0.0;
    // Publishing written columns, survivor appends, and attach result
    // columns into the PointView.
    double spillPublish = 0.0;
    // Exact compatibility repair after a bounded device proof declines rows.
    // This includes any compatibility-index build, ordered host recomputation,
    // and refresh of repaired columns back to a retained device product.
    double exactHostRepair = 0.0;
    // Planner-composable exact repair performed from the already-resident
    // coordinates without constructing another spatial index.
    double exactDeviceRepair = 0.0;
    std::size_t ambiguousRepairRows = 0U;
    // The following three counters belong exclusively to exact host repair.
    std::size_t incompleteRepairRows = 0U;
    std::size_t repairedRows = 0U;
    // Exact HAG-NN repair may lazily acquire one ground-domain KD2 product.
    // It is owned by the planner-scoped resident neighborhood, never by the
    // stage wrapper, and is built at most once for that region.
    std::size_t hagNnCompatibilityIndexBuilds = 0U;
    // LOF keeps its compatibility KD3 index singular and parallelizes only
    // independent exact row recomputation between the three pass barriers.
    std::size_t lofRepairWorkers = 0U;
    std::size_t lofParallelRepairRows = 0U;
    std::size_t lofCoordinateCacheRows = 0U;
    std::size_t deviceIncompleteRepairRows = 0U;
    std::size_t deviceRepairRows = 0U;
    // NNDistance attribution is explicit so proof-bearing benchmarks can
    // distinguish its bounded repair from another consumer in the same
    // shared-index region.
    double nnDistanceExactHostRepair = 0.0;
    double nnDistanceExactDeviceRepair = 0.0;
    std::size_t nnDistanceHostIncompleteRepairRows = 0U;
    std::size_t nnDistanceDeviceIncompleteRepairRows = 0U;
    std::size_t nnDistanceHostRepairRows = 0U;
    std::size_t nnDistanceDeviceRepairRows = 0U;
    std::size_t nnDistanceParallelDeviceRepairRows = 0U;
    // Statistical-outlier attribution is explicit because a single
    // incomplete mean row can otherwise hide a full compatibility-index
    // build inside the aggregate repair totals.
    double outlierExactHostRepair = 0.0;
    double outlierExactDeviceRepair = 0.0;
    std::size_t outlierHostIncompleteRepairRows = 0U;
    std::size_t outlierDeviceIncompleteRepairRows = 0U;
    std::size_t outlierHostRepairRows = 0U;
    std::size_t outlierDeviceRepairRows = 0U;
    std::size_t outlierParallelDeviceRepairRows = 0U;
    // Approximate-coplanar attribution is explicit because its exact
    // eigensystem repair can otherwise hide a host KD3 build and a complete
    // eigensystem round trip behind the aggregate counters.
    double approximateCoplanarExactHostRepair = 0.0;
    std::size_t approximateCoplanarRepairTriggers = 0U;
    std::size_t approximateCoplanarAmbiguousRepairRows = 0U;
    std::size_t approximateCoplanarIncompleteRepairRows = 0U;
    std::size_t approximateCoplanarRepairRows = 0U;
    std::size_t approximateCoplanarKd3Uses = 0U;
    std::size_t approximateCoplanarDeviceToHostRepairBytes = 0U;
    std::size_t approximateCoplanarHostToDeviceRepairBytes = 0U;
    // Neighbor-classifier attribution makes its conservative distance-tie
    // repair visible without conflating the device vote with a pinned KD3
    // rebuild and ordered host recomputation.
    double neighborClassifierExactHostRepair = 0.0;
    std::size_t neighborClassifierAmbiguousRepairRows = 0U;
    std::size_t neighborClassifierIncompleteRepairRows = 0U;
    std::size_t neighborClassifierRepairRows = 0U;
    std::size_t neighborClassifierKd3Uses = 0U;
};

// Stats-only host-wall call spans nested inside one resident wrapper interval.
// The four fields are mutually exclusive. Boundary and repair counters above
// are nested drill-downs inside the outer wrapper interval and therefore are
// not added to these spans; LOF repair is also inside query/projection.
struct ResidentNnDistanceDetailSeconds
{
    double outputPreparation = 0.0;
    double querySubmission = 0.0;
    double statusAllocation = 0.0;
    double resultTransferCall = 0.0;
    double statusTransferCall = 0.0;
    double explicitStreamWait = 0.0;
    double statusScanAndRepair = 0.0;
    double outputPublication = 0.0;
};

// Stats-only host-wall call spans nested inside one eigen-family
// (normal/eigenvalue/covariance/coplanar) wrapper interval. Mutually exclusive
// among themselves; the whole wrapper is neighborhoodQueryProjection.
struct ResidentEigenFamilyDetailSeconds
{
    double eigenSystemsSubmission = 0.0;
    double ambiguousRepair = 0.0;
    double ambiguousRepairStatusWait = 0.0;
    double ambiguousRepairSystemDownload = 0.0;
    double ambiguousRepairIndexBuild = 0.0;
    double ambiguousRepairRows = 0.0;
    double ambiguousRepairUpload = 0.0;
    std::size_t ambiguousRows = 0U;
    std::size_t incompleteRows = 0U;
    std::size_t repairedRows = 0U;
    double outputPreparation = 0.0;
    double projectionAndCopy = 0.0;
    double statusScan = 0.0;
    double transcendentalFeatures = 0.0;
    double columnPublication = 0.0;
    double hostColumnUpload = 0.0;
};

struct ResidentManagerDetailSeconds
{
    double residentProductSetup = 0.0;
    double directLasCoordinateHydration = 0.0;
    double directLasHydrationAllocation = 0.0;
    double directLasHydrationSubmission = 0.0;
    double directLasHydrationWait = 0.0;
    double indexConfiguration = 0.0;
    double indexConfigSelection = 0.0;
    double indexEnvelopeValidation = 0.0;
    double indexBuild = 0.0;
    double neighborhoodQueryProjection = 0.0;
    double adjacentPointProgramBridge = 0.0;
    ResidentNnDistanceDetailSeconds nnDistance;
    ResidentEigenFamilyDetailSeconds eigenFamily;
};

// Non-overlapping wall-clock partition of one stats-enabled, single-region
// rewritten manager execution. Boundary/repair phase counters are nested
// drill-downs inside residentWrapperIndexFilter.
struct ResidentManagerPhaseSeconds
{
    double managerGraphAndPrepare = 0.0;
    double readerRowTableMaterialization = 0.0;
    double residentWrapperIndexFilter = 0.0;
    double postSpillStageControl = 0.0;
    double total = 0.0;
    ResidentManagerDetailSeconds detail;
    bool complete = false;
};

// Planner-scoped execution state shared by private resident boundary filters
// and the point-program executor. The implementation owns bounded pinned and
// device tile lanes; no state is attached to PointView's neighborhood product.
class ResidentExecutionContext
{
public:
    ResidentExecutionContext(const pdg::Plan& plan,
                             const pdg::DimensionRegistry& planDimensions,
                             std::size_t deviceMemoryBudgetBytes,
                             std::size_t tilePointCapacity);
    ~ResidentExecutionContext();

    ResidentExecutionContext(const ResidentExecutionContext&) = delete;
    ResidentExecutionContext&
    operator=(const ResidentExecutionContext&) = delete;

    void preflight(PointLayout& layout, std::size_t pointCount,
                   std::span<const std::size_t> selectedRegions);

    void enterBoundary(PointView& view, std::size_t boundaryId,
                       ResidentBoundaryDirection direction,
                       std::size_t residentRegion,
                       bool requiresFullPointRecord);

    // A region whose planned stages declare a cardinality change must supply
    // the output view that receives surviving points in stable source order;
    // cardinality-preserving regions must not supply one.
    void beginRegion(PointView& view, std::size_t residentRegion,
                     std::span<const pdg::PackedPointColumn> columns,
                     PointView* compactionOutput = nullptr);
    // A shared-index neighborhood region executes whole-view through the
    // validated attach machinery rather than context tiles; the context still
    // owns its boundaries, planner budget verification, schedule facts, and
    // region events.
    void beginDelegatedRegion(PointView& view, std::size_t residentRegion);
    void endDelegatedRegion(PointView& view, std::size_t residentRegion);
    // A delegated Grid stage obtains both its memory resources and global
    // cell backing from this planner-scoped context. The product request is
    // checked against the selected stage descriptor before any cell storage
    // is allocated.
    [[nodiscard]] pdg::MemoryResource&
    delegatedStagingMemory(PointView& view, std::size_t residentRegion);
    [[nodiscard]] pdg::MemoryResource&
    delegatedDeviceMemory(PointView& view, std::size_t residentRegion);
    [[nodiscard]] pdg::RasterGridProduct&
    acquireRasterGridProduct(PointView& view, std::size_t residentRegion,
                             const pdg::RasterGridFrame& frame,
                             bool reuseExpected);
    [[nodiscard]] std::size_t tileCount() const noexcept;
    pdg::PointBatch& acquireTile(PointView& view, std::size_t tileIndex);
    // Device storage for the declared keep mask of the acquired tile. Valid
    // only between acquireTile and submitTile of a cardinality-changing
    // region.
    [[nodiscard]] std::uint8_t* tileKeepMask(std::size_t tileIndex);
    void beginStage(std::size_t tileIndex, std::size_t stageIndex);
    void endStage(std::size_t tileIndex, std::size_t stageIndex);
    void submitTile(PointView& view, std::size_t tileIndex,
                    pdg::PointBatch& result);
    void endRegion(PointView& view, std::size_t residentRegion);

    [[nodiscard]] const pdg::TiledSchedule& schedule() const noexcept;
    // Cardinality observed after the most recently completed region; equals
    // the input cardinality until a declared cardinality change executes.
    [[nodiscard]] std::size_t observedOutputPointCount() const noexcept;
    [[nodiscard]] const ResidentPhaseSeconds& phaseSeconds() const noexcept;
    // Mutable accumulator for attach-machinery boundary loops that execute
    // outside the context's tile lanes (the shared-index whole-view path).
    [[nodiscard]] ResidentPhaseSeconds& phaseAccumulator() noexcept;
    // Null unless the stats-enabled direct manager timer is armed. Callers
    // therefore avoid even reading the clock on ordinary execution.
    [[nodiscard]] ResidentManagerDetailSeconds*
    managerDetailAccumulator() noexcept;
    void beginManagerPhaseTiming(
        std::chrono::steady_clock::time_point managerStarted,
        std::chrono::steady_clock::time_point executeStarted) noexcept;
    [[nodiscard]] ResidentManagerPhaseSeconds finishManagerPhaseTiming(
        std::chrono::steady_clock::time_point executeEnded) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// The standard PDAL executor is synchronous on one thread. A thread-local
// scope prevents a private marker from accidentally binding to another
// pipeline and guarantees cleanup on every exception path.
class ResidentExecutionScope
{
public:
    ResidentExecutionScope(const pdg::Plan& plan,
                           const pdg::DimensionRegistry& planDimensions,
                           std::size_t deviceMemoryBudgetBytes,
                           std::size_t tilePointCapacity);
    ~ResidentExecutionScope();

    ResidentExecutionScope(const ResidentExecutionScope&) = delete;
    ResidentExecutionScope& operator=(const ResidentExecutionScope&) = delete;

    void preflight(PointLayout& layout, std::size_t pointCount,
                   std::span<const std::size_t> selectedRegions);

    [[nodiscard]] ResidentExecutionContext& context() noexcept;

private:
    ResidentExecutionContext m_context;
    ResidentExecutionContext* m_previous = nullptr;
};

[[nodiscard]] ResidentExecutionContext& requireResidentExecutionContext();

// The active context's boundary phase accumulator, or null when no resident
// execution context is active on this thread. Attach-machinery boundary
// loops record their host wall time here so `--stats` reports one aggregate
// `host_boundary_phase_seconds` surface for every resident executor.
[[nodiscard]] ResidentPhaseSeconds* activeResidentPhaseSeconds() noexcept;

// The active context's stats-only manager detail accumulator, or null when
// this invocation is outside the bounded direct manager timing envelope.
[[nodiscard]] ResidentManagerDetailSeconds*
activeResidentManagerDetailSeconds() noexcept;

} // namespace pdg_detail
} // namespace pdal
