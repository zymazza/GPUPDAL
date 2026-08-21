#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/ColorMap.hpp>
#include <pdg/stages/Csf.hpp>
#include <pdg/stages/Elm.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Ferry.hpp>
#include <pdg/stages/Grouping.hpp>
#include <pdg/stages/LabelDuplicates.hpp>
#include <pdg/stages/Locate.hpp>
#include <pdg/stages/Metadata.hpp>
#include <pdg/stages/Morton.hpp>
#include <pdg/stages/Neighborhood.hpp>
#include <pdg/stages/Ordering.hpp>
#include <pdg/stages/Ordinal.hpp>
#include <pdg/stages/Partition.hpp>
#include <pdg/stages/Pmf.hpp>
#include <pdg/stages/Robust.hpp>
#include <pdg/stages/Skewness.hpp>
#include <pdg/stages/Smrf.hpp>
#include <pdg/stages/Summary.hpp>
#include <pdg/stages/Transformation.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace pdg
{

class PointBatch;

class PlanError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum class StageKind
{
    Pointwise,
    Knn,
    Grid,
    Global,
    Split,
    Cpu
};

enum class StageRole
{
    Reader,
    Filter,
    Writer
};

enum class IndexKind
{
    None,
    Radius,
    Knn
};

struct IndexRequest
{
    IndexKind kind = IndexKind::None;
    double radius = 0.0;
    std::uint32_t neighbors = 0;
    std::uint8_t dimensions = 3;
};

enum class GridFramePolicy
{
    None,
    // Preserve PMF's pinned-oracle split between initial
    // floor(x - minimum) / cell binning and later
    // floor((x - minimum) / cell) lookup binning.
    PmfInitialLookupV1
};

struct GridRequest
{
    GridFramePolicy framePolicy = GridFramePolicy::None;
    double cellSize = 0.0;
    std::size_t deviceBytesPerCell = 0U;
    // Number of complete phase backings represented by
    // deviceBytesPerCell. When the complete frame fits the runtime budget,
    // the Grid product owns these backings for the whole resident region.
    std::size_t deviceBackingCount = 0U;
    // Complete-frame bytes per cell that may exist only while a Grid producer
    // is proving a generation. A successful proof may explicitly promote
    // this storage into the phase backings; failed proofs expose no product.
    std::size_t deviceProofBytesPerCell = 0U;
    std::size_t deviceFixedBytes = 0U;
    std::size_t hostBytesPerPoint = 0U;
    std::size_t hostBytesPerCell = 0U;
    std::size_t hostTileBytesPerExpandedCell = 0U;
    std::size_t maximumHaloCells = 0U;
    // Each stencil iteration mosaics every owner core into a completed
    // global backing before the next iteration begins.
    bool phaseSynchronized = false;
};

enum class WhereMergeMode
{
    NotApplicable,
    Auto,
    MergeSkipped,
    SeparateSkipped,
    Invalid
};

struct FusionSemantics
{
    bool pure = false;
    bool cardinalityPreserving = false;
    std::vector<DimensionId> dimsRead;
    std::vector<DimensionId> dimsWritten;
    // These flags declare that this stage may be embedded in a surrounding
    // consumer prologue or producer epilogue without changing its semantics.
    bool fusableAsPrologue = false;
    bool fusableAsEpilogue = false;
    // Kernel capabilities are separate from point-program legality: a pure
    // stage is not executable inside an anchor unless that anchor explicitly
    // declares the corresponding entry point.
    bool acceptsFusedPrologue = false;
    bool acceptsFusedEpilogue = false;
    // An anchor whose prologue machinery performs declared stable compaction
    // and final-size truncation may host a cardinality-changing chain; the
    // ordered LAS pack/summarize sink is the first such anchor.
    bool acceptsCompactingPrologue = false;
    bool deterministicSafe = false;
    // Most feature prologues may run independent point work but cannot consume
    // a value just written by that work. Writer pack kernels explicitly can.
    bool prologueConsumesPointWrites = false;
    bool hasWhere = false;
    WhereMergeMode whereMerge = WhereMergeMode::NotApplicable;
};

struct StageDescriptor
{
    std::string type;
    // Name of the measured D4 residual model to use when this stage is the
    // first device stage of a resident region.  This is deliberately
    // descriptor data, rather than a runtime placement stage-name switch.
    // An empty name means the stage has no measured residual and must make a
    // runtime placement request fail closed.
    std::string placementModel;
    StageKind kind = StageKind::Cpu;
    std::vector<DimensionId> reads;
    std::vector<DimensionId> writes;
    IndexRequest index;
    GridRequest grid;
    double maximumRadius = 0.0;
    bool mutatesCoordinates = false;
    bool preservesOrder = true;
    // Bytes returned to host to materialize non-column stage results at a
    // spill boundary (for example a keep mask, permutation, or partition
    // membership). These bytes are distinct from resident SoA columns and
    // must not be hidden inside a fitted compute coefficient.
    std::size_t deviceToHostBytesPerInputPoint = 0;
    std::size_t deviceToHostFixedBytes = 0;
    FusionSemantics fusion;
};

enum class FusionPlacement
{
    ProducerEpilogue,
    ConsumerPrologue
};

struct FusionCandidate
{
    std::size_t anchorStage = 0;
    FusionPlacement placement = FusionPlacement::ProducerEpilogue;
    std::vector<std::size_t> pointStages;
    std::vector<DimensionId> dimsRead;
    std::vector<DimensionId> dimsWritten;
    bool deterministicSafe = false;
};

enum class ResidencyBoundaryKind
{
    Upload,
    Spill
};

struct ResidencyBoundary
{
    std::size_t producer = 0;
    std::size_t consumer = 0;
    // All graph consumers served by this one materialized boundary. consumer
    // is the first transition point and remains for compact diagnostics.
    std::vector<std::size_t> consumers;
    ResidencyBoundaryKind kind = ResidencyBoundaryKind::Upload;
    std::vector<DimensionId> dimensions;
    // Columns whose final resident use is this spill. They remain allocated
    // until the asynchronous transfer associated with this boundary is done.
    std::vector<DimensionId> releaseDimensions;
    // Subset of releaseDimensions whose resident value was written in this
    // region and must be repacked into a physical record before the spill.
    // Read-only carried columns remain in releaseDimensions but are excluded.
    std::vector<DimensionId> repackDimensions;
    std::size_t repackBytesPerPoint = 0;
    std::size_t bytesPerPoint = 0;
    bool fallback = false;
    // A non-native stage may inspect layout fields that cannot be discovered
    // from pipeline JSON. Its executor must pack the complete input record in
    // addition to the known live columns listed above.
    bool requiresFullPointRecord = false;
};

inline constexpr std::size_t NoResidentRegion =
    (std::numeric_limits<std::size_t>::max)();

struct FileStagePlan
{
    std::string filename;
    // `writers.las(extra_dims=all)` is still an upstream-owned sink, but the
    // exact option is planner-visible so a separately proved direct publisher
    // can replace only that terminal boundary without broadening writer
    // selection.
    bool extraDimensionsAll = false;
    // B0151/D0213: true when the stage is an option-free LAS-family file stage
    // (`.las` or `.laz`). `native` remains uncompressed-only because it also
    // authorizes memory-mapped record access; this flag authorizes only
    // header-derived placement facts, which a LAZ public header supplies
    // identically. Both require the same option-free shape.
    bool optionFreeLasFamily = false;
};

struct FallbackStagePlan
{
    std::string optionsJson;
};

using StagePayload = std::variant<
    FileStagePlan, FerryProgram, AssignProgram, PredicateProgram,
    OrdinalProgram, LocateProgram, TransformationProgram, RobustProgram,
    OrderingProgram, MortonProgram, GroupByProgram, ReturnsProgram,
    MergeProgram, DividerProgram, SplitterProgram, ColorinterpProgram,
    StatsProgram, InfoProgram, ExpressionStatsProgram, OutlierProgram,
    RadialDensityProgram, RadiusAssignProgram, NnDistanceProgram, HagNnProgram,
    HagDelaunayProgram, LofProgram, LabelDuplicatesProgram, SmrfProgram,
    PmfProgram, CsfProgram, ElmProgram, NormalProgram,
    ApproximateCoplanarProgram, EigenvaluesProgram, CovarianceFeaturesProgram,
    EstimateRankProgram, OptimalNeighborhoodProgram, NeighborClassifierProgram,
    SkewnessProgram, FallbackStagePlan>;

struct PlannedStage
{
    std::size_t id = 0;
    std::string tag;
    std::vector<std::size_t> inputs;
    StageRole role = StageRole::Filter;
    StageDescriptor descriptor;
    StagePayload payload;
    MemoryKind preferredResidency = MemoryKind::Host;
    std::vector<DimensionId> liveBefore;
    std::vector<DimensionId> liveAfter;
    // D1 resident execution contract. The planner owns which SoA columns are
    // present on device at each stage, which allocations must be created, and
    // which become reusable after the stage's outputs have been consumed or
    // spilled at an explicit boundary.
    std::vector<DimensionId> deviceLiveBefore;
    std::vector<DimensionId> deviceLiveAfter;
    std::vector<DimensionId> deviceMaterialize;
    std::vector<DimensionId> deviceRelease;
    std::size_t residentRegion = NoResidentRegion;
    std::size_t deviceColumnBytesPerPoint = 0;
    std::size_t deviceIndexBytesPerPoint = 0;
    // One planner-owned ordered kNN rowset may remain live between adjacent
    // compatible projections. The width is explicit so the rewrite and
    // executor cannot infer reuse from stage names or a larger index envelope.
    std::uint32_t deviceKnnGatherNeighbors = 0U;
    std::size_t deviceQueryBytesPerPoint = 0U;
    std::size_t deviceIndexBuildBytesPerPoint = 0;
    std::size_t deviceIndexReleaseBytesPerPoint = 0;
    std::size_t deviceGridBuildBytesPerCell = 0U;
    std::size_t deviceGridProofBytesPerCell = 0U;
    std::size_t deviceGridFixedBytes = 0U;
    bool native = false;
};

struct PlanSummary
{
    std::vector<DimensionId> touchedDimensions;
    std::size_t bytesPerPoint = 0;
    std::size_t indexBytesPerPoint = 0;
    std::size_t peakDeviceColumnBytesPerPoint = 0;
    std::size_t peakDeviceQueryBytesPerPoint = 0U;
    std::size_t peakDeviceBytesPerPoint = 0;
    std::size_t hostDeviceTransferBytesPerPoint = 0;
    std::size_t hostDeviceTransfers = 0;
    std::size_t spillBoundaries = 0;
    std::size_t fallbackBoundaries = 0;
    std::size_t residentRegions = 0;
    std::size_t indexBuilds = 0;
    std::size_t gridBuilds = 0;
    std::size_t peakDeviceGridBytesPerCell = 0U;
    std::size_t peakDeviceGridProofBytesPerCell = 0U;
    std::size_t peakDeviceGridFixedBytes = 0U;
    double maximumRadius = 0.0;
    bool allStagesNative = false;
    bool deterministic = false;
    std::vector<ResidencyBoundary> residencyBoundaries;
    std::vector<FusionCandidate> fusionCandidates;
    std::vector<std::string> fallbackReasons;
};

class Plan
{
public:
    Plan(std::vector<PlannedStage> stages, PlanSummary summary);

    [[nodiscard]] const std::vector<PlannedStage>& stages() const noexcept;
    [[nodiscard]] const PlanSummary& summary() const noexcept;
    [[nodiscard]] std::size_t
    estimatedDeviceBytes(std::size_t pointCapacity) const;

private:
    std::vector<PlannedStage> m_stages;
    PlanSummary m_summary;
};

struct PlannerOptions
{
    bool strict = false;
    bool deterministic = false;
};

[[nodiscard]] Plan compilePipeline(std::string_view json,
                                   DimensionRegistry& dimensions,
                                   PlannerOptions options = {});

// Runtime hooks for a resident executor. Preparation materializes only the
// planner-declared columns missing from the current batch. Retirement returns
// dead column allocations to the owning stream-ordered pool, where a later
// materialization can reuse them. Boundary retirement is called only after a
// planned spill has completed.
void preparePlannedDeviceColumns(PointBatch& batch, const PlannedStage& stage);
void releasePlannedDeviceColumns(PointBatch& batch, const PlannedStage& stage);
void releaseSpilledDeviceColumns(PointBatch& batch,
                                 const ResidencyBoundary& boundary);

[[nodiscard]] bool pointFusionLegal(const FusionSemantics& pointProgram,
                                    const FusionSemantics& anchor,
                                    FusionPlacement placement,
                                    bool deterministic) noexcept;

} // namespace pdg
