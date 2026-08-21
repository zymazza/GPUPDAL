#pragma once

#include <pdg/index/SpatialIndex.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/LabelDuplicates.hpp>
#include <pdg/stages/Neighborhood.hpp>

#include <pdal/PointView.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <pdal/KDIndex.hpp>

namespace pdal
{
class DimRangeList;
namespace expr
{
class AssignStatement;
}
} // namespace pdal

namespace pdal::pdg_detail
{

enum class EigenStatus
{
    Success,
    CovarianceZero,
    SolverFailure
};

struct EigenResult
{
    pdg::EigenSystem3d system;
    EigenStatus status = EigenStatus::Success;
};

// Planner annotations for a maximal run of compatible neighborhood stages.
// Region zero is a one-stage, non-retained request.  A nonzero region keeps
// XYZ, the shared index, and cached eigensystems attached to the PointView
// until its final stage has received its result.
struct CudaNeighborhoodRegion
{
    std::uint64_t id = 0;
    std::uint32_t maximumNeighbors = 0;
    // Nonzero only for a planner-proved adjacent projection pair that shares
    // one ordered max-k result. It is independent of the index envelope.
    std::uint32_t gatherNeighbors = 0;
    double maximumRadius = 0.0;
    std::uint32_t dimensions = 3;
    bool radiusIndex = false;
    // A whole-view resident producer may retain columns without requesting a
    // spatial index. This remains true by default for every established
    // neighborhood wrapper.
    bool indexRequired = true;
    bool reuseExpected = false;
    bool last = true;
    // True only when the planner proved that nothing after this region can
    // consume a PointView KD3 product (the region is followed by the terminal
    // writer alone). The exact host tie repair may then use a private
    // cached-coordinate tree instead of building and publishing an uncached
    // one nobody reads (D0262/B0262). Never inferred at runtime.
    bool terminalSink = false;
};

struct CudaNeighborhoodResults
{
    std::vector<std::uint8_t> status;
};

struct CudaStatisticalOutlierResult
{
    std::size_t inliers = 0;
    std::size_t outliers = 0;
    std::size_t repairedRows = 0;
    std::size_t hostRepairedRows = 0;
    std::size_t deviceRepairedRows = 0;
};

struct CudaRadiusOutlierResult
{
    std::size_t inliers = 0;
    std::size_t outliers = 0;
};

// Internal deterministic backend selector. Every returned kNN configuration
// is derived from and validates the complete host coordinate envelope;
// callers need not rescan it. Test-only force switches exercise both
// traversal implementations without changing public PDAL options.
[[nodiscard]] pdg::UniformGridConfig
selectCudaKnnConfig(const pdg::PointBatch& hostBatch, std::uint8_t dimensions,
                    std::uint32_t neighbors);

// Shared exact radius execution for radius-bounded PDAL stages. The helper
// selects whole-view or capacity-bounded core/ghost tiles, owns the one shared
// grid implementation, and returns results in original PointView order.
[[nodiscard]] bool tryCudaRadiusCounts(PointView& view, double radius,
                                       std::vector<std::uint32_t>& counts,
                                       bool requireCuda);
// Whole-view resident radius-outlier form. Counts are returned in source
// order for the exact host classification finale; a nonterminal producer
// refreshes Classification for the next compatible resident consumer.
[[nodiscard]] bool
tryCudaRadiusOutlier(PointView& view, double radius, int minimumNeighbors,
                     std::uint8_t classification, CudaNeighborhoodRegion region,
                     CudaRadiusOutlierResult& result, bool requireCuda);
[[nodiscard]] bool tryCudaRadiusScaledValues(PointView& view, double radius,
                                             double factor,
                                             std::vector<double>& values,
                                             bool requireCuda);
// Whole-view resident form used when a following exact point program consumes
// RadialDensity. The generated column remains device-resident until the final
// bridge and is published only when this producer is terminal.
[[nodiscard]] bool
tryCudaResidentRadiusScaledValues(PointView& view, double radius, double factor,
                                  CudaNeighborhoodRegion region,
                                  bool requireCuda);
[[nodiscard]] bool
tryCudaRadiusAssign(PointView& view, double radius, bool search3d,
                    double maximumAbove, double maximumBelow,
                    const DimRangeList& sourceDomain,
                    const DimRangeList& referenceDomain,
                    std::vector<expr::AssignStatement>& updates,
                    CudaNeighborhoodRegion region, bool requireCuda);

// The only class allowed to access PointView's opaque fork-internal resident
// product.  Keeping the product type out of libpdal avoids coupling upstream
// PointView headers to CUDA or PDG implementation headers.
class ResidentPointViewAccess
{
public:
    [[nodiscard]] static std::shared_ptr<void> get(const PointView& view);
    static void set(PointView& view, std::shared_ptr<void> product);
    static void clear(PointView& view);
    // Stable backing-table identity for exact permutation-aware publishers.
    // The value is source identity, not the current logical PointView index.
    [[nodiscard]] static PointId tableId(const PointView& view, PointId point);
};

// Read-only access to the CLI-owned mapped LAS source on the current execution
// thread. A bounded non-neighborhood wrapper may consume it only after its own
// exact layout/cardinality validation, then mark the same source proof used.
[[nodiscard]] std::span<const std::byte> activeCudaLasSource() noexcept;
void markActiveCudaLasSourceUsed() noexcept;

// Bounded CLI-owned source scope for the exact direct-LAS resident endpoint.
// The mapped bytes outlive the scope; neighborhood construction consumes them
// only on the calling thread and retains ordinary owned host/device columns.
class CudaNeighborhoodLasSourceScope
{
public:
    explicit CudaNeighborhoodLasSourceScope(
        std::span<const std::byte> mappedLasBytes,
        bool deviceOnlyNnDistanceHandoff = false);
    ~CudaNeighborhoodLasSourceScope();

    CudaNeighborhoodLasSourceScope(const CudaNeighborhoodLasSourceScope&) =
        delete;
    CudaNeighborhoodLasSourceScope&
    operator=(const CudaNeighborhoodLasSourceScope&) = delete;

    [[nodiscard]] bool used() const noexcept;
    [[nodiscard]] bool recordSummaryUsed() const noexcept;
    [[nodiscard]] bool hostXyzMirrored() const noexcept;
    [[nodiscard]] bool nnDistanceDeviceOnlyHandoffUsed() const noexcept;
    [[nodiscard]] bool nnDistanceHostRestoreUsed() const noexcept;
    [[nodiscard]] bool nnDistanceAssignmentDeviceColumnReused() const noexcept;
    [[nodiscard]] bool knnGatherReuseUsed() const noexcept;
    [[nodiscard]] bool residentAssignmentExecuted() const noexcept;
    [[nodiscard]] bool hagDelaunayCudaUsed() const noexcept;

private:
    std::span<const std::byte> m_previous;
    bool* m_previousUsed = nullptr;
    bool* m_previousRecordSummaryUsed = nullptr;
    bool* m_previousHostXyzMirrored = nullptr;
    bool m_previousDeviceOnlyNnDistanceHandoff = false;
    bool* m_previousNnDistanceDeviceOnlyHandoffUsed = nullptr;
    bool* m_previousNnDistanceHostRestoreUsed = nullptr;
    bool* m_previousNnDistanceAssignmentDeviceColumnReused = nullptr;
    bool* m_previousKnnGatherReuseUsed = nullptr;
    bool* m_previousResidentAssignmentExecuted = nullptr;
    bool* m_previousHagDelaunayCudaUsed = nullptr;
    bool m_used = false;
    bool m_recordSummaryUsed = false;
    bool m_hostXyzMirrored = false;
    bool m_nnDistanceDeviceOnlyHandoffUsed = false;
    bool m_nnDistanceHostRestoreUsed = false;
    bool m_nnDistanceAssignmentDeviceColumnReused = false;
    bool m_knnGatherReuseUsed = false;
    bool m_residentAssignmentExecuted = false;
    bool m_hagDelaunayCudaUsed = false;
};

// Reproduces pdal::math::computeCovariance and the fixed-size
// SelfAdjointEigenSolver result without relying on private libpdal symbols.
[[nodiscard]] EigenResult computeEigenSystem(const PointView& view,
                                             const PointIdList& neighbors);

// Execute the common exact kNN/covariance/eigensystem backend, project the
// requested output columns on device, and publish only those columns through
// the current host PointView boundary. The caller supplies PDAL's actual
// requested neighbor count, including the query point. Device XYZ, index,
// eigensystems, and projected columns remain attached across one planned run.
[[nodiscard]] bool
tryCudaNormalColumns(PointView& view, std::uint32_t neighbors,
                     CudaNeighborhoodRegion region, bool alwaysUp,
                     std::shared_ptr<const CudaNeighborhoodResults>& result,
                     bool requireCuda);
[[nodiscard]] bool
tryCudaEigenvalueColumns(PointView& view, std::uint32_t neighbors,
                         CudaNeighborhoodRegion region, bool normalize,
                         std::shared_ptr<const CudaNeighborhoodResults>& result,
                         bool requireCuda);
// Verbatim transcription of the pinned MathUtils computeRank chain; the
// wrapper's host fallback and the tie/incomplete repair share it.
[[nodiscard]] std::uint8_t computeRankExact(const PointView& view,
                                            const PointIdList& ids,
                                            double threshold);
// Verbatim transcription of the pinned OptimalNeighborhood per-point
// selection; the wrapper's host fallback and the tie/incomplete repair
// share it.
void computeOptimalExact(const PointView& view, const KD3Index& index,
                         PointId point, std::uint32_t minimumK,
                         std::uint32_t maximumK, std::uint64_t& optimalK,
                         double& optimalRadius);

// Verbatim transcription of NeighborClassifierFilter's per-point vote;
// the wrapper's host fallback and the tie/incomplete repair share it.
[[nodiscard]] std::uint8_t computeNeighborVoteExact(const PointView& view,
                                                    const KD3Index& index,
                                                    PointId point,
                                                    std::uint32_t neighbors);

[[nodiscard]] bool
tryCudaNeighborClassifierColumn(PointView& view, std::uint32_t neighbors,
                                CudaNeighborhoodRegion region,
                                bool requireCuda);

[[nodiscard]] bool tryCudaOptimalNeighborhoodColumns(
    PointView& view, std::uint32_t minimumK, std::uint32_t maximumK,
    CudaNeighborhoodRegion region, bool requireCuda);

[[nodiscard]] bool tryCudaEstimateRankColumn(PointView& view,
                                             std::uint32_t neighbors,
                                             CudaNeighborhoodRegion region,
                                             double threshold,
                                             bool requireCuda);
[[nodiscard]] bool tryCudaApproximateCoplanarColumn(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    double threshold1, double threshold2,
    std::shared_ptr<const CudaNeighborhoodResults>& result, bool requireCuda);
[[nodiscard]] bool tryCudaNnDistanceColumns(PointView& view,
                                            std::uint32_t neighbors,
                                            CudaNeighborhoodRegion region,
                                            pdg::KnnDistanceMode mode,
                                            bool requireCuda);
[[nodiscard]] bool tryCudaStatisticalOutlier(
    PointView& view, std::uint32_t neighbors, double multiplier,
    std::uint8_t classification, CudaNeighborhoodRegion region,
    CudaStatisticalOutlierResult& result, bool requireCuda);
[[nodiscard]] bool tryCudaHagNnColumn(PointView& view,
                                      const pdg::HagNnProgram& program,
                                      CudaNeighborhoodRegion region,
                                      bool requireCuda);
[[nodiscard]] bool
tryCudaHagDelaunayColumn(PointView& view,
                         const pdg::HagDelaunayProgram& program,
                         CudaNeighborhoodRegion region, bool requireCuda);
// After the exact upstream host fallback handles a data-dependent HAG
// ambiguity, retain its published HAG column with the planner-owned 2D index
// so a following resident assignment/ferry bridge remains composable.
[[nodiscard]] bool retainCudaHagHostColumn(PointView& view,
                                           std::uint32_t neighbors,
                                           CudaNeighborhoodRegion region,
                                           bool requireCuda);
// `neighbors` is the stage's already-incremented minpts (self-inclusive).
// Distance ties are repaired on host over the exact affected closure: an
// ambiguous own row rebuilds its density and factor from the compatibility
// index, and any row that reaches an ambiguous row rebuilds its factor.
[[nodiscard]] bool tryCudaLofColumns(PointView& view, std::uint32_t neighbors,
                                     CudaNeighborhoodRegion region,
                                     bool requireCuda);
[[nodiscard]] bool tryCudaCovarianceFeatureColumns(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    pdg::EigenvalueMode mode, std::uint32_t features,
    std::shared_ptr<const CudaNeighborhoodResults>& result, bool requireCuda);

// Execute an assignment/ferry-only bridge directly against the retained
// neighborhood PointBatch. Standard feature columns already on the device are
// consumed without re-upload. A non-terminal bridge retains the device XYZ,
// index, and columns for the following compatible neighborhood stage.
[[nodiscard]] bool tryCudaResidentAssignments(PointView& view,
                                              CudaNeighborhoodRegion region,
                                              const pdg::AssignProgram& program,
                                              bool requireCuda);
[[nodiscard]] bool
tryCudaLabelDuplicatesColumn(PointView& view,
                             const pdg::LabelDuplicatesProgram& program,
                             const pdg::DimensionRegistry& dimensions,
                             CudaNeighborhoodRegion region, bool requireCuda);
void clearCudaNeighborhood(PointView& view);

} // namespace pdal::pdg_detail
