#pragma once

#include <pdg/Memory.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace pdg
{

class PointBatch;
class SpatialIndex;

// Packed upper triangle of the symmetric 3D covariance matrix. Values retain
// PDAL's float-demeaned, sample-covariance arithmetic contract.
struct Covariance3d
{
    double xx = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yy = 0.0;
    double yz = 0.0;
    double zz = 0.0;
};

// Eigenvalues are ascending. Eigenvector column `eigen` is stored at
// vectors[axis * 3 + eigen], matching the coefficient order observed by
// PDAL's neighborhood filters.
struct EigenSystem3d
{
    std::array<double, 3> values{0.0, 0.0, 0.0};
    std::array<double, 9> vectors{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

enum class KnnDistanceMode : std::uint8_t
{
    Kth,
    Average
};

namespace detail
{
struct MortonBvhBounds
{
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
};

void buildUniformGridDevice(SpatialIndex& index);
void radiusCountsDevice(const SpatialIndex& index, double radius,
                        std::uint32_t* counts);
void radiusScaledValuesDevice(const SpatialIndex& index, double radius,
                              double factor, double* values);
void radiusAnyDevice(const SpatialIndex& index, double radius,
                     const std::uint8_t* sourceMask,
                     const std::uint8_t* referenceMask, double maximumAbove,
                     double maximumBelow, std::uint8_t* matches);
void knnGatherDevice(const SpatialIndex& index, std::uint32_t neighbors,
                     std::uint32_t* pointIds, double* squaredDistances,
                     std::uint8_t* status);
void knnGatherMaskedDevice(
    const SpatialIndex& index, std::uint32_t neighbors,
    std::uint32_t referenceCount, const std::uint8_t* sourceMask,
    const std::uint8_t* referenceMask, std::uint32_t* pointIds,
    double* squaredDistances, std::uint8_t* status);
void knnMeanDistancesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            double* means, std::uint8_t* status);
void knnDistanceValuesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                             KnnDistanceMode mode, double* values,
                             std::uint8_t* status);
void projectKnnMeanDistancesDevice(const SpatialIndex& index,
                                   std::uint32_t rowNeighbors,
                                   std::uint32_t meanNeighbors,
                                   const double* squaredDistances,
                                   double* means);
void projectKnnDistanceValuesDevice(const SpatialIndex& index,
                                    std::uint32_t rowNeighbors,
                                    std::uint32_t distanceNeighbors,
                                    KnnDistanceMode mode,
                                    const double* squaredDistances,
                                    double* values);
void repairIncompleteKthDistanceValuesDevice(const SpatialIndex& index,
                                             std::uint32_t neighbors,
                                             double* values,
                                             const std::uint32_t* queryIds,
                                             std::size_t queryCount,
                                             bool parallelRepair);
void repairIncompleteMeanDistancesDevice(const SpatialIndex& index,
                                         std::uint32_t neighbors,
                                         double* values,
                                         const std::uint32_t* queryIds,
                                         std::size_t queryCount,
                                         bool parallelRepair);
void knnCovariancesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                          Covariance3d* covariances, std::uint8_t* status);
void knnEigenSystemsDevice(const SpatialIndex& index, std::uint32_t neighbors,
                           EigenSystem3d* systems, std::uint8_t* status);
void copyCoordinateColumnsToHost(const SpatialIndex& index, double* hostX,
                                 double* hostY, double* hostZ);
void knnAdjacencyHostDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            std::uint32_t* hostIds,
                            double* hostSquaredDistances,
                            std::uint8_t* hostStatus);
void knnRankCovariancesDevice(const SpatialIndex& index,
                              std::uint32_t neighbors,
                              Covariance3d* hostCovariances,
                              std::uint8_t* hostStatus);
void knnNeighborVotesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            const std::uint8_t* values, std::uint8_t* results,
                            std::uint8_t* status);
void knnLofValuesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                        double* kDistances, double* reachabilityDensities,
                        double* outlierFactors, std::uint8_t* status,
                        std::uint8_t* neighborStatus,
                        std::uint8_t* neighborNeighborStatus);
} // namespace detail

inline constexpr std::uint8_t KnnExact = 0U;
// D0271: never reported under the relaxed tie-order contract (`gpupdal --fast`,
// pdg::knnStatusMask()); the index's own tie choice then stands for every
// consumer and no exact tie repair runs.
inline constexpr std::uint8_t KnnDistanceTie = 1U;
inline constexpr std::uint8_t KnnSearchIncomplete = 2U;
inline constexpr std::uint8_t KnnCovarianceZero = 4U;
inline constexpr std::uint8_t KnnEigenFailure = 8U;
// Set by a downstream feature projection when PDAL would reject an otherwise
// solved eigensystem (currently the all-zero clamped feature spectrum).
inline constexpr std::uint8_t KnnFeatureInvalid = 16U;

enum class SpatialIndexBackend : std::uint8_t
{
    UniformGrid,
    MortonBvh
};

inline constexpr std::size_t KnnConfigMaximumProbePoints = 8192U;

// Returns the source point used by one established deterministic adaptive
// probe. Exposing the mapping keeps compact point-record summaries and the
// PointBatch selector on one exact sampling contract.
[[nodiscard]] std::size_t knnConfigProbePoint(std::size_t pointCount,
                                              std::size_t probe);

struct UniformGridConfig
{
    std::uint8_t dimensions = 3;
    double cellSize = 0.0;
    std::array<double, 3> origin{0.0, 0.0, 0.0};
    std::array<std::uint32_t, 3> maximumCell{0, 0, 0};
    SpatialIndexBackend backend = SpatialIndexBackend::UniformGrid;
    // Set by the kNN config builders: a device uniform-grid build then also
    // gathers the sorted-order candidate coordinate copies (three doubles
    // plus three prefilter floats per point) that the exact kNN kernels
    // require. Radius-only indexes never pay for them.
    bool knnCandidateArrays = false;
};

// Complete finite coordinate envelope plus the deterministic adaptive-probe
// coordinates needed to choose a planner-owned kNN index. Sources that already
// own a compact point-record representation can construct this summary without
// allocating full host XYZ columns. Probe order is the established
// deterministicProbe order used by makeAdaptiveKnnConfig(PointBatch, ...).
struct KnnConfigSummary
{
    std::size_t pointCount = 0;
    std::uint8_t dimensions = 3;
    std::array<double, 3> minimum{0.0, 0.0, 0.0};
    std::array<double, 3> maximum{0.0, 0.0, 0.0};
    std::vector<std::array<double, 3>> probes;
};

// Builds the exact grid frame from finite logical-double coordinates. The
// cell edge is normally the largest radius requested by a planned native
// region, so every radius query visits at most 3^dimensions cells.
[[nodiscard]] UniformGridConfig
makeUniformGridConfig(const PointBatch& hostBatch, std::uint8_t dimensions,
                      double cellSize);
[[nodiscard]] UniformGridConfig
makeUniformGridConfig(const KnnConfigSummary& summary, double cellSize);

// Chooses a bounded uniform-grid frame for an exact k-nearest candidate
// search. The current primitive supports one through 64 requested neighbors.
// Dense/adaptive point sets that exceed the bounded shell search are reported
// per point rather than being published as exact results.
[[nodiscard]] UniformGridConfig makeKnnGridConfig(const PointBatch& hostBatch,
                                                  std::uint8_t dimensions,
                                                  std::uint32_t neighbors);

// Summary overloads preserve the PointBatch builders' arithmetic and backend
// choice without requiring complete host coordinate columns.
[[nodiscard]] UniformGridConfig
makeKnnGridConfig(const KnnConfigSummary& summary, std::uint32_t neighbors);

// Builds the same 63-bit Morton frame at maximum useful resolution and marks
// it for an implicit linear BVH over the stable Morton point order. Unlike the
// bounded grid-shell traversal, this backend proves exact completion across
// sparse gaps and mixed-density point sets.
[[nodiscard]] UniformGridConfig makeMortonBvhConfig(const PointBatch& hostBatch,
                                                    std::uint8_t dimensions);
[[nodiscard]] UniformGridConfig
makeMortonBvhConfig(const KnnConfigSummary& summary);

// Selects the compact uniform grid for ordinary density and the Morton BVH
// when a bounded deterministic occupancy probe finds broad clustering and an
// estimated hot-cell population above the measured grid/BVH crossover.
// Selection affects only traversal; exact distance, ordering, tie, and status
// contracts are shared.
[[nodiscard]] UniformGridConfig
makeAdaptiveKnnConfig(const PointBatch& hostBatch, std::uint8_t dimensions,
                      std::uint32_t neighbors);
[[nodiscard]] UniformGridConfig
makeAdaptiveKnnConfig(const KnnConfigSummary& summary,
                      std::uint32_t neighbors);

// The current exact device envelope is finite logical-double XYZ (XY for a
// two-dimensional index), at most INT_MAX points, and no more than 21 cell
// bits per axis. The grid only selects candidates; consumers retain double
// distance arithmetic for compatibility.
[[nodiscard]] bool
uniformGridMaySupportExactDevice(const PointBatch& batch,
                                 const UniformGridConfig& config) noexcept;

// Planner/tile-owned persistent compact cell table. Coordinate-mutating
// stages invalidate this object; subsequent neighborhood consumers rebuild it
// once and share the resulting sorted point and cell arrays.
class SpatialIndex
{
public:
    SpatialIndex(PointBatch& batch, UniformGridConfig config);
    ~SpatialIndex();

    SpatialIndex(const SpatialIndex&) = delete;
    SpatialIndex& operator=(const SpatialIndex&) = delete;
    SpatialIndex(SpatialIndex&&) noexcept;
    SpatialIndex& operator=(SpatialIndex&&) noexcept;

    void build();
    void invalidate() noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::size_t buildCount() const noexcept;
    [[nodiscard]] std::size_t cellCount() const noexcept;
    [[nodiscard]] std::size_t allocatedBytes() const noexcept;
    [[nodiscard]] MemoryKind memoryKind() const noexcept;
    [[nodiscard]] const UniformGridConfig& config() const noexcept;
    [[nodiscard]] const PointBatch& batch() const noexcept;

private:
    void allocateStorage();
    void allocateBvhStorage();

    friend void detail::buildUniformGridDevice(SpatialIndex& index);
    friend void detail::radiusCountsDevice(const SpatialIndex& index,
                                           double radius,
                                           std::uint32_t* counts);
    friend void detail::radiusScaledValuesDevice(const SpatialIndex& index,
                                                 double radius, double factor,
                                                 double* values);
    friend void detail::radiusAnyDevice(
        const SpatialIndex& index, double radius,
        const std::uint8_t* sourceMask, const std::uint8_t* referenceMask,
        double maximumAbove, double maximumBelow, std::uint8_t* matches);
    friend void detail::knnGatherDevice(const SpatialIndex& index,
                                        std::uint32_t neighbors,
                                        std::uint32_t* pointIds,
                                        double* squaredDistances,
                                        std::uint8_t* status);
    friend void detail::knnGatherMaskedDevice(
        const SpatialIndex& index, std::uint32_t neighbors,
        std::uint32_t referenceCount, const std::uint8_t* sourceMask,
        const std::uint8_t* referenceMask, std::uint32_t* pointIds,
        double* squaredDistances, std::uint8_t* status);
    friend void detail::knnMeanDistancesDevice(const SpatialIndex& index,
                                               std::uint32_t neighbors,
                                               double* means,
                                               std::uint8_t* status);
    friend void detail::knnDistanceValuesDevice(const SpatialIndex& index,
                                                std::uint32_t neighbors,
                                                KnnDistanceMode mode,
                                                double* values,
                                                std::uint8_t* status);
    friend void detail::projectKnnMeanDistancesDevice(
        const SpatialIndex& index, std::uint32_t rowNeighbors,
        std::uint32_t meanNeighbors, const double* squaredDistances,
        double* means);
    friend void detail::projectKnnDistanceValuesDevice(
        const SpatialIndex& index, std::uint32_t rowNeighbors,
        std::uint32_t distanceNeighbors, KnnDistanceMode mode,
        const double* squaredDistances, double* values);
    friend void detail::repairIncompleteKthDistanceValuesDevice(
        const SpatialIndex& index, std::uint32_t neighbors, double* values,
        const std::uint32_t* queryIds, std::size_t queryCount,
        bool parallelRepair);
    friend void detail::repairIncompleteMeanDistancesDevice(
        const SpatialIndex& index, std::uint32_t neighbors, double* values,
        const std::uint32_t* queryIds, std::size_t queryCount,
        bool parallelRepair);
    friend void detail::knnCovariancesDevice(const SpatialIndex& index,
                                             std::uint32_t neighbors,
                                             Covariance3d* covariances,
                                             std::uint8_t* status);
    friend void detail::knnEigenSystemsDevice(const SpatialIndex& index,
                                              std::uint32_t neighbors,
                                              EigenSystem3d* systems,
                                              std::uint8_t* status);
    friend void detail::copyCoordinateColumnsToHost(const SpatialIndex& index,
                                                    double* hostX,
                                                    double* hostY,
                                                    double* hostZ);
    friend void detail::knnAdjacencyHostDevice(const SpatialIndex& index,
                                               std::uint32_t neighbors,
                                               std::uint32_t* hostIds,
                                               double* hostSquaredDistances,
                                               std::uint8_t* hostStatus);
    friend void detail::knnRankCovariancesDevice(const SpatialIndex& index,
                                                 std::uint32_t neighbors,
                                                 Covariance3d* hostCovariances,
                                                 std::uint8_t* hostStatus);
    friend void detail::knnNeighborVotesDevice(const SpatialIndex& index,
                                               std::uint32_t neighbors,
                                               const std::uint8_t* values,
                                               std::uint8_t* results,
                                               std::uint8_t* status);
    friend void detail::knnLofValuesDevice(
        const SpatialIndex& index, std::uint32_t neighbors, double* kDistances,
        double* reachabilityDensities, double* outlierFactors,
        std::uint8_t* status, std::uint8_t* neighborStatus,
        std::uint8_t* neighborNeighborStatus);
    friend void radiusCounts(const SpatialIndex& index, double radius,
                             std::uint32_t* counts);
    friend void radiusScaledValues(const SpatialIndex& index, double radius,
                                   double factor, double* values);
    friend void radiusAny(const SpatialIndex& index, double radius,
                          const std::uint8_t* sourceMask,
                          const std::uint8_t* referenceMask,
                          double maximumAbove, double maximumBelow,
                          std::uint8_t* matches);
    friend void knnGather(const SpatialIndex& index, std::uint32_t neighbors,
                          std::uint32_t* pointIds, double* squaredDistances,
                          std::uint8_t* status);
    friend void knnGatherMasked(
        const SpatialIndex& index, std::uint32_t neighbors,
        std::uint32_t referenceCount, const std::uint8_t* sourceMask,
        const std::uint8_t* referenceMask, std::uint32_t* pointIds,
        double* squaredDistances, std::uint8_t* status);
    friend void knnMeanDistances(const SpatialIndex& index,
                                 std::uint32_t neighbors, double* means,
                                 std::uint8_t* status);
    friend void knnDistanceValues(const SpatialIndex& index,
                                  std::uint32_t neighbors, KnnDistanceMode mode,
                                  double* values, std::uint8_t* status);
    friend void projectKnnMeanDistances(
        const SpatialIndex& index, std::uint32_t rowNeighbors,
        std::uint32_t meanNeighbors, const double* squaredDistances,
        double* means);
    friend void projectKnnDistanceValues(
        const SpatialIndex& index, std::uint32_t rowNeighbors,
        std::uint32_t distanceNeighbors, KnnDistanceMode mode,
        const double* squaredDistances, double* values);
    friend void knnCovariances(const SpatialIndex& index,
                               std::uint32_t neighbors,
                               Covariance3d* covariances, std::uint8_t* status);
    friend void knnEigenSystems(const SpatialIndex& index,
                                std::uint32_t neighbors, EigenSystem3d* systems,
                                std::uint8_t* status);
    friend void knnRankValues(const SpatialIndex& index,
                              std::uint32_t neighbors, double threshold,
                              std::uint8_t* ranks, std::uint8_t* status);
    friend void knnOptimalValues(const SpatialIndex& index,
                                 std::uint32_t minimumK, std::uint32_t maximumK,
                                 std::uint64_t* optimalK, double* optimalRadius,
                                 std::uint8_t* status);
    friend void knnNeighborVotes(const SpatialIndex& index,
                                 std::uint32_t neighbors,
                                 const std::uint8_t* values,
                                 std::uint8_t* results, std::uint8_t* status);
    friend void knnLofValues(const SpatialIndex& index, std::uint32_t neighbors,
                             double* kDistances, double* reachabilityDensities,
                             double* outlierFactors, std::uint8_t* status,
                             std::uint8_t* neighborStatus,
                             std::uint8_t* neighborNeighborStatus);

    PointBatch* m_batch = nullptr;
    UniformGridConfig m_config;
    bool m_valid = false;
    std::size_t m_buildCount = 0;
    std::size_t m_cellCount = 0;
    std::unique_ptr<Allocation> m_sortedKeys;
    std::unique_ptr<Allocation> m_sortedPointIds;
    std::unique_ptr<Allocation> m_uniqueKeys;
    std::unique_ptr<Allocation> m_cellOffsets;
    std::unique_ptr<Allocation> m_cellCounts;
    // Device-only copy of the batch coordinates gathered into sorted-point
    // order (x, then y, then z, each `capacity` doubles). Candidate loops
    // read it contiguously instead of gathering through sortedPointIds; the
    // stored doubles are bit-identical to the batch columns.
    std::unique_ptr<Allocation> m_sortedCoordinates;
    // Grid-origin-shifted float copies in the same order (x, y, z, each
    // `capacity` floats). They feed only the certified distance prefilter,
    // whose rounding margin makes skipping provably identical to the exact
    // double path; they never contribute a stored output bit.
    std::unique_ptr<Allocation> m_sortedFloatCoordinates;
    std::size_t m_bvhLeafBase = 0;
    std::unique_ptr<Allocation> m_bvhBounds;
};

// Emits the number of points whose squared distance is strictly less than
// radius^2 for every source point. The source point is therefore included for
// positive radii, matching PDAL's KD2Index/KD3Index radius contract.
void radiusCounts(const SpatialIndex& index, double radius,
                  std::uint32_t* counts);

// Applies one binary64 multiplication to each exact radius count. This is the
// reusable projection used by filters.radialdensity; callers compute and pass
// the stage's factor so its literal operation order remains explicit.
void radiusScaledValues(const SpatialIndex& index, double radius, double factor,
                        double* values);

// Emits one when a source point has any reference point at a squared distance
// strictly below radius^2. Null masks mean every point. Two-dimensional
// indexes optionally reject reference points farther than maximumAbove or
// maximumBelow in Z; a negative maximum disables that side. Three-dimensional
// indexes ignore the vertical caps. The query is the reusable selection
// primitive for filters.radiusassign and never constructs a private index.
void radiusAny(const SpatialIndex& index, double radius,
               const std::uint8_t* sourceMask,
               const std::uint8_t* referenceMask, double maximumAbove,
               double maximumBelow, std::uint8_t* matches);

// Emits `neighbors` entries per source point, ordered by increasing squared
// distance. Rows have a fixed `neighbors` stride. When the input has fewer
// points than requested, unused entries are UINT32_MAX/infinity. A nonzero
// status means a consumer must use its compatibility path: distance ties make
// nanoflann traversal order observable, while an incomplete status means the
// bounded grid search could not prove that every closer cell was visited.
void knnGather(const SpatialIndex& index, std::uint32_t neighbors,
               std::uint32_t* pointIds, double* squaredDistances,
               std::uint8_t* status);

// The same exact kNN query restricted to a planner-owned reference domain.
// A null source mask queries every row; a zero source row emits the sentinel
// id/infinite distance with KnnExact. The non-null reference mask must contain
// exactly referenceCount selected rows. Retaining one candidate beyond the
// requested width exposes an equal-distance boundary through KnnDistanceTie.
void knnGatherMasked(const SpatialIndex& index, std::uint32_t neighbors,
                     std::uint32_t referenceCount,
                     const std::uint8_t* sourceMask,
                     const std::uint8_t* referenceMask,
                     std::uint32_t* pointIds, double* squaredDistances,
                     std::uint8_t* status);

// Applies PDAL OutlierFilter's ordered online mean to neighbors 1..k-1 while
// preserving knnGather's tie/incomplete status. The source point at distance
// zero is entry 0. Distance-only consumers may accept KnnDistanceTie because
// tied values are identical; they must reject KnnSearchIncomplete.
void knnMeanDistances(const SpatialIndex& index, std::uint32_t neighbors,
                      double* means, std::uint8_t* status);

// Reproduces filters.nndistance: Kth emits sqrt(distance[k]) and Average
// performs a serial sum of sqrt(distance[1..k]) followed by one division.
// `neighbors` includes the query point and is therefore the stage's k + 1.
// Distance ties are value-invariant; incomplete searches must not be
// published.
void knnDistanceValues(const SpatialIndex& index, std::uint32_t neighbors,
                       KnnDistanceMode mode, double* values,
                       std::uint8_t* status);

// Reprojects a planner-owned ordered max-k squared-distance rowset without a
// second spatial query. `rowNeighbors` is the fixed row stride and must cover
// the requested self-inclusive projection width. These operations preserve
// the exact serial arithmetic of knnMeanDistances/knnDistanceValues; callers
// retain and interpret the rowset query status independently.
void projectKnnMeanDistances(const SpatialIndex& index,
                             std::uint32_t rowNeighbors,
                             std::uint32_t meanNeighbors,
                             const double* squaredDistances, double* means);
void projectKnnDistanceValues(const SpatialIndex& index,
                              std::uint32_t rowNeighbors,
                              std::uint32_t distanceNeighbors,
                              KnnDistanceMode mode,
                              const double* squaredDistances, double* values);

// Computes PDAL's exact sample covariance for every k-nearest row. The
// centroid follows upstream's ordered online mean; each demeaned coordinate is
// narrowed to float before the six products are accumulated. Consumers that
// observe point identities must reject KnnDistanceTie, and every consumer must
// reject KnnSearchIncomplete. At least three neighbors and at least k input
// points are required.
void knnCovariances(const SpatialIndex& index, std::uint32_t neighbors,
                    Covariance3d* covariances, std::uint8_t* status);

// Extends knnCovariances with the same SelfAdjointEigenSolver contract used by
// PDAL's normal, eigenvalue, covariance-feature, and optimal-neighborhood
// stages. Approximate-zero covariance and solver failure are explicit status
// bits; each stage preserves its corresponding upstream skip/error behavior.
void knnEigenSystems(const SpatialIndex& index, std::uint32_t neighbors,
                     EigenSystem3d* systems, std::uint8_t* status);

// Reproduces filters.lof's three passes over one retained k-nearest
// adjacency: the k-distance (sqrt of the last retained squared distance), the
// local reachability density (inverse of the ordered online mean of
// reachability distances against each neighbor's k-distance), and the local
// outlier factor (ordered online mean of neighbor/query density ratios).
// `neighbors` includes the query point and is therefore the stage's
// incremented minpts. The passes observe neighbor identities one and two hops
// out: a row is exact only when its own status and every neighbor row's
// status are clean. `neighborStatus` receives the OR of the row's neighbor
// statuses so consumers can repair exactly the affected closure — an
// ambiguous own row invalidates the density and the factor, while an
// ambiguous neighbor row alone invalidates only the factor. The k-distance is
// distance-valued and therefore tie-invariant. An incomplete row differs
// from a tie in that its k-distance itself is unknown, so wrongness reaches
// one hop further: a neighbor's incomplete k-distance invalidates this
// row's density, and a neighbor's invalidated density invalidates this
// row's factor. `neighborNeighborStatus` receives the OR of
// (status | neighborStatus) over the row's neighbors so consumers can
// identify that second hop; for an incomplete own row it mirrors the own
// status.
// Reproduces filters.estimaterank's per-point rank: the float-demeaned
// neighborhood covariance decomposed by Eigen's fixed 3x3 JacobiSVD with
// the stage threshold cast to float. `neighbors` is upstream's
// self-inclusive knn. Incomplete rows report zero and their status bit;
// consumers repair them and tie rows from the compatibility index.
void knnRankValues(const SpatialIndex& index, std::uint32_t neighbors,
                   double threshold, std::uint8_t* ranks, std::uint8_t* status);

// Reproduces filters.optimalneighborhood's per-point selection: one
// distance-sorted max_k adjacency, the Welford incremental covariance
// sweep over k in [min_k, max_k], the host eigen solve, and the
// host-transcendental eigenentropy minimization. Outputs and statuses
// are host-visible on both memory kinds.
void knnOptimalValues(const SpatialIndex& index, std::uint32_t minimumK,
                      std::uint32_t maximumK, std::uint64_t* optimalK,
                      double* optimalRadius, std::uint8_t* status);

// Reproduces filters.neighborclassifier's per-point vote over the k
// self-inclusive neighbors' integer values: smallest value among maximal
// counts, applied only when the count strictly exceeds k/2 and the value
// changes. `values` and `results` live in the index's memory kind; the
// result is a pure function of the neighbor set, so device and host
// agree bit for bit wherever the gather does.
void knnNeighborVotes(const SpatialIndex& index, std::uint32_t neighbors,
                      const std::uint8_t* values, std::uint8_t* results,
                      std::uint8_t* status);

void knnLofValues(const SpatialIndex& index, std::uint32_t neighbors,
                  double* kDistances, double* reachabilityDensities,
                  double* outlierFactors, std::uint8_t* status,
                  std::uint8_t* neighborStatus,
                  std::uint8_t* neighborNeighborStatus);

} // namespace pdg
