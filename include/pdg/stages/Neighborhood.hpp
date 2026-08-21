#pragma once

#include <pdg/index/SpatialIndex.hpp>
#include <pdg/stages/Assign.hpp>

#include <cstdint>

namespace pdg
{

enum class OutlierMethod
{
    Statistical,
    Radius,
    Unknown
};

struct OutlierProgram
{
    OutlierMethod method = OutlierMethod::Statistical;
    std::int32_t minimumNeighbors = 2;
    double radius = 1.0;
    std::int32_t meanNeighbors = 8;
    double multiplier = 2.0;
    std::uint8_t classification = 7;
};

struct RadialDensityProgram
{
    double radius = 1.0;
};

// filters.radiusassign selects source rows through a strict radius query over
// an independently masked reference domain. The original ordered PDAL
// assignment statements remain the exact host finale; this compiled program
// declares their reads and writes to the resident planner.
struct RadiusAssignProgram
{
    double radius = 0.0;
    bool search3d = false;
    double maximumAbove = -1.0;
    double maximumBelow = -1.0;
    AssignProgram updates;
};

struct NnDistanceProgram
{
    std::uint32_t k = 10;
    KnnDistanceMode mode = KnnDistanceMode::Kth;
};

// The bounded exact filters.hag_nn lane supports one through 64 ground
// neighbors, which is the shared spatial index's own maximumNeighbors cap.
// The one-neighbor branch ignores max_distance and allow_extrapolation, as the
// pinned implementation does. Wider branches preserve the original ordered
// inverse-squared-distance interpolation and ground-bounds check.
struct HagNnProgram
{
    std::uint32_t count = 1U;
    double maximumDistance = 0.0;
    bool allowExtrapolation = true;
    std::uint8_t groundClass = 2U;
};

// B0235's measured count-one mapped-source executor keeps three status bytes,
// one uint32 neighbor row, and one binary64 neighbor value per input point.
// The complete whole-view reservation also includes the planner-owned
// columns/index and the direct source/output products observed by the explicit
// executor.  Runtime placement charges the full high-water independently of
// its component estimate so admission cannot double-count omissions into a
// too-small VRAM budget.
inline constexpr std::size_t HagNnCountOneExactDeviceScratchBytesPerPoint =
    3U * sizeof(std::uint8_t) + sizeof(std::uint32_t) + sizeof(double);
inline constexpr std::size_t HagNnCountOneExactDevicePeakBytesPerPoint = 160U;

// The first exact filters.hag_delaunay lane is the minimum valid count=3
// contract. Three selected ground neighbors produce one Delaunator triangle,
// whose seed ordering and barycentric operation sequence are reproduced by the
// device projection. Wider local triangulations remain upstream-owned.
struct HagDelaunayProgram
{
    std::uint32_t count = 3U;
    bool allowExtrapolation = true;
    std::uint8_t groundClass = 2U;
};

// B0236's measured count-three mapped-source executor keeps one status byte
// for each of the three ordered ground neighbors, plus their uint32 row ids
// and binary64 squared distances. The complete whole-view reservation also
// includes the planner-owned columns/index and direct source/output products
// observed by the executor.
inline constexpr std::size_t
    HagDelaunayCountThreeExactDeviceScratchBytesPerPoint =
        3U * sizeof(std::uint8_t) +
        3U * (sizeof(std::uint32_t) + sizeof(double));
inline constexpr std::size_t HagDelaunayCountThreeExactDevicePeakBytesPerPoint =
    184U;

// filters.lof declares its un-incremented `minpts` option; the executing
// stage replicates upstream's self-inclusive increment when it queries.
struct LofProgram
{
    std::int32_t minimumPoints = 10;
};

// filters.estimaterank's self-inclusive knn and its SVD rank threshold.
struct EstimateRankProgram
{
    std::int32_t neighbors = 8;
    double threshold = 0.01;
};

// filters.optimalneighborhood's self-inclusive k range.
struct OptimalNeighborhoodProgram
{
    std::uint32_t minimumK = 10;
    std::uint32_t maximumK = 14;
};

// filters.neighborclassifier's self-inclusive vote width over the
// default Classification dimension.
struct NeighborClassifierProgram
{
    std::int32_t neighbors = 0;
};

struct NormalProgram
{
    std::int32_t neighbors = 8;
    bool alwaysUp = true;
};

struct EigenvaluesProgram
{
    std::int32_t neighbors = 8;
    bool normalize = false;
};

struct ApproximateCoplanarProgram
{
    std::int32_t neighbors = 8;
    double threshold1 = 25.0;
    double threshold2 = 6.0;
};

enum class EigenvalueMode : std::uint8_t
{
    Raw,
    Sqrt,
    Normalized
};

enum CovarianceFeature : std::uint32_t
{
    CovarianceLinearity = 1U << 0U,
    CovariancePlanarity = 1U << 1U,
    CovarianceScattering = 1U << 2U,
    CovarianceVerticality = 1U << 3U,
    CovarianceOmnivariance = 1U << 4U,
    CovarianceAnisotropy = 1U << 5U,
    CovarianceEigenentropy = 1U << 6U,
    CovarianceEigenvalueSum = 1U << 7U,
    CovarianceSurfaceVariation = 1U << 8U,
    CovarianceDemantkeVerticality = 1U << 9U
};

inline constexpr std::uint32_t CovarianceDimensionality =
    CovarianceLinearity | CovariancePlanarity | CovarianceScattering |
    CovarianceVerticality;

struct CovarianceFeaturesProgram
{
    std::int32_t neighbors = 10;
    EigenvalueMode mode = EigenvalueMode::Sqrt;
    std::uint32_t features = CovarianceDimensionality;
};

class PointBatch;

// Project exact device-resident eigensystems into the output columns written
// by the corresponding PDAL stages. Required columns are materialized in the
// supplied device batch. Entries with a non-exact/failed status retain their
// prior column values, matching the stages' skip behavior. Covariance feature
// projection may add KnnFeatureInvalid to its caller-owned, stage-local status
// buffer. Omnivariance and Eigenentropy are intentionally rejected here because
// CUDA cbrt/log did not pass the binary64 compatibility gate.
void projectNormalColumns(PointBatch& batch, const EigenSystem3d* systems,
                          const std::uint8_t* status,
                          const NormalProgram& program);
void projectEigenvalueColumns(PointBatch& batch, const EigenSystem3d* systems,
                              const std::uint8_t* status,
                              const EigenvaluesProgram& program);
void projectApproximateCoplanarColumn(
    PointBatch& batch, const EigenSystem3d* systems, const std::uint8_t* status,
    const ApproximateCoplanarProgram& program);
void projectCovarianceFeatureColumns(PointBatch& batch,
                                     const EigenSystem3d* systems,
                                     std::uint8_t* status,
                                     const CovarianceFeaturesProgram& program);

// Projects the bounded filters.hag_nn result from one masked, planner-owned 2D
// nearest-ground query. Ground rows become +0.0. Count one uses one explicitly
// rounded binary64 subtraction; wider counts preserve the pinned
// interpolation order. Non-exact rows retain their prior output for whole-stage
// host repair.
void projectHagNnColumn(PointBatch& batch, const std::uint8_t* groundMask,
                        const std::uint32_t* nearestPointIds,
                        const double* squaredDistances,
                        const std::uint8_t* status, std::uint32_t neighbors,
                        std::uint32_t groundCount,
                        double maximumDistanceSquared, bool allowExtrapolation,
                        double minimumX, double minimumY, double maximumX,
                        double maximumY);

// Projects the count=3 filters.hag_delaunay result from one masked,
// planner-owned 2D nearest-ground query. Non-exact query rows retain their
// prior output so the wrapper can repair the complete stage on the host.
void projectHagDelaunayColumn(PointBatch& batch, const std::uint8_t* groundMask,
                              const std::uint32_t* nearestPointIds,
                              const std::uint8_t* status,
                              std::uint32_t groundCount,
                              bool allowExtrapolation, double minimumX,
                              double minimumY, double maximumX,
                              double maximumY);

} // namespace pdg
