#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <filters/private/DimRange.hpp>
#include <filters/private/expr/AssignStatement.hpp>
#include <pdal/KDIndex.hpp>

#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/index/SpatialTile.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/io/LasTranslate.hpp>
#if PDG_HAS_CUDA
#include <pdg/io/LasCuda.hpp>
#endif

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace pdal::pdg_detail
{

// Verbatim transcription of the pinned MathUtils computeCentroid,
// computeCovariance, and computeRank (the symbols are not exported from
// libpdalcpp): the online double centroid, the float-cast demeaned matrix
// held in doubles, the sample covariance, and Eigen's JacobiSVD with the
// float-cast threshold.
std::uint8_t computeRankExact(const PointView& view, const PointIdList& ids,
                              double threshold)
{
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    int n = 0;
    for (auto const& j : ids)
    {
        auto update = [&n](double value, double average)
        {
            double delta = value - average;
            double deltaN = delta / n;
            return average + deltaN;
        };
        n++;
        mx = update(view.getFieldAs<double>(Dimension::Id::X, j), mx);
        my = update(view.getFieldAs<double>(Dimension::Id::Y, j), my);
        mz = update(view.getFieldAs<double>(Dimension::Id::Z, j), mz);
    }

    Eigen::MatrixXd demeaned(3, static_cast<Eigen::Index>(ids.size()));
    Eigen::Index k = 0;
    for (auto const& j : ids)
    {
        demeaned(0, k) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::X, j) - mx);
        demeaned(1, k) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::Y, j) - my);
        demeaned(2, k) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::Z, j) - mz);
        k++;
    }
    const Eigen::Matrix3d covariance =
        demeaned * demeaned.transpose() / static_cast<double>(ids.size() - 1U);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance);
    svd.setThreshold(static_cast<float>(threshold));
    return static_cast<std::uint8_t>(svd.rank());
}

// Verbatim transcription of the pinned OptimalNeighborhood per-point
// selection for the tie/incomplete repair: the KD3 max_k query, the
// Welford covariance sweep, and the host-transcendental eigenentropy
// minimization in upstream's accumulation order.
void computeOptimalExact(const PointView& view, const KD3Index& index,
                         PointId point, std::uint32_t minimumK,
                         std::uint32_t maximumK, std::uint64_t& optimalK,
                         double& optimalRadius)
{
    PointIdList ids(maximumK);
    std::vector<double> squaredDistances(maximumK);
    index.knnSearch(point, maximumK, &ids, &squaredDistances);
    double minimumEntropy = (std::numeric_limits<double>::max)();
    std::uint64_t bestK = 0U;
    double bestSquared = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    Eigen::Matrix3d accumulated = Eigen::Matrix3d::Zero(3, 3);
    const auto update = [&](std::size_t item)
    {
        const PointId q = ids[item];
        const double dx = view.getFieldAs<double>(Dimension::Id::X, q) - mx;
        const double dy = view.getFieldAs<double>(Dimension::Id::Y, q) - my;
        const double dz = view.getFieldAs<double>(Dimension::Id::Z, q) - mz;
        const double n = double(item + 1);
        mx += dx / n;
        my += dy / n;
        mz += dz / n;
        const double scale = (n - 1) / n;
        accumulated(0, 0) = accumulated(0, 0) + scale * dx * dx;
        accumulated(1, 1) = accumulated(1, 1) + scale * dy * dy;
        accumulated(2, 2) = accumulated(2, 2) + scale * dz * dz;
        accumulated(1, 0) = accumulated(0, 1) =
            accumulated(0, 1) + scale * dx * dy;
        accumulated(2, 0) = accumulated(0, 2) =
            accumulated(0, 2) + scale * dx * dz;
        accumulated(1, 2) = accumulated(2, 1) =
            accumulated(2, 1) + scale * dy * dz;
        return n;
    };
    for (std::size_t k = 0; k + 1U < minimumK; ++k)
        update(k);
    for (std::size_t k = minimumK - 1U; k < maximumK; ++k)
    {
        const double n = update(k);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(accumulated /
                                                              (n - 1));
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Cannot perform eigen decomposition.");
        const Eigen::Vector3d values = solver.eigenvalues();
        double lambda2 = (std::max)(values[2], 0.0);
        double lambda1 = (std::max)(values[1], 0.0);
        double lambda0 = (std::max)(values[0], 0.0);
        const double sum = lambda2 + lambda1 + lambda0;
        lambda2 /= sum;
        lambda1 /= sum;
        lambda0 /= sum;
        const double entropy =
            -(lambda0 * std::log(lambda0) + lambda1 * std::log(lambda1) +
              lambda2 * std::log(lambda2));
        if (entropy < minimumEntropy)
        {
            minimumEntropy = entropy;
            bestK = static_cast<std::uint64_t>(k + 1U);
            bestSquared = squaredDistances[k];
        }
    }
    optimalK = bestK;
    optimalRadius = std::sqrt(bestSquared);
}

// Verbatim transcription of NeighborClassifierFilter's per-point vote for
// the tie/incomplete repair and the wrapper's host fallback: the ordered
// count map, the first maximal element, the strict k/2 threshold, and the
// changed-only application against the original values.
std::uint8_t computeNeighborVoteExact(const PointView& view,
                                      const KD3Index& index, PointId point,
                                      std::uint32_t neighbors)
{
    PointIdList ids(neighbors);
    std::vector<double> squaredDistances(neighbors);
    index.knnSearch(point, neighbors, &ids, &squaredDistances);
    const double threshold = double(ids.size()) / 2.0;
    std::map<int, unsigned int> counts;
    for (const PointId id : ids)
        counts[view.getFieldAs<int>(Dimension::Id::Classification, id)]++;
    const auto winner =
        *std::max_element(counts.begin(), counts.end(),
                          [](const std::pair<const int, unsigned int>& left,
                             const std::pair<const int, unsigned int>& right)
                          { return left.second < right.second; });
    const int oldValue =
        view.getFieldAs<int>(Dimension::Id::Classification, point);
    if (winner.second > threshold && oldValue != winner.first)
        return static_cast<std::uint8_t>(winner.first);
    return static_cast<std::uint8_t>(oldValue);
}

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* message)
{
    if (right != 0U && left > (std::numeric_limits<std::size_t>::max)() / right)
        throw std::overflow_error(message);
    return left * right;
}

std::optional<std::size_t> configuredNativeWorkers()
{
    const char* configured = std::getenv("PDG_NATIVE_WORKERS");
    if (!configured || !*configured)
        return std::nullopt;
    const std::string_view text(configured);
    std::size_t value = 0U;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(
            "PDG_NATIVE_WORKERS must be a positive integer");
    return value;
}

constexpr std::size_t LofParallelRepairMinimumRows = 4096U;

std::size_t lofRepairWorkerCount(std::size_t repairRows)
{
    if (!repairRows)
        return 1U;
    // A worker owns a fixed contiguous range of original point ordinals.
    // Below this threshold the thread lifecycle costs more than the bounded
    // exact kNN work. PDG_NATIVE_WORKERS is a cap, matching the other native
    // host executors; it never widens the useful-worker calculation.
    const std::size_t usefulWorkers =
        repairRows / LofParallelRepairMinimumRows +
        static_cast<std::size_t>(repairRows % LofParallelRepairMinimumRows !=
                                 0U);
    const std::optional<std::size_t> configured = configuredNativeWorkers();
    const std::size_t availableWorkers = configured.value_or(
        (std::max<std::size_t>)(1U, std::thread::hardware_concurrency()));
    return (std::max<std::size_t>)(1U,
                                   (std::min)(usefulWorkers, availableWorkers));
}

template <typename Function>
void forEachLofRepairChunk(std::size_t pointCount, std::size_t workerCount,
                           Function&& function)
{
    if (workerCount <= 1U)
    {
        function(0U, pointCount);
        return;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(workerCount);
    workers.reserve(workerCount);
    const std::size_t pointsPerWorker = pointCount / workerCount;
    const std::size_t remainder = pointCount % workerCount;
    try
    {
        for (std::size_t worker = 0U; worker < workerCount; ++worker)
        {
            const std::size_t begin =
                worker * pointsPerWorker + (std::min)(worker, remainder);
            const std::size_t end =
                begin + pointsPerWorker +
                static_cast<std::size_t>(worker < remainder);
            workers.emplace_back(
                [&, worker, begin, end]
                {
                    try
                    {
                        function(begin, end);
                    }
                    catch (...)
                    {
                        errors[worker] = std::current_exception();
                    }
                });
        }
    }
    catch (...)
    {
        for (std::thread& worker : workers)
            worker.join();
        throw;
    }
    for (std::thread& worker : workers)
        worker.join();
    for (const std::exception_ptr& error : errors)
        if (error)
            std::rethrow_exception(error);
}

thread_local std::span<const std::byte> ActiveLasSource;
thread_local bool* ActiveLasSourceUsed = nullptr;
thread_local bool* ActiveLasRecordSummaryUsed = nullptr;
thread_local bool* ActiveLasHostXyzMirrored = nullptr;
thread_local bool ActiveDeviceOnlyNnDistanceHandoff = false;
thread_local bool* ActiveNnDistanceDeviceOnlyHandoffUsed = nullptr;
thread_local bool* ActiveNnDistanceHostRestoreUsed = nullptr;
thread_local bool* ActiveNnDistanceAssignmentDeviceColumnReused = nullptr;
thread_local bool* ActiveKnnGatherReuseUsed = nullptr;
thread_local bool* ActiveResidentAssignmentExecuted = nullptr;
thread_local bool* ActiveHagDelaunayCudaUsed = nullptr;

class NeighborhoodProofFailure final : public std::runtime_error
{
public:
    explicit NeighborhoodProofFailure(const char* message)
        : std::runtime_error(message)
    {
    }
};

// Aggregates one boundary loop's wall time into the active resident
// context's phase counters; a null accumulator (no active context, or a
// direct one-stage invocation) records nothing.
void accumulatePhaseSeconds(double* accumulator,
                            std::chrono::steady_clock::time_point started)
{
    if (accumulator)
        *accumulator += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count();
}

// One stats-only host-wall span inside the direct resident manager interval.
// The active accumulator is null on ordinary execution, so those paths do not
// read the clock. CUDA work remains asynchronously ordered on its existing
// stream; these spans describe call-wall ownership, not exclusive kernel time.
class ResidentManagerDetailTimer
{
public:
    using Field = double ResidentManagerDetailSeconds::*;

    explicit ResidentManagerDetailTimer(Field field) noexcept
    {
        if (ResidentManagerDetailSeconds* phases =
                activeResidentManagerDetailSeconds())
        {
            m_accumulator = &(phases->*field);
            m_started = std::chrono::steady_clock::now();
        }
    }

    ~ResidentManagerDetailTimer()
    {
        finish();
    }

    void finish() noexcept
    {
        accumulatePhaseSeconds(m_accumulator, m_started);
        m_accumulator = nullptr;
    }

    ResidentManagerDetailTimer(const ResidentManagerDetailTimer&) = delete;
    ResidentManagerDetailTimer&
    operator=(const ResidentManagerDetailTimer&) = delete;

private:
    double* m_accumulator = nullptr;
    std::chrono::steady_clock::time_point m_started{};
};

// Nested eigen-family host-call spans (D0215 diagnostic). They overlap the
// one broad neighborhoodQueryProjection span but not one another.
class ResidentEigenFamilyDetailTimer
{
public:
    using Field = double ResidentEigenFamilyDetailSeconds::*;

    explicit ResidentEigenFamilyDetailTimer(Field field) noexcept
    {
        if (ResidentManagerDetailSeconds* phases =
                activeResidentManagerDetailSeconds())
        {
            m_accumulator = &(phases->eigenFamily.*field);
            m_started = std::chrono::steady_clock::now();
        }
    }

    ~ResidentEigenFamilyDetailTimer()
    {
        finish();
    }

    void finish() noexcept
    {
        accumulatePhaseSeconds(m_accumulator, m_started);
        m_accumulator = nullptr;
    }

    ResidentEigenFamilyDetailTimer(const ResidentEigenFamilyDetailTimer&) =
        delete;
    ResidentEigenFamilyDetailTimer&
    operator=(const ResidentEigenFamilyDetailTimer&) = delete;

private:
    double* m_accumulator = nullptr;
    std::chrono::steady_clock::time_point m_started{};
};

// Nested NND-only host-call spans. They overlap the one broad
// neighborhoodQueryProjection span above but not one another.
class ResidentNnDistanceDetailTimer
{
public:
    using Field = double ResidentNnDistanceDetailSeconds::*;

    explicit ResidentNnDistanceDetailTimer(Field field) noexcept
    {
        if (ResidentManagerDetailSeconds* phases =
                activeResidentManagerDetailSeconds())
        {
            m_accumulator = &(phases->nnDistance.*field);
            m_started = std::chrono::steady_clock::now();
        }
    }

    ~ResidentNnDistanceDetailTimer()
    {
        accumulatePhaseSeconds(m_accumulator, m_started);
    }

    ResidentNnDistanceDetailTimer(const ResidentNnDistanceDetailTimer&) =
        delete;
    ResidentNnDistanceDetailTimer&
    operator=(const ResidentNnDistanceDetailTimer&) = delete;

private:
    double* m_accumulator = nullptr;
    std::chrono::steady_clock::time_point m_started{};
};

// Observed physical transfers of the whole-view attach machinery. The
// planner-selected region identifier follows the IndexBuild convention
// (region.id is one-based; an unplanned one-stage invocation reports
// region zero), and record() is a no-op without an execution scope.
void observeTransfer(pdg::ExecutionEventKind kind, std::uint64_t regionId,
                     std::size_t bytes)
{
    pdg::ExecutionObservationScope::record(
        kind, regionId ? static_cast<std::size_t>(regionId) - 1U : 0U, bytes);
}

// D0062 proved that PointView rows match the layout's dimension offsets
// byte-for-byte only when the declared offsets exactly partition the
// record — sorted, contiguous, and summing to the point stride. A layout
// whose offsets are not byte offsets (a finalized column table rewrites
// them into per-dimension ordinals) must keep the semantic field path.
bool rowOffsetsArePhysical(const PointLayout& layout)
{
    struct Span
    {
        std::size_t offset = 0;
        std::size_t size = 0;
    };
    const DimTypeList dims = layout.dimTypes();
    if (dims.empty() || !layout.pointSize())
        return false;
    std::vector<Span> spans;
    spans.reserve(dims.size());
    for (const DimType& dim : dims)
        spans.push_back(
            {layout.dimOffset(dim.m_id), Dimension::size(dim.m_type)});
    std::sort(spans.begin(), spans.end(),
              [](const Span& left, const Span& right)
              { return left.offset < right.offset; });
    std::size_t expected = 0;
    for (const Span& span : spans)
    {
        if (span.offset != expected)
            return false;
        expected += span.size;
    }
    return expected == layout.pointSize();
}

bool rowBoundaryUsable(PointView& view)
{
    return !std::getenv("PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY") &&
           rowOffsetsArePhysical(*view.layout());
}

void requireRowBoundary(bool direct)
{
    if (!direct && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY"))
        throw NeighborhoodProofFailure(
            "required row-backed neighborhood boundary did not engage");
}

// Copies one PointView column into a densely packed buffer. A row-backed
// table whose stored type equals the transferred type copies the identical
// bytes directly out of each physical row; every other table keeps the
// semantic per-field path, which performs the same conversion one point at
// a time.
void gatherColumnFromView(PointView& view, Dimension::Id id,
                          Dimension::Type type, bool physicalRows,
                          std::byte* destination, std::size_t stride)
{
    const bool direct = physicalRows && view.layout()->dimType(id) == type;
    requireRowBoundary(direct);
    const std::size_t offset = direct ? view.layout()->dimOffset(id) : 0U;
    for (PointId point = 0; point < view.size(); ++point)
    {
        std::byte* target =
            destination + static_cast<std::size_t>(point) * stride;
        if (direct)
            if (const char* row = view.getPoint(point))
            {
                std::memcpy(target, row + offset, stride);
                continue;
            }
        view.getField(reinterpret_cast<char*>(target), id, type, point);
    }
}

void publishColumnToView(PointView& view, Dimension::Id id,
                         Dimension::Type type, bool physicalRows,
                         const std::byte* source, std::size_t stride)
{
    const bool direct = physicalRows && view.layout()->dimType(id) == type;
    requireRowBoundary(direct);
    const std::size_t offset = direct ? view.layout()->dimOffset(id) : 0U;
    for (PointId point = 0; point < view.size(); ++point)
    {
        const std::byte* value =
            source + static_cast<std::size_t>(point) * stride;
        if (direct)
            if (char* row = view.getPoint(point))
            {
                std::memcpy(row + offset, value, stride);
                continue;
            }
        view.setField(id, type, point, value);
    }
}

pdg::PointBatch gather(PointView& view, pdg::DimensionRegistry& dimensions,
                       pdg::MemoryResource& memory)
{
    const std::size_t size = static_cast<std::size_t>(view.size());
    pdg::PointBatch batch(
        size, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    for (pdg::DimensionId dimension : {X, Y, Z})
        batch.materialize(dimension, pdg::DimensionType::Double);
    batch.setSize(size);
    ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
    const auto started = std::chrono::steady_clock::now();
    const bool physicalRows = rowBoundaryUsable(view);
    const std::pair<pdg::DimensionId, Dimension::Id> coordinates[]{
        {X, Dimension::Id::X}, {Y, Dimension::Id::Y}, {Z, Dimension::Id::Z}};
    for (const auto& [device, pointView] : coordinates)
        gatherColumnFromView(
            view, pointView, Dimension::Type::Double, physicalRows,
            reinterpret_cast<std::byte*>(batch.data<double>(device)),
            sizeof(double));
    accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr, started);
    return batch;
}

std::optional<double> requestedTileEdge()
{
    const char* text = std::getenv("PDG_SPATIAL_TILE_EDGE");
    if (!text)
        return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno || end == text || *end != '\0' || !std::isfinite(value) ||
        value <= 0.0)
        throw std::invalid_argument(
            "PDG_SPATIAL_TILE_EDGE must be finite and positive");
    return value;
}

std::size_t requestedSchedulerLanes()
{
    const char* text = std::getenv("PDG_CUDA_SCHEDULER_LANES");
    if (!text || !*text)
        return 0U;
    try
    {
        return pdg::parseSchedulerLaneCount(text);
    }
    catch (const std::invalid_argument&)
    {
        throw std::invalid_argument(
            "PDG_CUDA_SCHEDULER_LANES must be an integer in [2, 6]");
    }
}

std::array<double, 3> minimumCoordinates(const pdg::PointBatch& batch)
{
    std::array<double, 3> minimum{(std::numeric_limits<double>::infinity)(),
                                  (std::numeric_limits<double>::infinity)(),
                                  (std::numeric_limits<double>::infinity)()};
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        minimum[0] = (std::min)(minimum[0], batch.data<double>(X)[point]);
        minimum[1] = (std::min)(minimum[1], batch.data<double>(Y)[point]);
        minimum[2] = (std::min)(minimum[2], batch.data<double>(Z)[point]);
    }
    return minimum;
}

double automaticTileEdge(const pdg::PointBatch& batch, double radius,
                         std::size_t targetCorePoints,
                         const std::array<double, 3>& minimum)
{
    std::array<double, 2> maximum{minimum[0], minimum[1]};
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        maximum[0] = (std::max)(maximum[0], batch.data<double>(X)[point]);
        maximum[1] = (std::max)(maximum[1], batch.data<double>(Y)[point]);
    }
    const double width = maximum[0] - minimum[0];
    const double height = maximum[1] - minimum[1];
    const std::size_t desiredTiles =
        (batch.size() - 1U) / (std::max)(targetCorePoints, std::size_t{1}) + 1U;
    double edge = 0.0;
    if (width > 0.0 && height > 0.0)
        edge = std::sqrt(width * height / static_cast<double>(desiredTiles));
    else
        edge = (std::max)(width, height) / static_cast<double>(desiredTiles);
    if (!std::isfinite(edge) || edge <= 0.0)
        edge = radius;
    return (std::max)(edge, radius);
}
} // unnamed namespace

CudaNeighborhoodLasSourceScope::CudaNeighborhoodLasSourceScope(
    std::span<const std::byte> mappedLasBytes, bool deviceOnlyNnDistanceHandoff)
    : m_previous(ActiveLasSource), m_previousUsed(ActiveLasSourceUsed),
      m_previousRecordSummaryUsed(ActiveLasRecordSummaryUsed),
      m_previousHostXyzMirrored(ActiveLasHostXyzMirrored),
      m_previousDeviceOnlyNnDistanceHandoff(ActiveDeviceOnlyNnDistanceHandoff),
      m_previousNnDistanceDeviceOnlyHandoffUsed(
          ActiveNnDistanceDeviceOnlyHandoffUsed),
      m_previousNnDistanceHostRestoreUsed(ActiveNnDistanceHostRestoreUsed),
      m_previousNnDistanceAssignmentDeviceColumnReused(
          ActiveNnDistanceAssignmentDeviceColumnReused),
      m_previousKnnGatherReuseUsed(ActiveKnnGatherReuseUsed),
      m_previousResidentAssignmentExecuted(ActiveResidentAssignmentExecuted),
      m_previousHagDelaunayCudaUsed(ActiveHagDelaunayCudaUsed)
{
    if (mappedLasBytes.empty())
        throw std::invalid_argument(
            "direct resident LAS source requires mapped input bytes");
    ActiveLasSource = mappedLasBytes;
    ActiveLasSourceUsed = &m_used;
    ActiveLasRecordSummaryUsed = &m_recordSummaryUsed;
    ActiveLasHostXyzMirrored = &m_hostXyzMirrored;
    ActiveDeviceOnlyNnDistanceHandoff = deviceOnlyNnDistanceHandoff;
    ActiveNnDistanceDeviceOnlyHandoffUsed = &m_nnDistanceDeviceOnlyHandoffUsed;
    ActiveNnDistanceHostRestoreUsed = &m_nnDistanceHostRestoreUsed;
    ActiveNnDistanceAssignmentDeviceColumnReused =
        &m_nnDistanceAssignmentDeviceColumnReused;
    ActiveKnnGatherReuseUsed = &m_knnGatherReuseUsed;
    ActiveResidentAssignmentExecuted = &m_residentAssignmentExecuted;
    ActiveHagDelaunayCudaUsed = &m_hagDelaunayCudaUsed;
}

CudaNeighborhoodLasSourceScope::~CudaNeighborhoodLasSourceScope()
{
    ActiveLasSource = m_previous;
    ActiveLasSourceUsed = m_previousUsed;
    ActiveLasRecordSummaryUsed = m_previousRecordSummaryUsed;
    ActiveLasHostXyzMirrored = m_previousHostXyzMirrored;
    ActiveDeviceOnlyNnDistanceHandoff = m_previousDeviceOnlyNnDistanceHandoff;
    ActiveNnDistanceDeviceOnlyHandoffUsed =
        m_previousNnDistanceDeviceOnlyHandoffUsed;
    ActiveNnDistanceHostRestoreUsed = m_previousNnDistanceHostRestoreUsed;
    ActiveNnDistanceAssignmentDeviceColumnReused =
        m_previousNnDistanceAssignmentDeviceColumnReused;
    ActiveKnnGatherReuseUsed = m_previousKnnGatherReuseUsed;
    ActiveResidentAssignmentExecuted = m_previousResidentAssignmentExecuted;
    ActiveHagDelaunayCudaUsed = m_previousHagDelaunayCudaUsed;
}

bool CudaNeighborhoodLasSourceScope::used() const noexcept
{
    return m_used;
}

bool CudaNeighborhoodLasSourceScope::recordSummaryUsed() const noexcept
{
    return m_recordSummaryUsed;
}

bool CudaNeighborhoodLasSourceScope::hostXyzMirrored() const noexcept
{
    return m_hostXyzMirrored;
}

bool CudaNeighborhoodLasSourceScope::nnDistanceDeviceOnlyHandoffUsed()
    const noexcept
{
    return m_nnDistanceDeviceOnlyHandoffUsed;
}

bool CudaNeighborhoodLasSourceScope::nnDistanceHostRestoreUsed() const noexcept
{
    return m_nnDistanceHostRestoreUsed;
}

bool CudaNeighborhoodLasSourceScope::nnDistanceAssignmentDeviceColumnReused()
    const noexcept
{
    return m_nnDistanceAssignmentDeviceColumnReused;
}

bool CudaNeighborhoodLasSourceScope::knnGatherReuseUsed() const noexcept
{
    return m_knnGatherReuseUsed;
}

bool CudaNeighborhoodLasSourceScope::residentAssignmentExecuted() const noexcept
{
    return m_residentAssignmentExecuted;
}

bool CudaNeighborhoodLasSourceScope::hagDelaunayCudaUsed() const noexcept
{
    return m_hagDelaunayCudaUsed;
}

std::shared_ptr<void> ResidentPointViewAccess::get(const PointView& view)
{
    return view.m_pdgResidentProduct;
}

void ResidentPointViewAccess::set(PointView& view,
                                  std::shared_ptr<void> product)
{
    view.m_pdgResidentProduct = std::move(product);
}

void ResidentPointViewAccess::clear(PointView& view)
{
    view.m_pdgResidentProduct.reset();
}

PointId ResidentPointViewAccess::tableId(const PointView& view, PointId point)
{
    if (point >= view.size())
        throw pdal_error("Point view source identity index out of range.");
    return view.m_index.at(static_cast<std::size_t>(point));
}

std::span<const std::byte> activeCudaLasSource() noexcept
{
    return ActiveLasSource;
}

void markActiveCudaLasSourceUsed() noexcept
{
    if (ActiveLasSourceUsed)
        *ActiveLasSourceUsed = true;
}

void clearCudaNeighborhood(PointView& view)
{
    ResidentPointViewAccess::clear(view);
}

pdg::UniformGridConfig selectCudaKnnConfig(const pdg::PointBatch& hostBatch,
                                           std::uint8_t dimensions,
                                           std::uint32_t neighbors)
{
    const bool forceBvh = std::getenv("PDG_FORCE_MORTON_BVH") != nullptr;
    const bool forceGrid = std::getenv("PDG_FORCE_UNIFORM_GRID") != nullptr;
    if (forceBvh && forceGrid)
        throw std::invalid_argument(
            "conflicting forced CUDA spatial-index backends");
    if (forceBvh)
        return pdg::makeMortonBvhConfig(hostBatch, dimensions);
    if (forceGrid)
        return pdg::makeKnnGridConfig(hostBatch, dimensions, neighbors);
    return pdg::makeAdaptiveKnnConfig(hostBatch, dimensions, neighbors);
}

pdg::UniformGridConfig selectCudaKnnConfig(const pdg::KnnConfigSummary& summary,
                                           std::uint32_t neighbors)
{
    const bool forceBvh = std::getenv("PDG_FORCE_MORTON_BVH") != nullptr;
    const bool forceGrid = std::getenv("PDG_FORCE_UNIFORM_GRID") != nullptr;
    if (forceBvh && forceGrid)
        throw std::invalid_argument(
            "conflicting forced CUDA spatial-index backends");
    if (forceBvh)
        return pdg::makeMortonBvhConfig(summary);
    if (forceGrid)
        return pdg::makeKnnGridConfig(summary, neighbors);
    return pdg::makeAdaptiveKnnConfig(summary, neighbors);
}

EigenResult computeEigenSystem(const PointView& view,
                               const PointIdList& neighbors)
{
    EigenResult result;
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    point_count_t count = 0;
    for (PointId point : neighbors)
    {
        ++count;
        centroid[0] +=
            (view.getFieldAs<double>(Dimension::Id::X, point) - centroid[0]) /
            static_cast<double>(count);
        centroid[1] +=
            (view.getFieldAs<double>(Dimension::Id::Y, point) - centroid[1]) /
            static_cast<double>(count);
        centroid[2] +=
            (view.getFieldAs<double>(Dimension::Id::Z, point) - centroid[2]) /
            static_cast<double>(count);
    }

    Eigen::MatrixXd demeaned(3, static_cast<Eigen::Index>(neighbors.size()));
    std::size_t column = 0;
    for (PointId point : neighbors)
    {
        demeaned(0, static_cast<Eigen::Index>(column)) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::X, point) - centroid[0]);
        demeaned(1, static_cast<Eigen::Index>(column)) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::Y, point) - centroid[1]);
        demeaned(2, static_cast<Eigen::Index>(column)) = static_cast<float>(
            view.getFieldAs<double>(Dimension::Id::Z, point) - centroid[2]);
        ++column;
    }
    const Eigen::Matrix3d covariance =
        demeaned * demeaned.transpose() /
        static_cast<double>(neighbors.size() - 1U);
    if (covariance.isZero())
    {
        result.status = EigenStatus::CovarianceZero;
        return result;
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success)
    {
        result.status = EigenStatus::SolverFailure;
        return result;
    }
    for (std::size_t eigen = 0; eigen < 3; ++eigen)
    {
        result.system.values[eigen] =
            solver.eigenvalues()[static_cast<Eigen::Index>(eigen)];
        for (std::size_t axis = 0; axis < 3; ++axis)
            result.system.vectors[axis * 3U + eigen] =
                solver.eigenvectors()(static_cast<Eigen::Index>(axis),
                                      static_cast<Eigen::Index>(eigen));
    }
    return result;
}

#if PDG_HAS_CUDA
namespace
{
template <typename Output, typename WholeQuery, typename TiledQuery>
// NeighborhoodProofFailure must not be swallowed into a fallback: every
// tryCuda* entry rethrows it so a required proof that did not occur fails
// the invocation instead of silently changing paths.
bool tryCudaRadiusOutput(PointView& view, double radius,
                         std::vector<Output>& output, bool requireCuda,
                         WholeQuery&& wholeQuery, TiledQuery&& tiledQuery)
{
    if (!std::isfinite(radius) || radius <= 0.0)
        return false;
    output.resize(static_cast<std::size_t>(view.size()));
    const bool requireTiling =
        std::getenv("PDG_REQUIRE_SPATIAL_TILING") != nullptr;
    try
    {
        if (view.empty())
        {
            if (requireTiling)
                throw std::runtime_error(
                    "required spatial tiling cannot partition an empty view");
            return true;
        }
        if (pdg::cudaDevices().empty())
            return false;

        pdg::DimensionRegistry dimensions;
        std::unique_ptr<pdg::MemoryResource> pinnedMemory =
            pdg::makeCudaPinnedMemoryResource();
        pdg::PointBatch host = gather(view, dimensions, *pinnedMemory);
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        PDG_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        constexpr std::size_t MaximumWorkingSet =
            20ULL * 1024ULL * 1024ULL * 1024ULL;
        constexpr std::size_t ConservativeBytesPerPoint = 128U;
        const std::size_t freeBudget = freeBytes - freeBytes / 4U;
        const std::size_t totalBudget = totalBytes - totalBytes / 4U;
        const std::size_t workingBytes =
            (std::min)(MaximumWorkingSet, (std::min)(freeBudget, totalBudget));
        const std::size_t maximumWholePoints =
            (std::min)(static_cast<std::size_t>(
                           (std::numeric_limits<int>::max)()),
                       (std::max)(std::size_t{1},
                                  workingBytes / ConservativeBytesPerPoint));
        const std::size_t schedulerOverride = requestedSchedulerLanes();
        const std::size_t schedulerLanes =
            schedulerOverride
                ? schedulerOverride
                : pdg::fixedLaneCount(pdg::PipelineClass::RadiusNeighborhood);
        // Divide the conservative budget across the benchmark-fixed lanes.
        // Runtime scheduling receives the full budget and may clamp the
        // active width further for a ghost-heavy tile.
        const std::size_t maximumTilePoints =
            (std::min)(static_cast<std::size_t>(
                           (std::numeric_limits<int>::max)()),
                       (std::max)(std::size_t{1},
                                  workingBytes / (schedulerLanes *
                                                  ConservativeBytesPerPoint)));
        const std::optional<double> forcedEdge = requestedTileEdge();
        const bool needsTiling = host.size() > maximumWholePoints;
        if (forcedEdge || requireTiling || needsTiling)
        {
            const std::array<double, 3> origin = minimumCoordinates(host);
            const std::size_t targetCorePoints =
                requireTiling && !needsTiling
                    ? (std::max)(std::size_t{1}, host.size() / 4U)
                    : (std::max)(std::size_t{1}, maximumTilePoints / 2U);
            double edge = forcedEdge.value_or(
                automaticTileEdge(host, radius, targetCorePoints, origin));
            pdg::SpatialTileSet tiles;
            for (;;)
            {
                try
                {
                    tiles = pdg::makeSpatialTiles(
                        host, {2, edge, radius, origin, maximumTilePoints,
                               schedulerOverride, workingBytes});
                    break;
                }
                catch (const std::length_error&)
                {
                    if (forcedEdge || edge <= radius)
                        throw;
                    edge = (std::max)(radius, edge / 2.0);
                }
            }
            if (requireTiling && tiles.tiles().size() < 2U)
                throw std::runtime_error(
                    "required spatial tiling produced fewer than two core "
                    "tiles");
            std::unique_ptr<pdg::MemoryResource> deviceMemory =
                pdg::makeCudaMemoryResource(workingBytes / schedulerLanes);
            const pdg::SpatialTileExecutionStats stats = tiledQuery(
                host, tiles, dimensions, *pinnedMemory, *deviceMemory, output);
            if (requireTiling && stats.tileCount < 2U)
                throw std::runtime_error(
                    "required spatial tiled radius execution did not occur");
            if (schedulerOverride &&
                stats.executionLaneCount != schedulerOverride)
                throw std::runtime_error(
                    "requested CUDA scheduler lane count was not active");
            return true;
        }

        const pdg::UniformGridConfig config =
            pdg::makeUniformGridConfig(host, 3, radius);
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource(workingBytes);
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               dimensions, *deviceMemory);
        for (pdg::DimensionId dimension : {X, Y, Z})
            device.materialize(dimension, pdg::DimensionType::Double);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (pdg::DimensionId dimension : {X, Y, Z})
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                device.rawData(dimension), host.rawData(dimension),
                host.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        if (!pdg::uniformGridMaySupportExactDevice(device, config))
            return false;

        pdg::SpatialIndex index(device, config);
        index.build();
        std::unique_ptr<pdg::Allocation> deviceOutput = deviceMemory->allocate(
            host.size() * sizeof(Output), alignof(Output));
        wholeQuery(index, static_cast<Output*>(deviceOutput->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(output.data(), deviceOutput->data(),
                                       host.size() * sizeof(Output),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        throw;
    }
    catch (const pdg::CudaError&)
    {
        if (requireCuda)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        if (requireCuda || requireTiling)
            throw;
        return false;
    }
}

struct CachedCudaEigenSystems
{
    std::shared_ptr<CudaNeighborhoodResults> host;
    std::unique_ptr<pdg::Allocation> hostEigenvalues;
    std::unique_ptr<pdg::Allocation> deviceSystems;
    std::unique_ptr<pdg::Allocation> deviceStatus;
    std::unique_ptr<pdg::Allocation> deviceProjectionStatus;
    bool statusKnown = false;
    bool exact = false;
};

struct CachedCudaKnnGather
{
    std::uint32_t neighbors = 0U;
    std::unique_ptr<pdg::Allocation> pointIds;
    std::unique_ptr<pdg::Allocation> squaredDistances;
    std::unique_ptr<pdg::Allocation> status;
};

struct CachedHagNnCompatibilityGround
{
    PointViewPtr view;
    BOX2D bounds;
    const KD2Index* index = nullptr;
};

struct ResidentNeighborhood
{
    std::uint64_t region = 0;
    std::uint32_t maximumNeighbors = 0;
    double maximumRadius = 0.0;
    std::uint8_t spatialDimensions = 3;
    bool radiusIndex = false;
    bool indexRequired = true;
    std::size_t pointCount = 0;
    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> pinnedMemory;
    std::unique_ptr<pdg::PointBatch> host;
    std::unique_ptr<pdg::MemoryResource> deviceMemory;
    std::unique_ptr<pdg::PointBatch> device;
    std::unique_ptr<pdg::SpatialIndex> index;
    std::map<std::uint32_t, std::unique_ptr<CachedCudaEigenSystems>>
        eigenSystems;
    std::unique_ptr<CachedCudaKnnGather> knnGather;
    // Planner-owned, lazily constructed exact-repair product. Keeping both
    // the filtered view and its pinned KD2 tree on the resident product makes
    // the ownership/lifetime explicit and prevents a stage-private index.
    std::unique_ptr<CachedHagNnCompatibilityGround> hagNnCompatibilityGround;
};

std::int32_t directLasRawCoordinate(const std::byte* record,
                                    std::size_t axis) noexcept
{
    std::int32_t value = 0;
    std::memcpy(&value, record + axis * sizeof(value), sizeof(value));
    return value;
}

pdg::KnnConfigSummary
summarizeDirectLasCoordinates(const pdg::las::Header& header,
                              const std::byte* records, std::uint8_t dimensions)
{
    pdg::KnnConfigSummary summary;
    summary.pointCount = static_cast<std::size_t>(header.pointCount);
    summary.dimensions = dimensions;
    if (summary.pointCount == 0U)
        return summary;

    const pdg::CoordinateEncoding encoding = header.coordinateEncoding();
    const auto coordinate = [&](std::size_t point, std::size_t axis)
    {
        const std::byte* record = records + point * header.pointRecordLength;
        return encoding.decode(axis, directLasRawCoordinate(record, axis));
    };
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
        summary.minimum[axis] = summary.maximum[axis] = coordinate(0U, axis);
    for (std::size_t point = 1U; point < summary.pointCount; ++point)
        for (std::uint8_t axis = 0; axis < dimensions; ++axis)
        {
            const double value = coordinate(point, axis);
            summary.minimum[axis] = (std::min)(summary.minimum[axis], value);
            summary.maximum[axis] = (std::max)(summary.maximum[axis], value);
        }

    const std::size_t probeCount =
        (std::min)(summary.pointCount, pdg::KnnConfigMaximumProbePoints);
    summary.probes.resize(probeCount);
    for (std::size_t probe = 0U; probe < probeCount; ++probe)
    {
        const std::size_t point =
            pdg::knnConfigProbePoint(summary.pointCount, probe);
        for (std::uint8_t axis = 0; axis < dimensions; ++axis)
            summary.probes[probe][axis] = coordinate(point, axis);
    }
    return summary;
}

pdg::KnnConfigSummary
hydrateResidentCoordinatesFromLas(ResidentNeighborhood& resident,
                                  const PointView& view)
{
    ResidentManagerDetailTimer allocationTimer(
        &ResidentManagerDetailSeconds::directLasHydrationAllocation);
    if (ActiveLasSource.empty())
        throw NeighborhoodProofFailure(
            "direct resident LAS source mapping is unavailable");
    const pdg::las::FileView input(ActiveLasSource);
    if ((!pdg::las::supportsDefaultTranslation(input) &&
         !pdg::las::supportsExtraDoubleTranslation(input)) ||
        input.header().pointCount != view.size())
        throw NeighborhoodProofFailure(
            "direct resident LAS source differs from the accepted view");

    const pdg::las::Header& header = input.header();
    if (header.pointCount >
        std::numeric_limits<std::size_t>::max() / header.pointRecordLength)
        throw std::overflow_error(
            "direct resident LAS record extent overflows size_t");
    const std::size_t recordBytes =
        static_cast<std::size_t>(header.pointCount) * header.pointRecordLength;
    const std::span<const std::byte> mapped = input.bytes();
    if (header.pointDataOffset > mapped.size() ||
        recordBytes > mapped.size() - header.pointDataOffset)
        throw pdg::las::Error(
            "direct resident LAS point records exceed the mapped input");

    for (pdg::DimensionId dimension : {X, Y, Z})
        resident.device->materialize(dimension, pdg::DimensionType::Double);
    const std::size_t rawBytes = resident.pointCount * sizeof(std::int32_t);
    std::unique_ptr<pdg::Allocation> deviceRecords =
        resident.deviceMemory->allocate(recordBytes, alignof(std::max_align_t));
    std::unique_ptr<pdg::Allocation> rawX =
        resident.deviceMemory->allocate(rawBytes, alignof(std::int32_t));
    std::unique_ptr<pdg::Allocation> rawY =
        resident.deviceMemory->allocate(rawBytes, alignof(std::int32_t));
    std::unique_ptr<pdg::Allocation> rawZ =
        resident.deviceMemory->allocate(rawBytes, alignof(std::int32_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    const std::byte* records = mapped.data() + header.pointDataOffset;
    pdg::KnnConfigSummary summary = summarizeDirectLasCoordinates(
        header, records, resident.spatialDimensions);
    allocationTimer.finish();
    ResidentManagerDetailTimer submissionTimer(
        &ResidentManagerDetailSeconds::directLasHydrationSubmission);
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceRecords->data(), records, recordBytes,
                                   cudaMemcpyHostToDevice, stream));
    observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident.region,
                    recordBytes);
    pdg::las::decodeCoordinatesAsync(
        deviceRecords->data(), header.pointRecordLength, resident.pointCount,
        static_cast<std::int32_t*>(rawX->data()),
        static_cast<std::int32_t*>(rawY->data()),
        static_cast<std::int32_t*>(rawZ->data()), stream);
    pdg::las::expandCoordinatesAsync(
        static_cast<const std::int32_t*>(rawX->data()),
        static_cast<const std::int32_t*>(rawY->data()),
        static_cast<const std::int32_t*>(rawZ->data()), resident.pointCount,
        header.coordinateEncoding(), resident.device->data<double>(X),
        resident.device->data<double>(Y), resident.device->data<double>(Z),
        stream);
    submissionTimer.finish();
    {
        ResidentManagerDetailTimer waitTimer(
            &ResidentManagerDetailSeconds::directLasHydrationWait);
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (ActiveLasSourceUsed)
        *ActiveLasSourceUsed = true;
    if (ActiveLasHostXyzMirrored)
        *ActiveLasHostXyzMirrored = resident.host->has(X) ||
                                    resident.host->has(Y) ||
                                    resident.host->has(Z);
    return summary;
}

thread_local std::optional<std::uint64_t> RequiredBridgeRebuildRegion;

std::optional<std::uint32_t> requiredEigenSystemReuseNeighbors()
{
    const char* text =
        std::getenv("PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE");
    if (!text)
        return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || end == text || *end != '\0' || value < 3U || value > 64U)
        throw NeighborhoodProofFailure(
            "PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE must name an "
            "internal neighbor count from 3 through 64");
    return static_cast<std::uint32_t>(value);
}

bool requireBridgeRebuild(CudaNeighborhoodRegion region,
                          const std::shared_ptr<ResidentNeighborhood>& resident)
{
    if (!std::getenv("PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD") ||
        !region.reuseExpected)
        return false;
    if (!RequiredBridgeRebuildRegion ||
        *RequiredBridgeRebuildRegion != region.id)
        throw NeighborhoodProofFailure(
            "required unsupported resident bridge fallback did not occur");
    if (resident)
        throw NeighborhoodProofFailure(
            "resident neighborhood survived an unsupported bridge fallback");
    return true;
}

enum class EigenRepairAttribution
{
    Other,
    ApproximateCoplanar
};

void repairAmbiguousEigenSystems(ResidentNeighborhood& resident,
                                 PointView& view,
                                 CachedCudaEigenSystems& cached,
                                 std::uint32_t neighbors,
                                 EigenRepairAttribution attribution,
                                 bool terminalSink)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    ResidentManagerDetailSeconds* const detailPhases =
        activeResidentManagerDetailSeconds();
    std::vector<std::uint8_t> status(resident.pointCount);
    ResidentEigenFamilyDetailTimer statusWaitTimer(
        &ResidentEigenFamilyDetailSeconds::ambiguousRepairStatusWait);
    PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), cached.deviceStatus->data(),
                                   status.size(), cudaMemcpyDeviceToHost,
                                   stream));
    observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident.region,
                    status.size());
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    statusWaitTimer.finish();
    std::size_t ambiguousCount = 0;
    std::size_t incompleteCount = 0;
    std::size_t repairCount = 0;
    for (std::uint8_t value : status)
    {
        ambiguousCount +=
            static_cast<std::size_t>((value & pdg::KnnDistanceTie) != 0U);
        incompleteCount +=
            static_cast<std::size_t>((value & pdg::KnnSearchIncomplete) != 0U);
        repairCount += static_cast<std::size_t>(
            (value & (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) != 0U);
    }
    if (detailPhases)
    {
        detailPhases->eigenFamily.ambiguousRows += ambiguousCount;
        detailPhases->eigenFamily.incompleteRows += incompleteCount;
        detailPhases->eigenFamily.repairedRows += repairCount;
    }
    if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR") && repairCount == 0U)
        throw NeighborhoodProofFailure(
            "required resident neighborhood tie repair did not occur");
    if (repairCount == 0U)
        return;
    ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
    const bool attributeApproximateCoplanar =
        attribution == EigenRepairAttribution::ApproximateCoplanar;
    const auto repairStarted = std::chrono::steady_clock::now();
    if (attributeApproximateCoplanar && phases)
    {
        ++phases->approximateCoplanarRepairTriggers;
        phases->approximateCoplanarAmbiguousRepairRows += ambiguousCount;
        phases->approximateCoplanarIncompleteRepairRows += incompleteCount;
        phases->approximateCoplanarRepairRows += repairCount;
        phases->approximateCoplanarDeviceToHostRepairBytes += status.size();
    }
    std::vector<pdg::EigenSystem3d> systems(resident.pointCount);
    const std::size_t systemBytes = systems.size() * sizeof(pdg::EigenSystem3d);
    ResidentEigenFamilyDetailTimer downloadTimer(
        &ResidentEigenFamilyDetailSeconds::ambiguousRepairSystemDownload);
    PDG_CUDA_CHECK(cudaMemcpyAsync(systems.data(), cached.deviceSystems->data(),
                                   systemBytes, cudaMemcpyDeviceToHost,
                                   stream));
    observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident.region,
                    systemBytes);
    if (attributeApproximateCoplanar && phases)
        phases->approximateCoplanarDeviceToHostRepairBytes += systemBytes;
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    downloadTimer.finish();
    ResidentEigenFamilyDetailTimer indexTimer(
        &ResidentEigenFamilyDetailSeconds::ambiguousRepairIndexBuild);
    // The exact repair needs pinned nanoflann's tree, traversal, and tie
    // order over the whole view. When the planner proved that only the
    // terminal writer follows this region, no later stage can observe a
    // PointView KD3 product, so a private cached-coordinate tree (identical
    // results, about 4x cheaper to build and query; D0237/B0260) replaces the
    // published uncached one. Every other graph keeps publishing the uncached
    // tree so a later consumer sees exactly what pinned PDAL leaves it.
    std::unique_ptr<KD3Index> privateIndex;
    if (terminalSink)
    {
        privateIndex = std::make_unique<KD3Index>(view, true);
        privateIndex->build();
    }
    KD3Index& index = privateIndex ? *privateIndex : view.build3dIndex();
    indexTimer.finish();
    if (attributeApproximateCoplanar && phases)
        ++phases->approximateCoplanarKd3Uses;
    ResidentEigenFamilyDetailTimer rowsTimer(
        &ResidentEigenFamilyDetailSeconds::ambiguousRepairRows);
    for (PointId point = 0; point < view.size(); ++point)
    {
        const std::size_t offset = static_cast<std::size_t>(point);
        if ((status[offset] &
             (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) == 0U)
            continue;
        const EigenResult result = computeEigenSystem(
            view,
            index.neighbors(point, static_cast<point_count_t>(neighbors)));
        systems[offset] = result.system;
        status[offset] = result.status == EigenStatus::CovarianceZero
                             ? pdg::KnnCovarianceZero
                         : result.status == EigenStatus::SolverFailure
                             ? pdg::KnnEigenFailure
                             : pdg::KnnExact;
    }
    rowsTimer.finish();
    ResidentEigenFamilyDetailTimer uploadTimer(
        &ResidentEigenFamilyDetailSeconds::ambiguousRepairUpload);
    PDG_CUDA_CHECK(cudaMemcpyAsync(cached.deviceSystems->data(), systems.data(),
                                   systemBytes, cudaMemcpyHostToDevice,
                                   stream));
    observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident.region,
                    systemBytes);
    PDG_CUDA_CHECK(cudaMemcpyAsync(cached.deviceStatus->data(), status.data(),
                                   status.size(), cudaMemcpyHostToDevice,
                                   stream));
    observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident.region,
                    status.size());
    if (attributeApproximateCoplanar && phases)
    {
        phases->approximateCoplanarHostToDeviceRepairBytes +=
            systemBytes + status.size();
    }
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (attributeApproximateCoplanar && phases)
    {
        phases->ambiguousRepairRows += ambiguousCount;
        phases->incompleteRepairRows += incompleteCount;
        phases->repairedRows += repairCount;
        const double repairSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          repairStarted)
                .count();
        phases->exactHostRepair += repairSeconds;
        phases->approximateCoplanarExactHostRepair += repairSeconds;
    }
}

class UnsupportedResidentAssignments final : public std::runtime_error
{
public:
    UnsupportedResidentAssignments()
        : std::runtime_error(
              "point program cannot consume the resident neighborhood batch")
    {
    }
};

std::shared_ptr<ResidentNeighborhood>
makeResidentNeighborhood(PointView& view, CudaNeighborhoodRegion region,
                         std::uint32_t neighbors, double radius = 0.0)
{
    ResidentManagerDetailTimer setupTimer(
        &ResidentManagerDetailSeconds::residentProductSetup);
    auto resident = std::make_shared<ResidentNeighborhood>();
    resident->region = region.id;
    resident->radiusIndex = region.radiusIndex;
    resident->indexRequired = region.indexRequired;
    resident->spatialDimensions = static_cast<std::uint8_t>(region.dimensions);
    if (!resident->indexRequired)
    {
        resident->maximumNeighbors = 0U;
        resident->maximumRadius = 0.0;
    }
    else if (resident->radiusIndex)
    {
        resident->maximumRadius =
            region.maximumRadius > 0.0 ? region.maximumRadius : radius;
        if ((resident->spatialDimensions != 2U &&
             resident->spatialDimensions != 3U) ||
            !std::isfinite(radius) || radius <= 0.0 ||
            !std::isfinite(resident->maximumRadius) ||
            resident->maximumRadius < radius)
            throw std::invalid_argument(
                "invalid resident neighborhood radius envelope");
    }
    else
    {
        resident->maximumNeighbors =
            region.maximumNeighbors ? region.maximumNeighbors : neighbors;
        if ((resident->spatialDimensions != 2U &&
             resident->spatialDimensions != 3U) ||
            resident->maximumNeighbors < neighbors ||
            resident->maximumNeighbors > 64U)
            throw std::invalid_argument(
                "invalid resident neighborhood neighbor envelope");
    }
    resident->pointCount = static_cast<std::size_t>(view.size());
    resident->pinnedMemory = pdg::makeCudaPinnedMemoryResource();
    const bool directLasSource = !ActiveLasSource.empty();
    if (directLasSource &&
        (!resident->indexRequired || (resident->spatialDimensions != 2U &&
                                      resident->spatialDimensions != 3U)))
        throw NeighborhoodProofFailure(
            "direct resident LAS source reached an incompatible region");
    if (resident->indexRequired && !directLasSource)
        resident->host = std::make_unique<pdg::PointBatch>(
            gather(view, resident->dimensions, *resident->pinnedMemory));
    else
    {
        resident->host = std::make_unique<pdg::PointBatch>(
            resident->pointCount,
            pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            resident->dimensions, *resident->pinnedMemory);
        resident->host->setSize(resident->pointCount);
    }
    resident->deviceMemory = pdg::makeCudaMemoryResource();
    resident->device = std::make_unique<pdg::PointBatch>(
        resident->host->size(), resident->host->coordinateEncoding(),
        resident->dimensions, *resident->deviceMemory);
    resident->device->setSize(resident->host->size());
    setupTimer.finish();
    if (!resident->indexRequired)
        return resident;
    bool deviceCoordinatesReady = false;
    std::optional<pdg::KnnConfigSummary> directLasSummary;
    if (directLasSource)
    {
        ResidentManagerDetailTimer hydrationTimer(
            &ResidentManagerDetailSeconds::directLasCoordinateHydration);
        directLasSummary = hydrateResidentCoordinatesFromLas(*resident, view);
        deviceCoordinatesReady = true;
    }

    pdg::UniformGridConfig config;
    {
        ResidentManagerDetailTimer timer(
            &ResidentManagerDetailSeconds::indexConfiguration);
        {
            ResidentManagerDetailTimer selectionTimer(
                &ResidentManagerDetailSeconds::indexConfigSelection);
            if (directLasSummary)
                config = resident->radiusIndex
                             ? pdg::makeUniformGridConfig(
                                   *directLasSummary, resident->maximumRadius)
                             : selectCudaKnnConfig(*directLasSummary,
                                                   resident->maximumNeighbors);
            else
                config = resident->radiusIndex
                             ? pdg::makeUniformGridConfig(
                                   *resident->host, resident->spatialDimensions,
                                   resident->maximumRadius)
                             : selectCudaKnnConfig(*resident->host,
                                                   resident->spatialDimensions,
                                                   resident->maximumNeighbors);
            if (directLasSummary)
            {
                const char* required = std::getenv(
                    "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND");
                if (required)
                {
                    const bool uniform =
                        std::string_view(required) == "uniform_grid";
                    const bool morton =
                        std::string_view(required) == "morton_bvh";
                    if ((!uniform && !morton) ||
                        (uniform &&
                         config.backend !=
                             pdg::SpatialIndexBackend::UniformGrid) ||
                        (morton &&
                         config.backend != pdg::SpatialIndexBackend::MortonBvh))
                        throw NeighborhoodProofFailure(
                            "required direct LAS record-summary backend was "
                            "not selected");
                }
            }
        }
        if (resident->radiusIndex)
        {
            ResidentManagerDetailTimer validationTimer(
                &ResidentManagerDetailSeconds::indexEnvelopeValidation);
            const pdg::PointBatch& validationBatch =
                directLasSummary ? *resident->device : *resident->host;
            if (!pdg::uniformGridMaySupportExactDevice(validationBatch, config))
                return {};
        }
    }

    {
        ResidentManagerDetailTimer buildTimer(
            &ResidentManagerDetailSeconds::indexBuild);
        for (pdg::DimensionId dimension : {X, Y, Z})
            resident->device->materialize(dimension,
                                          pdg::DimensionType::Double);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        if (!deviceCoordinatesReady)
        {
            for (pdg::DimensionId dimension : {X, Y, Z})
            {
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(resident->device->rawData(dimension),
                                    resident->host->rawData(dimension),
                                    resident->host->size() * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
                observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                region.id,
                                resident->host->size() * sizeof(double));
            }
        }

        resident->index =
            std::make_unique<pdg::SpatialIndex>(*resident->device, config);
        resident->index->build();
        if (directLasSummary && ActiveLasRecordSummaryUsed)
            *ActiveLasRecordSummaryUsed = true;
        // Observed physical build of the shared spatial index.
        // Planner-selected regions carry their resident-region identifier;
        // an unplanned one-stage invocation reports region zero. record() is
        // a no-op without a scope.
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::IndexBuild,
            region.id ? static_cast<std::size_t>(region.id) - 1U : 0U,
            resident->index->allocatedBytes());
    }
    return resident;
}

bool matches(const ResidentNeighborhood& resident, const PointView& view,
             CudaNeighborhoodRegion region, std::uint32_t neighbors)
{
    const std::uint32_t maximumNeighbors =
        region.maximumNeighbors ? region.maximumNeighbors : neighbors;
    return !resident.radiusIndex && !region.radiusIndex && region.id != 0 &&
           resident.region == region.id &&
           resident.maximumNeighbors == maximumNeighbors &&
           resident.spatialDimensions == region.dimensions &&
           resident.pointCount == static_cast<std::size_t>(view.size()) &&
           neighbors <= resident.maximumNeighbors;
}

bool matchesRadius(const ResidentNeighborhood& resident, const PointView& view,
                   CudaNeighborhoodRegion region, double radius,
                   std::uint8_t dimensions)
{
    const double maximumRadius =
        region.maximumRadius > 0.0 ? region.maximumRadius : radius;
    return resident.radiusIndex && region.radiusIndex && region.id != 0 &&
           resident.region == region.id &&
           resident.maximumRadius == maximumRadius &&
           resident.spatialDimensions == dimensions &&
           resident.pointCount == static_cast<std::size_t>(view.size()) &&
           radius <= resident.maximumRadius;
}

struct OutputColumn
{
    pdg::DimensionId device;
    Dimension::Id pointView;
    pdg::DimensionType physical = pdg::DimensionType::Double;
    Dimension::Type pointViewType = Dimension::Type::Double;
};

constexpr OutputColumn NormalColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::NormalX), Dimension::Id::NormalX},
    {pdg::DimensionId(pdg::StandardDimension::NormalY), Dimension::Id::NormalY},
    {pdg::DimensionId(pdg::StandardDimension::NormalZ), Dimension::Id::NormalZ},
    {pdg::DimensionId(pdg::StandardDimension::Curvature),
     Dimension::Id::Curvature}};

constexpr OutputColumn EigenvalueColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::Eigenvalue0),
     Dimension::Id::Eigenvalue0},
    {pdg::DimensionId(pdg::StandardDimension::Eigenvalue1),
     Dimension::Id::Eigenvalue1},
    {pdg::DimensionId(pdg::StandardDimension::Eigenvalue2),
     Dimension::Id::Eigenvalue2}};

constexpr OutputColumn NnDistanceColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::NNDistance),
     Dimension::Id::NNDistance}};

constexpr OutputColumn RadialDensityColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::RadialDensity),
     Dimension::Id::RadialDensity}};

constexpr OutputColumn HagNnColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::HeightAboveGround),
     Dimension::Id::HeightAboveGround}};

constexpr OutputColumn ApproximateCoplanarColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::Coplanar),
     Dimension::Id::Coplanar, pdg::DimensionType::Unsigned8,
     Dimension::Type::Unsigned8}};

constexpr OutputColumn ClassificationColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::Classification),
     Dimension::Id::Classification, pdg::DimensionType::Unsigned8,
     Dimension::Type::Unsigned8}};

constexpr OutputColumn OptimalColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::OptimalKNN),
     Dimension::Id::OptimalKNN, pdg::DimensionType::Unsigned64,
     Dimension::Type::Unsigned64},
    {pdg::DimensionId(pdg::StandardDimension::OptimalRadius),
     Dimension::Id::OptimalRadius}};

constexpr OutputColumn RankColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::Rank), Dimension::Id::Rank,
     pdg::DimensionType::Unsigned8, Dimension::Type::Unsigned8}};

constexpr OutputColumn LofColumns[]{
    {pdg::DimensionId(pdg::StandardDimension::NNDistance),
     Dimension::Id::NNDistance},
    {pdg::DimensionId(pdg::StandardDimension::LocalReachabilityDistance),
     Dimension::Id::LocalReachabilityDistance},
    {pdg::DimensionId(pdg::StandardDimension::LocalOutlierFactor),
     Dimension::Id::LocalOutlierFactor}};

void prepareOutputColumns(ResidentNeighborhood& resident, PointView& view,
                          std::span<const OutputColumn> columns)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    for (const OutputColumn& column : columns)
    {
        if (resident.device->has(column.device))
        {
            if (resident.device->columnInfo(column.device).physicalType !=
                column.physical)
                throw std::invalid_argument(
                    "resident neighborhood output column type mismatch");
            continue;
        }
        resident.host->materialize(column.device, column.physical);
        resident.device->materialize(column.device, column.physical);
        std::byte* host =
            static_cast<std::byte*>(resident.host->rawData(column.device));
        const std::size_t stride = pdg::dimensionTypeSize(column.physical);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto started = std::chrono::steady_clock::now();
        gatherColumnFromView(view, column.pointView, column.pointViewType,
                             rowBoundaryUsable(view), host, stride);
        accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr, started);
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident.device->rawData(column.device),
                                       host, resident.pointCount * stride,
                                       cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident.region,
                        resident.pointCount * stride);
    }
}

void refreshOutputColumnsFromView(ResidentNeighborhood& resident,
                                  PointView& view,
                                  std::span<const OutputColumn> columns)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    const bool physicalRows = rowBoundaryUsable(view);
    for (const OutputColumn& column : columns)
    {
        if (!resident.device->has(column.device))
        {
            prepareOutputColumns(resident, view,
                                 std::span<const OutputColumn>(&column, 1U));
            continue;
        }
        if (resident.device->columnInfo(column.device).physicalType !=
            column.physical)
            throw std::invalid_argument(
                "resident neighborhood output column type mismatch");
        if (!resident.host->has(column.device))
            resident.host->materialize(column.device, column.physical);
        std::byte* host =
            static_cast<std::byte*>(resident.host->rawData(column.device));
        const std::size_t stride = pdg::dimensionTypeSize(column.physical);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto started = std::chrono::steady_clock::now();
        gatherColumnFromView(view, column.pointView, column.pointViewType,
                             physicalRows, host, stride);
        accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr, started);
        const std::size_t bytes = resident.pointCount * stride;
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident.device->rawData(column.device),
                                       host, bytes, cudaMemcpyHostToDevice,
                                       stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident.region,
                        bytes);
    }
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void copyOutputColumnsToHost(ResidentNeighborhood& resident,
                             CachedCudaEigenSystems& cached,
                             std::span<const OutputColumn> columns,
                             bool copyEigenvalues)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    for (const OutputColumn& column : columns)
    {
        const std::size_t bytes =
            resident.pointCount * pdg::dimensionTypeSize(column.physical);
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident.host->rawData(column.device),
                                       resident.device->rawData(column.device),
                                       bytes, cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident.region,
                        bytes);
    }
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        cached.host->status.data(), cached.deviceProjectionStatus->data(),
        resident.pointCount, cudaMemcpyDeviceToHost, stream));
    observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident.region,
                    resident.pointCount);
    if (copyEigenvalues)
    {
        if (!cached.hostEigenvalues)
            cached.hostEigenvalues = resident.pinnedMemory->allocate(
                resident.pointCount * 3U * sizeof(double), alignof(double));
        PDG_CUDA_CHECK(cudaMemcpy2DAsync(
            cached.hostEigenvalues->data(), 3U * sizeof(double),
            cached.deviceSystems->data(), sizeof(pdg::EigenSystem3d),
            3U * sizeof(double), resident.pointCount, cudaMemcpyDeviceToHost,
            stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident.region,
                        resident.pointCount * 3U * sizeof(double));
    }
    ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
    const auto waitStarted = std::chrono::steady_clock::now();
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr, waitStarted);
}

void publishOutputColumns(const ResidentNeighborhood& resident, PointView& view,
                          std::span<const OutputColumn> columns)
{
    ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
    const auto started = std::chrono::steady_clock::now();
    const bool physicalRows = rowBoundaryUsable(view);
    for (const OutputColumn& column : columns)
    {
        const std::byte* values = static_cast<const std::byte*>(
            resident.host->rawData(column.device));
        const std::size_t stride = pdg::dimensionTypeSize(column.physical);
        publishColumnToView(view, column.pointView, column.pointViewType,
                            physicalRows, values, stride);
    }
    accumulatePhaseSeconds(phases ? &phases->spillPublish : nullptr, started);
}

const CachedHagNnCompatibilityGround& acquireHagNnCompatibilityGround(
    ResidentNeighborhood& resident, PointView& view,
    std::span<const std::uint8_t> groundMask)
{
    if (resident.hagNnCompatibilityGround)
        return *resident.hagNnCompatibilityGround;

    // Build locally and publish only after the view, bounds, and pinned tree
    // are complete. A failure therefore leaves no partial planner product.
    auto product = std::make_unique<CachedHagNnCompatibilityGround>();
    product->view = view.makeNew();
    for (PointId point = 0U; point < view.size(); ++point)
        if (groundMask[static_cast<std::size_t>(point)] != 0U)
            product->view->appendPoint(view, point);
    product->view->calculateBounds(product->bounds);
    product->index = &product->view->build2dIndex();
    resident.hagNnCompatibilityGround = std::move(product);
    if (ResidentPhaseSeconds* phases = activeResidentPhaseSeconds())
        ++phases->hagNnCompatibilityIndexBuilds;
    return *resident.hagNnCompatibilityGround;
}

// Exact transcription of the pinned filters.hag_nn ground-domain query and
// interpolation for the rows whose shared CUDA search reports a distance tie
// or an incomplete bounded search. The CUDA result remains authoritative for
// every exact row; the planner-owned compatibility product exists only for
// the bounded repair set and preserves pinned insertion/query order.
void repairHagNnRowsExact(ResidentNeighborhood& resident, PointView& view,
                          const pdg::HagNnProgram& program,
                          std::span<const std::uint8_t> groundMask,
                          std::span<const std::uint8_t> sourceMask,
                          std::span<const std::uint8_t> status,
                          double* output)
{
    const CachedHagNnCompatibilityGround& ground =
        acquireHagNnCompatibilityGround(resident, view, groundMask);
    const KD2Index& index = *ground.index;
    const double maximumDistanceSquared =
        std::pow(program.maximumDistance, 2.0);
    for (PointId point = 0U; point < view.size(); ++point)
    {
        const std::size_t offset = static_cast<std::size_t>(point);
        if (sourceMask[offset] == 0U || status[offset] == pdg::KnnExact)
            continue;

        const double x0 =
            view.getFieldAs<double>(Dimension::Id::X, point);
        const double y0 =
            view.getFieldAs<double>(Dimension::Id::Y, point);
        const double z0 =
            view.getFieldAs<double>(Dimension::Id::Z, point);
        PointIdList ids(program.count);
        std::vector<double> squaredDistances(program.count);
        index.knnSearch(x0, y0, program.count, &ids, &squaredDistances);
        const double x =
            ground.view->getFieldAs<double>(Dimension::Id::X, ids[0]);
        const double y =
            ground.view->getFieldAs<double>(Dimension::Id::Y, ids[0]);
        const double z =
            ground.view->getFieldAs<double>(Dimension::Id::Z, ids[0]);
        double groundZ = z0;
        if ((x0 == x && y0 == y) || ids.size() == 1U)
            groundZ = z;
        else if (!ground.bounds.contains(x0, y0) &&
                 !program.allowExtrapolation)
            groundZ = z0;
        else
        {
            double weights = 0.0;
            double accumulatedZ = 0.0;
            for (std::size_t item = 0U; item < ids.size(); ++item)
            {
                const double groundPointZ = ground.view->getFieldAs<double>(
                    Dimension::Id::Z, ids[item]);
                const double squaredDistance = squaredDistances[item];
                if (maximumDistanceSquared > 0.0 &&
                    squaredDistance > maximumDistanceSquared)
                    break;
                const double weight = 1.0 / squaredDistance;
                weights += weight;
                accumulatedZ += weight * groundPointZ;
            }
            groundZ = weights ? accumulatedZ / weights : z0;
        }
        output[offset] = z0 - groundZ;
    }
}

std::vector<OutputColumn> covarianceColumns(std::uint32_t features)
{
    std::vector<OutputColumn> columns;
    const auto add = [&](std::uint32_t feature, pdg::StandardDimension device,
                         Dimension::Id pointView)
    {
        if ((features & feature) != 0U)
            columns.push_back({pdg::DimensionId(device), pointView});
    };
    add(pdg::CovarianceLinearity, pdg::StandardDimension::Linearity,
        Dimension::Id::Linearity);
    add(pdg::CovariancePlanarity, pdg::StandardDimension::Planarity,
        Dimension::Id::Planarity);
    add(pdg::CovarianceScattering, pdg::StandardDimension::Scattering,
        Dimension::Id::Scattering);
    add(pdg::CovarianceVerticality, pdg::StandardDimension::Verticality,
        Dimension::Id::Verticality);
    add(pdg::CovarianceOmnivariance, pdg::StandardDimension::Omnivariance,
        Dimension::Id::Omnivariance);
    add(pdg::CovarianceAnisotropy, pdg::StandardDimension::Anisotropy,
        Dimension::Id::Anisotropy);
    add(pdg::CovarianceEigenentropy, pdg::StandardDimension::Eigenentropy,
        Dimension::Id::Eigenentropy);
    add(pdg::CovarianceEigenvalueSum, pdg::StandardDimension::EigenvalueSum,
        Dimension::Id::EigenvalueSum);
    add(pdg::CovarianceSurfaceVariation,
        pdg::StandardDimension::SurfaceVariation,
        Dimension::Id::SurfaceVariation);
    add(pdg::CovarianceDemantkeVerticality,
        pdg::StandardDimension::DemantkeVerticality,
        Dimension::Id::DemantkeVerticality);
    return columns;
}

void computeHostTranscendentalFeatures(ResidentNeighborhood& resident,
                                       const CachedCudaEigenSystems& cached,
                                       pdg::EigenvalueMode mode,
                                       std::uint32_t features)
{
    if (!features)
        return;
    const double* systems =
        static_cast<const double*>(cached.hostEigenvalues->data());
    double* omnivariance = (features & pdg::CovarianceOmnivariance) != 0U
                               ? resident.host->data<double>(pdg::DimensionId(
                                     pdg::StandardDimension::Omnivariance))
                               : nullptr;
    double* eigenentropy = (features & pdg::CovarianceEigenentropy) != 0U
                               ? resident.host->data<double>(pdg::DimensionId(
                                     pdg::StandardDimension::Eigenentropy))
                               : nullptr;
    for (std::size_t point = 0; point < resident.pointCount; ++point)
    {
        if (cached.host->status[point] != pdg::KnnExact)
            continue;
        const double* values = systems + point * 3U;
        std::array<double, 3> lambda{(std::max)(values[2], 0.0),
                                     (std::max)(values[1], 0.0),
                                     (std::max)(values[0], 0.0)};
        const double sum = lambda[0] + lambda[1] + lambda[2];
        if (lambda[0] == 0.0)
            continue;
        if (mode == pdg::EigenvalueMode::Sqrt)
            for (double& value : lambda)
                value = std::sqrt(value);
        else if (mode == pdg::EigenvalueMode::Normalized)
            for (double& value : lambda)
                value /= sum;
        if (omnivariance)
            omnivariance[point] = std::cbrt(lambda[2] * lambda[1] * lambda[0]);
        if (eigenentropy)
            eigenentropy[point] = -(lambda[2] * std::log(lambda[2]) +
                                    lambda[1] * std::log(lambda[1]) +
                                    lambda[0] * std::log(lambda[0]));
    }
}

void uploadHostColumns(ResidentNeighborhood& resident,
                       std::span<const OutputColumn> columns,
                       std::uint32_t hostFeatures)
{
    if (!hostFeatures)
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(resident.device->nativeStreamHandle());
    for (const OutputColumn& column : columns)
    {
        const bool omnivariance =
            column.device ==
            pdg::DimensionId(pdg::StandardDimension::Omnivariance);
        const bool eigenentropy =
            column.device ==
            pdg::DimensionId(pdg::StandardDimension::Eigenentropy);
        if ((!omnivariance ||
             (hostFeatures & pdg::CovarianceOmnivariance) == 0U) &&
            (!eigenentropy ||
             (hostFeatures & pdg::CovarianceEigenentropy) == 0U))
            continue;
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            resident.device->rawData(column.device),
            resident.host->rawData(column.device),
            resident.pointCount * pdg::dimensionTypeSize(column.physical),
            cudaMemcpyHostToDevice, stream));
    }
}

template <typename Projection>
bool tryCudaColumnsImpl(PointView& view, std::uint32_t neighbors,
                        CudaNeighborhoodRegion region,
                        std::span<const OutputColumn> columns,
                        Projection&& projection, pdg::EigenvalueMode hostMode,
                        std::uint32_t hostFeatures,
                        EigenRepairAttribution repairAttribution,
                        std::shared_ptr<const CudaNeighborhoodResults>& result,
                        bool requireCuda)
{
    result.reset();
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 3U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    // An empty delegated region has no index or output work, but it must
    // remain an exact successful no-op so the terminal wrapper can close the
    // resident execution context. Nonterminal empty stages similarly leave
    // the context active for the next no-op bridge.
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        result = std::make_shared<CudaNeighborhoodResults>();
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (std::getenv("PDG_REQUIRE_HAG_NN_TIE_FALLBACK") ||
                    std::getenv("PDG_REQUIRE_HAG_NN_HOST_FALLBACK"))
                    throw NeighborhoodProofFailure(
                        "required HAG CUDA fallback had no CUDA device");
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        ResidentManagerDetailTimer queryTimer(
            &ResidentManagerDetailSeconds::neighborhoodQueryProjection);
        auto cached = resident->eigenSystems.find(neighbors);
        const std::optional<std::uint32_t> requiredReuse =
            requiredEigenSystemReuseNeighbors();
        if (region.reuseExpected && requiredReuse &&
            neighbors == *requiredReuse &&
            cached == resident->eigenSystems.end())
            throw NeighborhoodProofFailure(
                "required resident eigensystem cache reuse did not occur");
        if (cached == resident->eigenSystems.end())
        {
            auto entry = std::make_unique<CachedCudaEigenSystems>();
            entry->host = std::make_shared<CudaNeighborhoodResults>();
            entry->host->status.resize(resident->pointCount);
            entry->deviceSystems = resident->deviceMemory->allocate(
                resident->pointCount * sizeof(pdg::EigenSystem3d),
                alignof(pdg::EigenSystem3d));
            entry->deviceStatus = resident->deviceMemory->allocate(
                resident->pointCount, alignof(std::uint8_t));
            entry->deviceProjectionStatus = resident->deviceMemory->allocate(
                resident->pointCount, alignof(std::uint8_t));
            {
                ResidentEigenFamilyDetailTimer timer(
                    &ResidentEigenFamilyDetailSeconds::eigenSystemsSubmission);
                pdg::knnEigenSystems(
                    *resident->index, neighbors,
                    static_cast<pdg::EigenSystem3d*>(
                        entry->deviceSystems->data()),
                    static_cast<std::uint8_t*>(entry->deviceStatus->data()));
            }
            {
                ResidentEigenFamilyDetailTimer timer(
                    &ResidentEigenFamilyDetailSeconds::ambiguousRepair);
                repairAmbiguousEigenSystems(*resident, view, *entry, neighbors,
                                            repairAttribution,
                                            region.terminalSink);
            }
            cached = resident->eigenSystems.emplace(neighbors, std::move(entry))
                         .first;
        }
        CachedCudaEigenSystems& entry = *cached->second;
        {
            ResidentEigenFamilyDetailTimer timer(
                &ResidentEigenFamilyDetailSeconds::outputPreparation);
            prepareOutputColumns(*resident, view, columns);
        }
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
#ifdef PDG_UNIT_TESTING
        if (const char* configured =
                std::getenv("PDG_TEST_NEIGHBORHOOD_EIGEN_FAILURE_POINT"))
        {
            const std::string_view text(configured);
            std::size_t point = 0;
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), point);
            if (error != std::errc{} || end != text.data() + text.size() ||
                point >= resident->pointCount)
                throw std::invalid_argument(
                    "invalid injected neighborhood eigen-failure point");
            const std::uint8_t failure = pdg::KnnEigenFailure;
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::uint8_t*>(entry.deviceStatus->data()) + point,
                &failure, sizeof(failure), cudaMemcpyHostToDevice, stream));
            entry.statusKnown = false;
        }
#endif
        {
            ResidentEigenFamilyDetailTimer timer(
                &ResidentEigenFamilyDetailSeconds::projectionAndCopy);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                entry.deviceProjectionStatus->data(),
                entry.deviceStatus->data(), resident->pointCount,
                cudaMemcpyDeviceToDevice, stream));
            projection(*resident->device,
                       static_cast<const pdg::EigenSystem3d*>(
                           entry.deviceSystems->data()),
                       static_cast<std::uint8_t*>(
                           entry.deviceProjectionStatus->data()));
            copyOutputColumnsToHost(*resident, entry, columns,
                                    hostFeatures != 0U);
        }
        if (!entry.statusKnown)
        {
            ResidentEigenFamilyDetailTimer timer(
                &ResidentEigenFamilyDetailSeconds::statusScan);
            entry.exact = true;
            for (std::uint8_t pointStatus : entry.host->status)
            {
                if ((pointStatus &
                     (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) != 0U)
                    entry.exact = false;
            }
            entry.statusKnown = true;
        }
        if (entry.exact)
        {
            {
                ResidentEigenFamilyDetailTimer timer(
                    &ResidentEigenFamilyDetailSeconds::transcendentalFeatures);
                computeHostTranscendentalFeatures(*resident, entry, hostMode,
                                                  hostFeatures);
            }
            {
                ResidentEigenFamilyDetailTimer timer(
                    &ResidentEigenFamilyDetailSeconds::columnPublication);
                publishOutputColumns(*resident, view, columns);
            }
            {
                ResidentEigenFamilyDetailTimer timer(
                    &ResidentEigenFamilyDetailSeconds::hostColumnUpload);
                uploadHostColumns(*resident, columns, hostFeatures);
            }
        }
        result = entry.host;
        const bool exact = entry.exact;
        if (exact)
            closeRegion();
        else
            invalidateRegion();
        return exact;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}
} // unnamed namespace

bool tryCudaRadiusCounts(PointView& view, double radius,
                         std::vector<std::uint32_t>& counts, bool requireCuda)
{
    return tryCudaRadiusOutput(
        view, radius, counts, requireCuda,
        [radius](const pdg::SpatialIndex& index, std::uint32_t* output)
        { pdg::radiusCounts(index, radius, output); },
        [radius](const pdg::PointBatch& host, const pdg::SpatialTileSet& tiles,
                 pdg::DimensionRegistry& dimensions,
                 pdg::MemoryResource& staging, pdg::MemoryResource& device,
                 std::vector<std::uint32_t>& output)
        {
            return pdg::tiledRadiusCounts(host, tiles, 3, radius, dimensions,
                                          staging, device, output);
        });
}

bool tryCudaRadiusOutlier(PointView& view, double radius, int minimumNeighbors,
                          std::uint8_t classification,
                          CudaNeighborhoodRegion region,
                          CudaRadiusOutlierResult& result, bool requireCuda)
{
    result = {};
    region.radiusIndex = true;
    region.dimensions = 3U;
    if (region.maximumRadius <= 0.0)
        region.maximumRadius = radius;
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(region.maximumRadius) || region.maximumRadius < radius)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        closeRegion();
        return true;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matchesRadius(*resident, view, region, radius, 3U))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw NeighborhoodProofFailure(
            "required resident radius-outlier index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radius-outlier rebuild had no "
                        "CUDA device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, 0U, radius);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radius-outlier rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        const std::size_t countBytes =
            checkedProduct(resident->pointCount, sizeof(std::uint32_t),
                           "resident radius-outlier count bytes overflow");
        std::unique_ptr<pdg::Allocation> deviceCounts =
            resident->deviceMemory->allocate(countBytes,
                                             alignof(std::uint32_t));
        {
            ResidentManagerDetailTimer queryTimer(
                &ResidentManagerDetailSeconds::neighborhoodQueryProjection);
            pdg::radiusCounts(
                *resident->index, radius,
                static_cast<std::uint32_t*>(deviceCounts->data()));
        }

        std::vector<std::uint32_t> counts(resident->pointCount);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(counts.data(), deviceCounts->data(),
                                       countBytes, cudaMemcpyDeviceToHost,
                                       stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        countBytes);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);

        const std::size_t threshold =
            static_cast<std::size_t>(minimumNeighbors);
        for (std::uint32_t count : counts)
        {
            if (static_cast<std::size_t>(count) > threshold)
                ++result.inliers;
            else
                ++result.outliers;
        }
        if (result.inliers != 0U && result.outliers != 0U)
        {
            const auto publishStarted = std::chrono::steady_clock::now();
            for (PointId point = 0U; point < view.size(); ++point)
                if (static_cast<std::size_t>(
                        counts[static_cast<std::size_t>(point)]) <= threshold)
                    view.setField(Dimension::Id::Classification, point,
                                  classification);
            accumulatePhaseSeconds(phases ? &phases->spillPublish : nullptr,
                                   publishStarted);
            if (region.id != 0U && !region.last)
                refreshOutputColumnsFromView(*resident, view,
                                             ClassificationColumns);
        }
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaRadiusScaledValues(PointView& view, double radius, double factor,
                               std::vector<double>& values, bool requireCuda)
{
    return tryCudaRadiusOutput(
        view, radius, values, requireCuda,
        [radius, factor](const pdg::SpatialIndex& index, double* output)
        { pdg::radiusScaledValues(index, radius, factor, output); },
        [radius, factor](
            const pdg::PointBatch& host, const pdg::SpatialTileSet& tiles,
            pdg::DimensionRegistry& dimensions, pdg::MemoryResource& staging,
            pdg::MemoryResource& device, std::vector<double>& output)
        {
            return pdg::tiledRadiusScaledValues(host, tiles, 3, radius, factor,
                                                dimensions, staging, device,
                                                output);
        });
}

bool tryCudaResidentRadiusScaledValues(PointView& view, double radius,
                                       double factor,
                                       CudaNeighborhoodRegion region,
                                       bool requireCuda)
{
    region.radiusIndex = true;
    region.dimensions = 3U;
    if (region.maximumRadius <= 0.0)
        region.maximumRadius = radius;
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(region.maximumRadius) || region.maximumRadius < radius)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        closeRegion();
        return true;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matchesRadius(*resident, view, region, radius, 3U))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident radial-density index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radial-density rebuild had no "
                        "CUDA device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, 0U, radius);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radial-density rebuild was not "
                        "created");
                return false;
            }
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        const OutputColumn& column = RadialDensityColumns[0];
        if (resident->device->has(column.device))
        {
            if (resident->device->columnInfo(column.device).physicalType !=
                column.physical)
                throw std::invalid_argument(
                    "resident radial-density output column type mismatch");
        }
        else
        {
            resident->host->materialize(column.device, column.physical);
            resident->device->materialize(column.device, column.physical);
        }

        {
            ResidentManagerDetailTimer queryTimer(
                &ResidentManagerDetailSeconds::neighborhoodQueryProjection);
            pdg::radiusScaledValues(
                *resident->index, radius, factor,
                resident->device->data<double>(column.device));
        }

        if (!region.last)
        {
            ResidentPointViewAccess::set(view, resident);
            return true;
        }

        const std::size_t bytes =
            checkedProduct(resident->pointCount, sizeof(double),
                           "resident radial-density output bytes overflow");
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident->host->rawData(column.device),
                                       resident->device->rawData(column.device),
                                       bytes, cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        bytes);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);
        const auto publishStarted = std::chrono::steady_clock::now();
        publishColumnToView(view, column.pointView, column.pointViewType,
                            rowBoundaryUsable(view),
                            static_cast<const std::byte*>(
                                resident->host->rawData(column.device)),
                            sizeof(double));
        accumulatePhaseSeconds(phases ? &phases->spillPublish : nullptr,
                               publishStarted);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaRadiusAssign(PointView& view, double radius, bool search3d,
                         double maximumAbove, double maximumBelow,
                         const DimRangeList& sourceDomain,
                         const DimRangeList& referenceDomain,
                         std::vector<expr::AssignStatement>& updates,
                         CudaNeighborhoodRegion region, bool requireCuda)
{
    const std::uint8_t dimensions = search3d ? 3U : 2U;
    region.radiusIndex = true;
    region.dimensions = dimensions;
    if (region.maximumRadius <= 0.0)
        region.maximumRadius = radius;
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (!std::isfinite(radius) || radius <= 0.0 ||
        region.maximumRadius < radius)
    {
        invalidateRegion();
        return false;
    }
    std::vector<Dimension::Id> targets;
    for (expr::AssignStatement& update : updates)
    {
        const Dimension::Id target = update.identExpr().eval();
        if (target == Dimension::Id::Unknown || target == Dimension::Id::X ||
            target == Dimension::Id::Y || target == Dimension::Id::Z)
        {
            invalidateRegion();
            return false;
        }
        if (std::find(targets.begin(), targets.end(), target) == targets.end())
            targets.push_back(target);
    }
    if (view.empty())
    {
        closeRegion();
        return true;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matchesRadius(*resident, view, region, radius, dimensions))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident radius index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    const auto physicalType = [](Dimension::Type type)
    {
        switch (type)
        {
        case Dimension::Type::Signed8:
            return pdg::DimensionType::Signed8;
        case Dimension::Type::Signed16:
            return pdg::DimensionType::Signed16;
        case Dimension::Type::Signed32:
            return pdg::DimensionType::Signed32;
        case Dimension::Type::Signed64:
            return pdg::DimensionType::Signed64;
        case Dimension::Type::Unsigned8:
            return pdg::DimensionType::Unsigned8;
        case Dimension::Type::Unsigned16:
            return pdg::DimensionType::Unsigned16;
        case Dimension::Type::Unsigned32:
            return pdg::DimensionType::Unsigned32;
        case Dimension::Type::Unsigned64:
            return pdg::DimensionType::Unsigned64;
        case Dimension::Type::Float:
            return pdg::DimensionType::Float;
        case Dimension::Type::Double:
            return pdg::DimensionType::Double;
        case Dimension::Type::None:
            return pdg::DimensionType::None;
        }
        return pdg::DimensionType::None;
    };

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radius rebuild had no CUDA device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, 0U, radius);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident radius rebuild was not created");
                return false;
            }
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        const std::size_t count = resident->pointCount;
        std::vector<std::uint8_t> sourceMask(count, 1U);
        std::vector<std::uint8_t> referenceMask(count, 1U);
        std::vector<std::uint8_t> matchesMask(count, 0U);
        PointRef point(view, 0);
        const auto fillMask =
            [&](const DimRangeList& domain, std::vector<std::uint8_t>& mask)
        {
            if (domain.ranges().empty())
                return;
            std::fill(mask.begin(), mask.end(), 0U);
            for (PointId id = 0; id < view.size(); ++id)
            {
                point.setPointId(id);
                for (const DimRange& range : domain.ranges())
                    if (range.valuePasses(point.getFieldAs<double>(range.m_id)))
                    {
                        mask[static_cast<std::size_t>(id)] = 1U;
                        break;
                    }
            }
        };
        fillMask(sourceDomain, sourceMask);
        fillMask(referenceDomain, referenceMask);

        std::unique_ptr<pdg::Allocation> deviceSource =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceReference =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceMatches =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSource->data(), sourceMask.data(),
                                       count, cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceReference->data(),
                                       referenceMask.data(), count,
                                       cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        2U * count);
        pdg::radiusAny(
            *resident->index, radius,
            static_cast<const std::uint8_t*>(deviceSource->data()),
            static_cast<const std::uint8_t*>(deviceReference->data()),
            maximumAbove, maximumBelow,
            static_cast<std::uint8_t*>(deviceMatches->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(matchesMask.data(),
                                       deviceMatches->data(), count,
                                       cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        count);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);

        for (PointId id = 0; id < view.size(); ++id)
        {
            if (!matchesMask[static_cast<std::size_t>(id)])
                continue;
            point.setPointId(id);
            for (expr::AssignStatement& update : updates)
                if (update.conditionalExpr().eval(point))
                    point.setField(update.identExpr().eval(),
                                   update.valueExpr().eval(point));
        }

        const bool physicalRows = rowBoundaryUsable(view);
        for (Dimension::Id target : targets)
        {
            const std::string name = view.layout()->dimName(target);
            const Dimension::Type pointViewType =
                view.layout()->dimType(target);
            const pdg::DimensionType type = physicalType(pointViewType);
            if (type == pdg::DimensionType::None)
                throw UnsupportedResidentAssignments();
            const pdg::DimensionDefinition* definition =
                resident->dimensions.find(name);
            if (!definition)
                definition = &resident->dimensions.registerCustom(name, type);
            if (resident->device->has(definition->id) &&
                resident->device->columnInfo(definition->id).physicalType !=
                    type)
                throw UnsupportedResidentAssignments();
            if (!resident->device->has(definition->id))
            {
                resident->host->materialize(definition->id, type);
                resident->device->materialize(definition->id, type);
            }
            const std::size_t stride = pdg::dimensionTypeSize(type);
            const auto packStarted = std::chrono::steady_clock::now();
            gatherColumnFromView(view, target, pointViewType, physicalRows,
                                 static_cast<std::byte*>(
                                     resident->host->rawData(definition->id)),
                                 stride);
            accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr,
                                   packStarted);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                resident->device->rawData(definition->id),
                resident->host->rawData(definition->id), count * stride,
                cudaMemcpyHostToDevice, stream));
            observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                            resident->region, count * stride);
        }
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaNormalColumns(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    bool alwaysUp, std::shared_ptr<const CudaNeighborhoodResults>& result,
    bool requireCuda)
{
    const pdg::NormalProgram program{static_cast<std::int32_t>(neighbors - 1U),
                                     alwaysUp};
    return tryCudaColumnsImpl(
        view, neighbors, region, NormalColumns,
        [&program](pdg::PointBatch& batch, const pdg::EigenSystem3d* systems,
                   std::uint8_t* status)
        { pdg::projectNormalColumns(batch, systems, status, program); },
        pdg::EigenvalueMode::Raw, 0U, EigenRepairAttribution::Other, result,
        requireCuda);
}

bool tryCudaEigenvalueColumns(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    bool normalize, std::shared_ptr<const CudaNeighborhoodResults>& result,
    bool requireCuda)
{
    const pdg::EigenvaluesProgram program{
        static_cast<std::int32_t>(neighbors - 1U), normalize};
    return tryCudaColumnsImpl(
        view, neighbors, region, EigenvalueColumns,
        [&program](pdg::PointBatch& batch, const pdg::EigenSystem3d* systems,
                   std::uint8_t* status)
        { pdg::projectEigenvalueColumns(batch, systems, status, program); },
        pdg::EigenvalueMode::Raw, 0U, EigenRepairAttribution::Other, result,
        requireCuda);
}

bool tryCudaApproximateCoplanarColumn(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    double threshold1, double threshold2,
    std::shared_ptr<const CudaNeighborhoodResults>& result, bool requireCuda)
{
    const pdg::ApproximateCoplanarProgram program{
        static_cast<std::int32_t>(neighbors), threshold1, threshold2};
    return tryCudaColumnsImpl(
        view, neighbors, region, ApproximateCoplanarColumns,
        [&program](pdg::PointBatch& batch, const pdg::EigenSystem3d* systems,
                   std::uint8_t* status)
        {
            pdg::projectApproximateCoplanarColumn(batch, systems, status,
                                                  program);
        },
        pdg::EigenvalueMode::Raw, 0U,
        EigenRepairAttribution::ApproximateCoplanar, result, requireCuda);
}

bool tryCudaNnDistanceColumns(PointView& view, std::uint32_t neighbors,
                              CudaNeighborhoodRegion region,
                              pdg::KnnDistanceMode mode, bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 2U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        ResidentManagerDetailTimer queryTimer(
            &ResidentManagerDetailSeconds::neighborhoodQueryProjection);
        const pdg::DimensionId nnDistance(pdg::StandardDimension::NNDistance);
        const bool deviceOnlyHandoff = ActiveDeviceOnlyNnDistanceHandoff &&
                                       !ActiveLasSource.empty() &&
                                       region.id != 0U && !region.last;
        const bool plannerGatherReuse = region.gatherNeighbors != 0U;
        const bool requireGatherReuse =
            std::getenv("PDG_REQUIRE_KNN_GATHER_REUSE") != nullptr;
        if (requireGatherReuse && !plannerGatherReuse)
            throw NeighborhoodProofFailure(
                "required resident k-nearest gather reuse lacks planner "
                "provenance");
        if (plannerGatherReuse &&
            (region.gatherNeighbors < neighbors ||
             region.gatherNeighbors > resident->maximumNeighbors))
            throw NeighborhoodProofFailure(
                "planner-declared k-nearest gather reuse is outside the "
                "resident index envelope");
        CachedCudaKnnGather* cachedGather =
            plannerGatherReuse && resident->knnGather &&
                    neighbors <= resident->knnGather->neighbors &&
                    resident->knnGather->neighbors == region.gatherNeighbors
                ? resident->knnGather.get()
                : nullptr;
        if (requireGatherReuse && !cachedGather)
            throw NeighborhoodProofFailure(
                "required resident k-nearest gather reuse did not occur");
        if (cachedGather && ActiveKnnGatherReuseUsed)
            *ActiveKnnGatherReuseUsed = true;
        std::unique_ptr<pdg::Allocation> deviceStatus;
        std::uint8_t* deviceStatusData = nullptr;
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::outputPreparation);
            if (deviceOnlyHandoff)
            {
                if (!resident->device->has(nnDistance))
                    resident->device->materialize(nnDistance,
                                                  pdg::DimensionType::Double);
                else if (resident->device->columnInfo(nnDistance)
                             .physicalType != pdg::DimensionType::Double)
                    throw std::invalid_argument(
                        "resident NNDistance output column type mismatch");
            }
            else
                prepareOutputColumns(*resident, view, NnDistanceColumns);
            if (cachedGather)
                deviceStatusData =
                    static_cast<std::uint8_t*>(cachedGather->status->data());
            else
            {
                deviceStatus = resident->deviceMemory->allocate(
                    resident->pointCount, alignof(std::uint8_t));
                deviceStatusData =
                    static_cast<std::uint8_t*>(deviceStatus->data());
            }
        }
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::querySubmission);
            if (cachedGather)
                pdg::projectKnnDistanceValues(
                    *resident->index, cachedGather->neighbors, neighbors, mode,
                    static_cast<const double*>(
                        cachedGather->squaredDistances->data()),
                    resident->device->data<double>(nnDistance));
            else
                pdg::knnDistanceValues(
                    *resident->index, neighbors, mode,
                    resident->device->data<double>(nnDistance),
                    deviceStatusData);
        }

        std::vector<std::uint8_t> status;
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::statusAllocation);
            status.resize(resident->pointCount);
        }
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        if (!deviceOnlyHandoff)
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::resultTransferCall);
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(resident->host->rawData(nnDistance),
                                resident->device->rawData(nnDistance),
                                resident->pointCount * sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
            observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                            resident->region,
                            resident->pointCount * sizeof(double));
        }
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::statusTransferCall);
            PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatusData,
                                           resident->pointCount,
                                           cudaMemcpyDeviceToHost, stream));
            observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                            resident->region, resident->pointCount);
        }
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::explicitStreamWait);
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }
        // The cached rowset has reached its planner-declared consumer. The
        // projected NNDistance column and copied status are now canonical, so
        // release the adjacency before any selective repair allocations.
        if (cachedGather)
        {
            resident->knnGather.reset();
            cachedGather = nullptr;
            deviceStatusData = nullptr;
        }
        // The k-distance is tie-invariant, so only incomplete rows need
        // repair. The host path uses the compatibility index; the bounded
        // device path reads the same planner-owned resident coordinates.
        bool publishHostResult = !deviceOnlyHandoff;
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::statusScanAndRepair);
            std::vector<std::uint32_t> incompletePointIds;
            for (std::size_t point = 0U; point < status.size(); ++point)
                if ((status[point] & pdg::KnnSearchIncomplete) != 0U)
                    incompletePointIds.push_back(
                        static_cast<std::uint32_t>(point));
            const std::size_t incompleteRows = incompletePointIds.size();
            const bool requireDeviceRepair =
                std::getenv("PDG_REQUIRE_NND_DEVICE_REPAIR") != nullptr;
            const bool requireParallelRepair =
                std::getenv("PDG_REQUIRE_NND_PARALLEL_REPAIR") != nullptr;
            const bool parallelRepair =
                !std::getenv("PDG_DISABLE_NND_PARALLEL_REPAIR");
            constexpr std::size_t MaximumSelectiveDeviceRepairRows = 16U;
            const bool deviceRepairEligible =
                mode == pdg::KnnDistanceMode::Kth && neighbors <= 16U &&
                incompleteRows <= MaximumSelectiveDeviceRepairRows;
            const bool useDeviceRepair =
                !std::getenv("PDG_DISABLE_NND_DEVICE_REPAIR") &&
                deviceRepairEligible;
            if (requireDeviceRepair &&
                (incompleteRows == 0U || !useDeviceRepair))
                throw NeighborhoodProofFailure(
                    "required selective NND device repair was not used");
            if (requireParallelRepair &&
                (incompleteRows == 0U || !useDeviceRepair || !parallelRepair))
                throw NeighborhoodProofFailure(
                    "required parallel selective NND device repair was not "
                    "used");
            if (incompleteRows != 0U)
            {
                ResidentPhaseSeconds* phases =
                    activeResidentManagerDetailSeconds()
                        ? activeResidentPhaseSeconds()
                        : nullptr;
                std::optional<std::chrono::steady_clock::time_point>
                    repairStarted;
                if (phases)
                    repairStarted = std::chrono::steady_clock::now();
                if (useDeviceRepair)
                {
                    std::unique_ptr<pdg::Allocation> deviceIncomplete =
                        resident->deviceMemory->allocate(
                            incompleteRows * sizeof(std::uint32_t),
                            alignof(std::uint32_t));
                    PDG_CUDA_CHECK(cudaMemcpyAsync(
                        deviceIncomplete->data(), incompletePointIds.data(),
                        incompleteRows * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream));
                    observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                    resident->region,
                                    incompleteRows * sizeof(std::uint32_t));
                    pdg::detail::repairIncompleteKthDistanceValuesDevice(
                        *resident->index, neighbors,
                        resident->device->data<double>(nnDistance),
                        static_cast<const std::uint32_t*>(
                            deviceIncomplete->data()),
                        incompleteRows, parallelRepair);
                    if (!deviceOnlyHandoff)
                    {
                        double* hostValues =
                            resident->host->data<double>(nnDistance);
                        std::size_t repairedBytes = 0U;
                        for (std::uint32_t point : incompletePointIds)
                        {
                            PDG_CUDA_CHECK(cudaMemcpyAsync(
                                hostValues + point,
                                resident->device->data<double>(nnDistance) +
                                    point,
                                sizeof(double), cudaMemcpyDeviceToHost,
                                stream));
                            repairedBytes += sizeof(double);
                        }
                        observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                                        resident->region, repairedBytes);
                    }
                    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                    if (phases)
                    {
                        phases->deviceIncompleteRepairRows += incompleteRows;
                        phases->deviceRepairRows += incompleteRows;
                        phases->nnDistanceDeviceIncompleteRepairRows +=
                            incompleteRows;
                        phases->nnDistanceDeviceRepairRows += incompleteRows;
                        if (parallelRepair)
                            phases->nnDistanceParallelDeviceRepairRows +=
                                incompleteRows;
                        if (repairStarted)
                        {
                            accumulatePhaseSeconds(&phases->exactDeviceRepair,
                                                   *repairStarted);
                            accumulatePhaseSeconds(
                                &phases->nnDistanceExactDeviceRepair,
                                *repairStarted);
                        }
                    }
                }
                else
                {
                    if (deviceOnlyHandoff)
                    {
                        resident->host->materialize(nnDistance,
                                                    pdg::DimensionType::Double);
                        PDG_CUDA_CHECK(cudaMemcpyAsync(
                            resident->host->rawData(nnDistance),
                            resident->device->rawData(nnDistance),
                            resident->pointCount * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
                        observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                                        resident->region,
                                        resident->pointCount * sizeof(double));
                        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                        publishHostResult = true;
                        if (ActiveNnDistanceHostRestoreUsed)
                            *ActiveNnDistanceHostRestoreUsed = true;
                    }
                    if (phases)
                    {
                        phases->incompleteRepairRows += incompleteRows;
                        phases->repairedRows += incompleteRows;
                        phases->nnDistanceHostIncompleteRepairRows +=
                            incompleteRows;
                        phases->nnDistanceHostRepairRows += incompleteRows;
                    }
                    double* hostValues =
                        resident->host->data<double>(nnDistance);
                    KD3Index& index = view.build3dIndex();
                    for (PointId point = 0; point < view.size(); ++point)
                    {
                        if ((status[static_cast<std::size_t>(point)] &
                             pdg::KnnSearchIncomplete) == 0U)
                            continue;
                        PointIdList indices(neighbors);
                        std::vector<double> squaredDistances(neighbors);
                        index.knnSearch(point, neighbors, &indices,
                                        &squaredDistances);
                        double value = 0.0;
                        if (mode == pdg::KnnDistanceMode::Kth)
                            value = std::sqrt(squaredDistances[neighbors - 1U]);
                        else
                        {
                            for (std::size_t neighbor = 1U;
                                 neighbor < static_cast<std::size_t>(neighbors);
                                 ++neighbor)
                                value += std::sqrt(squaredDistances[neighbor]);
                            value /= static_cast<double>(neighbors - 1U);
                        }
                        hostValues[static_cast<std::size_t>(point)] = value;
                    }
                    // Keep the device column canonical for a retained
                    // region's bridges.
                    PDG_CUDA_CHECK(
                        cudaMemcpyAsync(resident->device->rawData(nnDistance),
                                        resident->host->rawData(nnDistance),
                                        resident->pointCount * sizeof(double),
                                        cudaMemcpyHostToDevice, stream));
                    observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                    resident->region,
                                    resident->pointCount * sizeof(double));
                    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                if (!useDeviceRepair && repairStarted)
                {
                    accumulatePhaseSeconds(&phases->exactHostRepair,
                                           *repairStarted);
                    accumulatePhaseSeconds(&phases->nnDistanceExactHostRepair,
                                           *repairStarted);
                }
            }
        }
        if (publishHostResult)
        {
            ResidentNnDistanceDetailTimer timer(
                &ResidentNnDistanceDetailSeconds::outputPublication);
            publishOutputColumns(*resident, view, NnDistanceColumns);
        }
        else if (ActiveNnDistanceDeviceOnlyHandoffUsed)
            *ActiveNnDistanceDeviceOnlyHandoffUsed = true;
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaStatisticalOutlier(PointView& view, std::uint32_t neighbors,
                               double multiplier, std::uint8_t classification,
                               CudaNeighborhoodRegion region,
                               CudaStatisticalOutlierResult& result,
                               bool requireCuda)
{
    result = {};
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 1U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw NeighborhoodProofFailure(
            "required resident statistical outlier index reuse did not "
            "occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        std::unique_ptr<pdg::Allocation> deviceDistances =
            resident->deviceMemory->allocate(
                resident->pointCount * sizeof(double), alignof(double));
        const bool plannerGatherReuse = region.gatherNeighbors != 0U;
        const bool requireGatherReuse =
            std::getenv("PDG_REQUIRE_KNN_GATHER_REUSE") != nullptr;
        if (requireGatherReuse && !plannerGatherReuse)
            throw NeighborhoodProofFailure(
                "required resident k-nearest gather reuse lacks planner "
                "provenance");
        const std::uint32_t gatherNeighbors = plannerGatherReuse
                                                  ? region.gatherNeighbors
                                                  : resident->maximumNeighbors;
        if (plannerGatherReuse &&
            (region.gatherNeighbors < neighbors ||
             region.gatherNeighbors > resident->maximumNeighbors))
            throw NeighborhoodProofFailure(
                "planner-declared k-nearest gather cache is outside the "
                "resident index envelope");
        CachedCudaKnnGather* cachedGather =
            plannerGatherReuse && resident->knnGather &&
                    neighbors <= resident->knnGather->neighbors &&
                    resident->knnGather->neighbors == region.gatherNeighbors
                ? resident->knnGather.get()
                : nullptr;
        if (!cachedGather && plannerGatherReuse && region.id != 0U &&
            !region.last && gatherNeighbors >= neighbors &&
            gatherNeighbors <= resident->maximumNeighbors &&
            resident->pointCount >= gatherNeighbors)
        {
            auto gathered = std::make_unique<CachedCudaKnnGather>();
            gathered->neighbors = gatherNeighbors;
            const std::size_t entries =
                checkedProduct(resident->pointCount, gathered->neighbors,
                               "resident k-nearest gather row count "
                               "overflows size_t");
            gathered->pointIds = resident->deviceMemory->allocate(
                checkedProduct(entries, sizeof(std::uint32_t),
                               "resident k-nearest gather id bytes overflow"),
                alignof(std::uint32_t));
            gathered->squaredDistances = resident->deviceMemory->allocate(
                checkedProduct(
                    entries, sizeof(double),
                    "resident k-nearest gather distance bytes overflow"),
                alignof(double));
            gathered->status = resident->deviceMemory->allocate(
                resident->pointCount, alignof(std::uint8_t));
            pdg::knnGather(
                *resident->index, gathered->neighbors,
                static_cast<std::uint32_t*>(gathered->pointIds->data()),
                static_cast<double*>(gathered->squaredDistances->data()),
                static_cast<std::uint8_t*>(gathered->status->data()));
            resident->knnGather = std::move(gathered);
            cachedGather = resident->knnGather.get();
        }
        if (requireGatherReuse && !cachedGather)
            throw NeighborhoodProofFailure(
                "required resident k-nearest gather cache was not created");

        std::unique_ptr<pdg::Allocation> deviceStatus;
        std::uint8_t* deviceStatusData = nullptr;
        if (cachedGather)
        {
            pdg::projectKnnMeanDistances(
                *resident->index, cachedGather->neighbors, neighbors,
                static_cast<const double*>(
                    cachedGather->squaredDistances->data()),
                static_cast<double*>(deviceDistances->data()));
            deviceStatusData =
                static_cast<std::uint8_t*>(cachedGather->status->data());
        }
        else
        {
            deviceStatus = resident->deviceMemory->allocate(
                resident->pointCount, alignof(std::uint8_t));
            deviceStatusData = static_cast<std::uint8_t*>(deviceStatus->data());
            pdg::knnMeanDistances(*resident->index, neighbors,
                                  static_cast<double*>(deviceDistances->data()),
                                  deviceStatusData);
        }

        std::vector<double> distances(resident->pointCount);
        std::vector<std::uint8_t> status(resident->pointCount);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(distances.data(),
                                       deviceDistances->data(),
                                       resident->pointCount * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatusData,
                                       resident->pointCount,
                                       cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount * (sizeof(double) + 1U));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::uint32_t> incompletePointIds;
        for (std::size_t point = 0U; point < status.size(); ++point)
            if ((status[point] & pdg::KnnSearchIncomplete) != 0U)
                incompletePointIds.push_back(static_cast<std::uint32_t>(point));
        const std::size_t incompleteRows = incompletePointIds.size();
        const bool requireDeviceRepair =
            std::getenv("PDG_REQUIRE_OUTLIER_DEVICE_REPAIR") != nullptr;
        const bool requireParallelRepair =
            std::getenv("PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR") != nullptr;
        const bool parallelRepair =
            std::getenv("PDG_DISABLE_OUTLIER_PARALLEL_REPAIR") == nullptr;
        constexpr std::size_t MaximumSelectiveDeviceRepairRows = 16U;
        const bool deviceRepairEligible =
            neighbors <= 16U &&
            incompleteRows <= MaximumSelectiveDeviceRepairRows;
        const bool useDeviceRepair =
            !std::getenv("PDG_DISABLE_OUTLIER_DEVICE_REPAIR") &&
            deviceRepairEligible;
        if (requireDeviceRepair && (incompleteRows == 0U || !useDeviceRepair))
            throw NeighborhoodProofFailure(
                "required selective statistical-outlier device repair was "
                "not used");
        if (requireParallelRepair &&
            (incompleteRows == 0U || !useDeviceRepair || !parallelRepair))
            throw NeighborhoodProofFailure(
                "required parallel statistical-outlier device repair was "
                "not used");
        if (incompleteRows != 0U)
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto repairStarted = std::chrono::steady_clock::now();
            if (useDeviceRepair)
            {
                std::unique_ptr<pdg::Allocation> deviceIncomplete =
                    resident->deviceMemory->allocate(
                        checkedProduct(
                            incompleteRows, sizeof(std::uint32_t),
                            "statistical-outlier repair id bytes overflow"),
                        alignof(std::uint32_t));
                PDG_CUDA_CHECK(cudaMemcpyAsync(
                    deviceIncomplete->data(), incompletePointIds.data(),
                    incompleteRows * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, stream));
                observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                resident->region,
                                incompleteRows * sizeof(std::uint32_t));
                pdg::detail::repairIncompleteMeanDistancesDevice(
                    *resident->index, neighbors,
                    static_cast<double*>(deviceDistances->data()),
                    static_cast<const std::uint32_t*>(deviceIncomplete->data()),
                    incompleteRows, parallelRepair);
                std::size_t repairedBytes = 0U;
                for (std::uint32_t point : incompletePointIds)
                {
                    PDG_CUDA_CHECK(cudaMemcpyAsync(
                        distances.data() + point,
                        static_cast<const double*>(deviceDistances->data()) +
                            point,
                        sizeof(double), cudaMemcpyDeviceToHost, stream));
                    repairedBytes += sizeof(double);
                }
                observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                                resident->region, repairedBytes);
                PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                result.repairedRows += incompleteRows;
                result.deviceRepairedRows += incompleteRows;
                if (phases)
                {
                    phases->deviceIncompleteRepairRows += incompleteRows;
                    phases->deviceRepairRows += incompleteRows;
                    phases->outlierDeviceIncompleteRepairRows += incompleteRows;
                    phases->outlierDeviceRepairRows += incompleteRows;
                    if (parallelRepair)
                        phases->outlierParallelDeviceRepairRows +=
                            incompleteRows;
                    accumulatePhaseSeconds(&phases->exactDeviceRepair,
                                           repairStarted);
                    accumulatePhaseSeconds(&phases->outlierExactDeviceRepair,
                                           repairStarted);
                }
            }
            else
            {
                KD3Index& index = view.build3dIndex();
                const point_count_t count =
                    static_cast<point_count_t>(neighbors);
                for (std::uint32_t point : incompletePointIds)
                {
                    ++result.repairedRows;
                    ++result.hostRepairedRows;
                    PointIdList pointIds(count);
                    std::vector<double> squaredDistances(count);
                    index.knnSearch(point, count, &pointIds, &squaredDistances);
                    double& value = distances[point];
                    value = 0.0;
                    for (std::size_t neighbor = 1U; neighbor < count;
                         ++neighbor)
                    {
                        const double delta =
                            std::sqrt(squaredDistances[neighbor]) - value;
                        value += delta / static_cast<double>(neighbor);
                    }
                }
                if (phases)
                {
                    phases->incompleteRepairRows += incompleteRows;
                    phases->repairedRows += incompleteRows;
                    phases->outlierHostIncompleteRepairRows += incompleteRows;
                    phases->outlierHostRepairRows += incompleteRows;
                    accumulatePhaseSeconds(&phases->exactHostRepair,
                                           repairStarted);
                    accumulatePhaseSeconds(&phases->outlierExactHostRepair,
                                           repairStarted);
                }
            }
        }

        std::size_t samples = 0U;
        double mean = 0.0;
        double secondMoment = 0.0;
        for (double distance : distances)
        {
            const std::size_t previous = samples;
            ++samples;
            const double delta = distance - mean;
            const double normalized = delta / static_cast<double>(samples);
            mean += normalized;
            secondMoment += delta * normalized * static_cast<double>(previous);
        }
        const double variance =
            secondMoment / (static_cast<double>(samples) - 1.0);
        const double threshold = mean + multiplier * std::sqrt(variance);
        for (double distance : distances)
        {
            if (distance < threshold)
                ++result.inliers;
            else
                ++result.outliers;
        }

        if (result.inliers != 0U && result.outliers != 0U)
        {
            for (PointId point = 0U; point < view.size(); ++point)
                if (!(distances[static_cast<std::size_t>(point)] < threshold))
                    view.setField(Dimension::Id::Classification, point,
                                  classification);
            // A later resident consumer may read Classification. Keep its
            // device column canonical after the exact host global finale.
            if (region.id != 0U && !region.last)
                refreshOutputColumnsFromView(*resident, view,
                                             ClassificationColumns);
        }
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaHagNnColumn(PointView& view, const pdg::HagNnProgram& program,
                        CudaNeighborhoodRegion region, bool requireCuda)
{
    const std::uint32_t neighbors = program.count;
    region.dimensions = 2U;
    if (region.maximumNeighbors == 0U)
        region.maximumNeighbors = neighbors;
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 1U || neighbors > 64U ||
        region.maximumNeighbors < neighbors || region.maximumNeighbors > 64U ||
        view.empty())
    {
        invalidateRegion();
        return false;
    }

    const bool debugPhases =
        std::getenv("PDG_DEBUG_HAG_NN_PHASES") != nullptr;
    using DebugClock = std::chrono::steady_clock;
    auto marked = DebugClock::now();
    const auto mark = [&](const char* label)
    {
        if (!debugPhases)
            return;
        const auto now = DebugClock::now();
        std::fprintf(stderr, "PDG_HAG_NN %-20s %.6f\n", label,
                     std::chrono::duration<double>(now - marked).count());
        marked = now;
    };

    // The shared CUDA index has an explicit finite-coordinate envelope while
    // the pinned host filter does not. Decline before allocation or mutation;
    // the wrapper can then run the original host algorithm on the untouched
    // view. Explicit require-CUDA calls still fail in the wrapper because this
    // helper reports that CUDA was not used.
    bool finiteXy = true;
    bool finiteZ = true;
    for (PointId point = 0U; point < view.size(); ++point)
    {
        finiteXy =
            finiteXy &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::X, point)) &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::Y, point));
        finiteZ =
            finiteZ &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::Z, point));
    }
    mark("coordinate_scan");
    const bool nonfiniteZFallback = neighbors > 1U && !finiteZ;
    if (std::getenv("PDG_REQUIRE_HAG_NN_NONFINITE_Z_FALLBACK") &&
        !nonfiniteZFallback)
    {
        invalidateRegion();
        throw NeighborhoodProofFailure(
            "required HAG nonfinite-Z fallback did not occur");
    }
    if (!finiteXy || nonfiniteZFallback)
    {
        invalidateRegion();
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(view.size());
    std::vector<std::uint8_t> sourceMask(count, 0U);
    std::vector<std::uint8_t> groundMask(count, 0U);
    std::uint32_t groundCount = 0U;
    double minimumX = (std::numeric_limits<double>::max)();
    double minimumY = (std::numeric_limits<double>::max)();
    double maximumX = (std::numeric_limits<double>::lowest)();
    double maximumY = (std::numeric_limits<double>::lowest)();
    for (PointId point = 0U; point < view.size(); ++point)
    {
        const bool ground =
            view.getFieldAs<std::uint8_t>(Dimension::Id::Classification,
                                          point) == program.groundClass;
        groundMask[static_cast<std::size_t>(point)] = ground ? 1U : 0U;
        sourceMask[static_cast<std::size_t>(point)] = ground ? 0U : 1U;
        groundCount += static_cast<std::uint32_t>(ground);
        if (ground)
        {
            const double x = view.getFieldAs<double>(Dimension::Id::X, point);
            const double y = view.getFieldAs<double>(Dimension::Id::Y, point);
            minimumX = (std::min)(minimumX, x);
            minimumY = (std::min)(minimumY, y);
            maximumX = (std::max)(maximumX, x);
            maximumY = (std::max)(maximumY, y);
        }
    }
    mark("ground_partition");
    const bool insufficientGroundFallback =
        groundCount != 0U && groundCount < neighbors;
    if (std::getenv("PDG_REQUIRE_HAG_NN_INSUFFICIENT_GROUND_FALLBACK") &&
        !insufficientGroundFallback)
    {
        invalidateRegion();
        throw NeighborhoodProofFailure(
            "required HAG insufficient-ground fallback did not occur");
    }
    if (groundCount == 0U)
    {
        invalidateRegion();
        return false;
    }
    // The pinned KD2Index call leaves the requested row width observable when
    // fewer references exist than count (count two with one ground point
    // produces its historical NaN result). The compact masked query reports
    // only real references, so preserve that edge byte-for-byte on host.
    if (insufficientGroundFallback)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident HAG neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident HAG neighborhood rebuild had no "
                        "CUDA device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident HAG neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }
        mark("resident_index_setup");

        prepareOutputColumns(*resident, view, HagNnColumns);
        std::unique_ptr<pdg::Allocation> deviceSource =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceGround =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceIds =
            resident->deviceMemory->allocate(count * neighbors *
                                                 sizeof(std::uint32_t),
                                             alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> deviceDistances =
            resident->deviceMemory->allocate(count * neighbors * sizeof(double),
                                             alignof(double));
        std::unique_ptr<pdg::Allocation> deviceStatus =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSource->data(), sourceMask.data(),
                                       count, cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceGround->data(), groundMask.data(),
                                       count, cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        2U * count);
        pdg::knnGatherMasked(
            *resident->index, neighbors, groundCount,
            static_cast<const std::uint8_t*>(deviceSource->data()),
            static_cast<const std::uint8_t*>(deviceGround->data()),
            static_cast<std::uint32_t*>(deviceIds->data()),
            static_cast<double*>(deviceDistances->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()));
        pdg::projectHagNnColumn(
            *resident->device,
            static_cast<const std::uint8_t*>(deviceGround->data()),
            static_cast<const std::uint32_t*>(deviceIds->data()),
            static_cast<const double*>(deviceDistances->data()),
            static_cast<const std::uint8_t*>(deviceStatus->data()), neighbors,
            groundCount, std::pow(program.maximumDistance, 2.0),
            program.allowExtrapolation, minimumX, minimumY, maximumX, maximumY);

        std::vector<std::uint8_t> status(count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatus->data(),
                                       count, cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            resident->host->rawData(
                pdg::DimensionId(pdg::StandardDimension::HeightAboveGround)),
            resident->device->rawData(
                pdg::DimensionId(pdg::StandardDimension::HeightAboveGround)),
            count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        count * (sizeof(double) + 1U));
        mark("query_submit");
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);
        mark("query_wait");

        bool exact = true;
        bool ambiguous = false;
        std::size_t nonExactRows = 0U;
        std::size_t tieRows = 0U;
        std::size_t incompleteRows = 0U;
        for (std::size_t point = 0U; point < count; ++point)
            if (sourceMask[point] != 0U && status[point] != pdg::KnnExact)
            {
                exact = false;
                ++nonExactRows;
                tieRows += static_cast<std::size_t>(
                    (status[point] & pdg::KnnDistanceTie) != 0U);
                incompleteRows += static_cast<std::size_t>(
                    (status[point] & pdg::KnnSearchIncomplete) != 0U);
                ambiguous =
                    ambiguous || (status[point] & pdg::KnnDistanceTie) != 0U;
            }
        if (debugPhases)
            std::fprintf(stderr,
                         "PDG_HAG_NN rows points=%zu ground=%u non_exact=%zu "
                         "tie=%zu incomplete=%zu\n",
                         count, groundCount, nonExactRows, tieRows,
                         incompleteRows);
        mark("status_scan");
        if (std::getenv("PDG_REQUIRE_HAG_NN_TIE_FALLBACK") && !ambiguous)
            throw NeighborhoodProofFailure(
                "required HAG nearest-neighbor tie fallback did not occur");
        if (std::getenv("PDG_REQUIRE_HAG_NN_HOST_FALLBACK") && exact)
            throw NeighborhoodProofFailure(
                "required HAG CUDA host fallback did not occur");
        const bool requireSelectiveRepair =
            std::getenv("PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR") != nullptr;
        if (requireSelectiveRepair && exact)
            throw NeighborhoodProofFailure(
                "required HAG selective exact repair did not occur");
        if (!exact)
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto repairStarted = std::chrono::steady_clock::now();
            if (phases)
            {
                phases->ambiguousRepairRows += tieRows;
                phases->incompleteRepairRows += incompleteRows;
                phases->repairedRows += nonExactRows;
            }
            repairHagNnRowsExact(
                *resident, view, program, groundMask, sourceMask, status,
                resident->host->data<double>(pdg::DimensionId(
                    pdg::StandardDimension::HeightAboveGround)));
            mark("exact_host_repair");

            // A retained planner region may hand HeightAboveGround to a
            // following resident stage. Restore the repaired host column to
            // the device before that bridge; terminal regions publish only.
            if (region.id != 0U && !region.last)
            {
                const pdg::DimensionId heightAboveGround(
                    pdg::StandardDimension::HeightAboveGround);
                PDG_CUDA_CHECK(cudaMemcpyAsync(
                    resident->device->rawData(heightAboveGround),
                    resident->host->rawData(heightAboveGround),
                    count * sizeof(double), cudaMemcpyHostToDevice, stream));
                observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                resident->region, count * sizeof(double));
                PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            accumulatePhaseSeconds(phases ? &phases->exactHostRepair : nullptr,
                                   repairStarted);
        }

        publishOutputColumns(*resident, view, HagNnColumns);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaHagDelaunayColumn(PointView& view,
                              const pdg::HagDelaunayProgram& program,
                              CudaNeighborhoodRegion region, bool requireCuda)
{
    constexpr std::uint32_t Neighbors = 3U;
    region.dimensions = 2U;
    if (region.maximumNeighbors == 0U)
        region.maximumNeighbors = Neighbors;
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (program.count != Neighbors || region.maximumNeighbors < Neighbors ||
        region.maximumNeighbors > 64U || view.empty())
    {
        invalidateRegion();
        return false;
    }

    bool finiteCoordinates = true;
    for (PointId point = 0U; point < view.size(); ++point)
        finiteCoordinates =
            finiteCoordinates &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::X, point)) &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::Y, point)) &&
            std::isfinite(view.getFieldAs<double>(Dimension::Id::Z, point));
    if (std::getenv("PDG_REQUIRE_HAG_DELAUNAY_NONFINITE_FALLBACK") &&
        finiteCoordinates)
    {
        invalidateRegion();
        throw NeighborhoodProofFailure(
            "required HAG Delaunay nonfinite fallback did not occur");
    }
    if (!finiteCoordinates)
    {
        invalidateRegion();
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(view.size());
    std::vector<std::uint8_t> sourceMask(count, 0U);
    std::vector<std::uint8_t> groundMask(count, 0U);
    std::uint32_t groundCount = 0U;
    double minimumX = (std::numeric_limits<double>::max)();
    double minimumY = (std::numeric_limits<double>::max)();
    double maximumX = (std::numeric_limits<double>::lowest)();
    double maximumY = (std::numeric_limits<double>::lowest)();
    for (PointId point = 0U; point < view.size(); ++point)
    {
        const bool ground =
            view.getFieldAs<std::uint8_t>(Dimension::Id::Classification,
                                          point) == program.groundClass;
        groundMask[static_cast<std::size_t>(point)] = ground ? 1U : 0U;
        sourceMask[static_cast<std::size_t>(point)] = ground ? 0U : 1U;
        groundCount += static_cast<std::uint32_t>(ground);
        if (ground)
        {
            const double x = view.getFieldAs<double>(Dimension::Id::X, point);
            const double y = view.getFieldAs<double>(Dimension::Id::Y, point);
            minimumX = (std::min)(minimumX, x);
            minimumY = (std::min)(minimumY, y);
            maximumX = (std::max)(maximumX, x);
            maximumY = (std::max)(maximumY, y);
        }
    }
    const bool insufficientGround = groundCount < Neighbors;
    if (std::getenv("PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK") &&
        !insufficientGround)
    {
        invalidateRegion();
        throw NeighborhoodProofFailure(
            "required HAG Delaunay insufficient-ground fallback did not "
            "occur");
    }
    if (insufficientGround)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, Neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident HAG Delaunay index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident HAG Delaunay rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, Neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident HAG Delaunay rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        prepareOutputColumns(*resident, view, HagNnColumns);
        std::unique_ptr<pdg::Allocation> deviceSource =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceGround =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceIds =
            resident->deviceMemory->allocate(count * Neighbors *
                                                 sizeof(std::uint32_t),
                                             alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> deviceDistances =
            resident->deviceMemory->allocate(count * Neighbors * sizeof(double),
                                             alignof(double));
        std::unique_ptr<pdg::Allocation> deviceStatus =
            resident->deviceMemory->allocate(count, alignof(std::uint8_t));
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSource->data(), sourceMask.data(),
                                       count, cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceGround->data(), groundMask.data(),
                                       count, cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        2U * count);
        pdg::knnGatherMasked(
            *resident->index, Neighbors, groundCount,
            static_cast<const std::uint8_t*>(deviceSource->data()),
            static_cast<const std::uint8_t*>(deviceGround->data()),
            static_cast<std::uint32_t*>(deviceIds->data()),
            static_cast<double*>(deviceDistances->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()));
        pdg::projectHagDelaunayColumn(
            *resident->device,
            static_cast<const std::uint8_t*>(deviceGround->data()),
            static_cast<const std::uint32_t*>(deviceIds->data()),
            static_cast<const std::uint8_t*>(deviceStatus->data()), groundCount,
            program.allowExtrapolation, minimumX, minimumY, maximumX, maximumY);

        std::vector<std::uint8_t> status(count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatus->data(),
                                       count, cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            resident->host->rawData(
                pdg::DimensionId(pdg::StandardDimension::HeightAboveGround)),
            resident->device->rawData(
                pdg::DimensionId(pdg::StandardDimension::HeightAboveGround)),
            count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        count * (sizeof(double) + 1U));
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);

        bool exact = true;
        bool ambiguous = false;
        for (std::size_t point = 0U; point < count; ++point)
            if (sourceMask[point] != 0U && status[point] != pdg::KnnExact)
            {
                exact = false;
                ambiguous =
                    ambiguous || (status[point] & pdg::KnnDistanceTie) != 0U;
            }
        if (std::getenv("PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK") && !ambiguous)
            throw NeighborhoodProofFailure(
                "required HAG Delaunay tie fallback did not occur");
        if (std::getenv("PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK") && exact)
            throw NeighborhoodProofFailure(
                "required HAG Delaunay host fallback did not occur");
        if (!exact)
        {
            if (region.id == 0U || region.last)
                invalidateRegion();
            return false;
        }

        publishOutputColumns(*resident, view, HagNnColumns);
        if (ActiveHagDelaunayCudaUsed)
            *ActiveHagDelaunayCudaUsed = true;
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool retainCudaHagHostColumn(PointView& view, std::uint32_t neighbors,
                             CudaNeighborhoodRegion region, bool requireCuda)
{
    region.dimensions = 2U;
    if (region.maximumNeighbors == 0U)
        region.maximumNeighbors = neighbors;
    if (region.id == 0U || region.last)
        return false;
    if (view.empty())
        return true;
    try
    {
        bool finiteCoordinates = true;
        for (PointId point = 0U; point < view.size(); ++point)
            finiteCoordinates =
                finiteCoordinates &&
                std::isfinite(
                    view.getFieldAs<double>(Dimension::Id::X, point)) &&
                std::isfinite(view.getFieldAs<double>(Dimension::Id::Y, point));
        std::shared_ptr<ResidentNeighborhood> resident;
        if (region.id != 0U)
        {
            const std::shared_ptr<void> product =
                ResidentPointViewAccess::get(view);
            if (product)
            {
                resident =
                    std::static_pointer_cast<ResidentNeighborhood>(product);
                const bool compatibleIndex =
                    finiteCoordinates &&
                    matches(*resident, view, region, neighbors);
                const bool compatibleBridgeProduct =
                    !finiteCoordinates && !resident->indexRequired &&
                    resident->region == region.id &&
                    resident->pointCount ==
                        static_cast<std::size_t>(view.size());
                if (!compatibleIndex && !compatibleBridgeProduct)
                    resident.reset();
            }
        }
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
                return false;
            CudaNeighborhoodRegion retainedRegion = region;
            retainedRegion.indexRequired = finiteCoordinates;
            resident =
                makeResidentNeighborhood(view, retainedRegion, neighbors);
            if (!resident)
                return false;
            if (region.id != 0U)
                ResidentPointViewAccess::set(view, resident);
        }

        const OutputColumn& column = HagNnColumns[0];
        if (!resident->device->has(column.device))
            prepareOutputColumns(*resident, view, HagNnColumns);
        else
        {
            std::byte* host =
                static_cast<std::byte*>(resident->host->rawData(column.device));
            const std::size_t stride = pdg::dimensionTypeSize(column.physical);
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto started = std::chrono::steady_clock::now();
            gatherColumnFromView(view, column.pointView, column.pointViewType,
                                 rowBoundaryUsable(view), host, stride);
            accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr,
                                   started);
            const cudaStream_t stream = static_cast<cudaStream_t>(
                resident->device->nativeStreamHandle());
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                resident->device->rawData(column.device), host,
                resident->pointCount * stride, cudaMemcpyHostToDevice, stream));
            observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                            resident->region, resident->pointCount * stride);
        }
        return true;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda)
            throw;
        return false;
    }
}

bool tryCudaNeighborClassifierColumn(PointView& view, std::uint32_t neighbors,
                                     CudaNeighborhoodRegion region,
                                     bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 1U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        // The vote reads every neighbor's original value and applies the
        // changed results afterwards, so the results stage in their own
        // buffers until the repair completes.
        prepareOutputColumns(*resident, view, ClassificationColumns);
        const pdg::DimensionId classification(
            pdg::StandardDimension::Classification);
        std::unique_ptr<pdg::Allocation> deviceStatus =
            resident->deviceMemory->allocate(resident->pointCount,
                                             alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceResults =
            resident->deviceMemory->allocate(resident->pointCount,
                                             alignof(std::uint8_t));
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            pdg::knnNeighborVotes(
                *resident->index, neighbors,
                static_cast<const std::uint8_t*>(
                    resident->device->rawData(classification)),
                static_cast<std::uint8_t*>(deviceResults->data()),
                static_cast<std::uint8_t*>(deviceStatus->data()));
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }
        std::vector<std::uint8_t> status(resident->pointCount);
        std::vector<std::uint8_t> results(resident->pointCount);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatus->data(),
                                       resident->pointCount,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(results.data(), deviceResults->data(),
                                       resident->pointCount,
                                       cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount * 2U);
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }

        std::size_t ambiguousRows = 0U;
        std::size_t incompleteRows = 0U;
        std::size_t repairRows = 0U;
        for (std::uint8_t value : status)
        {
            ambiguousRows +=
                static_cast<std::size_t>((value & pdg::KnnDistanceTie) != 0U);
            incompleteRows += static_cast<std::size_t>(
                (value & pdg::KnnSearchIncomplete) != 0U);
            repairRows += static_cast<std::size_t>(
                (value & (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) !=
                0U);
        }
        if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR") &&
            repairRows == 0U)
            throw NeighborhoodProofFailure(
                "required resident neighborhood tie repair did not occur");
        if (repairRows != 0U)
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto repairStarted = std::chrono::steady_clock::now();
            const KD3Index& index = view.build3dIndex();
            for (PointId point = 0; point < view.size(); ++point)
            {
                if ((status[static_cast<std::size_t>(point)] &
                     (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) == 0U)
                    continue;
                results[static_cast<std::size_t>(point)] =
                    computeNeighborVoteExact(view, index, point, neighbors);
            }
            if (phases)
            {
                phases->ambiguousRepairRows += ambiguousRows;
                phases->incompleteRepairRows += incompleteRows;
                phases->repairedRows += repairRows;
                phases->neighborClassifierAmbiguousRepairRows += ambiguousRows;
                phases->neighborClassifierIncompleteRepairRows +=
                    incompleteRows;
                phases->neighborClassifierRepairRows += repairRows;
                ++phases->neighborClassifierKd3Uses;
                accumulatePhaseSeconds(&phases->exactHostRepair, repairStarted);
                accumulatePhaseSeconds(
                    &phases->neighborClassifierExactHostRepair, repairStarted);
            }
        }
        // Publish the final values and keep the device column canonical.
        std::memcpy(resident->host->rawData(classification), results.data(),
                    resident->pointCount);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            resident->device->rawData(classification),
            resident->host->rawData(classification), resident->pointCount,
            cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        resident->pointCount);
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        publishOutputColumns(*resident, view, ClassificationColumns);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaOptimalNeighborhoodColumns(PointView& view, std::uint32_t minimumK,
                                       std::uint32_t maximumK,
                                       CudaNeighborhoodRegion region,
                                       bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (minimumK < 1U || minimumK > maximumK || maximumK > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < maximumK)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, maximumK))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, maximumK);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        prepareOutputColumns(*resident, view, OptimalColumns);
        const pdg::DimensionId optimalKnn(pdg::StandardDimension::OptimalKNN);
        const pdg::DimensionId optimalRadius(
            pdg::StandardDimension::OptimalRadius);
        std::vector<std::uint8_t> status(resident->pointCount);
        auto* hostK =
            static_cast<std::uint64_t*>(resident->host->rawData(optimalKnn));
        auto* hostRadius =
            static_cast<double*>(resident->host->rawData(optimalRadius));
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            pdg::knnOptimalValues(*resident->index, minimumK, maximumK, hostK,
                                  hostRadius, status.data());
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount *
                            (static_cast<std::size_t>(maximumK) *
                                 (sizeof(std::uint32_t) + sizeof(double)) +
                             1U));

        std::size_t repairRows = 0;
        for (std::uint8_t value : status)
            repairRows += static_cast<std::size_t>(
                (value & (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) !=
                0U);
        if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR") &&
            repairRows == 0U)
            throw NeighborhoodProofFailure(
                "required resident neighborhood tie repair did not occur");
        if (repairRows != 0U)
        {
            const KD3Index& index = view.build3dIndex();
            for (PointId point = 0; point < view.size(); ++point)
            {
                if ((status[static_cast<std::size_t>(point)] &
                     (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) == 0U)
                    continue;
                computeOptimalExact(
                    view, index, point, minimumK, maximumK,
                    hostK[static_cast<std::size_t>(point)],
                    hostRadius[static_cast<std::size_t>(point)]);
            }
        }
        // Keep the device columns canonical for a retained region's bridges.
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(resident->device->rawData(optimalKnn),
                            resident->host->rawData(optimalKnn),
                            resident->pointCount * sizeof(std::uint64_t),
                            cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident->device->rawData(optimalRadius),
                                       resident->host->rawData(optimalRadius),
                                       resident->pointCount * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        resident->pointCount *
                            (sizeof(std::uint64_t) + sizeof(double)));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        publishOutputColumns(*resident, view, OptimalColumns);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaEstimateRankColumn(PointView& view, std::uint32_t neighbors,
                               CudaNeighborhoodRegion region, double threshold,
                               bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 3U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        prepareOutputColumns(*resident, view, RankColumns);
        const pdg::DimensionId rank(pdg::StandardDimension::Rank);
        // knnRankValues lands host-visible ranks and statuses: the device
        // side computes the exact float-demeaned covariances and the host
        // runs the oracle's own JacobiSVD.
        std::vector<std::uint8_t> status(resident->pointCount);
        auto* hostRanks =
            static_cast<std::uint8_t*>(resident->host->rawData(rank));
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            pdg::knnRankValues(*resident->index, neighbors, threshold,
                               hostRanks, status.data());
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount *
                            (sizeof(pdg::Covariance3d) + 1U));

        // Tie or incomplete rows recompute through the compatibility index
        // with upstream's own computeRank, which is the oracle by identity.
        std::size_t repairRows = 0;
        for (std::uint8_t value : status)
            repairRows += static_cast<std::size_t>(
                (value & (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) !=
                0U);
        if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR") &&
            repairRows == 0U)
            throw NeighborhoodProofFailure(
                "required resident neighborhood tie repair did not occur");
        if (repairRows != 0U)
        {
            KD3Index& index = view.build3dIndex();
            for (PointId point = 0; point < view.size(); ++point)
            {
                if ((status[static_cast<std::size_t>(point)] &
                     (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) == 0U)
                    continue;
                const PointIdList ids = index.neighbors(
                    point, static_cast<point_count_t>(neighbors));
                hostRanks[static_cast<std::size_t>(point)] =
                    computeRankExact(view, ids, threshold);
            }
        }
        // Keep the device column canonical for a retained region's bridges.
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            resident->device->rawData(rank), resident->host->rawData(rank),
            resident->pointCount, cudaMemcpyHostToDevice, stream));
        observeTransfer(pdg::ExecutionEventKind::HostToDevice, resident->region,
                        resident->pointCount);
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        publishOutputColumns(*resident, view, RankColumns);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaLofColumns(PointView& view, std::uint32_t neighbors,
                       CudaNeighborhoodRegion region, bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0 && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (neighbors < 2U || neighbors > 64U)
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        if (region.id == 0U)
            return false;
        closeRegion();
        return true;
    }
    if (static_cast<std::size_t>(view.size()) < neighbors)
    {
        invalidateRegion();
        return false;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            if (!matches(*resident, view, region, neighbors))
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident neighborhood index reuse did not occur");
    const bool bridgeRebuild = requireBridgeRebuild(region, resident);

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild had no CUDA "
                        "device");
                return false;
            }
            resident = makeResidentNeighborhood(view, region, neighbors);
            if (!resident)
            {
                invalidateRegion();
                if (bridgeRebuild)
                    throw NeighborhoodProofFailure(
                        "required resident neighborhood rebuild was not "
                        "created");
                return false;
            }
            if (region.id != 0)
                ResidentPointViewAccess::set(view, resident);
            if (bridgeRebuild)
                RequiredBridgeRebuildRegion.reset();
        }

        ResidentManagerDetailTimer queryTimer(
            &ResidentManagerDetailSeconds::neighborhoodQueryProjection);
        prepareOutputColumns(*resident, view, LofColumns);
        const pdg::DimensionId nnDistance(pdg::StandardDimension::NNDistance);
        const pdg::DimensionId reachability(
            pdg::StandardDimension::LocalReachabilityDistance);
        const pdg::DimensionId outlierFactor(
            pdg::StandardDimension::LocalOutlierFactor);
        std::unique_ptr<pdg::Allocation> deviceStatus =
            resident->deviceMemory->allocate(resident->pointCount,
                                             alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceNeighborStatus =
            resident->deviceMemory->allocate(resident->pointCount,
                                             alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceClosureStatus =
            resident->deviceMemory->allocate(resident->pointCount,
                                             alignof(std::uint8_t));
        pdg::knnLofValues(
            *resident->index, neighbors,
            resident->device->data<double>(nnDistance),
            resident->device->data<double>(reachability),
            resident->device->data<double>(outlierFactor),
            static_cast<std::uint8_t*>(deviceStatus->data()),
            static_cast<std::uint8_t*>(deviceNeighborStatus->data()),
            static_cast<std::uint8_t*>(deviceClosureStatus->data()));

        std::vector<std::uint8_t> status(resident->pointCount);
        std::vector<std::uint8_t> neighborStatus(resident->pointCount);
        std::vector<std::uint8_t> closureStatus(resident->pointCount);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        for (const pdg::DimensionId column :
             {nnDistance, reachability, outlierFactor})
        {
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(resident->host->rawData(column),
                                resident->device->rawData(column),
                                resident->pointCount * sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
            observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                            resident->region,
                            resident->pointCount * sizeof(double));
        }
        PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatus->data(),
                                       resident->pointCount,
                                       cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            neighborStatus.data(), deviceNeighborStatus->data(),
            resident->pointCount, cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            closureStatus.data(), deviceClosureStatus->data(),
            resident->pointCount, cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount);
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto waitStarted = std::chrono::steady_clock::now();
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                                   waitStarted);
        }

        // Host closure repair. A tie leaves the k-distance exact but makes
        // the retained adjacency ambiguous: the own row rebuilds density
        // and factor, and any row reaching it rebuilds only its factor. An
        // incomplete row's k-distance itself is unknown, so wrongness
        // reaches one hop further: the own row rebuilds k-distance,
        // density, and factor; a row reaching it rebuilds density and
        // factor (its reachability read the unknown k-distance); and a row
        // reaching one of those rebuilds only its factor. All rebuilt
        // values come from the compatibility index in upstream's exact
        // operation order, k-distances first so densities read repaired
        // values and densities before factors.
        std::size_t ambiguousRows = 0;
        const auto repairsKDistance = [&](std::size_t point)
        { return (status[point] & pdg::KnnSearchIncomplete) != 0U; };
        const auto repairsDensity = [&](std::size_t point)
        {
            return (status[point] &
                    (pdg::KnnDistanceTie | pdg::KnnSearchIncomplete)) != 0U ||
                   (neighborStatus[point] & pdg::KnnSearchIncomplete) != 0U;
        };
        const auto repairsFactor = [&](std::size_t point)
        {
            return repairsDensity(point) ||
                   (neighborStatus[point] & pdg::KnnDistanceTie) != 0U ||
                   (closureStatus[point] & pdg::KnnSearchIncomplete) != 0U;
        };
        std::vector<PointId> kDistanceRepairPoints;
        std::vector<PointId> densityRepairPoints;
        std::vector<PointId> factorRepairPoints;
        for (std::size_t point = 0; point < resident->pointCount; ++point)
        {
            ambiguousRows += static_cast<std::size_t>(
                (status[point] & pdg::KnnDistanceTie) != 0U);
            if (repairsKDistance(point))
                kDistanceRepairPoints.push_back(static_cast<PointId>(point));
            if (repairsDensity(point))
                densityRepairPoints.push_back(static_cast<PointId>(point));
            if (repairsFactor(point))
                factorRepairPoints.push_back(static_cast<PointId>(point));
        }
        const std::size_t incompleteRows = kDistanceRepairPoints.size();
        const std::size_t repairRows = factorRepairPoints.size();
        if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR") &&
            ambiguousRows == 0U)
            throw NeighborhoodProofFailure(
                "required resident neighborhood tie repair did not occur");
        const std::size_t repairWorkerCount = lofRepairWorkerCount(repairRows);
        if (std::getenv("PDG_REQUIRE_LOF_PARALLEL_REPAIR") &&
            repairWorkerCount <= 1U)
            throw NeighborhoodProofFailure(
                "required parallel LOF repair did not occur");
        const bool requireCoordinateCache =
            std::getenv("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE") != nullptr;
        if (requireCoordinateCache &&
            (repairRows < LofParallelRepairMinimumRows ||
             std::getenv("PDG_DISABLE_LOF_KD3_COORDINATE_CACHE")))
            throw NeighborhoodProofFailure(
                "required LOF KD3 coordinate cache was not used");
        if (repairRows != 0U)
        {
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto repairStarted = std::chrono::steady_clock::now();
            if (phases)
            {
                phases->ambiguousRepairRows += ambiguousRows;
                phases->incompleteRepairRows += incompleteRows;
                phases->repairedRows += repairRows;
                phases->lofRepairWorkers =
                    (std::max)(phases->lofRepairWorkers, repairWorkerCount);
                if (repairWorkerCount > 1U)
                    phases->lofParallelRepairRows += repairRows;
            }
            double* hostDensities = resident->host->data<double>(reachability);
            double* hostFactors = resident->host->data<double>(outlierFactor);
            double* hostKDistances = resident->host->data<double>(nnDistance);
            const bool requestCoordinateCache =
                repairRows >= LofParallelRepairMinimumRows &&
                !std::getenv("PDG_DISABLE_LOF_KD3_COORDINATE_CACHE");
            KD3Index* index = nullptr;
            try
            {
                index = &view.build3dIndex(requestCoordinateCache);
            }
            catch (const std::exception&)
            {
                // The contiguous backing is only an optimization. PointView
                // publishes an index after a complete build, so an ordinary
                // cache allocation/build failure can safely retry the same
                // exact nanoflann index against the original view adapter.
                // Proof-bearing runs must instead expose the failed cache.
                if (!requestCoordinateCache || requireCoordinateCache)
                    throw;
                index = &view.build3dIndex(false);
            }
            if (requireCoordinateCache && !index->coordinatesCached())
                throw NeighborhoodProofFailure(
                    "required LOF KD3 coordinate cache was not used");
            if (phases && index->coordinatesCached())
                phases->lofCoordinateCacheRows += resident->pointCount;
            // KD3Index and its vendored nanoflann query are const. Every
            // query owns its result set, query point, and distance vector;
            // the built tree and PointView coordinates are read-only here.
            forEachLofRepairChunk(
                kDistanceRepairPoints.size(),
                (std::min)(repairWorkerCount, kDistanceRepairPoints.size()),
                [&](std::size_t begin, std::size_t end)
                {
                    PointIdList indices(neighbors);
                    std::vector<double> squaredDistances(neighbors);
                    for (std::size_t item = begin; item < end; ++item)
                    {
                        const PointId point = kDistanceRepairPoints[item];
                        index->knnSearch(point, neighbors, &indices,
                                         &squaredDistances);
                        hostKDistances[static_cast<std::size_t>(point)] =
                            std::sqrt(squaredDistances[neighbors - 1U]);
                    }
                });
            const auto onlineLofPass =
                [&](PointId point, PointIdList& indices,
                    std::vector<double>& squaredDistances,
                    const auto& contribution)
            {
                index->knnSearch(point, neighbors, &indices, &squaredDistances);
                double mean = 0.0;
                for (std::size_t item = 0;
                     item < static_cast<std::size_t>(neighbors); ++item)
                    mean +=
                        (contribution(indices[item], squaredDistances[item]) -
                         mean) /
                        static_cast<double>(item + 1U);
                return mean;
            };
            // Join above is the k-distance -> density dependency barrier.
            forEachLofRepairChunk(
                densityRepairPoints.size(),
                (std::min)(repairWorkerCount, densityRepairPoints.size()),
                [&](std::size_t begin, std::size_t end)
                {
                    PointIdList indices(neighbors);
                    std::vector<double> squaredDistances(neighbors);
                    for (std::size_t item = begin; item < end; ++item)
                    {
                        const PointId point = densityRepairPoints[item];
                        const std::size_t offset =
                            static_cast<std::size_t>(point);
                        const double mean = onlineLofPass(
                            point, indices, squaredDistances,
                            [&](PointId neighbor, double squaredDistance)
                            {
                                const double distance =
                                    std::sqrt(squaredDistance);
                                return (std::max)(hostKDistances[static_cast<
                                                      std::size_t>(neighbor)],
                                                  distance);
                            });
                        hostDensities[offset] = 1.0 / mean;
                    }
                });
            // Join above is the density -> factor dependency barrier.
            forEachLofRepairChunk(
                factorRepairPoints.size(),
                (std::min)(repairWorkerCount, factorRepairPoints.size()),
                [&](std::size_t begin, std::size_t end)
                {
                    PointIdList indices(neighbors);
                    std::vector<double> squaredDistances(neighbors);
                    for (std::size_t item = begin; item < end; ++item)
                    {
                        const PointId point = factorRepairPoints[item];
                        const std::size_t offset =
                            static_cast<std::size_t>(point);
                        const double density = hostDensities[offset];
                        hostFactors[offset] = onlineLofPass(
                            point, indices, squaredDistances,
                            [&](PointId neighbor, double)
                            {
                                return hostDensities[static_cast<std::size_t>(
                                           neighbor)] /
                                       density;
                            });
                    }
                });
            // Keep device columns canonical for a retained region's bridges.
            for (const pdg::DimensionId column :
                 {nnDistance, reachability, outlierFactor})
            {
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(resident->device->rawData(column),
                                    resident->host->rawData(column),
                                    resident->pointCount * sizeof(double),
                                    cudaMemcpyHostToDevice, stream));
                observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                resident->region,
                                resident->pointCount * sizeof(double));
            }
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            accumulatePhaseSeconds(phases ? &phases->exactHostRepair : nullptr,
                                   repairStarted);
        }
        publishOutputColumns(*resident, view, LofColumns);
        closeRegion();
        return true;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda || bridgeRebuild)
            throw;
        return false;
    }
}

bool tryCudaCovarianceFeatureColumns(
    PointView& view, std::uint32_t neighbors, CudaNeighborhoodRegion region,
    pdg::EigenvalueMode mode, std::uint32_t features,
    std::shared_ptr<const CudaNeighborhoodResults>& result, bool requireCuda)
{
    const std::vector<OutputColumn> columns = covarianceColumns(features);
    constexpr std::uint32_t HostFeatures =
        pdg::CovarianceOmnivariance | pdg::CovarianceEigenentropy;
    const std::uint32_t hostFeatures = features & HostFeatures;
    const pdg::CovarianceFeaturesProgram program{
        static_cast<std::int32_t>(neighbors - 1U), mode,
        features & ~HostFeatures};
    return tryCudaColumnsImpl(
        view, neighbors, region, columns,
        [&program](pdg::PointBatch& batch, const pdg::EigenSystem3d* systems,
                   std::uint8_t* status)
        {
            pdg::projectCovarianceFeatureColumns(batch, systems, status,
                                                 program);
        },
        mode, hostFeatures, EigenRepairAttribution::Other, result, requireCuda);
}

bool tryCudaLabelDuplicatesColumn(PointView& view,
                                  const pdg::LabelDuplicatesProgram& program,
                                  const pdg::DimensionRegistry& dimensions,
                                  CudaNeighborhoodRegion region,
                                  bool requireCuda)
{
    const auto invalidateRegion = [&]()
    {
        if (region.id != 0U)
            ResidentPointViewAccess::clear(view);
    };
    const auto closeRegion = [&]()
    {
        if (region.id != 0U && region.last)
            ResidentPointViewAccess::clear(view);
    };
    if (std::find(program.dimensions.begin(), program.dimensions.end(),
                  pdg::DimensionId(pdg::StandardDimension::Duplicate)) !=
        program.dimensions.end())
    {
        invalidateRegion();
        return false;
    }
    if (view.empty())
    {
        closeRegion();
        return true;
    }

    std::shared_ptr<ResidentNeighborhood> resident;
    if (region.id != 0U)
    {
        const std::shared_ptr<void> product =
            ResidentPointViewAccess::get(view);
        if (product)
        {
            resident = std::static_pointer_cast<ResidentNeighborhood>(product);
            bool compatible =
                resident->region == region.id &&
                resident->pointCount == static_cast<std::size_t>(view.size());
            if (compatible && region.indexRequired)
            {
                compatible =
                    region.radiusIndex
                        ? matchesRadius(
                              *resident, view, region, region.maximumRadius,
                              static_cast<std::uint8_t>(region.dimensions))
                        : matches(*resident, view, region,
                                  region.maximumNeighbors);
            }
            if (!compatible)
                resident.reset();
        }
    }
    if (region.reuseExpected && std::getenv("PDG_REQUIRE_NEIGHBORHOOD_REUSE") &&
        !resident)
        throw std::runtime_error(
            "required resident label_duplicates column reuse did not occur");

    const auto physicalType = [](Dimension::Type type)
    {
        switch (type)
        {
        case Dimension::Type::Signed8:
            return pdg::DimensionType::Signed8;
        case Dimension::Type::Signed16:
            return pdg::DimensionType::Signed16;
        case Dimension::Type::Signed32:
            return pdg::DimensionType::Signed32;
        case Dimension::Type::Signed64:
            return pdg::DimensionType::Signed64;
        case Dimension::Type::Unsigned8:
            return pdg::DimensionType::Unsigned8;
        case Dimension::Type::Unsigned16:
            return pdg::DimensionType::Unsigned16;
        case Dimension::Type::Unsigned32:
            return pdg::DimensionType::Unsigned32;
        case Dimension::Type::Unsigned64:
            return pdg::DimensionType::Unsigned64;
        case Dimension::Type::Float:
            return pdg::DimensionType::Float;
        case Dimension::Type::Double:
            return pdg::DimensionType::Double;
        case Dimension::Type::None:
            return pdg::DimensionType::None;
        }
        return pdg::DimensionType::None;
    };

    try
    {
        if (!resident)
        {
            if (pdg::cudaDevices().empty())
            {
                invalidateRegion();
                return false;
            }
            const std::uint32_t neighbors =
                region.indexRequired && !region.radiusIndex
                    ? region.maximumNeighbors
                    : 0U;
            const double radius = region.indexRequired && region.radiusIndex
                                      ? region.maximumRadius
                                      : 0.0;
            resident =
                makeResidentNeighborhood(view, region, neighbors, radius);
            if (!resident)
            {
                invalidateRegion();
                return false;
            }
        }

        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        const bool physicalRows = rowBoundaryUsable(view);
        pdg::LabelDuplicatesProgram residentProgram;
        const auto ensureColumn = [&](pdg::DimensionId source)
        {
            const pdg::DimensionDefinition& sourceDefinition =
                dimensions.require(source);
            const pdg::DimensionDefinition* residentDefinition =
                resident->dimensions.find(sourceDefinition.name);
            if (!residentDefinition)
                residentDefinition = &resident->dimensions.registerCustom(
                    sourceDefinition.name, sourceDefinition.type);
            const Dimension::Id pointView =
                view.layout()->findDim(sourceDefinition.name);
            if (pointView == Dimension::Id::Unknown)
                throw std::invalid_argument(
                    "resident label_duplicates input dimension is missing");
            const Dimension::Type pointViewType =
                view.layout()->dimType(pointView);
            const pdg::DimensionType physical = physicalType(pointViewType);
            if (physical == pdg::DimensionType::None)
                throw std::invalid_argument(
                    "resident label_duplicates input type is unsupported");
            const pdg::DimensionId target = residentDefinition->id;
            if (resident->device->has(target))
            {
                if (resident->device->columnInfo(target).physicalType !=
                    physical)
                    throw std::invalid_argument(
                        "resident label_duplicates column type changed");
                return target;
            }

            resident->host->materialize(target, physical);
            resident->device->materialize(target, physical);
            const std::size_t stride = pdg::dimensionTypeSize(physical);
            ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
            const auto started = std::chrono::steady_clock::now();
            gatherColumnFromView(
                view, pointView, pointViewType, physicalRows,
                static_cast<std::byte*>(resident->host->rawData(target)),
                stride);
            accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr,
                                   started);
            const std::size_t bytes = resident->pointCount * stride;
            PDG_CUDA_CHECK(cudaMemcpyAsync(resident->device->rawData(target),
                                           resident->host->rawData(target),
                                           bytes, cudaMemcpyHostToDevice,
                                           stream));
            observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                            resident->region, bytes);
            return target;
        };

        residentProgram.dimensions.reserve(program.dimensions.size());
        for (pdg::DimensionId dimension : program.dimensions)
            residentProgram.dimensions.push_back(ensureColumn(dimension));
        const pdg::DimensionId duplicate =
            ensureColumn(pdg::DimensionId(pdg::StandardDimension::Duplicate));
        if (!pdg::labelDuplicatesMaySupportExactDevice(*resident->host,
                                                       residentProgram))
            throw std::invalid_argument(
                "resident label_duplicates exact envelope was not met");

        pdg::labelDuplicates(*resident->device, residentProgram,
                             resident->device->data<std::uint8_t>(duplicate));
        PDG_CUDA_CHECK(cudaMemcpyAsync(resident->host->rawData(duplicate),
                                       resident->device->rawData(duplicate),
                                       resident->pointCount,
                                       cudaMemcpyDeviceToHost, stream));
        observeTransfer(pdg::ExecutionEventKind::DeviceToHost, resident->region,
                        resident->pointCount);
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);
        const auto publishStarted = std::chrono::steady_clock::now();
        publishColumnToView(
            view, Dimension::Id::Duplicate, Dimension::Type::Unsigned8,
            physicalRows,
            static_cast<const std::byte*>(resident->host->rawData(duplicate)),
            sizeof(std::uint8_t));
        accumulatePhaseSeconds(phases ? &phases->spillPublish : nullptr,
                               publishStarted);

        if (region.id != 0U && !region.last)
            ResidentPointViewAccess::set(view, resident);
        else
            closeRegion();
        return true;
    }
    catch (const pdg::CudaError&)
    {
        invalidateRegion();
        if (requireCuda)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        invalidateRegion();
        if (requireCuda)
            throw;
        return false;
    }
}

bool tryCudaResidentAssignments(PointView& view, CudaNeighborhoodRegion region,
                                const pdg::AssignProgram& program,
                                bool requireCuda)
{
    // The resident spatial index is valid only for the original XYZ values.
    // Reject an XYZ-mutating bridge before it can enqueue a device kernel;
    // the caller then takes its existing exact host path, whose
    // invalidateProducts() closes this region before any later neighborhood
    // client is considered.
    for (const pdg::DimensionId coordinate : {X, Y, Z})
        if (std::find(program.writes.begin(), program.writes.end(),
                      coordinate) != program.writes.end())
            return false;

    const bool requireColumnReuse =
        std::getenv("PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE") ||
        (!region.last &&
         std::getenv("PDG_REQUIRE_NONTERMINAL_NEIGHBORHOOD_COLUMN_REUSE"));

    // No device column is needed for an empty view. Treat the bridge as the
    // exact no-op it is upstream, while still honoring the terminal resident
    // product lifetime.
    if (view.empty())
    {
        if (region.last)
            ResidentPointViewAccess::clear(view);
        return true;
    }

    const std::shared_ptr<void> product = ResidentPointViewAccess::get(view);
    std::shared_ptr<ResidentNeighborhood> resident;
    if (product)
    {
        resident = std::static_pointer_cast<ResidentNeighborhood>(product);
        if (!region.id || resident->region != region.id ||
            resident->pointCount != static_cast<std::size_t>(view.size()))
            resident.reset();
    }
    if (!resident)
    {
        ResidentPointViewAccess::clear(view);
        if (requireColumnReuse)
            throw std::runtime_error(
                "required resident neighborhood column reuse did not occur");
        return false;
    }

    ResidentManagerDetailTimer bridgeTimer(
        &ResidentManagerDetailSeconds::adjacentPointProgramBridge);

    const pdg::DimensionId coplanar(pdg::StandardDimension::Coplanar);
    if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_COPLANAR_COLUMN_REUSE") &&
        std::find(program.reads.begin(), program.reads.end(), coplanar) !=
            program.reads.end() &&
        !resident->device->has(coplanar))
        throw std::runtime_error(
            "required resident Coplanar column reuse did not occur");

    const auto physicalType = [](Dimension::Type type)
    {
        switch (type)
        {
        case Dimension::Type::Signed8:
            return pdg::DimensionType::Signed8;
        case Dimension::Type::Signed16:
            return pdg::DimensionType::Signed16;
        case Dimension::Type::Signed32:
            return pdg::DimensionType::Signed32;
        case Dimension::Type::Signed64:
            return pdg::DimensionType::Signed64;
        case Dimension::Type::Unsigned8:
            return pdg::DimensionType::Unsigned8;
        case Dimension::Type::Unsigned16:
            return pdg::DimensionType::Unsigned16;
        case Dimension::Type::Unsigned32:
            return pdg::DimensionType::Unsigned32;
        case Dimension::Type::Unsigned64:
            return pdg::DimensionType::Unsigned64;
        case Dimension::Type::Float:
            return pdg::DimensionType::Float;
        case Dimension::Type::Double:
            return pdg::DimensionType::Double;
        case Dimension::Type::None:
            return pdg::DimensionType::None;
        }
        return pdg::DimensionType::None;
    };
    struct AssignmentColumn
    {
        pdg::DimensionId device;
        Dimension::Id pointView = Dimension::Id::Unknown;
        Dimension::Type pointViewType = Dimension::Type::None;
        pdg::DimensionType physical = pdg::DimensionType::None;
    };

    try
    {
        const pdg::DimensionId nnDistance(pdg::StandardDimension::NNDistance);
        bool nnDistanceDeviceColumnReused = false;
        std::vector<pdg::DimensionId> dimensions = program.reads;
        for (pdg::DimensionId write : program.writes)
            if (std::find(dimensions.begin(), dimensions.end(), write) ==
                dimensions.end())
                dimensions.push_back(write);
        std::vector<AssignmentColumn> columns;
        columns.reserve(dimensions.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(resident->device->nativeStreamHandle());
        for (pdg::DimensionId dimension : dimensions)
        {
            if (dimension.value() >= pdg::CustomDimensionBase)
                throw UnsupportedResidentAssignments();
            const pdg::DimensionDefinition& definition =
                resident->dimensions.require(dimension);
            const Dimension::Id pointView =
                view.layout()->findDim(definition.name);
            if (pointView == Dimension::Id::Unknown)
                throw UnsupportedResidentAssignments();
            const Dimension::Type pointViewType =
                view.layout()->dimType(pointView);
            const pdg::DimensionType physical = physicalType(pointViewType);
            if (physical == pdg::DimensionType::None)
                throw UnsupportedResidentAssignments();
            if (resident->device->has(dimension))
            {
                if (resident->device->columnInfo(dimension).physicalType !=
                    physical)
                    throw UnsupportedResidentAssignments();
                if (dimension == nnDistance &&
                    std::find(program.reads.begin(), program.reads.end(),
                              dimension) != program.reads.end())
                    nnDistanceDeviceColumnReused = true;
            }
            else
            {
                resident->host->materialize(dimension, physical);
                resident->device->materialize(dimension, physical);
                std::byte* destination =
                    static_cast<std::byte*>(resident->host->rawData(dimension));
                const std::size_t stride = pdg::dimensionTypeSize(physical);
                ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
                const auto started = std::chrono::steady_clock::now();
                gatherColumnFromView(view, pointView, pointViewType,
                                     rowBoundaryUsable(view), destination,
                                     stride);
                accumulatePhaseSeconds(phases ? &phases->uploadPack : nullptr,
                                       started);
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(resident->device->rawData(dimension),
                                    destination, resident->pointCount * stride,
                                    cudaMemcpyHostToDevice, stream));
                observeTransfer(pdg::ExecutionEventKind::HostToDevice,
                                resident->region,
                                resident->pointCount * stride);
            }
            columns.push_back({dimension, pointView, pointViewType, physical});
        }
        if (!pdg::assignSupportsExactDevice(*resident->device, program))
            throw UnsupportedResidentAssignments();
        pdg::executeAssign(*resident->device, program);
        if (ActiveDeviceOnlyNnDistanceHandoff && nnDistanceDeviceColumnReused &&
            ActiveNnDistanceAssignmentDeviceColumnReused)
            *ActiveNnDistanceAssignmentDeviceColumnReused = true;
        for (pdg::DimensionId write : program.writes)
        {
            const auto column =
                std::find_if(columns.begin(), columns.end(),
                             [&](const AssignmentColumn& candidate)
                             { return candidate.device == write; });
            if (column == columns.end())
                throw std::logic_error(
                    "resident assignment output binding is missing");
            const std::size_t bytes =
                resident->pointCount * pdg::dimensionTypeSize(column->physical);
            PDG_CUDA_CHECK(cudaMemcpyAsync(resident->host->rawData(write),
                                           resident->device->rawData(write),
                                           bytes, cudaMemcpyDeviceToHost,
                                           stream));
            observeTransfer(pdg::ExecutionEventKind::DeviceToHost,
                            resident->region, bytes);
        }
        ResidentPhaseSeconds* phases = activeResidentPhaseSeconds();
        const auto waitStarted = std::chrono::steady_clock::now();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        accumulatePhaseSeconds(phases ? &phases->spillWait : nullptr,
                               waitStarted);
        const auto publishStarted = std::chrono::steady_clock::now();
        const bool physicalRows = rowBoundaryUsable(view);
        for (pdg::DimensionId write : program.writes)
        {
            const auto column =
                std::find_if(columns.begin(), columns.end(),
                             [&](const AssignmentColumn& candidate)
                             { return candidate.device == write; });
            const std::size_t stride = pdg::dimensionTypeSize(column->physical);
            const std::byte* source =
                static_cast<const std::byte*>(resident->host->rawData(write));
            publishColumnToView(view, column->pointView, column->pointViewType,
                                physicalRows, source, stride);
        }
        accumulatePhaseSeconds(phases ? &phases->spillPublish : nullptr,
                               publishStarted);
        if (ActiveResidentAssignmentExecuted)
            *ActiveResidentAssignmentExecuted = true;
        if (region.last)
            ResidentPointViewAccess::clear(view);
        return true;
    }
    catch (const UnsupportedResidentAssignments&)
    {
        ResidentPointViewAccess::clear(view);
        if (std::getenv("PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD") &&
            region.id != 0 && !region.last)
            RequiredBridgeRebuildRegion = region.id;
        if (requireColumnReuse)
            throw;
        return false;
    }
    catch (const NeighborhoodProofFailure&)
    {
        ResidentPointViewAccess::clear(view);
        throw;
    }
    catch (const pdg::CudaError&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda)
            throw;
        return false;
    }
    catch (const std::exception&)
    {
        ResidentPointViewAccess::clear(view);
        if (requireCuda)
            throw;
        return false;
    }
}
#else
bool tryCudaRadiusCounts(PointView&, double, std::vector<std::uint32_t>&, bool)
{
    return false;
}

bool tryCudaRadiusOutlier(PointView&, double, int, std::uint8_t,
                          CudaNeighborhoodRegion, CudaRadiusOutlierResult&,
                          bool)
{
    return false;
}

bool tryCudaRadiusScaledValues(PointView&, double, double, std::vector<double>&,
                               bool)
{
    return false;
}

bool tryCudaResidentRadiusScaledValues(PointView&, double, double,
                                       CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaRadiusAssign(PointView&, double, bool, double, double,
                         const DimRangeList&, const DimRangeList&,
                         std::vector<expr::AssignStatement>&,
                         CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaNormalColumns(PointView&, std::uint32_t, CudaNeighborhoodRegion,
                          bool, std::shared_ptr<const CudaNeighborhoodResults>&,
                          bool)
{
    return false;
}

bool tryCudaEigenvalueColumns(PointView&, std::uint32_t, CudaNeighborhoodRegion,
                              bool,
                              std::shared_ptr<const CudaNeighborhoodResults>&,
                              bool)
{
    return false;
}

bool tryCudaApproximateCoplanarColumn(
    PointView&, std::uint32_t, CudaNeighborhoodRegion, double, double,
    std::shared_ptr<const CudaNeighborhoodResults>&, bool)
{
    return false;
}

bool tryCudaNnDistanceColumns(PointView&, std::uint32_t, CudaNeighborhoodRegion,
                              pdg::KnnDistanceMode, bool)
{
    return false;
}

bool tryCudaStatisticalOutlier(PointView&, std::uint32_t, double, std::uint8_t,
                               CudaNeighborhoodRegion,
                               CudaStatisticalOutlierResult&, bool)
{
    return false;
}

bool tryCudaHagNnColumn(PointView&, const pdg::HagNnProgram&,
                        CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaHagDelaunayColumn(PointView&, const pdg::HagDelaunayProgram&,
                              CudaNeighborhoodRegion, bool)
{
    return false;
}

bool retainCudaHagHostColumn(PointView&, std::uint32_t, CudaNeighborhoodRegion,
                             bool)
{
    return false;
}

bool tryCudaNeighborClassifierColumn(PointView&, std::uint32_t,
                                     CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaOptimalNeighborhoodColumns(PointView&, std::uint32_t, std::uint32_t,
                                       CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaEstimateRankColumn(PointView&, std::uint32_t,
                               CudaNeighborhoodRegion, double, bool)
{
    return false;
}

bool tryCudaLofColumns(PointView&, std::uint32_t, CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaCovarianceFeatureColumns(
    PointView&, std::uint32_t, CudaNeighborhoodRegion, pdg::EigenvalueMode,
    std::uint32_t, std::shared_ptr<const CudaNeighborhoodResults>&, bool)
{
    return false;
}

bool tryCudaLabelDuplicatesColumn(PointView&,
                                  const pdg::LabelDuplicatesProgram&,
                                  const pdg::DimensionRegistry&,
                                  CudaNeighborhoodRegion, bool)
{
    return false;
}

bool tryCudaResidentAssignments(PointView&, CudaNeighborhoodRegion,
                                const pdg::AssignProgram&, bool)
{
    return false;
}
#endif

} // namespace pdal::pdg_detail
