#include <pdg/Cuda.hpp>
#include <pdg/FastMode.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <Eigen/Eigenvalues>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

namespace pdg::detail
{

namespace
{
constexpr int BlockSize = 256;
constexpr int KnnBlockSize = 64;
constexpr std::uint32_t MaximumSelectiveRepairNeighbors = 16U;
constexpr std::uint32_t MaximumSelectiveRepairPartitions = 128U;
constexpr std::uint32_t MaximumCell = (1U << 21U) - 1U;
constexpr std::uint32_t MaximumKnnNeighbors = 64U;
constexpr std::uint32_t MaximumKnnShell = 4096U;
constexpr double DoubleEpsilon = 2.220446049250313080847263336181640625e-16;
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);

class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

// D0271: mask applied to every kNN status the gather kernels publish. All
// bits under the default contract; KnnDistanceTie cleared under the relaxed
// tie-order contract (`gpupal --fast`). Written per index build from
// pdg::knnStatusMask() so the device and host paths agree.
__constant__ std::uint8_t c_knnStatusMask = 0xFFU;

unsigned int launchBlocks(std::size_t size, int blockSize = BlockSize)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(blockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__device__ std::uint32_t cellCoordinate(double value, double origin,
                                        double cellSize) noexcept
{
    return static_cast<std::uint32_t>(
        floor(__ddiv_rn(__dsub_rn(value, origin), cellSize)));
}

__host__ __device__ std::uint64_t cellKey(std::uint32_t x, std::uint32_t y,
                                          std::uint32_t z) noexcept
{
    std::uint64_t key = 0;
    for (unsigned int bit = 0; bit < 21U; ++bit)
    {
        key |= static_cast<std::uint64_t>((x >> bit) & 1U) << (3U * bit);
        key |= static_cast<std::uint64_t>((y >> bit) & 1U) << (3U * bit + 1U);
        key |= static_cast<std::uint64_t>((z >> bit) & 1U) << (3U * bit + 2U);
    }
    return key;
}

__global__ void generateCellsKernel(const double* x, const double* y,
                                    const double* z, std::size_t size,
                                    std::uint8_t dimensions, double cellSize,
                                    double originX, double originY,
                                    double originZ, std::uint64_t* keys,
                                    std::uint32_t* pointIds)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const std::uint32_t cellX = cellCoordinate(x[point], originX, cellSize);
        const std::uint32_t cellY = cellCoordinate(y[point], originY, cellSize);
        const std::uint32_t cellZ =
            dimensions == 3 ? cellCoordinate(z[point], originZ, cellSize) : 0U;
        keys[point] = cellKey(cellX, cellY, cellZ);
        pointIds[point] = static_cast<std::uint32_t>(point);
    }
}

__device__ std::uint32_t findCell(const std::uint64_t* keys,
                                  std::uint32_t count,
                                  std::uint64_t key) noexcept
{
    std::uint32_t begin = 0;
    std::uint32_t end = count;
    while (begin < end)
    {
        const std::uint32_t middle = begin + (end - begin) / 2U;
        if (keys[middle] < key)
            begin = middle + 1U;
        else
            end = middle;
    }
    return begin < count && keys[begin] == key ? begin : count;
}

__global__ void radiusCountsKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double cellSize, double originX, double originY,
    double originZ, const std::uint64_t* uniqueKeys,
    const std::uint32_t* cellOffsets, const std::uint32_t* cellCounts,
    std::uint32_t cellCount, const std::uint32_t* sortedPointIds,
    double radiusSquared, std::uint32_t* outputCounts, double* outputValues,
    double factor)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        const std::uint32_t queryCellX =
            cellCoordinate(x[query], originX, cellSize);
        const std::uint32_t queryCellY =
            cellCoordinate(y[query], originY, cellSize);
        const std::uint32_t queryCellZ =
            dimensions == 3 ? cellCoordinate(z[query], originZ, cellSize) : 0U;
        std::uint32_t count = 0;
        const int zMinimum = dimensions == 3 ? -1 : 0;
        const int zMaximum = dimensions == 3 ? 1 : 0;
        for (int dzCell = zMinimum; dzCell <= zMaximum; ++dzCell)
            for (int dyCell = -1; dyCell <= 1; ++dyCell)
                for (int dxCell = -1; dxCell <= 1; ++dxCell)
                {
                    const long long cellX =
                        static_cast<long long>(queryCellX) + dxCell;
                    const long long cellY =
                        static_cast<long long>(queryCellY) + dyCell;
                    const long long cellZ =
                        static_cast<long long>(queryCellZ) + dzCell;
                    if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                        cellX > MaximumCell || cellY > MaximumCell ||
                        cellZ > MaximumCell)
                        continue;
                    const std::uint64_t key =
                        cellKey(static_cast<std::uint32_t>(cellX),
                                static_cast<std::uint32_t>(cellY),
                                static_cast<std::uint32_t>(cellZ));
                    const std::uint32_t cell =
                        findCell(uniqueKeys, cellCount, key);
                    if (cell == cellCount)
                        continue;
                    const std::uint32_t end =
                        cellOffsets[cell] + cellCounts[cell];
                    for (std::uint32_t item = cellOffsets[cell]; item < end;
                         ++item)
                    {
                        const std::uint32_t candidate = sortedPointIds[item];
                        const double dx = __dsub_rn(x[query], x[candidate]);
                        const double dy = __dsub_rn(y[query], y[candidate]);
                        double distance =
                            __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                        if (dimensions == 3)
                        {
                            const double dz = __dsub_rn(z[query], z[candidate]);
                            distance = __dadd_rn(distance, __dmul_rn(dz, dz));
                        }
                        if (distance < radiusSquared)
                            ++count;
                    }
                }
        if (outputCounts)
            outputCounts[query] = count;
        if (outputValues)
            outputValues[query] = __dmul_rn(static_cast<double>(count), factor);
    }
}

__device__ bool
radiusCandidateAccepted(const double* x, const double* y, const double* z,
                        std::size_t query, std::uint32_t candidate,
                        std::uint8_t dimensions, double radiusSquared,
                        const std::uint8_t* referenceMask, double maximumAbove,
                        double maximumBelow) noexcept
{
    if (referenceMask && referenceMask[candidate] == 0U)
        return false;
    const double dx = __dsub_rn(x[query], x[candidate]);
    const double dy = __dsub_rn(y[query], y[candidate]);
    double distance = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
    if (dimensions == 3)
    {
        const double dz = __dsub_rn(z[query], z[candidate]);
        distance = __dadd_rn(distance, __dmul_rn(dz, dz));
    }
    if (!(distance < radiusSquared))
        return false;
    if (dimensions == 2)
    {
        const double difference = __dsub_rn(z[candidate], z[query]);
        if (maximumAbove >= 0.0 && difference > 0.0 &&
            difference > maximumAbove)
            return false;
        if (maximumBelow >= 0.0 && difference < 0.0 &&
            -difference > maximumBelow)
            return false;
    }
    return true;
}

__global__ void radiusAnyKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double cellSize, double originX, double originY,
    double originZ, const std::uint64_t* uniqueKeys,
    const std::uint32_t* cellOffsets, const std::uint32_t* cellCounts,
    std::uint32_t cellCount, const std::uint32_t* sortedPointIds,
    double radiusSquared, const std::uint8_t* sourceMask,
    const std::uint8_t* referenceMask, double maximumAbove, double maximumBelow,
    std::uint8_t* matches)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        bool found = false;
        if (!sourceMask || sourceMask[query] != 0U)
        {
            const std::uint32_t queryCellX =
                cellCoordinate(x[query], originX, cellSize);
            const std::uint32_t queryCellY =
                cellCoordinate(y[query], originY, cellSize);
            const std::uint32_t queryCellZ =
                dimensions == 3 ? cellCoordinate(z[query], originZ, cellSize)
                                : 0U;
            const int zMinimum = dimensions == 3 ? -1 : 0;
            const int zMaximum = dimensions == 3 ? 1 : 0;
            for (int dzCell = zMinimum; dzCell <= zMaximum && !found; ++dzCell)
                for (int dyCell = -1; dyCell <= 1 && !found; ++dyCell)
                    for (int dxCell = -1; dxCell <= 1 && !found; ++dxCell)
                    {
                        const long long cellX =
                            static_cast<long long>(queryCellX) + dxCell;
                        const long long cellY =
                            static_cast<long long>(queryCellY) + dyCell;
                        const long long cellZ =
                            static_cast<long long>(queryCellZ) + dzCell;
                        if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                            cellX > MaximumCell || cellY > MaximumCell ||
                            cellZ > MaximumCell)
                            continue;
                        const std::uint64_t key =
                            cellKey(static_cast<std::uint32_t>(cellX),
                                    static_cast<std::uint32_t>(cellY),
                                    static_cast<std::uint32_t>(cellZ));
                        const std::uint32_t cell =
                            findCell(uniqueKeys, cellCount, key);
                        if (cell == cellCount)
                            continue;
                        const std::uint32_t end =
                            cellOffsets[cell] + cellCounts[cell];
                        for (std::uint32_t item = cellOffsets[cell]; item < end;
                             ++item)
                            if (radiusCandidateAccepted(
                                    x, y, z, query, sortedPointIds[item],
                                    dimensions, radiusSquared, referenceMask,
                                    maximumAbove, maximumBelow))
                            {
                                found = true;
                                break;
                            }
                    }
        }
        matches[query] = static_cast<std::uint8_t>(found);
    }
}

// Gathers the batch coordinates into sorted-point order so candidate loops
// read them contiguously. The double copies are bit-identical to their
// batch column sources; the float copies are grid-origin-shifted and feed
// only the certified distance prefilter, never a stored output bit.
__global__ void gatherSortedCoordinatesKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    double originX, double originY, double originZ,
    const std::uint32_t* sortedPointIds, double* sortedX, double* sortedY,
    double* sortedZ, float* sortedFloatX, float* sortedFloatY,
    float* sortedFloatZ)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t item = thread; item < size; item += grid)
    {
        const std::uint32_t point = sortedPointIds[item];
        const double pointX = x[point];
        const double pointY = y[point];
        const double pointZ = z ? z[point] : 0.0;
        sortedX[item] = pointX;
        sortedY[item] = pointY;
        sortedZ[item] = pointZ;
        sortedFloatX[item] = __double2float_rn(__dsub_rn(pointX, originX));
        sortedFloatY[item] = __double2float_rn(__dsub_rn(pointY, originY));
        sortedFloatZ[item] =
            z ? __double2float_rn(__dsub_rn(pointZ, originZ)) : 0.0F;
    }
}

// Certified float prefilter for the exact kNN candidate loop. A candidate
// whose float squared distance, reduced by a rigorous rounding margin,
// still exceeds the round-up float of the current worst retained entry is
// provably a candidate the exact double path would discard, so the double
// arithmetic can be skipped without changing any output bit.
//
// Margin derivation, per axis with shifted coordinates in [0, E):
// stored/query floats err by at most E*u (u = 2^-24 relative rounding),
// their float difference by 3*E*u =: e, each squared term by
// e*(2*|difference| + e), and the three-term float sum by a factor within
// (1 +- 4u). With S the computed float sum, the true squared distance is
// at least S*(1-4u) - 2*e*sqrt(3*S) - 3*e^2. The constants below are built
// from uu = 2^-23 (double the unit) with a further 2x inflation on the
// absolute terms and 64*uu of relative slack, an order of magnitude beyond
// every rounding factor the chain consumes, including the final float
// evaluation of the bound itself and the double path's own rounding.
struct KnnPrefilterConstants
{
    bool enabled = false;
    float scale = 0.0F;
    float linear = 0.0F;
    float constant = 0.0F;
};

// Device shell budget for the exact kNN walk. A query that cannot prove
// completeness within this many shells reports KnnSearchIncomplete and is
// repaired exactly on the host through the per-family repair contracts;
// this keeps a handful of isolated outlier points from serializing the
// kernel while the rest of the device drains (B0016). The ceiling stays
// MaximumKnnShell; the budget only decides where the device stops trying.
std::uint32_t knnDeviceShellBudget()
{
    constexpr std::uint32_t DefaultBudget = 32U;
    const char* text = std::getenv("PDG_KNN_DEVICE_SHELL_BUDGET");
    if (!text || !*text)
        return DefaultBudget;
    errno = 0;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || end == text || *end != '\0' || value == 0UL ||
        value > MaximumKnnShell)
        throw std::invalid_argument(
            "PDG_KNN_DEVICE_SHELL_BUDGET must be an integer in [1, 4096]");
    return static_cast<std::uint32_t>(value);
}

KnnPrefilterConstants knnPrefilterConstants(const UniformGridConfig& config)
{
    if (std::getenv("PDG_DISABLE_KNN_DISTANCE_PREFILTER"))
        return {};
    double extent = 0.0;
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
        extent =
            (std::max)(extent,
                       (static_cast<double>(config.maximumCell[axis]) + 1.0) *
                           config.cellSize);
    constexpr double uu = 1.1920928955078125e-7; // 2^-23
    const double e = 3.0 * extent * uu;
    const double linear = 2.0 * 2.0 * std::sqrt(3.0) * e;
    const double constant = 2.0 * 3.0 * e * e;
    const double scale = 1.0 - 64.0 * uu;
    // A frame too large for float squared distances (4*E^2 must stay well
    // inside float range) disables the prefilter; the exact path is the
    // unconditional fallback.
    if (!std::isfinite(extent) || extent <= 0.0 ||
        4.0 * extent * extent > 1.0e37 || !std::isfinite(linear))
        return {};
    KnnPrefilterConstants constants;
    constants.enabled = true;
    constants.scale = std::nextafterf(static_cast<float>(scale), 0.0F);
    constants.linear = std::nextafterf(static_cast<float>(linear),
                                       (std::numeric_limits<float>::max)());
    constants.constant = std::nextafterf(static_cast<float>(constant),
                                         (std::numeric_limits<float>::max)());
    return constants;
}

__device__ bool neighborPrecedes(double leftDistance, std::uint32_t leftPoint,
                                 double rightDistance,
                                 std::uint32_t rightPoint) noexcept
{
    return leftDistance < rightDistance ||
           (leftDistance == rightDistance && leftPoint < rightPoint);
}

__device__ void retainNeighbor(std::uint32_t candidate, double distance,
                               std::uint32_t capacity, std::uint32_t& retained,
                               std::uint32_t* bestPoints,
                               double* bestDistances) noexcept
{
    std::uint32_t position = 0U;
    while (position < retained &&
           neighborPrecedes(bestDistances[position], bestPoints[position],
                            distance, candidate))
        ++position;
    if (position >= capacity)
        return;
    const std::uint32_t newCount =
        retained < capacity ? retained + 1U : retained;
    for (std::uint32_t item = newCount; item > position + 1U; --item)
    {
        bestPoints[item - 1U] = bestPoints[item - 2U];
        bestDistances[item - 1U] = bestDistances[item - 2U];
    }
    bestPoints[position] = candidate;
    bestDistances[position] = distance;
    retained = newCount;
}

using BvhBounds = MortonBvhBounds;

__device__ BvhBounds invalidBvhBounds() noexcept
{
    BvhBounds bounds;
    bounds.minimum = {CUDART_INF_F, CUDART_INF_F, CUDART_INF_F};
    bounds.maximum = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
    return bounds;
}

__device__ bool validBvhBounds(const BvhBounds& bounds) noexcept
{
    return bounds.minimum[0] <= bounds.maximum[0];
}

__device__ BvhBounds mergeBvhBounds(const BvhBounds& left,
                                    const BvhBounds& right) noexcept
{
    if (!validBvhBounds(left))
        return right;
    if (!validBvhBounds(right))
        return left;
    BvhBounds result;
    for (unsigned int axis = 0U; axis < 3U; ++axis)
    {
        result.minimum[axis] = left.minimum[axis] < right.minimum[axis]
                                   ? left.minimum[axis]
                                   : right.minimum[axis];
        result.maximum[axis] = left.maximum[axis] > right.maximum[axis]
                                   ? left.maximum[axis]
                                   : right.maximum[axis];
    }
    return result;
}

__device__ BvhBounds pointBvhBounds(const double* x, const double* y,
                                    const double* z, std::uint8_t dimensions,
                                    const double* origin,
                                    std::uint32_t point) noexcept
{
    const double coordinates[3] = {x[point], y[point],
                                   dimensions == 3 ? z[point] : 0.0};
    BvhBounds bounds;
    for (std::uint8_t axis = 0U; axis < dimensions; ++axis)
    {
        const double local = __dsub_rn(coordinates[axis], origin[axis]);
        bounds.minimum[axis] = __double2float_rd(local);
        bounds.maximum[axis] = __double2float_ru(local);
    }
    for (std::uint8_t axis = dimensions; axis < 3U; ++axis)
        bounds.minimum[axis] = bounds.maximum[axis] = 0.0F;
    return bounds;
}

__device__ BvhBounds heapBvhBounds(std::uint32_t node, std::uint32_t leafBase,
                                   std::uint32_t pointCount,
                                   const std::uint32_t* sortedPointIds,
                                   const BvhBounds* internalBounds,
                                   const double* x, const double* y,
                                   const double* z, std::uint8_t dimensions,
                                   const double* origin) noexcept
{
    const std::uint32_t leafStart = leafBase - 1U;
    if (node < leafStart)
        return internalBounds[node];
    const std::uint32_t position = node - leafStart;
    if (position >= pointCount)
        return invalidBvhBounds();
    return pointBvhBounds(x, y, z, dimensions, origin,
                          sortedPointIds[position]);
}

__device__ double bvhDistanceSquared(const double* query,
                                     std::uint8_t dimensions,
                                     const double* origin,
                                     const BvhBounds& bounds) noexcept
{
    if (!validBvhBounds(bounds))
        return CUDART_INF;
    double distance = 0.0;
    for (std::uint8_t axis = 0U; axis < dimensions; ++axis)
    {
        const double local = __dsub_rn(query[axis], origin[axis]);
        double delta = 0.0;
        if (local < static_cast<double>(bounds.minimum[axis]))
            delta = __dsub_ru(static_cast<double>(bounds.minimum[axis]), local);
        else if (local > static_cast<double>(bounds.maximum[axis]))
            delta = __dsub_ru(local, static_cast<double>(bounds.maximum[axis]));
        double scale = __dadd_ru(fabs(query[axis]), fabs(origin[axis]));
        scale = __dadd_ru(scale, fabs(local));
        scale = __dadd_ru(scale, fabs(delta));
        scale = __dadd_ru(scale, 1.0);
        const double error = __dmul_ru(scale, 16.0 * DoubleEpsilon);
        delta = delta > error ? __dsub_rd(delta, error) : 0.0;
        distance = __dadd_rd(distance, __dmul_rd(delta, delta));
    }
    return distance;
}

__global__ void buildMortonBvhLevelKernel(
    const double* x, const double* y, const double* z, std::uint32_t pointCount,
    std::uint8_t dimensions, double originX, double originY, double originZ,
    const std::uint32_t* sortedPointIds, std::uint32_t leafBase,
    std::uint32_t firstNode, std::uint32_t nodeCount, BvhBounds* internalBounds)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double origin[3] = {originX, originY, originZ};
    for (std::size_t offset = thread; offset < nodeCount; offset += grid)
    {
        const std::uint32_t node =
            firstNode + static_cast<std::uint32_t>(offset);
        const std::uint32_t left = node * 2U + 1U;
        const std::uint32_t right = left + 1U;
        internalBounds[node] = mergeBvhBounds(
            heapBvhBounds(left, leafBase, pointCount, sortedPointIds,
                          internalBounds, x, y, z, dimensions, origin),
            heapBvhBounds(right, leafBase, pointCount, sortedPointIds,
                          internalBounds, x, y, z, dimensions, origin));
    }
}

__global__ void bvhRadiusCountsKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double originX, double originY, double originZ,
    const std::uint32_t* sortedPointIds, const BvhBounds* internalBounds,
    std::uint32_t leafBase, double radiusSquared, std::uint32_t* outputCounts,
    double* outputValues, double factor)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double origin[3] = {originX, originY, originZ};
    const std::uint32_t pointCount = static_cast<std::uint32_t>(size);
    const std::uint32_t leafStart = leafBase - 1U;
    for (std::size_t query = thread; query < size; query += grid)
    {
        const double queryCoordinates[3] = {x[query], y[query],
                                            dimensions == 3 ? z[query] : 0.0};
        std::uint32_t stack[64];
        std::uint32_t stackSize = 1U;
        stack[0] = leafBase > 1U ? 0U : leafStart;
        std::uint32_t count = 0U;
        while (stackSize != 0U)
        {
            const std::uint32_t node = stack[--stackSize];
            const BvhBounds bounds =
                heapBvhBounds(node, leafBase, pointCount, sortedPointIds,
                              internalBounds, x, y, z, dimensions, origin);
            if (bvhDistanceSquared(queryCoordinates, dimensions, origin,
                                   bounds) >= radiusSquared)
                continue;
            if (node >= leafStart)
            {
                const std::uint32_t position = node - leafStart;
                if (position >= pointCount)
                    continue;
                const std::uint32_t candidate = sortedPointIds[position];
                const double dx = __dsub_rn(x[query], x[candidate]);
                const double dy = __dsub_rn(y[query], y[candidate]);
                double distance =
                    __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                if (dimensions == 3)
                {
                    const double dz = __dsub_rn(z[query], z[candidate]);
                    distance = __dadd_rn(distance, __dmul_rn(dz, dz));
                }
                if (distance < radiusSquared)
                    ++count;
                continue;
            }
            stack[stackSize++] = node * 2U + 2U;
            stack[stackSize++] = node * 2U + 1U;
        }
        if (outputCounts)
            outputCounts[query] = count;
        if (outputValues)
            outputValues[query] = __dmul_rn(static_cast<double>(count), factor);
    }
}

__global__ void bvhRadiusAnyKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double originX, double originY, double originZ,
    const std::uint32_t* sortedPointIds, const BvhBounds* internalBounds,
    std::uint32_t leafBase, double radiusSquared,
    const std::uint8_t* sourceMask, const std::uint8_t* referenceMask,
    double maximumAbove, double maximumBelow, std::uint8_t* matches)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double origin[3] = {originX, originY, originZ};
    const std::uint32_t pointCount = static_cast<std::uint32_t>(size);
    const std::uint32_t leafStart = leafBase - 1U;
    for (std::size_t query = thread; query < size; query += grid)
    {
        bool found = false;
        if (!sourceMask || sourceMask[query] != 0U)
        {
            const double queryCoordinates[3] = {
                x[query], y[query], dimensions == 3 ? z[query] : 0.0};
            std::uint32_t stack[64];
            std::uint32_t stackSize = 1U;
            stack[0] = leafBase > 1U ? 0U : leafStart;
            while (stackSize != 0U && !found)
            {
                const std::uint32_t node = stack[--stackSize];
                const BvhBounds bounds =
                    heapBvhBounds(node, leafBase, pointCount, sortedPointIds,
                                  internalBounds, x, y, z, dimensions, origin);
                if (bvhDistanceSquared(queryCoordinates, dimensions, origin,
                                       bounds) >= radiusSquared)
                    continue;
                if (node >= leafStart)
                {
                    const std::uint32_t position = node - leafStart;
                    if (position < pointCount &&
                        radiusCandidateAccepted(
                            x, y, z, query, sortedPointIds[position],
                            dimensions, radiusSquared, referenceMask,
                            maximumAbove, maximumBelow))
                        found = true;
                    continue;
                }
                stack[stackSize++] = node * 2U + 2U;
                stack[stackSize++] = node * 2U + 1U;
            }
        }
        matches[query] = static_cast<std::uint8_t>(found);
    }
}

__device__ double
conservativeFaceDistanceSquared(double query, double origin, double cellSize,
                                std::uint32_t boundaryCell) noexcept
{
    const double offset =
        __dmul_rn(static_cast<double>(boundaryCell), cellSize);
    const double boundary = __dadd_rn(origin, offset);
    const double raw = fabs(__dsub_rn(query, boundary));
    double scale = __dadd_ru(fabs(query), fabs(origin));
    scale = __dadd_ru(scale, fabs(offset));
    scale = __dadd_ru(scale, fabs(cellSize));
    // Grid classification and face construction each use rounded binary64
    // operations. Subtracting a deliberately wide error interval makes the
    // stopping bound one-sided: it can cause extra shell visits, never omit a
    // candidate that could be closer.
    const double error = __dmul_ru(scale, 32.0 * DoubleEpsilon);
    const double lower = __dsub_rd(raw, error);
    return lower > 0.0 ? __dmul_rd(lower, lower) : 0.0;
}

__device__ double outsideShellDistanceSquared(
    const double* queryCoordinates, const std::uint32_t* queryCell,
    const std::uint32_t* maximumCell, std::uint8_t dimensions,
    std::uint32_t shell, double cellSize, const double* origin) noexcept
{
    double lower = CUDART_INF;
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
    {
        if (queryCell[axis] > shell)
        {
            const double candidate = conservativeFaceDistanceSquared(
                queryCoordinates[axis], origin[axis], cellSize,
                queryCell[axis] - shell);
            lower = lower < candidate ? lower : candidate;
        }
        if (queryCell[axis] + shell < maximumCell[axis])
        {
            const double candidate = conservativeFaceDistanceSquared(
                queryCoordinates[axis], origin[axis], cellSize,
                queryCell[axis] + shell + 1U);
            lower = lower < candidate ? lower : candidate;
        }
    }
    return lower;
}

__global__ void knnGatherKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double cellSize, double originX, double originY,
    double originZ, std::uint32_t maximumCellX, std::uint32_t maximumCellY,
    std::uint32_t maximumCellZ, const std::uint64_t* uniqueKeys,
    const std::uint32_t* cellOffsets, const std::uint32_t* cellCounts,
    std::uint32_t cellCount, const std::uint32_t* sortedPointIds,
    const double* sortedX, const double* sortedY, const double* sortedZ,
    const float* sortedFloatX, const float* sortedFloatY,
    const float* sortedFloatZ, KnnPrefilterConstants prefilter,
    std::uint32_t shellBudget, std::uint32_t neighbors,
    std::uint32_t* outputPoints, double* outputDistances, double* outputMeans,
    double* outputNnDistances, KnnDistanceMode nnDistanceMode,
    Covariance3d* outputCovariances, std::uint8_t* outputStatus,
    const std::uint8_t* sourceMask, const std::uint8_t* referenceMask,
    std::uint32_t referenceCount)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    // Walk queries in the index's Morton order so each warp processes
    // spatially adjacent points: shells, cell lookups, and candidate loops
    // become warp-coherent. The per-query computation and its output slot
    // are unchanged — this is purely a thread-to-query assignment.
    for (std::size_t slot = thread; slot < size; slot += grid)
    {
        const std::size_t query = sortedPointIds[slot];
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        if (sourceMask && sourceMask[query] == 0U)
        {
            outputStatus[query] = KnnExact;
            for (std::uint32_t item = 0U; item < neighbors; ++item)
            {
                if (outputPoints)
                    outputPoints[row + item] = 0xffffffffU;
                if (outputDistances)
                    outputDistances[row + item] = CUDART_INF;
            }
            continue;
        }
        const double queryCoordinates[3] = {x[query], y[query],
                                            dimensions == 3 ? z[query] : 0.0};
        const double origin[3] = {originX, originY, originZ};
        const std::uint32_t queryCell[3] = {
            cellCoordinate(queryCoordinates[0], originX, cellSize),
            cellCoordinate(queryCoordinates[1], originY, cellSize),
            dimensions == 3
                ? cellCoordinate(queryCoordinates[2], originZ, cellSize)
                : 0U};
        const std::uint32_t maximumCell[3] = {
            maximumCellX, maximumCellY, dimensions == 3 ? maximumCellZ : 0U};
        const std::uint32_t pointCount = static_cast<std::uint32_t>(size);
        const std::uint32_t candidateCount =
            referenceMask ? referenceCount : pointCount;
        const std::uint32_t outputCount =
            candidateCount < neighbors ? candidateCount : neighbors;
        const std::uint32_t retainedCapacity =
            candidateCount <
                    neighbors + (candidateCount > neighbors ? 1U : 0U)
                ? candidateCount
                : neighbors + (candidateCount > neighbors ? 1U : 0U);
        std::uint32_t bestPoints[MaximumKnnNeighbors + 1U];
        double bestDistances[MaximumKnnNeighbors + 1U];
        std::uint32_t retained = 0U;
        std::uint32_t visited = 0U;
        // Register-cached copy of the worst retained entry, valid only once
        // the retained set is full. Rejecting a candidate against it is
        // exactly the insertion-position-past-capacity discard: the
        // (distance, id) precedence is a strict total order, so "worst
        // precedes candidate" if and only if the candidate would land at or
        // past capacity. This keeps the common rejected candidate out of
        // the local-memory arrays entirely.
        double worstDistance = 0.0;
        std::uint32_t worstPoint = 0U;
        // Round-up float image of worstDistance for the certified
        // prefilter; +inf until the retained set is full, which makes the
        // prefilter test vacuously false before any eviction can happen.
        float worstFloat = CUDART_INF_F;
        const float queryFloat[3] = {
            __double2float_rn(__dsub_rn(queryCoordinates[0], originX)),
            __double2float_rn(__dsub_rn(queryCoordinates[1], originY)),
            dimensions == 3
                ? __double2float_rn(__dsub_rn(queryCoordinates[2], originZ))
                : 0.0F};
        std::uint32_t maximumShell =
            queryCell[0] > maximumCell[0] - queryCell[0]
                ? queryCell[0]
                : maximumCell[0] - queryCell[0];
        const std::uint32_t yShell =
            queryCell[1] > maximumCell[1] - queryCell[1]
                ? queryCell[1]
                : maximumCell[1] - queryCell[1];
        maximumShell = maximumShell > yShell ? maximumShell : yShell;
        if (dimensions == 3)
        {
            const std::uint32_t zShell =
                queryCell[2] > maximumCell[2] - queryCell[2]
                    ? queryCell[2]
                    : maximumCell[2] - queryCell[2];
            maximumShell = maximumShell > zShell ? maximumShell : zShell;
        }
        std::uint32_t shellLimit =
            maximumShell < MaximumKnnShell ? maximumShell : MaximumKnnShell;
        // The budget bounds device effort, not correctness: an unproven
        // walk reports KnnSearchIncomplete exactly as an exhausted
        // MaximumKnnShell walk always has, and the host repairs the row.
        shellLimit = shellLimit < shellBudget ? shellLimit : shellBudget;
        bool provenComplete = false;
        for (std::uint32_t shell = 0U; shell <= shellLimit; ++shell)
        {
            const int signedShell = static_cast<int>(shell);
            const std::uint32_t leftX =
                shell < queryCell[0] ? shell : queryCell[0];
            const std::uint32_t rightXAvailable = maximumCell[0] - queryCell[0];
            const std::uint32_t rightX =
                shell < rightXAvailable ? shell : rightXAvailable;
            const std::uint32_t leftY =
                shell < queryCell[1] ? shell : queryCell[1];
            const std::uint32_t rightYAvailable = maximumCell[1] - queryCell[1];
            const std::uint32_t rightY =
                shell < rightYAvailable ? shell : rightYAvailable;
            const std::uint32_t leftZ =
                shell < queryCell[2] ? shell : queryCell[2];
            const std::uint32_t rightZAvailable = maximumCell[2] - queryCell[2];
            const std::uint32_t rightZ =
                shell < rightZAvailable ? shell : rightZAvailable;
            const int xMinimum = -static_cast<int>(leftX);
            const int xMaximum = static_cast<int>(rightX);
            const int yMinimum = -static_cast<int>(leftY);
            const int yMaximum = static_cast<int>(rightY);
            const int zMinimum = dimensions == 3 ? -static_cast<int>(leftZ) : 0;
            const int zMaximum = dimensions == 3 ? static_cast<int>(rightZ) : 0;
            for (int dzCell = zMinimum; dzCell <= zMaximum; ++dzCell)
                for (int dyCell = yMinimum; dyCell <= yMaximum; ++dyCell)
                    for (int dxCell = xMinimum; dxCell <= xMaximum; ++dxCell)
                    {
                        const int absoluteX = dxCell < 0 ? -dxCell : dxCell;
                        const int absoluteY = dyCell < 0 ? -dyCell : dyCell;
                        const int absoluteZ = dzCell < 0 ? -dzCell : dzCell;
                        const int yzMaximum =
                            absoluteY > absoluteZ ? absoluteY : absoluteZ;
                        const int cellShell =
                            absoluteX > yzMaximum ? absoluteX : yzMaximum;
                        if (cellShell != signedShell)
                            continue;
                        const long long cellX =
                            static_cast<long long>(queryCell[0]) + dxCell;
                        const long long cellY =
                            static_cast<long long>(queryCell[1]) + dyCell;
                        const long long cellZ =
                            static_cast<long long>(queryCell[2]) + dzCell;
                        if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                            cellX > maximumCellX || cellY > maximumCellY ||
                            cellZ > maximumCellZ)
                            continue;
                        const std::uint64_t key =
                            cellKey(static_cast<std::uint32_t>(cellX),
                                    static_cast<std::uint32_t>(cellY),
                                    static_cast<std::uint32_t>(cellZ));
                        const std::uint32_t cell =
                            findCell(uniqueKeys, cellCount, key);
                        if (cell == cellCount)
                            continue;
                        const std::uint32_t end =
                            cellOffsets[cell] + cellCounts[cell];
                        for (std::uint32_t item = cellOffsets[cell]; item < end;
                             ++item)
                        {
                            const std::uint32_t candidate =
                                sortedPointIds[item];
                            if (referenceMask && referenceMask[candidate] == 0U)
                                continue;
                            // Certified float prefilter: skip the double
                            // path when the margin-reduced float squared
                            // distance provably exceeds the current worst.
                            // worstFloat is +inf until the set is full.
                            if (prefilter.enabled)
                            {
                                const float dxF = __fsub_rn(queryFloat[0],
                                                            sortedFloatX[item]);
                                const float dyF = __fsub_rn(queryFloat[1],
                                                            sortedFloatY[item]);
                                float squaredF = __fadd_rn(__fmul_rn(dxF, dxF),
                                                           __fmul_rn(dyF, dyF));
                                if (dimensions == 3)
                                {
                                    const float dzF = __fsub_rn(
                                        queryFloat[2], sortedFloatZ[item]);
                                    squaredF = __fadd_rn(squaredF,
                                                         __fmul_rn(dzF, dzF));
                                }
                                const float lower = __fsub_rn(
                                    __fmul_rn(squaredF, prefilter.scale),
                                    __fmaf_rn(prefilter.linear,
                                              __fsqrt_rn(squaredF),
                                              prefilter.constant));
                                if (lower > worstFloat)
                                {
                                    ++visited;
                                    continue;
                                }
                            }
                            // The sorted copies hold bit-identical doubles,
                            // so the exactly-rounded arithmetic sees the
                            // same operands as a gather through the ids.
                            const double dx =
                                __dsub_rn(queryCoordinates[0], sortedX[item]);
                            const double dy =
                                __dsub_rn(queryCoordinates[1], sortedY[item]);
                            double distance =
                                __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                            if (dimensions == 3)
                            {
                                const double dz = __dsub_rn(queryCoordinates[2],
                                                            sortedZ[item]);
                                distance =
                                    __dadd_rn(distance, __dmul_rn(dz, dz));
                            }
                            if (retained == retainedCapacity &&
                                neighborPrecedes(worstDistance, worstPoint,
                                                 distance, candidate))
                            {
                                ++visited;
                                continue;
                            }
                            retainNeighbor(candidate, distance,
                                           retainedCapacity, retained,
                                           bestPoints, bestDistances);
                            if (retained == retainedCapacity)
                            {
                                worstDistance = bestDistances[retained - 1U];
                                worstPoint = bestPoints[retained - 1U];
                                worstFloat = __double2float_ru(worstDistance);
                            }
                            ++visited;
                        }
                    }

            if (visited == candidateCount)
            {
                provenComplete = true;
                break;
            }
            if (retained >= outputCount && outputCount != 0U)
            {
                const double outside = outsideShellDistanceSquared(
                    queryCoordinates, queryCell, maximumCell, dimensions, shell,
                    cellSize, origin);
                if (bestDistances[outputCount - 1U] < outside)
                {
                    provenComplete = true;
                    break;
                }
            }
        }

        std::uint8_t status = provenComplete ? KnnExact : KnnSearchIncomplete;
        for (std::uint32_t item = 1U; item < retained; ++item)
            if (bestDistances[item - 1U] == bestDistances[item])
                status = static_cast<std::uint8_t>(status | KnnDistanceTie);
        outputStatus[query] = static_cast<std::uint8_t>(status & c_knnStatusMask);
        const std::uint32_t available =
            outputCount < retained ? outputCount : retained;

        if (outputMeans)
        {
            double mean = 0.0;
            for (std::uint32_t neighbor = 1U; neighbor < neighbors; ++neighbor)
            {
                const double squaredDistance =
                    neighbor < available ? bestDistances[neighbor] : CUDART_INF;
                const double distance = __dsqrt_rn(squaredDistance);
                const double delta = __dsub_rn(distance, mean);
                mean = __dadd_rn(
                    mean, __ddiv_rn(delta, static_cast<double>(neighbor)));
            }
            outputMeans[query] = mean;
        }

        if (outputNnDistances)
        {
            if (nnDistanceMode == KnnDistanceMode::Kth)
            {
                const double squaredDistance =
                    neighbors - 1U < available ? bestDistances[neighbors - 1U]
                                               : CUDART_INF;
                outputNnDistances[query] = __dsqrt_rn(squaredDistance);
            }
            else
            {
                double value = 0.0;
                for (std::uint32_t neighbor = 1U; neighbor < neighbors;
                     ++neighbor)
                {
                    const double squaredDistance = neighbor < available
                                                       ? bestDistances[neighbor]
                                                       : CUDART_INF;
                    value = __dadd_rn(value, __dsqrt_rn(squaredDistance));
                }
                outputNnDistances[query] =
                    __ddiv_rn(value, static_cast<double>(neighbors - 1U));
            }
        }

        if (outputCovariances)
        {
            if (available < neighbors)
            {
                outputCovariances[query] = {CUDART_NAN, CUDART_NAN, CUDART_NAN,
                                            CUDART_NAN, CUDART_NAN, CUDART_NAN};
            }
            else
            {
                double meanX = 0.0;
                double meanY = 0.0;
                double meanZ = 0.0;
                for (std::uint32_t item = 0U; item < neighbors; ++item)
                {
                    const std::uint32_t point = bestPoints[item];
                    const double count = static_cast<double>(item + 1U);
                    meanX = __dadd_rn(
                        meanX, __ddiv_rn(__dsub_rn(x[point], meanX), count));
                    meanY = __dadd_rn(
                        meanY, __ddiv_rn(__dsub_rn(y[point], meanY), count));
                    meanZ = __dadd_rn(
                        meanZ, __ddiv_rn(__dsub_rn(z[point], meanZ), count));
                }
                Covariance3d covariance;
                for (std::uint32_t item = 0U; item < neighbors; ++item)
                {
                    const std::uint32_t point = bestPoints[item];
                    const double dx = static_cast<double>(
                        __double2float_rn(__dsub_rn(x[point], meanX)));
                    const double dy = static_cast<double>(
                        __double2float_rn(__dsub_rn(y[point], meanY)));
                    const double dz = static_cast<double>(
                        __double2float_rn(__dsub_rn(z[point], meanZ)));
                    covariance.xx = __dadd_rn(covariance.xx, __dmul_rn(dx, dx));
                    covariance.xy = __dadd_rn(covariance.xy, __dmul_rn(dx, dy));
                    covariance.xz = __dadd_rn(covariance.xz, __dmul_rn(dx, dz));
                    covariance.yy = __dadd_rn(covariance.yy, __dmul_rn(dy, dy));
                    covariance.yz = __dadd_rn(covariance.yz, __dmul_rn(dy, dz));
                    covariance.zz = __dadd_rn(covariance.zz, __dmul_rn(dz, dz));
                }
                const double divisor = static_cast<double>(neighbors - 1U);
                covariance.xx = __ddiv_rn(covariance.xx, divisor);
                covariance.xy = __ddiv_rn(covariance.xy, divisor);
                covariance.xz = __ddiv_rn(covariance.xz, divisor);
                covariance.yy = __ddiv_rn(covariance.yy, divisor);
                covariance.yz = __ddiv_rn(covariance.yz, divisor);
                covariance.zz = __ddiv_rn(covariance.zz, divisor);
                outputCovariances[query] = covariance;
            }
        }

        for (std::uint32_t item = 0U; item < available; ++item)
        {
            if (outputPoints)
                outputPoints[row + item] = bestPoints[item];
            if (outputDistances)
                outputDistances[row + item] = bestDistances[item];
        }
        for (std::uint32_t item = available; item < neighbors; ++item)
        {
            if (outputPoints)
                outputPoints[row + item] = 0xffffffffU;
            if (outputDistances)
                outputDistances[row + item] = CUDART_INF;
        }
    }
}

__global__ void projectKnnMeanDistancesKernel(
    const double* squaredDistances, std::size_t size,
    std::uint32_t rowNeighbors, std::uint32_t meanNeighbors, double* means)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const std::size_t row =
            point * static_cast<std::size_t>(rowNeighbors);
        double mean = 0.0;
        for (std::uint32_t neighbor = 1U; neighbor < meanNeighbors; ++neighbor)
        {
            const double distance =
                __dsqrt_rn(squaredDistances[row + neighbor]);
            const double delta = __dsub_rn(distance, mean);
            mean = __dadd_rn(
                mean, __ddiv_rn(delta, static_cast<double>(neighbor)));
        }
        means[point] = mean;
    }
}

__global__ void projectKnnDistanceValuesKernel(
    const double* squaredDistances, std::size_t size,
    std::uint32_t rowNeighbors, std::uint32_t distanceNeighbors,
    KnnDistanceMode mode, double* values)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const std::size_t row =
            point * static_cast<std::size_t>(rowNeighbors);
        if (mode == KnnDistanceMode::Kth)
        {
            values[point] = __dsqrt_rn(
                squaredDistances[row + distanceNeighbors - 1U]);
            continue;
        }
        double value = 0.0;
        for (std::uint32_t neighbor = 1U; neighbor < distanceNeighbors;
             ++neighbor)
            value = __dadd_rn(
                value, __dsqrt_rn(squaredDistances[row + neighbor]));
        values[point] = __ddiv_rn(
            value, static_cast<double>(distanceNeighbors - 1U));
    }
}

__device__ double repairedDistanceValue(
    const double* squaredDistances, std::uint32_t neighbors, bool meanValue)
{
    if (!meanValue)
        return __dsqrt_rn(squaredDistances[neighbors - 1U]);

    // Preserve PDAL's serial online-mean order exactly. The selection may be
    // parallel, but finalization never is.
    double mean = 0.0;
    for (std::uint32_t neighbor = 1U; neighbor < neighbors; ++neighbor)
    {
        const double distance = __dsqrt_rn(squaredDistances[neighbor]);
        const double delta = __dsub_rn(distance, mean);
        mean = __dadd_rn(
            mean, __ddiv_rn(delta, static_cast<double>(neighbor)));
    }
    return mean;
}

// Bounded selective fallback for distance-valued consumers. A full scan is
// planner-composable because it reads the resident coordinate columns and
// allocates no index state. Both kth distance and mean distance are
// independent of tie ordering.
__global__ void repairIncompleteDistanceKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, std::uint32_t neighbors, double* values,
    const std::uint32_t* queryIds, bool meanValue)
{
    const std::uint32_t query = queryIds[blockIdx.x];
    const double queryX = x[query];
    const double queryY = y[query];
    const double queryZ = dimensions == 3 ? z[query] : 0.0;
    std::uint32_t bestPoints[MaximumSelectiveRepairNeighbors];
    double bestDistances[MaximumSelectiveRepairNeighbors];
    std::uint32_t retained = 0U;
    for (std::uint32_t candidate = threadIdx.x;
         candidate < static_cast<std::uint32_t>(size);
         candidate += blockDim.x)
    {
        const double dx = __dsub_rn(queryX, x[candidate]);
        const double dy = __dsub_rn(queryY, y[candidate]);
        double distance =
            __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
        if (dimensions == 3)
        {
            const double dz = __dsub_rn(queryZ, z[candidate]);
            distance = __dadd_rn(distance, __dmul_rn(dz, dz));
        }
        retainNeighbor(candidate, distance, neighbors, retained, bestPoints,
                       bestDistances);
    }

    extern __shared__ unsigned char shared[];
    std::uint32_t* sharedPoints =
        reinterpret_cast<std::uint32_t*>(shared);
    double* sharedDistances = reinterpret_cast<double*>(
        sharedPoints + blockDim.x * neighbors);
    const std::size_t row =
        static_cast<std::size_t>(threadIdx.x) * neighbors;
    for (std::uint32_t item = 0U; item < neighbors; ++item)
    {
        sharedPoints[row + item] =
            item < retained ? bestPoints[item] : 0xffffffffU;
        sharedDistances[row + item] =
            item < retained ? bestDistances[item] : CUDART_INF;
    }
    __syncthreads();

    if (threadIdx.x == 0U)
    {
        std::uint32_t mergedPoints[MaximumSelectiveRepairNeighbors];
        double mergedDistances[MaximumSelectiveRepairNeighbors];
        std::uint32_t merged = 0U;
        const std::size_t candidates =
            static_cast<std::size_t>(blockDim.x) * neighbors;
        for (std::size_t item = 0U; item < candidates; ++item)
        {
            if (sharedPoints[item] == 0xffffffffU)
                continue;
            retainNeighbor(sharedPoints[item], sharedDistances[item], neighbors,
                           merged, mergedPoints, mergedDistances);
        }
        // Every input row contains at least `neighbors` points. The union of
        // each thread's local top-k contains the global top-k.
        values[query] =
            repairedDistanceValue(mergedDistances, neighbors, meanValue);
    }
}

__global__ void repairIncompleteDistancePartialsKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, std::uint32_t neighbors,
    const std::uint32_t* queryIds, std::uint32_t partitions,
    std::uint32_t* partialPoints, double* partialDistances)
{
    const std::uint32_t queryRow = blockIdx.x / partitions;
    const std::uint32_t partition = blockIdx.x % partitions;
    const std::uint32_t query = queryIds[queryRow];
    const double queryX = x[query];
    const double queryY = y[query];
    const double queryZ = dimensions == 3 ? z[query] : 0.0;
    std::uint32_t bestPoints[MaximumSelectiveRepairNeighbors];
    double bestDistances[MaximumSelectiveRepairNeighbors];
    std::uint32_t retained = 0U;
    const std::uint32_t first = partition * blockDim.x + threadIdx.x;
    const std::uint32_t stride = partitions * blockDim.x;
    for (std::uint32_t candidate = first;
         candidate < static_cast<std::uint32_t>(size); candidate += stride)
    {
        const double dx = __dsub_rn(queryX, x[candidate]);
        const double dy = __dsub_rn(queryY, y[candidate]);
        double distance =
            __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
        if (dimensions == 3)
        {
            const double dz = __dsub_rn(queryZ, z[candidate]);
            distance = __dadd_rn(distance, __dmul_rn(dz, dz));
        }
        retainNeighbor(candidate, distance, neighbors, retained, bestPoints,
                       bestDistances);
    }

    extern __shared__ unsigned char shared[];
    std::uint32_t* sharedPoints =
        reinterpret_cast<std::uint32_t*>(shared);
    double* sharedDistances = reinterpret_cast<double*>(
        sharedPoints + blockDim.x * neighbors);
    const std::size_t row =
        static_cast<std::size_t>(threadIdx.x) * neighbors;
    for (std::uint32_t item = 0U; item < neighbors; ++item)
    {
        sharedPoints[row + item] =
            item < retained ? bestPoints[item] : 0xffffffffU;
        sharedDistances[row + item] =
            item < retained ? bestDistances[item] : CUDART_INF;
    }
    __syncthreads();

    if (threadIdx.x == 0U)
    {
        std::uint32_t mergedPoints[MaximumSelectiveRepairNeighbors];
        double mergedDistances[MaximumSelectiveRepairNeighbors];
        std::uint32_t merged = 0U;
        const std::size_t candidates =
            static_cast<std::size_t>(blockDim.x) * neighbors;
        for (std::size_t item = 0U; item < candidates; ++item)
        {
            if (sharedPoints[item] == 0xffffffffU)
                continue;
            retainNeighbor(sharedPoints[item], sharedDistances[item], neighbors,
                           merged, mergedPoints, mergedDistances);
        }
        const std::size_t partialRow =
            (static_cast<std::size_t>(queryRow) * partitions + partition) *
            neighbors;
        for (std::uint32_t item = 0U; item < neighbors; ++item)
        {
            partialPoints[partialRow + item] =
                item < merged ? mergedPoints[item] : 0xffffffffU;
            partialDistances[partialRow + item] =
                item < merged ? mergedDistances[item] : CUDART_INF;
        }
    }
}

__global__ void mergeIncompleteDistancePartialsKernel(
    std::uint32_t neighbors, const std::uint32_t* queryIds,
    std::uint32_t queryCount, std::uint32_t partitions,
    const std::uint32_t* partialPoints, const double* partialDistances,
    double* values, bool meanValue)
{
    const std::uint32_t queryRow = blockIdx.x;
    if (queryRow >= queryCount || threadIdx.x != 0U)
        return;
    std::uint32_t mergedPoints[MaximumSelectiveRepairNeighbors];
    double mergedDistances[MaximumSelectiveRepairNeighbors];
    std::uint32_t merged = 0U;
    const std::size_t begin =
        static_cast<std::size_t>(queryRow) * partitions * neighbors;
    const std::size_t end = begin +
        static_cast<std::size_t>(partitions) * neighbors;
    for (std::size_t item = begin; item < end; ++item)
    {
        if (partialPoints[item] == 0xffffffffU)
            continue;
        retainNeighbor(partialPoints[item], partialDistances[item], neighbors,
                       merged, mergedPoints, mergedDistances);
    }
    values[queryIds[queryRow]] =
        repairedDistanceValue(mergedDistances, neighbors, meanValue);
}

__global__ void bvhKnnGatherKernel(
    const double* x, const double* y, const double* z, std::size_t size,
    std::uint8_t dimensions, double originX, double originY, double originZ,
    const std::uint32_t* sortedPointIds, const BvhBounds* internalBounds,
    std::uint32_t leafBase, std::uint32_t neighbors,
    std::uint32_t* outputPoints, double* outputDistances, double* outputMeans,
    double* outputNnDistances, KnnDistanceMode nnDistanceMode,
    Covariance3d* outputCovariances, std::uint8_t* outputStatus,
    const std::uint8_t* sourceMask, const std::uint8_t* referenceMask,
    std::uint32_t referenceCount)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double origin[3] = {originX, originY, originZ};
    const std::uint32_t pointCount = static_cast<std::uint32_t>(size);
    const std::uint32_t leafStart = leafBase - 1U;
    for (std::size_t query = thread; query < size; query += grid)
    {
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        if (sourceMask && sourceMask[query] == 0U)
        {
            outputStatus[query] = KnnExact;
            for (std::uint32_t item = 0U; item < neighbors; ++item)
            {
                if (outputPoints)
                    outputPoints[row + item] = 0xffffffffU;
                if (outputDistances)
                    outputDistances[row + item] = CUDART_INF;
            }
            continue;
        }
        const double queryCoordinates[3] = {x[query], y[query],
                                            dimensions == 3 ? z[query] : 0.0};
        const std::uint32_t candidateCount =
            referenceMask ? referenceCount : pointCount;
        const std::uint32_t outputCount =
            candidateCount < neighbors ? candidateCount : neighbors;
        const std::uint32_t retainedCapacity =
            candidateCount <
                    neighbors + (candidateCount > neighbors ? 1U : 0U)
                ? candidateCount
                : neighbors + (candidateCount > neighbors ? 1U : 0U);
        std::uint32_t bestPoints[MaximumKnnNeighbors + 1U];
        double bestDistances[MaximumKnnNeighbors + 1U];
        std::uint32_t retained = 0U;
        std::uint32_t stack[64];
        std::uint32_t stackSize = 1U;
        stack[0] = leafBase > 1U ? 0U : leafStart;
        while (stackSize != 0U)
        {
            const std::uint32_t node = stack[--stackSize];
            const BvhBounds bounds =
                heapBvhBounds(node, leafBase, pointCount, sortedPointIds,
                              internalBounds, x, y, z, dimensions, origin);
            const double lower = bvhDistanceSquared(queryCoordinates,
                                                    dimensions, origin, bounds);
            if (retained >= outputCount && outputCount != 0U &&
                lower > bestDistances[outputCount - 1U])
                continue;

            if (node >= leafStart)
            {
                const std::uint32_t position = node - leafStart;
                if (position >= pointCount)
                    continue;
                const std::uint32_t candidate = sortedPointIds[position];
                if (referenceMask && referenceMask[candidate] == 0U)
                    continue;
                const double dx = __dsub_rn(queryCoordinates[0], x[candidate]);
                const double dy = __dsub_rn(queryCoordinates[1], y[candidate]);
                double distance =
                    __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                if (dimensions == 3)
                {
                    const double dz =
                        __dsub_rn(queryCoordinates[2], z[candidate]);
                    distance = __dadd_rn(distance, __dmul_rn(dz, dz));
                }
                retainNeighbor(candidate, distance, retainedCapacity, retained,
                               bestPoints, bestDistances);
                continue;
            }

            const std::uint32_t left = node * 2U + 1U;
            const std::uint32_t right = left + 1U;
            const double leftDistance = bvhDistanceSquared(
                queryCoordinates, dimensions, origin,
                heapBvhBounds(left, leafBase, pointCount, sortedPointIds,
                              internalBounds, x, y, z, dimensions, origin));
            const double rightDistance = bvhDistanceSquared(
                queryCoordinates, dimensions, origin,
                heapBvhBounds(right, leafBase, pointCount, sortedPointIds,
                              internalBounds, x, y, z, dimensions, origin));
            if (leftDistance <= rightDistance)
            {
                stack[stackSize++] = right;
                stack[stackSize++] = left;
            }
            else
            {
                stack[stackSize++] = left;
                stack[stackSize++] = right;
            }
        }

        std::uint8_t status = KnnExact;
        for (std::uint32_t item = 1U; item < retained; ++item)
            if (bestDistances[item - 1U] == bestDistances[item])
                status = static_cast<std::uint8_t>(status | KnnDistanceTie);
        outputStatus[query] = static_cast<std::uint8_t>(status & c_knnStatusMask);
        const std::uint32_t available =
            outputCount < retained ? outputCount : retained;

        if (outputMeans)
        {
            double mean = 0.0;
            for (std::uint32_t neighbor = 1U; neighbor < neighbors; ++neighbor)
            {
                const double squaredDistance =
                    neighbor < available ? bestDistances[neighbor] : CUDART_INF;
                const double distance = __dsqrt_rn(squaredDistance);
                const double delta = __dsub_rn(distance, mean);
                mean = __dadd_rn(
                    mean, __ddiv_rn(delta, static_cast<double>(neighbor)));
            }
            outputMeans[query] = mean;
        }

        if (outputNnDistances)
        {
            if (nnDistanceMode == KnnDistanceMode::Kth)
            {
                const double squaredDistance =
                    neighbors - 1U < available ? bestDistances[neighbors - 1U]
                                               : CUDART_INF;
                outputNnDistances[query] = __dsqrt_rn(squaredDistance);
            }
            else
            {
                double value = 0.0;
                for (std::uint32_t neighbor = 1U; neighbor < neighbors;
                     ++neighbor)
                {
                    const double squaredDistance = neighbor < available
                                                       ? bestDistances[neighbor]
                                                       : CUDART_INF;
                    value = __dadd_rn(value, __dsqrt_rn(squaredDistance));
                }
                outputNnDistances[query] =
                    __ddiv_rn(value, static_cast<double>(neighbors - 1U));
            }
        }

        if (outputCovariances)
        {
            if (available < neighbors)
            {
                outputCovariances[query] = {CUDART_NAN, CUDART_NAN, CUDART_NAN,
                                            CUDART_NAN, CUDART_NAN, CUDART_NAN};
            }
            else
            {
                double meanX = 0.0;
                double meanY = 0.0;
                double meanZ = 0.0;
                for (std::uint32_t item = 0U; item < neighbors; ++item)
                {
                    const std::uint32_t point = bestPoints[item];
                    const double count = static_cast<double>(item + 1U);
                    meanX = __dadd_rn(
                        meanX, __ddiv_rn(__dsub_rn(x[point], meanX), count));
                    meanY = __dadd_rn(
                        meanY, __ddiv_rn(__dsub_rn(y[point], meanY), count));
                    meanZ = __dadd_rn(
                        meanZ, __ddiv_rn(__dsub_rn(z[point], meanZ), count));
                }
                Covariance3d covariance;
                for (std::uint32_t item = 0U; item < neighbors; ++item)
                {
                    const std::uint32_t point = bestPoints[item];
                    const double dx = static_cast<double>(
                        __double2float_rn(__dsub_rn(x[point], meanX)));
                    const double dy = static_cast<double>(
                        __double2float_rn(__dsub_rn(y[point], meanY)));
                    const double dz = static_cast<double>(
                        __double2float_rn(__dsub_rn(z[point], meanZ)));
                    covariance.xx = __dadd_rn(covariance.xx, __dmul_rn(dx, dx));
                    covariance.xy = __dadd_rn(covariance.xy, __dmul_rn(dx, dy));
                    covariance.xz = __dadd_rn(covariance.xz, __dmul_rn(dx, dz));
                    covariance.yy = __dadd_rn(covariance.yy, __dmul_rn(dy, dy));
                    covariance.yz = __dadd_rn(covariance.yz, __dmul_rn(dy, dz));
                    covariance.zz = __dadd_rn(covariance.zz, __dmul_rn(dz, dz));
                }
                const double divisor = static_cast<double>(neighbors - 1U);
                covariance.xx = __ddiv_rn(covariance.xx, divisor);
                covariance.xy = __ddiv_rn(covariance.xy, divisor);
                covariance.xz = __ddiv_rn(covariance.xz, divisor);
                covariance.yy = __ddiv_rn(covariance.yy, divisor);
                covariance.yz = __ddiv_rn(covariance.yz, divisor);
                covariance.zz = __ddiv_rn(covariance.zz, divisor);
                outputCovariances[query] = covariance;
            }
        }

        for (std::uint32_t item = 0U; item < available; ++item)
        {
            if (outputPoints)
                outputPoints[row + item] = bestPoints[item];
            if (outputDistances)
                outputDistances[row + item] = bestDistances[item];
        }
        for (std::uint32_t item = available; item < neighbors; ++item)
        {
            if (outputPoints)
                outputPoints[row + item] = 0xffffffffU;
            if (outputDistances)
                outputDistances[row + item] = CUDART_INF;
        }
    }
}

__global__ void lofKDistanceKernel(const double* squaredDistances,
                                   const std::uint8_t* status, std::size_t size,
                                   std::uint32_t neighbors, double* kDistances)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        kDistances[query] = (status[query] & KnnSearchIncomplete) == 0U
                                ? sqrt(squaredDistances[row + neighbors - 1U])
                                : CUDART_NAN;
    }
}

__global__ void lofReachabilityKernel(const std::uint32_t* pointIds,
                                      const double* squaredDistances,
                                      const double* kDistances,
                                      const std::uint8_t* status,
                                      std::size_t size, std::uint32_t neighbors,
                                      double* densities)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        if ((status[query] & KnnSearchIncomplete) != 0U)
        {
            densities[query] = CUDART_NAN;
            continue;
        }
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        double mean = 0.0;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const double distance = sqrt(squaredDistances[row + item]);
            const double kDistance = kDistances[pointIds[row + item]];
            // std::max: the second argument wins only when strictly greater.
            const double reachability =
                kDistance < distance ? distance : kDistance;
            mean += (reachability - mean) /
                    static_cast<double>(static_cast<std::size_t>(item) + 1U);
        }
        densities[query] = 1.0 / mean;
    }
}

__global__ void lofFactorKernel(const std::uint32_t* pointIds,
                                const double* densities,
                                const std::uint8_t* status, std::size_t size,
                                std::uint32_t neighbors, double* factors,
                                std::uint8_t* neighborStatus)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        if ((status[query] & KnnSearchIncomplete) != 0U)
        {
            factors[query] = CUDART_NAN;
            neighborStatus[query] = KnnSearchIncomplete;
            continue;
        }
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        const double density = densities[query];
        double mean = 0.0;
        std::uint8_t observed = KnnExact;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t neighbor = pointIds[row + item];
            observed = static_cast<std::uint8_t>(observed | status[neighbor]);
            const double ratio = densities[neighbor] / density;
            mean += (ratio - mean) /
                    static_cast<double>(static_cast<std::size_t>(item) + 1U);
        }
        factors[query] = mean;
        neighborStatus[query] = observed;
    }
}

// Second status propagation for the LOF closure: the OR of
// (status | neighborStatus) over the row's retained neighbors identifies
// rows whose factor read a density that itself read an incomplete
// k-distance. An incomplete own row mirrors its own status.
__global__ void lofClosureKernel(const std::uint32_t* pointIds,
                                 const std::uint8_t* status,
                                 const std::uint8_t* neighborStatus,
                                 std::size_t size, std::uint32_t neighbors,
                                 std::uint8_t* neighborNeighborStatus)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        if ((status[query] & KnnSearchIncomplete) != 0U)
        {
            neighborNeighborStatus[query] = KnnSearchIncomplete;
            continue;
        }
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        std::uint8_t observed = KnnExact;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t neighbor = pointIds[row + item];
            observed = static_cast<std::uint8_t>(observed | status[neighbor] |
                                                 neighborStatus[neighbor]);
        }
        neighborNeighborStatus[query] = observed;
    }
}

// Reproduces NeighborClassifierFilter's per-point vote: the k
// self-inclusive neighbors' integer values counted with the winner
// taken as the smallest value among maximal counts (std::map iteration
// order), applied only when the count strictly exceeds k/2 and the
// value changes. Purely integer arithmetic; the result is a function of
// the neighbor set alone, so it is exact wherever the gather is.
__global__ void neighborVoteKernel(const std::uint32_t* pointIds,
                                   const std::uint8_t* values,
                                   const std::uint8_t* status, std::size_t size,
                                   std::uint32_t neighbors,
                                   std::uint8_t* results)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t query = thread; query < size; query += grid)
    {
        const std::uint8_t oldValue = values[query];
        if ((status[query] & KnnSearchIncomplete) != 0U)
        {
            results[query] = oldValue;
            continue;
        }
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        std::uint32_t bestCount = 0U;
        std::uint32_t bestValue = 0U;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t candidate = values[pointIds[row + item]];
            std::uint32_t count = 0U;
            for (std::uint32_t other = 0; other < neighbors; ++other)
                count += static_cast<std::uint32_t>(
                    values[pointIds[row + other]] == candidate);
            if (count > bestCount ||
                (count == bestCount && candidate < bestValue))
            {
                bestCount = count;
                bestValue = candidate;
            }
        }
        const bool apply = 2U * bestCount > neighbors &&
                           bestValue != static_cast<std::uint32_t>(oldValue);
        results[query] =
            apply ? static_cast<std::uint8_t>(bestValue) : oldValue;
    }
}

__global__ void eigenSystemsKernel(const Covariance3d* covariances,
                                   std::size_t size, EigenSystem3d* systems,
                                   std::uint8_t* status)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        EigenSystem3d result;
        if ((status[point] & KnnSearchIncomplete) != 0U)
        {
            systems[point] = result;
            continue;
        }
        const Covariance3d& covariance = covariances[point];
        Eigen::Matrix3d matrix;
        matrix(0, 0) = covariance.xx;
        matrix(0, 1) = covariance.xy;
        matrix(0, 2) = covariance.xz;
        matrix(1, 0) = covariance.xy;
        matrix(1, 1) = covariance.yy;
        matrix(1, 2) = covariance.yz;
        matrix(2, 0) = covariance.xz;
        matrix(2, 1) = covariance.yz;
        matrix(2, 2) = covariance.zz;
        if (matrix.isZero())
        {
            status[point] =
                static_cast<std::uint8_t>(status[point] | KnnCovarianceZero);
            systems[point] = result;
            continue;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(matrix);
        if (solver.info() != Eigen::Success)
        {
            status[point] =
                static_cast<std::uint8_t>(status[point] | KnnEigenFailure);
            systems[point] = result;
            continue;
        }
        for (std::size_t eigen = 0; eigen < 3U; ++eigen)
        {
            result.values[eigen] =
                solver.eigenvalues()[static_cast<Eigen::Index>(eigen)];
            for (std::size_t axis = 0; axis < 3U; ++axis)
                result.vectors[axis * 3U + eigen] =
                    solver.eigenvectors()(static_cast<Eigen::Index>(axis),
                                          static_cast<Eigen::Index>(eigen));
        }
        systems[point] = result;
    }
}
} // unnamed namespace

void buildUniformGridDevice(SpatialIndex& index)
{
    PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA uniform-grid construction requires a device batch");
    {
        // D0271: publish the contract's status mask before any query kernel
        // of this index can run.
        const std::uint8_t mask = knnStatusMask();
        PDG_CUDA_CHECK(cudaMemcpyToSymbol(c_knnStatusMask, &mask,
                                          sizeof(mask)));
    }
    if (batch.size() == 0)
        return;

    NvtxRange range("pdg::index.uniform_grid.build");
    MemoryResource& memory = batch.memoryResource();
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const std::size_t keyBytes = batch.size() * sizeof(std::uint64_t);
    const std::size_t pointBytes = batch.size() * sizeof(std::uint32_t);
    std::unique_ptr<Allocation> inputKeys =
        memory.allocate(keyBytes, alignof(std::uint64_t));
    std::unique_ptr<Allocation> inputPointIds =
        memory.allocate(pointBytes, alignof(std::uint32_t));
    auto* keys = static_cast<std::uint64_t*>(inputKeys->data());
    auto* pointIds = static_cast<std::uint32_t*>(inputPointIds->data());
    const UniformGridConfig& config = index.m_config;
    generateCellsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2], keys, pointIds);
    PDG_CUDA_CHECK(cudaGetLastError());

    std::size_t sortTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sortTemporaryBytes, keys,
        static_cast<std::uint64_t*>(index.m_sortedKeys->data()), pointIds,
        static_cast<std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<int>(batch.size()), 0, 63, stream));
    std::unique_ptr<Allocation> sortTemporary =
        memory.allocate(sortTemporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sortTemporary->data(), sortTemporaryBytes, keys,
        static_cast<std::uint64_t*>(index.m_sortedKeys->data()), pointIds,
        static_cast<std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<int>(batch.size()), 0, 63, stream));

    std::unique_ptr<Allocation> deviceCellCount =
        memory.allocate(sizeof(int), alignof(int));
    std::size_t encodeTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, encodeTemporaryBytes,
        static_cast<const std::uint64_t*>(index.m_sortedKeys->data()),
        static_cast<std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<int*>(deviceCellCount->data()),
        static_cast<int>(batch.size()), stream));
    std::unique_ptr<Allocation> encodeTemporary =
        memory.allocate(encodeTemporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        encodeTemporary->data(), encodeTemporaryBytes,
        static_cast<const std::uint64_t*>(index.m_sortedKeys->data()),
        static_cast<std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<int*>(deviceCellCount->data()),
        static_cast<int>(batch.size()), stream));

    int hostCellCount = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&hostCellCount, deviceCellCount->data(),
                                   sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hostCellCount < 0)
        throw std::runtime_error("CUDA uniform-grid cell count is negative");
    index.m_cellCount = static_cast<std::size_t>(hostCellCount);

    std::size_t scanTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scanTemporaryBytes,
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t*>(index.m_cellOffsets->data()), hostCellCount,
        stream));
    std::unique_ptr<Allocation> scanTemporary =
        memory.allocate(scanTemporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scanTemporary->data(), scanTemporaryBytes,
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t*>(index.m_cellOffsets->data()), hostCellCount,
        stream));

    if (index.m_sortedCoordinates)
    {
        auto* sortedCoordinates =
            static_cast<double*>(index.m_sortedCoordinates->data());
        auto* sortedFloatCoordinates =
            static_cast<float*>(index.m_sortedFloatCoordinates->data());
        gatherSortedCoordinatesKernel<<<launchBlocks(batch.size()), BlockSize,
                                        0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.origin[0], config.origin[1], config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            sortedCoordinates, sortedCoordinates + batch.capacity(),
            sortedCoordinates + 2U * batch.capacity(), sortedFloatCoordinates,
            sortedFloatCoordinates + batch.capacity(),
            sortedFloatCoordinates + 2U * batch.capacity());
        PDG_CUDA_CHECK(cudaGetLastError());
    }

    if (config.backend == SpatialIndexBackend::MortonBvh &&
        index.m_bvhLeafBase > 1U)
    {
        const std::uint32_t leafBase =
            static_cast<std::uint32_t>(index.m_bvhLeafBase);
        for (std::uint32_t levelWidth = leafBase / 2U; levelWidth != 0U;
             levelWidth /= 2U)
        {
            const std::uint32_t firstNode = levelWidth - 1U;
            buildMortonBvhLevelKernel<<<launchBlocks(levelWidth), BlockSize, 0,
                                        stream>>>(
                batch.data<double>(X), batch.data<double>(Y),
                config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
                static_cast<std::uint32_t>(batch.size()), config.dimensions,
                config.origin[0], config.origin[1], config.origin[2],
                static_cast<const std::uint32_t*>(
                    index.m_sortedPointIds->data()),
                leafBase, firstNode, levelWidth,
                static_cast<BvhBounds*>(index.m_bvhBounds->data()));
            PDG_CUDA_CHECK(cudaGetLastError());
        }
    }
}

void radiusCountsDevice(const SpatialIndex& index, double radius,
                        std::uint32_t* counts)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA radius counts require a device spatial index");
    if (batch.size() == 0)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.radius_counts"
                        : "pdg::index.uniform_grid.radius_counts");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const double radiusSquared = radius * radius;
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhRadiusCountsKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), radiusSquared,
            counts, nullptr, 0.0);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    radiusCountsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        radiusSquared, counts, nullptr, 0.0);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void radiusAnyDevice(const SpatialIndex& index, double radius,
                     const std::uint8_t* sourceMask,
                     const std::uint8_t* referenceMask, double maximumAbove,
                     double maximumBelow, std::uint8_t* matches)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA radius-any query requires a device spatial index");
    if (batch.size() == 0U)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.radius_any"
                        : "pdg::index.uniform_grid.radius_any");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const double radiusSquared = radius * radius;
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhRadiusAnyKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                             stream>>>(
            batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), radiusSquared,
            sourceMask, referenceMask, maximumAbove, maximumBelow, matches);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    radiusAnyKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), config.dimensions, config.cellSize, config.origin[0],
        config.origin[1], config.origin[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        radiusSquared, sourceMask, referenceMask, maximumAbove, maximumBelow,
        matches);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void radiusScaledValuesDevice(const SpatialIndex& index, double radius,
                              double factor, double* values)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA scaled radius values require a device spatial index");
    if (batch.size() == 0U)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.radius_scaled_values"
                        : "pdg::index.uniform_grid.radius_scaled_values");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const double radiusSquared = radius * radius;
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhRadiusCountsKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), radiusSquared,
            nullptr, values, factor);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    radiusCountsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        radiusSquared, nullptr, values, factor);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnGatherDevice(const SpatialIndex& index, std::uint32_t neighbors,
                     std::uint32_t* pointIds, double* squaredDistances,
                     std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest gather requires a device spatial index");
    if (batch.size() == 0)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_gather"
                        : "pdg::index.uniform_grid.knn_gather");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhKnnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize),
                             KnnBlockSize, 0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), neighbors,
            pointIds, squaredDistances, nullptr, nullptr, KnnDistanceMode::Kth,
            nullptr, status, nullptr, nullptr,
            static_cast<std::uint32_t>(batch.size()));
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!index.m_sortedCoordinates)
        throw std::invalid_argument(
            "exact kNN queries require a kNN-configured device spatial index");
    knnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize), KnnBlockSize, 0,
                      stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2], config.maximumCell[0], config.maximumCell[1],
        config.maximumCell[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            batch.capacity(),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            2U * batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            2U * batch.capacity(),
        knnPrefilterConstants(config), knnDeviceShellBudget(), neighbors,
        pointIds, squaredDistances, nullptr, nullptr, KnnDistanceMode::Kth,
        nullptr, status, nullptr, nullptr,
        static_cast<std::uint32_t>(batch.size()));
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnGatherMaskedDevice(
    const SpatialIndex& index, std::uint32_t neighbors,
    std::uint32_t referenceCount, const std::uint8_t* sourceMask,
    const std::uint8_t* referenceMask, std::uint32_t* pointIds,
    double* squaredDistances, std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA masked k-nearest gather requires a device spatial index");
    if (batch.size() == 0U)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_gather_masked"
                        : "pdg::index.uniform_grid.knn_gather_masked");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhKnnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize),
                             KnnBlockSize, 0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), neighbors,
            pointIds, squaredDistances, nullptr, nullptr, KnnDistanceMode::Kth,
            nullptr, status, sourceMask, referenceMask, referenceCount);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!index.m_sortedCoordinates)
        throw std::invalid_argument(
            "exact masked kNN queries require a kNN-configured device "
            "spatial index");
    knnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize), KnnBlockSize, 0,
                      stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2], config.maximumCell[0], config.maximumCell[1],
        config.maximumCell[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            batch.capacity(),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            2U * batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            2U * batch.capacity(),
        knnPrefilterConstants(config), knnDeviceShellBudget(), neighbors,
        pointIds, squaredDistances, nullptr, nullptr, KnnDistanceMode::Kth,
        nullptr, status, sourceMask, referenceMask, referenceCount);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnMeanDistancesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            double* means, std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest means require a device spatial index");
    if (batch.size() == 0)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_mean_distances"
                        : "pdg::index.uniform_grid.knn_mean_distances");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhKnnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize),
                             KnnBlockSize, 0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), neighbors, nullptr,
            nullptr, means, nullptr, KnnDistanceMode::Kth, nullptr, status,
            nullptr, nullptr, static_cast<std::uint32_t>(batch.size()));
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!index.m_sortedCoordinates)
        throw std::invalid_argument(
            "exact kNN queries require a kNN-configured device spatial index");
    knnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize), KnnBlockSize, 0,
                      stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2], config.maximumCell[0], config.maximumCell[1],
        config.maximumCell[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            batch.capacity(),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            2U * batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            2U * batch.capacity(),
        knnPrefilterConstants(config), knnDeviceShellBudget(), neighbors,
        nullptr, nullptr, means, nullptr, KnnDistanceMode::Kth, nullptr,
        status, nullptr, nullptr, static_cast<std::uint32_t>(batch.size()));
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnDistanceValuesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                             KnnDistanceMode mode, double* values,
                             std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest distance requires a device spatial index");
    if (batch.size() == 0)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_distance_values"
                        : "pdg::index.uniform_grid.knn_distance_values");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhKnnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize),
                             KnnBlockSize, 0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), neighbors, nullptr,
            nullptr, nullptr, values, mode, nullptr, status, nullptr, nullptr,
            static_cast<std::uint32_t>(batch.size()));
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!index.m_sortedCoordinates)
        throw std::invalid_argument(
            "exact kNN queries require a kNN-configured device spatial index");
    knnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize), KnnBlockSize, 0,
                      stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, config.cellSize, config.origin[0], config.origin[1],
        config.origin[2], config.maximumCell[0], config.maximumCell[1],
        config.maximumCell[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            batch.capacity(),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            2U * batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            2U * batch.capacity(),
        knnPrefilterConstants(config), knnDeviceShellBudget(), neighbors,
        nullptr, nullptr, nullptr, values, mode, nullptr, status, nullptr,
        nullptr, static_cast<std::uint32_t>(batch.size()));
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectKnnMeanDistancesDevice(const SpatialIndex& index,
                                   std::uint32_t rowNeighbors,
                                   std::uint32_t meanNeighbors,
                                   const double* squaredDistances,
                                   double* means)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest mean projection requires a device spatial index");
    if (batch.size() == 0U)
        return;

    NvtxRange range("pdg::index.knn_mean_distances.project_cached");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    projectKnnMeanDistancesKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                    stream>>>(
        squaredDistances, batch.size(), rowNeighbors, meanNeighbors, means);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void projectKnnDistanceValuesDevice(const SpatialIndex& index,
                                    std::uint32_t rowNeighbors,
                                    std::uint32_t distanceNeighbors,
                                    KnnDistanceMode mode,
                                    const double* squaredDistances,
                                    double* values)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest distance projection requires a device spatial "
            "index");
    if (batch.size() == 0U)
        return;

    NvtxRange range("pdg::index.knn_distance_values.project_cached");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    projectKnnDistanceValuesKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                     stream>>>(
        squaredDistances, batch.size(), rowNeighbors, distanceNeighbors, mode,
        values);
    PDG_CUDA_CHECK(cudaGetLastError());
}

namespace
{
void repairIncompleteDistanceValuesDevice(
    const PointBatch& batch, const UniformGridConfig& config,
    std::uint32_t neighbors, double* values, const std::uint32_t* queryIds,
    std::size_t queryCount, bool parallelRepair, bool meanValue)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA selective distance repair requires a device spatial index");
    if (batch.size() == 0 || queryCount == 0U)
        return;
    if (neighbors < 2U || neighbors > MaximumSelectiveRepairNeighbors ||
        queryCount > MaximumSelectiveRepairNeighbors ||
        batch.size() < static_cast<std::size_t>(neighbors) || !values ||
        !queryIds)
        throw std::invalid_argument(
            "invalid CUDA selective distance repair request");

    NvtxRange range(meanValue
                        ? "pdg::index.knn_mean.selective_full_scan_repair"
                        : "pdg::index.knn_distance.selective_full_scan_repair");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const std::size_t sharedBytes =
        static_cast<std::size_t>(KnnBlockSize) * neighbors *
        (sizeof(std::uint32_t) + sizeof(double));
    if (parallelRepair)
    {
        const std::size_t naturalPartitions =
            (batch.size() + KnnBlockSize - 1U) / KnnBlockSize;
        const std::uint32_t partitions = static_cast<std::uint32_t>(
            (std::min)(naturalPartitions,
                       static_cast<std::size_t>(
                           MaximumSelectiveRepairPartitions)));
        const std::size_t partialCount = queryCount * partitions * neighbors;
        MemoryResource& memory = batch.memoryResource();
        std::unique_ptr<Allocation> partialPoints = memory.allocate(
            partialCount * sizeof(std::uint32_t), alignof(std::uint32_t));
        std::unique_ptr<Allocation> partialDistances = memory.allocate(
            partialCount * sizeof(double), alignof(double));
        repairIncompleteDistancePartialsKernel<<<
            static_cast<unsigned int>(queryCount * partitions), KnnBlockSize,
            sharedBytes, stream>>>(
            batch.data<double>(X), batch.data<double>(Y),
            config.dimensions == 3 ? batch.data<double>(Z) : nullptr,
            batch.size(), config.dimensions, neighbors, queryIds, partitions,
            static_cast<std::uint32_t*>(partialPoints->data()),
            static_cast<double*>(partialDistances->data()));
        PDG_CUDA_CHECK(cudaGetLastError());
        mergeIncompleteDistancePartialsKernel<<<
            static_cast<unsigned int>(queryCount), 1U, 0, stream>>>(
            neighbors, queryIds, static_cast<std::uint32_t>(queryCount),
            partitions,
            static_cast<const std::uint32_t*>(partialPoints->data()),
            static_cast<const double*>(partialDistances->data()), values,
            meanValue);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    repairIncompleteDistanceKernel<<<static_cast<unsigned int>(queryCount),
                                     KnnBlockSize, sharedBytes, stream>>>(
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr, batch.size(),
        config.dimensions, neighbors, values, queryIds, meanValue);
    PDG_CUDA_CHECK(cudaGetLastError());
}
} // unnamed namespace

void repairIncompleteKthDistanceValuesDevice(const SpatialIndex& index,
                                             std::uint32_t neighbors,
                                             double* values,
                                             const std::uint32_t* queryIds,
                                             std::size_t queryCount,
                                             bool parallelRepair)
{
    repairIncompleteDistanceValuesDevice(
        *index.m_batch, index.m_config, neighbors, values, queryIds,
        queryCount, parallelRepair, false);
}

void repairIncompleteMeanDistancesDevice(const SpatialIndex& index,
                                         std::uint32_t neighbors,
                                         double* values,
                                         const std::uint32_t* queryIds,
                                         std::size_t queryCount,
                                         bool parallelRepair)
{
    repairIncompleteDistanceValuesDevice(
        *index.m_batch, index.m_config, neighbors, values, queryIds,
        queryCount, parallelRepair, true);
}

void knnCovariancesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                          Covariance3d* covariances, std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest covariance requires a device spatial index");
    if (batch.size() == 0)
        return;

    const UniformGridConfig& config = index.m_config;
    NvtxRange range(config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_covariances"
                        : "pdg::index.uniform_grid.knn_covariances");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (config.backend == SpatialIndexBackend::MortonBvh)
    {
        bvhKnnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize),
                             KnnBlockSize, 0, stream>>>(
            batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
            batch.size(), config.dimensions, config.origin[0], config.origin[1],
            config.origin[2],
            static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
            static_cast<const BvhBounds*>(
                index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr),
            static_cast<std::uint32_t>(index.m_bvhLeafBase), neighbors, nullptr,
            nullptr, nullptr, nullptr, KnnDistanceMode::Kth, covariances,
            status, nullptr, nullptr,
            static_cast<std::uint32_t>(batch.size()));
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!index.m_sortedCoordinates)
        throw std::invalid_argument(
            "exact kNN queries require a kNN-configured device spatial index");
    knnGatherKernel<<<launchBlocks(batch.size(), KnnBlockSize), KnnBlockSize, 0,
                      stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), config.dimensions, config.cellSize, config.origin[0],
        config.origin[1], config.origin[2], config.maximumCell[0],
        config.maximumCell[1], config.maximumCell[2],
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data()),
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data()),
        static_cast<const std::uint32_t*>(index.m_cellCounts->data()),
        static_cast<std::uint32_t>(index.m_cellCount),
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            batch.capacity(),
        static_cast<const double*>(index.m_sortedCoordinates->data()) +
            2U * batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            batch.capacity(),
        static_cast<const float*>(index.m_sortedFloatCoordinates->data()) +
            2U * batch.capacity(),
        knnPrefilterConstants(config), knnDeviceShellBudget(), neighbors,
        nullptr, nullptr, nullptr, nullptr, KnnDistanceMode::Kth, covariances,
        status, nullptr, nullptr, static_cast<std::uint32_t>(batch.size()));
    PDG_CUDA_CHECK(cudaGetLastError());
}

void copyCoordinateColumnsToHost(const SpatialIndex& index, double* hostX,
                                 double* hostY, double* hostZ)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "coordinate copy requires a device spatial index");
    if (batch.size() == 0)
        return;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostX, batch.rawData(X),
                                   batch.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostY, batch.rawData(Y),
                                   batch.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostZ, batch.rawData(Z),
                                   batch.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void knnAdjacencyHostDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            std::uint32_t* hostIds,
                            double* hostSquaredDistances,
                            std::uint8_t* hostStatus)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest adjacency requires a device spatial index");
    if (batch.size() == 0)
        return;
    NvtxRange range(index.m_config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_adjacency"
                        : "pdg::index.uniform_grid.knn_adjacency");
    MemoryResource& memory = batch.memoryResource();
    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::unique_ptr<Allocation> deviceIds = memory.allocate(
        resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
    std::unique_ptr<Allocation> deviceDistances =
        memory.allocate(resultCount * sizeof(double), alignof(double));
    std::unique_ptr<Allocation> deviceStatus =
        memory.allocate(batch.size(), alignof(std::uint8_t));
    knnGatherDevice(index, neighbors,
                    static_cast<std::uint32_t*>(deviceIds->data()),
                    static_cast<double*>(deviceDistances->data()),
                    static_cast<std::uint8_t*>(deviceStatus->data()));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostIds, deviceIds->data(),
                                   resultCount * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        hostSquaredDistances, deviceDistances->data(),
        resultCount * sizeof(double), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus, deviceStatus->data(),
                                   batch.size(), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void knnRankCovariancesDevice(const SpatialIndex& index,
                              std::uint32_t neighbors,
                              Covariance3d* hostCovariances,
                              std::uint8_t* hostStatus)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest rank requires a device spatial index");
    if (batch.size() == 0)
        return;
    NvtxRange range(index.m_config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_rank_covariances"
                        : "pdg::index.uniform_grid.knn_rank_covariances");
    MemoryResource& memory = batch.memoryResource();
    std::unique_ptr<Allocation> deviceCovariances = memory.allocate(
        batch.size() * sizeof(Covariance3d), alignof(Covariance3d));
    std::unique_ptr<Allocation> deviceStatus =
        memory.allocate(batch.size(), alignof(std::uint8_t));
    knnCovariancesDevice(index, neighbors,
                         static_cast<Covariance3d*>(deviceCovariances->data()),
                         static_cast<std::uint8_t*>(deviceStatus->data()));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostCovariances, deviceCovariances->data(),
                                   batch.size() * sizeof(Covariance3d),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus, deviceStatus->data(),
                                   batch.size(), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void knnNeighborVotesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                            const std::uint8_t* values, std::uint8_t* results,
                            std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA neighbor vote requires a device spatial index");
    if (batch.size() == 0)
        return;
    NvtxRange range(index.m_config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_neighbor_votes"
                        : "pdg::index.uniform_grid.knn_neighbor_votes");
    MemoryResource& memory = batch.memoryResource();
    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::unique_ptr<Allocation> pointIds = memory.allocate(
        resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
    std::unique_ptr<Allocation> squaredDistances =
        memory.allocate(resultCount * sizeof(double), alignof(double));
    knnGatherDevice(index, neighbors,
                    static_cast<std::uint32_t*>(pointIds->data()),
                    static_cast<double*>(squaredDistances->data()), status);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    neighborVoteKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const std::uint32_t*>(pointIds->data()), values, status,
        batch.size(), neighbors, results);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnLofValuesDevice(const SpatialIndex& index, std::uint32_t neighbors,
                        double* kDistances, double* reachabilityDensities,
                        double* outlierFactors, std::uint8_t* status,
                        std::uint8_t* neighborStatus,
                        std::uint8_t* neighborNeighborStatus)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA local outlier factor requires a device spatial index");
    if (batch.size() == 0)
        return;

    NvtxRange range(index.m_config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_lof_values"
                        : "pdg::index.uniform_grid.knn_lof_values");
    MemoryResource& memory = batch.memoryResource();
    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::unique_ptr<Allocation> pointIds = memory.allocate(
        resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
    std::unique_ptr<Allocation> squaredDistances =
        memory.allocate(resultCount * sizeof(double), alignof(double));
    knnGatherDevice(index, neighbors,
                    static_cast<std::uint32_t*>(pointIds->data()),
                    static_cast<double*>(squaredDistances->data()), status);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    lofKDistanceKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const double*>(squaredDistances->data()), status,
        batch.size(), neighbors, kDistances);
    lofReachabilityKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const std::uint32_t*>(pointIds->data()),
        static_cast<const double*>(squaredDistances->data()), kDistances,
        status, batch.size(), neighbors, reachabilityDensities);
    lofFactorKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const std::uint32_t*>(pointIds->data()),
        reachabilityDensities, status, batch.size(), neighbors, outlierFactors,
        neighborStatus);
    lofClosureKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const std::uint32_t*>(pointIds->data()), status,
        neighborStatus, batch.size(), neighbors, neighborNeighborStatus);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void knnEigenSystemsDevice(const SpatialIndex& index, std::uint32_t neighbors,
                           EigenSystem3d* systems, std::uint8_t* status)
{
    const PointBatch& batch = *index.m_batch;
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA k-nearest eigensystem requires a device spatial index");
    if (batch.size() == 0)
        return;

    NvtxRange range(index.m_config.backend == SpatialIndexBackend::MortonBvh
                        ? "pdg::index.morton_bvh.knn_eigen_systems"
                        : "pdg::index.uniform_grid.knn_eigen_systems");
    MemoryResource& memory = batch.memoryResource();
    std::unique_ptr<Allocation> covariances = memory.allocate(
        batch.size() * sizeof(Covariance3d), alignof(Covariance3d));
    knnCovariancesDevice(index, neighbors,
                         static_cast<Covariance3d*>(covariances->data()),
                         status);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    eigenSystemsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        static_cast<const Covariance3d*>(covariances->data()), batch.size(),
        systems, status);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg::detail
