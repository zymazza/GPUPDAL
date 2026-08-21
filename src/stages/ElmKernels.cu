#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace pdg
{
namespace
{
constexpr int BlockSize = 256;
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId Classification(StandardDimension::Classification);

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

struct DeviceFrame
{
    double minimumX;
    double minimumY;
    std::size_t rows;
    std::size_t columns;
};

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

template <typename T>
std::unique_ptr<Allocation> allocate(PointBatch& batch, std::size_t count)
{
    if (count && count > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
        throw std::overflow_error("CUDA ELM allocation size overflows");
    return batch.memoryResource().allocate(count * sizeof(T), alignof(T));
}

template <typename T> T* pointer(const std::unique_ptr<Allocation>& allocation)
{
    return static_cast<T*>(allocation->data());
}

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* message)
{
    if (left > (std::numeric_limits<std::size_t>::max)() - right)
        throw std::overflow_error(message);
    return left + right;
}

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* message)
{
    if (left && right > (std::numeric_limits<std::size_t>::max)() / left)
        throw std::overflow_error(message);
    return left * right;
}

__device__ std::size_t threadIndex()
{
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ std::size_t gridStride()
{
    return static_cast<std::size_t>(blockDim.x) * gridDim.x;
}

__global__ void validateFiniteKernel(const double* x, const double* y,
                                     const double* z, std::size_t pointCount,
                                     int* invalid)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
        if (!isfinite(x[point]) || !isfinite(y[point]) || !isfinite(z[point]))
            atomicExch(invalid, 1);
}

__device__ bool cellFor(double x, double y, DeviceFrame frame, double cell,
                        std::uint32_t& key)
{
    const double columnValue =
        __ddiv_rn(floor(__dsub_rn(x, frame.minimumX)), cell);
    const double rowValue =
        __ddiv_rn(floor(__dsub_rn(y, frame.minimumY)), cell);
    if (!isfinite(columnValue) || !isfinite(rowValue) || columnValue < 0.0 ||
        rowValue < 0.0 || columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        return false;
    const std::size_t index =
        static_cast<std::size_t>(columnValue) * frame.rows +
        static_cast<std::size_t>(rowValue);
    if (index > 0xffffffffU)
        return false;
    key = static_cast<std::uint32_t>(index);
    return true;
}

__global__ void initializeSortKernel(const double* x, const double* y,
                                     const double* z, std::size_t pointCount,
                                     DeviceFrame frame, double cell,
                                     double* zKeys, std::uint32_t* pointIds,
                                     int* invalid)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::uint32_t key = 0U;
        if (!cellFor(x[point], y[point], frame, cell, key))
            atomicExch(invalid, 1);
        // std::less<double> treats the two signed zeros as equivalent and
        // multimap retains their insertion order. Canonicalizing the radix
        // key gives CUB the same equivalence class.
        zKeys[point] = z[point] == 0.0 ? 0.0 : z[point];
        pointIds[point] = static_cast<std::uint32_t>(point);
    }
}

__global__ void cellKeyKernel(const double* x, const double* y,
                              const std::uint32_t* pointIds,
                              std::size_t pointCount, DeviceFrame frame,
                              double cell, std::uint32_t* keys, int* invalid)
{
    for (std::size_t sorted = threadIndex(); sorted < pointCount;
         sorted += gridStride())
    {
        const std::uint32_t point = pointIds[sorted];
        std::uint32_t key = 0U;
        if (!cellFor(x[point], y[point], frame, cell, key))
            atomicExch(invalid, 1);
        keys[sorted] = key;
    }
}

__global__ void segmentBoundsKernel(const std::uint32_t* keys,
                                    std::size_t pointCount,
                                    std::uint32_t* starts, std::uint32_t* ends)
{
    for (std::size_t sorted = threadIndex(); sorted < pointCount;
         sorted += gridStride())
    {
        const std::uint32_t key = keys[sorted];
        if (sorted == 0U || keys[sorted - 1U] != key)
            starts[key] = static_cast<std::uint32_t>(sorted);
        if (sorted + 1U == pointCount || keys[sorted + 1U] != key)
            ends[key] = static_cast<std::uint32_t>(sorted + 1U);
    }
}

__global__ void classifyKernel(const double* z, const std::uint32_t* pointIds,
                               const std::uint32_t* starts,
                               const std::uint32_t* ends, std::size_t cellCount,
                               double threshold, std::uint8_t outputClass,
                               std::uint8_t* classification,
                               unsigned long long* count)
{
    for (std::size_t cell = threadIndex(); cell < cellCount;
         cell += gridStride())
    {
        const std::uint32_t begin = starts[cell];
        const std::uint32_t end = ends[cell];
        if (begin == 0xffffffffU || end <= begin + 1U)
            continue;
        for (std::uint32_t sorted = begin; sorted + 1U < end; ++sorted)
        {
            const std::uint32_t point = pointIds[sorted];
            const std::uint32_t next = pointIds[sorted + 1U];
            if (fabs(__dsub_rn(z[point], z[next])) < threshold)
                break;
            classification[point] = outputClass;
            atomicAdd(count, 1ULL);
        }
    }
}

void throwIfInvalid(PointBatch& batch,
                    const std::unique_ptr<Allocation>& invalid,
                    const char* message)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    int host = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&host, invalid->data(), sizeof(host),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host)
        throw std::invalid_argument(message);
}

DeviceFrame reduceFrame(PointBatch& batch, double cell)
{
    const int count = static_cast<int>(batch.size());
    std::unique_ptr<Allocation> results = allocate<double>(batch, 4U);
    auto* output = pointer<double>(results);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::size_t minimumBytes = 0U;
    std::size_t maximumBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(
        nullptr, minimumBytes, batch.data<double>(X), output, count, stream));
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(nullptr, maximumBytes,
                                          batch.data<double>(X), output + 1U,
                                          count, stream));
    const std::size_t temporaryBytes = (std::max)(minimumBytes, maximumBytes);
    std::unique_ptr<Allocation> temporary = batch.memoryResource().allocate(
        temporaryBytes, alignof(std::max_align_t));
    minimumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(temporary->data(), minimumBytes,
                                          batch.data<double>(X), output, count,
                                          stream));
    maximumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(temporary->data(), maximumBytes,
                                          batch.data<double>(X), output + 1U,
                                          count, stream));
    minimumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(temporary->data(), minimumBytes,
                                          batch.data<double>(Y), output + 2U,
                                          count, stream));
    maximumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(temporary->data(), maximumBytes,
                                          batch.data<double>(Y), output + 3U,
                                          count, stream));
    double host[4]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(host, output, sizeof(host),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    const double columnsValue = (host[1] - host[0]) / cell + 1.0;
    const double rowsValue = (host[3] - host[2]) / cell + 1.0;
    if (!std::isfinite(columnsValue) || !std::isfinite(rowsValue) ||
        columnsValue < 1.0 || rowsValue < 1.0 ||
        columnsValue > static_cast<double>(ElmExactDeviceMaximumGridCells) ||
        rowsValue > static_cast<double>(ElmExactDeviceMaximumGridCells))
        throw std::invalid_argument("CUDA ELM grid dimensions are invalid");
    const DeviceFrame frame{host[0], host[2],
                            static_cast<std::size_t>(rowsValue),
                            static_cast<std::size_t>(columnsValue)};
    if (frame.rows > ElmExactDeviceMaximumGridCells / frame.columns)
        throw std::invalid_argument(
            "CUDA ELM grid exceeds the exact device envelope");
    return frame;
}
} // unnamed namespace

std::size_t elmExactDeviceScratchBytes(std::size_t pointCount)
{
    if (!pointCount)
        return 0U;
    if (pointCount >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA ELM point count exceeds the exact device envelope");
    const int count = static_cast<int>(pointCount);
    const cudaStream_t stream{};

    // Temporary-storage queries do not dereference their input/output
    // iterators. Real host objects provide distinct, correctly typed pointer
    // identities without allocating device memory during planner preflight.
    double doubleInput = 0.0;
    double doubleOutput = 0.0;
    std::uint32_t idInput = 0U;
    std::uint32_t idOutput = 0U;
    std::uint32_t keyInput = 0U;
    std::uint32_t keyOutput = 0U;

    std::size_t reduceMinimumBytes = 0U;
    std::size_t reduceMaximumBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(nullptr, reduceMinimumBytes,
                                          &doubleInput, &doubleOutput, count,
                                          stream));
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(nullptr, reduceMaximumBytes,
                                          &doubleInput, &doubleOutput, count,
                                          stream));

    cub::DoubleBuffer<double> zKeys(&doubleInput, &doubleOutput);
    cub::DoubleBuffer<std::uint32_t> zIds(&idInput, &idOutput);
    std::size_t zSortBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, zSortBytes, zKeys, zIds, count, 0,
        static_cast<int>(sizeof(double) * 8U), stream));

    cub::DoubleBuffer<std::uint32_t> cellKeys(&keyInput, &keyOutput);
    cub::DoubleBuffer<std::uint32_t> cellIds(&idInput, &idOutput);
    std::size_t cellSortBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, cellSortBytes, cellKeys, cellIds, count, 0,
        static_cast<int>(sizeof(std::uint32_t) * 8U), stream));

    const std::size_t invalidBytes = sizeof(int);
    const std::size_t reductionPeak =
        checkedAdd(checkedAdd(invalidBytes, 4U * sizeof(double),
                              "ELM reduction scratch estimate overflows"),
                   (std::max)(reduceMinimumBytes, reduceMaximumBytes),
                   "ELM reduction scratch estimate overflows");
    const std::size_t zSortPeak = checkedAdd(
        checkedAdd(
            invalidBytes,
            checkedProduct(pointCount,
                           2U * sizeof(double) + 2U * sizeof(std::uint32_t),
                           "ELM Z-sort scratch estimate overflows"),
            "ELM Z-sort scratch estimate overflows"),
        zSortBytes, "ELM Z-sort scratch estimate overflows");
    const std::size_t cellSortPeak = checkedAdd(
        checkedAdd(invalidBytes,
                   checkedProduct(pointCount, 4U * sizeof(std::uint32_t),
                                  "ELM cell-sort scratch estimate overflows"),
                   "ELM cell-sort scratch estimate overflows"),
        cellSortBytes, "ELM cell-sort scratch estimate overflows");
    const std::size_t segmentPeak = checkedAdd(
        checkedAdd(invalidBytes,
                   checkedProduct(pointCount, 4U * sizeof(std::uint32_t),
                                  "ELM segment scratch estimate overflows"),
                   "ELM segment scratch estimate overflows"),
        ElmExactDeviceMaximumGridCells * 2U * sizeof(std::uint32_t) +
            sizeof(unsigned long long),
        "ELM segment scratch estimate overflows");
    return checkedAdd(
        (std::max)({reductionPeak, zSortPeak, cellSortPeak, segmentPeak}),
        ElmExactDeviceAllocatorSlackBytes,
        "ELM allocator scratch estimate overflows");
}

ElmResult classifyElmDevice(PointBatch& batch, const ElmProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA ELM requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA ELM point count exceeds the exact device envelope");
    if (!elmProgramWithinExactDeviceEnvelope(program))
        throw std::invalid_argument(
            "CUDA ELM program exceeds the exact device envelope");

    NvtxRange range("pdg::filters.elm");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::unique_ptr<Allocation> invalid = allocate<int>(batch, 1U);
    PDG_CUDA_CHECK(cudaMemsetAsync(invalid->data(), 0, sizeof(int), stream));
    validateFiniteKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    throwIfInvalid(batch, invalid,
                   "CUDA ELM requires finite logical-double XYZ");

    const DeviceFrame frame = reduceFrame(batch, program.cell);
    const std::size_t cellCount = frame.rows * frame.columns;
    const int count = static_cast<int>(batch.size());

    std::unique_ptr<Allocation> zKeysA = allocate<double>(batch, batch.size());
    std::unique_ptr<Allocation> zKeysB = allocate<double>(batch, batch.size());
    std::unique_ptr<Allocation> idsA =
        allocate<std::uint32_t>(batch, batch.size());
    std::unique_ptr<Allocation> idsB =
        allocate<std::uint32_t>(batch, batch.size());
    initializeSortKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), frame, program.cell, pointer<double>(zKeysA),
        pointer<std::uint32_t>(idsA), pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    throwIfInvalid(batch, invalid,
                   "CUDA ELM point lies outside its grid frame");

    cub::DoubleBuffer<double> zKeys(pointer<double>(zKeysA),
                                    pointer<double>(zKeysB));
    cub::DoubleBuffer<std::uint32_t> zIds(pointer<std::uint32_t>(idsA),
                                          pointer<std::uint32_t>(idsB));
    std::size_t zSortBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, zSortBytes, zKeys, zIds, count, 0,
        static_cast<int>(sizeof(double) * 8U), stream));
    std::unique_ptr<Allocation> zSortTemporary =
        batch.memoryResource().allocate(zSortBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        zSortTemporary->data(), zSortBytes, zKeys, zIds, count, 0,
        static_cast<int>(sizeof(double) * 8U), stream));

    const bool idsInA = zIds.Current() == pointer<std::uint32_t>(idsA);
    zSortTemporary.reset();
    zKeysA.reset();
    zKeysB.reset();
    std::unique_ptr<Allocation> cellKeysA =
        allocate<std::uint32_t>(batch, batch.size());
    std::unique_ptr<Allocation> cellKeysB =
        allocate<std::uint32_t>(batch, batch.size());
    std::uint32_t* zSortedIds =
        idsInA ? pointer<std::uint32_t>(idsA) : pointer<std::uint32_t>(idsB);
    std::uint32_t* alternateIds =
        idsInA ? pointer<std::uint32_t>(idsB) : pointer<std::uint32_t>(idsA);
    cellKeyKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), zSortedIds, batch.size(),
        frame, program.cell, pointer<std::uint32_t>(cellKeysA),
        pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    throwIfInvalid(batch, invalid,
                   "CUDA ELM point lies outside its grid frame");

    cub::DoubleBuffer<std::uint32_t> cellKeys(
        pointer<std::uint32_t>(cellKeysA), pointer<std::uint32_t>(cellKeysB));
    cub::DoubleBuffer<std::uint32_t> cellIds(zSortedIds, alternateIds);
    std::size_t cellSortBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, cellSortBytes, cellKeys, cellIds, count, 0,
        static_cast<int>(sizeof(std::uint32_t) * 8U), stream));
    std::unique_ptr<Allocation> cellSortTemporary =
        batch.memoryResource().allocate(cellSortBytes,
                                        alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        cellSortTemporary->data(), cellSortBytes, cellKeys, cellIds, count, 0,
        static_cast<int>(sizeof(std::uint32_t) * 8U), stream));
    cellSortTemporary.reset();

    std::unique_ptr<Allocation> starts =
        allocate<std::uint32_t>(batch, cellCount);
    std::unique_ptr<Allocation> ends =
        allocate<std::uint32_t>(batch, cellCount);
    PDG_CUDA_CHECK(cudaMemsetAsync(starts->data(), 0xff,
                                   cellCount * sizeof(std::uint32_t), stream));
    PDG_CUDA_CHECK(cudaMemsetAsync(ends->data(), 0,
                                   cellCount * sizeof(std::uint32_t), stream));
    segmentBoundsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        cellKeys.Current(), batch.size(), pointer<std::uint32_t>(starts),
        pointer<std::uint32_t>(ends));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> classified =
        allocate<unsigned long long>(batch, 1U);
    PDG_CUDA_CHECK(cudaMemsetAsync(classified->data(), 0,
                                   sizeof(unsigned long long), stream));
    classifyKernel<<<launchBlocks(cellCount), BlockSize, 0, stream>>>(
        batch.data<double>(Z), cellIds.Current(),
        pointer<std::uint32_t>(starts), pointer<std::uint32_t>(ends), cellCount,
        program.threshold, program.classification,
        batch.data<std::uint8_t>(Classification),
        pointer<unsigned long long>(classified));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned long long hostClassified = 0ULL;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&hostClassified, classified->data(),
                                   sizeof(hostClassified),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {frame.rows, frame.columns,
            static_cast<std::size_t>(hostClassified)};
}

} // namespace pdg
