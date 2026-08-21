#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;

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

struct DeviceBin
{
    std::uint64_t count = 0;
    std::uint64_t firstIndex = 0;
};

struct MergeBins
{
    __host__ __device__ DeviceBin operator()(const DeviceBin& first,
                                             const DeviceBin& second) const
    {
        DeviceBin result;
        result.count = first.count + second.count;
        result.firstIndex = first.firstIndex < second.firstIndex
                                ? first.firstIndex
                                : second.firstIndex;
        return result;
    }
};

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__global__ void initializeIndicesKernel(std::uint64_t* indices,
                                        std::size_t size)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
        indices[point] = static_cast<std::uint64_t>(point);
}

__global__ void finiteTargetKernel(const double* values, std::size_t size,
                                   int* invalid)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
        if (!isfinite(values[point]))
            atomicExch(invalid, 1);
}

__global__ void initializeBinsKernel(const std::uint64_t* sortedIndices,
                                     std::size_t size,
                                     std::uint64_t indexOffset, DeviceBin* bins)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
        bins[point] = {1, indexOffset + sortedIndices[point]};
}

__global__ void restoreFirstValueKernel(double* uniqueValues,
                                        const DeviceBin* bins, std::size_t size,
                                        std::uint64_t indexOffset,
                                        const double* originalValues)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t bin = thread; bin < size; bin += grid)
        uniqueValues[bin] = originalValues[bins[bin].firstIndex - indexOffset];
}
} // unnamed namespace

std::vector<HistogramBin>
selectedHistogramDevice(PointBatch& batch, DimensionId target,
                        const PredicateProgram& predicate,
                        std::uint64_t indexOffset)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA histogram requires a device batch");
    if (batch.columnInfo(target).physicalType != DimensionType::Double)
        throw std::invalid_argument(
            "CUDA histogram target must be a Double column");
    if (!predicateSupportsExactDevice(batch, predicate))
        throw std::invalid_argument(
            "histogram predicate is outside the exact CUDA envelope");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::overflow_error("CUDA histogram exceeds the CUB item limit");
    if (!batch.size())
        return {};

    NvtxRange range("pdg::filters.expressionstats::histogram");
    const int count = static_cast<int>(batch.size());
    const std::size_t size = batch.size();
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const double* source = batch.data<double>(target);

    std::unique_ptr<Allocation> invalidAllocation =
        batch.memoryResource().allocate(sizeof(int), alignof(int));
    auto* invalid = static_cast<int*>(invalidAllocation->data());
    PDG_CUDA_CHECK(cudaMemsetAsync(invalid, 0, sizeof(int), stream));
    finiteTargetKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        source, size, invalid);
    PDG_CUDA_CHECK(cudaGetLastError());
    int hostInvalid = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&hostInvalid, invalid, sizeof(hostInvalid),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hostInvalid)
        throw std::invalid_argument(
            "non-finite histogram targets are outside the exact CUDA "
            "envelope");

    std::unique_ptr<Allocation> keepAllocation =
        batch.memoryResource().allocate(size, alignof(std::uint8_t));
    auto* keep = static_cast<std::uint8_t*>(keepAllocation->data());
    evaluatePredicate(batch, predicate, keep);

    const std::size_t keyBytes = size * sizeof(double);
    const std::size_t indexBytes = size * sizeof(std::uint64_t);
    std::unique_ptr<Allocation> inputIndicesAllocation =
        batch.memoryResource().allocate(indexBytes, alignof(std::uint64_t));
    std::unique_ptr<Allocation> selectedKeysAllocation =
        batch.memoryResource().allocate(keyBytes, alignof(double));
    std::unique_ptr<Allocation> selectedIndicesAllocation =
        batch.memoryResource().allocate(indexBytes, alignof(std::uint64_t));
    auto* inputIndices =
        static_cast<std::uint64_t*>(inputIndicesAllocation->data());
    auto* selectedKeys = static_cast<double*>(selectedKeysAllocation->data());
    auto* selectedIndices =
        static_cast<std::uint64_t*>(selectedIndicesAllocation->data());
    initializeIndicesKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        inputIndices, size);
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> selectedCountAllocation =
        batch.memoryResource().allocate(sizeof(int), alignof(int));
    auto* selectedCount = static_cast<int*>(selectedCountAllocation->data());
    std::size_t selectKeyTemporaryBytes = 0;
    std::size_t selectIndexTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, selectKeyTemporaryBytes,
                                              source, keep, selectedKeys,
                                              selectedCount, count, stream));
    PDG_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, selectIndexTemporaryBytes, inputIndices, keep, selectedIndices,
        selectedCount, count, stream));
    const std::size_t selectTemporaryBytes =
        (std::max)(selectKeyTemporaryBytes, selectIndexTemporaryBytes);
    std::unique_ptr<Allocation> selectTemporary =
        batch.memoryResource().allocate(selectTemporaryBytes,
                                        alignof(std::max_align_t));
    selectKeyTemporaryBytes = selectTemporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceSelect::Flagged(
        selectTemporary->data(), selectKeyTemporaryBytes, source, keep,
        selectedKeys, selectedCount, count, stream));
    selectIndexTemporaryBytes = selectTemporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceSelect::Flagged(
        selectTemporary->data(), selectIndexTemporaryBytes, inputIndices, keep,
        selectedIndices, selectedCount, count, stream));
    int selected = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&selected, selectedCount, sizeof(selected),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (selected < 0 || selected > count)
        throw std::runtime_error("CUDA histogram returned an invalid count");
    if (!selected)
        return {};

    std::unique_ptr<Allocation> sortedKeysAllocation =
        batch.memoryResource().allocate(keyBytes, alignof(double));
    std::unique_ptr<Allocation> sortedIndicesAllocation =
        batch.memoryResource().allocate(indexBytes, alignof(std::uint64_t));
    auto* sortedKeys = static_cast<double*>(sortedKeysAllocation->data());
    auto* sortedIndices =
        static_cast<std::uint64_t*>(sortedIndicesAllocation->data());
    std::size_t sortTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sortTemporaryBytes, selectedKeys, sortedKeys, selectedIndices,
        sortedIndices, selected, 0, 64, stream));
    std::unique_ptr<Allocation> sortTemporary = batch.memoryResource().allocate(
        sortTemporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sortTemporary->data(), sortTemporaryBytes, selectedKeys, sortedKeys,
        selectedIndices, sortedIndices, selected, 0, 64, stream));

    const std::size_t binBytes = size * sizeof(DeviceBin);
    std::unique_ptr<Allocation> inputBinsAllocation =
        batch.memoryResource().allocate(binBytes, alignof(DeviceBin));
    std::unique_ptr<Allocation> uniqueKeysAllocation =
        batch.memoryResource().allocate(keyBytes, alignof(double));
    std::unique_ptr<Allocation> outputBinsAllocation =
        batch.memoryResource().allocate(binBytes, alignof(DeviceBin));
    std::unique_ptr<Allocation> runCountAllocation =
        batch.memoryResource().allocate(sizeof(int), alignof(int));
    auto* inputBins = static_cast<DeviceBin*>(inputBinsAllocation->data());
    auto* uniqueKeys = static_cast<double*>(uniqueKeysAllocation->data());
    auto* outputBins = static_cast<DeviceBin*>(outputBinsAllocation->data());
    auto* runCount = static_cast<int*>(runCountAllocation->data());
    initializeBinsKernel<<<launchBlocks(static_cast<std::size_t>(selected)),
                           BlockSize, 0, stream>>>(
        sortedIndices, static_cast<std::size_t>(selected), indexOffset,
        inputBins);
    PDG_CUDA_CHECK(cudaGetLastError());

    std::size_t reduceTemporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, reduceTemporaryBytes, sortedKeys, uniqueKeys, inputBins,
        outputBins, runCount, MergeBins{}, selected, stream));
    std::unique_ptr<Allocation> reduceTemporary =
        batch.memoryResource().allocate(reduceTemporaryBytes,
                                        alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        reduceTemporary->data(), reduceTemporaryBytes, sortedKeys, uniqueKeys,
        inputBins, outputBins, runCount, MergeBins{}, selected, stream));
    int runs = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&runs, runCount, sizeof(runs),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (runs <= 0 || runs > selected)
        throw std::runtime_error(
            "CUDA histogram returned an invalid run count");

    restoreFirstValueKernel<<<launchBlocks(static_cast<std::size_t>(runs)),
                              BlockSize, 0, stream>>>(
        uniqueKeys, outputBins, static_cast<std::size_t>(runs), indexOffset,
        source);
    PDG_CUDA_CHECK(cudaGetLastError());
    std::vector<double> hostKeys(static_cast<std::size_t>(runs));
    std::vector<DeviceBin> hostBins(static_cast<std::size_t>(runs));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostKeys.data(), uniqueKeys,
                                   hostKeys.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostBins.data(), outputBins,
                                   hostBins.size() * sizeof(DeviceBin),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<HistogramBin> result;
    result.reserve(hostKeys.size());
    for (std::size_t bin = 0; bin < hostKeys.size(); ++bin)
        result.push_back(
            {hostKeys[bin], hostBins[bin].count, hostBins[bin].firstIndex});
    return result;
}

} // namespace pdg
