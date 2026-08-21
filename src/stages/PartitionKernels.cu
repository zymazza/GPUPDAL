#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
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
constexpr DimensionId ReturnNumber(StandardDimension::ReturnNumber);
constexpr DimensionId NumberOfReturns(StandardDimension::NumberOfReturns);

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

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__device__ std::uint8_t classifyReturn(std::uint8_t returnNumber,
                                       std::uint8_t numberOfReturns,
                                       std::uint8_t groups)
{
    if ((groups & ReturnFirst) && returnNumber == 1U && numberOfReturns > 1U)
        return 0U;
    if ((groups & ReturnIntermediate) && returnNumber > 1U &&
        returnNumber < numberOfReturns && numberOfReturns > 2U)
        return 1U;
    if ((groups & ReturnLast) && returnNumber == numberOfReturns &&
        numberOfReturns > 1U)
        return 2U;
    if ((groups & ReturnOnly) && numberOfReturns == 1U)
        return 3U;
    return UnselectedReturnGroup;
}

__global__ void classifyReturnsKernel(const std::uint8_t* returnNumbers,
                                      const std::uint8_t* numbersOfReturns,
                                      std::uint8_t groups, std::size_t size,
                                      std::uint8_t* keys,
                                      std::uint64_t* indices,
                                      unsigned long long* counts)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const std::uint8_t group = classifyReturn(
            returnNumbers[point], numbersOfReturns[point], groups);
        keys[point] = group;
        indices[point] = static_cast<std::uint64_t>(point);
        if (group < 4U)
            atomicAdd(counts + group, 1ULL);
    }
}

__global__ void fillDividerKernel(std::size_t size, std::uint32_t count,
                                  std::uint16_t* keys, std::uint64_t* indices)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        if (keys)
            keys[point] = static_cast<std::uint16_t>(point % count);
        indices[point] = static_cast<std::uint64_t>(point);
    }
}

__global__ void splitterCellsKernel(const double* x, const double* y,
                                    std::size_t size, double originX,
                                    double originY, double length,
                                    std::int32_t* xCells, std::int32_t* yCells)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const double dx = __dsub_rn(x[point], originX);
        const double dy = __dsub_rn(y[point], originY);
        int xCell = static_cast<int>(__ddiv_rn(dx, length));
        int yCell = static_cast<int>(__ddiv_rn(dy, length));
        if (dx < 0.0)
            --xCell;
        if (dy < 0.0)
            --yCell;
        xCells[point] = static_cast<std::int32_t>(xCell);
        yCells[point] = static_cast<std::int32_t>(yCell);
    }
}
} // unnamed namespace

ReturnsPartitionResult partitionReturnsDevice(PointBatch& batch,
                                              const ReturnsProgram& program,
                                              std::uint64_t* permutation)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA returns requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("CUDA returns exceeds the CUB item limit");
    if (!batch.size())
        return {};

    NvtxRange range("pdg::filters.returns");
    MemoryResource& memory = batch.memoryResource();
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const std::size_t keyBytes = batch.size() * sizeof(std::uint8_t);
    const std::size_t indexBytes = batch.size() * sizeof(std::uint64_t);
    std::unique_ptr<Allocation> inputKeys =
        memory.allocate(keyBytes, alignof(std::uint8_t));
    std::unique_ptr<Allocation> outputKeys =
        memory.allocate(keyBytes, alignof(std::uint8_t));
    std::unique_ptr<Allocation> inputIndices =
        memory.allocate(indexBytes, alignof(std::uint64_t));
    std::unique_ptr<Allocation> countAllocation = memory.allocate(
        4U * sizeof(unsigned long long), alignof(unsigned long long));
    auto* counts = static_cast<unsigned long long*>(countAllocation->data());
    PDG_CUDA_CHECK(
        cudaMemsetAsync(counts, 0, 4U * sizeof(unsigned long long), stream));
    classifyReturnsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<std::uint8_t>(ReturnNumber),
        batch.data<std::uint8_t>(NumberOfReturns), program.groups, batch.size(),
        static_cast<std::uint8_t*>(inputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()), counts);
    PDG_CUDA_CHECK(cudaGetLastError());

    std::size_t temporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temporaryBytes, static_cast<std::uint8_t*>(inputKeys->data()),
        static_cast<std::uint8_t*>(outputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()), permutation,
        static_cast<int>(batch.size()), 0, 3, stream));
    std::unique_ptr<Allocation> temporary =
        memory.allocate(temporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temporary->data(), temporaryBytes,
        static_cast<std::uint8_t*>(inputKeys->data()),
        static_cast<std::uint8_t*>(outputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()), permutation,
        static_cast<int>(batch.size()), 0, 3, stream));

    std::array<unsigned long long, 4> hostCounts{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostCounts.data(), counts,
                                   4U * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    ReturnsPartitionResult result;
    for (std::size_t group = 0; group < result.counts.size(); ++group)
        result.counts[group] = static_cast<std::uint64_t>(hostCounts[group]);
    return result;
}

DividerPartitionResult partitionDividerDevice(PointBatch& batch,
                                              const DividerProgram& program,
                                              std::uint64_t* permutation)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA divider requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("CUDA divider exceeds the CUB item limit");

    DividerPartitionResult result;
    result.counts.assign(program.count, 0U);
    if (!batch.size())
        return result;

    NvtxRange range("pdg::filters.divider");
    MemoryResource& memory = batch.memoryResource();
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const std::size_t count = static_cast<std::size_t>(program.count);
    if (program.mode == DividerMode::Partition)
    {
        const std::size_t limit = (batch.size() - 1U) / count + 1U;
        std::size_t remaining = batch.size();
        for (std::size_t view = 0; view < count; ++view)
        {
            const std::size_t points = (std::min)(limit, remaining);
            result.counts[view] = static_cast<std::uint64_t>(points);
            remaining -= points;
        }
        fillDividerKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
            batch.size(), program.count, nullptr, permutation);
        PDG_CUDA_CHECK(cudaGetLastError());
        return result;
    }

    for (std::size_t view = 0; view < count; ++view)
        result.counts[view] = static_cast<std::uint64_t>(
            batch.size() / count + (view < batch.size() % count));

    const std::size_t keyBytes = batch.size() * sizeof(std::uint16_t);
    const std::size_t indexBytes = batch.size() * sizeof(std::uint64_t);
    std::unique_ptr<Allocation> inputKeys =
        memory.allocate(keyBytes, alignof(std::uint16_t));
    std::unique_ptr<Allocation> outputKeys =
        memory.allocate(keyBytes, alignof(std::uint16_t));
    std::unique_ptr<Allocation> inputIndices =
        memory.allocate(indexBytes, alignof(std::uint64_t));
    fillDividerKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.size(), program.count,
        static_cast<std::uint16_t*>(inputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::size_t temporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temporaryBytes, static_cast<std::uint16_t*>(inputKeys->data()),
        static_cast<std::uint16_t*>(outputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()), permutation,
        static_cast<int>(batch.size()), 0, 10, stream));
    std::unique_ptr<Allocation> temporary =
        memory.allocate(temporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temporary->data(), temporaryBytes,
        static_cast<std::uint16_t*>(inputKeys->data()),
        static_cast<std::uint16_t*>(outputKeys->data()),
        static_cast<std::uint64_t*>(inputIndices->data()), permutation,
        static_cast<int>(batch.size()), 0, 10, stream));
    return result;
}

void computeSplitterCellsDevice(PointBatch& batch,
                                const SplitterProgram& program,
                                std::int32_t* xCells, std::int32_t* yCells)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA splitter requires a device batch");
    if (!batch.size())
        return;

    NvtxRange range("pdg::filters.splitter");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    splitterCellsKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(DimensionId(StandardDimension::X)),
        batch.data<double>(DimensionId(StandardDimension::Y)), batch.size(),
        program.originX, program.originY, program.length, xCells, yCells);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
