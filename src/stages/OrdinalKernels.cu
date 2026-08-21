#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordinal.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;

int gridSize(std::size_t count)
{
    const std::size_t blocks = (count + BlockSize - 1U) / BlockSize;
    return static_cast<int>(std::min<std::size_t>(blocks, 65535U));
}

__global__ void evaluateHeadTailKernel(std::uint8_t* keep, std::size_t size,
                                       std::uint64_t begin,
                                       std::uint64_t inputTotal,
                                       std::uint64_t count, bool invert,
                                       bool tail)
{
    const std::uint64_t selected = count < inputTotal ? count : inputTotal;
    const std::uint64_t tailStart = inputTotal - selected;
    for (std::size_t local = blockIdx.x * blockDim.x + threadIdx.x;
         local < size; local += blockDim.x * gridDim.x)
    {
        const std::uint64_t point = begin + local;
        bool retain = tail ? point >= tailStart : point < count;
        if (invert)
            retain = !retain;
        keep[local] = static_cast<std::uint8_t>(retain);
    }
}

__global__ void evaluateDecimationKernel(std::uint8_t* keep, std::size_t size,
                                         std::uint64_t begin,
                                         std::uint64_t offset,
                                         std::uint64_t limit, double step,
                                         std::uint64_t firstSequence,
                                         std::uint64_t sequenceCount)
{
    for (std::uint64_t relative = blockIdx.x * blockDim.x + threadIdx.x;
         relative < sequenceCount; relative += blockDim.x * gridDim.x)
    {
        const std::uint64_t sequence = firstSequence + relative;
        constexpr double MaximumRounded = 0x1p63;
        const double product = static_cast<double>(sequence) * step;
        if (!isfinite(product) || product >= MaximumRounded)
            continue;
        const long long rounded = llround(product);
        if (rounded < 0)
            continue;
        const std::uint64_t distance = static_cast<std::uint64_t>(rounded);
        if (distance > UINT64_MAX - offset)
            continue;
        const std::uint64_t point = offset + distance;
        if (point >= begin && point - begin < size && point < limit)
            keep[point - begin] = 1U;
    }
}
} // unnamed namespace

void evaluateOrdinalDevice(PointBatch& batch, const OrdinalProgram& program,
                           std::uint64_t begin, std::uint64_t inputTotal,
                           std::uint64_t firstSequence,
                           std::uint64_t sequenceCount, std::uint8_t* keep)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA ordinal evaluation requires a device batch");
    if (!keep)
        throw std::invalid_argument("CUDA ordinal predicate is null");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    if (program.kind == OrdinalKind::Decimation)
    {
        PDG_CUDA_CHECK(cudaMemsetAsync(keep, 0, batch.size(), stream));
        if (sequenceCount)
        {
            evaluateDecimationKernel<<<gridSize(static_cast<std::size_t>(
                                           sequenceCount)),
                                       BlockSize, 0, stream>>>(
                keep, batch.size(), begin, program.offset, program.limit,
                program.step, firstSequence, sequenceCount);
            PDG_CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    evaluateHeadTailKernel<<<gridSize(batch.size()), BlockSize, 0, stream>>>(
        keep, batch.size(), begin, inputTotal, program.count, program.invert,
        program.kind == OrdinalKind::Tail);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
