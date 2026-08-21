#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Morton.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);

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

__device__ std::uint64_t forwardCode(std::uint32_t x, std::uint32_t y) noexcept
{
    std::uint64_t code = 0;
#pragma unroll
    for (unsigned int bit = 0; bit < 31U; ++bit)
    {
        code |= static_cast<std::uint64_t>((y >> bit) & 1U) << (2U * bit);
        code |= static_cast<std::uint64_t>((x >> bit) & 1U) << (2U * bit + 1U);
    }
    return code;
}

__device__ std::uint32_t part1By1(std::uint32_t value) noexcept
{
    value &= 0x0000ffffU;
    value = (value ^ (value << 8U)) & 0x00ff00ffU;
    value = (value ^ (value << 4U)) & 0x0f0f0f0fU;
    value = (value ^ (value << 2U)) & 0x33333333U;
    value = (value ^ (value << 1U)) & 0x55555555U;
    return value;
}

__device__ std::uint32_t reverseBits(std::uint32_t value) noexcept
{
    value = ((value >> 1U) & 0x55555555U) | ((value & 0x55555555U) << 1U);
    value = ((value >> 2U) & 0x33333333U) | ((value & 0x33333333U) << 2U);
    value = ((value >> 4U) & 0x0f0f0f0fU) | ((value & 0x0f0f0f0fU) << 4U);
    value = ((value >> 8U) & 0x00ff00ffU) | ((value & 0x00ff00ffU) << 8U);
    return (value >> 16U) | (value << 16U);
}

__global__ void mortonKeysKernel(const double* x, const double* y,
                                 std::size_t size, MortonBounds bounds,
                                 bool reverse, double cellWidth,
                                 double cellHeight, std::uint64_t* keys)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double xRange = __dsub_rn(bounds.maxX, bounds.minX);
    const double yRange = __dsub_rn(bounds.maxY, bounds.minY);
    for (std::size_t point = thread; point < size; point += grid)
    {
        if (reverse)
        {
            const auto xPosition = static_cast<std::int32_t>(
                floor(__ddiv_rn(__dsub_rn(x[point], bounds.minX), cellWidth)));
            const auto yPosition = static_cast<std::int32_t>(
                floor(__ddiv_rn(__dsub_rn(y[point], bounds.minY), cellHeight)));
            const std::uint32_t code =
                (part1By1(static_cast<std::uint32_t>(yPosition)) << 1U) +
                part1By1(static_cast<std::uint32_t>(xPosition));
            keys[point] = reverseBits(code);
        }
        else
        {
            const int xPosition = static_cast<int>(
                __dmul_rn(__ddiv_rn(__dsub_rn(x[point], bounds.minX), xRange),
                          static_cast<double>(INT_MAX)));
            const int yPosition = static_cast<int>(
                __dmul_rn(__ddiv_rn(__dsub_rn(y[point], bounds.minY), yRange),
                          static_cast<double>(INT_MAX)));
            keys[point] = forwardCode(static_cast<std::uint32_t>(xPosition),
                                      static_cast<std::uint32_t>(yPosition));
        }
    }
}
} // unnamed namespace

void generateMortonKeysDevice(PointBatch& batch, const MortonProgram& program,
                              std::uint64_t* keys)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA Morton keys require a device batch");
    NvtxRange range("pdg::filters.mortonorder.keys");
    const std::size_t naturalBlocks =
        (batch.size() - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    const unsigned int blocks = static_cast<unsigned int>(
        (std::min)(naturalBlocks, static_cast<std::size_t>(65535)));
    const std::int32_t cell =
        static_cast<std::int32_t>(std::sqrt(batch.size()));
    const double xRange = program.bounds.maxX - program.bounds.minX;
    const double yRange = program.bounds.maxY - program.bounds.minY;
    const double cellWidth = xRange / static_cast<double>(cell);
    const double cellHeight = yRange / static_cast<double>(cell);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    mortonKeysKernel<<<blocks, BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.size(),
        program.bounds, program.reverse, cellWidth, cellHeight, keys);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
