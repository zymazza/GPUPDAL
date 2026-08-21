#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Summary.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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
constexpr unsigned int BlockSize = 256U;

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

__device__ void insertExact(SummaryState& state, double value)
{
    ++state.count;
    state.minimum = value < state.minimum ? value : state.minimum;
    state.maximum = state.maximum < value ? value : state.maximum;

    const double delta = __dsub_rn(value, state.m1);
    const double deltaN = __ddiv_rn(delta, static_cast<double>(state.count));
    const double term1 = __dmul_rn(__dmul_rn(delta, deltaN),
                                   static_cast<double>(state.count - 1U));
    state.m1 = __dadd_rn(state.m1, deltaN);
    state.m2 = __dadd_rn(state.m2, term1);
}

__global__ void summaryKernel(const double* const* columns,
                              std::size_t pointCount,
                              std::size_t dimensionCount, SummaryState* states)
{
    const std::size_t dimension = static_cast<std::size_t>(blockIdx.x);
    if (dimension >= dimensionCount)
        return;

    __shared__ double tile[BlockSize];
    SummaryState state;
    if (threadIdx.x == 0U)
        state = states[dimension];

    const double* values = columns[dimension];
    for (std::size_t base = 0; base < pointCount; base += BlockSize)
    {
        const std::size_t point = base + threadIdx.x;
        if (point < pointCount)
            tile[threadIdx.x] = values[point];
        __syncthreads();

        if (threadIdx.x == 0U)
        {
            const std::size_t remaining = pointCount - base;
            const std::size_t limit =
                remaining < BlockSize ? remaining : BlockSize;
            for (std::size_t offset = 0; offset < limit; ++offset)
                insertExact(state, tile[offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0U)
        states[dimension] = state;
}
} // unnamed namespace

void updateSummariesDevice(PointBatch& batch,
                           std::span<const DimensionId> dimensions,
                           SummaryState* states, bool advanced)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA summaries require a device batch");
    if (advanced)
        throw std::invalid_argument(
            "advanced summaries are outside the exact CUDA envelope");
    if (dimensions.empty() || !batch.size())
        return;
    if (dimensions.size() >
        static_cast<std::size_t>((std::numeric_limits<unsigned int>::max)()))
        throw std::overflow_error("too many CUDA summary dimensions");

    NvtxRange range("pdg::filters.stats");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::vector<const double*> hostColumns;
    hostColumns.reserve(dimensions.size());
    for (DimensionId dimension : dimensions)
        hostColumns.push_back(batch.data<double>(dimension));

    std::unique_ptr<Allocation> deviceColumns = batch.memoryResource().allocate(
        dimensions.size() * sizeof(const double*), alignof(const double*));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceColumns->data(), hostColumns.data(),
                                   dimensions.size() * sizeof(const double*),
                                   cudaMemcpyHostToDevice, stream));
    summaryKernel<<<static_cast<unsigned int>(dimensions.size()), BlockSize, 0,
                    stream>>>(
        static_cast<const double* const*>(deviceColumns->data()), batch.size(),
        dimensions.size(), states);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
