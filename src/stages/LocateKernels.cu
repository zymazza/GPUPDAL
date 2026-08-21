#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Locate.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr std::size_t MaximumBlocks = 1024;

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

__device__ double loadPhysical(const void* data, DimensionType type,
                               std::size_t index)
{
    switch (type)
    {
    case DimensionType::Signed8:
        return static_cast<const std::int8_t*>(data)[index];
    case DimensionType::Signed16:
        return static_cast<const std::int16_t*>(data)[index];
    case DimensionType::Signed32:
        return static_cast<const std::int32_t*>(data)[index];
    case DimensionType::Signed64:
        return static_cast<double>(
            static_cast<const std::int64_t*>(data)[index]);
    case DimensionType::Unsigned8:
        return static_cast<const std::uint8_t*>(data)[index];
    case DimensionType::Unsigned16:
        return static_cast<const std::uint16_t*>(data)[index];
    case DimensionType::Unsigned32:
        return static_cast<const std::uint32_t*>(data)[index];
    case DimensionType::Unsigned64:
        return static_cast<double>(
            static_cast<const std::uint64_t*>(data)[index]);
    case DimensionType::Float:
        return static_cast<const float*>(data)[index];
    case DimensionType::Double:
        return static_cast<const double*>(data)[index];
    case DimensionType::None:
        return 0.0;
    }
    return 0.0;
}

__device__ bool improves(LocateKind kind, double candidate, double current)
{
    return kind == LocateKind::Minimum ? candidate < current
                                       : candidate > current;
}

__device__ LocateResult merge(LocateKind kind, LocateResult first,
                              LocateResult second)
{
    if (!first.hasPoints)
        return second;
    if (!second.hasPoints)
        return first;
    if (first.comparable != second.comparable)
        return first.comparable ? first : second;
    if (!first.comparable)
        return first.index <= second.index ? first : second;
    if (improves(kind, second.value, first.value))
        return second;
    if (improves(kind, first.value, second.value))
        return first;
    return first.index <= second.index ? first : second;
}

__device__ LocateResult emptyResult(LocateKind kind)
{
    LocateResult result;
    result.value = kind == LocateKind::Minimum ? DBL_MAX : -DBL_MAX;
    return result;
}

__global__ void locateBlocksKernel(const void* source, DimensionType type,
                                   std::size_t size, std::uint64_t indexOffset,
                                   LocateKind kind, int coordinateAxis,
                                   double coordinateScale,
                                   double coordinateOffset,
                                   LocateResult* blockResults)
{
    LocateResult local = emptyResult(kind);
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += blockDim.x * gridDim.x)
    {
        if (!local.hasPoints)
        {
            local.hasPoints = 1;
            local.index = indexOffset + static_cast<std::uint64_t>(point);
        }
        double value = loadPhysical(source, type, point);
        if (coordinateAxis >= 0 && type == DimensionType::Signed32)
            value = __dadd_rn(
                __dmul_rn(static_cast<double>(
                              static_cast<const std::int32_t*>(source)[point]),
                          coordinateScale),
                coordinateOffset);
        if (improves(kind, value, local.value))
        {
            local.value = value;
            local.index = indexOffset + static_cast<std::uint64_t>(point);
            local.comparable = 1;
        }
    }

    __shared__ LocateResult shared[BlockSize];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BlockSize / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] =
                merge(kind, shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x)
        blockResults[blockIdx.x] = shared[0];
}

__global__ void locateFinalKernel(const LocateResult* blockResults,
                                  std::size_t count, LocateKind kind,
                                  LocateResult* output)
{
    LocateResult local = emptyResult(kind);
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x)
        local = merge(kind, local, blockResults[index]);

    __shared__ LocateResult shared[BlockSize];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BlockSize / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] =
                merge(kind, shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x)
        *output = shared[0];
}

int coordinateAxis(DimensionId id) noexcept
{
    if (id == DimensionId(StandardDimension::X))
        return 0;
    if (id == DimensionId(StandardDimension::Y))
        return 1;
    if (id == DimensionId(StandardDimension::Z))
        return 2;
    return -1;
}
} // unnamed namespace

LocateResult locateExtremeDevice(PointBatch& batch,
                                 const LocateProgram& program,
                                 std::uint64_t indexOffset)
{
    NvtxRange range("pdg::filters.locate::reduce");
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA locate reduction requires a device batch");
    LocateResult empty;
    empty.value = program.kind == LocateKind::Minimum ? DBL_MAX : -DBL_MAX;
    empty.index = indexOffset;
    if (!batch.size())
        return empty;

    const std::size_t naturalBlocks =
        (batch.size() - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    const std::size_t blocks = (std::min)(naturalBlocks, MaximumBlocks);
    if (blocks > (static_cast<std::size_t>(-1) / sizeof(LocateResult)) - 1U)
        throw std::overflow_error("locate reduction allocation overflows");
    std::unique_ptr<Allocation> scratch = batch.memoryResource().allocate(
        (blocks + 1U) * sizeof(LocateResult), alignof(LocateResult));
    auto* partial = static_cast<LocateResult*>(scratch->data());
    LocateResult* output = partial + blocks;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const ColumnInfo& column = batch.columnInfo(program.dimension);
    const int axis = coordinateAxis(program.dimension);
    const double scale =
        axis >= 0
            ? batch.coordinateEncoding().scale()[static_cast<std::size_t>(axis)]
            : 1.0;
    const double offset = axis >= 0
                              ? batch.coordinateEncoding()
                                    .offset()[static_cast<std::size_t>(axis)]
                              : 0.0;
    locateBlocksKernel<<<static_cast<unsigned int>(blocks), BlockSize, 0,
                         stream>>>(
        batch.rawData(program.dimension), column.physicalType, batch.size(),
        indexOffset, program.kind, axis, scale, offset, partial);
    PDG_CUDA_CHECK(cudaGetLastError());
    locateFinalKernel<<<1, BlockSize, 0, stream>>>(partial, blocks,
                                                   program.kind, output);
    PDG_CUDA_CHECK(cudaGetLastError());

    LocateResult result;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&result, output, sizeof(result),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace pdg
