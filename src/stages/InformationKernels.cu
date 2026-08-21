#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
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

struct DeviceParameters
{
    const void* columns[3]{};
    DimensionType types[3]{};
    double scales[3]{};
    double offsets[3]{};
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

__device__ void consider(LocateResult& result, bool minimum, double value,
                         std::uint64_t index)
{
    if (!result.hasPoints)
    {
        result.hasPoints = 1;
        result.index = index;
    }
    const bool improves = minimum ? value < result.value : value > result.value;
    if (improves)
    {
        result.value = value;
        result.index = index;
        result.comparable = 1;
    }
}

__device__ LocateResult mergeExtreme(LocateResult first, LocateResult second,
                                     bool minimum)
{
    if (!first.hasPoints)
        return second;
    if (!second.hasPoints)
        return first;
    if (first.comparable != second.comparable)
        return first.comparable ? first : second;
    if (!first.comparable)
        return first.index <= second.index ? first : second;
    const bool secondImproves =
        minimum ? second.value < first.value : second.value > first.value;
    if (secondImproves)
        return second;
    const bool firstImproves =
        minimum ? first.value < second.value : first.value > second.value;
    if (firstImproves)
        return first;
    return first.index <= second.index ? first : second;
}

__device__ BoundsResult mergeBounds(BoundsResult first, BoundsResult second)
{
    BoundsResult result;
    result.count = first.count + second.count;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.minimum[axis] =
            mergeExtreme(first.minimum[axis], second.minimum[axis], true);
        result.maximum[axis] =
            mergeExtreme(first.maximum[axis], second.maximum[axis], false);
    }
    return result;
}

__global__ void boundsBlocksKernel(const DeviceParameters* parameters,
                                   std::size_t size, std::uint64_t indexOffset,
                                   BoundsResult* blockResults)
{
    BoundsResult local;
    for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
         point < size; point += blockDim.x * gridDim.x)
    {
        ++local.count;
        const std::uint64_t index =
            indexOffset + static_cast<std::uint64_t>(point);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            double value = loadPhysical(parameters->columns[axis],
                                        parameters->types[axis], point);
            if (parameters->types[axis] == DimensionType::Signed32)
                value = __dadd_rn(
                    __dmul_rn(
                        static_cast<double>(static_cast<const std::int32_t*>(
                            parameters->columns[axis])[point]),
                        parameters->scales[axis]),
                    parameters->offsets[axis]);
            consider(local.minimum[axis], true, value, index);
            consider(local.maximum[axis], false, value, index);
        }
    }

    __shared__ BoundsResult shared[BlockSize];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BlockSize / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] =
                mergeBounds(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x)
        blockResults[blockIdx.x] = shared[0];
}

__global__ void boundsFinalKernel(const BoundsResult* blockResults,
                                  std::size_t count, BoundsResult* output)
{
    BoundsResult local;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x)
        local = mergeBounds(local, blockResults[index]);

    __shared__ BoundsResult shared[BlockSize];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BlockSize / 2; stride; stride /= 2)
    {
        if (threadIdx.x < stride)
            shared[threadIdx.x] =
                mergeBounds(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x)
        *output = shared[0];
}
} // unnamed namespace

BoundsResult summarizeBoundsDevice(PointBatch& batch, std::uint64_t indexOffset)
{
    NvtxRange range("pdg::filters.info::bounds");
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA bounds require a device batch");
    BoundsResult empty;
    if (!batch.size())
        return empty;

    const std::array<DimensionId, 3> dimensions = {
        DimensionId(StandardDimension::X), DimensionId(StandardDimension::Y),
        DimensionId(StandardDimension::Z)};
    std::array<const void*, 3> columns{};
    std::array<DimensionType, 3> types{};
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
    {
        if (!batch.has(dimensions[axis]))
            throw std::invalid_argument(
                "bounds coordinate column is not materialized");
        columns[axis] = batch.rawData(dimensions[axis]);
        types[axis] = batch.columnInfo(dimensions[axis]).physicalType;
    }

    const std::size_t naturalBlocks =
        (batch.size() - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    const std::size_t blocks = (std::min)(naturalBlocks, MaximumBlocks);
    std::unique_ptr<Allocation> scratch = batch.memoryResource().allocate(
        (blocks + 1U) * sizeof(BoundsResult), alignof(BoundsResult));
    auto* partial = static_cast<BoundsResult*>(scratch->data());
    BoundsResult* output = partial + blocks;

    DeviceParameters hostParameters;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        hostParameters.columns[axis] = columns[axis];
        hostParameters.types[axis] = types[axis];
        hostParameters.scales[axis] = batch.coordinateEncoding().scale()[axis];
        hostParameters.offsets[axis] =
            batch.coordinateEncoding().offset()[axis];
    }
    std::unique_ptr<Allocation> parameters = batch.memoryResource().allocate(
        sizeof(DeviceParameters), alignof(DeviceParameters));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(parameters->data(), &hostParameters,
                                   sizeof(hostParameters),
                                   cudaMemcpyHostToDevice, stream));

    boundsBlocksKernel<<<static_cast<unsigned int>(blocks), BlockSize, 0,
                         stream>>>(
        static_cast<const DeviceParameters*>(parameters->data()), batch.size(),
        indexOffset, partial);
    PDG_CUDA_CHECK(cudaGetLastError());
    boundsFinalKernel<<<1, BlockSize, 0, stream>>>(partial, blocks, output);
    PDG_CUDA_CHECK(cudaGetLastError());

    BoundsResult result;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&result, output, sizeof(result),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace pdg
