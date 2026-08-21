#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/LabelDuplicates.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__global__ void initializeDuplicatesKernel(std::uint8_t* output,
                                           std::size_t size)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread + 1U; point < size; point += grid)
        output[point] = 1U;
}

template <typename T>
__global__ void compareAdjacentKernel(const T* values, std::size_t size,
                                      std::uint8_t* output)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread + 1U; point < size; point += grid)
        if (static_cast<double>(values[point]) !=
            static_cast<double>(values[point - 1U]))
            output[point] = 0U;
}

__global__ void compareCoordinateKernel(const std::int32_t* values,
                                        std::size_t size, double scale,
                                        double offset, std::uint8_t* output)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread + 1U; point < size; point += grid)
    {
        const double current = __dadd_rn(
            __dmul_rn(static_cast<double>(values[point]), scale), offset);
        const double previous = __dadd_rn(
            __dmul_rn(static_cast<double>(values[point - 1U]), scale), offset);
        if (current != previous)
            output[point] = 0U;
    }
}

template <typename T>
void compareTyped(PointBatch& batch, DimensionId dimension,
                  std::uint8_t* output)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    compareAdjacentKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<T>(dimension), batch.size(), output);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void compareDimension(PointBatch& batch, DimensionId dimension,
                      std::uint8_t* output)
{
    const ColumnInfo& column = batch.columnInfo(dimension);
    const int axis = coordinateAxis(dimension);
    if (axis >= 0 && column.physicalType == DimensionType::Signed32)
    {
        const std::size_t coordinate = static_cast<std::size_t>(axis);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(batch.nativeStreamHandle());
        compareCoordinateKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                  stream>>>(
            batch.data<std::int32_t>(dimension), batch.size(),
            batch.coordinateEncoding().scale()[coordinate],
            batch.coordinateEncoding().offset()[coordinate], output);
        PDG_CUDA_CHECK(cudaGetLastError());
        return;
    }
    switch (column.physicalType)
    {
    case DimensionType::Signed8:
        return compareTyped<std::int8_t>(batch, dimension, output);
    case DimensionType::Signed16:
        return compareTyped<std::int16_t>(batch, dimension, output);
    case DimensionType::Signed32:
        return compareTyped<std::int32_t>(batch, dimension, output);
    case DimensionType::Signed64:
        return compareTyped<std::int64_t>(batch, dimension, output);
    case DimensionType::Unsigned8:
        return compareTyped<std::uint8_t>(batch, dimension, output);
    case DimensionType::Unsigned16:
        return compareTyped<std::uint16_t>(batch, dimension, output);
    case DimensionType::Unsigned32:
        return compareTyped<std::uint32_t>(batch, dimension, output);
    case DimensionType::Unsigned64:
        return compareTyped<std::uint64_t>(batch, dimension, output);
    case DimensionType::Float:
        return compareTyped<float>(batch, dimension, output);
    case DimensionType::Double:
        return compareTyped<double>(batch, dimension, output);
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument("label_duplicates input has no physical type");
}
} // unnamed namespace

void labelDuplicatesDevice(PointBatch& batch,
                           const LabelDuplicatesProgram& program,
                           std::uint8_t* output)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA label_duplicates requires a device batch");
    if (batch.size() < 2U)
        return;
    NvtxRange range("pdg::filters.label_duplicates");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    initializeDuplicatesKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                 stream>>>(output, batch.size());
    PDG_CUDA_CHECK(cudaGetLastError());
    for (DimensionId dimension : program.dimensions)
        compareDimension(batch, dimension, output);
}

} // namespace pdg
