#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
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

__global__ void initializePermutationKernel(std::uint64_t* permutation,
                                            std::size_t size)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
        permutation[point] = static_cast<std::uint64_t>(point);
}

template <typename T>
__global__ void gatherKeysKernel(const T* source,
                                 const std::uint64_t* permutation,
                                 std::size_t size, T* keys)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
        keys[point] = source[permutation[point]];
}

__global__ void gatherCoordinateKeysKernel(const std::int32_t* source,
                                           const std::uint64_t* permutation,
                                           std::size_t size, double scale,
                                           double offset, double* keys)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const std::int32_t raw = source[permutation[point]];
        keys[point] =
            __dadd_rn(__dmul_rn(static_cast<double>(raw), scale), offset);
    }
}

template <typename T>
__global__ void equivalentAdjacentKernel(const T* sortedKeys, std::size_t size,
                                         int* duplicate)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread + 1U; point < size; point += grid)
        if (sortedKeys[point] == sortedKeys[point - 1U])
            atomicExch(duplicate, 1);
}

template <typename Key>
bool sortMaterializedKeys(PointBatch& batch, Key* keysInput, Key* keysOutput,
                          const std::uint64_t* indicesInput,
                          std::uint64_t* indicesOutput,
                          OrderingDirection direction, bool checkUnique)
{
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::size_t temporaryBytes = 0;
    if (direction == OrderingDirection::Ascending)
    {
        PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, temporaryBytes, keysInput, keysOutput, indicesInput,
            indicesOutput, static_cast<int>(batch.size()), 0,
            static_cast<int>(sizeof(Key) * 8U), stream));
    }
    else
    {
        PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temporaryBytes, keysInput, keysOutput, indicesInput,
            indicesOutput, static_cast<int>(batch.size()), 0,
            static_cast<int>(sizeof(Key) * 8U), stream));
    }
    std::unique_ptr<Allocation> temporary = batch.memoryResource().allocate(
        temporaryBytes, alignof(std::max_align_t));
    if (direction == OrderingDirection::Ascending)
    {
        PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temporary->data(), temporaryBytes, keysInput, keysOutput,
            indicesInput, indicesOutput, static_cast<int>(batch.size()), 0,
            static_cast<int>(sizeof(Key) * 8U), stream));
    }
    else
    {
        PDG_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            temporary->data(), temporaryBytes, keysInput, keysOutput,
            indicesInput, indicesOutput, static_cast<int>(batch.size()), 0,
            static_cast<int>(sizeof(Key) * 8U), stream));
    }

    if (!checkUnique)
        return true;
    std::unique_ptr<Allocation> duplicateAllocation =
        batch.memoryResource().allocate(sizeof(int), alignof(int));
    auto* duplicate = static_cast<int*>(duplicateAllocation->data());
    PDG_CUDA_CHECK(cudaMemsetAsync(duplicate, 0, sizeof(int), stream));
    equivalentAdjacentKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                               stream>>>(keysOutput, batch.size(), duplicate);
    PDG_CUDA_CHECK(cudaGetLastError());
    int hostDuplicate = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&hostDuplicate, duplicate, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return hostDuplicate == 0;
}

template <typename T>
bool typedPass(PointBatch& batch, const T* source,
               const std::uint64_t* indicesInput, std::uint64_t* indicesOutput,
               OrderingDirection direction, bool checkUnique)
{
    const std::size_t bytes = batch.size() * sizeof(T);
    std::unique_ptr<Allocation> inputAllocation =
        batch.memoryResource().allocate(bytes, alignof(T));
    std::unique_ptr<Allocation> outputAllocation =
        batch.memoryResource().allocate(bytes, alignof(T));
    auto* keysInput = static_cast<T*>(inputAllocation->data());
    auto* keysOutput = static_cast<T*>(outputAllocation->data());
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    gatherKeysKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        source, indicesInput, batch.size(), keysInput);
    PDG_CUDA_CHECK(cudaGetLastError());
    return sortMaterializedKeys(batch, keysInput, keysOutput, indicesInput,
                                indicesOutput, direction, checkUnique);
}

bool coordinatePass(PointBatch& batch, DimensionId dimension,
                    const std::int32_t* source,
                    const std::uint64_t* indicesInput,
                    std::uint64_t* indicesOutput, OrderingDirection direction,
                    bool checkUnique)
{
    const std::size_t bytes = batch.size() * sizeof(double);
    std::unique_ptr<Allocation> inputAllocation =
        batch.memoryResource().allocate(bytes, alignof(double));
    std::unique_ptr<Allocation> outputAllocation =
        batch.memoryResource().allocate(bytes, alignof(double));
    auto* keysInput = static_cast<double*>(inputAllocation->data());
    auto* keysOutput = static_cast<double*>(outputAllocation->data());
    const int axis = coordinateAxis(dimension);
    const std::size_t coordinate = static_cast<std::size_t>(axis);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    gatherCoordinateKeysKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                 stream>>>(
        source, indicesInput, batch.size(),
        batch.coordinateEncoding().scale()[coordinate],
        batch.coordinateEncoding().offset()[coordinate], keysInput);
    PDG_CUDA_CHECK(cudaGetLastError());
    return sortMaterializedKeys(batch, keysInput, keysOutput, indicesInput,
                                indicesOutput, direction, checkUnique);
}

bool executePass(PointBatch& batch, DimensionId dimension,
                 const std::uint64_t* indicesInput,
                 std::uint64_t* indicesOutput, OrderingDirection direction,
                 bool checkUnique)
{
    const ColumnInfo& column = batch.columnInfo(dimension);
    const void* data = batch.rawData(dimension);
    if (coordinateAxis(dimension) >= 0 &&
        column.physicalType == DimensionType::Signed32)
        return coordinatePass(
            batch, dimension, static_cast<const std::int32_t*>(data),
            indicesInput, indicesOutput, direction, checkUnique);
    switch (column.physicalType)
    {
    case DimensionType::Signed8:
        return typedPass(batch, static_cast<const std::int8_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Signed16:
        return typedPass(batch, static_cast<const std::int16_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Signed32:
        return typedPass(batch, static_cast<const std::int32_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Signed64:
        return typedPass(batch, static_cast<const std::int64_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Unsigned8:
        return typedPass(batch, static_cast<const std::uint8_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Unsigned16:
        return typedPass(batch, static_cast<const std::uint16_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Unsigned32:
        return typedPass(batch, static_cast<const std::uint32_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Unsigned64:
        return typedPass(batch, static_cast<const std::uint64_t*>(data),
                         indicesInput, indicesOutput, direction, checkUnique);
    case DimensionType::Float:
        return typedPass(batch, static_cast<const float*>(data), indicesInput,
                         indicesOutput, direction, checkUnique);
    case DimensionType::Double:
        return typedPass(batch, static_cast<const double*>(data), indicesInput,
                         indicesOutput, direction, checkUnique);
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument("ordering key has no physical type");
}
} // unnamed namespace

OrderingResult orderPointsDevice(PointBatch& batch,
                                 const OrderingProgram& program,
                                 std::uint64_t* permutation)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA ordering requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("CUDA ordering exceeds the CUB item limit");
    if (!batch.size())
        return {};

    NvtxRange range("pdg::filters.sort");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    initializePermutationKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                  stream>>>(permutation, batch.size());
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> alternateAllocation =
        batch.memoryResource().allocate(batch.size() * sizeof(std::uint64_t),
                                        alignof(std::uint64_t));
    auto* alternate = static_cast<std::uint64_t*>(alternateAllocation->data());
    std::uint64_t* current = permutation;
    std::uint64_t* next = alternate;
    bool exact = true;
    for (std::size_t pass = 0; pass < program.dimensions.size(); ++pass)
    {
        const bool last = pass + 1U == program.dimensions.size();
        const bool checkUnique =
            last && (program.dimensions.size() > 1U ||
                     program.algorithm == OrderingAlgorithm::Normal);
        const bool unique =
            executePass(batch, program.dimensions[pass], current, next,
                        program.direction, checkUnique);
        if (checkUnique && !unique)
            exact = false;
        std::swap(current, next);
    }
    if (current != permutation)
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation, current,
                                       batch.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToDevice, stream));
    return {exact};
}

} // namespace pdg
