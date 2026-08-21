#include <pdg/Cuda.hpp>
#include <pdg/PackedPointBatch.hpp>
#include <pdg/PointBatch.hpp>

#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr std::size_t MaximumPackedColumns = 64;

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

struct DevicePackedColumn
{
    std::uint8_t* column = nullptr;
    std::size_t offset = 0;
    std::uint8_t size = 0;
    bool written = false;
};

struct DevicePackedLayout
{
    DevicePackedColumn columns[MaximumPackedColumns];
    std::size_t pointStride = 0;
    std::uint16_t columnCount = 0;
};

__global__ void unpackKernel(const std::uint8_t* packed,
                             const std::uint64_t* pointIds,
                             std::size_t pointCount, DevicePackedLayout layout,
                             std::uint64_t* originalIndexes)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        const std::size_t packedPoint =
            pointIds ? static_cast<std::size_t>(pointIds[point]) : point;
        const std::uint8_t* source = packed + packedPoint * layout.pointStride;
        for (std::size_t column = 0; column < layout.columnCount; ++column)
        {
            const DevicePackedColumn& target = layout.columns[column];
            std::uint8_t* destination =
                target.column + point * static_cast<std::size_t>(target.size);
#pragma unroll
            for (std::size_t byte = 0; byte < 8; ++byte)
                if (byte < target.size)
                    destination[byte] = source[target.offset + byte];
        }
        if (originalIndexes)
            originalIndexes[point] = packedPoint;
    }
}

__global__ void repackKernel(const DevicePackedLayout layout,
                             const std::uint64_t* pointIds,
                             std::size_t pointCount, std::uint8_t* packed)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t point = thread; point < pointCount; point += grid)
    {
        const std::size_t packedPoint =
            pointIds ? static_cast<std::size_t>(pointIds[point]) : point;
        std::uint8_t* destination = packed + packedPoint * layout.pointStride;
        for (std::size_t column = 0; column < layout.columnCount; ++column)
        {
            const DevicePackedColumn& source = layout.columns[column];
            if (!source.written)
                continue;
            const std::uint8_t* input =
                source.column + point * static_cast<std::size_t>(source.size);
#pragma unroll
            for (std::size_t byte = 0; byte < 8; ++byte)
                if (byte < source.size)
                    destination[source.offset + byte] = input[byte];
        }
    }
}

int gridSize(std::size_t count)
{
    const std::size_t blocks =
        count / BlockSize + static_cast<std::size_t>(count % BlockSize != 0);
    return static_cast<int>(
        std::min<std::size_t>(blocks, std::numeric_limits<int>::max()));
}

DevicePackedLayout bindLayout(const PointBatch& batch, std::size_t pointStride,
                              std::span<const PackedPointColumn> columns)
{
    if (!pointStride)
        throw std::invalid_argument("packed point stride is zero");
    if (columns.size() > MaximumPackedColumns)
        throw std::invalid_argument(
            "packed point layout exceeds the CUDA column limit");

    DevicePackedLayout result;
    result.pointStride = pointStride;
    result.columnCount = static_cast<std::uint16_t>(columns.size());
    for (std::size_t index = 0; index < columns.size(); ++index)
    {
        const PackedPointColumn& source = columns[index];
        if (!batch.has(source.id))
            throw std::invalid_argument(
                "packed point column is not materialized");
        const ColumnInfo& column = batch.columnInfo(source.id);
        if (column.physicalType != source.physicalType)
            throw std::invalid_argument(
                "packed point column physical type does not match");
        const std::size_t size = dimensionTypeSize(source.physicalType);
        if (!size || size > 8 || source.offset > pointStride ||
            size > pointStride - source.offset)
            throw std::invalid_argument(
                "packed point column lies outside the record");
        result.columns[index] = {static_cast<std::uint8_t*>(const_cast<void*>(
                                     batch.rawData(source.id))),
                                 source.offset, static_cast<std::uint8_t>(size),
                                 source.written};
    }
    return result;
}
} // unnamed namespace

void unpackPackedPointBatchDevice(const void* packed, std::size_t pointStride,
                                  const std::uint64_t* pointIds,
                                  std::size_t pointCount,
                                  std::span<const PackedPointColumn> columns,
                                  PointBatch& destination,
                                  std::uint64_t* originalIndexes)
{
    if (destination.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "packed CUDA transpose requires a device PointBatch");
    if (!packed && pointCount)
        throw std::invalid_argument("packed CUDA input is null");
    if (pointCount > destination.capacity())
        throw std::out_of_range(
            "packed CUDA input exceeds destination capacity");
    const DevicePackedLayout layout =
        bindLayout(destination, pointStride, columns);
    destination.setSize(pointCount);
    if (!pointCount)
        return;

    NvtxRange range("pdg::packed_to_soa");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(destination.nativeStreamHandle());
    unpackKernel<<<gridSize(pointCount), BlockSize, 0, stream>>>(
        static_cast<const std::uint8_t*>(packed), pointIds, pointCount, layout,
        originalIndexes);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void repackPackedPointBatchDevice(const PointBatch& source, void* packed,
                                  std::size_t pointStride,
                                  const std::uint64_t* pointIds,
                                  std::span<const PackedPointColumn> columns)
{
    if (source.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "packed CUDA repack requires a device PointBatch");
    if (!packed && source.size())
        throw std::invalid_argument("packed CUDA output is null");
    const DevicePackedLayout layout = bindLayout(source, pointStride, columns);
    if (!source.size())
        return;

    NvtxRange range("pdg::soa_to_packed");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(source.nativeStreamHandle());
    repackKernel<<<gridSize(source.size()), BlockSize, 0, stream>>>(
        layout, pointIds, source.size(), static_cast<std::uint8_t*>(packed));
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
