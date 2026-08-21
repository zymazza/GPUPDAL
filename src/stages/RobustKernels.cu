#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Robust.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
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

__global__ void materializeLogicalKernel(const void* source, DimensionType type,
                                         std::size_t size, int coordinateAxis,
                                         double coordinateScale,
                                         double coordinateOffset,
                                         double* values)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        double value = loadPhysical(source, type, point);
        if (coordinateAxis >= 0 && type == DimensionType::Signed32)
            value = __dadd_rn(
                __dmul_rn(static_cast<double>(
                              static_cast<const std::int32_t*>(source)[point]),
                          coordinateScale),
                coordinateOffset);
        values[point] = value;
    }
}

__global__ void deviationKernel(const void* source, DimensionType type,
                                std::size_t size, int coordinateAxis,
                                double coordinateScale, double coordinateOffset,
                                double median, double* deviations)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        double value = loadPhysical(source, type, point);
        if (coordinateAxis >= 0 && type == DimensionType::Signed32)
            value = __dadd_rn(
                __dmul_rn(static_cast<double>(
                              static_cast<const std::int32_t*>(source)[point]),
                          coordinateScale),
                coordinateOffset);
        deviations[point] = fabs(__dsub_rn(value, median));
    }
}

__global__ void iqrMaskKernel(const void* source, DimensionType type,
                              std::size_t size, int coordinateAxis,
                              double coordinateScale, double coordinateOffset,
                              double lowFence, double highFence,
                              std::uint8_t* keep)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        double value = loadPhysical(source, type, point);
        if (coordinateAxis >= 0 && type == DimensionType::Signed32)
            value = __dadd_rn(
                __dmul_rn(static_cast<double>(
                              static_cast<const std::int32_t*>(source)[point]),
                          coordinateScale),
                coordinateOffset);
        keep[point] =
            static_cast<std::uint8_t>(value > lowFence && value < highFence);
    }
}

__global__ void madMaskKernel(const void* source, DimensionType type,
                              std::size_t size, int coordinateAxis,
                              double coordinateScale, double coordinateOffset,
                              double median, double mad, double multiplier,
                              std::uint8_t* keep)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < size; point += grid)
    {
        double value = loadPhysical(source, type, point);
        if (coordinateAxis >= 0 && type == DimensionType::Signed32)
            value = __dadd_rn(
                __dmul_rn(static_cast<double>(
                              static_cast<const std::int32_t*>(source)[point]),
                          coordinateScale),
                coordinateOffset);
        const double deviation = fabs(__dsub_rn(value, median));
        keep[point] =
            static_cast<std::uint8_t>(__ddiv_rn(deviation, mad) < multiplier);
    }
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

RobustResult evaluateRobustDevice(PointBatch& batch,
                                  const RobustProgram& program,
                                  std::uint8_t* keep)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA robust statistics require a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA robust statistics exceed the CUB item limit");

    NvtxRange range(program.kind == RobustKind::Iqr ? "pdg::filters.iqr"
                                                    : "pdg::filters.mad");
    const std::size_t bytes = batch.size() * sizeof(double);
    std::unique_ptr<Allocation> primaryAllocation =
        batch.memoryResource().allocate(bytes, alignof(double));
    std::unique_ptr<Allocation> alternateAllocation =
        batch.memoryResource().allocate(bytes, alignof(double));
    auto* primary = static_cast<double*>(primaryAllocation->data());
    auto* alternate = static_cast<double*>(alternateAllocation->data());
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
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    const std::size_t naturalBlocks =
        (batch.size() - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    const unsigned int blocks = static_cast<unsigned int>(
        (std::min)(naturalBlocks, static_cast<std::size_t>(65535)));
    materializeLogicalKernel<<<blocks, BlockSize, 0, stream>>>(
        batch.rawData(program.dimension), column.physicalType, batch.size(),
        axis, scale, offset, primary);
    PDG_CUDA_CHECK(cudaGetLastError());

    cub::DoubleBuffer<double> keys(primary, alternate);
    std::size_t temporaryBytes = 0;
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, temporaryBytes, keys, static_cast<int>(batch.size()), 0, 64,
        stream));
    std::unique_ptr<Allocation> temporary = batch.memoryResource().allocate(
        temporaryBytes, alignof(std::max_align_t));
    PDG_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        temporary->data(), temporaryBytes, keys, static_cast<int>(batch.size()),
        0, 64, stream));

    RobustResult result;
    if (program.kind == RobustKind::Iqr)
    {
        const int lowerIndex =
            static_cast<int>(static_cast<double>(batch.size()) * 0.25);
        const int upperIndex =
            static_cast<int>(static_cast<double>(batch.size()) * 0.75);
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(&result.first, keys.Current() + lowerIndex,
                            sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(&result.second, keys.Current() + upperIndex,
                            sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        result.scale = result.second - result.first;
        result.lowFence = result.first - program.multiplier * result.scale;
        result.highFence = result.second + program.multiplier * result.scale;
        iqrMaskKernel<<<blocks, BlockSize, 0, stream>>>(
            batch.rawData(program.dimension), column.physicalType, batch.size(),
            axis, scale, offset, result.lowFence, result.highFence, keep);
        PDG_CUDA_CHECK(cudaGetLastError());
    }
    else
    {
        const std::size_t medianIndex = batch.size() / 2U;
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(&result.first, keys.Current() + medianIndex,
                            sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        deviationKernel<<<blocks, BlockSize, 0, stream>>>(
            batch.rawData(program.dimension), column.physicalType, batch.size(),
            axis, scale, offset, result.first, primary);
        PDG_CUDA_CHECK(cudaGetLastError());
        cub::DoubleBuffer<double> deviationKeys(primary, alternate);
        PDG_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            temporary->data(), temporaryBytes, deviationKeys,
            static_cast<int>(batch.size()), 0, 64, stream));
        double deviationMedian = 0.0;
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            &deviationMedian, deviationKeys.Current() + medianIndex,
            sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        result.scale = deviationMedian * program.madMultiplier;
        result.second = result.scale;
        result.lowFence = result.first - program.multiplier * result.scale;
        result.highFence = result.first + program.multiplier * result.scale;
        madMaskKernel<<<blocks, BlockSize, 0, stream>>>(
            batch.rawData(program.dimension), column.physicalType, batch.size(),
            axis, scale, offset, result.first, result.scale, program.multiplier,
            keep);
        PDG_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

} // namespace pdg
