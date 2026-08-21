#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Smrf.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pdg
{
namespace
{
constexpr int BlockSize = 256;
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId Classification(StandardDimension::Classification);

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

struct DeviceFrame
{
    double minimumX;
    double minimumY;
    std::size_t rows;
    std::size_t columns;
};

struct Neighbor
{
    std::uint64_t squaredDistance;
    std::size_t cell;
};

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

template <typename T>
std::unique_ptr<Allocation> allocate(PointBatch& batch, std::size_t count)
{
    if (count && count > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
        throw std::overflow_error("CUDA SMRF allocation size overflows");
    return batch.memoryResource().allocate(count * sizeof(T), alignof(T));
}

template <typename T> T* pointer(const std::unique_ptr<Allocation>& allocation)
{
    return static_cast<T*>(allocation->data());
}

__device__ std::size_t threadIndex()
{
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ std::size_t gridStride()
{
    return static_cast<std::size_t>(blockDim.x) * gridDim.x;
}

__global__ void initializeDoubleKernel(double* values, std::size_t size,
                                       double value)
{
    for (std::size_t index = threadIndex(); index < size; index += gridStride())
        values[index] = value;
}

__global__ void initializeIntKernel(int* values, std::size_t size, int value)
{
    for (std::size_t index = threadIndex(); index < size; index += gridStride())
        values[index] = value;
}

__device__ void atomicMinimum(double* address, double value)
{
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long observed = atomicCAS(bits, 0ULL, 0ULL);
    while (true)
    {
        const double current =
            __longlong_as_double(static_cast<long long>(observed));
        if (!isnan(current) && !(value < current))
            return;
        const unsigned long long desired =
            static_cast<unsigned long long>(__double_as_longlong(value));
        const unsigned long long previous = atomicCAS(bits, observed, desired);
        if (previous == observed)
            return;
        observed = previous;
    }
}

__device__ bool rasterCell(double x, double y, const DeviceFrame& frame,
                           double cell, std::size_t& index)
{
    const double columnValue =
        floor(__ddiv_rn(__dsub_rn(x, frame.minimumX), cell));
    const double rowValue =
        floor(__ddiv_rn(__dsub_rn(y, frame.minimumY), cell));
    if (columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        return false;
    const std::size_t column = static_cast<std::size_t>(columnValue);
    const std::size_t row = static_cast<std::size_t>(rowValue);
    index = column * frame.rows + row;
    return true;
}

__global__ void minimumRasterKernel(const double* x, const double* y,
                                    const double* z, std::size_t pointCount,
                                    DeviceFrame frame, double cell,
                                    double* minimum, int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::size_t index = 0U;
        if (!rasterCell(x[point], y[point], frame, cell, index))
        {
            atomicExch(invalidFrame, 1);
            continue;
        }
        atomicMinimum(minimum + index, z[point]);
    }
}

__global__ void minimumSourceKernel(const double* x, const double* y,
                                    const double* z, std::size_t pointCount,
                                    DeviceFrame frame, double cell,
                                    const double* minimum,
                                    unsigned int* sourcePoint)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::size_t index = 0U;
        if (rasterCell(x[point], y[point], frame, cell, index) &&
            z[point] == minimum[index])
            atomicMin(sourcePoint + index, static_cast<unsigned int>(point));
    }
}

__global__ void materializeMinimumKernel(const double* z,
                                         const unsigned int* sourcePoint,
                                         std::size_t size, double* minimum)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
    {
        const unsigned int point = sourcePoint[cell];
        if (point != 0xffffffffU)
            minimum[cell] = z[point];
    }
}

__device__ void insertNeighbor(Neighbor (&neighbors)[8], std::size_t& count,
                               Neighbor candidate)
{
    std::size_t position = count;
    while (position > 0U)
    {
        const Neighbor previous = neighbors[position - 1U];
        if (previous.squaredDistance < candidate.squaredDistance ||
            (previous.squaredDistance == candidate.squaredDistance &&
             previous.cell <= candidate.cell))
            break;
        if (position < 8U)
            neighbors[position] = previous;
        --position;
    }
    if (position < 8U)
        neighbors[position] = candidate;
    if (count < 8U)
        ++count;
}

__global__ void fillRasterKernel(const double* source, DeviceFrame frame,
                                 double* output)
{
    const std::size_t size = frame.rows * frame.columns;
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
    {
        if (!isnan(source[cell]))
        {
            output[cell] = source[cell];
            continue;
        }
        const std::size_t column = cell / frame.rows;
        const std::size_t row = cell % frame.rows;
        Neighbor nearest[8]{};
        std::size_t count = 0U;
        for (std::size_t candidate = 0U; candidate < size; ++candidate)
        {
            if (isnan(source[candidate]))
                continue;
            const std::size_t sourceColumn = candidate / frame.rows;
            const std::size_t sourceRow = candidate % frame.rows;
            const std::uint64_t deltaColumn = column > sourceColumn
                                                  ? column - sourceColumn
                                                  : sourceColumn - column;
            const std::uint64_t deltaRow =
                row > sourceRow ? row - sourceRow : sourceRow - row;
            insertNeighbor(
                nearest, count,
                {deltaColumn * deltaColumn + deltaRow * deltaRow, candidate});
        }
        double mean = 0.0;
        for (std::size_t index = 0U; index < count; ++index)
        {
            const double delta = __dsub_rn(source[nearest[index].cell], mean);
            mean = __dadd_rn(mean,
                             __ddiv_rn(delta, static_cast<double>(index + 1U)));
        }
        output[cell] = count ? mean : nan("");
    }
}

__global__ void morphKernel(const double* input, std::size_t rows,
                            std::size_t columns, bool dilation, double* output)
{
    const std::size_t size = rows * columns;
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
    {
        const std::size_t column = cell / rows;
        const std::size_t row = cell % rows;
        double result = input[cell];
        const auto include = [&](double value)
        {
            if ((dilation && value > result) || (!dilation && value < result))
                result = value;
        };
        if (row > 0U)
            include(input[cell - 1U]);
        if (row + 1U < rows)
            include(input[cell + 1U]);
        if (column > 0U)
            include(input[cell - rows]);
        if (column + 1U < columns)
            include(input[cell + rows]);
        output[cell] = result;
    }
}

__global__ void progressiveMaskKernel(const double* previous,
                                      const double* opening, std::size_t size,
                                      double threshold, int* objects)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
        if (fabs(__dsub_rn(previous[cell], opening[cell])) > threshold)
            objects[cell] = 1;
}

__global__ void negateKernel(const double* input, std::size_t size,
                             double* output)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
        output[cell] = -input[cell];
}

__global__ void netMaskKernel(std::size_t rows, std::size_t columns,
                              std::size_t spacing, int* mask)
{
    const std::size_t size = rows * columns;
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
    {
        const std::size_t column = cell / rows;
        const std::size_t row = cell % rows;
        mask[cell] = column % spacing == 0U || row % spacing == 0U ? 1 : 0;
    }
}

__global__ void applyNetKernel(const double* minimum, const double* opening,
                               const int* net, std::size_t size, double* output)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
        output[cell] = net[cell] == 1 ? opening[cell] : minimum[cell];
}

__global__ void provisionalKernel(const double* minimum, const int* low,
                                  const int* net, const int* objects,
                                  std::size_t size, double* provisional)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
        provisional[cell] =
            low[cell] == 1 || net[cell] == 1 || objects[cell] == 1
                ? nan("")
                : minimum[cell];
}

__global__ void scaleKernel(const double* values, std::size_t size, double cell,
                            double* scaled)
{
    for (std::size_t index = threadIndex(); index < size; index += gridStride())
        scaled[index] = __ddiv_rn(values[index], cell);
}

__global__ void gradientKernel(const double* values, DeviceFrame frame,
                               double* gradient)
{
    const std::size_t size = frame.rows * frame.columns;
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
    {
        const std::size_t column = cell / frame.rows;
        const std::size_t row = cell % frame.rows;
        double gx;
        if (column == 0U)
            gx = __dsub_rn(values[cell + frame.rows], values[cell]);
        else if (column + 1U == frame.columns)
            gx = __dsub_rn(values[cell], values[cell - frame.rows]);
        else
            gx = __dmul_rn(0.5, __dsub_rn(values[cell + frame.rows],
                                          values[cell - frame.rows]));
        double gy;
        if (row == 0U)
            gy = __dsub_rn(values[cell + 1U], values[cell]);
        else if (row + 1U == frame.rows)
            gy = __dsub_rn(values[cell], values[cell - 1U]);
        else
            gy =
                __dmul_rn(0.5, __dsub_rn(values[cell + 1U], values[cell - 1U]));
        gradient[cell] = sqrt(__dadd_rn(__dmul_rn(gx, gx), __dmul_rn(gy, gy)));
    }
}

__global__ void classifyKernel(const double* x, const double* y,
                               const double* z, std::uint8_t* classification,
                               std::size_t pointCount, DeviceFrame frame,
                               double cell, const double* provisional,
                               const double* gradient, double threshold,
                               double scalar, std::uint8_t groundClass,
                               std::uint8_t otherClass, bool onlyGround,
                               unsigned long long* counts)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        if (!onlyGround)
            classification[point] = otherClass;
        std::size_t index = 0U;
        if (!rasterCell(x[point], y[point], frame, cell, index) ||
            isnan(provisional[index]) || isnan(gradient[index]))
            continue;
        const double elevationThreshold =
            __dadd_rn(threshold, __dmul_rn(scalar, gradient[index]));
        if (fabs(__dsub_rn(provisional[index], z[point])) > elevationThreshold)
        {
            atomicAdd(counts + 1U, 1ULL);
            if (!onlyGround)
                classification[point] = otherClass;
        }
        else
        {
            atomicAdd(counts, 1ULL);
            classification[point] = groundClass;
        }
    }
}

std::unique_ptr<Allocation> copyDoubles(PointBatch& batch, const double* source,
                                        std::size_t size)
{
    std::unique_ptr<Allocation> output = allocate<double>(batch, size);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(output->data(), source,
                                   size * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
    return output;
}

std::unique_ptr<Allocation> morphology(PointBatch& batch, const double* input,
                                       const DeviceFrame& frame, int iterations,
                                       bool dilation)
{
    const std::size_t size = frame.rows * frame.columns;
    std::unique_ptr<Allocation> current = copyDoubles(batch, input, size);
    std::unique_ptr<Allocation> next = allocate<double>(batch, size);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        morphKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            pointer<double>(current), frame.rows, frame.columns, dilation,
            pointer<double>(next));
        PDG_CUDA_CHECK(cudaGetLastError());
        current.swap(next);
    }
    return current;
}

std::unique_ptr<Allocation> progressiveFilter(PointBatch& batch,
                                              const double* input,
                                              const DeviceFrame& frame,
                                              double cell, double slope,
                                              int maximumRadius)
{
    const std::size_t size = frame.rows * frame.columns;
    std::unique_ptr<Allocation> previous = copyDoubles(batch, input, size);
    std::unique_ptr<Allocation> erosion = copyDoubles(batch, input, size);
    std::unique_ptr<Allocation> objects = allocate<int>(batch, size);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    initializeIntKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<int>(objects), size, 0);
    PDG_CUDA_CHECK(cudaGetLastError());
    for (int radius = 1; radius <= maximumRadius; ++radius)
    {
        erosion = morphology(batch, pointer<double>(erosion), frame, 1, false);
        std::unique_ptr<Allocation> opening =
            morphology(batch, pointer<double>(erosion), frame, radius, true);
        const double elevationThreshold =
            slope * cell * static_cast<double>(radius);
        progressiveMaskKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            pointer<double>(previous), pointer<double>(opening), size,
            elevationThreshold, pointer<int>(objects));
        PDG_CUDA_CHECK(cudaGetLastError());
        previous = std::move(opening);
    }
    return objects;
}

DeviceFrame reduceFrame(PointBatch& batch, double cell)
{
    const int count = static_cast<int>(batch.size());
    std::unique_ptr<Allocation> results = allocate<double>(batch, 4U);
    auto* output = pointer<double>(results);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::size_t minimumBytes = 0U;
    std::size_t maximumBytes = 0U;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(
        nullptr, minimumBytes, batch.data<double>(X), output, count, stream));
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(nullptr, maximumBytes,
                                          batch.data<double>(X), output + 1U,
                                          count, stream));
    const std::size_t temporaryBytes = (std::max)(minimumBytes, maximumBytes);
    std::unique_ptr<Allocation> temporary = batch.memoryResource().allocate(
        temporaryBytes, alignof(std::max_align_t));
    minimumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(temporary->data(), minimumBytes,
                                          batch.data<double>(X), output, count,
                                          stream));
    maximumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(temporary->data(), maximumBytes,
                                          batch.data<double>(X), output + 1U,
                                          count, stream));
    minimumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Min(temporary->data(), minimumBytes,
                                          batch.data<double>(Y), output + 2U,
                                          count, stream));
    maximumBytes = temporaryBytes;
    PDG_CUDA_CHECK(cub::DeviceReduce::Max(temporary->data(), maximumBytes,
                                          batch.data<double>(Y), output + 3U,
                                          count, stream));
    double host[4]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(host, output, sizeof(host),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    const double columnsValue = (host[1] - host[0]) / cell + 1.0;
    const double rowsValue = (host[3] - host[2]) / cell + 1.0;
    if (!std::isfinite(columnsValue) || !std::isfinite(rowsValue) ||
        columnsValue < 2.0 || rowsValue < 2.0 ||
        columnsValue > static_cast<double>(SmrfExactDeviceMaximumRasterCells) ||
        rowsValue > static_cast<double>(SmrfExactDeviceMaximumRasterCells))
        throw std::invalid_argument("CUDA SMRF raster dimensions are invalid");
    const DeviceFrame frame{host[0], host[2],
                            static_cast<std::size_t>(rowsValue),
                            static_cast<std::size_t>(columnsValue)};
    if (frame.rows > SmrfExactDeviceMaximumRasterCells / frame.columns)
        throw std::invalid_argument(
            "CUDA SMRF raster exceeds the exact device envelope");
    return frame;
}
} // unnamed namespace

SmrfResult classifySmrfDevice(PointBatch& batch, const SmrfProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA SMRF requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA SMRF point count exceeds the exact device envelope");
    const double objectRadiusValue = std::ceil(program.window / program.cell);
    const double netSpacingValue =
        program.cut > 0.0 ? std::ceil(program.cut / program.cell) : 0.0;
    if (objectRadiusValue > SmrfExactDeviceMaximumMorphologyRadius ||
        2.0 * netSpacingValue > SmrfExactDeviceMaximumMorphologyRadius)
        throw std::invalid_argument(
            "CUDA SMRF morphology exceeds the exact device envelope");

    NvtxRange range("pdg::filters.smrf");
    const DeviceFrame frame = reduceFrame(batch, program.cell);
    const std::size_t size = frame.rows * frame.columns;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());

    std::unique_ptr<Allocation> sparseMinimum = allocate<double>(batch, size);
    std::unique_ptr<Allocation> invalidFrame = allocate<int>(batch, 1U);
    PDG_CUDA_CHECK(
        cudaMemsetAsync(invalidFrame->data(), 0, sizeof(int), stream));
    initializeDoubleKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(sparseMinimum), size,
        std::numeric_limits<double>::quiet_NaN());
    PDG_CUDA_CHECK(cudaGetLastError());
    minimumRasterKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), frame, program.cell, pointer<double>(sparseMinimum),
        pointer<int>(invalidFrame));
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> minimumSource =
        allocate<unsigned int>(batch, size);
    PDG_CUDA_CHECK(cudaMemsetAsync(minimumSource->data(), 0xff,
                                   size * sizeof(unsigned int), stream));
    minimumSourceKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), frame, program.cell, pointer<double>(sparseMinimum),
        pointer<unsigned int>(minimumSource));
    PDG_CUDA_CHECK(cudaGetLastError());
    materializeMinimumKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        batch.data<double>(Z), pointer<unsigned int>(minimumSource), size,
        pointer<double>(sparseMinimum));
    PDG_CUDA_CHECK(cudaGetLastError());
    int invalidFrameHost = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidFrameHost, invalidFrame->data(),
                                   sizeof(invalidFrameHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidFrameHost != 0)
        throw std::out_of_range(
            "CUDA SMRF point lies outside its raster frame");
    std::unique_ptr<Allocation> minimum = allocate<double>(batch, size);
    fillRasterKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(sparseMinimum), frame, pointer<double>(minimum));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> negative = allocate<double>(batch, size);
    negateKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(minimum), size, pointer<double>(negative));
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> low = progressiveFilter(
        batch, pointer<double>(negative), frame, program.cell, 5.0, 1);

    std::unique_ptr<Allocation> net = allocate<int>(batch, size);
    initializeIntKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<int>(net), size, 0);
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> netMinimum;
    if (program.cut > 0.0)
    {
        const int spacing = static_cast<int>(netSpacingValue);
        netMaskKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            frame.rows, frame.columns, static_cast<std::size_t>(spacing),
            pointer<int>(net));
        PDG_CUDA_CHECK(cudaGetLastError());
        std::unique_ptr<Allocation> opening = morphology(
            batch, pointer<double>(minimum), frame, 2 * spacing, false);
        opening = morphology(batch, pointer<double>(opening), frame,
                             2 * spacing, true);
        netMinimum = allocate<double>(batch, size);
        applyNetKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            pointer<double>(minimum), pointer<double>(opening),
            pointer<int>(net), size, pointer<double>(netMinimum));
        PDG_CUDA_CHECK(cudaGetLastError());
    }
    else
        netMinimum = copyDoubles(batch, pointer<double>(minimum), size);

    const int objectRadius = static_cast<int>(objectRadiusValue);
    std::unique_ptr<Allocation> objects =
        progressiveFilter(batch, pointer<double>(netMinimum), frame,
                          program.cell, program.slope, objectRadius);
    std::unique_ptr<Allocation> sparseProvisional =
        allocate<double>(batch, size);
    provisionalKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(minimum), pointer<int>(low), pointer<int>(net),
        pointer<int>(objects), size, pointer<double>(sparseProvisional));
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> provisional = allocate<double>(batch, size);
    fillRasterKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(sparseProvisional), frame,
        pointer<double>(provisional));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> scaled = allocate<double>(batch, size);
    scaleKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(provisional), size, program.cell,
        pointer<double>(scaled));
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> gradient = allocate<double>(batch, size);
    gradientKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(scaled), frame, pointer<double>(gradient));
    PDG_CUDA_CHECK(cudaGetLastError());
    // The finite-difference output is already dense whenever the provisional
    // raster is dense. Preserve upstream's second inpaint phase exactly for
    // the all-masked edge case as well.
    std::unique_ptr<Allocation> filledGradient = allocate<double>(batch, size);
    fillRasterKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(gradient), frame, pointer<double>(filledGradient));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> counts =
        allocate<unsigned long long>(batch, 2U);
    PDG_CUDA_CHECK(cudaMemsetAsync(counts->data(), 0,
                                   2U * sizeof(unsigned long long), stream));
    classifyKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.data<std::uint8_t>(Classification), batch.size(), frame,
        program.cell, pointer<double>(provisional),
        pointer<double>(filledGradient), program.threshold, program.scalar,
        program.groundClass, program.otherClass, program.onlyGround,
        pointer<unsigned long long>(counts));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned long long hostCounts[2]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostCounts, counts->data(),
                                   sizeof(hostCounts), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return {frame.rows, frame.columns, static_cast<std::size_t>(hostCounts[0]),
            static_cast<std::size_t>(hostCounts[1])};
}

} // namespace pdg
