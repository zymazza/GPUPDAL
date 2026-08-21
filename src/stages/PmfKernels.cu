#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/RasterGrid.hpp>
#include <pdg/stages/Pmf.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

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

struct PmfPass
{
    int radius;
    double threshold;
};

struct DeviceTile
{
    std::size_t coreRow;
    std::size_t coreColumn;
    std::size_t coreRows;
    std::size_t coreColumns;
    std::size_t expandedRow;
    std::size_t expandedColumn;
    std::size_t expandedRows;
    std::size_t expandedColumns;
};

struct ProofCandidate
{
    double distance;
    unsigned long long bits;
    unsigned int found;
    unsigned int ambiguous;
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
        throw std::overflow_error("CUDA PMF allocation size overflows");
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

__device__ bool initialRasterCell(double x, double y, const DeviceFrame& frame,
                                  double cellSize, std::size_t& index)
{
    if (!isfinite(x) || !isfinite(y))
        return false;
    const double columnValue =
        __ddiv_rn(floor(__dsub_rn(x, frame.minimumX)), cellSize);
    const double rowValue =
        __ddiv_rn(floor(__dsub_rn(y, frame.minimumY)), cellSize);
    if (columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        return false;
    index = static_cast<std::size_t>(columnValue) * frame.rows +
            static_cast<std::size_t>(rowValue);
    return true;
}

__device__ bool lookupRasterCell(double x, double y, const DeviceFrame& frame,
                                 double cellSize, std::size_t& index)
{
    if (!isfinite(x) || !isfinite(y))
        return false;
    const double columnValue =
        floor(__ddiv_rn(__dsub_rn(x, frame.minimumX), cellSize));
    const double rowValue =
        floor(__ddiv_rn(__dsub_rn(y, frame.minimumY), cellSize));
    if (columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        return false;
    index = static_cast<std::size_t>(columnValue) * frame.rows +
            static_cast<std::size_t>(rowValue);
    return true;
}

__device__ double rasterCenter(double minimum, std::size_t cell,
                               double cellSize)
{
    return __dadd_rn(
        minimum,
        __dmul_rn(__dadd_rn(static_cast<double>(cell), 0.5), cellSize));
}

__device__ double
rasterCenterDistanceSquared(std::size_t firstColumn, std::size_t firstRow,
                            std::size_t secondColumn, std::size_t secondRow,
                            const DeviceFrame& frame, double cellSize)
{
    const double deltaX =
        __dsub_rn(rasterCenter(frame.minimumX, firstColumn, cellSize),
                  rasterCenter(frame.minimumX, secondColumn, cellSize));
    const double deltaY =
        __dsub_rn(rasterCenter(frame.minimumY, firstRow, cellSize),
                  rasterCenter(frame.minimumY, secondRow, cellSize));
    return __dadd_rn(__dmul_rn(deltaX, deltaX), __dmul_rn(deltaY, deltaY));
}

__global__ void minimumRasterKernel(const double* x, const double* y,
                                    const double* z, std::size_t pointCount,
                                    DeviceFrame frame, double cellSize,
                                    double* minimum, int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        if (!isfinite(z[point]))
        {
            atomicExch(invalidFrame, 1);
            continue;
        }
        std::size_t index = 0U;
        if (!initialRasterCell(x[point], y[point], frame, cellSize, index))
        {
            atomicExch(invalidFrame, 1);
            continue;
        }
        atomicMinimum(minimum + index, z[point]);
    }
}

__global__ void validateLookupKernel(const double* x, const double* y,
                                     std::size_t pointCount, DeviceFrame frame,
                                     double cellSize, int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::size_t index = 0U;
        if (!lookupRasterCell(x[point], y[point], frame, cellSize, index))
            atomicExch(invalidFrame, 1);
    }
}

__global__ void validateTiledFrameKernel(const double* x, const double* y,
                                         std::size_t pointCount,
                                         DeviceFrame frame, double cellSize,
                                         int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::size_t initial = 0U;
        std::size_t lookup = 0U;
        if (!initialRasterCell(x[point], y[point], frame, cellSize, initial) ||
            !lookupRasterCell(x[point], y[point], frame, cellSize, lookup))
            atomicExch(invalidFrame, 1);
    }
}

__global__ void minimumSourceKernel(const double* x, const double* y,
                                    const double* z, std::size_t pointCount,
                                    DeviceFrame frame, double cellSize,
                                    const double* minimum,
                                    unsigned int* sourcePoint)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        std::size_t index = 0U;
        if (initialRasterCell(x[point], y[point], frame, cellSize, index) &&
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

__global__ void compactMinimumSourcesKernel(const unsigned int* sourcePoint,
                                            std::size_t size,
                                            unsigned int* sourceCells,
                                            unsigned int* sourceCount)
{
    for (std::size_t cell = threadIndex(); cell < size; cell += gridStride())
        if (sourcePoint[cell] != 0xffffffffU)
        {
            const unsigned int slot = atomicAdd(sourceCount, 1U);
            sourceCells[slot] = static_cast<unsigned int>(cell);
        }
}

__device__ ProofCandidate mergeProofCandidate(ProofCandidate left,
                                              ProofCandidate right)
{
    if (!left.found)
        return right;
    if (!right.found)
        return left;
    if (right.distance < left.distance)
        return right;
    if (left.distance < right.distance)
        return left;
    left.ambiguous =
        left.ambiguous || right.ambiguous || left.bits != right.bits;
    return left;
}

__global__ void fillNearestProofKernel(double* raster,
                                       const unsigned int* sourcePoint,
                                       const unsigned int* sourceCells,
                                       unsigned int sourceCount,
                                       DeviceFrame frame, double cellSize,
                                       int* ambiguous)
{
    __shared__ double sharedDistance[BlockSize];
    __shared__ unsigned long long sharedBits[BlockSize];
    __shared__ unsigned int sharedFound[BlockSize];
    __shared__ unsigned int sharedAmbiguous[BlockSize];

    const std::size_t size = frame.rows * frame.columns;
    for (std::size_t cell = blockIdx.x; cell < size; cell += gridDim.x)
    {
        const bool occupied = sourcePoint[cell] != 0xffffffffU;
        ProofCandidate candidate{0.0, 0ULL, 0U, 0U};
        if (!occupied)
        {
            const std::size_t column = cell / frame.rows;
            const std::size_t row = cell % frame.rows;
            for (unsigned int slot = threadIdx.x; slot < sourceCount;
                 slot += blockDim.x)
            {
                const std::size_t source = sourceCells[slot];
                const std::size_t sourceColumn = source / frame.rows;
                const std::size_t sourceRow = source % frame.rows;
                const double distance = rasterCenterDistanceSquared(
                    column, row, sourceColumn, sourceRow, frame, cellSize);
                const unsigned long long bits = static_cast<unsigned long long>(
                    __double_as_longlong(raster[source]));
                candidate =
                    mergeProofCandidate(candidate, {distance, bits, 1U, 0U});
            }
        }

        sharedDistance[threadIdx.x] = candidate.distance;
        sharedBits[threadIdx.x] = candidate.bits;
        sharedFound[threadIdx.x] = candidate.found;
        sharedAmbiguous[threadIdx.x] = candidate.ambiguous;
        __syncthreads();

        for (unsigned int stride = BlockSize / 2U; stride != 0U; stride /= 2U)
        {
            if (threadIdx.x < stride)
            {
                ProofCandidate left{
                    sharedDistance[threadIdx.x], sharedBits[threadIdx.x],
                    sharedFound[threadIdx.x], sharedAmbiguous[threadIdx.x]};
                const unsigned int rightIndex = threadIdx.x + stride;
                const ProofCandidate right{
                    sharedDistance[rightIndex], sharedBits[rightIndex],
                    sharedFound[rightIndex], sharedAmbiguous[rightIndex]};
                left = mergeProofCandidate(left, right);
                sharedDistance[threadIdx.x] = left.distance;
                sharedBits[threadIdx.x] = left.bits;
                sharedFound[threadIdx.x] = left.found;
                sharedAmbiguous[threadIdx.x] = left.ambiguous;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0U && !occupied)
        {
            if (!sharedFound[0] || sharedAmbiguous[0])
                atomicExch(ambiguous, 1);
            else
                raster[cell] =
                    __longlong_as_double(static_cast<long long>(sharedBits[0]));
        }
        __syncthreads();
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

__global__ void initializeGroundKernel(std::uint8_t* ground,
                                       std::size_t pointCount)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
        ground[point] = 1U;
}

__global__ void filterGroundKernel(const double* x, const double* y,
                                   const double* z, std::uint8_t* ground,
                                   std::size_t pointCount, DeviceFrame frame,
                                   double cellSize, const double* surface,
                                   double threshold, int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        if (!ground[point])
            continue;
        std::size_t index = 0U;
        if (!lookupRasterCell(x[point], y[point], frame, cellSize, index))
        {
            atomicExch(invalidFrame, 1);
            ground[point] = 0U;
            continue;
        }
        if (!(__dsub_rn(z[point], surface[index]) < threshold))
            ground[point] = 0U;
    }
}

__global__ void filterGroundTileKernel(const double* x, const double* y,
                                       const double* z, std::uint8_t* ground,
                                       std::size_t pointCount,
                                       DeviceFrame frame, double cellSize,
                                       DeviceTile tile, const double* surface,
                                       double threshold, int* invalidFrame)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        if (!ground[point])
            continue;
        std::size_t cell = 0U;
        if (!lookupRasterCell(x[point], y[point], frame, cellSize, cell))
        {
            atomicExch(invalidFrame, 1);
            continue;
        }
        const std::size_t column = cell / frame.rows;
        const std::size_t row = cell % frame.rows;
        if (column < tile.coreColumn ||
            column >= tile.coreColumn + tile.coreColumns ||
            row < tile.coreRow || row >= tile.coreRow + tile.coreRows)
            continue;
        const std::size_t local =
            (column - tile.expandedColumn) * tile.expandedRows +
            (row - tile.expandedRow);
        if (!(__dsub_rn(z[point], surface[local]) < threshold))
            ground[point] = 0U;
    }
}

__global__ void classifyKernel(std::uint8_t* classification,
                               const std::uint8_t* ground,
                               std::size_t pointCount, std::uint8_t groundClass,
                               std::uint8_t otherClass, bool onlyGround,
                               unsigned long long* counts)
{
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        if (ground[point])
        {
            classification[point] = groundClass;
            atomicAdd(counts, 1ULL);
        }
        else
        {
            if (!onlyGround)
                classification[point] = otherClass;
            atomicAdd(counts + 1U, 1ULL);
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

std::vector<PmfPass> makeSchedule(const PmfProgram& program)
{
    std::vector<PmfPass> passes;
    double previousWindow = 0.0;
    for (std::size_t iteration = 0U; previousWindow < program.maxWindowSize;
         ++iteration)
    {
        if (iteration >= PmfExactDeviceMaximumPasses)
            throw std::invalid_argument(
                "CUDA PMF pass count exceeds the exact device envelope");
        const double window =
            program.exponential
                ? program.cellSize *
                      (2.0 * std::pow(2.0, static_cast<int>(iteration)) + 1.0)
                : program.cellSize *
                      (2.0 * static_cast<double>((iteration + 1U) * 2U) + 1.0);
        double threshold = program.initialDistance;
        if (iteration != 0U)
            threshold =
                program.slope * (window - previousWindow) * program.cellSize +
                program.initialDistance;
        if (threshold > program.maxDistance)
            threshold = program.maxDistance;
        const int radius = static_cast<int>(0.5 * (window - 1.0));
        if (radius > PmfExactDeviceMaximumMorphologyRadius)
            throw std::invalid_argument(
                "CUDA PMF radius exceeds the exact device envelope");
        passes.push_back({radius, threshold});
        previousWindow = window;
    }
    return passes;
}

DeviceFrame reduceFrame(PointBatch& batch, double cellSize)
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

    const double columnsValue = (host[1] - host[0]) / cellSize + 1.0;
    const double rowsValue = (host[3] - host[2]) / cellSize + 1.0;
    if (!std::isfinite(columnsValue) || !std::isfinite(rowsValue) ||
        columnsValue < 1.0 || rowsValue < 1.0 ||
        columnsValue > static_cast<double>(PmfExactDeviceMaximumRasterCells) ||
        rowsValue > static_cast<double>(PmfExactDeviceMaximumRasterCells))
        throw std::invalid_argument("CUDA PMF raster dimensions are invalid");
    const DeviceFrame frame{host[0], host[2],
                            static_cast<std::size_t>(rowsValue),
                            static_cast<std::size_t>(columnsValue)};
    if (frame.rows > PmfExactDeviceMaximumRasterCells / frame.columns)
        throw std::invalid_argument(
            "CUDA PMF raster exceeds the exact device envelope");
    return frame;
}

DeviceTile deviceTile(const RasterGridTile& tile) noexcept
{
    return {tile.coreRow,      tile.coreColumn,     tile.coreRows,
            tile.coreColumns,  tile.expandedRow,    tile.expandedColumn,
            tile.expandedRows, tile.expandedColumns};
}
} // unnamed namespace

PmfResult classifyPmfDevice(PointBatch& batch, const PmfProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA PMF requires a device batch");
    // CUB's reduction overload below accepts a signed-int item count.
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA PMF point count exceeds the exact device envelope");
    if (!pmfProgramWithinExactDeviceEnvelope(program))
        throw std::invalid_argument(
            "CUDA PMF program exceeds the exact device envelope");
    const std::vector<PmfPass> passes = makeSchedule(program);

    NvtxRange range("pdg::filters.pmf");
    const DeviceFrame frame = reduceFrame(batch, program.cellSize);
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
        batch.size(), frame, program.cellSize, pointer<double>(sparseMinimum),
        pointer<int>(invalidFrame));
    PDG_CUDA_CHECK(cudaGetLastError());
    validateLookupKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.size(), frame,
        program.cellSize, pointer<int>(invalidFrame));
    PDG_CUDA_CHECK(cudaGetLastError());
    std::unique_ptr<Allocation> minimumSource =
        allocate<unsigned int>(batch, size);
    PDG_CUDA_CHECK(cudaMemsetAsync(minimumSource->data(), 0xff,
                                   size * sizeof(unsigned int), stream));
    minimumSourceKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), frame, program.cellSize, pointer<double>(sparseMinimum),
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
        throw std::out_of_range("CUDA PMF point lies outside its raster frame");

    std::unique_ptr<Allocation> sourceCells =
        allocate<unsigned int>(batch, size);
    std::unique_ptr<Allocation> sourceCount = allocate<unsigned int>(batch, 1U);
    std::unique_ptr<Allocation> ambiguous = allocate<int>(batch, 1U);
    PDG_CUDA_CHECK(
        cudaMemsetAsync(sourceCount->data(), 0, sizeof(unsigned int), stream));
    PDG_CUDA_CHECK(cudaMemsetAsync(ambiguous->data(), 0, sizeof(int), stream));
    compactMinimumSourcesKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<unsigned int>(minimumSource), size,
        pointer<unsigned int>(sourceCells), pointer<unsigned int>(sourceCount));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned int sourceCountHost = 0U;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&sourceCountHost, sourceCount->data(),
                                   sizeof(sourceCountHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (sourceCountHost == 0U)
        throw std::logic_error("CUDA PMF raster proof found no sources");
    fillNearestProofKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
        pointer<double>(sparseMinimum), pointer<unsigned int>(minimumSource),
        pointer<unsigned int>(sourceCells), sourceCountHost, frame,
        program.cellSize, pointer<int>(ambiguous));
    PDG_CUDA_CHECK(cudaGetLastError());
    int ambiguousHost = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&ambiguousHost, ambiguous->data(),
                                   sizeof(ambiguousHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ambiguousHost != 0)
        throw PmfRasterTieError();
    std::unique_ptr<Allocation> minimum = std::move(sparseMinimum);
    std::unique_ptr<Allocation> ground =
        allocate<std::uint8_t>(batch, batch.size());
    initializeGroundKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                             stream>>>(pointer<std::uint8_t>(ground),
                                       batch.size());
    PDG_CUDA_CHECK(cudaGetLastError());

    for (const PmfPass& pass : passes)
    {
        minimum = morphology(batch, pointer<double>(minimum), frame,
                             pass.radius, false);
        minimum = morphology(batch, pointer<double>(minimum), frame,
                             pass.radius, true);
        filterGroundKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                             stream>>>(
            batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
            pointer<std::uint8_t>(ground), batch.size(), frame,
            program.cellSize, pointer<double>(minimum), pass.threshold,
            pointer<int>(invalidFrame));
        PDG_CUDA_CHECK(cudaGetLastError());
    }

    std::unique_ptr<Allocation> counts =
        allocate<unsigned long long>(batch, 2U);
    PDG_CUDA_CHECK(cudaMemsetAsync(counts->data(), 0,
                                   2U * sizeof(unsigned long long), stream));
    classifyKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<std::uint8_t>(Classification), pointer<std::uint8_t>(ground),
        batch.size(), program.groundClass, program.otherClass,
        program.onlyGround, pointer<unsigned long long>(counts));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned long long hostCounts[2]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostCounts, counts->data(),
                                   sizeof(hostCounts), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidFrameHost, invalidFrame->data(),
                                   sizeof(invalidFrameHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidFrameHost != 0)
        throw std::out_of_range("CUDA PMF point lies outside its raster frame");
    return {frame.rows, frame.columns, static_cast<std::size_t>(hostCounts[0]),
            static_cast<std::size_t>(hostCounts[1])};
}

void buildPmfTiledRasterDevice(PointBatch& deviceBatch,
                               const PmfProgram& program,
                               RasterGridProduct& product,
                               PmfRasterBuildFacts* facts)
{
    if (facts)
        *facts = {};
    if (deviceBatch.memoryKind() != MemoryKind::Device ||
        &deviceBatch.memoryResource() != &product.executionMemory())
        throw std::invalid_argument(
            "CUDA PMF raster proof requires its planner-owned device resource");
    if (!deviceBatch.size() ||
        deviceBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        !deviceBatch.has(X) || !deviceBatch.has(Y) || !deviceBatch.has(Z) ||
        deviceBatch.columnInfo(X).physicalType != DimensionType::Double ||
        deviceBatch.columnInfo(Y).physicalType != DimensionType::Double ||
        deviceBatch.columnInfo(Z).physicalType != DimensionType::Double ||
        !pmfProgramWithinExactDeviceEnvelope(program))
        throw std::invalid_argument(
            "CUDA PMF raster proof input is outside the exact envelope");

    const RasterGridFrame& rasterFrame = product.frame();
    if (rasterFrame.policy != RasterGridFramePolicy::PmfV1 ||
        rasterFrame.cellSize != program.cellSize || !rasterFrame.size() ||
        rasterFrame.size() > static_cast<std::size_t>(
                                 (std::numeric_limits<unsigned int>::max)()) ||
        product.hasPendingRasterBuild() ||
        product.deviceProofWorkspaceBytes() !=
            rasterFrame.size() * PmfTiledDeviceProofBytesPerCell ||
        product.backingBytes() != rasterFrame.size() * sizeof(double))
        throw std::invalid_argument(
            "CUDA PMF raster proof product is outside the exact envelope");
    if (!product.materializeDeviceProofWorkspace())
        throw std::invalid_argument(
            "CUDA PMF raster proof workspace exceeds its planner budget");

    try
    {
        NvtxRange range("pdg::filters.pmf.raster_proof");
        const DeviceFrame frame{rasterFrame.minimumX, rasterFrame.minimumY,
                                rasterFrame.rows, rasterFrame.columns};
        const std::size_t size = rasterFrame.size();
        auto* workspace =
            static_cast<std::byte*>(product.deviceProofWorkspace());
        auto* raster = reinterpret_cast<double*>(workspace);
        auto* sourcePoint =
            reinterpret_cast<unsigned int*>(workspace + size * sizeof(double));
        auto* sourceCells = sourcePoint + size;
        const cudaStream_t stream =
            static_cast<cudaStream_t>(deviceBatch.nativeStreamHandle());
        std::unique_ptr<Allocation> status =
            allocate<unsigned int>(deviceBatch, 3U);
        auto* statusValues = pointer<unsigned int>(status);

        PDG_CUDA_CHECK(cudaMemsetAsync(status->data(), 0,
                                       3U * sizeof(unsigned int), stream));
        PDG_CUDA_CHECK(cudaMemsetAsync(sourcePoint, 0xff,
                                       size * sizeof(unsigned int), stream));
        initializeDoubleKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            raster, size, std::numeric_limits<double>::quiet_NaN());
        PDG_CUDA_CHECK(cudaGetLastError());
        minimumRasterKernel<<<launchBlocks(deviceBatch.size()), BlockSize, 0,
                              stream>>>(
            deviceBatch.data<double>(X), deviceBatch.data<double>(Y),
            deviceBatch.data<double>(Z), deviceBatch.size(), frame,
            program.cellSize, raster, reinterpret_cast<int*>(statusValues));
        PDG_CUDA_CHECK(cudaGetLastError());
        validateLookupKernel<<<launchBlocks(deviceBatch.size()), BlockSize, 0,
                               stream>>>(
            deviceBatch.data<double>(X), deviceBatch.data<double>(Y),
            deviceBatch.size(), frame, program.cellSize,
            reinterpret_cast<int*>(statusValues));
        PDG_CUDA_CHECK(cudaGetLastError());
        minimumSourceKernel<<<launchBlocks(deviceBatch.size()), BlockSize, 0,
                              stream>>>(
            deviceBatch.data<double>(X), deviceBatch.data<double>(Y),
            deviceBatch.data<double>(Z), deviceBatch.size(), frame,
            program.cellSize, raster, sourcePoint);
        PDG_CUDA_CHECK(cudaGetLastError());
        materializeMinimumKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            deviceBatch.data<double>(Z), sourcePoint, size, raster);
        PDG_CUDA_CHECK(cudaGetLastError());
        compactMinimumSourcesKernel<<<launchBlocks(size), BlockSize, 0,
                                      stream>>>(sourcePoint, size, sourceCells,
                                                statusValues + 1U);
        PDG_CUDA_CHECK(cudaGetLastError());

        std::array<unsigned int, 2U> hostStatus{};
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus.data(), statusValues,
                                       hostStatus.size() * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (hostStatus[0] != 0U)
            throw std::out_of_range(
                "CUDA PMF point lies outside its raster frame");
        if (hostStatus[1] == 0U)
            throw std::logic_error("CUDA PMF raster proof found no sources");

        fillNearestProofKernel<<<launchBlocks(size), BlockSize, 0, stream>>>(
            raster, sourcePoint, sourceCells, hostStatus[1], frame,
            program.cellSize, reinterpret_cast<int*>(statusValues + 2U));
        PDG_CUDA_CHECK(cudaGetLastError());
        unsigned int ambiguous = 0U;
        PDG_CUDA_CHECK(cudaMemcpyAsync(&ambiguous, statusValues + 2U,
                                       sizeof(ambiguous),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (ambiguous != 0U)
            throw PmfRasterTieError();

        if (facts)
        {
            facts->populatedCells = hostStatus[1];
            const std::size_t voidCells = size - hostStatus[1];
            facts->sourceSlotsVisited =
                hostStatus[1] != 0U &&
                        voidCells > (std::numeric_limits<std::size_t>::max)() /
                                        hostStatus[1]
                    ? (std::numeric_limits<std::size_t>::max)()
                    : voidCells * hostStatus[1];
            facts->deviceNativeSourceBuild = true;
            facts->usedDeviceTieProof = true;
        }
        product.publishDeviceRasterBuild();
    }
    catch (...)
    {
        if (product.hasDeviceProofWorkspace())
            product.discardDeviceProofWorkspace();
        throw;
    }
}

PmfResult classifyPmfTiledDevice(PointBatch& batch, const PmfProgram& program,
                                 RasterGridProduct& product,
                                 PmfTiledExecutionFacts* facts)
{
    if (facts)
        *facts = {};
    if (batch.memoryKind() != MemoryKind::Device ||
        &batch.memoryResource() != &product.executionMemory())
        throw std::invalid_argument(
            "CUDA tiled PMF requires its planner-owned device resource");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA tiled PMF point count exceeds the exact device envelope");
    if (!pmfProgramWithinExactDeviceEnvelope(program))
        throw std::invalid_argument(
            "CUDA tiled PMF program exceeds the exact device envelope");
    const RasterGridFrame& rasterFrame = product.frame();
    if (rasterFrame.policy != RasterGridFramePolicy::PmfV1 ||
        rasterFrame.cellSize != program.cellSize || !rasterFrame.size() ||
        rasterFrame.rows >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        rasterFrame.columns >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        product.backingBytes() != rasterFrame.size() * sizeof(double) ||
        (product.hasResidentDeviceBackings() &&
         (product.deviceBackingBytes() != product.backingBytes() ||
          !product.currentDeviceBacking() || !product.nextDeviceBacking())) ||
        product.tileScratchBytes() <
            product.tiles().peakExpandedCellCount() * sizeof(double))
        throw std::invalid_argument(
            "CUDA tiled PMF raster product is outside the exact envelope");

    NvtxRange range("pdg::filters.pmf.tiled");
    const DeviceFrame frame{rasterFrame.minimumX, rasterFrame.minimumY,
                            rasterFrame.rows, rasterFrame.columns};
    const std::vector<PmfPass> passes = makeSchedule(program);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::unique_ptr<Allocation> invalidFrame = allocate<int>(batch, 1U);
    PDG_CUDA_CHECK(
        cudaMemsetAsync(invalidFrame->data(), 0, sizeof(int), stream));
    validateTiledFrameKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                               stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.size(), frame,
        program.cellSize, pointer<int>(invalidFrame));
    PDG_CUDA_CHECK(cudaGetLastError());
    int invalidFrameHost = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidFrameHost, invalidFrame->data(),
                                   sizeof(invalidFrameHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidFrameHost != 0)
        throw std::out_of_range(
            "CUDA tiled PMF point lies outside its raster frame");

    product.consumeRasterBuild();
    PmfTiledExecutionFacts observed;
    observed.deviceResidentPhases = product.hasResidentDeviceBackings();
    observed.deviceNativeRaster = product.deviceRasterBuild();
    const auto recordTransfer = [](std::size_t& transfers, std::size_t& bytes,
                                   std::size_t transferredBytes)
    {
        if (transfers != (std::numeric_limits<std::size_t>::max)())
            ++transfers;
        if (transferredBytes >
            (std::numeric_limits<std::size_t>::max)() - bytes)
            bytes = (std::numeric_limits<std::size_t>::max)();
        else
            bytes += transferredBytes;
    };
    const std::size_t peakTileCells = product.tiles().peakExpandedCellCount();
    std::unique_ptr<Allocation> currentTile;
    std::unique_ptr<Allocation> nextTile;
    if (!observed.deviceResidentPhases)
    {
        currentTile = allocate<double>(batch, peakTileCells);
        nextTile = allocate<double>(batch, peakTileCells);
    }

    auto current = std::span<double>(
        static_cast<double*>(product.currentBacking()), rasterFrame.size());
    auto next = std::span<double>(static_cast<double*>(product.nextBacking()),
                                  rasterFrame.size());
    auto scratch = std::span<double>(
        static_cast<double*>(product.tileScratch()), peakTileCells);

    if (observed.deviceResidentPhases && !observed.deviceNativeRaster)
    {
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            product.currentDeviceBacking(), product.currentBacking(),
            product.backingBytes(), cudaMemcpyHostToDevice, stream));
        recordTransfer(observed.rasterHostToDeviceTransfers,
                       observed.rasterHostToDeviceBytes,
                       product.backingBytes());
    }

    std::unique_ptr<Allocation> ground =
        allocate<std::uint8_t>(batch, batch.size());
    initializeGroundKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                             stream>>>(pointer<std::uint8_t>(ground),
                                       batch.size());
    PDG_CUDA_CHECK(cudaGetLastError());

    const auto morphologyPhase = [&](int iterations, bool dilation)
    {
        if (observed.deviceResidentPhases)
        {
            for (int iteration = 0; iteration < iterations; ++iteration)
            {
                morphKernel<<<launchBlocks(rasterFrame.size()), BlockSize, 0,
                              stream>>>(
                    static_cast<const double*>(product.currentDeviceBacking()),
                    rasterFrame.rows, rasterFrame.columns, dilation,
                    static_cast<double*>(product.nextDeviceBacking()));
                PDG_CUDA_CHECK(cudaGetLastError());
                product.swapDeviceBackings();
            }
            return;
        }
        for (int iteration = 0; iteration < iterations; ++iteration)
        {
            current = std::span<double>(
                static_cast<double*>(product.currentBacking()),
                rasterFrame.size());
            next =
                std::span<double>(static_cast<double*>(product.nextBacking()),
                                  rasterFrame.size());
            for (const RasterGridTile& tile : product.tiles().tiles())
            {
                const std::size_t cells = tile.expandedCellCount();
                gatherRasterGridExpanded(tile, std::span<const double>(current),
                                         scratch);
                PDG_CUDA_CHECK(cudaMemcpyAsync(
                    currentTile->data(), scratch.data(), cells * sizeof(double),
                    cudaMemcpyHostToDevice, stream));
                recordTransfer(observed.rasterHostToDeviceTransfers,
                               observed.rasterHostToDeviceBytes,
                               cells * sizeof(double));
                morphKernel<<<launchBlocks(cells), BlockSize, 0, stream>>>(
                    pointer<double>(currentTile), tile.expandedRows,
                    tile.expandedColumns, dilation, pointer<double>(nextTile));
                PDG_CUDA_CHECK(cudaGetLastError());
                PDG_CUDA_CHECK(cudaMemcpyAsync(scratch.data(), nextTile->data(),
                                               cells * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
                recordTransfer(observed.rasterDeviceToHostTransfers,
                               observed.rasterDeviceToHostBytes,
                               cells * sizeof(double));
                PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                mosaicRasterGridOwned(
                    tile, std::span<const double>(scratch.data(), cells), next);
            }
            product.swapBackings();
        }
    };

    for (const PmfPass& pass : passes)
    {
        morphologyPhase(pass.radius, false);
        morphologyPhase(pass.radius, true);
        if (observed.deviceResidentPhases)
        {
            filterGroundKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                 stream>>>(
                batch.data<double>(X), batch.data<double>(Y),
                batch.data<double>(Z), pointer<std::uint8_t>(ground),
                batch.size(), frame, program.cellSize,
                static_cast<const double*>(product.currentDeviceBacking()),
                pass.threshold, pointer<int>(invalidFrame));
            PDG_CUDA_CHECK(cudaGetLastError());
            continue;
        }
        current = std::span<double>(
            static_cast<double*>(product.currentBacking()), rasterFrame.size());
        for (const RasterGridTile& tile : product.tiles().tiles())
        {
            const std::size_t cells = tile.expandedCellCount();
            gatherRasterGridExpanded(tile, std::span<const double>(current),
                                     scratch);
            PDG_CUDA_CHECK(cudaMemcpyAsync(currentTile->data(), scratch.data(),
                                           cells * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
            recordTransfer(observed.rasterHostToDeviceTransfers,
                           observed.rasterHostToDeviceBytes,
                           cells * sizeof(double));
            filterGroundTileKernel<<<launchBlocks(batch.size()), BlockSize, 0,
                                     stream>>>(
                batch.data<double>(X), batch.data<double>(Y),
                batch.data<double>(Z), pointer<std::uint8_t>(ground),
                batch.size(), frame, program.cellSize, deviceTile(tile),
                pointer<double>(currentTile), pass.threshold,
                pointer<int>(invalidFrame));
            PDG_CUDA_CHECK(cudaGetLastError());
            // The pinned scratch is reused for the next tile, so its async
            // upload must be complete before the host overwrites it.
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    std::unique_ptr<Allocation> counts =
        allocate<unsigned long long>(batch, 2U);
    PDG_CUDA_CHECK(cudaMemsetAsync(counts->data(), 0,
                                   2U * sizeof(unsigned long long), stream));
    classifyKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<std::uint8_t>(Classification), pointer<std::uint8_t>(ground),
        batch.size(), program.groundClass, program.otherClass,
        program.onlyGround, pointer<unsigned long long>(counts));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned long long hostCounts[2]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(hostCounts, counts->data(),
                                   sizeof(hostCounts), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidFrameHost, invalidFrame->data(),
                                   sizeof(invalidFrameHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidFrameHost != 0)
        throw std::out_of_range(
            "CUDA tiled PMF point lies outside its raster frame");
    if (facts)
        *facts = observed;
    return {frame.rows, frame.columns, static_cast<std::size_t>(hostCounts[0]),
            static_cast<std::size_t>(hostCounts[1])};
}

} // namespace pdg
