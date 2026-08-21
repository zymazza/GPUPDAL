/******************************************************************************
 * Exact bounded CUDA CSF lane derived from the Apache-2.0 CSF implementation
 * under filters/private/csf and PDAL's BSD-licensed filters.csf wrapper.
 * Copyright (c) 2017 State Key Laboratory of Remote Sensing Science,
 * Institute of Remote Sensing Science and Engineering,
 * Beijing Normal University
 * Copyright (c) 2019 Bradley J Chambers
 * Copyright (c) 2026 PDAL-GPU contributors
 ******************************************************************************/

#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
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
constexpr int BlockSize = 256;
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId Classification(StandardDimension::Classification);
constexpr double MinimumHeight = -9999999999.0;
constexpr double MaximumDistance = 9999999999.0;

__device__ __constant__ double SingleMove[15] = {
    0.0,     0.3,     0.51,    0.657,   0.7599,  0.83193, 0.88235, 0.91765,
    0.94235, 0.95965, 0.97175, 0.98023, 0.98616, 0.99031, 0.99322};
__device__ __constant__ double DoubleMove[15] = {
    0.0,    0.3,    0.42,   0.468, 0.4872, 0.4949, 0.498, 0.4992,
    0.4997, 0.4999, 0.4999, 0.5,   0.5,    0.5,    0.5};

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
    double originX = 0.0;
    double originY = 0.0;
    double originZ = 0.0;
    std::size_t width = 0U;
    std::size_t height = 0U;
};

template <typename T>
std::unique_ptr<Allocation> allocate(PointBatch& batch, std::size_t count)
{
    if (count && count > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
        throw std::overflow_error("CUDA CSF allocation size overflows");
    return batch.memoryResource().allocate(count * sizeof(T), alignof(T));
}

template <typename T> T* pointer(const std::unique_ptr<Allocation>& allocation)
{
    return static_cast<T*>(allocation->data());
}

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__device__ std::size_t threadIndex()
{
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ std::size_t gridStride()
{
    return static_cast<std::size_t>(blockDim.x) * gridDim.x;
}

__device__ double add(double first, double second)
{
    return __dadd_rn(first, second);
}

__device__ double subtract(double first, double second)
{
    return __dsub_rn(first, second);
}

__device__ double multiply(double first, double second)
{
    return __dmul_rn(first, second);
}

__device__ double divide(double first, double second)
{
    return __ddiv_rn(first, second);
}

__device__ std::size_t cell(std::size_t column, std::size_t row,
                            const DeviceFrame& frame)
{
    return row * frame.width + column;
}

__global__ void frameKernel(const double* x, const double* y, const double* z,
                            std::size_t pointCount, double resolution,
                            DeviceFrame* output, int* invalid)
{
    if (blockIdx.x || threadIdx.x)
        return;
    if (!isfinite(x[0]) || !isfinite(y[0]) || !isfinite(z[0]))
    {
        *invalid = 1;
        return;
    }
    double minimumX = x[0];
    double maximumX = x[0];
    double minimumY = y[0];
    double maximumY = y[0];
    double minimumInvertedZ = -z[0];
    double maximumInvertedZ = -z[0];
    for (std::size_t point = 1U; point < pointCount; ++point)
    {
        if (!isfinite(x[point]) || !isfinite(y[point]) || !isfinite(z[point]))
        {
            *invalid = 1;
            return;
        }
        if (x[point] < minimumX)
            minimumX = x[point];
        else if (x[point] > maximumX)
            maximumX = x[point];
        if (y[point] < minimumY)
            minimumY = y[point];
        else if (y[point] > maximumY)
            maximumY = y[point];
        const double invertedZ = -z[point];
        if (invertedZ < minimumInvertedZ)
            minimumInvertedZ = invertedZ;
        else if (invertedZ > maximumInvertedZ)
            maximumInvertedZ = invertedZ;
    }
    const double widthValue =
        add(floor(divide(subtract(maximumX, minimumX), resolution)), 4.0);
    const double heightValue =
        add(floor(divide(subtract(maximumY, minimumY), resolution)), 4.0);
    if (!isfinite(widthValue) || !isfinite(heightValue) || widthValue < 1.0 ||
        heightValue < 1.0 ||
        widthValue > static_cast<double>(CsfExactDeviceMaximumClothCells) ||
        heightValue > static_cast<double>(CsfExactDeviceMaximumClothCells))
    {
        *invalid = 1;
        return;
    }
    DeviceFrame frame;
    frame.originX = subtract(minimumX, multiply(2.0, resolution));
    frame.originY = add(maximumInvertedZ, 0.05);
    frame.originZ = subtract(minimumY, multiply(2.0, resolution));
    frame.width = static_cast<std::size_t>(widthValue);
    frame.height = static_cast<std::size_t>(heightValue);
    if (!isfinite(frame.originX) || !isfinite(frame.originY) ||
        !isfinite(frame.originZ) || !frame.width || !frame.height ||
        frame.width > CsfExactDeviceMaximumClothCells / frame.height)
    {
        *invalid = 1;
        return;
    }
    *output = frame;
}

__device__ double
breadthFirstHeight(std::size_t root, const unsigned int* offsets,
                   const unsigned int* neighbors, double* nearestHeight,
                   unsigned char* visited, unsigned int* queue,
                   unsigned int* backlist, std::size_t queueCapacity,
                   int* invalid)
{
    std::size_t head = 0U;
    std::size_t tail = 0U;
    std::size_t backCount = 0U;
    for (unsigned int index = offsets[root]; index < offsets[root + 1U];
         ++index)
    {
        visited[root] = 1U;
        if (tail >= queueCapacity)
        {
            *invalid = 1;
            return MinimumHeight;
        }
        queue[tail++] = neighbors[index];
    }
    while (head < tail)
    {
        const unsigned int current = queue[head++];
        if (backCount >= queueCapacity)
        {
            *invalid = 1;
            return MinimumHeight;
        }
        backlist[backCount++] = current;
        if (nearestHeight[current] > MinimumHeight)
        {
            for (std::size_t index = 0U; index < backCount; ++index)
                visited[backlist[index]] = 0U;
            for (std::size_t index = head; index < tail; ++index)
                visited[queue[index]] = 0U;
            return nearestHeight[current];
        }
        for (unsigned int index = offsets[current];
             index < offsets[current + 1U]; ++index)
        {
            const unsigned int neighbor = neighbors[index];
            if (!visited[neighbor])
            {
                visited[neighbor] = 1U;
                if (tail >= queueCapacity)
                {
                    *invalid = 1;
                    return MinimumHeight;
                }
                queue[tail++] = neighbor;
            }
        }
    }
    return MinimumHeight;
}

__global__ void rasterKernel(const double* x, const double* y, const double* z,
                             std::size_t pointCount, DeviceFrame frame,
                             double resolution, const unsigned int* offsets,
                             const unsigned int* neighbors, double* position,
                             double* oldPosition, double* nearestHeight,
                             double* temporaryDistance, double* heights,
                             unsigned char* movable, unsigned char* visited,
                             unsigned int* queue, unsigned int* backlist,
                             int* invalid)
{
    if (blockIdx.x || threadIdx.x)
        return;
    const std::size_t size = frame.width * frame.height;
    for (std::size_t index = 0U; index < size; ++index)
    {
        position[index] = frame.originY;
        oldPosition[index] = frame.originY;
        nearestHeight[index] = MinimumHeight;
        temporaryDistance[index] = MaximumDistance;
        movable[index] = 1U;
        visited[index] = 0U;
    }
    for (std::size_t point = 0U; point < pointCount; ++point)
    {
        const double deltaX = subtract(x[point], frame.originX);
        const double deltaZ = subtract(y[point], frame.originZ);
        const int column =
            static_cast<int>(add(divide(deltaX, resolution), 0.5));
        const int row = static_cast<int>(add(divide(deltaZ, resolution), 0.5));
        if (column < 0 || row < 0 ||
            static_cast<std::size_t>(column) >= frame.width ||
            static_cast<std::size_t>(row) >= frame.height)
        {
            *invalid = 1;
            return;
        }
        const std::size_t index = cell(static_cast<std::size_t>(column),
                                       static_cast<std::size_t>(row), frame);
        const double particleX = add(
            frame.originX, multiply(static_cast<double>(column), resolution));
        const double particleZ =
            add(frame.originZ, multiply(static_cast<double>(row), resolution));
        const double dx = subtract(x[point], particleX);
        const double dz = subtract(y[point], particleZ);
        const double distance = add(multiply(dx, dx), multiply(dz, dz));
        if (distance < temporaryDistance[index])
        {
            temporaryDistance[index] = distance;
            nearestHeight[index] = -z[point];
        }
    }
    const std::size_t queueCapacity = size * 2U;
    for (std::size_t index = 0U; index < size; ++index)
    {
        if (nearestHeight[index] > MinimumHeight)
        {
            heights[index] = nearestHeight[index];
            continue;
        }
        const std::size_t column = index % frame.width;
        const std::size_t row = index / frame.width;
        double value = MinimumHeight;
        for (std::size_t candidate = column + 1U; candidate < frame.width;
             ++candidate)
            if (nearestHeight[cell(candidate, row, frame)] > MinimumHeight)
            {
                value = nearestHeight[cell(candidate, row, frame)];
                break;
            }
        if (value <= MinimumHeight)
            for (std::size_t candidate = column; candidate-- > 0U;)
                if (nearestHeight[cell(candidate, row, frame)] > MinimumHeight)
                {
                    value = nearestHeight[cell(candidate, row, frame)];
                    break;
                }
        if (value <= MinimumHeight)
            for (std::size_t candidate = row; candidate-- > 0U;)
                if (nearestHeight[cell(column, candidate, frame)] >
                    MinimumHeight)
                {
                    value = nearestHeight[cell(column, candidate, frame)];
                    break;
                }
        if (value <= MinimumHeight)
            for (std::size_t candidate = row + 1U; candidate < frame.height;
                 ++candidate)
                if (nearestHeight[cell(column, candidate, frame)] >
                    MinimumHeight)
                {
                    value = nearestHeight[cell(column, candidate, frame)];
                    break;
                }
        if (value <= MinimumHeight)
            value = breadthFirstHeight(index, offsets, neighbors, nearestHeight,
                                       visited, queue, backlist, queueCapacity,
                                       invalid);
        heights[index] = value;
        if (*invalid)
            return;
    }
}

__global__ void simulationKernel(DeviceFrame frame, CsfProgram program,
                                 const unsigned int* offsets,
                                 const unsigned int* neighbors,
                                 const double* heights, double* position,
                                 double* oldPosition, unsigned char* movable,
                                 unsigned int* iterationsExecuted,
                                 const int* invalid)
{
    if (blockIdx.x || threadIdx.x)
        return;
    if (*invalid)
        return;
    const std::size_t size = frame.width * frame.height;
    const double stepSquared = multiply(program.timeStep, program.timeStep);
    const double acceleration = multiply(-0.2, stepSquared);
    const double doubleCoefficient =
        program.rigidness > 14 ? 0.5 : DoubleMove[program.rigidness];
    const double singleCoefficient =
        program.rigidness > 14 ? 1.0 : SingleMove[program.rigidness];
    *iterationsExecuted = 0U;
    for (int iteration = 0; iteration < program.iterations; ++iteration)
    {
        ++*iterationsExecuted;
        for (std::size_t index = 0U; index < size; ++index)
            if (movable[index])
            {
                const double previous = position[index];
                const double velocity = multiply(
                    subtract(position[index], oldPosition[index]), 0.99);
                const double force = multiply(acceleration, stepSquared);
                position[index] = add(add(position[index], velocity), force);
                oldPosition[index] = previous;
            }
        for (std::size_t first = 0U; first < size; ++first)
            for (unsigned int offset = offsets[first];
                 offset < offsets[first + 1U]; ++offset)
            {
                const std::size_t second = neighbors[offset];
                const double correction =
                    subtract(position[second], position[first]);
                if (movable[first] && movable[second])
                {
                    const double half = multiply(correction, doubleCoefficient);
                    position[first] = add(position[first], half);
                    position[second] = add(position[second], -half);
                }
                else if (movable[first] && !movable[second])
                    position[first] =
                        add(position[first],
                            multiply(correction, singleCoefficient));
                else if (!movable[first] && movable[second])
                    position[second] =
                        add(position[second],
                            -multiply(correction, singleCoefficient));
            }
        double maximumDifference = 0.0;
        for (std::size_t index = 0U; index < size; ++index)
            if (movable[index])
            {
                const double difference =
                    fabs(subtract(oldPosition[index], position[index]));
                if (difference > maximumDifference)
                    maximumDifference = difference;
            }
        for (std::size_t index = 0U; index < size; ++index)
            if (position[index] < heights[index])
            {
                if (movable[index])
                    position[index] =
                        add(position[index],
                            subtract(heights[index], position[index]));
                movable[index] = 0U;
            }
        if (maximumDifference != 0.0 && maximumDifference < 0.005)
            break;
    }
}

__global__ void classificationKernel(const double* x, const double* y,
                                     const double* z,
                                     std::uint8_t* classification,
                                     std::size_t pointCount, DeviceFrame frame,
                                     CsfProgram program, const double* position,
                                     unsigned long long* counts, int* invalid)
{
    if (*invalid)
        return;
    for (std::size_t point = threadIndex(); point < pointCount;
         point += gridStride())
    {
        const double pointY = -z[point];
        const double deltaX = subtract(x[point], frame.originX);
        const double deltaZ = subtract(y[point], frame.originZ);
        const int column0 =
            static_cast<int>(divide(deltaX, program.resolution));
        const int row0 = static_cast<int>(divide(deltaZ, program.resolution));
        if (column0 < 0 || row0 < 0 ||
            static_cast<std::size_t>(column0 + 1) >= frame.width ||
            static_cast<std::size_t>(row0 + 1) >= frame.height)
        {
            atomicExch(invalid, 1);
            continue;
        }
        const double subdeltaX =
            divide(subtract(deltaX, multiply(static_cast<double>(column0),
                                             program.resolution)),
                   program.resolution);
        const double subdeltaZ =
            divide(subtract(deltaZ, multiply(static_cast<double>(row0),
                                             program.resolution)),
                   program.resolution);
        const std::size_t c0 = static_cast<std::size_t>(column0);
        const std::size_t r0 = static_cast<std::size_t>(row0);
        const double oneMinusX = subtract(1.0, subdeltaX);
        const double oneMinusZ = subtract(1.0, subdeltaZ);
        const double term0 = multiply(
            multiply(position[cell(c0, r0, frame)], oneMinusX), oneMinusZ);
        const double term1 = multiply(
            multiply(position[cell(c0, r0 + 1U, frame)], oneMinusX), subdeltaZ);
        const double term2 = multiply(
            multiply(position[cell(c0 + 1U, r0 + 1U, frame)], subdeltaX),
            subdeltaZ);
        const double term3 = multiply(
            multiply(position[cell(c0 + 1U, r0, frame)], subdeltaX), oneMinusZ);
        const double interpolated = add(add(add(term0, term1), term2), term3);
        const bool ground =
            fabs(subtract(interpolated, pointY)) < program.classThreshold;
        if (ground)
        {
            classification[point] = program.groundClass;
            atomicAdd(counts, 1ULL);
        }
        else
        {
            if (!program.onlyGround)
                classification[point] = program.otherClass;
            atomicAdd(counts + 1U, 1ULL);
        }
    }
}

void addConstraint(std::vector<std::vector<unsigned int>>& adjacency,
                   std::size_t first, std::size_t second)
{
    adjacency[first].push_back(static_cast<unsigned int>(second));
    adjacency[second].push_back(static_cast<unsigned int>(first));
}

void makeAdjacency(const DeviceFrame& frame, std::vector<unsigned int>& offsets,
                   std::vector<unsigned int>& neighbors)
{
    const auto index = [&](std::size_t column, std::size_t row)
    { return row * frame.width + column; };
    std::vector<std::vector<unsigned int>> adjacency(frame.width *
                                                     frame.height);
    for (std::size_t column = 0U; column < frame.width; ++column)
        for (std::size_t row = 0U; row < frame.height; ++row)
        {
            const std::size_t current = index(column, row);
            if (column + 1U < frame.width)
                addConstraint(adjacency, current, index(column + 1U, row));
            if (row + 1U < frame.height)
                addConstraint(adjacency, current, index(column, row + 1U));
            if (column + 1U < frame.width && row + 1U < frame.height)
                addConstraint(adjacency, current, index(column + 1U, row + 1U));
            if (column + 1U < frame.width && row + 1U < frame.height)
                addConstraint(adjacency, index(column + 1U, row),
                              index(column, row + 1U));
        }
    for (std::size_t column = 0U; column < frame.width; ++column)
        for (std::size_t row = 0U; row < frame.height; ++row)
        {
            const std::size_t current = index(column, row);
            if (column + 2U < frame.width)
                addConstraint(adjacency, current, index(column + 2U, row));
            if (row + 2U < frame.height)
                addConstraint(adjacency, current, index(column, row + 2U));
            if (column + 2U < frame.width && row + 2U < frame.height)
                addConstraint(adjacency, current, index(column + 2U, row + 2U));
            if (column + 2U < frame.width && row + 2U < frame.height)
                addConstraint(adjacency, index(column + 2U, row),
                              index(column, row + 2U));
        }
    offsets.resize(adjacency.size() + 1U);
    for (std::size_t current = 0U; current < adjacency.size(); ++current)
    {
        offsets[current] = static_cast<unsigned int>(neighbors.size());
        neighbors.insert(neighbors.end(), adjacency[current].begin(),
                         adjacency[current].end());
    }
    offsets.back() = static_cast<unsigned int>(neighbors.size());
}
} // unnamed namespace

CsfResult classifyCsfDevice(PointBatch& batch, const CsfProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA CSF requires a device batch");
    if (batch.size() >
        static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument(
            "CUDA CSF point count exceeds the exact device envelope");
    if (!csfProgramWithinExactDeviceEnvelope(program))
        throw std::invalid_argument(
            "CUDA CSF program exceeds the exact device envelope");

    NvtxRange range("pdg::filters.csf");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    std::unique_ptr<Allocation> frameAllocation =
        allocate<DeviceFrame>(batch, 1U);
    std::unique_ptr<Allocation> invalid = allocate<int>(batch, 1U);
    PDG_CUDA_CHECK(cudaMemsetAsync(invalid->data(), 0, sizeof(int), stream));
    frameKernel<<<1, 1, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), program.resolution, pointer<DeviceFrame>(frameAllocation),
        pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    int invalidHost = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidHost, invalid->data(),
                                   sizeof(invalidHost), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidHost)
        throw std::invalid_argument(
            "CUDA CSF cloth frame exceeds the exact device envelope");
    DeviceFrame frame;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&frame, frameAllocation->data(),
                                   sizeof(frame), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t size = frame.width * frame.height;

    std::vector<unsigned int> hostOffsets;
    std::vector<unsigned int> hostNeighbors;
    makeAdjacency(frame, hostOffsets, hostNeighbors);
    std::unique_ptr<Allocation> offsets =
        allocate<unsigned int>(batch, hostOffsets.size());
    std::unique_ptr<Allocation> neighbors =
        allocate<unsigned int>(batch, hostNeighbors.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(offsets->data(), hostOffsets.data(),
                                   hostOffsets.size() * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(neighbors->data(), hostNeighbors.data(),
                                   hostNeighbors.size() * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));

    std::unique_ptr<Allocation> position = allocate<double>(batch, size);
    std::unique_ptr<Allocation> oldPosition = allocate<double>(batch, size);
    std::unique_ptr<Allocation> nearestHeight = allocate<double>(batch, size);
    std::unique_ptr<Allocation> temporaryDistance =
        allocate<double>(batch, size);
    std::unique_ptr<Allocation> heights = allocate<double>(batch, size);
    std::unique_ptr<Allocation> movable = allocate<unsigned char>(batch, size);
    std::unique_ptr<Allocation> visited = allocate<unsigned char>(batch, size);
    std::unique_ptr<Allocation> queue =
        allocate<unsigned int>(batch, size * 2U);
    std::unique_ptr<Allocation> backlist =
        allocate<unsigned int>(batch, size * 2U);
    rasterKernel<<<1, 1, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.size(), frame, program.resolution, pointer<unsigned int>(offsets),
        pointer<unsigned int>(neighbors), pointer<double>(position),
        pointer<double>(oldPosition), pointer<double>(nearestHeight),
        pointer<double>(temporaryDistance), pointer<double>(heights),
        pointer<unsigned char>(movable), pointer<unsigned char>(visited),
        pointer<unsigned int>(queue), pointer<unsigned int>(backlist),
        pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidHost, invalid->data(),
                                   sizeof(invalidHost), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidHost)
        throw std::out_of_range(
            "CUDA CSF raster exceeds the exact device envelope");
    std::unique_ptr<Allocation> iterations = allocate<unsigned int>(batch, 1U);
    PDG_CUDA_CHECK(
        cudaMemsetAsync(iterations->data(), 0, sizeof(unsigned int), stream));
    simulationKernel<<<1, 1, 0, stream>>>(
        frame, program, pointer<unsigned int>(offsets),
        pointer<unsigned int>(neighbors), pointer<double>(heights),
        pointer<double>(position), pointer<double>(oldPosition),
        pointer<unsigned char>(movable), pointer<unsigned int>(iterations),
        pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());

    std::unique_ptr<Allocation> counts =
        allocate<unsigned long long>(batch, 2U);
    PDG_CUDA_CHECK(cudaMemsetAsync(counts->data(), 0,
                                   2U * sizeof(unsigned long long), stream));
    classificationKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(X), batch.data<double>(Y), batch.data<double>(Z),
        batch.data<std::uint8_t>(Classification), batch.size(), frame, program,
        pointer<double>(position), pointer<unsigned long long>(counts),
        pointer<int>(invalid));
    PDG_CUDA_CHECK(cudaGetLastError());
    unsigned int iterationsHost = 0U;
    unsigned long long countsHost[2]{};
    PDG_CUDA_CHECK(cudaMemcpyAsync(&iterationsHost, iterations->data(),
                                   sizeof(iterationsHost),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(countsHost, counts->data(),
                                   sizeof(countsHost), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(&invalidHost, invalid->data(),
                                   sizeof(invalidHost), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (invalidHost)
        throw std::out_of_range("CUDA CSF point lies outside its cloth frame");
    return {frame.width, frame.height, iterationsHost,
            static_cast<std::size_t>(countsHost[0]),
            static_cast<std::size_t>(countsHost[1])};
}

} // namespace pdg
