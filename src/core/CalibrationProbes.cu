#include <pdg/CalibrationProbes.hpp>

#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/SyntheticCloud.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pdg
{
namespace
{
using Clock = std::chrono::steady_clock;

[[nodiscard]] double nanosecondsSince(Clock::time_point start) noexcept
{
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             start)
            .count());
}

[[nodiscard]] double median(std::vector<double> values)
{
    if (values.empty())
        throw std::invalid_argument("median of no samples");
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2U;
    return values.size() % 2U == 1U ? values[mid]
                                    : 0.5 * (values[mid - 1U] + values[mid]);
}

__global__ void emptyKernel() {}
} // unnamed namespace

double probeCudaStartupNanoseconds()
{
    const Clock::time_point start = Clock::now();
    PDG_CUDA_CHECK(cudaFree(nullptr));
    void* scratch = nullptr;
    PDG_CUDA_CHECK(cudaMalloc(&scratch, 256U));
    emptyKernel<<<1, 32>>>();
    PDG_CUDA_CHECK(cudaGetLastError());
    PDG_CUDA_CHECK(cudaDeviceSynchronize());
    PDG_CUDA_CHECK(cudaFree(scratch));
    return nanosecondsSince(start);
}

CalibrationTransferProbe probeCudaTransfers(std::size_t bytes, int repeats)
{
    if (bytes == 0U || repeats <= 0)
        throw std::invalid_argument("transfer probe needs bytes and repeats");
    PDG_CUDA_CHECK(cudaFree(nullptr));
    std::vector<unsigned char> host(bytes, 0x5AU);
    void* device = nullptr;
    PDG_CUDA_CHECK(cudaMalloc(&device, bytes));
    // One untimed round trip so page mapping and allocation are settled.
    PDG_CUDA_CHECK(cudaMemcpy(device, host.data(), bytes,
                              cudaMemcpyHostToDevice));
    PDG_CUDA_CHECK(cudaMemcpy(host.data(), device, bytes,
                              cudaMemcpyDeviceToHost));
    std::vector<double> up;
    std::vector<double> down;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        Clock::time_point start = Clock::now();
        PDG_CUDA_CHECK(cudaMemcpy(device, host.data(), bytes,
                                  cudaMemcpyHostToDevice));
        PDG_CUDA_CHECK(cudaDeviceSynchronize());
        up.push_back(nanosecondsSince(start) / static_cast<double>(bytes));
        start = Clock::now();
        PDG_CUDA_CHECK(cudaMemcpy(host.data(), device, bytes,
                                  cudaMemcpyDeviceToHost));
        PDG_CUDA_CHECK(cudaDeviceSynchronize());
        down.push_back(nanosecondsSince(start) / static_cast<double>(bytes));
    }
    PDG_CUDA_CHECK(cudaFree(device));
    return {.hostToDeviceNanosecondsPerByte = median(std::move(up)),
            .deviceToHostNanosecondsPerByte = median(std::move(down))};
}

double probeCudaSynchronizationNanoseconds(int repeats)
{
    if (repeats <= 0)
        throw std::invalid_argument("synchronization probe needs repeats");
    PDG_CUDA_CHECK(cudaFree(nullptr));
    emptyKernel<<<1, 32>>>();
    PDG_CUDA_CHECK(cudaGetLastError());
    PDG_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> samples;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        const Clock::time_point start = Clock::now();
        emptyKernel<<<1, 32>>>();
        PDG_CUDA_CHECK(cudaGetLastError());
        PDG_CUDA_CHECK(cudaDeviceSynchronize());
        samples.push_back(nanosecondsSince(start));
    }
    return median(std::move(samples));
}

double probeIndexBuildNanosecondsPerByte(std::size_t points, int repeats)
{
    if (points == 0U || repeats <= 0)
        throw std::invalid_argument("index probe needs points and repeats");
    constexpr DimensionId X(StandardDimension::X);
    constexpr DimensionId Y(StandardDimension::Y);
    constexpr DimensionId Z(StandardDimension::Z);
    DimensionRegistry dimensions;
    const CoordinateEncoding coordinates{{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}};
    HostMemoryResource hostMemory;
    std::unique_ptr<MemoryResource> deviceMemory =
        makeCudaMemoryResource(256U * 1024U * 1024U);
    PointBatch host(points, coordinates, dimensions, hostMemory);
    PointBatch device(points, coordinates, dimensions, *deviceMemory);
    for (PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, DimensionType::Double);
        batch->materialize(Y, DimensionType::Double);
        batch->materialize(Z, DimensionType::Double);
        batch->setSize(points);
    }
    const SyntheticCloudGenerator generator({.points = points});
    for (std::size_t index = 0; index < points; ++index)
    {
        const SyntheticPoint point = generator.point(index);
        // Quantise like the LAS fixture so ties match what the stages see.
        const auto quantize = [](double value)
        { return static_cast<double>(std::llround(value / 0.01)) * 0.01; };
        host.hostSpan<double>(X)[index] = quantize(point.x);
        host.hostSpan<double>(Y)[index] = quantize(point.y);
        host.hostSpan<double>(Z)[index] = quantize(point.z);
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            points * sizeof(double), cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    const UniformGridConfig config = makeAdaptiveKnnConfig(host, 3, 8);
    std::vector<double> samples;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        SpatialIndex index(device, config);
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        const Clock::time_point start = Clock::now();
        index.build();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        const double nanoseconds = nanosecondsSince(start);
        const std::size_t bytes = index.allocatedBytes();
        if (bytes == 0U)
            throw std::runtime_error("index probe reported no persistent bytes");
        samples.push_back(nanoseconds / static_cast<double>(bytes));
    }
    return median(std::move(samples));
}

} // namespace pdg
