#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Morton.hpp>
#include <pdg/stages/Ordering.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool mortonDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

TEST(CudaMorton, KeyBitsAndStablePermutationMatchHost)
{
    if (!mortonDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    constexpr double MaximumX = 250000.75;
    constexpr double MaximumY = 175000.25;
    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("MortonKey", pdg::DimensionType::Unsigned64);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(128U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(x, pdg::DimensionType::Double);
        batch->materialize(y, pdg::DimensionType::Double);
        batch->materialize(key.id, pdg::DimensionType::Unsigned64);
        batch->setSize(Count);
    }
    auto xValues = host.hostSpan<double>(x);
    auto yValues = host.hostSpan<double>(y);
    for (std::size_t point = 0; point < Count; ++point)
    {
        xValues[point] = static_cast<double>((point * 7919U) % 1000003U) * 0.25;
        yValues[point] = static_cast<double>((point * 1543U) % 700001U) * 0.25;
    }
    xValues[0] = 0.0;
    yValues[0] = 0.0;
    xValues[1] = MaximumX;
    yValues[1] = MaximumY;

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(x), host.rawData(x),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(y), host.rawData(y),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

    for (bool reverse : {false, true})
    {
        pdg::MortonProgram morton;
        morton.bounds = {0.0, 0.0, MaximumX, MaximumY};
        morton.reverse = reverse;
        ASSERT_TRUE(pdg::mortonMaySupportExactDevice(host, morton));
        pdg::generateMortonKeys(
            host, morton, static_cast<std::uint64_t*>(host.rawData(key.id)));
        pdg::generateMortonKeys(
            device, morton,
            static_cast<std::uint64_t*>(device.rawData(key.id)));

        std::vector<std::uint64_t> deviceKeys(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            deviceKeys.data(), device.rawData(key.id),
            Count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(deviceKeys, (std::vector<std::uint64_t>(
                                  host.hostSpan<std::uint64_t>(key.id).begin(),
                                  host.hostSpan<std::uint64_t>(key.id).end())));

        pdg::OrderingProgram ordering;
        ordering.dimensions = {key.id};
        ordering.algorithm = pdg::OrderingAlgorithm::Stable;
        std::vector<std::uint64_t> expected(Count);
        std::vector<std::uint64_t> actual(Count);
        static_cast<void>(pdg::orderPoints(host, ordering, expected.data()));
        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory->allocate(Count * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
        ASSERT_TRUE(pdg::orderPoints(
                        device, ordering,
                        static_cast<std::uint64_t*>(devicePermutation->data()))
                        .exact);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), devicePermutation->data(),
                                       Count * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, expected);
    }
}
} // unnamed namespace
