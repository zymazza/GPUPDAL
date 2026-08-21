#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Robust.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool robustDeviceAvailable()
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

void expectDeviceMatchesHost(pdg::RobustKind kind)
{
    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionDefinition& value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double);
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(64U * 1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    host.materialize(value.id, pdg::DimensionType::Double);
    device.materialize(value.id, pdg::DimensionType::Double);
    host.setSize(Count);
    device.setSize(Count);
    auto values = host.hostSpan<double>(value.id);
    for (std::size_t point = 0; point < Count; ++point)
        values[point] =
            static_cast<double>((point * 7919U) % 10007U) * 0.03125 + 0.125;
    values[17] = 1000000.25;
    values[Count - 3U] = 1000000.25;

    pdg::RobustProgram program;
    program.dimension = value.id;
    program.kind = kind;
    program.multiplier = kind == pdg::RobustKind::Iqr ? 1.75 : 2.5;
    program.madMultiplier = 1.4862;
    ASSERT_TRUE(pdg::robustSupportsExactDevice(host, program));
    std::vector<std::uint8_t> expected(Count);
    const pdg::RobustResult expectedResult =
        pdg::evaluateRobust(host, program, expected.data());

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        device.rawData(value.id), host.rawData(value.id),
        Count * sizeof(double), cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    const pdg::RobustResult actualResult = pdg::evaluateRobust(
        device, program, static_cast<std::uint8_t*>(deviceKeep->data()));
    std::vector<std::uint8_t> actual(Count);
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), Count,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(std::bit_cast<std::uint64_t>(actualResult.first),
              std::bit_cast<std::uint64_t>(expectedResult.first));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actualResult.second),
              std::bit_cast<std::uint64_t>(expectedResult.second));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actualResult.scale),
              std::bit_cast<std::uint64_t>(expectedResult.scale));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actualResult.lowFence),
              std::bit_cast<std::uint64_t>(expectedResult.lowFence));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actualResult.highFence),
              std::bit_cast<std::uint64_t>(expectedResult.highFence));
    EXPECT_EQ(actual, expected);
}

TEST(CudaRobust, IqrAndMadMatchHostBitsAcrossLargeSorts)
{
    if (!robustDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";
    expectDeviceMatchesHost(pdg::RobustKind::Iqr);
    expectDeviceMatchesHost(pdg::RobustKind::Mad);
}
} // unnamed namespace
