#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordinal.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool ordinalDeviceAvailable()
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

void expectDeviceMatchesHost(const pdg::OrdinalProgram& program,
                             pdg::OrdinalMode mode, std::uint64_t total,
                             const std::vector<std::size_t>& chunks)
{
    const std::size_t capacity =
        *std::max_element(chunks.begin(), chunks.end());
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(4U * 1024U * 1024U);
    pdg::PointBatch host(
        capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, hostMemory);
    pdg::PointBatch device(
        capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    std::unique_ptr<pdg::Allocation> deviceKeep =
        deviceMemory->allocate(capacity, alignof(std::uint8_t));
    std::vector<std::uint8_t> expected(capacity);
    std::vector<std::uint8_t> actual(capacity);
    pdg::OrdinalState hostState = pdg::makeOrdinalState(program, mode, total);
    pdg::OrdinalState deviceState = pdg::makeOrdinalState(program, mode, total);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (std::size_t size : chunks)
    {
        host.setSize(size);
        device.setSize(size);
        pdg::evaluateOrdinal(host, program, hostState, expected.data());
        pdg::evaluateOrdinal(device, program, deviceState,
                             static_cast<std::uint8_t*>(deviceKeep->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceKeep->data(), size,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(
            std::vector<std::uint8_t>(actual.begin(), actual.begin() + size),
            std::vector<std::uint8_t>(expected.begin(),
                                      expected.begin() + size));
    }
    EXPECT_EQ(deviceState.inputProcessed, hostState.inputProcessed);
    EXPECT_EQ(deviceState.sequence, hostState.sequence);
}

TEST(CudaOrdinal, MatchesHostAcrossModesKindsAndChunkBoundaries)
{
    if (!ordinalDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::OrdinalProgram decimation;
    decimation.kind = pdg::OrdinalKind::Decimation;
    decimation.step = 2.6;
    decimation.offset = 10;
    decimation.limit = 1000;
    expectDeviceMatchesHost(decimation, pdg::OrdinalMode::Streaming, 1065,
                            {1, 7, 131, 511, 415});
    decimation.offset = 0;
    expectDeviceMatchesHost(decimation, pdg::OrdinalMode::Standard, 1065,
                            {127, 128, 255, 555});

    pdg::OrdinalProgram head;
    head.kind = pdg::OrdinalKind::Head;
    head.count = 300;
    head.invert = true;
    expectDeviceMatchesHost(head, pdg::OrdinalMode::Streaming, 1065,
                            {64, 129, 512, 360});

    pdg::OrdinalProgram tail;
    tail.kind = pdg::OrdinalKind::Tail;
    tail.count = 300;
    expectDeviceMatchesHost(tail, pdg::OrdinalMode::Standard, 1065,
                            {64, 129, 512, 360});
}
} // unnamed namespace
