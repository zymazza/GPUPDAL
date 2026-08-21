#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Summary.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace
{
bool summaryDeviceAvailable()
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

TEST(CudaSummary, MatchesOrderedHostRecurrenceAcrossDimensionsAndTiles)
{
    if (!summaryDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const std::array<pdg::DimensionId, 3> ids = {
        dimensions.registerCustom("First", pdg::DimensionType::Double).id,
        dimensions.registerCustom("Second", pdg::DimensionType::Double).id,
        dimensions.registerCustom("Third", pdg::DimensionType::Double).id};
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::DimensionId id : ids)
    {
        host.materialize(id, pdg::DimensionType::Double);
        device.materialize(id, pdg::DimensionType::Double);
    }
    host.setSize(Count);
    device.setSize(Count);
    for (std::size_t point = 0; point < Count; ++point)
    {
        host.data<double>(ids[0])[point] =
            static_cast<double>(static_cast<std::int64_t>(point % 8191U) -
                                4095) /
            7.0;
        host.data<double>(ids[1])[point] =
            static_cast<double>((point * 65537U) % 10000019U) / 19.0;
        host.data<double>(ids[2])[point] =
            point % 2U ? 9007199254740992.0 : -9007199254740992.0;
    }

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId id : ids)
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), host.rawData(id),
                                       Count * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

    std::array<pdg::SummaryState, 3> expected{};
    for (std::size_t dimension = 0; dimension < ids.size(); ++dimension)
        for (std::size_t prefix = 0; prefix < 17U; ++prefix)
            pdg::insertSummary(expected[dimension],
                               static_cast<double>(prefix) -
                                   static_cast<double>(dimension),
                               false);
    std::array<pdg::SummaryState, 3> actual = expected;
    std::unique_ptr<pdg::Allocation> deviceStates =
        deviceMemory->allocate(sizeof(actual), alignof(pdg::SummaryState));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceStates->data(), actual.data(),
                                   sizeof(actual), cudaMemcpyHostToDevice,
                                   stream));

    pdg::updateSummaries(host, ids, expected.data(), false);
    pdg::updateSummaries(device, ids,
                         static_cast<pdg::SummaryState*>(deviceStates->data()),
                         false);
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceStates->data(),
                                   sizeof(actual), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t dimension = 0; dimension < ids.size(); ++dimension)
    {
        EXPECT_EQ(actual[dimension].count, expected[dimension].count);
        for (const auto member :
             {&pdg::SummaryState::minimum, &pdg::SummaryState::maximum,
              &pdg::SummaryState::m1, &pdg::SummaryState::m2,
              &pdg::SummaryState::m3, &pdg::SummaryState::m4})
            EXPECT_EQ(
                std::bit_cast<std::uint64_t>(actual[dimension].*member),
                std::bit_cast<std::uint64_t>(expected[dimension].*member));
    }
}
} // unnamed namespace
