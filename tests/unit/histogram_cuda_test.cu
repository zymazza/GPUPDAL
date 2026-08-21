#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace
{
bool deviceAvailable()
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

TEST(CudaHistogram, MatchesHostBinsCountsOrderAndFirstBits)
{
    if (!deviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t count = 131103;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double).id;
    const pdg::DimensionId flag =
        dimensions.registerCustom("PredicateFlag", pdg::DimensionType::Double)
            .id;
    const pdg::PredicateProgram predicate = pdg::compilePredicate(
        "PredicateFlag >= 2 && PredicateFlag <= 5 && Value >= -31", dimensions);
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(32U * 1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch host(count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(value);
        batch->materialize(flag);
        batch->setSize(count);
    }
    for (std::size_t point = 0; point < count; ++point)
    {
        host.data<double>(value)[point] =
            static_cast<double>(static_cast<std::int64_t>(point % 67U) - 33);
        host.data<double>(flag)[point] = static_cast<double>(point % 8U);
    }
    host.data<double>(value)[16] = +0.0;
    host.data<double>(value)[24] = -0.0;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {value, flag})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const std::vector<pdg::HistogramBin> expected =
        pdg::selectedHistogram(host, value, predicate, 700, 1);
    const std::vector<pdg::HistogramBin> actual =
        pdg::selectedHistogram(device, value, predicate, 700);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t bin = 0; bin < expected.size(); ++bin)
    {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[bin].value),
                  std::bit_cast<std::uint64_t>(expected[bin].value));
        EXPECT_EQ(actual[bin].count, expected[bin].count);
        EXPECT_EQ(actual[bin].firstIndex, expected[bin].firstIndex);
    }
}
} // unnamed namespace
