#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
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

TEST(CudaInformation, MatchesHostAcrossBlocksAndCoordinateSpecialCases)
{
    if (!deviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t count = 131103;
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({0.01, 0.02, 0.03},
                                              {0.0, 0.0, 0.0});
    pdg::PointBatch host(count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(count, coordinates, dimensions, *deviceMemory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    for (pdg::DimensionId dimension : {x, y, z})
    {
        host.materialize(dimension, pdg::DimensionType::Double);
        device.materialize(dimension, pdg::DimensionType::Double);
    }
    host.setSize(count);
    device.setSize(count);
    auto xs = host.hostSpan<double>(x);
    auto ys = host.hostSpan<double>(y);
    auto zs = host.hostSpan<double>(z);
    for (std::size_t point = 0; point < count; ++point)
    {
        xs[point] = static_cast<double>(point % 997U) - 500.0;
        ys[point] = static_cast<double>(point % 701U) - 350.0;
        zs[point] = static_cast<double>(point % 503U) - 250.0;
    }
    xs[0] = -0.0;
    xs[997] = +0.0;
    ys[17] = std::numeric_limits<double>::quiet_NaN();
    zs[130999] = -std::numeric_limits<double>::infinity();
    zs[131000] = std::numeric_limits<double>::infinity();

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {x, y, z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const pdg::BoundsResult expected = pdg::summarizeBounds(host, 29);
    const pdg::BoundsResult actual = pdg::summarizeBounds(device, 29);
    EXPECT_EQ(actual.count, expected.count);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.minimum[axis].value),
                  std::bit_cast<std::uint64_t>(expected.minimum[axis].value));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.maximum[axis].value),
                  std::bit_cast<std::uint64_t>(expected.maximum[axis].value));
        EXPECT_EQ(actual.minimum[axis].index, expected.minimum[axis].index);
        EXPECT_EQ(actual.maximum[axis].index, expected.maximum[axis].index);
        EXPECT_EQ(actual.minimum[axis].comparable,
                  expected.minimum[axis].comparable);
        EXPECT_EQ(actual.maximum[axis].comparable,
                  expected.maximum[axis].comparable);
    }
}
} // unnamed namespace
