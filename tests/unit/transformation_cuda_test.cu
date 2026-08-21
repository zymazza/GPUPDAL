#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Transformation.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool transformationDeviceAvailable()
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

TEST(CudaTransformation, AffineKernelMatchesHostBitForBit)
{
    if (!transformationDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::X, pdg::StandardDimension::Y,
          pdg::StandardDimension::Z})
    {
        host.materialize(pdg::DimensionId(dimension),
                         pdg::DimensionType::Double);
        device.materialize(pdg::DimensionId(dimension),
                           pdg::DimensionType::Double);
    }
    host.setSize(Count);
    device.setSize(Count);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    for (std::size_t point = 0; point < Count; ++point)
    {
        host.hostSpan<double>(x)[point] =
            static_cast<double>(point) * 0.03125 - 1000.125;
        host.hostSpan<double>(y)[point] =
            static_cast<double>(point % 997U) * -0.0625 + 3.75;
        host.hostSpan<double>(z)[point] =
            static_cast<double>(point % 37U) / 11.0;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (const pdg::DimensionId id : {x, y, z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), host.rawData(id),
                                       Count * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));

    const pdg::TransformationProgram program = pdg::compileTransformation(
        "1.25 -0.5 0.125 123.75 0.25 2.5 -0.75 -9.125 "
        "-1.5 0.0625 3.25 0.03125 0 0 0 1");
    pdg::executeTransformation(host, program);
    pdg::executeTransformation(device, program);

    std::vector<double> actual(Count);
    for (const pdg::DimensionId id : {x, y, z})
    {
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), device.rawData(id),
                                       Count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto expected = host.hostSpan<double>(id);
        for (std::size_t point = 0; point < Count; ++point)
            EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[point]),
                      std::bit_cast<std::uint64_t>(expected[point]))
                << "dimension " << id.value() << ", point " << point;
    }
}
} // unnamed namespace
