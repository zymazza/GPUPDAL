#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Locate.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace
{
bool locateDeviceAvailable()
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

template <typename T>
void expectDeviceMatchesHost(const std::vector<T>& values,
                             pdg::DimensionType type, pdg::LocateKind kind)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionDefinition& scratch =
        dimensions.registerCustom("Scratch", type);
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(4U * 1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch host(values.size(), coordinates, dimensions, hostMemory);
    pdg::PointBatch device(values.size(), coordinates, dimensions,
                           *deviceMemory);
    host.materialize(scratch.id, type);
    device.materialize(scratch.id, type);
    host.setSize(values.size());
    device.setSize(values.size());
    auto target = host.hostSpan<T>(scratch.id);
    std::copy(values.begin(), values.end(), target.begin());
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        device.rawData(scratch.id), host.rawData(scratch.id),
        values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));

    const pdg::LocateProgram program{scratch.id, kind};
    const pdg::LocateResult expected = pdg::locateExtreme(host, program, 29);
    const pdg::LocateResult actual = pdg::locateExtreme(device, program, 29);
    EXPECT_EQ(actual.hasPoints, expected.hasPoints);
    EXPECT_EQ(actual.comparable, expected.comparable);
    EXPECT_EQ(actual.index, expected.index);
    EXPECT_DOUBLE_EQ(actual.value, expected.value);
}

TEST(CudaLocate, MatchesHostForTypesTiesSentinelsAndLargeReductions)
{
    if (!locateDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    std::vector<std::uint16_t> unsignedValues(131103, 7);
    unsignedValues[17] = 65535;
    unsignedValues[130999] = 65535;
    expectDeviceMatchesHost(unsignedValues, pdg::DimensionType::Unsigned16,
                            pdg::LocateKind::Maximum);

    const std::vector<std::int64_t> signedValues = {
        (std::numeric_limits<std::int64_t>::max)(),
        (std::numeric_limits<std::int64_t>::max)() - 1, -5, -5};
    expectDeviceMatchesHost(signedValues, pdg::DimensionType::Signed64,
                            pdg::LocateKind::Minimum);

    const std::vector<double> nonComparable = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::lowest(),
        -std::numeric_limits<double>::infinity()};
    expectDeviceMatchesHost(nonComparable, pdg::DimensionType::Double,
                            pdg::LocateKind::Maximum);
}

TEST(CudaLocate, MatchesExactCoordinateDecode)
{
    if (!locateDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(1024U * 1024U);
    const pdg::CoordinateEncoding coordinates({0.01, 0.02, 0.03},
                                              {1000.0, 2000.0, 3000.0});
    pdg::PointBatch host(4, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(4, coordinates, dimensions, *deviceMemory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    host.materialize(x, pdg::DimensionType::Signed32);
    device.materialize(x, pdg::DimensionType::Signed32);
    host.setSize(4);
    device.setSize(4);
    auto raw = host.hostSpan<std::int32_t>(x);
    raw[0] = -10;
    raw[1] = 500;
    raw[2] = 499;
    raw[3] = 500;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(x), host.rawData(x),
                                   4U * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
    const pdg::LocateProgram program{x, pdg::LocateKind::Maximum};
    const pdg::LocateResult expected = pdg::locateExtreme(host, program, 100);
    const pdg::LocateResult actual = pdg::locateExtreme(device, program, 100);
    EXPECT_EQ(actual.index, expected.index);
    EXPECT_DOUBLE_EQ(actual.value, expected.value);
}
} // unnamed namespace
