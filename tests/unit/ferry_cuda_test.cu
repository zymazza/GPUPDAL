#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ferry.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
pdg::CoordinateEncoding cudaFerryCoordinates()
{
    return {{0.001, 0.001, 0.001}, {500000.0, 4800000.0, 100.0}};
}

bool cudaDeviceAvailable()
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

struct Destination
{
    pdg::DimensionId id;
    pdg::DimensionType type;
};
} // unnamed namespace

TEST(CudaFerry, AllDestinationTypesMatchExactHostPath)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 4097;
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const std::array<pdg::DimensionType, 10> types = {
        pdg::DimensionType::Signed8,    pdg::DimensionType::Signed16,
        pdg::DimensionType::Signed32,   pdg::DimensionType::Signed64,
        pdg::DimensionType::Unsigned8,  pdg::DimensionType::Unsigned16,
        pdg::DimensionType::Unsigned32, pdg::DimensionType::Unsigned64,
        pdg::DimensionType::Float,      pdg::DimensionType::Double};
    std::vector<Destination> destinations;
    for (std::size_t index = 0; index < types.size(); ++index)
        destinations.push_back(
            {dimensions
                 .registerCustom("Destination" + std::to_string(index),
                                 types[index])
                 .id,
             types[index]});
    const auto empty =
        dimensions.registerCustom("Empty", pdg::DimensionType::Double).id;

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaFerryCoordinates(), dimensions, hostMemory);
    pdg::PointBatch device(Count, cudaFerryCoordinates(), dimensions,
                           *deviceMemory);
    host.materialize(source);
    device.materialize(source);
    host.setSize(Count);
    device.setSize(Count);

    constexpr std::array<double, 23> BoundaryValues = {-9007199254740991.0,
                                                       -2147483648.5,
                                                       -2147483648.49,
                                                       -255.5,
                                                       -128.5,
                                                       -127.5,
                                                       -2.5,
                                                       -1.5,
                                                       -0.5,
                                                       0.0,
                                                       0.49,
                                                       0.5,
                                                       1.5,
                                                       127.49,
                                                       127.5,
                                                       255.49,
                                                       255.5,
                                                       65535.49,
                                                       65535.5,
                                                       2147483647.49,
                                                       2147483647.5,
                                                       4294967295.49,
                                                       9007199254740991.0};
    for (std::size_t index = 0; index < Count; ++index)
        host.data<double>(source)[index] =
            BoundaryValues[index % BoundaryValues.size()];

    const auto stream = static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(source), host.rawData(source),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

    pdg::FerryProgram program;
    for (const Destination& destination : destinations)
        program.copies.push_back({true, source, destination.id, true});
    program.copies.push_back({false, {}, empty, true});
    pdg::executeFerry(host, program, 1);
    pdg::executeFerry(device, program);

    for (const Destination& destination : destinations)
    {
        const std::size_t bytes =
            Count * pdg::dimensionTypeSize(destination.type);
        std::vector<std::byte> actual(bytes);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(),
                                       device.rawData(destination.id), bytes,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(
            std::memcmp(actual.data(), host.rawData(destination.id), bytes), 0)
            << "destination type " << static_cast<unsigned>(destination.type);
    }

    std::vector<double> emptyActual(Count, -1.0);
    PDG_CUDA_CHECK(cudaMemcpyAsync(emptyActual.data(), device.rawData(empty),
                                   Count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_TRUE(std::all_of(emptyActual.begin(), emptyActual.end(),
                            [](double value) { return value == 0.0; }));
}

TEST(CudaFerry, OrderedProgramMatchesHostAcrossKernelLaunchChunks)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    constexpr std::size_t OperationCount = 18;
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    std::array<pdg::DimensionId, OperationCount> chain;
    for (std::size_t index = 0; index < chain.size(); ++index)
        chain[index] = dimensions
                           .registerCustom("Chain" + std::to_string(index),
                                           pdg::DimensionType::Signed16)
                           .id;

    pdg::HostMemoryResource hostMemory;
    auto deviceMemory = pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, cudaFerryCoordinates(), dimensions, hostMemory);
    pdg::PointBatch device(Count, cudaFerryCoordinates(), dimensions,
                           *deviceMemory);
    host.materialize(source);
    device.materialize(source);
    host.setSize(Count);
    device.setSize(Count);
    for (std::size_t index = 0; index < Count; ++index)
        host.data<double>(source)[index] =
            static_cast<double>((index * 7919U) % 80000U) - 40000.5;

    const auto stream = static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(source), host.rawData(source),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    pdg::FerryProgram program;
    program.copies.push_back({true, source, chain.front(), true});
    for (std::size_t index = 1; index < chain.size(); ++index)
        program.copies.push_back({true, chain[index - 1], chain[index], true});

    pdg::executeFerry(host, program, 1);
    pdg::executeFerry(device, program);
    std::vector<std::int16_t> actual(Count);
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), device.rawData(chain.back()),
                                   Count * sizeof(std::int16_t),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(std::memcmp(actual.data(), host.rawData(chain.back()),
                          Count * sizeof(std::int16_t)),
              0);
}

TEST(CudaFerry, RejectsCoordinateMappingsOutsideCurrentDeviceEnvelope)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    auto memory = pdg::makeCudaMemoryResource();
    pdg::PointBatch batch(0, cudaFerryCoordinates(), dimensions, *memory);
    const auto x = pdg::DimensionId(pdg::StandardDimension::X);
    const auto intensity = pdg::DimensionId(pdg::StandardDimension::Intensity);
    EXPECT_THROW(pdg::executeFerry(batch, {{{true, x, intensity, false}}}),
                 std::invalid_argument);
}
