#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool colorMapDeviceAvailable()
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

TEST(CudaColorMap, MatchesHostAtChunkAndRampBoundaries)
{
    if (!colorMapDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    constexpr std::array<std::uint8_t, 7> RampRed = {3, 5, 8, 13, 21, 34, 55};
    constexpr std::array<std::uint8_t, 7> RampGreen = {1, 4, 9, 16, 25, 36, 49};
    constexpr std::array<std::uint8_t, 7> RampBlue = {2, 6, 10, 14, 18, 22, 26};
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId value =
        dimensions.registerCustom("PdgColorValue", pdg::DimensionType::Double)
            .id;
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(32U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(value, pdg::DimensionType::Double);
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
            batch->materialize(pdg::DimensionId(dimension),
                               pdg::DimensionType::Unsigned16);
        batch->setSize(Count);
    }
    auto values = host.hostSpan<double>(value);
    for (std::size_t point = 0; point < Count; ++point)
    {
        const std::int64_t centered =
            static_cast<std::int64_t>(point % 1001U) - 500;
        values[point] = static_cast<double>(centered) / 7.0;
    }
    values[0] = -25.0;
    values[1] = 75.0;
    values[2] = -25.000000000000004;
    values[3] = 75.00000000000001;
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
          pdg::StandardDimension::Blue})
    {
        auto colors = host.hostSpan<std::uint16_t>(pdg::DimensionId(dimension));
        for (std::size_t point = 0; point < Count; ++point)
            colors[point] = static_cast<std::uint16_t>((point * 37U) % 65536U);
    }

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (const pdg::DimensionId id :
         {value, pdg::DimensionId(pdg::StandardDimension::Red),
          pdg::DimensionId(pdg::StandardDimension::Green),
          pdg::DimensionId(pdg::StandardDimension::Blue)})
    {
        const std::size_t bytes =
            Count * pdg::dimensionTypeSize(host.columnInfo(id).physicalType);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id), host.rawData(id),
                                       bytes, cudaMemcpyHostToDevice, stream));
    }
    std::unique_ptr<pdg::Allocation> deviceRed =
        deviceMemory->allocate(RampRed.size(), alignof(std::uint8_t));
    std::unique_ptr<pdg::Allocation> deviceGreen =
        deviceMemory->allocate(RampGreen.size(), alignof(std::uint8_t));
    std::unique_ptr<pdg::Allocation> deviceBlue =
        deviceMemory->allocate(RampBlue.size(), alignof(std::uint8_t));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceRed->data(), RampRed.data(),
                                   RampRed.size(), cudaMemcpyHostToDevice,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceGreen->data(), RampGreen.data(),
                                   RampGreen.size(), cudaMemcpyHostToDevice,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceBlue->data(), RampBlue.data(),
                                   RampBlue.size(), cudaMemcpyHostToDevice,
                                   stream));

    pdg::ColorMapProgram map;
    map.value = value;
    map.minimum = -25.0;
    map.maximum = 75.0;
    map.invert = true;
    pdg::applyColorMap(
        host, map,
        {RampRed.data(), RampGreen.data(), RampBlue.data(), RampRed.size()});
    pdg::applyColorMap(device, map,
                       {static_cast<const std::uint8_t*>(deviceRed->data()),
                        static_cast<const std::uint8_t*>(deviceGreen->data()),
                        static_cast<const std::uint8_t*>(deviceBlue->data()),
                        RampRed.size()});

    std::vector<std::uint16_t> actual(Count);
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
          pdg::StandardDimension::Blue})
    {
        const pdg::DimensionId id(dimension);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), device.rawData(id),
                                       Count * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_TRUE(std::equal(actual.begin(), actual.end(),
                               host.data<std::uint16_t>(id)));
    }
}
} // unnamed namespace
