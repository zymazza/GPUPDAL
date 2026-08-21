#include <pdg/Compaction.hpp>
#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{

bool compactionDeviceAvailable()
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

TEST(CudaCompaction, MatchesStableHostSelection)
{
    if (!compactionDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 65537;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId integer =
        dimensions.registerCustom("Integer", pdg::DimensionType::Unsigned64).id;
    const pdg::DimensionId value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double).id;
    const std::vector<pdg::DimensionId> columns = {integer, value};
    auto memory = pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch source(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *memory);
    pdg::PointBatch destination(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *memory);
    for (pdg::PointBatch* batch : {&source, &destination})
        for (pdg::DimensionId id : columns)
            batch->materialize(id);
    source.setSize(Count);

    std::vector<std::uint64_t> integers(Count);
    std::vector<double> values(Count);
    std::vector<std::uint8_t> keep(Count);
    std::vector<std::uint64_t> expectedIntegers;
    std::vector<double> expectedValues;
    for (std::size_t index = 0; index < Count; ++index)
    {
        integers[index] = index * 7919U;
        values[index] = static_cast<double>(index) * -0.03125;
        keep[index] = static_cast<std::uint8_t>(index % 13U < 5U);
        if (keep[index])
        {
            expectedIntegers.push_back(integers[index]);
            expectedValues.push_back(values[index]);
        }
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(memory->nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(source.rawData(integer), integers.data(),
                                   Count * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(source.rawData(value), values.data(),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> deviceKeep =
        memory->allocate(Count, alignof(std::uint8_t));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceKeep->data(), keep.data(), Count,
                                   cudaMemcpyHostToDevice, stream));

    ASSERT_EQ(pdg::compactPointBatch(
                  source, destination, columns,
                  static_cast<const std::uint8_t*>(deviceKeep->data())),
              expectedIntegers.size());
    std::vector<std::uint64_t> actualIntegers(expectedIntegers.size());
    std::vector<double> actualValues(expectedValues.size());
    PDG_CUDA_CHECK(
        cudaMemcpyAsync(actualIntegers.data(), destination.rawData(integer),
                        actualIntegers.size() * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        actualValues.data(), destination.rawData(value),
        actualValues.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actualIntegers, expectedIntegers);
    EXPECT_EQ(actualValues, expectedValues);
}

} // unnamed namespace
