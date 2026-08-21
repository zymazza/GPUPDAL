#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

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
bool partitionDeviceAvailable()
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

TEST(CudaReturnPartition, MatchesHostForAllGroupsAndBoundaries)
{
    if (!partitionDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(48U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    const pdg::DimensionId returnNumber(pdg::StandardDimension::ReturnNumber);
    const pdg::DimensionId numberOfReturns(
        pdg::StandardDimension::NumberOfReturns);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(returnNumber, pdg::DimensionType::Unsigned8);
        batch->materialize(numberOfReturns, pdg::DimensionType::Unsigned8);
        batch->setSize(Count);
    }
    auto returnNumbers = host.hostSpan<std::uint8_t>(returnNumber);
    auto numbersOfReturns = host.hostSpan<std::uint8_t>(numberOfReturns);
    for (std::size_t point = 0; point < Count; ++point)
    {
        const std::uint8_t number =
            static_cast<std::uint8_t>((point * 17U) % 8U);
        numbersOfReturns[point] = number;
        returnNumbers[point] = static_cast<std::uint8_t>(
            (point * 29U + (point % 11U == 0U ? 7U : 0U)) % 9U);
    }

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(returnNumber),
                                   host.rawData(returnNumber), Count,
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(numberOfReturns),
                                   host.rawData(numberOfReturns), Count,
                                   cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> devicePermutation = deviceMemory->allocate(
        Count * sizeof(std::uint64_t), alignof(std::uint64_t));

    for (std::uint8_t groups :
         {pdg::AllReturnGroups, pdg::ReturnLast,
          static_cast<std::uint8_t>(pdg::ReturnFirst | pdg::ReturnOnly),
          std::uint8_t{0}})
    {
        pdg::ReturnsProgram program;
        program.groups = groups;
        std::vector<std::uint64_t> expected(Count);
        std::vector<std::uint64_t> actual(Count);
        const pdg::ReturnsPartitionResult hostResult =
            pdg::partitionReturns(host, program, expected.data());
        const pdg::ReturnsPartitionResult deviceResult = pdg::partitionReturns(
            device, program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        ASSERT_EQ(deviceResult.counts, hostResult.counts);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), devicePermutation->data(),
                                       Count * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t selected =
            static_cast<std::size_t>(hostResult.selectedCount());
        EXPECT_TRUE(std::equal(expected.begin(), expected.begin() + selected,
                               actual.begin()));
    }
}

TEST(CudaDividerPartition, MatchesHostForBothModes)
{
    if (!partitionDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(48U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    host.setSize(Count);
    device.setSize(Count);
    std::vector<std::uint64_t> expected(Count);
    std::vector<std::uint64_t> actual(Count);
    std::unique_ptr<pdg::Allocation> devicePermutation = deviceMemory->allocate(
        Count * sizeof(std::uint64_t), alignof(std::uint64_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());

    for (pdg::DividerMode mode :
         {pdg::DividerMode::Partition, pdg::DividerMode::RoundRobin})
    {
        pdg::DividerProgram program;
        program.mode = mode;
        program.count = 17;
        const pdg::DividerPartitionResult hostResult =
            pdg::partitionDivider(host, program, expected.data());
        const pdg::DividerPartitionResult deviceResult = pdg::partitionDivider(
            device, program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        ASSERT_EQ(deviceResult.counts, hostResult.counts);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), devicePermutation->data(),
                                       Count * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaSplitterCells, MatchesHostAtChunkAndBoundaryValues)
{
    if (!partitionDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(48U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(x, pdg::DimensionType::Double);
        batch->materialize(y, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    auto xValues = host.hostSpan<double>(x);
    auto yValues = host.hostSpan<double>(y);
    for (std::size_t point = 0; point < Count; ++point)
    {
        const std::int64_t centered =
            static_cast<std::int64_t>(point % 2001U) - 1000;
        xValues[point] = static_cast<double>(centered) * 10.0;
        yValues[point] = static_cast<double>(centered) * 10.0 + 0.125;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(x), host.rawData(x),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(y), host.rawData(y),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> deviceX = deviceMemory->allocate(
        Count * sizeof(std::int32_t), alignof(std::int32_t));
    std::unique_ptr<pdg::Allocation> deviceY = deviceMemory->allocate(
        Count * sizeof(std::int32_t), alignof(std::int32_t));
    std::vector<std::int32_t> expectedX(Count);
    std::vector<std::int32_t> expectedY(Count);
    std::vector<std::int32_t> actualX(Count);
    std::vector<std::int32_t> actualY(Count);
    pdg::SplitterProgram program;
    program.length = 10.0;
    program.originX = 0.0;
    program.originY = 0.0;
    pdg::computeSplitterCells(host, program, expectedX.data(),
                              expectedY.data());
    pdg::computeSplitterCells(device, program,
                              static_cast<std::int32_t*>(deviceX->data()),
                              static_cast<std::int32_t*>(deviceY->data()));
    PDG_CUDA_CHECK(cudaMemcpyAsync(actualX.data(), deviceX->data(),
                                   Count * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(actualY.data(), deviceY->data(),
                                   Count * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actualX, expectedX);
    EXPECT_EQ(actualY, expectedY);
}
} // unnamed namespace
