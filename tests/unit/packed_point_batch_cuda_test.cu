#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PackedPointBatch.hpp>
#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

bool packedBatchDeviceAvailable()
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

class ScopedLegacyAllocator final
{
public:
    ScopedLegacyAllocator()
    {
        const char* previous = std::getenv("PDG_FORCE_LEGACY_CUDA_ALLOCATOR");
        if (previous)
        {
            m_hadPrevious = true;
            m_previous = previous;
        }
        if (setenv("PDG_FORCE_LEGACY_CUDA_ALLOCATOR", "1", 1) != 0)
            throw std::runtime_error(
                "could not force the legacy CUDA allocator for this test");
    }

    ~ScopedLegacyAllocator()
    {
        if (m_hadPrevious)
            setenv("PDG_FORCE_LEGACY_CUDA_ALLOCATOR", m_previous.c_str(), 1);
        else
            unsetenv("PDG_FORCE_LEGACY_CUDA_ALLOCATOR");
    }

private:
    bool m_hadPrevious = false;
    std::string m_previous;
};

TEST(CudaMemoryResource, ForcesClassicAllocator)
{
    if (!packedBatchDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    ScopedLegacyAllocator legacy;
    auto memory = pdg::makeCudaMemoryResource();
    auto allocation = memory->allocate(4096, alignof(std::max_align_t));
    ASSERT_NE(allocation->data(), nullptr);
    EXPECT_EQ(allocation->kind(), pdg::MemoryKind::Device);

    std::array<std::uint32_t, 4> expected{17U, 29U, 43U, 71U};
    std::array<std::uint32_t, 4> actual{};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(memory->nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(allocation->data(), expected.data(),
                                   sizeof(expected), cudaMemcpyHostToDevice,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), allocation->data(),
                                   sizeof(actual), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(actual, expected);
}

TEST(CudaPackedPointBatch, PreservesUnalignedPhysicalFieldsAndMapping)
{
    if (!packedBatchDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t PointCount = 4097;
    constexpr std::size_t ActiveCount = 2049;
    constexpr std::size_t PointStride = 31;
    pdg::DimensionRegistry dimensions;
    const std::array<pdg::DimensionId, 6> ids = {
        dimensions.registerCustom("Byte", pdg::DimensionType::Unsigned8).id,
        dimensions.registerCustom("Short", pdg::DimensionType::Signed16).id,
        dimensions.registerCustom("Word", pdg::DimensionType::Unsigned32).id,
        dimensions.registerCustom("Double", pdg::DimensionType::Double).id,
        dimensions.registerCustom("Float", pdg::DimensionType::Float).id,
        dimensions.registerCustom("Long", pdg::DimensionType::Unsigned64).id};
    const std::array<pdg::DimensionType, 6> types = {
        pdg::DimensionType::Unsigned8,  pdg::DimensionType::Signed16,
        pdg::DimensionType::Unsigned32, pdg::DimensionType::Double,
        pdg::DimensionType::Float,      pdg::DimensionType::Unsigned64};
    const std::array<std::size_t, 6> offsets = {0, 1, 3, 7, 15, 19};

    std::vector<pdg::PackedPointColumn> columns;
    for (std::size_t index = 0; index < ids.size(); ++index)
        columns.push_back(
            {ids[index], types[index], offsets[index], index % 2U == 0});

    auto memory = pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch batch(
        PointCount, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *memory);
    for (std::size_t index = 0; index < ids.size(); ++index)
        batch.materialize(ids[index], types[index]);

    std::vector<std::uint8_t> packed(PointCount * PointStride);
    for (std::size_t index = 0; index < packed.size(); ++index)
        packed[index] = static_cast<std::uint8_t>(index * 37U + 19U);
    std::vector<std::uint64_t> mapping(ActiveCount);
    for (std::size_t index = 0; index < ActiveCount; ++index)
        mapping[index] = static_cast<std::uint64_t>(index * 2U);

    std::unique_ptr<pdg::Allocation> devicePacked =
        memory->allocate(packed.size(), alignof(std::max_align_t));
    std::unique_ptr<pdg::Allocation> deviceOutput =
        memory->allocate(packed.size(), alignof(std::max_align_t));
    std::unique_ptr<pdg::Allocation> deviceMapping = memory->allocate(
        mapping.size() * sizeof(std::uint64_t), alignof(std::uint64_t));
    std::unique_ptr<pdg::Allocation> deviceOriginal = memory->allocate(
        mapping.size() * sizeof(std::uint64_t), alignof(std::uint64_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(memory->nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(devicePacked->data(), packed.data(),
                                   packed.size(), cudaMemcpyHostToDevice,
                                   stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceMapping->data(), mapping.data(),
                                   mapping.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(
        cudaMemsetAsync(deviceOutput->data(), 0xa5, packed.size(), stream));

    pdg::unpackPackedPointBatchDevice(
        devicePacked->data(), PointStride,
        static_cast<const std::uint64_t*>(deviceMapping->data()), ActiveCount,
        columns, batch, static_cast<std::uint64_t*>(deviceOriginal->data()));
    pdg::repackPackedPointBatchDevice(
        batch, deviceOutput->data(), PointStride,
        static_cast<const std::uint64_t*>(deviceOriginal->data()), columns);

    std::vector<std::vector<std::uint8_t>> unpacked(ids.size());
    for (std::size_t column = 0; column < ids.size(); ++column)
    {
        const std::size_t size = pdg::dimensionTypeSize(types[column]);
        unpacked[column].resize(ActiveCount * size);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            unpacked[column].data(), batch.rawData(ids[column]),
            unpacked[column].size(), cudaMemcpyDeviceToHost, stream));
    }
    std::vector<std::uint8_t> output(packed.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(output.data(), deviceOutput->data(),
                                   output.size(), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (std::size_t point = 0; point < ActiveCount; ++point)
        for (std::size_t column = 0; column < ids.size(); ++column)
        {
            const std::size_t size = pdg::dimensionTypeSize(types[column]);
            EXPECT_EQ(std::memcmp(unpacked[column].data() + point * size,
                                  packed.data() + mapping[point] * PointStride +
                                      offsets[column],
                                  size),
                      0);
        }

    std::vector<std::uint8_t> expected(output.size(), 0xa5U);
    for (std::size_t point = 0; point < ActiveCount; ++point)
        for (std::size_t column = 0; column < ids.size(); ++column)
            if (columns[column].written)
            {
                const std::size_t size = pdg::dimensionTypeSize(types[column]);
                std::memcpy(expected.data() + mapping[point] * PointStride +
                                offsets[column],
                            packed.data() + mapping[point] * PointStride +
                                offsets[column],
                            size);
            }
    EXPECT_EQ(output, expected);
}

} // unnamed namespace
