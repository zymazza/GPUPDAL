#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/LabelDuplicates.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
template <typename T>
void fillPairs(pdg::PointBatch& batch, pdg::DimensionId dimension)
{
    const T values[] = {T{1}, T{1}, T{2}, T{2}, T{3}, T{3}};
    std::copy(std::begin(values), std::end(values), batch.data<T>(dimension));
}

TEST(CudaLabelDuplicates, MatchesHostAcrossPhysicalTypesAndSpecialValues)
{
    if (pdg::cudaDevices().empty())
        GTEST_SKIP() << "No CUDA device available";

    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const auto& s8 =
        dimensions.registerCustom("S8", pdg::DimensionType::Signed8);
    const auto& s16 =
        dimensions.registerCustom("S16", pdg::DimensionType::Signed16);
    const auto& s32 =
        dimensions.registerCustom("S32", pdg::DimensionType::Signed32);
    const auto& s64 =
        dimensions.registerCustom("S64", pdg::DimensionType::Signed64);
    const auto& u8 =
        dimensions.registerCustom("U8", pdg::DimensionType::Unsigned8);
    const auto& u16 =
        dimensions.registerCustom("U16", pdg::DimensionType::Unsigned16);
    const auto& u32 =
        dimensions.registerCustom("U32", pdg::DimensionType::Unsigned32);
    const auto& u64 =
        dimensions.registerCustom("U64", pdg::DimensionType::Unsigned64);
    const auto& f32 =
        dimensions.registerCustom("F32", pdg::DimensionType::Float);
    const auto& f64 =
        dimensions.registerCustom("F64", pdg::DimensionType::Double);
    const pdg::CoordinateEncoding coordinates({0.25, 1.0, 1.0},
                                              {1000.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    pdg::PointBatch host(6U, coordinates, dimensions, hostMemory);
    const std::vector<std::pair<pdg::DimensionId, pdg::DimensionType>> columns{
        {x, pdg::DimensionType::Signed32},
        {s8.id, pdg::DimensionType::Signed8},
        {s16.id, pdg::DimensionType::Signed16},
        {s32.id, pdg::DimensionType::Signed32},
        {s64.id, pdg::DimensionType::Signed64},
        {u8.id, pdg::DimensionType::Unsigned8},
        {u16.id, pdg::DimensionType::Unsigned16},
        {u32.id, pdg::DimensionType::Unsigned32},
        {u64.id, pdg::DimensionType::Unsigned64},
        {f32.id, pdg::DimensionType::Float},
        {f64.id, pdg::DimensionType::Double},
    };
    for (const auto& [dimension, type] : columns)
        host.materialize(dimension, type);
    host.setSize(6U);
    fillPairs<std::int32_t>(host, x);
    fillPairs<std::int8_t>(host, s8.id);
    fillPairs<std::int16_t>(host, s16.id);
    fillPairs<std::int32_t>(host, s32.id);
    fillPairs<std::int64_t>(host, s64.id);
    fillPairs<std::uint8_t>(host, u8.id);
    fillPairs<std::uint16_t>(host, u16.id);
    fillPairs<std::uint32_t>(host, u32.id);
    fillPairs<std::uint64_t>(host, u64.id);
    const float f32Values[] = {1.0F,
                               1.0F,
                               -0.0F,
                               0.0F,
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN()};
    const double f64Values[] = {1.0,
                                1.0,
                                -0.0,
                                0.0,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()};
    std::copy(std::begin(f32Values), std::end(f32Values),
              host.data<float>(f32.id));
    std::copy(std::begin(f64Values), std::end(f64Values),
              host.data<double>(f64.id));

    pdg::LabelDuplicatesProgram program;
    for (const auto& [dimension, type] : columns)
    {
        static_cast<void>(type);
        program.dimensions.push_back(dimension);
    }
    ASSERT_TRUE(pdg::labelDuplicatesMaySupportExactDevice(host, program));
    std::vector<std::uint8_t> expected(host.size(), 9U);
    pdg::labelDuplicates(host, program, expected.data());

    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(host.size(), coordinates, dimensions, *deviceMemory);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (const auto& [dimension, type] : columns)
    {
        device.materialize(dimension, type);
        const std::size_t bytes = host.size() * pdg::dimensionTypeSize(type);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                       host.rawData(dimension), bytes,
                                       cudaMemcpyHostToDevice, stream));
    }
    device.setSize(host.size());
    std::unique_ptr<pdg::Allocation> output = deviceMemory->allocate(
        host.size() * sizeof(std::uint8_t), alignof(std::uint8_t));
    PDG_CUDA_CHECK(cudaMemsetAsync(output->data(), 9, host.size(), stream));
    pdg::labelDuplicates(device, program,
                         static_cast<std::uint8_t*>(output->data()));
    std::vector<std::uint8_t> actual(host.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), output->data(), actual.size(),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(expected, (std::vector<std::uint8_t>{9U, 1U, 0U, 1U, 0U, 0U}));
    EXPECT_EQ(actual, expected);
}
} // unnamed namespace
