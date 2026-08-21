#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
bool orderingDeviceAvailable()
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

struct AllocationTally
{
    std::size_t current = 0U;
    std::size_t peak = 0U;
};

class TalliedAllocation final : public pdg::Allocation
{
public:
    TalliedAllocation(std::unique_ptr<pdg::Allocation> allocation,
                      std::shared_ptr<AllocationTally> tally)
        : m_allocation(std::move(allocation)), m_tally(std::move(tally))
    {
    }

    ~TalliedAllocation() override
    {
        const std::size_t bytes = m_allocation->size();
        m_allocation.reset();
        m_tally->current -= bytes;
    }

    void* data() noexcept override
    {
        return m_allocation->data();
    }

    const void* data() const noexcept override
    {
        return m_allocation->data();
    }

    std::size_t size() const noexcept override
    {
        return m_allocation->size();
    }

    pdg::MemoryKind kind() const noexcept override
    {
        return m_allocation->kind();
    }

private:
    std::unique_ptr<pdg::Allocation> m_allocation;
    std::shared_ptr<AllocationTally> m_tally;
};

class TalliedDeviceResource final : public pdg::MemoryResource
{
public:
    explicit TalliedDeviceResource(std::size_t releaseThreshold)
        : m_resource(pdg::makeCudaMemoryResource(releaseThreshold)),
          m_tally(std::make_shared<AllocationTally>())
    {
    }

    std::unique_ptr<pdg::Allocation> allocate(std::size_t bytes,
                                              std::size_t alignment) override
    {
        std::unique_ptr<pdg::Allocation> allocation =
            m_resource->allocate(bytes, alignment);
        m_tally->current += bytes;
        m_tally->peak = (std::max)(m_tally->peak, m_tally->current);
        return std::make_unique<TalliedAllocation>(std::move(allocation),
                                                   m_tally);
    }

    pdg::MemoryKind kind() const noexcept override
    {
        return m_resource->kind();
    }

    void* nativeStreamHandle() const noexcept override
    {
        return m_resource->nativeStreamHandle();
    }

    std::size_t peak() const noexcept
    {
        return m_tally->peak;
    }

private:
    std::unique_ptr<pdg::MemoryResource> m_resource;
    std::shared_ptr<AllocationTally> m_tally;
};

TEST(CudaOrdering, BoundsDirectDoubleKeyHighWater)
{
    if (!orderingDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 600'000U;
    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("Key", pdg::DimensionType::Double);
    TalliedDeviceResource deviceMemory(128U * 1024U * 1024U);
    pdg::PointBatch device(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, deviceMemory);
    device.materialize(key.id, pdg::DimensionType::Double);
    device.setSize(Count);
    std::vector<double> values(Count);
    for (std::size_t point = 0U; point < Count; ++point)
        values[point] = static_cast<double>(point) + 0.25;
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(key.id), values.data(),
                                   Count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> permutation = deviceMemory.allocate(
        Count * sizeof(std::uint64_t), alignof(std::uint64_t));
    pdg::OrderingProgram program;
    program.dimensions = {key.id};
    ASSERT_TRUE(
        pdg::orderPoints(device, program,
                         static_cast<std::uint64_t*>(permutation->data()))
            .exact);
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_GT(deviceMemory.peak(), Count * 40U);
    EXPECT_LE(deviceMemory.peak(),
              Count * pdg::OrderingExactDevicePeakBytesPerPoint);
}

TEST(CudaOrdering, StableAndUniqueNormalOrdersMatchHost)
{
    if (!orderingDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    const auto& repeated =
        dimensions.registerCustom("Repeated", pdg::DimensionType::Signed32);
    const auto& unique =
        dimensions.registerCustom("Unique", pdg::DimensionType::Double);
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(96U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(repeated.id, pdg::DimensionType::Signed32);
        batch->materialize(unique.id, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    auto repeatedValues = host.hostSpan<std::int32_t>(repeated.id);
    auto uniqueValues = host.hostSpan<double>(unique.id);
    for (std::size_t point = 0; point < Count; ++point)
    {
        repeatedValues[point] =
            static_cast<std::int32_t>((point * 7919U) % 257U);
        uniqueValues[point] =
            static_cast<double>((point * 7919U) % 262147U) + 0.25;
    }

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        device.rawData(repeated.id), host.rawData(repeated.id),
        Count * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        device.rawData(unique.id), host.rawData(unique.id),
        Count * sizeof(double), cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> devicePermutation = deviceMemory->allocate(
        Count * sizeof(std::uint64_t), alignof(std::uint64_t));

    const auto compare = [&](const pdg::OrderingProgram& program)
    {
        std::vector<std::uint64_t> expected(Count);
        std::vector<std::uint64_t> actual(Count);
        ASSERT_TRUE(pdg::orderPoints(host, program, expected.data()).exact);
        const pdg::OrderingResult result = pdg::orderPoints(
            device, program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        ASSERT_TRUE(result.exact);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), devicePermutation->data(),
                                       Count * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, expected);
    };

    pdg::OrderingProgram stable;
    stable.dimensions = {repeated.id};
    stable.algorithm = pdg::OrderingAlgorithm::Stable;
    compare(stable);

    pdg::OrderingProgram normal;
    normal.dimensions = {unique.id};
    normal.direction = pdg::OrderingDirection::Descending;
    compare(normal);

    pdg::OrderingProgram multi;
    multi.dimensions = {repeated.id, unique.id};
    compare(multi);
}

TEST(CudaOrdering, ReportsNormalTieEnvelopeBeforePublication)
{
    if (!orderingDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("Key", pdg::DimensionType::Unsigned16);
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(
        4, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    device.materialize(key.id, pdg::DimensionType::Unsigned16);
    device.setSize(4);
    const std::uint16_t values[] = {2, 1, 2, 0};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(key.id), values,
                                   sizeof(values), cudaMemcpyHostToDevice,
                                   stream));
    std::unique_ptr<pdg::Allocation> permutation = deviceMemory->allocate(
        4 * sizeof(std::uint64_t), alignof(std::uint64_t));
    pdg::OrderingProgram program;
    program.dimensions = {key.id};
    EXPECT_FALSE(
        pdg::orderPoints(device, program,
                         static_cast<std::uint64_t*>(permutation->data()))
            .exact);
}

TEST(CudaOrdering, ReportsSignedZeroComparatorTieBeforePublication)
{
    if (!orderingDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("Key", pdg::DimensionType::Double);
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(
        4, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    device.materialize(key.id, pdg::DimensionType::Double);
    device.setSize(4);
    const double values[] = {2.0, +0.0, -0.0, 1.0};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(key.id), values,
                                   sizeof(values), cudaMemcpyHostToDevice,
                                   stream));
    std::unique_ptr<pdg::Allocation> permutation = deviceMemory->allocate(
        4 * sizeof(std::uint64_t), alignof(std::uint64_t));
    pdg::OrderingProgram program;
    program.dimensions = {key.id};
    EXPECT_FALSE(
        pdg::orderPoints(device, program,
                         static_cast<std::uint64_t*>(permutation->data()))
            .exact);
}
} // unnamed namespace
