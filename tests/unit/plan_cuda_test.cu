#include <pdg/Cuda.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <memory>

namespace
{
bool cudaPlanDeviceAvailable()
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
} // unnamed namespace

TEST(CudaPlan, ResidentLivenessReturnsActualDeviceColumns)
{
    if (!cudaPlanDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign","value":"TmpA = Intensity * 2"},
             {"type":"filters.assign","value":"TmpB = TmpA + 1"},
             {"type":"filters.assign","value":"Classification = TmpB"},
             "output.las"])",
        dimensions);
    std::unique_ptr<pdg::MemoryResource> memory = pdg::makeCudaMemoryResource();
    pdg::PointBatch batch(10U, {{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}},
                          dimensions, *memory);

    pdg::preparePlannedDeviceColumns(batch, plan.stages()[1]);
    EXPECT_EQ(batch.allocatedBytes(), 100U);
    pdg::releasePlannedDeviceColumns(batch, plan.stages()[1]);
    EXPECT_EQ(batch.allocatedBytes(), 80U);
    pdg::preparePlannedDeviceColumns(batch, plan.stages()[2]);
    EXPECT_EQ(batch.allocatedBytes(), 160U);
    pdg::releasePlannedDeviceColumns(batch, plan.stages()[2]);
    EXPECT_EQ(batch.allocatedBytes(), 80U);
    pdg::preparePlannedDeviceColumns(batch, plan.stages()[3]);
    EXPECT_EQ(batch.allocatedBytes(), 90U);
    pdg::releasePlannedDeviceColumns(batch, plan.stages()[3]);
    EXPECT_EQ(batch.allocatedBytes(), 10U);

    const auto spill = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(), [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Spill; });
    ASSERT_NE(spill, plan.summary().residencyBoundaries.end());
    pdg::releaseSpilledDeviceColumns(batch, *spill);
    EXPECT_EQ(batch.allocatedBytes(), 0U);
    PDG_CUDA_CHECK(cudaStreamSynchronize(
        static_cast<cudaStream_t>(memory->nativeStreamHandle())));
}
