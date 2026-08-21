#include <pdg/Memory.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

TEST(HostMemoryResource, AllocatesAlignedOwnedMemory)
{
    pdg::HostMemoryResource memory;
    EXPECT_EQ(memory.nativeStreamHandle(), nullptr);
    auto allocation = memory.allocate(4096, 64);
    ASSERT_NE(allocation->data(), nullptr);
    EXPECT_EQ(allocation->size(), 4096U);
    EXPECT_EQ(allocation->kind(), pdg::MemoryKind::Host);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(allocation->data()) % 64U, 0U);
}

TEST(HostMemoryResource, HandlesZeroAndRejectsBadAlignment)
{
    pdg::HostMemoryResource memory;
    const auto empty = memory.allocate(0, 1);
    EXPECT_EQ(empty->data(), nullptr);
    EXPECT_EQ(empty->size(), 0U);
    EXPECT_THROW(static_cast<void>(memory.allocate(1, 24)),
                 std::invalid_argument);
}

TEST(CudaRuntimeFacts, FormatsToolkitVersionForCalibrationKeys)
{
    EXPECT_EQ(pdg::formatCudaToolkitVersion(13030), "13.3");
    EXPECT_EQ(pdg::formatCudaToolkitVersion(12041), "12.4.1");
    EXPECT_EQ(pdg::formatCudaToolkitVersion(0), "");
    EXPECT_EQ(pdg::formatCudaToolkitVersion(-1), "");
}

TEST(CudaRuntimeFacts, ParsesExactNvidiaKernelDriverVersion)
{
    constexpr std::string_view Valid =
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
        "610.43.03  Release Build  (builder@example)  Wed Aug  6\n"
        "GCC version:  gcc version 13.3.0\n";
    EXPECT_EQ(pdg::parseNvidiaKernelDriverVersion(Valid), "610.43.03");
}

TEST(CudaRuntimeFacts, RejectsMalformedNvidiaKernelDriverVersion)
{
    for (const std::string_view invalid : {
             "",
             "NVRM version: NVIDIA UNIX 610.43 Release Build",
             "NVRM version: NVIDIA UNIX 610.43.03.1 Release Build",
             "NVRM version: NVIDIA UNIX v610.43.03 Release Build",
             "GCC version: 610.43.03",
         })
        EXPECT_TRUE(pdg::parseNvidiaKernelDriverVersion(invalid).empty())
            << invalid;
}

TEST(CudaRuntimeFacts, StubReportsNoFreeDeviceMemory)
{
    if (!pdg::cudaBackendCompiled())
        EXPECT_EQ(pdg::cudaCurrentDeviceFreeMemoryBytes(), 0U);
}
