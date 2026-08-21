#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace pdg
{

enum class MemoryKind
{
    Host,
    PinnedHost,
    Device
};

class Allocation
{
public:
    virtual ~Allocation() = default;
    [[nodiscard]] virtual void* data() noexcept = 0;
    [[nodiscard]] virtual const void* data() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual MemoryKind kind() const noexcept = 0;
};

class MemoryResource
{
public:
    virtual ~MemoryResource() = default;
    [[nodiscard]] virtual std::unique_ptr<Allocation>
    allocate(std::size_t bytes, std::size_t alignment) = 0;
    [[nodiscard]] virtual MemoryKind kind() const noexcept = 0;
    [[nodiscard]] virtual void* nativeStreamHandle() const noexcept = 0;
};

class HostMemoryResource final : public MemoryResource
{
public:
    [[nodiscard]] std::unique_ptr<Allocation>
    allocate(std::size_t bytes, std::size_t alignment) override;
    [[nodiscard]] MemoryKind kind() const noexcept override;
    [[nodiscard]] void* nativeStreamHandle() const noexcept override;
};

struct CudaDeviceSummary
{
    int ordinal = -1;
    std::string name;
    int computeMajor = 0;
    int computeMinor = 0;
    std::size_t totalMemory = 0;
    bool memoryPoolsSupported = false;
};

[[nodiscard]] bool cudaBackendCompiled() noexcept;
[[nodiscard]] int cudaCompiledToolkitVersion() noexcept;
// Formats CUDA's integer version encoding as the stable major.minor[.patch]
// key used by placement calibration profiles. Zero and malformed encodings
// have no usable representation and return an empty string.
[[nodiscard]] std::string formatCudaToolkitVersion(int version);
[[nodiscard]] int cudaRuntimeVersion();
[[nodiscard]] int cudaDriverVersion();
[[nodiscard]] std::vector<CudaDeviceSummary> cudaDevices();
// Returns free bytes on the CUDA device current in this thread.  It returns
// zero when CUDA has no usable device, allowing placement to fail closed.
[[nodiscard]] std::size_t cudaCurrentDeviceFreeMemoryBytes();

// Extracts the exact dotted kernel-module driver version from Linux's
// /proc/driver/nvidia/version content. This pure parser is kept public for
// deterministic unit coverage; an empty result denotes malformed input.
[[nodiscard]] std::string parseNvidiaKernelDriverVersion(std::string_view text);
// Reads the Linux NVIDIA kernel-module version without spawning a process.
// Returns an empty string when the proc file is unavailable or malformed.
[[nodiscard]] std::string nvidiaKernelDriverVersion();
[[nodiscard]] std::unique_ptr<MemoryResource>
makeCudaMemoryResource(std::size_t releaseThresholdBytes = 0);
[[nodiscard]] std::unique_ptr<MemoryResource> makeCudaPinnedMemoryResource();

} // namespace pdg
