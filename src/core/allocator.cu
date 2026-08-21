#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pdg
{

CudaError::CudaError(int code, std::string message)
    : std::runtime_error(std::move(message)), m_code(code)
{
}

int CudaError::code() const noexcept
{
    return m_code;
}

namespace cuda
{
namespace
{
std::string message(cudaError_t result, const char* expression,
                    const char* file, int line)
{
    std::ostringstream output;
    output << file << ':' << line << ": CUDA call " << expression
           << " failed: " << cudaGetErrorName(result) << " ("
           << cudaGetErrorString(result) << ')';
    return output.str();
}
} // unnamed namespace

void check(cudaError_t result, const char* expression, const char* file,
           int line)
{
    if (result != cudaSuccess)
        throw CudaError(static_cast<int>(result),
                        message(result, expression, file, line));
}

void checkNoexcept(cudaError_t result, const char* expression, const char* file,
                   int line) noexcept
{
    if (result == cudaSuccess)
        return;
    const std::string error = message(result, expression, file, line);
    std::fputs(error.c_str(), stderr);
    std::fputc('\n', stderr);
    std::terminate();
}
} // namespace cuda

namespace
{
struct PoolState
{
    cudaMemPool_t pool = nullptr;
    cudaStream_t stream = nullptr;

    ~PoolState()
    {
        if (stream)
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(stream));
        if (stream)
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream));
        if (pool)
            PDG_CUDA_CHECK_NOEXCEPT(cudaMemPoolDestroy(pool));
    }
};

struct ClassicState
{
    cudaStream_t stream = nullptr;

    ~ClassicState()
    {
        if (stream)
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(stream));
        if (stream)
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream));
    }
};

class CudaAllocation final : public Allocation
{
public:
    CudaAllocation(std::shared_ptr<PoolState> state, void* data,
                   std::size_t size)
        : m_state(std::move(state)), m_data(data), m_size(size)
    {
    }

    ~CudaAllocation() override
    {
        if (m_data)
            PDG_CUDA_CHECK_NOEXCEPT(cudaFreeAsync(m_data, m_state->stream));
    }

    void* data() noexcept override
    {
        return m_data;
    }

    const void* data() const noexcept override
    {
        return m_data;
    }

    std::size_t size() const noexcept override
    {
        return m_size;
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::Device;
    }

private:
    std::shared_ptr<PoolState> m_state;
    void* m_data;
    std::size_t m_size;
};

class CudaMemoryResource final : public MemoryResource
{
public:
    explicit CudaMemoryResource(std::size_t releaseThreshold)
        : m_state(std::make_shared<PoolState>())
    {
        int device = 0;
        PDG_CUDA_CHECK(cudaGetDevice(&device));

        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = device;

        try
        {
            PDG_CUDA_CHECK(cudaMemPoolCreate(&m_state->pool, &properties));
            std::uint64_t threshold = releaseThreshold;
            PDG_CUDA_CHECK(cudaMemPoolSetAttribute(
                m_state->pool, cudaMemPoolAttrReleaseThreshold, &threshold));
            PDG_CUDA_CHECK(cudaStreamCreateWithFlags(&m_state->stream,
                                                     cudaStreamNonBlocking));
        }
        catch (...)
        {
            if (m_state->stream)
            {
                PDG_CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(m_state->stream));
                m_state->stream = nullptr;
            }
            if (m_state->pool)
            {
                PDG_CUDA_CHECK_NOEXCEPT(cudaMemPoolDestroy(m_state->pool));
                m_state->pool = nullptr;
            }
            throw;
        }
    }

    std::unique_ptr<Allocation> allocate(std::size_t bytes,
                                         std::size_t alignment) override
    {
        if (!alignment || (alignment & (alignment - 1U)) != 0)
            throw std::invalid_argument(
                "CUDA allocation alignment must be a power of two");
        if (alignment > 256)
            throw std::invalid_argument(
                "CUDA pool cannot guarantee alignment above 256 bytes");
        void* data = nullptr;
        if (bytes)
            PDG_CUDA_CHECK(cudaMallocFromPoolAsync(&data, bytes, m_state->pool,
                                                   m_state->stream));
        return std::make_unique<CudaAllocation>(m_state, data, bytes);
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::Device;
    }

    void* nativeStreamHandle() const noexcept override
    {
        return m_state->stream;
    }

private:
    std::shared_ptr<PoolState> m_state;
};

class ClassicCudaAllocation final : public Allocation
{
public:
    ClassicCudaAllocation(std::shared_ptr<ClassicState> state, void* data,
                          std::size_t size)
        : m_state(std::move(state)), m_data(data), m_size(size)
    {
    }

    ~ClassicCudaAllocation() override
    {
        if (m_data)
        {
            // cudaFree is not stream ordered. Synchronizing the allocator's
            // stream first keeps the resource contract equivalent to the
            // async-pool backend before the legacy free can reclaim storage.
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(m_state->stream));
            PDG_CUDA_CHECK_NOEXCEPT(cudaFree(m_data));
        }
    }

    void* data() noexcept override
    {
        return m_data;
    }

    const void* data() const noexcept override
    {
        return m_data;
    }

    std::size_t size() const noexcept override
    {
        return m_size;
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::Device;
    }

private:
    std::shared_ptr<ClassicState> m_state;
    void* m_data;
    std::size_t m_size;
};

class ClassicCudaMemoryResource final : public MemoryResource
{
public:
    ClassicCudaMemoryResource() : m_state(std::make_shared<ClassicState>())
    {
        PDG_CUDA_CHECK(
            cudaStreamCreateWithFlags(&m_state->stream, cudaStreamNonBlocking));
    }

    std::unique_ptr<Allocation> allocate(std::size_t bytes,
                                         std::size_t alignment) override
    {
        if (!alignment || (alignment & (alignment - 1U)) != 0)
            throw std::invalid_argument(
                "CUDA allocation alignment must be a power of two");
        if (alignment > 256)
            throw std::invalid_argument(
                "CUDA allocator cannot guarantee alignment above 256 bytes");
        void* data = nullptr;
        if (bytes)
            PDG_CUDA_CHECK(cudaMalloc(&data, bytes));
        return std::make_unique<ClassicCudaAllocation>(m_state, data, bytes);
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::Device;
    }

    void* nativeStreamHandle() const noexcept override
    {
        return m_state->stream;
    }

private:
    std::shared_ptr<ClassicState> m_state;
};

bool forceLegacyCudaAllocator() noexcept
{
    return std::getenv("PDG_FORCE_LEGACY_CUDA_ALLOCATOR") != nullptr;
}

bool memoryPoolsSupported(int device)
{
    int supported = 0;
    PDG_CUDA_CHECK(cudaDeviceGetAttribute(
        &supported, cudaDevAttrMemoryPoolsSupported, device));
    return supported != 0;
}

class CudaPinnedAllocation final : public Allocation
{
public:
    CudaPinnedAllocation(void* data, std::size_t size)
        : m_data(data), m_size(size)
    {
    }

    ~CudaPinnedAllocation() override
    {
        if (m_data)
            PDG_CUDA_CHECK_NOEXCEPT(cudaFreeHost(m_data));
    }

    void* data() noexcept override
    {
        return m_data;
    }

    const void* data() const noexcept override
    {
        return m_data;
    }

    std::size_t size() const noexcept override
    {
        return m_size;
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::PinnedHost;
    }

private:
    void* m_data;
    std::size_t m_size;
};

class CudaPinnedMemoryResource final : public MemoryResource
{
public:
    std::unique_ptr<Allocation> allocate(std::size_t bytes,
                                         std::size_t alignment) override
    {
        if (!alignment || (alignment & (alignment - 1U)) != 0)
            throw std::invalid_argument(
                "pinned allocation alignment must be a power of two");
        if (alignment > 256)
            throw std::invalid_argument(
                "pinned allocation cannot guarantee alignment above 256 bytes");
        void* data = nullptr;
        if (bytes)
            PDG_CUDA_CHECK(cudaHostAlloc(&data, bytes, cudaHostAllocPortable));
        return std::make_unique<CudaPinnedAllocation>(data, bytes);
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::PinnedHost;
    }

    void* nativeStreamHandle() const noexcept override
    {
        return nullptr;
    }
};
} // unnamed namespace

bool cudaBackendCompiled() noexcept
{
    return true;
}

int cudaCompiledToolkitVersion() noexcept
{
    return CUDART_VERSION;
}

int cudaRuntimeVersion()
{
    int version = 0;
    PDG_CUDA_CHECK(cudaRuntimeGetVersion(&version));
    return version;
}

int cudaDriverVersion()
{
    int version = 0;
    PDG_CUDA_CHECK(cudaDriverGetVersion(&version));
    return version;
}

std::size_t cudaCurrentDeviceFreeMemoryBytes()
{
    std::size_t freeBytes = 0U;
    std::size_t totalBytes = 0U;
    const cudaError_t result = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver)
        return 0U;
    PDG_CUDA_CHECK(result);
    return freeBytes;
}

std::vector<CudaDeviceSummary> cudaDevices()
{
    int count = 0;
    PDG_CUDA_CHECK(cudaGetDeviceCount(&count));
    std::vector<CudaDeviceSummary> devices;
    devices.reserve(static_cast<std::size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        cudaDeviceProp properties{};
        PDG_CUDA_CHECK(cudaGetDeviceProperties(&properties, ordinal));
        devices.push_back({ordinal, properties.name, properties.major,
                           properties.minor, properties.totalGlobalMem,
                           memoryPoolsSupported(ordinal)});
    }
    return devices;
}

std::unique_ptr<MemoryResource>
makeCudaMemoryResource(std::size_t releaseThresholdBytes)
{
    int device = 0;
    PDG_CUDA_CHECK(cudaGetDevice(&device));
    if (forceLegacyCudaAllocator() || !memoryPoolsSupported(device))
        return std::make_unique<ClassicCudaMemoryResource>();
    return std::make_unique<CudaMemoryResource>(releaseThresholdBytes);
}

std::unique_ptr<MemoryResource> makeCudaPinnedMemoryResource()
{
    return std::make_unique<CudaPinnedMemoryResource>();
}

} // namespace pdg
