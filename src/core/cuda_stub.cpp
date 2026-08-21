#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>

#include <stdexcept>

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

bool cudaBackendCompiled() noexcept
{
    return false;
}

int cudaCompiledToolkitVersion() noexcept
{
    return 0;
}

int cudaRuntimeVersion()
{
    return 0;
}

int cudaDriverVersion()
{
    return 0;
}

std::size_t cudaCurrentDeviceFreeMemoryBytes()
{
    return 0U;
}

std::vector<CudaDeviceSummary> cudaDevices()
{
    return {};
}

std::unique_ptr<MemoryResource> makeCudaMemoryResource(std::size_t)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

std::unique_ptr<MemoryResource> makeCudaPinnedMemoryResource()
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
