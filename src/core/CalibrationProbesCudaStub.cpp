#include <pdg/CalibrationProbes.hpp>

#include <stdexcept>

namespace pdg
{

double probeCudaStartupNanoseconds()
{
    throw std::runtime_error("CUDA backend not compiled");
}

CalibrationTransferProbe probeCudaTransfers(std::size_t, int)
{
    throw std::runtime_error("CUDA backend not compiled");
}

double probeCudaSynchronizationNanoseconds(int)
{
    throw std::runtime_error("CUDA backend not compiled");
}

double probeIndexBuildNanosecondsPerByte(std::size_t, int)
{
    throw std::runtime_error("CUDA backend not compiled");
}

} // namespace pdg
