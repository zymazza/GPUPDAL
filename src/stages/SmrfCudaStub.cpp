#include <pdg/PointBatch.hpp>
#include <pdg/stages/Smrf.hpp>

#include <stdexcept>

namespace pdg
{

SmrfResult classifySmrfDevice(PointBatch&, const SmrfProgram&)
{
    throw std::runtime_error("CUDA SMRF support is not available");
}

} // namespace pdg
