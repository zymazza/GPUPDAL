#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

#include <stdexcept>

namespace pdg
{

CsfResult classifyCsfDevice(PointBatch&, const CsfProgram&)
{
    throw std::runtime_error("CUDA CSF support was not compiled");
}

} // namespace pdg
