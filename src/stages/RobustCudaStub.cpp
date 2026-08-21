#include <pdg/stages/Robust.hpp>

#include <stdexcept>

namespace pdg
{

RobustResult evaluateRobustDevice(PointBatch&, const RobustProgram&,
                                  std::uint8_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
