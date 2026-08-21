#include <pdg/stages/Locate.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

LocateResult locateExtremeDevice(PointBatch&, const LocateProgram&,
                                 std::uint64_t)
{
    throw std::runtime_error("CUDA support is not enabled");
}

} // namespace pdg
