#include <pdg/stages/Ordinal.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void evaluateOrdinalDevice(PointBatch&, const OrdinalProgram&, std::uint64_t,
                           std::uint64_t, std::uint64_t, std::uint64_t,
                           std::uint8_t*)
{
    throw std::runtime_error("CUDA support is not enabled");
}

} // namespace pdg
