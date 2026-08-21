#include <pdg/stages/Morton.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void generateMortonKeysDevice(PointBatch&, const MortonProgram&, std::uint64_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
