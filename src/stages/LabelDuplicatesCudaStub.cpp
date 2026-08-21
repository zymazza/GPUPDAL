#include <pdg/stages/LabelDuplicates.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void labelDuplicatesDevice(PointBatch&, const LabelDuplicatesProgram&,
                           std::uint8_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
