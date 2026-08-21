#include <pdg/stages/ColorMap.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void applyColorMapDevice(PointBatch&, const ColorMapProgram&, ColorRampView)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
