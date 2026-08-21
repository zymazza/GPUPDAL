#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ferry.hpp>

#include <stdexcept>

namespace pdg
{

void executeFerryDevice(PointBatch&, const FerryProgram&)
{
    throw std::runtime_error("PDG was built without CUDA ferry support");
}

} // namespace pdg
