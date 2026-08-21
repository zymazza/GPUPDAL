#include <pdg/stages/Transformation.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void executeTransformationDevice(PointBatch&, const TransformationProgram&)
{
    throw std::runtime_error("CUDA support is not enabled");
}

} // namespace pdg
