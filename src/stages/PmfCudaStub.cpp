#include <pdg/PointBatch.hpp>
#include <pdg/stages/Pmf.hpp>

#include <stdexcept>

namespace pdg
{

PmfResult classifyPmfDevice(PointBatch&, const PmfProgram&)
{
    throw std::runtime_error("CUDA PMF support is not available");
}

void buildPmfTiledRasterDevice(PointBatch&, const PmfProgram&,
                               RasterGridProduct&, PmfRasterBuildFacts*)
{
    throw std::runtime_error("CUDA PMF support is not compiled");
}

PmfResult classifyPmfTiledDevice(PointBatch&, const PmfProgram&,
                                 RasterGridProduct&, PmfTiledExecutionFacts*)
{
    throw std::runtime_error("CUDA tiled PMF support is not available");
}

} // namespace pdg
