#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <stdexcept>
#include <vector>

namespace pdg
{

std::vector<HistogramBin> selectedHistogramDevice(PointBatch&, DimensionId,
                                                  const PredicateProgram&,
                                                  std::uint64_t)
{
    throw std::runtime_error("CUDA support is not compiled in");
}

} // namespace pdg
