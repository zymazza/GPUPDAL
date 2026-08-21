#include <pdg/stages/Summary.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

void updateSummariesDevice(PointBatch&, std::span<const DimensionId>,
                           SummaryState*, bool)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
