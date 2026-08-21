#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <stdexcept>

namespace pdg
{

BoundsResult summarizeBoundsDevice(PointBatch&, std::uint64_t)
{
    throw std::runtime_error("CUDA support is not compiled in");
}

} // namespace pdg
