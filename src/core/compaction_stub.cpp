#include <pdg/Compaction.hpp>

#include <stdexcept>

namespace pdg
{

std::size_t compactPointBatchDevice(const PointBatch&, PointBatch&,
                                    std::span<const DimensionId>,
                                    const std::uint8_t*)
{
    throw std::runtime_error("CUDA backend was not compiled");
}

} // namespace pdg
