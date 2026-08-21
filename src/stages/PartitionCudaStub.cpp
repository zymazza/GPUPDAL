#include <pdg/stages/Partition.hpp>

#include <stdexcept>

namespace pdg
{

class PointBatch;

ReturnsPartitionResult
partitionReturnsDevice(PointBatch&, const ReturnsProgram&, std::uint64_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

DividerPartitionResult
partitionDividerDevice(PointBatch&, const DividerProgram&, std::uint64_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

void computeSplitterCellsDevice(PointBatch&, const SplitterProgram&,
                                std::int32_t*, std::int32_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
