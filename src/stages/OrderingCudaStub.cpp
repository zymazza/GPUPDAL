#include <pdg/stages/Ordering.hpp>

#include <stdexcept>

namespace pdg
{

OrderingResult orderPointsDevice(PointBatch&, const OrderingProgram&,
                                 std::uint64_t*)
{
    throw std::runtime_error("PDG was built without CUDA support");
}

} // namespace pdg
