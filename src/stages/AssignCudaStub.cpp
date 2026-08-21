#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>

#include <stdexcept>

namespace pdg
{

void executeAssignDevice(PointBatch&, const AssignProgram&)
{
    throw std::runtime_error("PDG was built without CUDA assignment support");
}

void evaluatePredicateDevice(PointBatch&, const PredicateProgram&,
                             std::uint8_t*)
{
    throw std::runtime_error("PDG was built without CUDA predicate support");
}

} // namespace pdg
