#pragma once

#include <pdg/stages/Expression.hpp>

#include <span>
#include <string>

namespace pdg
{

// Compiles PDAL's filters.range limits into the same stable predicate VM used
// by filters.expression. Ranges for one dimension are ORed; dimension groups
// are ANDed. Bound inclusivity, negation, omitted bounds, and NaN behavior
// match the pinned upstream DimRange implementation.
[[nodiscard]] PredicateProgram
compileRangePredicate(std::span<const std::string> specifications,
                      DimensionRegistry& dimensions);

} // namespace pdg
