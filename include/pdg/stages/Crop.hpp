#pragma once

#include <pdg/stages/Expression.hpp>

#include <string_view>

namespace pdg
{

// Compiles the first exact filters.crop envelope: one canonical PDAL 2D or
// 3D bounds string, with optional outside inversion. Bounds are inclusive and
// NaN handling follows BOX2D/BOX3D::contains.
[[nodiscard]] PredicateProgram
compileCropPredicate(std::string_view bounds, bool outside,
                     DimensionRegistry& dimensions);

} // namespace pdg
