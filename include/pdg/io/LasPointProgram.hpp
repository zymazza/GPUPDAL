#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Ordinal.hpp>

#include <cstddef>
#include <span>
#include <variant>
#include <vector>

namespace pdg::las
{

using PointOperation =
    std::variant<AssignProgram, PredicateProgram, OrdinalProgram>;

// Ordered point-local operations lowered from a linear PDAL region. Assign
// and ferry stages remain in declaration order, while value predicates and
// global ordinal selectors retain stable survivor order.
struct OrderedPointProgram
{
    std::vector<PointOperation> operations;
    std::vector<DimensionId> reads;
    std::vector<DimensionId> writes;
    bool filtersPoints = false;
    OrdinalMode ordinalMode = OrdinalMode::Streaming;
};

// Executes an ordered, fused point program against the canonical format-7
// output produced by translateDefaultInto. The selector is intentionally
// conservative: unsupported input fields and layout conversions remain on
// pinned PDAL.
[[nodiscard]] bool
supportsDefaultPointProgram(const FileView& input, const AssignProgram& program,
                            const DimensionRegistry& dimensions) noexcept;

[[nodiscard]] bool
supportsDefaultPointProgram(const FileView& input,
                            const OrderedPointProgram& program,
                            const DimensionRegistry& dimensions) noexcept;

void applyDefaultPointProgram(std::span<std::byte> canonicalOutput,
                              const FileView& input,
                              const AssignProgram& program,
                              DimensionRegistry& dimensions,
                              std::size_t maximumHostWorkers = 0);

} // namespace pdg::las
