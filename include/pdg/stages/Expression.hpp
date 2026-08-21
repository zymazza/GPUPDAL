#pragma once

#include <pdg/Expression.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace pdg
{

class PointBatch;

struct PredicateProgram
{
    CompiledExpression expression;
    std::vector<DimensionId> reads;
};

[[nodiscard]] PredicateProgram compilePredicate(std::string_view specification,
                                                DimensionRegistry& dimensions);

[[nodiscard]] bool
predicateSupportsExactDevice(const PredicateProgram& program) noexcept;

// B0162: plan-time residency for predicates, permitting coordinate loads.
//
// `predicateSupportsExactDevice(program)` refuses any predicate reading X, Y or
// Z, because a coordinate load is exact on device only for a binary64 column,
// or a signed-32 column whose scale offset is zero — a property of the runtime
// batch rather than of the plan. Refusing it at plan time made `filters.crop`,
// which is inherently a coordinate predicate, permanently unplaceable, and
// refused `filters.range` on `Z` while admitting the identical filter on
// `Intensity` (B0159).
//
// This variant answers the question the plan can actually answer — is every
// operation exact on device — and defers the coordinate-column question to the
// two places that can answer it against a real batch: the resident preflight,
// which fails closed before commitment, and `evaluatePredicateDevice`. B0161
// and B0162 verified that every device predicate path in the engine builds its
// batch with a zero coordinate offset, so those checks agree with each other.
[[nodiscard]] bool
predicateMaySupportExactDevice(const PredicateProgram& program) noexcept;
[[nodiscard]] bool
predicateSupportsExactDevice(const PointBatch& batch,
                             const PredicateProgram& program) noexcept;

// Writes one byte per point (zero rejects, one keeps). Host calls complete
// before returning; device calls enqueue onto the batch allocator's stream.
void evaluatePredicate(PointBatch& batch, const PredicateProgram& program,
                       std::uint8_t* keep, std::size_t maximumHostWorkers = 0);

} // namespace pdg
