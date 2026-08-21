#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/stages/Expression.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pdg
{

class PointBatch;

struct HistogramBin
{
    double value = 0.0;
    std::uint64_t count = 0;
    std::uint64_t firstIndex = 0;
};

// Clean RTX 4090 process benchmarks with persistent staging establish
// thresholds at 4M/2M/1M points for one/two/three-or-more expressions. Keep
// automatic execution inside that measured work curve; exact device-envelope
// checks still run before launch.
[[nodiscard]] bool
preferDefaultCudaExpressionStats(std::size_t points,
                                 std::size_t expressions) noexcept;

// The exact device envelope requires a finite materialized Double target and
// an expression supported by the exact predicate VM. Predicate inputs may
// still contain non-finite values.
[[nodiscard]] bool
histogramMaySupportExactDevice(const PointBatch& hostBatch, DimensionId target,
                               const PredicateProgram& predicate) noexcept;

// Returns bins in std::map<double> order. `value` preserves the bit pattern of
// the first matching point in an equivalence class (notably +0 versus -0).
[[nodiscard]] std::vector<HistogramBin> selectedHistogram(
    PointBatch& batch, DimensionId target, const PredicateProgram& predicate,
    std::uint64_t indexOffset = 0, std::size_t maximumHostWorkers = 0);

} // namespace pdg
