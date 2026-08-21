#pragma once

#include <pdg/stages/Ordering.hpp>

#include <cstddef>
#include <cstdint>

namespace pdg
{

struct SkewnessProgram
{
    std::uint8_t groundClass = 2U;
    std::uint8_t otherClass = 1U;
    bool onlyGround = false;
};

struct SkewnessResult
{
    std::size_t groundPoints = 0U;
    std::size_t otherPoints = 0U;
};

// The exact skewness device prefix is the same one-key ordering allocation
// used by the direct sort route: one planned binary64 Z column, the caller
// permutation, the alternate permutation, two radix-key buffers, CUB scratch,
// and the duplicate flag. Scratch stays tied to the measured ordering
// high-water; the whole-view peak additionally carries the planner's logical
// binary64 Z and byte Classification columns.
inline constexpr std::size_t SkewnessExactDeviceScratchBytesPerPoint =
    OrderingExactDeviceScratchBytesPerPoint;
inline constexpr std::size_t SkewnessExactDevicePeakBytesPerPoint =
    SkewnessExactDeviceScratchBytesPerPoint + sizeof(double) +
    sizeof(std::uint8_t);

[[nodiscard]] bool
skewnessProgramValid(const SkewnessProgram& program) noexcept;

[[nodiscard]] bool
skewnessOrderingSizeWithinDeviceEnvelope(std::size_t size) noexcept;

// Applies the pinned Bartels/Wei sequential moment recurrence to values that
// are already in the exact upstream Z order. No reduction, reassociation, or
// device libm operation is permitted in this arithmetic-sensitive finale.
[[nodiscard]] SkewnessResult
classifySkewnessSorted(const double* z, std::uint8_t* classification,
                       std::size_t size, const SkewnessProgram& program);

} // namespace pdg
