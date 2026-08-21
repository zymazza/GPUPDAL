#pragma once

#include <pdg/Dimension.hpp>

#include <cstdint>

namespace pdg
{

class PointBatch;

enum class RobustKind : std::uint8_t
{
    Iqr,
    Mad
};

struct RobustProgram
{
    DimensionId dimension;
    RobustKind kind = RobustKind::Iqr;
    double multiplier = 1.5;
    double madMultiplier = 1.4862;
};

struct RobustResult
{
    double first = 0.0;
    double second = 0.0;
    double scale = 0.0;
    double lowFence = 0.0;
    double highFence = 0.0;
};

// Device order statistics are exact for finite logical values without a
// negative-zero key. Host execution retains the full upstream nth_element
// behavior for every physical dimension type.
[[nodiscard]] bool
robustSupportsExactDevice(const PointBatch& hostBatch,
                          const RobustProgram& program) noexcept;

[[nodiscard]] RobustResult evaluateRobust(PointBatch& batch,
                                          const RobustProgram& program,
                                          std::uint8_t* keep);

} // namespace pdg
