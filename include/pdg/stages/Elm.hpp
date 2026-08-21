#pragma once

#include <cstddef>
#include <cstdint>

namespace pdg
{

class PointBatch;

// ELM's first exact device lane owns one immutable whole-view XY frame.  The
// point buckets are stage scratch, not a reusable neighborhood index; the
// planner accounts for them and keeps adjacent Grid composition disabled.
inline constexpr std::size_t ElmExactDeviceMaximumGridCells = 4096U;
inline constexpr std::size_t ElmExactDeviceAllocatorSlackBytes = 65536U;

struct ElmProgram
{
    double cell = 10.0;
    std::uint8_t classification = 7U;
    double threshold = 1.0;
};

struct ElmResult
{
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    std::size_t classifiedPoints = 0U;
};

[[nodiscard]] bool
elmProgramWithinExactDeviceEnvelope(const ElmProgram& program) noexcept;

[[nodiscard]] bool elmSupportsExactDevice(const PointBatch& hostBatch,
                                          const ElmProgram& program) noexcept;

// Queries the CUDA toolkit's CUB reduction/radix-sort temporary-storage
// requirements and returns the maximum live scratch for the bounded ELM
// phase schedule. The resident planner uses this value instead of a fitted
// per-point coefficient.
[[nodiscard]] std::size_t elmExactDeviceScratchBytes(std::size_t pointCount);

// Reproduces the pinned ELM frame, including its deliberate
// floor(coordinate - minimum) / cell bin expression, and the insertion order
// of equal-Z points within each cell.
[[nodiscard]] ElmResult classifyElm(PointBatch& batch,
                                    const ElmProgram& program);

} // namespace pdg
