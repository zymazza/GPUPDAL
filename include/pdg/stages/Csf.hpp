#pragma once

#include <cstddef>
#include <cstdint>

namespace pdg
{

class PointBatch;

// The first exact CSF lane uses a bounded whole-view cloth and preserves the
// pinned non-OpenMP serial constraint order. It excludes slope smoothing and
// is not yet reusable across stages.
inline constexpr std::size_t CsfExactDeviceMaximumClothCells = 4096U;
inline constexpr int CsfExactDeviceMaximumIterations = 64;
inline constexpr std::size_t CsfExactDeviceMaximumFixedScratchBytes =
    CsfExactDeviceMaximumClothCells * 160U + 65536U;
inline constexpr std::size_t CsfExactDeviceScratchBytesPerPoint = 1U;

// The pinned oracle build does not enable OpenMP, so its constraint and raster
// loops execute in source order. An OpenMP-enabled build has racy shared
// particle/visited writes and is not part of this exact CUDA envelope.
#if defined(_OPENMP)
inline constexpr bool CsfPinnedOracleHasSerialExecution = false;
#else
inline constexpr bool CsfPinnedOracleHasSerialExecution = true;
#endif

struct CsfProgram
{
    bool smooth = true;
    double timeStep = 0.65;
    double classThreshold = 0.5;
    double heightThreshold = 0.3;
    double resolution = 1.0;
    int rigidness = 3;
    int iterations = 500;
    std::uint8_t groundClass = 2U;
    std::uint8_t otherClass = 1U;
    bool onlyGround = false;
};

struct CsfResult
{
    std::size_t width = 0U;
    std::size_t height = 0U;
    std::size_t iterationsExecuted = 0U;
    std::size_t groundPoints = 0U;
    std::size_t nongroundPoints = 0U;
};

[[nodiscard]] bool
csfProgramWithinExactDeviceEnvelope(const CsfProgram& program) noexcept;

[[nodiscard]] bool csfSupportsExactDevice(const PointBatch& hostBatch,
                                          const CsfProgram& program) noexcept;

// Reproduces the pinned serial CSF coordinate transform, cloth graph,
// rasterization/fill order, in-place constraints, collision, and strict final
// interpolation. OpenMP-enabled and smoothed variants remain upstream work.
[[nodiscard]] CsfResult classifyCsf(PointBatch& batch,
                                    const CsfProgram& program);

} // namespace pdg
