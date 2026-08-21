#pragma once

// D0277: direct measurements of the planner-owned placement coefficients on
// this machine, used only by `gpupdal calibrate`.  Each probe is a small,
// self-contained CUDA measurement; none of them touch the planner.

#include <cstddef>

namespace pdg
{

struct CalibrationTransferProbe
{
    double hostToDeviceNanosecondsPerByte = 0.0;
    double deviceToHostNanosecondsPerByte = 0.0;
};

// Creates the CUDA context on the current device and touches it once; call it
// first in a fresh process to measure the process cold-start cost.  Throws
// when the CUDA backend is unavailable.
[[nodiscard]] double probeCudaStartupNanoseconds();

// Median over `repeats` pageable memcpy round trips of `bytes` bytes.
[[nodiscard]] CalibrationTransferProbe
probeCudaTransfers(std::size_t bytes, int repeats);

// Median over `repeats` of an empty-kernel launch followed by
// cudaDeviceSynchronize (the planner's per-synchronization charge).
[[nodiscard]] double probeCudaSynchronizationNanoseconds(int repeats);

// Builds the shared kNN spatial index over `points` synthetic points on the
// device `repeats` times and returns the median build wall divided by the
// index's persistent bytes (the reference profile's definition).  The
// synthetic points follow the same terrain generator as the calibration
// fixture so the cell occupancy is representative.
[[nodiscard]] double probeIndexBuildNanosecondsPerByte(std::size_t points,
                                                       int repeats);

} // namespace pdg
