#pragma once

#include <cstddef>
#include <functional>
#include <cstdint>

namespace pdg
{

class PointBatch;

// The provisional device lane uses a global raster and a deterministic direct
// eight-neighbor inpaint. B0214--B0216 proved that its cutoff-tie rule differs
// from the pinned oracle's KD2Index traversal. Keep the compiled limits for
// scratch planning and future repair, but do not qualify the lane until its
// void fill matches the oracle byte for byte.
inline constexpr bool SmrfExactDeviceQualified = false;
inline constexpr std::size_t SmrfExactDeviceMaximumRasterCells = 4096U;
inline constexpr int SmrfExactDeviceMaximumMorphologyRadius = 64;
inline constexpr std::size_t SmrfExactDeviceMaximumFixedScratchBytes =
    SmrfExactDeviceMaximumRasterCells * 128U + 65536U;

struct SmrfProgram
{
    double cell = 1.0;
    double slope = 0.15;
    double window = 18.0;
    double scalar = 1.25;
    double threshold = 0.5;
    double cut = 0.0;
    std::uint8_t groundClass = 2U;
    std::uint8_t otherClass = 1U;
    bool onlyGround = false;
};

struct SmrfResult
{
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    std::size_t groundPoints = 0U;
    std::size_t nongroundPoints = 0U;
};

// The provisional primitive consumes logical-double XYZ and an unsigned-byte
// Classification column. This predicate must remain false while
// SmrfExactDeviceQualified is false, even for inputs inside the compiled
// resource envelope. Return, ignore, class-bit, and where selection stay in
// the PDAL wrapper so unselected records remain byte-identical.
[[nodiscard]] bool smrfSupportsExactDevice(const PointBatch& hostBatch,
                                           const SmrfProgram& program) noexcept;

// Reproduces the pinned SMRF raster phase order. Host execution is the exact
// reference and device execution uses the same column-major grid contract.
// B0216: the void fill's neighbour choice must match the pinned oracle's.
//
// Upstream fills a raster void with the mean of its eight nearest non-void
// cells, chosen through a 2D KD-tree over cell centres. B0214 proved the two
// implementations disagree in exactly the cells where a distance tie at the
// eighth neighbour must be broken, and B0215 showed upstream's resolution is
// that tree's traversal order, which reduces to no short rule worth copying.
//
// So the selection is supplied by the caller rather than approximated here:
// the PDAL wrapper builds the same index type the oracle builds and matches by
// construction. `values` is column-major with `cell = column * rows + row`,
// NaN marking a void; the callee must fill every NaN and change nothing else.
// An empty function keeps pdg's own nearest-cell rule, which is exact for
// every void whose eighth neighbour is uncontested.
using RasterVoidFill =
    std::function<void(std::vector<double>& values, std::size_t rows,
                       std::size_t columns, double minimumX, double minimumY,
                       double cell)>;

[[nodiscard]] SmrfResult classifySmrf(PointBatch& batch,
                                      const SmrfProgram& program,
                                      const RasterVoidFill& fill = {});

} // namespace pdg
