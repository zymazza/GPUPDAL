#pragma once

#include <pdg/stages/Locate.hpp>

#include <cstdint>
#include <limits>

namespace pdg
{

class PointBatch;

// Extrema retain the first global point index for equal values. This makes
// signed-zero ties and chunk merges reproduce BOX3D::grow's ordered behavior.
struct BoundsResult
{
    LocateResult minimum[3] = {
        LocateResult{(std::numeric_limits<double>::max)(), 0, 0, 0},
        LocateResult{(std::numeric_limits<double>::max)(), 0, 0, 0},
        LocateResult{(std::numeric_limits<double>::max)(), 0, 0, 0}};
    LocateResult maximum[3] = {
        LocateResult{(std::numeric_limits<double>::lowest)(), 0, 0, 0},
        LocateResult{(std::numeric_limits<double>::lowest)(), 0, 0, 0},
        LocateResult{(std::numeric_limits<double>::lowest)(), 0, 0, 0}};
    std::uint64_t count = 0;
};

// Reduces X/Y/Z and count without reassociating coordinate decode. Device
// execution performs all six extrema in one reduction pipeline.
[[nodiscard]] BoundsResult summarizeBounds(PointBatch& batch,
                                           std::uint64_t indexOffset = 0);

// Inputs must describe consecutive ranges in increasing global point order.
[[nodiscard]] BoundsResult mergeBoundsResults(const BoundsResult& first,
                                              const BoundsResult& second);

} // namespace pdg
