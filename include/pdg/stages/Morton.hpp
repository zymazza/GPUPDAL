#pragma once

#include <cstddef>
#include <cstdint>

namespace pdg
{

class PointBatch;

struct MortonBounds
{
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 0.0;
    double maxY = 0.0;
};

struct MortonProgram
{
    MortonBounds bounds;
    bool reverse = false;
};

// Clean RTX 4090 process benchmarks cross over between one and two million
// points. The default starts at the first size with at least 20% median margin
// over the pinned PDAL process; exact envelope checks still run before launch.
[[nodiscard]] bool
preferDefaultCudaMorton(std::size_t points,
                        const MortonProgram& program) noexcept;

// The exact device envelope covers finite, nondegenerate XY bounds and
// logical double coordinate columns. Degenerate and nonfinite views retain
// upstream PDAL's implementation-defined host behavior.
[[nodiscard]] bool
mortonMaySupportExactDevice(const PointBatch& hostBatch,
                            const MortonProgram& program) noexcept;

// Emits one sortable key per input point. Stable ascending key order matches
// PDAL's multimap traversal for both ordinary and reverse Morton modes inside
// the exact envelope.
void generateMortonKeys(PointBatch& batch, const MortonProgram& program,
                        std::uint64_t* keys);

} // namespace pdg
