#pragma once

#include <pdg/Dimension.hpp>

#include <cstdint>
#include <vector>

namespace pdg
{

class PointBatch;

struct LabelDuplicatesProgram
{
    // Upstream compares each listed field after conversion to double. An
    // empty list is intentionally valid and makes every row after the first
    // vacuously equal to its predecessor.
    std::vector<DimensionId> dimensions;
};

// Reports whether every declared input column can be copied to a device batch
// without changing the pinned upstream double-conversion comparison.
[[nodiscard]] bool labelDuplicatesMaySupportExactDevice(
    const PointBatch& hostBatch,
    const LabelDuplicatesProgram& program) noexcept;

// Writes one byte for rows [1, size); row zero is deliberately untouched to
// match filters.label_duplicates when Duplicate already exists on input. The
// output pointer belongs to the same memory domain as batch.
void labelDuplicates(PointBatch& batch, const LabelDuplicatesProgram& program,
                     std::uint8_t* output);

} // namespace pdg
