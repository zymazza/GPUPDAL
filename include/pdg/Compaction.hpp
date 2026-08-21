#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace pdg
{

class PointBatch;

// Stable-selects every listed column with the same byte predicate. Source and
// destination must have identical materialized types and distinct storage.
// Host calls complete before returning; device calls synchronize only long
// enough to return the exact selected size needed by the next stage.
[[nodiscard]] std::size_t
compactPointBatch(const PointBatch& source, PointBatch& destination,
                  std::span<const DimensionId> dimensions,
                  const std::uint8_t* keep);

} // namespace pdg
