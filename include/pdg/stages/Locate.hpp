#pragma once

#include <pdg/Dimension.hpp>

#include <cstdint>

namespace pdg
{

class PointBatch;

enum class LocateKind : std::uint8_t
{
    Minimum,
    Maximum,
    None
};

struct LocateProgram
{
    DimensionId dimension;
    LocateKind kind = LocateKind::Maximum;
};

// `comparable` distinguishes a real extremum from PDAL's initial sentinel.
// This is required for exact behavior when a batch contains only NaNs,
// -infinity for maximum, +infinity for minimum, or the sentinel itself.
struct LocateResult
{
    double value = 0.0;
    std::uint64_t index = 0;
    // 32-bit flags keep the host/device transfer representation free of
    // uninitialized tail padding (important under Compute Sanitizer).
    std::uint32_t hasPoints = 0;
    std::uint32_t comparable = 0;
};

[[nodiscard]] LocateResult locateExtreme(PointBatch& batch,
                                         const LocateProgram& program,
                                         std::uint64_t indexOffset = 0);

// Merges two results whose input ranges are in increasing index order. Ties
// retain the first point, matching LocateFilter's strict comparison loop.
[[nodiscard]] LocateResult mergeLocateResults(const LocateProgram& program,
                                              const LocateResult& first,
                                              const LocateResult& second);

} // namespace pdg
