#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace pdg
{

class PointBatch;

enum class SummaryMode : std::uint8_t
{
    None,
    Enumerate,
    Count,
    Global
};

// The field order is part of the host/device ABI used by SummaryKernels.cu.
struct SummaryState
{
    std::uint64_t count = 0;
    double minimum = (std::numeric_limits<double>::max)();
    double maximum = (std::numeric_limits<double>::lowest)();
    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;
    double m4 = 0.0;
};

struct StatsProgram
{
    std::vector<DimensionId> dimensions;
    std::vector<SummaryMode> modes;
    bool advanced = false;
    std::string commonSrs = "EPSG:4326";
};

// These formulas deliberately mirror the pinned PDAL stats::Summary update
// order. Reassociation or parallel reduction changes exact output metadata.
void insertSummary(SummaryState& state, double value, bool advanced);

[[nodiscard]] double summaryAverage(const SummaryState& state) noexcept;
[[nodiscard]] double
summaryPopulationVariance(const SummaryState& state) noexcept;
[[nodiscard]] double summarySampleVariance(const SummaryState& state) noexcept;
[[nodiscard]] double
summaryPopulationStddev(const SummaryState& state) noexcept;
[[nodiscard]] double summarySampleStddev(const SummaryState& state) noexcept;
[[nodiscard]] double summaryPopulationSkewness(const SummaryState& state,
                                               bool advanced) noexcept;
[[nodiscard]] double summarySampleSkewness(const SummaryState& state,
                                           bool advanced) noexcept;
[[nodiscard]] double summaryPopulationKurtosis(const SummaryState& state,
                                               bool advanced) noexcept;
[[nodiscard]] double summarySampleKurtosis(const SummaryState& state,
                                           bool advanced) noexcept;
[[nodiscard]] double summarySampleExcessKurtosis(const SummaryState& state,
                                                 bool advanced) noexcept;

// Exact CUDA execution currently accepts finite, materialized Double columns
// and the non-advanced recurrence. Other inputs retain the exact host path.
[[nodiscard]] bool
summariesMaySupportExactDevice(const PointBatch& hostBatch,
                               std::span<const DimensionId> dimensions,
                               bool advanced) noexcept;

// Updates one persistent state per dimension. Point order within each
// dimension is never changed, including across repeated calls.
void updateSummaries(PointBatch& batch, std::span<const DimensionId> dimensions,
                     SummaryState* states, bool advanced);

} // namespace pdg
