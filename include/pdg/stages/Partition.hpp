#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace pdg
{

class PointBatch;

inline constexpr std::uint8_t ReturnFirst = 1U;
inline constexpr std::uint8_t ReturnIntermediate = 2U;
inline constexpr std::uint8_t ReturnLast = 4U;
inline constexpr std::uint8_t ReturnOnly = 8U;
inline constexpr std::uint8_t AllReturnGroups =
    ReturnFirst | ReturnIntermediate | ReturnLast | ReturnOnly;
inline constexpr std::uint8_t UnselectedReturnGroup = 4U;

struct ReturnsProgram
{
    std::uint8_t groups = ReturnLast;
};

struct ReturnsPartitionResult
{
    // Counts are always ordered first, intermediate, last, only.
    std::array<std::uint64_t, 4> counts{};

    [[nodiscard]] std::uint64_t selectedCount() const noexcept;
};

enum class DividerMode : std::uint8_t
{
    Partition,
    RoundRobin
};

struct DividerProgram
{
    DividerMode mode = DividerMode::Partition;
    std::uint32_t count = 2U;
};

struct DividerPartitionResult
{
    // Counts are ordered by output-view creation order.
    std::vector<std::uint64_t> counts;

    [[nodiscard]] std::uint64_t selectedCount() const noexcept;
};

struct SplitterProgram
{
    double length = 1000.0;
    double originX = (std::numeric_limits<double>::quiet_NaN)();
    double originY = (std::numeric_limits<double>::quiet_NaN)();
    double buffer = 0.0;
};

// Returns a fixed output-group index in [0, 3], or
// UnselectedReturnGroup when the point isn't selected.
[[nodiscard]] std::uint8_t returnGroupIndex(std::uint8_t returnNumber,
                                            std::uint8_t numberOfReturns,
                                            std::uint8_t groups) noexcept;

[[nodiscard]] bool
returnsMaySupportExactDevice(const PointBatch& hostBatch,
                             const ReturnsProgram& program) noexcept;

// Writes a stable output-position -> input-position permutation. Only the
// first selectedCount() entries are selected; their group boundaries are the
// cumulative counts in the result.
[[nodiscard]] ReturnsPartitionResult
partitionReturns(PointBatch& batch, const ReturnsProgram& program,
                 std::uint64_t* permutation);

[[nodiscard]] bool
dividerMaySupportExactDevice(const PointBatch& hostBatch,
                             const DividerProgram& program) noexcept;

// Writes a stable output-position -> input-position permutation. Empty output
// views remain represented by zero counts.
[[nodiscard]] DividerPartitionResult
partitionDivider(PointBatch& batch, const DividerProgram& program,
                 std::uint64_t* permutation);

[[nodiscard]] bool
splitterCellsMaySupportExactDevice(const PointBatch& hostBatch,
                                   const SplitterProgram& program) noexcept;

// Computes the primary splitter cell for every point. Buffered secondary
// membership is intentionally constructed in source order by the PDAL stage.
void computeSplitterCells(PointBatch& batch, const SplitterProgram& program,
                          std::int32_t* xCells, std::int32_t* yCells);

} // namespace pdg
