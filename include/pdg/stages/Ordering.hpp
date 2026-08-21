#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pdg
{

class PointBatch;

enum class OrderingDirection : std::uint8_t
{
    Ascending,
    Descending
};

enum class OrderingAlgorithm : std::uint8_t
{
    Normal,
    Stable
};

struct OrderingProgram
{
    // PDAL applies these passes from first to last. For more than one
    // dimension the first pass is std::sort and every later pass is stable,
    // making the final listed dimension the primary key.
    std::vector<DimensionId> dimensions;
    OrderingDirection direction = OrderingDirection::Ascending;
    OrderingAlgorithm algorithm = OrderingAlgorithm::Normal;
};

struct OrderingResult
{
    // Normal single-key and multi-key CUDA sorting is exact only when the
    // final key has no comparator-equivalent ties. The device implementation
    // reports that data-dependent condition after sorting.
    bool exact = true;
};

// The bounded whole-view exact lane's plan already owns one binary64 key
// column.  Its stage-local reservation additionally covers the caller-owned
// permutation, the alternate permutation, two materialized binary64 key
// buffers, CUB radix-sort temporary storage, and the duplicate flag.  On the
// pinned CUDA 13.3 / SM 89 lane the requested-allocation high-water was
// 33,769,475 bytes at 600K and 900,275,715 bytes at 16M (56.283 and 56.267
// bytes/point including the planned key).  Reserve 64 bytes/point across the
// calibrated envelope rather than treating only the persistent columns as
// the peak.
inline constexpr std::size_t OrderingExactDeviceScratchBytesPerPoint = 56U;
inline constexpr std::size_t OrderingExactDevicePeakBytesPerPoint = 64U;

// Checks the static CUDA envelope. A successful check may still produce
// OrderingResult::exact == false when a normal/multi-key final dimension has
// ties; callers must then retain the host result.
[[nodiscard]] bool
orderingMaySupportExactDevice(const PointBatch& hostBatch,
                              const OrderingProgram& program) noexcept;

// Writes output-position -> input-position indices into permutation. The
// pointer belongs to the same memory domain as batch. Host execution exactly
// reproduces PDAL's std::sort/stable_sort pass sequence.
[[nodiscard]] OrderingResult orderPoints(PointBatch& batch,
                                         const OrderingProgram& program,
                                         std::uint64_t* permutation);

} // namespace pdg
