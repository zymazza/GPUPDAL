#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pdg
{

// These are observed execution facts, not placement predictions.  The region
// identifier is planner-owned so an eventual --stats surface can compare the
// selected plan with its actual residency boundaries without inferring either
// one from stage names.
enum class ExecutionEventKind : std::uint8_t
{
    DeviceRegionBegin,
    DeviceRegionEnd,
    HostToDevice,
    DeviceToHost,
    // For these two kinds regionId carries the stable index into
    // PlanSummary::residencyBoundaries. Transfer events continue to carry the
    // planner-owned resident region.
    BoundaryUpload,
    BoundarySpill,
    FallbackBoundary,
    IndexBuild,
    IndexRebuild,
    GridBuild,
    RasterBuild,
    // Aggregate surface transfers performed inside one Grid execution. They
    // are distinct from the point-column residency boundary crossings above.
    RasterUpload,
    RasterDownload,
    Count
};

inline constexpr std::size_t ExecutionEventKindCount =
    static_cast<std::size_t>(ExecutionEventKind::Count);

struct ExecutionEvent
{
    ExecutionEventKind kind = ExecutionEventKind::DeviceRegionBegin;
    std::size_t regionId = 0;
    std::size_t bytes = 0;
    // Bytes processed by an explicit pack/repack operation at the execution
    // site. This is not inferred from transfer volume.
    std::size_t packingBytes = 0;
};

struct ExecutionEventTotals
{
    std::size_t count = 0;
    std::size_t bytes = 0;
    std::size_t packingBytes = 0;
};

// The event list preserves the caller's execution order.  Totals are indexed
// by ExecutionEventKind and are deterministic for a fixed event sequence.
struct ExecutionStatsSnapshot
{
    std::vector<ExecutionEvent> events;
    std::array<ExecutionEventTotals, ExecutionEventKindCount> totals{};
};

// An opt-in observer for one execution thread.  It is intentionally inactive
// unless a scope exists, and record() is noexcept: instrumentation must never
// alter compatibility execution, including if diagnostics run out of memory.
// Nested scopes are independent and restore the enclosing scope when they end.
class ExecutionObservationScope
{
public:
    ExecutionObservationScope() noexcept;
    ~ExecutionObservationScope() noexcept;

    ExecutionObservationScope(const ExecutionObservationScope&) = delete;
    ExecutionObservationScope&
    operator=(const ExecutionObservationScope&) = delete;
    ExecutionObservationScope(ExecutionObservationScope&&) = delete;
    ExecutionObservationScope& operator=(ExecutionObservationScope&&) = delete;

    [[nodiscard]] static bool active() noexcept;
    static void record(ExecutionEventKind kind, std::size_t regionId,
                       std::size_t bytes = 0,
                       std::size_t packingBytes = 0) noexcept;

    [[nodiscard]] ExecutionStatsSnapshot snapshot() const;

private:
    void recordEvent(ExecutionEventKind kind, std::size_t regionId,
                     std::size_t bytes, std::size_t packingBytes) noexcept;

    ExecutionObservationScope* previous_ = nullptr;
    std::vector<ExecutionEvent> events_;
    std::array<ExecutionEventTotals, ExecutionEventKindCount> totals_{};
};

} // namespace pdg
