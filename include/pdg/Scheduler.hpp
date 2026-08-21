#pragma once

#include <cstddef>
#include <string_view>

namespace pdg
{

enum class PipelineClass
{
    LasTranslation,
    FusedPointProgram,
    OrderedPointProgram,
    RadiusNeighborhood
};

inline constexpr std::size_t MinimumSweptLaneCount = 2U;
inline constexpr std::size_t MaximumSweptLaneCount = 6U;

// Zero requests the benchmark-fixed default for the pipeline class. A
// nonzero lane count is an explicit test/benchmark choice and must remain in
// the bounded sweep [2, 6]. memoryBudgetBytes is planner-owned; zero means the
// caller has not yet supplied a D4 budget.
struct TiledScheduleRequest
{
    PipelineClass pipelineClass = PipelineClass::LasTranslation;
    std::size_t itemCount = 0;
    std::size_t tileItems = 0;
    std::size_t bytesPerLane = 0;
    std::size_t memoryBudgetBytes = 0;
    std::size_t requestedLanes = 0;
    bool serialDependency = false;
};

struct TiledSchedule
{
    std::size_t itemCount = 0;
    std::size_t tileItemCapacity = 0;
    std::size_t tileCount = 0;
    std::size_t configuredLaneCount = 0;
    std::size_t activeLaneCount = 0;
    std::size_t laneReuseCount = 0;
    std::size_t peakLaneBytes = 0;
    // Runtime executors may populate this with the greatest aggregate current
    // device allocation sampled across their active lanes at planned
    // materialize/release transitions. The pure scheduler leaves it at zero.
    std::size_t observedPeakLaneBytes = 0;
    bool memoryLimited = false;
    bool serialDependency = false;
};

// Defaults are constants backed by an append-only same-machine sweep. This is
// deliberately not an autotuner and never probes the GPU or times work.
[[nodiscard]] std::size_t fixedLaneCount(PipelineClass pipelineClass) noexcept;

// Absence is represented by the caller as zero. A present benchmark/test
// override must be one of the widths covered by the mandatory D3 sweep.
[[nodiscard]] std::size_t parseSchedulerLaneCount(std::string_view text);

[[nodiscard]] TiledSchedule
makeTiledSchedule(const TiledScheduleRequest& request);

} // namespace pdg
