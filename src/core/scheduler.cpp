#include <pdg/Scheduler.hpp>

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace pdg
{

std::size_t fixedLaneCount(PipelineClass pipelineClass) noexcept
{
    // These conservative initial values preserve the existing two-lane
    // behavior. D3 promotes a different value only after its class-specific
    // N=2..6 sweep is recorded in BENCHMARKS.md.
    switch (pipelineClass)
    {
    case PipelineClass::LasTranslation:
    case PipelineClass::FusedPointProgram:
    case PipelineClass::OrderedPointProgram:
    case PipelineClass::RadiusNeighborhood:
        return 2U;
    }
    return 2U;
}

std::size_t parseSchedulerLaneCount(std::string_view text)
{
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value < MinimumSweptLaneCount || value > MaximumSweptLaneCount)
        throw std::invalid_argument(
            "scheduler lane count must be an integer in [2, 6]");
    return value;
}

TiledSchedule makeTiledSchedule(const TiledScheduleRequest& request)
{
    if (!request.tileItems)
        throw std::invalid_argument("scheduler tile size is zero");
    if (request.requestedLanes &&
        (request.requestedLanes < MinimumSweptLaneCount ||
         request.requestedLanes > MaximumSweptLaneCount))
        throw std::invalid_argument(
            "scheduler lane count must be zero or in [2, 6]");

    TiledSchedule result;
    result.itemCount = request.itemCount;
    result.tileItemCapacity = request.tileItems;
    result.configuredLaneCount = request.requestedLanes
                                     ? request.requestedLanes
                                     : fixedLaneCount(request.pipelineClass);
    result.serialDependency = request.serialDependency;
    if (!request.itemCount)
        return result;
    result.tileCount =
        request.itemCount / request.tileItems +
        static_cast<std::size_t>(request.itemCount % request.tileItems != 0U);
    result.activeLaneCount =
        request.serialDependency || result.tileCount == 1U
            ? 1U
            : (std::min)(result.configuredLaneCount, result.tileCount);

    if (request.bytesPerLane && request.memoryBudgetBytes)
    {
        const std::size_t affordable =
            request.memoryBudgetBytes / request.bytesPerLane;
        if (!affordable)
            throw std::runtime_error(
                "scheduler VRAM budget cannot hold one tile lane");
        if (affordable < result.activeLaneCount)
        {
            result.activeLaneCount = affordable;
            result.memoryLimited = true;
        }
    }
    if (request.bytesPerLane &&
        result.activeLaneCount >
            (std::numeric_limits<std::size_t>::max)() / request.bytesPerLane)
        throw std::overflow_error("scheduler peak lane bytes overflow");
    result.peakLaneBytes = result.activeLaneCount * request.bytesPerLane;
    result.laneReuseCount = result.tileCount > result.activeLaneCount
                                ? result.tileCount - result.activeLaneCount
                                : 0U;
    return result;
}

} // namespace pdg
