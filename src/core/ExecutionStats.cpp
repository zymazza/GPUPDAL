#include <pdg/ExecutionStats.hpp>

#include <limits>

namespace pdg
{
namespace
{

thread_local ExecutionObservationScope* ActiveScope = nullptr;

void saturatingAdd(std::size_t& target, std::size_t value) noexcept
{
    const std::size_t maximum = (std::numeric_limits<std::size_t>::max)();
    target = value > maximum - target ? maximum : target + value;
}

} // namespace

ExecutionObservationScope::ExecutionObservationScope() noexcept
    : previous_(ActiveScope)
{
    ActiveScope = this;
}

ExecutionObservationScope::~ExecutionObservationScope() noexcept
{
    // Scopes are non-movable and normal C++ lifetime nesting is LIFO.  Restore
    // the exact enclosing observer so an inner diagnostic cannot consume the
    // outer execution's events.
    ActiveScope = previous_;
}

bool ExecutionObservationScope::active() noexcept
{
    return ActiveScope != nullptr;
}

void ExecutionObservationScope::record(ExecutionEventKind kind,
                                       std::size_t regionId, std::size_t bytes,
                                       std::size_t packingBytes) noexcept
{
    if (ActiveScope)
        ActiveScope->recordEvent(kind, regionId, bytes, packingBytes);
}

ExecutionStatsSnapshot ExecutionObservationScope::snapshot() const
{
    return {.events = events_, .totals = totals_};
}

void ExecutionObservationScope::recordEvent(ExecutionEventKind kind,
                                            std::size_t regionId,
                                            std::size_t bytes,
                                            std::size_t packingBytes) noexcept
{
    const std::size_t index = static_cast<std::size_t>(kind);
    if (index >= ExecutionEventKindCount)
        return;

    try
    {
        events_.push_back({.kind = kind,
                           .regionId = regionId,
                           .bytes = bytes,
                           .packingBytes = packingBytes});
    }
    catch (...)
    {
        return;
    }

    saturatingAdd(totals_[index].count, 1U);
    saturatingAdd(totals_[index].bytes, bytes);
    saturatingAdd(totals_[index].packingBytes, packingBytes);
}

} // namespace pdg
