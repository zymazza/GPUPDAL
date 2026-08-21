#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordinal.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pdg
{

namespace
{
struct BatchSelection
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t firstSequence = 0;
    std::uint64_t sequenceCount = 0;
};

void validate(const OrdinalProgram& program)
{
    if (program.kind == OrdinalKind::Decimation &&
        (!std::isfinite(program.step) || program.step < 1.0))
        throw std::invalid_argument(
            "filters.decimation step must be a finite value >= 1.0");
}

std::uint64_t checkedEnd(std::uint64_t begin, std::size_t size)
{
    if (size > std::numeric_limits<std::uint64_t>::max() - begin)
        throw std::overflow_error("ordinal point index overflow");
    return begin + static_cast<std::uint64_t>(size);
}

std::uint64_t candidate(const OrdinalProgram& program, std::uint64_t sequence)
{
    constexpr double MaximumRounded =
        static_cast<double>((std::numeric_limits<long long>::max)());
    const double product = static_cast<double>(sequence) * program.step;
    if (!std::isfinite(product) || product >= MaximumRounded)
        return (std::numeric_limits<std::uint64_t>::max)();
    const long long rounded = std::llround(product);
    if (rounded < 0)
        return (std::numeric_limits<std::uint64_t>::max)();
    const std::uint64_t distance = static_cast<std::uint64_t>(rounded);
    if (distance > (std::numeric_limits<std::uint64_t>::max)() - program.offset)
        return (std::numeric_limits<std::uint64_t>::max)();
    return program.offset + distance;
}

std::uint64_t firstSequenceAtOrAfter(const OrdinalProgram& program,
                                     std::uint64_t first,
                                     std::uint64_t sequenceLimit,
                                     std::uint64_t point)
{
    if (first >= sequenceLimit || candidate(program, first) >= point)
        return first;

    std::uint64_t high = sequenceLimit;
    if (point > program.offset)
    {
        const std::uint64_t distance = point - program.offset;
        // step >= 1 means candidate(k) >= offset + k, so no sequence at or
        // beyond distance can select a point below `point`.
        const std::uint64_t bounded =
            distance == (std::numeric_limits<std::uint64_t>::max)()
                ? distance
                : distance + 1U;
        high = std::min(high, bounded);
    }
    if (high <= first)
        return first;

    std::uint64_t low = first;
    while (low < high)
    {
        const std::uint64_t middle = low + (high - low) / 2U;
        if (candidate(program, middle) < point)
            low = middle + 1U;
        else
            high = middle;
    }
    return low;
}

BatchSelection advance(const OrdinalProgram& program, OrdinalState& state,
                       std::size_t size)
{
    BatchSelection selection;
    selection.begin = state.inputProcessed;
    selection.end = checkedEnd(selection.begin, size);
    selection.firstSequence = state.sequence;
    if (program.kind == OrdinalKind::Decimation)
    {
        const std::uint64_t pointLimit = std::min(selection.end, program.limit);
        const std::uint64_t sequenceEnd = firstSequenceAtOrAfter(
            program, state.sequence, state.sequenceLimit, pointLimit);
        selection.sequenceCount = sequenceEnd - state.sequence;
        state.sequence = sequenceEnd;
    }
    state.inputProcessed = selection.end;
    return selection;
}

void evaluateDecimationHost(const OrdinalProgram& program,
                            const BatchSelection& selection, std::uint8_t* keep,
                            std::size_t size)
{
    std::memset(keep, 0, size);
    for (std::uint64_t relative = 0; relative < selection.sequenceCount;
         ++relative)
    {
        const std::uint64_t point =
            candidate(program, selection.firstSequence + relative);
        if (point >= selection.begin && point < selection.end &&
            point < program.limit)
            keep[static_cast<std::size_t>(point - selection.begin)] = 1U;
    }
}
} // unnamed namespace

void evaluateOrdinalDevice(PointBatch& batch, const OrdinalProgram& program,
                           std::uint64_t begin, std::uint64_t inputTotal,
                           std::uint64_t firstSequence,
                           std::uint64_t sequenceCount, std::uint8_t* keep);

bool ordinalSupportsMode(const OrdinalProgram& program,
                         OrdinalMode mode) noexcept
{
    if (program.kind == OrdinalKind::Decimation &&
        (!std::isfinite(program.step) || program.step < 1.0))
        return false;
    return mode == OrdinalMode::Standard || program.kind != OrdinalKind::Tail;
}

std::uint64_t ordinalStandardOutputCount(const OrdinalProgram& program,
                                         std::uint64_t inputCount)
{
    validate(program);
    if (program.kind == OrdinalKind::Head || program.kind == OrdinalKind::Tail)
    {
        const std::uint64_t selected = std::min(program.count, inputCount);
        return program.invert ? inputCount - selected : selected;
    }

    const std::uint64_t last = std::min(program.limit, inputCount);
    if (program.offset > last)
        throw std::invalid_argument(
            "filters.decimation offset exceeds its standard-mode limit");
    const double quotient =
        static_cast<double>(last - program.offset) / program.step;
    constexpr double MaximumRounded =
        static_cast<double>((std::numeric_limits<long long>::max)());
    if (!std::isfinite(quotient) || quotient >= MaximumRounded)
        throw std::overflow_error(
            "filters.decimation output count exceeds the exact domain");
    const long long rounded = std::llround(quotient);
    if (rounded < 0)
        throw std::overflow_error(
            "filters.decimation returned a negative output count");
    return static_cast<std::uint64_t>(rounded);
}

OrdinalState makeOrdinalState(const OrdinalProgram& program, OrdinalMode mode,
                              std::uint64_t inputTotal)
{
    validate(program);
    if (!ordinalSupportsMode(program, mode))
        throw std::invalid_argument(
            "filters.tail does not support streaming execution");
    OrdinalState state;
    state.mode = mode;
    state.inputTotal = inputTotal;
    if (program.kind == OrdinalKind::Decimation &&
        mode == OrdinalMode::Standard)
        state.sequenceLimit = ordinalStandardOutputCount(program, inputTotal);
    return state;
}

void evaluateOrdinal(PointBatch& batch, const OrdinalProgram& program,
                     OrdinalState& state, std::uint8_t* keep)
{
    if (!keep && batch.size())
        throw std::invalid_argument("ordinal predicate is null");
    if (!ordinalSupportsMode(program, state.mode))
        throw std::invalid_argument(
            "ordinal program does not support the requested execution mode");
    const BatchSelection selection = advance(program, state, batch.size());
    if (!batch.size())
        return;
    if (batch.memoryKind() == MemoryKind::Device)
    {
        evaluateOrdinalDevice(batch, program, selection.begin, state.inputTotal,
                              selection.firstSequence, selection.sequenceCount,
                              keep);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported ordinal memory kind");

    if (program.kind == OrdinalKind::Decimation)
    {
        evaluateDecimationHost(program, selection, keep, batch.size());
        return;
    }

    const std::uint64_t selected = std::min(program.count, state.inputTotal);
    const std::uint64_t tailStart = state.inputTotal - selected;
    for (std::size_t local = 0; local < batch.size(); ++local)
    {
        const std::uint64_t point =
            selection.begin + static_cast<std::uint64_t>(local);
        bool retain = program.kind == OrdinalKind::Head ? point < program.count
                                                        : point >= tailStart;
        if (program.invert)
            retain = !retain;
        keep[local] = static_cast<std::uint8_t>(retain);
    }
}

} // namespace pdg
