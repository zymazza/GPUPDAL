#pragma once

#include <cstddef>
#include <cstdint>

namespace pdg
{

class PointBatch;

enum class OrdinalKind : std::uint8_t
{
    Decimation,
    Head,
    Tail
};

enum class OrdinalMode : std::uint8_t
{
    Standard,
    Streaming
};

// A dimension-independent, stable point-selection program.  The fields map
// directly to PDAL's filters.decimation, filters.head, and filters.tail
// options.  Unused fields retain their PDAL defaults.
struct OrdinalProgram
{
    OrdinalKind kind = OrdinalKind::Head;
    double step = 1.0;
    std::uint64_t offset = 0;
    std::uint64_t limit = UINT64_MAX;
    std::uint64_t count = 10;
    bool invert = false;
};

// Mutable state is deliberately separate from the immutable program.  One
// state object is retained per operation so chunk boundaries cannot change
// the selected point sequence.
struct OrdinalState
{
    OrdinalMode mode = OrdinalMode::Standard;
    std::uint64_t inputProcessed = 0;
    std::uint64_t inputTotal = 0;
    std::uint64_t sequence = 0;
    std::uint64_t sequenceLimit = UINT64_MAX;
};

[[nodiscard]] bool ordinalSupportsMode(const OrdinalProgram& program,
                                       OrdinalMode mode) noexcept;

// Returns the exact number of points PDAL's standard-mode implementation
// emits.  It rejects the upstream decimation underflow domain so callers can
// delegate that pipeline before executing it.
[[nodiscard]] std::uint64_t
ordinalStandardOutputCount(const OrdinalProgram& program,
                           std::uint64_t inputCount);

[[nodiscard]] OrdinalState makeOrdinalState(const OrdinalProgram& program,
                                            OrdinalMode mode,
                                            std::uint64_t inputTotal = 0);

// Evaluates one consecutive batch and advances state.  Host and device masks
// use identical global ordinal semantics; compaction remains stable.
void evaluateOrdinal(PointBatch& batch, const OrdinalProgram& program,
                     OrdinalState& state, std::uint8_t* keep);

} // namespace pdg
