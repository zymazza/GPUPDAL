#pragma once

#include <cstdint>

namespace pdg
{

// D0261: `gpupal --fast` arms the internal marker PDG_INTERNAL_FAST_MODE=1 for
// the engine and the sibling PDAL; nothing else may set it (the launcher
// strips an external value). This is the one place engine code reads it.
[[nodiscard]] bool fastModeEnabled() noexcept;

// D0271: under the fast contract kNN distance ties are not reported. The
// device (or host-index) tie choice stands for those rows — an equally valid
// neighbor set or accumulation order at identical distances — and no exact
// tie repair runs. Coordinates, record count, and order are unchanged; the
// default contract still reports every tie and repairs it bit-for-bit.
[[nodiscard]] bool relaxedTieOrder() noexcept;

// Mask applied to every kNN status byte the spatial index publishes: all
// bits under the default contract, all bits but KnnDistanceTie under the
// relaxed one.
[[nodiscard]] std::uint8_t knnStatusMask() noexcept;

} // namespace pdg
