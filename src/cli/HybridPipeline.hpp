#pragma once

#include <optional>

namespace pdg::cli
{

// Returns no value when the command is outside the conservative hybrid
// envelope. A returned status is final: execution occurred in-process.
[[nodiscard]] std::optional<int> tryHybridPipeline(int argc, char** argv);

} // namespace pdg::cli
