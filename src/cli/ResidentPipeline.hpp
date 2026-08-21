#pragma once

#include <optional>

namespace pdg::cli
{

// Runs the explicit PDG-only resident execution command. Unlike `pipeline`,
// this command is outside the byte-compatible PDAL CLI namespace and may emit
// a requested placement/execution statistics document.
[[nodiscard]] int runResidentPipeline(int argc, char** argv);

// Attempts the one performance-qualified public pipeline envelope. A refusal
// has no output or diagnostic side effects and lets the caller continue with
// the existing direct/hybrid/oracle selection chain.
[[nodiscard]] std::optional<int>
tryAutomaticResidentLasPipeline(int argc, char** argv);

} // namespace pdg::cli
