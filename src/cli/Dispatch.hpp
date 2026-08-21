#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace pdg::cli
{

// The launcher uses this deliberately conservative distinction before either
// PDAL or CUDA code is loaded. Engine means that the existing implementation
// remains responsible for selection, validation, diagnostics, and execution.
enum class DispatchRoute
{
    Oracle,
    Engine
};

struct DispatchInputFacts
{
    std::uint64_t pointCount{};
    bool measuredReferenceLayout{};
    bool measuredR14ReferenceLayout{};
};

// Returns Engine for malformed or otherwise ambiguous pipeline JSON. A parsed
// pipeline is sent directly to the oracle only when no measured engine route
// can improve it. Direct-delegation shapes are literal and conservative.
[[nodiscard]] DispatchRoute classifyPipelineForDispatch(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts = std::nullopt) noexcept;

// Returns true only for literal measured direct-delegation routes calibrated
// for `gpupdal pipeline FILE` without stream, metadata, or other CLI modifiers.
[[nodiscard]] bool
dispatchRequiresPlainPipelineInvocation(std::string_view pipelineJson) noexcept;

// Returns the LAS-family reader filename only for a literal route whose
// dispatch decision needs a fixed-header point-count fact.
[[nodiscard]] std::optional<std::string>
dispatchPointCountProbeFilename(std::string_view pipelineJson) noexcept;

// True only for the B0239 complete r2 workflow and its fixed measured input
// facts. The thin launcher uses this to arm the internal hybrid selector
// before replacing itself with the engine.
[[nodiscard]] bool dispatchUsesAutomaticR2Hybrid(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts = std::nullopt) noexcept;

// True only for the measured r7 grammar and compressed format-7 reference
// facts. The launcher appends the exact four-worker readers.las option before
// replacing itself with sibling PDAL.
[[nodiscard]] bool dispatchUsesAutomaticR7ReaderThreads(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts = std::nullopt) noexcept;

// True only for the measured r14 LAS -> LAZ grammar and its fixed
// uncompressed format-7 reference facts. The launcher uses this to arm the
// exact two-worker lazperf path before replacing itself with sibling PDAL.
[[nodiscard]] bool dispatchUsesAutomaticR14ParallelCompression(
    std::string_view pipelineJson,
    std::optional<DispatchInputFacts> inputFacts = std::nullopt) noexcept;

// Testable fail-closed policy for environment variables which can alter PDG
// dispatch, diagnostics, or native execution. Every current or future PDG_*
// name requires the engine. PDG_ORACLE_PDAL is the sole exception because it
// identifies the same oracle for either route.
[[nodiscard]] bool dispatchEnvironmentRequiresEngine(
    std::span<const std::string_view> presentVariables) noexcept;

[[nodiscard]] std::span<const std::string_view>
dispatchEngineEnvironmentVariables() noexcept;

// D0261 fast mode. The public launcher consumes a leading `--fast` and arms
// this internal marker; an externally supplied marker is removed before the
// engine route so it cannot relax the default exact contract from outside.
inline constexpr std::string_view FastModeFlag = "--fast";
inline constexpr std::string_view FastModeMarker = "PDG_INTERNAL_FAST_MODE";

// True inside the engine or sibling when the launcher armed fast mode.
[[nodiscard]] bool fastModeEnabled() noexcept;

} // namespace pdg::cli
