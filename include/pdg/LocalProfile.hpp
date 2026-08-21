#pragma once

// D0277: locally calibrated placement profiles.
//
// The embedded SM-89 placement profile is a performance promise measured on
// one physical machine.  Every other machine fails closed to the host path
// unless the maintainer runs the explicit `gpupal calibrate` command, which
// re-measures the placement calibration cases on that machine and writes a
// profile file.  This module owns that file: its schema, its machine key, and
// the once-per-process lookup consulted by `placementCalibrationFor` when the
// embedded profile does not match.  It never fits or measures anything itself.

#include <pdg/Placement.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pdg
{

// The complete identity a local profile is keyed to.  Device identity uses
// the same four fields as PlacementDeviceKey; the host fields are part of the
// key because every stage model compares device work with host work measured
// on this CPU.  A profile whose key differs in any field is ignored.
struct LocalProfileMachineKey
{
    std::string deviceName;
    std::string computeCapability;
    std::string driverVersion;
    std::string cudaToolkitVersion;
    std::string cpuModel;
    unsigned logicalCpus = 0;
    std::string pdgVersion;

    [[nodiscard]] bool operator==(const LocalProfileMachineKey&) const =
        default;
};

// One row of the human-readable per-model summary carried by the profile so
// `gpupal calibrate --status` can explain the file without re-deriving
// anything.
struct LocalProfileModelSummary
{
    std::string name;
    std::size_t minimumDevicePoints = 0;
    std::size_t maximumDevicePoints = 0;
    // Median measured complete-process host/device seconds at the largest
    // measured size, and whether every measured pair was byte-identical.
    double hostSecondsAtMaximum = 0.0;
    double deviceSecondsAtMaximum = 0.0;
    bool byteExact = false;
    bool deviceWinsSomewhere = false;
};

struct LocalPlacementProfile
{
    std::string id;
    std::string createdUtc;
    std::string sourcePath;
    // "local" (measured on this exact machine, the default), "shipped" (a
    // GPU-class profile measured on a rented host and embedded in the build,
    // D0279), or "generic" (the conservative fallback for any other CUDA
    // device, D0279). Only "local" profiles are keyed on the host fields.
    std::string tier = "local";
    // Generic-tier applicability: the smallest compute capability and device
    // memory the profile was measured on (empty/zero = no bound).
    std::string minimumComputeCapability;
    std::uint64_t minimumDeviceMemoryBytes = 0;
    LocalProfileMachineKey machine;
    PlacementModelCoefficients coefficients;
    // Storage for the string_view names inside stageModels.
    std::vector<std::string> modelNames;
    std::vector<PlacementStageCalibration> stageModels;
    std::vector<LocalProfileModelSummary> summaries;

    // A borrowed view for the planner.  Valid only while this object is
    // alive and unmodified.
    [[nodiscard]] PlacementCalibrationProfile view() const noexcept;
};

inline constexpr std::string_view LocalProfileSchema =
    "pdg-local-placement-profile-v1";

// Environment controls (all routed to the engine like every PDG_* name):
//   PDG_PROFILE_PATH            explicit profile file (read and written)
//   PDG_DISABLE_LOCAL_PROFILE   never load a local profile
//   PDG_TEST_IGNORE_BUILTIN_PLACEMENT_PROFILE
//                               test hook: the embedded SM-89 profile is
//                               treated as absent so the local path can be
//                               exercised on the reference machine
inline constexpr std::string_view LocalProfilePathEnvironment =
    "PDG_PROFILE_PATH";
inline constexpr std::string_view LocalProfileDisableEnvironment =
    "PDG_DISABLE_LOCAL_PROFILE";
inline constexpr std::string_view IgnoreBuiltinProfileTestEnvironment =
    "PDG_TEST_IGNORE_BUILTIN_PLACEMENT_PROFILE";
// D0279: never consult the shipped GPU-class or generic tiers (a user knob to
// force "measured on this machine only", and what the tests use to obtain a
// clean host baseline).
inline constexpr std::string_view DisableShippedProfilesEnvironment =
    "PDG_DISABLE_SHIPPED_PROFILES";
// Test hooks: load additional shipped/generic profiles from a directory, and
// optionally use only those (ignoring the embedded table).
inline constexpr std::string_view ShippedProfileDirTestEnvironment =
    "PDG_TEST_SHIPPED_PROFILE_DIR";
inline constexpr std::string_view ShippedProfileDirOnlyTestEnvironment =
    "PDG_TEST_SHIPPED_PROFILE_DIR_ONLY";
// Calibration-only override honoured by the explicit `resident` command: the
// planner sees a profile that prices every calibrated stage as a device win
// so `gpupal calibrate` can time the device executor before any local profile
// exists.  The automatic `pipeline` route declines whenever it is set.
inline constexpr std::string_view CalibrationForceDeviceEnvironment =
    "PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT";

// ${PDG_PROFILE_PATH} or
// ${XDG_CONFIG_HOME:-$HOME/.config}/gpupal/placement-profile.json
[[nodiscard]] std::filesystem::path defaultLocalProfilePath();

// Reads the CPU model name from /proc/cpuinfo (empty when unavailable).
[[nodiscard]] std::string hostCpuModelName();

// The current machine key.  Throws when no CUDA device is usable.
[[nodiscard]] LocalProfileMachineKey currentLocalProfileMachineKey();

// Parses a profile document.  On failure returns nullopt and sets `error`.
[[nodiscard]] std::optional<LocalPlacementProfile>
parseLocalProfile(std::string_view json, std::string& error);

// Serialises a profile in the schema parseLocalProfile accepts.  `evidence`
// is an optional JSON object text appended verbatim under "evidence" (the
// calibrate command stores its raw measurements there); it is not parsed
// back and does not affect placement.
[[nodiscard]] std::string
serializeLocalProfile(const LocalPlacementProfile& profile,
                      std::string_view evidenceJson = {});

enum class LocalProfileStatus
{
    NotAttempted, // CUDA backend not compiled or no device
    Disabled,     // PDG_DISABLE_LOCAL_PROFILE
    NotFound,     // no file at the resolved path
    Malformed,    // file exists but does not parse
    MachineMismatch,
    Applied
};

struct LocalProfileLookup
{
    LocalProfileStatus status = LocalProfileStatus::NotAttempted;
    std::filesystem::path path;
    std::string detail;
    // Present when status is Applied or MachineMismatch (so --status can show
    // what the file was keyed to).
    std::optional<LocalPlacementProfile> profile;
    std::optional<LocalProfileMachineKey> currentMachine;
};

// Loads the local profile once per process (thread-safe) and returns the
// cached result.  Never throws.
[[nodiscard]] const LocalProfileLookup& loadedLocalProfile() noexcept;

// Same as loadedLocalProfile() but performs the lookup now, without caching;
// used by `gpupal calibrate --status` and tests.
[[nodiscard]] LocalProfileLookup lookupLocalProfile() noexcept;

[[nodiscard]] std::string_view
localProfileStatusName(LocalProfileStatus status) noexcept;

// The calibration override profile (see CalibrationForceDeviceEnvironment):
// present only when the environment is set.  The returned pointer stays
// valid for the process lifetime.
[[nodiscard]] const PlacementCalibrationProfile*
calibrationForcedDevicePlacementProfile(
    const PlacementDeviceKey& device) noexcept;

// The embedded stage-model names, so the forced profile and the calibrate
// command can enumerate them without duplicating the table.
[[nodiscard]] std::vector<std::string_view> embeddedPlacementModelNames();

// D0279: profiles embedded in the build from data/placement-profiles/*.json.
struct ShippedProfileSource
{
    std::string_view name;
    std::string_view sha256;
    std::string_view json;
};
[[nodiscard]] std::span<const ShippedProfileSource> shippedProfileSources();

// The shipped GPU-class profile for this device (name, compute capability,
// compiled CUDA toolkit; not driver or host), or nullptr. Parsed once per
// process; the returned pointer stays valid for the process lifetime.
[[nodiscard]] const PlacementCalibrationProfile*
shippedPlacementProfileFor(const PlacementDeviceKey& device) noexcept;
// The generic fallback for any CUDA device inside its applicability bounds
// (compute capability and device memory of CUDA ordinal zero), or nullptr.
[[nodiscard]] const PlacementCalibrationProfile*
genericPlacementProfileFor(const PlacementDeviceKey& device) noexcept;
// Which tier an active profile came from: "embedded", "local", "shipped",
// "generic", or "" for none / the calibration override.
[[nodiscard]] std::string_view
placementProfileTier(const PlacementCalibrationProfile* profile) noexcept;
// Every parsed shipped/generic profile (for `gpupal calibrate --status`).
[[nodiscard]] const std::vector<LocalPlacementProfile>& shippedProfiles();

} // namespace pdg
