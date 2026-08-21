#include "Calibrate.hpp"

#include <pdg/CalibrationProbes.hpp>
#include <pdg/LocalProfile.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Placement.hpp>
#include <pdg/PlacementCalibration.hpp>
#include <pdg/Plan.hpp>
#include <pdg/SyntheticCloud.hpp>
#include <pdg/Version.hpp>

#include <nlohmann/json.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace pdg::cli
{
namespace
{
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// The calibration plan: the placement models `gpupal calibrate` measures and
// the pipelines it runs for them.  Pipelines are verbatim copies of the
// corresponding cases in test/data/pdg/placement-calibration-sm89.json (the
// embedded profile's evidence record); "input.las"/"output.las" are replaced
// by the fixture paths.  Every model here is cardinality-preserving and runs
// as one resident region through the shared neighborhood engine or the fused
// point program, which is what the automatic `pipeline` route can select on
// a calibrated machine.  Direct whole-view executors (sort/skewness/HAG/
// radius/outlier direct compositions) need executor-specific proof flags and
// stay uncalibrated (host) in this version.
constexpr std::string_view EmbeddedCalibrationPlan = R"json({
  "pipelines": {
    "lof": ["input.las", {"type": "filters.lof", "minpts": 10}, "output.las"],
    "normal": ["input.las", {"type": "filters.normal", "knn": 8}, "output.las"],
    "eigenvalues": ["input.las", {"type": "filters.eigenvalues", "knn": 8}, "output.las"],
    "covariancefeatures": ["input.las", {"type": "filters.covariancefeatures", "knn": 8}, "output.las"],
    "nndistance": ["input.las", {"type": "filters.nndistance", "k": 10}, "output.las"],
    "estimaterank": ["input.las", {"type": "filters.estimaterank", "knn": 8}, {"type": "filters.assign", "value": "UserData = 1 WHERE Rank >= 3"}, "output.las"],
    "optimalneighborhood": ["input.las", {"type": "filters.optimalneighborhood", "min_k": 10, "max_k": 14}, {"type": "filters.assign", "value": "UserData = 1 WHERE OptimalKNN >= 12"}, "output.las"],
    "neighborclassifier": ["input.las", {"type": "filters.neighborclassifier", "k": 7}, "output.las"],
    "approximatecoplanar": ["input.las", {"type": "filters.approximatecoplanar", "knn": 8}, "output.las"],
    "radiusassign": ["input.las", {"type": "filters.radiusassign", "radius": 2.0, "src_domain": "ReturnNumber[1:1]", "reference_domain": "ReturnNumber[2:15]", "is3d": true, "update_expression": "UserData = 9"}, "output.las"],
    "normal-covariancefeatures-compose": ["input.las", {"type": "filters.normal", "knn": 8}, {"type": "filters.covariancefeatures", "knn": 8, "feature_set": "Dimensionality"}, "output.las"],
    "normal-covariancefeatures-compose-extradims": ["input.las", {"type": "filters.normal", "knn": 8}, {"type": "filters.covariancefeatures", "knn": 8, "feature_set": "Dimensionality"}, {"type": "writers.las", "filename": "output.las", "extra_dims": "all"}],
    "eigen-family-compose": ["input.las", {"type": "filters.normal", "knn": 12, "always_up": false}, {"type": "filters.eigenvalues", "knn": 12, "normalize": true}, {"type": "filters.covariancefeatures", "knn": 12, "mode": "raw", "feature_set": "dimensionality"}, {"type": "filters.assign", "value": ["Classification = Linearity * 10", "Intensity = Curvature * 1000", "UserData = Eigenvalue0 * 100"]}, "output.las"],
    "rank-optimal-compose": ["input.las", {"type": "filters.estimaterank", "knn": 14, "thresh": 0.01}, {"type": "filters.optimalneighborhood", "min_k": 10, "max_k": 14}, {"type": "filters.assign", "value": ["Classification = Rank", "Intensity = OptimalKNN", "PointSourceId = OptimalRadius"]}, "output.las"],
    "fused": ["input.las", {"type": "filters.assign", "value": ["Scratch = Intensity * 2 - 1", "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]}, {"type": "filters.ferry", "dimensions": "Classification=>UserData"}, {"type": "filters.assign", "value": ["PointSourceId = Scratch / 2 WHERE Scratch <= 131070", "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]}, "output.las"],
    "ferry": ["input.las", {"type": "filters.ferry", "dimensions": "Intensity=>PointSourceId"}, "output.las"]
  },
  "models": [
    {"model": "normal-covariancefeatures-compose", "pipeline": "normal-covariancefeatures-compose"},
    {"model": "normal-covariancefeatures-compose-extradims", "pipeline": "normal-covariancefeatures-compose-extradims"},
    {"model": "covariancefeatures", "pipeline": "covariancefeatures"},
    {"model": "normal", "pipeline": "normal"},
    {"model": "eigenvalues", "pipeline": "eigenvalues"},
    {"model": "eigen-family-compose", "pipeline": "eigen-family-compose"},
    {"model": "lof", "pipeline": "lof"},
    {"model": "nndistance", "pipeline": "nndistance"},
    {"model": "estimaterank", "pipeline": "estimaterank"},
    {"model": "optimalneighborhood", "pipeline": "optimalneighborhood"},
    {"model": "rank-optimal-compose", "pipeline": "rank-optimal-compose"},
    {"model": "neighborclassifier", "pipeline": "neighborclassifier"},
    {"model": "approximatecoplanar", "pipeline": "approximatecoplanar"},
    {"model": "radiusassign", "pipeline": "radiusassign"},
    {"model": "fused-point-program", "pipeline": "fused"},
    {"model": "simple-ferry", "pipeline": "ferry"}
  ]
})json";

// The reference profile's LAS packing coefficient (Nsight-derived, D0049
// evidence).  Packing has no independent probe in this version; the residual
// fit absorbs the difference for every measured shape, and the profile marks
// the value as inherited rather than measured.
constexpr double ReferencePackingNanosecondsPerByte = 0.002759959539676;

struct Options
{
    bool status = false;
    bool dryRun = false;
    bool force = false;
    bool quick = false;
    bool keepWork = false;
    bool quiet = false;
    bool append = false;
    std::optional<std::string> probe;
    std::filesystem::path output;
    std::filesystem::path input;
    std::filesystem::path work;
    std::vector<std::size_t> points;
    std::vector<std::string> models;
    int repeats = 3;
};

void printUsage(std::ostream& out)
{
    out << "Usage: gpupal calibrate [options]\n"
           "\n"
           "Measures the placement calibration cases on this machine and "
           "writes a local\n"
           "placement profile so the automatic pipeline route can select "
           "the GPU where it\n"
           "measured a win.  Nothing runs this implicitly.\n"
           "\n"
           "  --status              show the machine key and profile status; "
           "do not measure\n"
           "  --output PATH         profile file (default: $PDG_PROFILE_PATH "
           "or\n"
           "                        ${XDG_CONFIG_HOME:-~/.config}/pdg/"
           "placement-profile.json)\n"
           "  --input FILE          calibrate on this LAS/LAZ file (subsets "
           "are taken from\n"
           "                        its head); default: a synthetic terrain "
           "cloud\n"
           "  --points N[,N...]     input sizes (default 250000,1000000,"
           "4000000)\n"
           "  --repeats N           timed host/device pairs per size "
           "(default 3, median)\n"
           "  --models A[,B...]     restrict to these placement models\n"
           "  --quick               250000,1000000 points, one pair each\n"
           "  --work DIR            work directory (default: a temporary "
           "directory)\n"
           "  --keep-work           keep the work directory\n"
           "  --append              keep the models of an existing profile for "
           "this machine\n"
           "                        (and its coefficients); measure only the "
           "selected models\n"
           "  --dry-run             print the plan and exit\n"
           "  --force               measure even when the embedded reference "
           "profile applies\n"
           "  --quiet               no progress lines\n";
}

std::vector<std::string> splitList(std::string_view text)
{
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t comma = text.find(',', start);
        const std::string_view item =
            text.substr(start, comma == std::string_view::npos
                                   ? std::string_view::npos
                                   : comma - start);
        if (!item.empty())
            result.emplace_back(item);
        if (comma == std::string_view::npos)
            break;
        start = comma + 1;
    }
    return result;
}

std::size_t parseCount(std::string_view text, const char* what)
{
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0)
        throw std::invalid_argument(std::string("invalid ") + what + ": " +
                                    std::string(text));
    return value;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 2; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> std::string_view
        {
            if (index + 1 >= argc)
                throw std::invalid_argument(std::string(name) +
                                            " needs a value");
            return argv[++index];
        };
        if (argument == "--status")
            options.status = true;
        else if (argument == "--dry-run")
            options.dryRun = true;
        else if (argument == "--force")
            options.force = true;
        else if (argument == "--quick")
            options.quick = true;
        else if (argument == "--keep-work")
            options.keepWork = true;
        else if (argument == "--quiet")
            options.quiet = true;
        else if (argument == "--append")
            options.append = true;
        else if (argument == "--probe")
            options.probe = std::string(value("--probe"));
        else if (argument == "--output")
            options.output = std::string(value("--output"));
        else if (argument == "--input")
            options.input = std::string(value("--input"));
        else if (argument == "--work")
            options.work = std::string(value("--work"));
        else if (argument == "--repeats")
            options.repeats = static_cast<int>(
                parseCount(value("--repeats"), "repeat count"));
        else if (argument == "--points")
        {
            for (const std::string& item : splitList(value("--points")))
                options.points.push_back(parseCount(item, "point count"));
        }
        else if (argument == "--models")
            options.models = splitList(value("--models"));
        else if (argument == "--help" || argument == "-h")
        {
            printUsage(std::cout);
            std::exit(0);
        }
        else
            throw std::invalid_argument("unknown option: " +
                                        std::string(argument));
    }
    if (options.points.empty())
        options.points = options.quick
                             ? std::vector<std::size_t>{250000U, 1000000U}
                             : std::vector<std::size_t>{250000U, 1000000U,
                                                        4000000U};
    if (options.quick && options.repeats == 3)
        options.repeats = 1;
    std::sort(options.points.begin(), options.points.end());
    options.points.erase(
        std::unique(options.points.begin(), options.points.end()),
        options.points.end());
    if (options.output.empty())
        options.output = defaultLocalProfilePath();
    return options;
}

std::string utcNow()
{
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

std::filesystem::path selfExecutable(const char* argv0)
{
    std::error_code error;
    std::filesystem::path self =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error || self.empty())
        self = std::filesystem::absolute(argv0, error);
    return self;
}

std::filesystem::path siblingPdal(const char* argv0)
{
    return selfExecutable(argv0).parent_path() / "pdal";
}

struct ProcessResult
{
    int exitStatus = -1;
    double wallSeconds = 0.0;
    std::string stderrTail;
};

// Runs a child with a scrubbed environment (no inherited PDG_* controls,
// LC_ALL=C, TZ=UTC, plus `extra`), stdout/stderr to files, and returns the
// complete-process wall clock.
ProcessResult
runProcess(const std::vector<std::string>& arguments,
           const std::vector<std::pair<std::string, std::string>>& extra,
           const std::filesystem::path& stdoutPath,
           const std::filesystem::path& stderrPath)
{
    std::vector<std::string> environmentStrings;
    for (char** entry = environ; entry && *entry; ++entry)
    {
        const std::string_view text(*entry);
        if (text.rfind("PDG_", 0) == 0 || text.rfind("PDAL_TEST_", 0) == 0 ||
            text.rfind("LC_ALL=", 0) == 0 || text.rfind("TZ=", 0) == 0)
            continue;
        environmentStrings.emplace_back(text);
    }
    environmentStrings.emplace_back("LC_ALL=C");
    environmentStrings.emplace_back("TZ=UTC");
    for (const auto& [name, value] : extra)
        environmentStrings.push_back(name + "=" + value);
    std::vector<char*> envp;
    for (std::string& item : environmentStrings)
        envp.push_back(item.data());
    envp.push_back(nullptr);
    std::vector<char*> argv;
    std::vector<std::string> argumentStorage = arguments;
    for (std::string& item : argumentStorage)
        argv.push_back(item.data());
    argv.push_back(nullptr);

    const Clock::time_point start = Clock::now();
    const pid_t child = ::fork();
    if (child < 0)
        throw std::runtime_error(std::string("fork failed: ") +
                                 std::strerror(errno));
    if (child == 0)
    {
        if (!::freopen(stdoutPath.c_str(), "w", stdout) ||
            !::freopen(stderrPath.c_str(), "w", stderr))
            std::_Exit(126);
        ::execve(argv[0], argv.data(), envp.data());
        std::_Exit(errno == ENOENT ? 127 : 126);
    }
    int status = 0;
    while (::waitpid(child, &status, 0) < 0)
        if (errno != EINTR)
            throw std::runtime_error(std::string("waitpid failed: ") +
                                     std::strerror(errno));
    ProcessResult result;
    result.wallSeconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    result.exitStatus = WIFEXITED(status)   ? WEXITSTATUS(status)
                        : WIFSIGNALED(status) ? 128 + WTERMSIG(status)
                                              : -1;
    std::ifstream errors(stderrPath);
    const std::string text((std::istreambuf_iterator<char>(errors)),
                           std::istreambuf_iterator<char>());
    result.stderrTail = text.size() > 600 ? text.substr(text.size() - 600) : text;
    return result;
}

struct LasFacts
{
    std::uint64_t pointCount = 0;
    std::uint16_t recordBytes = 0;
    std::uint8_t pointFormat = 0;
    std::uint8_t minorVersion = 0;
};

LasFacts lasFacts(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> header(375);
    input.read(reinterpret_cast<char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
    if (input.gcount() < 227 || std::memcmp(header.data(), "LASF", 4) != 0)
        throw std::runtime_error("not a LAS file: " + path.string());
    LasFacts facts;
    facts.minorVersion = header[25];
    facts.pointFormat = header[104] & 0x3FU;
    facts.recordBytes = static_cast<std::uint16_t>(header[105] |
                                                    (header[106] << 8));
    std::uint32_t legacy = 0;
    for (int i = 3; i >= 0; --i)
        legacy = (legacy << 8) | header[107 + static_cast<std::size_t>(i)];
    facts.pointCount = legacy;
    if (facts.minorVersion >= 4 && input.gcount() >= 255)
    {
        std::uint64_t modern = 0;
        for (int i = 7; i >= 0; --i)
            modern = (modern << 8) | header[247 + static_cast<std::size_t>(i)];
        if (modern != 0)
            facts.pointCount = modern;
    }
    return facts;
}

// Byte comparison of two LAS files ignoring only the header's creation
// day/year (offsets 90-93), which depend on the wall clock of each run.
bool sameLasBytes(const std::filesystem::path& a,
                  const std::filesystem::path& b, std::string& detail)
{
    std::error_code error;
    const auto sizeA = std::filesystem::file_size(a, error);
    if (error)
    {
        detail = "missing " + a.string();
        return false;
    }
    const auto sizeB = std::filesystem::file_size(b, error);
    if (error)
    {
        detail = "missing " + b.string();
        return false;
    }
    if (sizeA != sizeB)
    {
        detail = "sizes differ: " + std::to_string(sizeA) + " vs " +
                 std::to_string(sizeB);
        return false;
    }
    std::ifstream inA(a, std::ios::binary);
    std::ifstream inB(b, std::ios::binary);
    std::vector<char> bufferA(1U << 20);
    std::vector<char> bufferB(1U << 20);
    std::uint64_t offset = 0;
    while (inA && inB)
    {
        inA.read(bufferA.data(), static_cast<std::streamsize>(bufferA.size()));
        inB.read(bufferB.data(), static_cast<std::streamsize>(bufferB.size()));
        const std::streamsize got = inA.gcount();
        if (got != inB.gcount())
        {
            detail = "read lengths differ";
            return false;
        }
        for (std::streamsize i = 0; i < got; ++i)
        {
            const std::uint64_t at = offset + static_cast<std::uint64_t>(i);
            if (at >= 90 && at <= 93)
                continue;
            if (bufferA[static_cast<std::size_t>(i)] !=
                bufferB[static_cast<std::size_t>(i)])
            {
                detail = "first difference at byte " + std::to_string(at);
                return false;
            }
        }
        offset += static_cast<std::uint64_t>(got);
        if (got == 0)
            break;
    }
    return true;
}

double medianOf(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2U;
    return values.size() % 2U == 1U ? values[mid]
                                    : 0.5 * (values[mid - 1U] + values[mid]);
}

struct SizeMeasurement
{
    std::size_t points = 0;
    std::vector<double> hostSeconds;
    std::vector<double> deviceSeconds;
    bool byteExact = true;
    bool deviceUsed = false;
    std::string note;
    std::size_t inputRecordBytes = 0;
    std::size_t outputRecordBytes = 0;
    double plannerOwnedDeviceNanoseconds = 0.0;
};

struct ModelMeasurement
{
    std::string model;
    std::string pipelineKey;
    std::vector<SizeMeasurement> sizes;
};

void progress(const Options& options, const std::string& line)
{
    if (!options.quiet)
        std::cerr << "gpupal calibrate: " << line << '\n';
}

int printStatus(std::ostream& out)
{
    out << "gpupal " << Version << "\n";
    LocalProfileMachineKey machine;
    try
    {
        machine = currentLocalProfileMachineKey();
    }
    catch (const std::exception& error)
    {
        out << "machine: unavailable (" << error.what() << ")\n";
        out << "embedded_profile: not-applicable\n";
        out << "local_profile: not-attempted\n";
        return 2;
    }
    out << "device_name: " << machine.deviceName << '\n'
        << "compute_capability: " << machine.computeCapability << '\n'
        << "driver: " << machine.driverVersion << '\n'
        << "cuda: " << machine.cudaToolkitVersion << '\n'
        << "cpu_model: " << machine.cpuModel << '\n'
        << "logical_cpus: " << machine.logicalCpus << '\n';
    const bool ignoreBuiltin =
        std::getenv(IgnoreBuiltinProfileTestEnvironment.data()) != nullptr;
    // Ask the lookup without the local profile so the answer isolates the
    // embedded one: temporarily the local lookup is irrelevant because the
    // embedded profile is consulted first.
    const PlacementDeviceKey key{.name = machine.deviceName,
                                 .computeCapability = machine.computeCapability,
                                 .driverVersion = machine.driverVersion,
                                 .cudaToolkitVersion =
                                     machine.cudaToolkitVersion};
    const PlacementCalibrationProfile* active = placementCalibrationFor(key);
    const LocalProfileLookup lookup = lookupLocalProfile();
    const bool localApplied =
        lookup.status == LocalProfileStatus::Applied && lookup.profile;
    const bool embeddedApplies =
        active && !(localApplied && active->id == lookup.profile->id);
    out << "embedded_profile: "
        << (embeddedApplies ? std::string("applies (") +
                                  std::string(active->id) + ")"
            : ignoreBuiltin ? "ignored by " +
                                  std::string(IgnoreBuiltinProfileTestEnvironment)
                            : "does not apply (this is not the reference "
                              "machine)")
        << '\n';
    out << "local_profile_path: " << lookup.path.string() << '\n';
    out << "local_profile: " << localProfileStatusName(lookup.status);
    if (!lookup.detail.empty())
        out << " (" << lookup.detail << ")";
    out << '\n';
    if (lookup.profile)
    {
        out << "local_profile_id: " << lookup.profile->id << '\n'
            << "local_profile_created_utc: " << lookup.profile->createdUtc
            << '\n'
            << "local_profile_models: " << lookup.profile->stageModels.size()
            << '\n';
        if (!lookup.profile->summaries.empty())
        {
            out << std::left << std::setw(36) << "model" << std::right
                << std::setw(11) << "min_pts" << std::setw(11) << "max_pts"
                << std::setw(11) << "host_s" << std::setw(11) << "device_s"
                << std::setw(8) << "exact" << std::setw(8) << "wins" << '\n';
            for (const LocalProfileModelSummary& summary :
                 lookup.profile->summaries)
                out << std::left << std::setw(36) << summary.name << std::right
                    << std::setw(11) << summary.minimumDevicePoints
                    << std::setw(11) << summary.maximumDevicePoints
                    << std::setw(11) << std::fixed << std::setprecision(3)
                    << summary.hostSecondsAtMaximum << std::setw(11)
                    << summary.deviceSecondsAtMaximum << std::setw(8)
                    << (summary.byteExact ? "yes" : "NO") << std::setw(8)
                    << (summary.deviceWinsSomewhere ? "yes" : "no") << '\n';
        }
    }
    // D0279: shipped GPU-class and generic tiers.
    const PlacementCalibrationProfile* shipped = shippedPlacementProfileFor(key);
    const PlacementCalibrationProfile* generic = genericPlacementProfileFor(key);
    const std::string_view activeTier =
        active ? placementProfileTier(active) : std::string_view("none");
    std::string_view activeProfileSource = "none";
    std::string_view activeProfileSha256 = "unavailable";
    if (active && (activeTier == "shipped" || activeTier == "generic"))
    {
        for (const LocalPlacementProfile& profile : shippedProfiles())
        {
            if (profile.id != active->id)
                continue;
            activeProfileSource = profile.sourcePath;
            for (const ShippedProfileSource& source : shippedProfileSources())
            {
                if (source.name == activeProfileSource)
                {
                    activeProfileSha256 = source.sha256;
                    break;
                }
            }
            break;
        }
    }
    out << "shipped_profile: "
        << (shipped ? std::string(shipped->id) : std::string("none for this GPU"))
        << '\n'
        << "generic_profile: "
        << (generic ? std::string(generic->id)
                    : std::string("not applicable"))
        << '\n';
    out << "active_profile: "
        << (active ? std::string(active->id) : std::string("none (host path)"))
        << '\n'
        << "active_profile_tier: "
        << activeTier
        << '\n'
        << "active_profile_source: " << activeProfileSource
        << '\n'
        << "active_profile_sha256: " << activeProfileSha256
        << '\n';
    return 0;
}

int runProbe(const std::string& probe)
{
    std::cout << std::setprecision(17);
    if (probe == "cuda-startup")
    {
        std::cout << probeCudaStartupNanoseconds() << '\n';
        return 0;
    }
    if (probe == "transfers")
    {
        const CalibrationTransferProbe result =
            probeCudaTransfers(64U * 1024U * 1024U, 5);
        std::cout << result.hostToDeviceNanosecondsPerByte << ' '
                  << result.deviceToHostNanosecondsPerByte << '\n';
        return 0;
    }
    if (probe == "synchronization")
    {
        std::cout << probeCudaSynchronizationNanoseconds(200) << '\n';
        return 0;
    }
    if (probe == "index-build")
    {
        std::cout << probeIndexBuildNanosecondsPerByte(1000000U, 5) << '\n';
        return 0;
    }
    throw std::invalid_argument("unknown probe: " + probe);
}

std::string materializePipeline(const Json& pipeline,
                                const std::filesystem::path& input,
                                const std::filesystem::path& output)
{
    Json result = Json::array();
    for (const Json& stage : pipeline)
    {
        if (stage.is_string() && stage.get<std::string>() == "input.las")
            result.push_back(input.string());
        else if (stage.is_string() && stage.get<std::string>() == "output.las")
            result.push_back(output.string());
        else if (stage.is_object() && stage.value("filename", std::string()) == "output.las")
        {
            Json copy = stage;
            copy["filename"] = output.string();
            result.push_back(std::move(copy));
        }
        else if (stage.is_object() && stage.value("filename", std::string()) == "input.las")
        {
            Json copy = stage;
            copy["filename"] = input.string();
            result.push_back(std::move(copy));
        }
        else
            result.push_back(stage);
    }
    return result.dump();
}

void writeText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    if (!out)
        throw std::runtime_error("unable to write " + path.string());
}

// Prior measurements of `model` carried in a profile's evidence chain
// (--append nests the previous evidence under "previous_evidence"), so an
// appended run refits over every size ever measured with this coefficient set
// instead of replacing the model's fit and envelope with the new sizes only.
struct PriorCase
{
    std::size_t points = 0;
    double hostSeconds = 0.0;
    double deviceSeconds = 0.0;
    double plannerOwnedDeviceNanoseconds = 0.0;
    bool byteExact = false;
};

void collectPriorCases(const Json& evidence, std::string_view model,
                       std::vector<PriorCase>& out)
{
    if (!evidence.is_object())
        return;
    if (const auto cases = evidence.find("cases");
        cases != evidence.end() && cases->is_array())
        for (const Json& entry : *cases)
        {
            if (entry.value("model", std::string()) != model ||
                !entry.value("device_used", false))
                continue;
            std::vector<double> host = entry.value("host_seconds", std::vector<double>{});
            std::vector<double> device =
                entry.value("device_seconds", std::vector<double>{});
            if (host.empty() || device.empty())
                continue;
            out.push_back({.points = entry.value("points", std::size_t{0}),
                           .hostSeconds = medianOf(host),
                           .deviceSeconds = medianOf(device),
                           .plannerOwnedDeviceNanoseconds =
                               entry.value("planner_owned_device_ns", 0.0),
                           .byteExact = entry.value("byte_exact", false)});
        }
    if (const auto previous = evidence.find("previous_evidence");
        previous != evidence.end())
        collectPriorCases(*previous, model, out);
}

} // unnamed namespace

int runCalibrate(int argc, char** argv)
{
    Options options;
    try
    {
        options = parseOptions(argc, argv);
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << "gpupal calibrate: " << error.what() << "\n\n";
        printUsage(std::cerr);
        return 2;
    }
    try
    {
        if (options.probe)
            return runProbe(*options.probe);
        if (options.status)
            return printStatus(std::cout);

        if (!cudaBackendCompiled())
        {
            std::cerr << "gpupal calibrate: this build has no CUDA backend; "
                         "nothing to calibrate\n";
            return 2;
        }
        LocalProfileMachineKey machine;
        try
        {
            machine = currentLocalProfileMachineKey();
        }
        catch (const std::exception& error)
        {
            std::cerr << "gpupal calibrate: no usable CUDA device ("
                      << error.what() << ")\n";
            return 2;
        }
        const PlacementDeviceKey deviceKey{
            .name = machine.deviceName,
            .computeCapability = machine.computeCapability,
            .driverVersion = machine.driverVersion,
            .cudaToolkitVersion = machine.cudaToolkitVersion};
        const bool ignoreBuiltin =
            std::getenv(IgnoreBuiltinProfileTestEnvironment.data()) != nullptr;
        {
            // Is the embedded reference profile in force?  (The local lookup
            // is only consulted after it, so an active profile whose id is
            // not a local one is the embedded profile.)
            const PlacementCalibrationProfile* active =
                placementCalibrationFor(deviceKey);
            const LocalProfileLookup lookup = lookupLocalProfile();
            const bool embeddedApplies =
                active && !(lookup.status == LocalProfileStatus::Applied &&
                            lookup.profile && active->id == lookup.profile->id);
            if (embeddedApplies && !options.force && !ignoreBuiltin)
            {
                std::cout
                    << "gpupal calibrate: the embedded reference profile '"
                    << active->id
                    << "' applies to this machine; nothing to "
                       "calibrate (use --force to measure and write a "
                       "local profile anyway)\n";
                return 0;
            }
        }

        const Json plan = Json::parse(EmbeddedCalibrationPlan);
        std::vector<std::pair<std::string, std::string>> selected;
        for (const Json& entry : plan.at("models"))
        {
            const std::string model = entry.at("model").get<std::string>();
            if (!options.models.empty() &&
                std::find(options.models.begin(), options.models.end(),
                          model) == options.models.end())
                continue;
            selected.emplace_back(model, entry.at("pipeline").get<std::string>());
        }
        if (selected.empty())
        {
            std::cerr << "gpupal calibrate: no models selected\n";
            return 2;
        }
        for (const std::string& requested : options.models)
            if (std::none_of(selected.begin(), selected.end(),
                             [&](const auto& item)
                             { return item.first == requested; }))
            {
                std::cerr << "gpupal calibrate: unknown model '" << requested
                          << "'\n";
                return 2;
            }

        const std::filesystem::path engine = selfExecutable(argv[0]);
        const std::filesystem::path pdal = siblingPdal(argv[0]);
        if (!std::filesystem::is_regular_file(pdal))
        {
            std::cerr << "gpupal calibrate: sibling pdal not found at " << pdal
                      << '\n';
            return 2;
        }

        std::cout << "gpupal calibrate\n"
                  << "  device:   " << machine.deviceName << " (sm "
                  << machine.computeCapability << ", driver "
                  << machine.driverVersion << ", CUDA "
                  << machine.cudaToolkitVersion << ")\n"
                  << "  host:     " << machine.cpuModel << " ("
                  << machine.logicalCpus << " logical CPUs)\n"
                  << "  models:   " << selected.size() << "\n  sizes:    ";
        for (std::size_t i = 0; i < options.points.size(); ++i)
            std::cout << (i ? "," : "") << options.points[i];
        std::cout << "\n  repeats:  " << options.repeats << " host/device pairs per size\n"
                  << "  input:    "
                  << (options.input.empty() ? std::string("synthetic terrain cloud")
                                            : options.input.string())
                  << "\n  output:   " << options.output.string() << '\n';
        if (options.dryRun)
        {
            for (const auto& [model, key] : selected)
                std::cout << "  case:     " << model << "  <- "
                          << plan.at("pipelines").at(key).dump() << '\n';
            return 0;
        }

        // Work directory.
        std::filesystem::path work = options.work;
        if (work.empty())
            work = std::filesystem::temp_directory_path() /
                   ("pdg-calibrate-" + std::to_string(::getpid()));
        std::filesystem::create_directories(work);
        struct WorkGuard
        {
            std::filesystem::path path;
            bool keep;
            ~WorkGuard()
            {
                if (!keep)
                {
                    std::error_code error;
                    std::filesystem::remove_all(path, error);
                }
            }
        } workGuard{work, options.keepWork};

        // Fixtures.
        std::map<std::size_t, std::filesystem::path> fixtures;
        std::vector<std::size_t> sizes = options.points;
        if (options.input.empty())
        {
            for (std::size_t points : sizes)
            {
                const std::filesystem::path path =
                    work / ("synthetic-" + std::to_string(points) + ".las");
                progress(options, "writing synthetic fixture " +
                                      std::to_string(points) + " points");
                writeSyntheticLas(path, {.points = points});
                fixtures[points] = path;
            }
        }
        else
        {
            const LasFacts source = lasFacts(options.input);
            std::vector<std::size_t> usable;
            for (std::size_t points : sizes)
                if (points <= source.pointCount)
                    usable.push_back(points);
            if (usable.empty())
            {
                std::cerr << "gpupal calibrate: " << options.input
                          << " has only " << source.pointCount
                          << " points; no requested size fits\n";
                return 2;
            }
            if (usable.size() != sizes.size())
                progress(options, "input has " + std::to_string(source.pointCount) +
                                      " points; larger sizes dropped");
            sizes = usable;
            for (std::size_t points : sizes)
            {
                const std::filesystem::path path =
                    work / ("subset-" + std::to_string(points) + ".las");
                progress(options, "materializing " + std::to_string(points) +
                                      "-point subset of " +
                                      options.input.string());
                const ProcessResult result = runProcess(
                    {pdal.string(), "translate", options.input.string(),
                     path.string(),
                     "--readers.las.count=" + std::to_string(points),
                     "--writers.las.forward=all"},
                    {}, work / "subset.stdout", work / "subset.stderr");
                if (result.exitStatus != 0)
                {
                    std::cerr << "gpupal calibrate: subset materialization "
                                 "failed: "
                              << result.stderrTail << '\n';
                    return 1;
                }
                fixtures[points] = path;
            }
        }

        // --append: an existing profile for this exact machine keeps the
        // models not measured now and supplies the coefficient set, so every
        // residual fit in the file shares one set of planner-owned terms.
        std::optional<LocalPlacementProfile> existing;
        Json existingEvidence;
        if (options.append && std::filesystem::exists(options.output))
        {
            std::ifstream in(options.output, std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            std::string error;
            existing = parseLocalProfile(text, error);
            if (existing)
                existingEvidence = Json::parse(text).value("evidence", Json());
            if (!existing)
                throw std::runtime_error("--append: existing profile is not "
                                         "readable: " + error);
            if (existing->machine != machine)
                throw std::runtime_error(
                    "--append: the existing profile is keyed to a different "
                    "machine; run without --append to replace it");
            progress(options, "appending to " + options.output.string() +
                                  " (" + std::to_string(existing->stageModels.size()) +
                                  " models, coefficients reused)");
        }

        // Coefficient probes.
        std::vector<double> startups;
        PlacementModelCoefficients coefficients;
        if (existing)
            coefficients = existing->coefficients;
        else
        {
            progress(options, "probing CUDA start-up, transfers, "
                              "synchronization, index build");
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                const ProcessResult result = runProcess(
                    {engine.string(), "calibrate", "--probe", "cuda-startup"},
                    {}, work / "probe.stdout", work / "probe.stderr");
                if (result.exitStatus != 0)
                    throw std::runtime_error("cuda-startup probe failed: " +
                                             result.stderrTail);
                std::ifstream in(work / "probe.stdout");
                double value = 0.0;
                in >> value;
                startups.push_back(value);
            }
            const CalibrationTransferProbe transfers =
                probeCudaTransfers(64U * 1024U * 1024U, 5);
            coefficients = {
                .cudaStartupNanoseconds = medianOf(startups),
                .hostToDeviceNanosecondsPerByte =
                    transfers.hostToDeviceNanosecondsPerByte,
                .deviceToHostNanosecondsPerByte =
                    transfers.deviceToHostNanosecondsPerByte,
                .packingNanosecondsPerByte = ReferencePackingNanosecondsPerByte,
                .indexBuildNanosecondsPerByte =
                    probeIndexBuildNanosecondsPerByte(
                        std::min<std::size_t>(sizes.back(), 1000000U), 5),
                .synchronizationNanoseconds =
                    probeCudaSynchronizationNanoseconds(200)};
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "  cuda_startup_ms:        "
                  << coefficients.cudaStartupNanoseconds / 1e6
                  << "\n  host_to_device_ns/B:    "
                  << coefficients.hostToDeviceNanosecondsPerByte
                  << "\n  device_to_host_ns/B:    "
                  << coefficients.deviceToHostNanosecondsPerByte
                  << "\n  synchronization_us:     "
                  << coefficients.synchronizationNanoseconds / 1e3
                  << "\n  index_build_ns/B:       "
                  << coefficients.indexBuildNanosecondsPerByte
                  << "\n  packing_ns/B:           "
                  << coefficients.packingNanosecondsPerByte
                  << (existing ? " (reused from the existing profile)"
                               : " (inherited from the reference profile)")
                  << '\n';

        // Measurements.
        std::vector<ModelMeasurement> measurements;
        bool warmed = false;
        const std::vector<std::pair<std::string, std::string>> deviceEnvironment{
            {std::string(CalibrationForceDeviceEnvironment), "1"},
            {std::string(IgnoreBuiltinProfileTestEnvironment), "1"}};
        for (const auto& [model, key] : selected)
        {
            ModelMeasurement measurement{model, key, {}};
            for (std::size_t points : sizes)
            {
                SizeMeasurement sample;
                sample.points = points;
                const std::filesystem::path input = fixtures.at(points);
                const std::string tag = model + "-" + std::to_string(points);
                const std::filesystem::path hostOutput =
                    work / (tag + "-host.las");
                const std::filesystem::path deviceOutput =
                    work / (tag + "-device.las");
                const std::filesystem::path hostPipeline =
                    work / (tag + "-host.json");
                const std::filesystem::path devicePipeline =
                    work / (tag + "-device.json");
                const std::filesystem::path stats = work / (tag + "-stats.json");
                writeText(hostPipeline,
                          materializePipeline(plan.at("pipelines").at(key),
                                              input, hostOutput));
                writeText(devicePipeline,
                          materializePipeline(plan.at("pipelines").at(key),
                                              input, deviceOutput));
                sample.inputRecordBytes = lasFacts(input).recordBytes;

                const auto runHost = [&]() -> std::optional<double>
                {
                    // Outputs are removed before every run, as the reference
                    // runner does; the direct resident LAS executor refuses to
                    // overwrite an existing output.
                    std::error_code ignored;
                    std::filesystem::remove(hostOutput, ignored);
                    const ProcessResult result = runProcess(
                        {pdal.string(), "pipeline", hostPipeline.string()}, {},
                        work / (tag + "-host.stdout"),
                        work / (tag + "-host.stderr"));
                    if (result.exitStatus != 0)
                    {
                        sample.note = "host run failed: " + result.stderrTail;
                        return std::nullopt;
                    }
                    return result.wallSeconds;
                };
                const auto runDevice = [&]() -> std::optional<double>
                {
                    std::error_code ignored;
                    std::filesystem::remove(deviceOutput, ignored);
                    std::filesystem::remove(stats, ignored);
                    const ProcessResult result = runProcess(
                        {engine.string(), "resident", devicePipeline.string(),
                         "--stats", stats.string()},
                        deviceEnvironment, work / (tag + "-device.stdout"),
                        work / (tag + "-device.stderr"));
                    if (result.exitStatus != 0)
                    {
                        sample.note = "device run failed: " + result.stderrTail;
                        return std::nullopt;
                    }
                    try
                    {
                        std::ifstream in(stats);
                        const Json document = Json::parse(in);
                        const Json& placement = document.at("placement");
                        if (placement.value("choice", std::string()) != "device")
                        {
                            sample.note =
                                "device placement was not selected (" +
                                placement.value("unavailable_reason",
                                                std::string("?")) +
                                ", reason " +
                                placement.value("reason", std::string("?")) +
                                ")";
                            return std::nullopt;
                        }
                    }
                    catch (const std::exception& error)
                    {
                        sample.note =
                            std::string("stats unreadable: ") + error.what();
                        return std::nullopt;
                    }
                    return result.wallSeconds;
                };

                if (!warmed)
                {
                    progress(options, "warm-up device run (" + tag + ")");
                    static_cast<void>(runDevice());
                    warmed = true;
                }
                bool usable = true;
                for (int repeat = 0; repeat < options.repeats && usable;
                     ++repeat)
                {
                    progress(options, tag + " pair " + std::to_string(repeat + 1) +
                                          "/" + std::to_string(options.repeats));
                    const std::optional<double> host = runHost();
                    if (!host)
                    {
                        usable = false;
                        break;
                    }
                    const std::optional<double> device = runDevice();
                    if (!device)
                    {
                        usable = false;
                        break;
                    }
                    sample.hostSeconds.push_back(*host);
                    sample.deviceSeconds.push_back(*device);
                    std::string detail;
                    if (!sameLasBytes(hostOutput, deviceOutput, detail))
                    {
                        sample.byteExact = false;
                        sample.note = "host/device outputs differ: " + detail;
                    }
                }
                sample.deviceUsed = usable && !sample.deviceSeconds.empty();
                if (sample.deviceUsed)
                {
                    sample.outputRecordBytes = lasFacts(hostOutput).recordBytes;
                    // Planner-owned device terms for the residual fit, from
                    // the same request builder the audit uses.
                    DimensionRegistry dimensions;
                    const Plan compiled = compilePipeline(
                        materializePipeline(plan.at("pipelines").at(key),
                                            "input.las", "output.las"),
                        dimensions);
                    const PlacementCalibrationShape shape{
                        .id = tag,
                        .model = model,
                        .pipelineJson = {},
                        .inputPoints = points,
                        .outputPoints = points,
                        .pointCapacity = points,
                        .inputRecordBytes = sample.inputRecordBytes,
                        .outputRecordBytes = sample.outputRecordBytes,
                        .fallbackRecordBytes = 0U,
                        .additionalSynchronizations = 0U,
                        .deviceMemoryBudgetBytes =
                            (std::numeric_limits<std::size_t>::max)(),
                        .cudaContextWarm = false,
                        .intrinsicSingleLaneExecutor = false};
                    const PlacementRequest request =
                        makePlacementCalibrationRequest(compiled, shape);
                    const PlacementEstimate estimate =
                        evaluatePlacement(compiled, request, coefficients);
                    sample.plannerOwnedDeviceNanoseconds =
                        estimate.device.totalNanoseconds;
                }
                std::error_code ignored;
                std::filesystem::remove(hostOutput, ignored);
                std::filesystem::remove(deviceOutput, ignored);
                measurement.sizes.push_back(std::move(sample));
            }
            measurements.push_back(std::move(measurement));
        }

        // Fit.
        LocalPlacementProfile profile;
        profile.id = "local-" + machine.deviceName + "-" + utcNow();
        for (char& c : profile.id)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' ||
                  c == '.'))
                c = '-';
        profile.createdUtc = utcNow();
        profile.machine = machine;
        profile.coefficients = coefficients;
        Json evidenceCases = Json::array();
        if (existing)
        {
            const auto measuredNow = [&](std::string_view name)
            {
                return std::any_of(measurements.begin(), measurements.end(),
                                   [&](const ModelMeasurement& m)
                                   { return m.model == name; });
            };
            for (std::size_t index = 0; index < existing->stageModels.size();
                 ++index)
                if (!measuredNow(existing->modelNames[index]))
                {
                    profile.modelNames.push_back(existing->modelNames[index]);
                    profile.stageModels.push_back(
                        {std::string_view(),
                         existing->stageModels[index].cost});
                }
            for (const LocalProfileModelSummary& summary : existing->summaries)
                if (!measuredNow(summary.name))
                    profile.summaries.push_back(summary);
        }
        std::cout << "\n"
                  << std::left << std::setw(36) << "model" << std::right
                  << std::setw(10) << "points" << std::setw(11) << "host_s"
                  << std::setw(11) << "device_s" << std::setw(9) << "speedup"
                  << std::setw(7) << "exact" << "  note\n";
        for (const ModelMeasurement& measurement : measurements)
        {
            std::vector<PlacementResidualSample> residuals;
            std::size_t minimumWin = 0;
            std::size_t maximumMeasured = 0;
            bool anyWin = false;
            bool allExact = true;
            double hostAtMax = 0.0;
            double deviceAtMax = 0.0;
            // Prior sizes measured under the same coefficient set (--append).
            std::vector<PriorCase> prior;
            if (existing)
                collectPriorCases(existingEvidence, measurement.model, prior);
            for (const PriorCase& past : prior)
            {
                if (!past.byteExact)
                {
                    allExact = false;
                    continue;
                }
                residuals.push_back(
                    {static_cast<double>(past.points),
                     (past.deviceSeconds - past.hostSeconds) * 1e9 -
                         past.plannerOwnedDeviceNanoseconds});
                if (past.points >= maximumMeasured)
                {
                    maximumMeasured = past.points;
                    hostAtMax = past.hostSeconds;
                    deviceAtMax = past.deviceSeconds;
                }
                if (past.deviceSeconds < past.hostSeconds)
                {
                    anyWin = true;
                    minimumWin = minimumWin == 0
                                     ? past.points
                                     : std::min(minimumWin, past.points);
                }
            }
            for (const SizeMeasurement& sample : measurement.sizes)
            {
                Json entry{{"model", measurement.model},
                           {"points", sample.points},
                           {"host_seconds", sample.hostSeconds},
                           {"device_seconds", sample.deviceSeconds},
                           {"byte_exact", sample.byteExact},
                           {"device_used", sample.deviceUsed},
                           {"input_record_bytes", sample.inputRecordBytes},
                           {"output_record_bytes", sample.outputRecordBytes},
                           {"planner_owned_device_ns",
                            sample.plannerOwnedDeviceNanoseconds},
                           {"note", sample.note}};
                evidenceCases.push_back(entry);
                std::cout << std::left << std::setw(36) << measurement.model
                          << std::right << std::setw(10) << sample.points;
                if (!sample.deviceUsed)
                {
                    std::cout << std::setw(11) << "-" << std::setw(11) << "-"
                              << std::setw(9) << "-" << std::setw(7) << "-"
                              << "  " << sample.note << '\n';
                    continue;
                }
                const double host = medianOf(sample.hostSeconds);
                const double device = medianOf(sample.deviceSeconds);
                std::cout << std::setw(11) << std::fixed << std::setprecision(3)
                          << host << std::setw(11) << device << std::setw(8)
                          << std::setprecision(2) << host / device << "x"
                          << std::setw(7) << (sample.byteExact ? "yes" : "NO")
                          << "  " << sample.note << '\n';
                if (!sample.byteExact)
                {
                    allExact = false;
                    continue; // never admit a route that changed bytes
                }
                residuals.push_back(
                    {static_cast<double>(sample.points),
                     (device - host) * 1e9 -
                         sample.plannerOwnedDeviceNanoseconds});
                if (sample.points >= maximumMeasured)
                {
                    maximumMeasured = sample.points;
                    hostAtMax = host;
                    deviceAtMax = device;
                }
                if (device < host)
                {
                    anyWin = true;
                    minimumWin = minimumWin == 0
                                     ? sample.points
                                     : std::min(minimumWin, sample.points);
                }
            }
            LocalProfileModelSummary summary;
            summary.name = measurement.model;
            summary.byteExact = allExact;
            summary.deviceWinsSomewhere = anyWin;
            summary.hostSecondsAtMaximum = hostAtMax;
            summary.deviceSecondsAtMaximum = deviceAtMax;
            if (anyWin && allExact && !residuals.empty())
            {
                StagePlacementCost cost = fitPlacementResidualModel(residuals);
                cost.minimumDevicePointCount = minimumWin;
                cost.maximumDevicePointCount = maximumMeasured;
                summary.minimumDevicePoints = minimumWin;
                summary.maximumDevicePoints = maximumMeasured;
                profile.modelNames.push_back(measurement.model);
                profile.stageModels.push_back({std::string_view(), cost});
            }
            profile.summaries.push_back(std::move(summary));
        }
        for (std::size_t index = 0; index < profile.stageModels.size(); ++index)
            profile.stageModels[index].name = profile.modelNames[index];

        Json evidence{
            {"protocol",
             "complete-process wall clock; host = sibling pdal pipeline, "
             "device = pdg-engine resident under the calibration placement "
             "override; alternating pairs; medians; outputs byte-compared"},
            {"engine", engine.string()},
            {"pdal", pdal.string()},
            {"sizes", sizes},
            {"repeats", options.repeats},
            {"input", options.input.empty() ? std::string("synthetic")
                                            : options.input.string()},
            {"packing_ns_per_byte_source", "reference profile (not measured)"},
            {"appended_to", existing ? Json(existing->id) : Json(nullptr)},
            {"previous_evidence", existingEvidence},
            {"probes",
             {{"cuda_startup_ns_samples", startups},
              {"transfer_bytes", 64U * 1024U * 1024U},
              {"index_probe_points",
               std::min<std::size_t>(sizes.back(), 1000000U)}}},
            {"cases", std::move(evidenceCases)}};
        {
            std::ifstream quota("/sys/fs/cgroup/cpu.max");
            std::string text;
            if (quota && std::getline(quota, text))
                evidence["cgroup_cpu_max"] = text;
        }

        std::filesystem::create_directories(options.output.parent_path());
        if (std::filesystem::exists(options.output))
        {
            std::error_code ignored;
            std::filesystem::copy_file(
                options.output, options.output.string() + ".previous",
                std::filesystem::copy_options::overwrite_existing, ignored);
        }
        writeText(options.output, serializeLocalProfile(profile, evidence.dump()));
        std::size_t admittedNow = 0;
        for (const ModelMeasurement& measurement : measurements)
            if (std::find(profile.modelNames.begin(), profile.modelNames.end(),
                          measurement.model) != profile.modelNames.end())
                ++admittedNow;
        std::cout << "\nwrote " << options.output.string() << " ("
                  << admittedNow << " of " << measurements.size()
                  << " measured models admitted for device placement; "
                  << profile.stageModels.size() << " models in the profile)\n";
        if (admittedNow == 0)
            std::cout << "no measured model showed a byte-exact device win; "
                         "the automatic route stays on the host path for "
                         "them\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "gpupal calibrate: " << error.what() << '\n';
        return 1;
    }
}

} // namespace pdg::cli
