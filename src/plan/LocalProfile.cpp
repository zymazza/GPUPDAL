#include <pdg/LocalProfile.hpp>

#include <pdg/Memory.hpp>
#include <pdg/Version.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pdg
{
namespace
{
using Json = nlohmann::json;

std::string trimmed(std::string value)
{
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    while (!value.empty() && !notSpace(static_cast<unsigned char>(value.back())))
        value.pop_back();
    std::size_t start = 0;
    while (start < value.size() &&
           !notSpace(static_cast<unsigned char>(value[start])))
        ++start;
    return value.substr(start);
}

double finiteNonnegative(const Json& object, const char* key)
{
    const Json& value = object.at(key);
    if (!value.is_number())
        throw std::invalid_argument(std::string("not a number: ") + key);
    const double number = value.get<double>();
    if (!(number >= 0.0) ||
        number == (std::numeric_limits<double>::infinity)())
        throw std::invalid_argument(std::string("invalid coefficient: ") +
                                    key);
    return number;
}

std::size_t optionalSize(const Json& object, const char* key,
                         std::size_t fallback)
{
    const auto position = object.find(key);
    if (position == object.end() || position->is_null())
        return fallback;
    if (!position->is_number_unsigned())
        throw std::invalid_argument(std::string("not an unsigned count: ") +
                                    key);
    return position->get<std::size_t>();
}

std::string requiredString(const Json& object, const char* key)
{
    const Json& value = object.at(key);
    if (!value.is_string())
        throw std::invalid_argument(std::string("not a string: ") + key);
    return value.get<std::string>();
}

Json machineJson(const LocalProfileMachineKey& machine)
{
    return {{"device_name", machine.deviceName},
            {"compute_capability", machine.computeCapability},
            {"driver", machine.driverVersion},
            {"cuda", machine.cudaToolkitVersion},
            {"cpu_model", machine.cpuModel},
            {"logical_cpus", machine.logicalCpus},
            {"pdg_version", machine.pdgVersion}};
}

LocalProfileMachineKey parseMachine(const Json& object)
{
    LocalProfileMachineKey machine;
    machine.deviceName = requiredString(object, "device_name");
    machine.computeCapability = requiredString(object, "compute_capability");
    machine.driverVersion = requiredString(object, "driver");
    machine.cudaToolkitVersion = requiredString(object, "cuda");
    machine.cpuModel = requiredString(object, "cpu_model");
    const Json& cpus = object.at("logical_cpus");
    if (!cpus.is_number_unsigned())
        throw std::invalid_argument("logical_cpus must be an unsigned count");
    machine.logicalCpus = cpus.get<unsigned>();
    machine.pdgVersion = requiredString(object, "pdg_version");
    return machine;
}

std::string describeMismatch(const LocalProfileMachineKey& file,
                             const LocalProfileMachineKey& current)
{
    std::ostringstream out;
    const auto field = [&](const char* name, const std::string& a,
                           const std::string& b)
    {
        if (a != b)
            out << name << " '" << a << "' != '" << b << "'; ";
    };
    field("device_name", file.deviceName, current.deviceName);
    field("compute_capability", file.computeCapability,
          current.computeCapability);
    field("driver", file.driverVersion, current.driverVersion);
    field("cuda", file.cudaToolkitVersion, current.cudaToolkitVersion);
    field("cpu_model", file.cpuModel, current.cpuModel);
    if (file.logicalCpus != current.logicalCpus)
        out << "logical_cpus " << file.logicalCpus
            << " != " << current.logicalCpus << "; ";
    field("pdg_version", file.pdgVersion, current.pdgVersion);
    std::string text = out.str();
    if (text.size() >= 2)
        text.resize(text.size() - 2);
    return text;
}

LocalProfileLookup performLookup() noexcept
{
    LocalProfileLookup lookup;
    try
    {
        lookup.path = defaultLocalProfilePath();
        if (std::getenv(LocalProfileDisableEnvironment.data()))
        {
            lookup.status = LocalProfileStatus::Disabled;
            lookup.detail = std::string(LocalProfileDisableEnvironment) +
                            " is set";
            return lookup;
        }
        if (!cudaBackendCompiled())
        {
            lookup.status = LocalProfileStatus::NotAttempted;
            lookup.detail = "CUDA backend not compiled";
            return lookup;
        }
        std::ifstream input(lookup.path, std::ios::binary);
        if (!input)
        {
            lookup.status = LocalProfileStatus::NotFound;
            lookup.detail = "no profile file";
            return lookup;
        }
        const std::string text((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
        std::string error;
        std::optional<LocalPlacementProfile> profile =
            parseLocalProfile(text, error);
        if (!profile)
        {
            lookup.status = LocalProfileStatus::Malformed;
            lookup.detail = error;
            return lookup;
        }
        profile->sourcePath = lookup.path.string();
        try
        {
            lookup.currentMachine = currentLocalProfileMachineKey();
        }
        catch (const std::exception& exception)
        {
            lookup.status = LocalProfileStatus::NotAttempted;
            lookup.detail = exception.what();
            lookup.profile = std::move(profile);
            return lookup;
        }
        if (profile->machine != *lookup.currentMachine)
        {
            lookup.status = LocalProfileStatus::MachineMismatch;
            lookup.detail =
                describeMismatch(profile->machine, *lookup.currentMachine);
            lookup.profile = std::move(profile);
            return lookup;
        }
        lookup.status = LocalProfileStatus::Applied;
        lookup.detail = profile->id;
        lookup.profile = std::move(profile);
        return lookup;
    }
    catch (const std::exception& exception)
    {
        lookup.status = LocalProfileStatus::Malformed;
        lookup.detail = exception.what();
        lookup.profile.reset();
        return lookup;
    }
}
} // unnamed namespace

PlacementCalibrationProfile LocalPlacementProfile::view() const noexcept
{
    return PlacementCalibrationProfile{
        .id = id,
        .device = {.name = machine.deviceName,
                   .computeCapability = machine.computeCapability,
                   .driverVersion = machine.driverVersion,
                   .cudaToolkitVersion = machine.cudaToolkitVersion},
        .coefficients = coefficients,
        .stageModels = std::span<const PlacementStageCalibration>(stageModels)};
}

std::filesystem::path defaultLocalProfilePath()
{
    if (const char* explicitPath = std::getenv(LocalProfilePathEnvironment.data());
        explicitPath && *explicitPath)
        return std::filesystem::path(explicitPath);
    std::filesystem::path base;
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg)
        base = xdg;
    else if (const char* home = std::getenv("HOME"); home && *home)
        base = std::filesystem::path(home) / ".config";
    else
        base = std::filesystem::path(".");
    return base / "gpupal" / "placement-profile.json";
}

std::string hostCpuModelName()
{
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line))
    {
        if (line.rfind("model name", 0) != 0)
            continue;
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        return trimmed(line.substr(colon + 1));
    }
    return {};
}

LocalProfileMachineKey currentLocalProfileMachineKey()
{
    if (!cudaBackendCompiled())
        throw std::runtime_error("CUDA backend not compiled");
    const std::vector<CudaDeviceSummary> devices = cudaDevices();
    if (devices.empty())
        throw std::runtime_error("no CUDA device");
    const CudaDeviceSummary& device = devices.front();
    LocalProfileMachineKey key;
    key.deviceName = device.name;
    key.computeCapability = std::to_string(device.computeMajor) + "." +
                            std::to_string(device.computeMinor);
    key.driverVersion = nvidiaKernelDriverVersion();
    key.cudaToolkitVersion =
        formatCudaToolkitVersion(cudaCompiledToolkitVersion());
    key.cpuModel = hostCpuModelName();
    key.logicalCpus = std::thread::hardware_concurrency();
    key.pdgVersion = std::string(Version);
    return key;
}

std::optional<LocalPlacementProfile> parseLocalProfile(std::string_view json,
                                                       std::string& error)
{
    try
    {
        const Json document = Json::parse(json);
        if (!document.is_object())
            throw std::invalid_argument("profile is not a JSON object");
        if (document.at("schema") != LocalProfileSchema)
            throw std::invalid_argument("unsupported local profile schema");
        LocalPlacementProfile profile;
        profile.id = requiredString(document, "id");
        profile.createdUtc = requiredString(document, "created_utc");
        profile.tier = document.value("tier", std::string("local"));
        if (profile.tier != "local" && profile.tier != "shipped" &&
            profile.tier != "generic")
            throw std::invalid_argument("unknown profile tier: " + profile.tier);
        if (const auto applies = document.find("applies");
            applies != document.end() && applies->is_object())
        {
            profile.minimumComputeCapability =
                applies->value("minimum_compute_capability", std::string());
            profile.minimumDeviceMemoryBytes =
                applies->value("minimum_device_memory_bytes", std::uint64_t{0});
        }
        profile.machine = parseMachine(document.at("machine"));
        const Json& coefficients = document.at("coefficients");
        profile.coefficients = {
            .cudaStartupNanoseconds =
                finiteNonnegative(coefficients, "cuda_startup_ns"),
            .hostToDeviceNanosecondsPerByte =
                finiteNonnegative(coefficients, "host_to_device_ns_per_byte"),
            .deviceToHostNanosecondsPerByte =
                finiteNonnegative(coefficients, "device_to_host_ns_per_byte"),
            .packingNanosecondsPerByte =
                finiteNonnegative(coefficients, "packing_ns_per_byte"),
            .indexBuildNanosecondsPerByte =
                finiteNonnegative(coefficients, "index_build_ns_per_byte"),
            .synchronizationNanoseconds =
                finiteNonnegative(coefficients, "synchronization_ns")};
        const Json& models = document.at("stage_models");
        if (!models.is_object())
            throw std::invalid_argument("stage_models must be an object");
        profile.modelNames.reserve(models.size());
        profile.stageModels.reserve(models.size());
        for (const auto& [name, value] : models.items())
        {
            if (name.empty())
                throw std::invalid_argument("empty stage model name");
            for (const std::string& existing : profile.modelNames)
                if (existing == name)
                    throw std::invalid_argument("duplicate stage model: " +
                                                name);
            StagePlacementCost cost{
                .hostFixedNanoseconds = finiteNonnegative(value, "host_fixed_ns"),
                .deviceFixedNanoseconds =
                    finiteNonnegative(value, "device_fixed_ns"),
                .hostNanosecondsPerPoint =
                    finiteNonnegative(value, "host_ns_per_point"),
                .deviceNanosecondsPerPoint =
                    finiteNonnegative(value, "device_ns_per_point"),
                .calibrated = true};
            cost.minimumDevicePointCount =
                optionalSize(value, "minimum_device_points", 0U);
            cost.maximumDevicePointCount = optionalSize(
                value, "maximum_device_points",
                (std::numeric_limits<std::size_t>::max)());
            if (cost.minimumDevicePointCount > cost.maximumDevicePointCount)
                throw std::invalid_argument("invalid envelope for stage model " +
                                            name);
            profile.modelNames.push_back(name);
            profile.stageModels.push_back({std::string_view(), cost});
        }
        // Names are stable now that the vector is fully sized.
        for (std::size_t index = 0; index < profile.stageModels.size(); ++index)
            profile.stageModels[index].name = profile.modelNames[index];
        if (const auto summaries = document.find("summaries");
            summaries != document.end() && summaries->is_array())
        {
            for (const Json& item : *summaries)
            {
                LocalProfileModelSummary summary;
                summary.name = requiredString(item, "model");
                summary.minimumDevicePoints =
                    optionalSize(item, "minimum_device_points", 0U);
                summary.maximumDevicePoints =
                    optionalSize(item, "maximum_device_points", 0U);
                summary.hostSecondsAtMaximum =
                    item.value("host_seconds_at_maximum", 0.0);
                summary.deviceSecondsAtMaximum =
                    item.value("device_seconds_at_maximum", 0.0);
                summary.byteExact = item.value("byte_exact", false);
                summary.deviceWinsSomewhere =
                    item.value("device_wins_somewhere", false);
                profile.summaries.push_back(std::move(summary));
            }
        }
        return profile;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return std::nullopt;
    }
}

std::string serializeLocalProfile(const LocalPlacementProfile& profile,
                                  std::string_view evidenceJson)
{
    Json models = Json::object();
    for (const PlacementStageCalibration& model : profile.stageModels)
    {
        Json entry{{"host_fixed_ns", model.cost.hostFixedNanoseconds},
                   {"device_fixed_ns", model.cost.deviceFixedNanoseconds},
                   {"host_ns_per_point", model.cost.hostNanosecondsPerPoint},
                   {"device_ns_per_point",
                    model.cost.deviceNanosecondsPerPoint}};
        if (model.cost.minimumDevicePointCount != 0U)
            entry["minimum_device_points"] = model.cost.minimumDevicePointCount;
        if (model.cost.maximumDevicePointCount !=
            (std::numeric_limits<std::size_t>::max)())
            entry["maximum_device_points"] = model.cost.maximumDevicePointCount;
        models[std::string(model.name)] = std::move(entry);
    }
    Json summaries = Json::array();
    for (const LocalProfileModelSummary& summary : profile.summaries)
        summaries.push_back(
            {{"model", summary.name},
             {"minimum_device_points", summary.minimumDevicePoints},
             {"maximum_device_points", summary.maximumDevicePoints},
             {"host_seconds_at_maximum", summary.hostSecondsAtMaximum},
             {"device_seconds_at_maximum", summary.deviceSecondsAtMaximum},
             {"byte_exact", summary.byteExact},
             {"device_wins_somewhere", summary.deviceWinsSomewhere}});
    Json document{
        {"schema", std::string(LocalProfileSchema)},
        {"id", profile.id},
        {"created_utc", profile.createdUtc},
        {"tier", profile.tier},
        {"machine", machineJson(profile.machine)},
        {"coefficients",
         {{"cuda_startup_ns", profile.coefficients.cudaStartupNanoseconds},
          {"host_to_device_ns_per_byte",
           profile.coefficients.hostToDeviceNanosecondsPerByte},
          {"device_to_host_ns_per_byte",
           profile.coefficients.deviceToHostNanosecondsPerByte},
          {"packing_ns_per_byte", profile.coefficients.packingNanosecondsPerByte},
          {"index_build_ns_per_byte",
           profile.coefficients.indexBuildNanosecondsPerByte},
          {"synchronization_ns",
           profile.coefficients.synchronizationNanoseconds}}},
        {"stage_models", std::move(models)},
        {"summaries", std::move(summaries)}};
    if (!profile.minimumComputeCapability.empty() ||
        profile.minimumDeviceMemoryBytes != 0U)
        document["applies"] = {
            {"minimum_compute_capability", profile.minimumComputeCapability},
            {"minimum_device_memory_bytes", profile.minimumDeviceMemoryBytes}};
    if (!evidenceJson.empty())
        document["evidence"] = Json::parse(evidenceJson);
    return document.dump(2) + "\n";
}

const LocalProfileLookup& loadedLocalProfile() noexcept
{
    static const LocalProfileLookup lookup = performLookup();
    return lookup;
}

LocalProfileLookup lookupLocalProfile() noexcept
{
    return performLookup();
}

std::string_view localProfileStatusName(LocalProfileStatus status) noexcept
{
    switch (status)
    {
    case LocalProfileStatus::NotAttempted:
        return "not-attempted";
    case LocalProfileStatus::Disabled:
        return "disabled";
    case LocalProfileStatus::NotFound:
        return "not-found";
    case LocalProfileStatus::Malformed:
        return "malformed";
    case LocalProfileStatus::MachineMismatch:
        return "machine-mismatch";
    case LocalProfileStatus::Applied:
        return "applied";
    }
    return "unknown";
}

const PlacementCalibrationProfile*
calibrationForcedDevicePlacementProfile(const PlacementDeviceKey& device) noexcept
{
    if (!std::getenv(CalibrationForceDeviceEnvironment.data()))
        return nullptr;
    try
    {
        // Built once for the process from the first key requested; the
        // calibrate child asks for exactly one device.
        static const std::unique_ptr<LocalPlacementProfile> forced = [&]
        {
            auto profile = std::make_unique<LocalPlacementProfile>();
            profile->id = "calibration-forced-device";
            profile->machine.deviceName = std::string(device.name);
            profile->machine.computeCapability =
                std::string(device.computeCapability);
            profile->machine.driverVersion = std::string(device.driverVersion);
            profile->machine.cudaToolkitVersion =
                std::string(device.cudaToolkitVersion);
            // Planner-owned terms are irrelevant to the forced choice; zero
            // keeps every device estimate below the host estimate.
            profile->coefficients = {};
            std::vector<std::string_view> names = embeddedPlacementModelNames();
            for (std::string_view name : names)
            {
                profile->modelNames.emplace_back(name);
                profile->stageModels.push_back(
                    {std::string_view(),
                     StagePlacementCost{.hostFixedNanoseconds = 1.0e9,
                                        .deviceFixedNanoseconds = 0.0,
                                        .hostNanosecondsPerPoint = 1.0e6,
                                        .deviceNanosecondsPerPoint = 0.0,
                                        .calibrated = true}});
            }
            for (std::size_t index = 0; index < profile->stageModels.size();
                 ++index)
                profile->stageModels[index].name = profile->modelNames[index];
            return profile;
        }();
        static const PlacementCalibrationProfile view = forced->view();
        if (view.device.name != device.name ||
            view.device.computeCapability != device.computeCapability ||
            view.device.driverVersion != device.driverVersion ||
            view.device.cudaToolkitVersion != device.cudaToolkitVersion)
            return nullptr;
        return &view;
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

namespace
{
struct ShippedProfileTable
{
    std::vector<LocalPlacementProfile> profiles;
    // Views into `profiles` (stable: the vector is never resized after
    // construction).
    std::vector<PlacementCalibrationProfile> views;
    // Generic-tier view bound to the current device key at first use.
    std::unique_ptr<LocalPlacementProfile> genericBound;
    std::unique_ptr<PlacementCalibrationProfile> genericView;
    std::string tierOfGeneric;
};

ShippedProfileTable& shippedTable()
{
    static ShippedProfileTable table = []
    {
        ShippedProfileTable result;
        const char* extraDir = std::getenv(ShippedProfileDirTestEnvironment.data());
        const bool onlyExtra =
            extraDir && *extraDir &&
            std::getenv(ShippedProfileDirOnlyTestEnvironment.data());
        for (const ShippedProfileSource& source : shippedProfileSources())
        {
            if (onlyExtra)
                break;
            std::string error;
            std::optional<LocalPlacementProfile> profile =
                parseLocalProfile(source.json, error);
            if (!profile)
            {
                if (std::getenv("PDG_DEBUG_HYBRID"))
                    std::cerr << "gpupal: shipped placement profile "
                              << source.name << " ignored: " << error << '\n';
                continue;
            }
            profile->sourcePath = std::string(source.name);
            result.profiles.push_back(std::move(*profile));
        }
        // Test hook: additional shipped/generic profiles from a directory, so
        // the tier order can be exercised without rebuilding the embedded
        // table.
        if (const char* extra = extraDir; extra && *extra)
        {
            std::error_code error;
            std::vector<std::filesystem::path> files;
            for (const auto& entry :
                 std::filesystem::directory_iterator(extra, error))
                if (entry.path().extension() == ".json")
                    files.push_back(entry.path());
            std::sort(files.begin(), files.end());
            for (const std::filesystem::path& file : files)
            {
                std::ifstream in(file, std::ios::binary);
                const std::string text((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
                std::string parseError;
                std::optional<LocalPlacementProfile> profile =
                    parseLocalProfile(text, parseError);
                if (!profile)
                    continue;
                profile->sourcePath = file.string();
                result.profiles.push_back(std::move(*profile));
            }
        }
        result.views.reserve(result.profiles.size());
        for (const LocalPlacementProfile& profile : result.profiles)
            result.views.push_back(profile.view());
        return result;
    }();
    return table;
}

[[nodiscard]] bool capabilityAtLeast(std::string_view actual,
                                     std::string_view minimum) noexcept
{
    const auto parse = [](std::string_view text, int& major, int& minor)
    {
        const std::size_t dot = text.find('.');
        try
        {
            major = std::stoi(std::string(text.substr(0, dot)));
            minor = dot == std::string_view::npos
                        ? 0
                        : std::stoi(std::string(text.substr(dot + 1)));
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    };
    int aMajor = 0, aMinor = 0, mMajor = 0, mMinor = 0;
    if (!parse(actual, aMajor, aMinor) || !parse(minimum, mMajor, mMinor))
        return false;
    return aMajor > mMajor || (aMajor == mMajor && aMinor >= mMinor);
}
} // unnamed namespace

const std::vector<LocalPlacementProfile>& shippedProfiles()
{
    return shippedTable().profiles;
}

const PlacementCalibrationProfile*
shippedPlacementProfileFor(const PlacementDeviceKey& device) noexcept
{
    if (std::getenv(DisableShippedProfilesEnvironment.data()))
        return nullptr;
    try
    {
        const ShippedProfileTable& table = shippedTable();
        for (std::size_t index = 0; index < table.profiles.size(); ++index)
        {
            const LocalPlacementProfile& profile = table.profiles[index];
            if (profile.tier != "shipped")
                continue;
            if (profile.machine.deviceName == device.name &&
                profile.machine.computeCapability == device.computeCapability &&
                profile.machine.cudaToolkitVersion == device.cudaToolkitVersion)
                return &table.views[index];
        }
        return nullptr;
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

const PlacementCalibrationProfile*
genericPlacementProfileFor(const PlacementDeviceKey& device) noexcept
{
    if (std::getenv(DisableShippedProfilesEnvironment.data()))
        return nullptr;
    try
    {
        ShippedProfileTable& table = shippedTable();
        const LocalPlacementProfile* generic = nullptr;
        for (const LocalPlacementProfile& profile : table.profiles)
            if (profile.tier == "generic")
            {
                generic = &profile;
                break;
            }
        if (!generic || !cudaBackendCompiled())
            return nullptr;
        if (!generic->minimumComputeCapability.empty() &&
            !capabilityAtLeast(device.computeCapability,
                               generic->minimumComputeCapability))
            return nullptr;
        if (generic->machine.cudaToolkitVersion != device.cudaToolkitVersion)
            return nullptr;
        if (generic->minimumDeviceMemoryBytes != 0U)
        {
            const std::vector<CudaDeviceSummary> devices = cudaDevices();
            if (devices.empty() ||
                devices.front().totalMemory < generic->minimumDeviceMemoryBytes)
                return nullptr;
        }
        // Bind the generic view to the first device key that asked for it, so
        // the planner's exact-profile identity check holds for this process.
        if (!table.genericBound)
        {
            table.genericBound =
                std::make_unique<LocalPlacementProfile>(*generic);
            table.genericBound->machine.deviceName = std::string(device.name);
            table.genericBound->machine.computeCapability =
                std::string(device.computeCapability);
            table.genericBound->machine.driverVersion =
                std::string(device.driverVersion);
            table.genericBound->machine.cudaToolkitVersion =
                std::string(device.cudaToolkitVersion);
            table.genericView = std::make_unique<PlacementCalibrationProfile>(
                table.genericBound->view());
        }
        const PlacementCalibrationProfile& view = *table.genericView;
        if (view.device.name != device.name ||
            view.device.computeCapability != device.computeCapability ||
            view.device.driverVersion != device.driverVersion ||
            view.device.cudaToolkitVersion != device.cudaToolkitVersion)
            return nullptr;
        return &view;
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

std::string_view
placementProfileTier(const PlacementCalibrationProfile* profile) noexcept
{
    if (!profile)
        return "";
    try
    {
        const ShippedProfileTable& table = shippedTable();
        for (const PlacementCalibrationProfile& view : table.views)
            if (&view == profile)
                return "shipped";
        if (table.genericView && table.genericView.get() == profile)
            return "generic";
        const LocalProfileLookup& local = loadedLocalProfile();
        if (local.status == LocalProfileStatus::Applied && local.profile &&
            profile->id == local.profile->id)
            return "local";
        if (profile->id == "calibration-forced-device")
            return "";
        return "embedded";
    }
    catch (const std::exception&)
    {
        return "";
    }
}

} // namespace pdg
