#include <pdg/Placement.hpp>
#include <pdg/PlacementCalibration.hpp>
#include <pdg/Plan.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using Json = nlohmann::json;

struct Options
{
    std::string calibrationPath;
    bool featuresOnly = false;
    bool suggestModels = false;
};

struct CalibrationCase
{
    std::string id;
    std::string model;
    std::string pipeline;
    std::size_t inputPoints = 0;
    std::size_t outputPoints = 0;
    std::size_t pointCapacity = 0;
    std::size_t inputRecordBytes = 0;
    std::size_t outputRecordBytes = 0;
    std::size_t fallbackRecordBytes = 0;
    std::size_t additionalSynchronizations = 0;
    std::size_t deviceMemoryBudgetBytes =
        (std::numeric_limits<std::size_t>::max)();
    double hostSeconds = 0.0;
    double deviceSeconds = 0.0;
    bool cudaContextWarm = false;
    bool intrinsicSingleLaneExecutor = false;
};

[[nodiscard]] Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--features-only")
            options.featuresOnly = true;
        else if (argument == "--suggest-models")
            options.suggestModels = true;
        else if (argument == "--calibration" && index + 1 < argc)
            options.calibrationPath = argv[++index];
        else
            throw std::invalid_argument(
                "usage: pdg_placement_audit --calibration FILE "
                "[--features-only|--suggest-models]");
    }
    if (options.calibrationPath.empty())
        throw std::invalid_argument("--calibration is required");
    if (options.featuresOnly && options.suggestModels)
        throw std::invalid_argument(
            "--features-only and --suggest-models are mutually exclusive");
    return options;
}

[[nodiscard]] Json readJson(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("unable to open calibration file: " + path);
    Json document;
    input >> document;
    return document;
}

[[nodiscard]] double nonnegativeNumber(const Json& object, std::string_view key)
{
    const double value = object.at(std::string(key)).get<double>();
    if (!(value >= 0.0) || value == (std::numeric_limits<double>::infinity)())
        throw std::invalid_argument("invalid nonnegative calibration value: " +
                                    std::string(key));
    return value;
}

[[nodiscard]] std::size_t optionalSize(const Json& object, std::string_view key,
                                       std::size_t fallback)
{
    const auto position = object.find(std::string(key));
    return position == object.end() ? fallback : position->get<std::size_t>();
}

[[nodiscard]] bool optionalBool(const Json& object, std::string_view key,
                                bool fallback)
{
    const auto position = object.find(std::string(key));
    return position == object.end() ? fallback : position->get<bool>();
}

[[nodiscard]] pdg::PlacementModelCoefficients
parseCoefficients(const Json& document)
{
    const Json& values = document.at("coefficients");
    return {
        .cudaStartupNanoseconds = nonnegativeNumber(values, "cuda_startup_ns"),
        .hostToDeviceNanosecondsPerByte =
            nonnegativeNumber(values, "host_to_device_ns_per_byte"),
        .deviceToHostNanosecondsPerByte =
            nonnegativeNumber(values, "device_to_host_ns_per_byte"),
        .packingNanosecondsPerByte =
            nonnegativeNumber(values, "packing_ns_per_byte"),
        .indexBuildNanosecondsPerByte =
            nonnegativeNumber(values, "index_build_ns_per_byte"),
        .synchronizationNanoseconds =
            nonnegativeNumber(values, "synchronization_ns"),
    };
}

[[nodiscard]] pdg::StagePlacementCost parseStageCost(const Json& document,
                                                     const std::string& key)
{
    const Json& value = document.at("stage_models").at(key);
    pdg::StagePlacementCost result{
        .hostFixedNanoseconds = nonnegativeNumber(value, "host_fixed_ns"),
        .deviceFixedNanoseconds = nonnegativeNumber(value, "device_fixed_ns"),
        .hostNanosecondsPerPoint =
            nonnegativeNumber(value, "host_ns_per_point"),
        .deviceNanosecondsPerPoint =
            nonnegativeNumber(value, "device_ns_per_point"),
        .calibrated = true,
    };
    result.minimumDevicePointCount =
        optionalSize(value, "minimum_device_points", 0U);
    result.maximumDevicePointCount =
        optionalSize(value, "maximum_device_points",
                     (std::numeric_limits<std::size_t>::max)());
    if (result.minimumDevicePointCount > result.maximumDevicePointCount)
        throw std::invalid_argument("invalid stage calibration envelope: " +
                                    key);
    return result;
}

[[nodiscard]] bool
sameCoefficients(const pdg::PlacementModelCoefficients& left,
                 const pdg::PlacementModelCoefficients& right) noexcept
{
    return left.cudaStartupNanoseconds == right.cudaStartupNanoseconds &&
           left.hostToDeviceNanosecondsPerByte ==
               right.hostToDeviceNanosecondsPerByte &&
           left.deviceToHostNanosecondsPerByte ==
               right.deviceToHostNanosecondsPerByte &&
           left.packingNanosecondsPerByte == right.packingNanosecondsPerByte &&
           left.indexBuildNanosecondsPerByte ==
               right.indexBuildNanosecondsPerByte &&
           left.synchronizationNanoseconds == right.synchronizationNanoseconds;
}

[[nodiscard]] bool sameStageCost(const pdg::StagePlacementCost& left,
                                 const pdg::StagePlacementCost& right) noexcept
{
    return left.hostFixedNanoseconds == right.hostFixedNanoseconds &&
           left.deviceFixedNanoseconds == right.deviceFixedNanoseconds &&
           left.hostNanosecondsPerPoint == right.hostNanosecondsPerPoint &&
           left.deviceNanosecondsPerPoint == right.deviceNanosecondsPerPoint &&
           left.minimumDevicePointCount == right.minimumDevicePointCount &&
           left.maximumDevicePointCount == right.maximumDevicePointCount &&
           left.calibrated == right.calibrated;
}

[[nodiscard]] CalibrationCase parseCase(const Json& document, const Json& value)
{
    CalibrationCase result;
    result.id = value.at("id").get<std::string>();
    result.model = value.at("model").get<std::string>();
    const Json& pipeline = value.at("pipeline");
    result.pipeline =
        pipeline.is_string()
            ? document.at("pipelines").at(pipeline.get<std::string>()).dump()
            : pipeline.dump();
    result.inputPoints = value.at("input_points").get<std::size_t>();
    result.outputPoints =
        optionalSize(value, "output_points", result.inputPoints);
    result.pointCapacity =
        optionalSize(value, "point_capacity", result.inputPoints);
    result.inputRecordBytes = value.at("input_record_bytes").get<std::size_t>();
    result.outputRecordBytes =
        value.at("output_record_bytes").get<std::size_t>();
    result.fallbackRecordBytes =
        optionalSize(value, "fallback_record_bytes", 0U);
    result.additionalSynchronizations =
        optionalSize(value, "additional_synchronizations", 0U);
    result.deviceMemoryBudgetBytes =
        optionalSize(value, "device_memory_budget_bytes",
                     (std::numeric_limits<std::size_t>::max)());
    result.hostSeconds = nonnegativeNumber(value, "host_seconds");
    result.deviceSeconds = nonnegativeNumber(value, "device_seconds");
    result.cudaContextWarm = optionalBool(value, "cuda_context_warm", false);
    result.intrinsicSingleLaneExecutor =
        optionalBool(value, "intrinsic_single_lane_executor", false);
    if (result.inputPoints == 0U || result.pointCapacity == 0U)
        throw std::invalid_argument(
            "placement calibration counts must be positive");
    return result;
}

[[nodiscard]] pdg::PlacementCalibrationShape
shapeOf(const CalibrationCase& calibrationCase)
{
    return {.id = calibrationCase.id,
            .model = calibrationCase.model,
            .pipelineJson = calibrationCase.pipeline,
            .inputPoints = calibrationCase.inputPoints,
            .outputPoints = calibrationCase.outputPoints,
            .pointCapacity = calibrationCase.pointCapacity,
            .inputRecordBytes = calibrationCase.inputRecordBytes,
            .outputRecordBytes = calibrationCase.outputRecordBytes,
            .fallbackRecordBytes = calibrationCase.fallbackRecordBytes,
            .additionalSynchronizations =
                calibrationCase.additionalSynchronizations,
            .deviceMemoryBudgetBytes = calibrationCase.deviceMemoryBudgetBytes,
            .cudaContextWarm = calibrationCase.cudaContextWarm,
            .intrinsicSingleLaneExecutor =
                calibrationCase.intrinsicSingleLaneExecutor};
}

// D0277: the request builder is shared with `pdg calibrate` so a local
// profile is fitted by exactly the audit's procedure.
[[nodiscard]] pdg::PlacementRequest
makeRequest(const pdg::Plan& plan, const CalibrationCase& calibrationCase)
{
    return pdg::makePlacementCalibrationRequest(plan, shapeOf(calibrationCase));
}

[[nodiscard]] const char* choiceName(pdg::PlacementChoice choice)
{
    return choice == pdg::PlacementChoice::Device ? "device" : "host";
}

int run(const Options& options)
{
    const Json document = readJson(options.calibrationPath);
    if (document.at("schema") != "pdg-placement-calibration-v1")
        throw std::invalid_argument("unsupported placement calibration schema");
    const Json& device = document.at("device");
    const pdg::PlacementDeviceKey deviceKey{
        .name = device.at("name").get_ref<const std::string&>(),
        .computeCapability =
            device.at("compute_capability").get_ref<const std::string&>(),
        .driverVersion = device.at("driver").get_ref<const std::string&>(),
        .cudaToolkitVersion = device.at("cuda").get_ref<const std::string&>()};
    const pdg::PlacementCalibrationProfile* profile =
        pdg::placementCalibrationFor(deviceKey);
    if (!profile)
        throw std::invalid_argument(
            "calibration device has no embedded placement profile");
    if (profile->id != document.at("calibration_id").get<std::string>())
        throw std::invalid_argument(
            "embedded placement profile id differs from calibration record");
    const pdg::PlacementModelCoefficients parsedCoefficients =
        parseCoefficients(document);
    if (!sameCoefficients(profile->coefficients, parsedCoefficients))
        throw std::invalid_argument(
            "embedded placement coefficients differ from calibration record");
    if (profile->stageModels.size() != document.at("stage_models").size())
        throw std::invalid_argument("embedded placement stage-model count "
                                    "differs from calibration record");
    for (const auto& [name, unused] : document.at("stage_models").items())
    {
        static_cast<void>(unused);
        const pdg::StagePlacementCost* embedded =
            pdg::placementStageCalibration(*profile, name);
        if (!embedded ||
            !sameStageCost(*embedded, parseStageCost(document, name)))
            throw std::invalid_argument("embedded placement stage model "
                                        "differs from calibration record: " +
                                        name);
    }
    const pdg::PlacementModelCoefficients& coefficients = profile->coefficients;
    const double minimumAccuracy =
        nonnegativeNumber(document, "minimum_accuracy");

    std::size_t correct = 0U;
    std::size_t total = 0U;
    std::set<std::string> coveredModels;
    std::map<std::string, std::size_t> maximumWinningEvidencePoints;
    std::map<std::string, std::vector<std::pair<double, double>>> modelSamples;
    std::cout << std::fixed << std::setprecision(6);
    for (const Json& value : document.at("cases"))
    {
        const CalibrationCase calibrationCase = parseCase(document, value);
        coveredModels.insert(calibrationCase.model);
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan =
            pdg::compilePipeline(calibrationCase.pipeline, dimensions);
        const bool withoutStageModel =
            options.featuresOnly || options.suggestModels;
        const pdg::PlacementRequest planRequest =
            makeRequest(plan, calibrationCase);
        pdg::PlacementRequest calibratedRequest = planRequest;
        if (!withoutStageModel)
        {
            if (plan.summary().residentRegions != 1U)
                throw std::invalid_argument("single-model calibration case has "
                                            "multiple resident regions: " +
                                            calibrationCase.id);
            const pdg::PlacementRegionCalibration calibration{
                .residentRegion = 0U, .model = calibrationCase.model};
            if (!pdg::applyPlacementRegionCalibrations(
                    plan, *profile, std::span(&calibration, 1U),
                    calibratedRequest))
                throw std::invalid_argument("calibration case references an "
                                            "unknown embedded stage model: " +
                                            calibrationCase.model);
        }
        const pdg::PlacementEstimate estimate =
            pdg::evaluatePlacement(plan, calibratedRequest, coefficients);
        const pdg::PlanPlacementEstimate planEstimate =
            pdg::evaluatePlanPlacement(plan, calibratedRequest, coefficients);
        const pdg::PlacementChoice measured =
            calibrationCase.deviceSeconds < calibrationCase.hostSeconds
                ? pdg::PlacementChoice::Device
                : pdg::PlacementChoice::Host;
        if (measured == pdg::PlacementChoice::Device)
            maximumWinningEvidencePoints[calibrationCase.model] = std::max(
                maximumWinningEvidencePoints[calibrationCase.model],
                calibrationCase.inputPoints);
        if (options.suggestModels)
        {
            const double measuredDeltaNanoseconds =
                (calibrationCase.deviceSeconds - calibrationCase.hostSeconds) *
                1'000'000'000.0;
            modelSamples[calibrationCase.model].emplace_back(
                static_cast<double>(calibrationCase.inputPoints),
                measuredDeltaNanoseconds - estimate.device.totalNanoseconds);
            continue;
        }
        if (!options.featuresOnly)
        {
            ++total;
            if (planEstimate.choice == measured)
                ++correct;
        }
        const std::size_t hostToDeviceBytes =
            options.featuresOnly ? estimate.hostToDeviceBytes
                                 : planEstimate.hostToDeviceBytes;
        const std::size_t deviceToHostBytes =
            options.featuresOnly ? estimate.deviceToHostBytes
                                 : planEstimate.deviceToHostBytes;
        const std::size_t packingBytes = options.featuresOnly
                                             ? estimate.packingBytes
                                             : planEstimate.packingBytes;
        const std::size_t indexBuildBytes = options.featuresOnly
                                                ? estimate.indexBuildBytes
                                                : planEstimate.indexBuildBytes;
        const std::size_t synchronizationCount =
            options.featuresOnly ? estimate.synchronizationCount
                                 : planEstimate.synchronizationCount;
        const double hostNanoseconds =
            options.featuresOnly
                ? estimate.host.totalNanoseconds
                : planEstimate.allHostPlacement.totalNanoseconds;
        const double selectedNanoseconds =
            options.featuresOnly
                ? estimate.device.totalNanoseconds
                : planEstimate.selectedPlacement.totalNanoseconds;
        std::cout << calibrationCase.id << '\t' << calibrationCase.model << '\t'
                  << calibrationCase.inputPoints << '\t' << hostToDeviceBytes
                  << '\t' << deviceToHostBytes << '\t' << packingBytes << '\t'
                  << indexBuildBytes << '\t' << synchronizationCount << '\t'
                  << hostNanoseconds / 1'000'000'000.0 << '\t'
                  << selectedNanoseconds / 1'000'000'000.0;
        if (!options.featuresOnly)
            std::cout << '\t' << choiceName(measured) << '\t'
                      << choiceName(planEstimate.choice) << '\t'
                      << (planEstimate.choice == measured ? "match" : "MISS");
        std::cout << '\n';
    }
    for (const auto& [name, unused] : document.at("stage_models").items())
    {
        static_cast<void>(unused);
        if (!coveredModels.contains(name))
            throw std::invalid_argument(
                "embedded placement stage model has no calibration case: " +
                name);
        const pdg::StagePlacementCost stageCost =
            parseStageCost(document, name);
        const Json& stageModel = document.at("stage_models").at(name);
        // Legacy models without an explicit upper bound predate this audit.
        // Enforce the evidence ceiling whenever the profile claims one; D0281's
        // two repaired composition models are both explicitly bounded.
        if (stageModel.contains("maximum_device_points") &&
            stageCost.maximumDevicePointCount >
            maximumWinningEvidencePoints[name])
            throw std::invalid_argument(
                "embedded placement stage model exceeds its largest recorded "
                "device-winning case: " +
                name);
    }
    if (options.suggestModels)
    {
        std::cout << std::setprecision(12);
        for (const auto& [name, samples] : modelSamples)
        {
            std::vector<pdg::PlacementResidualSample> residuals;
            residuals.reserve(samples.size());
            for (const auto& [points, residual] : samples)
                residuals.push_back({points, residual});
            pdg::StagePlacementCost fit;
            try
            {
                fit = pdg::fitPlacementResidualModel(residuals);
            }
            catch (const std::invalid_argument& error)
            {
                throw std::invalid_argument(std::string(error.what()) + ": " +
                                            name);
            }
            std::cout << name << '\t' << fit.hostFixedNanoseconds << '\t'
                      << fit.deviceFixedNanoseconds << '\t'
                      << fit.hostNanosecondsPerPoint << '\t'
                      << fit.deviceNanosecondsPerPoint << '\n';
        }
        return 0;
    }
    if (options.featuresOnly)
        return 0;
    const double accuracy =
        total == 0U ? 0.0
                    : static_cast<double>(correct) / static_cast<double>(total);
    std::cout << "accuracy\t" << correct << '/' << total << '\t' << accuracy
              << '\n';
    if (accuracy < minimumAccuracy)
        return 2;
    return 0;
}
} // unnamed namespace

int main(int argc, char** argv)
{
    try
    {
        return run(parseOptions(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << "pdg_placement_audit: " << error.what() << '\n';
        return 1;
    }
}
