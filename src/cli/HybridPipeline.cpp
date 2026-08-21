#include "HybridPipeline.hpp"

#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>

#include <nlohmann/json.hpp>

#include <pdal/Kernel.hpp>
#include <pdal/Log.hpp>
#include <pdal/PDALUtils.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PluginManager.hpp>
#include <pdal/PointLayout.hpp>
#include <pdal/Stage.hpp>
#include <pdal/util/FileUtils.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pdg::cli
{

namespace
{
using LayoutSignature =
    std::vector<std::pair<std::string, pdal::Dimension::Type>>;

bool matchesAutomaticR2LiteralGrammar(std::string_view text) noexcept
{
    using Json = nlohmann::json;
    const auto stringOption = [](const Json& stage, const char* name,
                                 std::string_view expected) noexcept
    {
        const auto value = stage.find(name);
        return value != stage.end() && value->is_string() &&
               value->get_ref<const std::string&>() == expected;
    };
    const auto filenameOption = [](const Json& stage) noexcept
    {
        const auto value = stage.find("filename");
        if (value == stage.end() || !value->is_string())
            return false;
        const std::string& filename = value->get_ref<const std::string&>();
        constexpr std::string_view Laz = ".laz";
        constexpr std::string_view CopcLaz = ".copc.laz";
        if (filename.size() < Laz.size() ||
            filename.substr(filename.size() - Laz.size()) != Laz)
            return false;
        if (filename.size() < CopcLaz.size())
            return true;
        const std::string_view suffix(
            filename.data() + filename.size() - CopcLaz.size(),
            CopcLaz.size());
        for (std::size_t index = 0U; index < CopcLaz.size(); ++index)
        {
            char character = suffix[index];
            if (character >= 'A' && character <= 'Z')
                character = static_cast<char>(character - 'A' + 'a');
            if (character != CopcLaz[index])
                return true;
        }
        return false;
    };
    try
    {
        const Json root = Json::parse(text, nullptr, true, true);
        if (!root.is_object() || root.size() != 1U)
            return false;
        const auto found = root.find("pipeline");
        if (found == root.end() || !found->is_array() || found->size() != 4U)
            return false;
        const Json& reader = found->at(0U);
        const Json& smrf = found->at(1U);
        const Json& hag = found->at(2U);
        const Json& writer = found->at(3U);
        return reader.is_object() && reader.size() == 2U &&
               stringOption(reader, "type", "readers.las") &&
               filenameOption(reader) && smrf.is_object() &&
               smrf.size() == 1U &&
               stringOption(smrf, "type", "filters.smrf") &&
               hag.is_object() && hag.size() == 1U &&
               stringOption(hag, "type", "filters.hag_nn") &&
               writer.is_object() && writer.size() == 4U &&
               stringOption(writer, "type", "writers.las") &&
               filenameOption(writer) &&
               stringOption(writer, "compression", "true") &&
               stringOption(writer, "extra_dims",
                            "HeightAboveGround=float32");
    }
    catch (const std::exception&)
    {
        return false;
    }
}

LayoutSignature layoutSignature(const pdal::PointLayoutPtr& layout)
{
    LayoutSignature signature;
    signature.reserve(layout->dims().size());
    for (const pdal::Dimension::Id dimension : layout->dims())
        signature.emplace_back(layout->dimName(dimension),
                               layout->dimType(dimension));
    return signature;
}

struct AutomaticLasInputFacts
{
    std::uint64_t pointCount{};
    bool measuredR4Input{};
    bool measuredLabelNnDistanceInput{};
};

bool measuredR4Fingerprint(std::ifstream& input)
{
    constexpr std::uint64_t OffsetBasis = 14'695'981'039'346'656'037ULL;
    constexpr std::uint64_t Prime = 1'099'511'628'211ULL;
    constexpr std::uint64_t Reference = 0x0ddc8c913fb8553fULL;
    std::uint64_t fingerprint = OffsetBasis;
    std::array<char, 64U * 1024U> block{};
    input.clear();
    input.seekg(0, std::ios::beg);
    while (input)
    {
        input.read(block.data(), static_cast<std::streamsize>(block.size()));
        const std::streamsize bytes = input.gcount();
        for (std::streamsize index = 0; index < bytes; ++index)
        {
            fingerprint ^= static_cast<unsigned char>(
                block[static_cast<std::size_t>(index)]);
            fingerprint *= Prime;
        }
    }
    return input.eof() && fingerprint == Reference;
}

std::optional<AutomaticLasInputFacts>
lasInputFacts(const HybridPipelineRewrite& rewrite)
{
    if (rewrite.automaticPointCountFilename.empty())
        return std::nullopt;
    const std::filesystem::path path(rewrite.automaticPointCountFilename);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        return std::nullopt;
    const std::uintmax_t fileBytes = std::filesystem::file_size(path, error);
    if (error)
        return std::nullopt;
    std::string filename = path.filename().string();
    std::transform(filename.begin(), filename.end(), filename.begin(),
                   [](unsigned char character)
                   { return static_cast<char>(std::tolower(character)); });
    if (filename.ends_with(".copc.laz") ||
        (!filename.ends_with(".las") && !filename.ends_with(".laz")))
        return std::nullopt;

    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    constexpr std::size_t Las12HeaderBytes = 227;
    constexpr std::size_t Las14HeaderBytes = 375;
    std::array<char, Las14HeaderBytes> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::streamsize count = input.gcount();
    if (count < static_cast<std::streamsize>(Las12HeaderBytes) ||
        std::memcmp(bytes.data(), "LASF", 4) != 0)
        return std::nullopt;
    const auto read = [&]<typename T>(std::size_t offset)
    {
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    };
    const std::uint8_t versionMajor = read.operator()<std::uint8_t>(24);
    const std::uint8_t versionMinor = read.operator()<std::uint8_t>(25);
    if (versionMajor != 1U || versionMinor > 4U)
        return std::nullopt;
    std::uint64_t points = read.operator()<std::uint32_t>(107);
    if (versionMinor >= 4U)
    {
        if (count < static_cast<std::streamsize>(Las14HeaderBytes))
            return std::nullopt;
        points = read.operator()<std::uint64_t>(247);
    }
    const bool measuredR4HeaderSummary =
        fileBytes == 6'747'641U && versionMajor == 1U && versionMinor == 4U &&
        read.operator()<std::uint16_t>(94) == 375U &&
        read.operator()<std::uint32_t>(96) == 475U &&
        read.operator()<std::uint8_t>(104) == 0x87U &&
        read.operator()<std::uint16_t>(105) == 36U &&
        read.operator()<double>(131) == 0.01 &&
        read.operator()<double>(139) == 0.01 &&
        read.operator()<double>(147) == 0.01 &&
        read.operator()<double>(155) == 0.0 &&
        read.operator()<double>(163) == 0.0 &&
        read.operator()<double>(171) == 0.0 &&
        read.operator()<double>(179) == 185999.99 &&
        read.operator()<double>(187) == 184500.0 &&
        read.operator()<double>(195) == 494999.99 &&
        read.operator()<double>(203) == 494923.21 &&
        read.operator()<double>(211) == 500.41 &&
        read.operator()<double>(219) == 367.44;
    const std::uint32_t pointOffset = read.operator()<std::uint32_t>(96);
    const std::uint16_t recordBytes = read.operator()<std::uint16_t>(105);
    const bool measuredLabelNnDistanceLayout =
        filename.ends_with(".las") && versionMajor == 1U &&
        versionMinor == 4U && pointOffset == Las14HeaderBytes &&
        read.operator()<std::uint32_t>(100) == 0U &&
        read.operator()<std::uint8_t>(104) == 7U && recordBytes == 36U &&
        read.operator()<std::uint64_t>(235) == 0U &&
        read.operator()<std::uint32_t>(243) == 0U &&
        points <= ((std::numeric_limits<std::uintmax_t>::max)() - pointOffset) /
                      recordBytes &&
        fileBytes == pointOffset + points * recordBytes;
    return AutomaticLasInputFacts{
        (std::min)(points, rewrite.automaticPointCountLimit),
        measuredR4HeaderSummary && measuredR4Fingerprint(input),
        measuredLabelNnDistanceLayout};
}

bool automaticCudaAvailable() noexcept
{
    if (std::getenv("PDG_DISABLE_CUDA_HYBRID") || !pdg::cudaBackendCompiled())
        return false;
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const std::exception&)
    {
        return false;
    }
}
} // unnamed namespace

static std::optional<int> tryHybridPipelineImpl(int argc, char** argv)
{
    if (argc < 3 || std::string_view(argv[1]) != "pipeline" ||
        std::getenv("PDG_DISABLE_HYBRID"))
        return std::nullopt;

    bool forceStream = false;
    bool forceStandard = false;
    std::string metadataFile;
    std::optional<pdal::LogLevel> verbosity;
    const auto parseVerbosity =
        [](std::string_view text) -> std::optional<pdal::LogLevel>
    {
        std::istringstream input{std::string(text)};
        pdal::LogLevel value = pdal::LogLevel::None;
        input >> value;
        std::string trailing;
        if (!input || (input >> trailing))
            return std::nullopt;
        return value;
    };
    for (int index = 3; index < argc; ++index)
    {
        const std::string_view option(argv[index]);
        if (option == "--stream")
            forceStream = true;
        else if (option == "--nostream")
            forceStandard = true;
        else if (option == "--metadata")
        {
            if (++index >= argc)
                return std::nullopt;
            metadataFile = argv[index];
        }
        else if (option.starts_with("--metadata="))
            metadataFile =
                option.substr(std::string_view("--metadata=").size());
        else if (option == "--verbose" || option == "-v")
        {
            if (++index >= argc)
                return std::nullopt;
            verbosity = parseVerbosity(argv[index]);
            if (!verbosity)
                return std::nullopt;
        }
        else if (option.starts_with("--verbose="))
        {
            verbosity = parseVerbosity(
                option.substr(std::string_view("--verbose=").size()));
            if (!verbosity)
                return std::nullopt;
        }
        else
            return std::nullopt;
    }
    if (forceStream && forceStandard)
        return std::nullopt;

    const std::filesystem::path path(argv[2]);
    std::error_code error;
    constexpr std::uintmax_t MaximumPipelineBytes = 16U * 1024U * 1024U;
    if (!std::filesystem::is_regular_file(path, error) || error ||
        std::filesystem::file_size(path, error) > MaximumPipelineBytes || error)
        return std::nullopt;

    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    if (input.bad())
        return std::nullopt;

    HybridPipelineRewrite rewritten;
    bool enableExperimentalReplacements = false;
    const bool automaticR2GroundNormalize =
        std::getenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID") != nullptr;
    const bool requireAutomaticR2 =
        std::getenv("PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE") != nullptr;
    if (automaticR2GroundNormalize && std::getenv("PDG_DEBUG_HYBRID"))
        std::cerr << "gpupal: evaluating automatic r2 hybrid\n";
    if (requireAutomaticR2 && !automaticR2GroundNormalize)
    {
        pdal::Utils::printError(
            "required automatic exact r2 ground-normalization hybrid path "
            "was not selected");
        return 1;
    }
    if (automaticR2GroundNormalize &&
        (!automaticCudaAvailable() ||
         !automaticR2GroundNormalizeDeviceQualified()))
    {
        if (requireAutomaticR2)
        {
            pdal::Utils::printError(
                "required automatic exact r2 ground-normalization CUDA "
                "profile was not available");
            return 1;
        }
        return std::nullopt;
    }
    try
    {
        enableExperimentalReplacements =
            std::getenv("PDG_REQUIRE_HYBRID") ||
            std::getenv("PDG_REQUIRE_STREAMING_HYBRID") ||
            std::getenv("PDG_REQUIRE_CUDA_HYBRID") ||
            std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
            automaticR2GroundNormalize;
        rewritten = rewriteHybridPipeline(text, !metadataFile.empty(),
                                          enableExperimentalReplacements);
    }
    catch (const std::exception& exception)
    {
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupal: hybrid rewrite failed: " << exception.what()
                      << '\n';
        return std::nullopt;
    }
    // The inherited marker is only a routing hint. The engine independently
    // revalidates the complete literal grammar so an externally injected
    // PDG_INTERNAL_* value cannot widen the public selector.
    const bool automaticR2LiteralGrammar =
        automaticR2GroundNormalize && matchesAutomaticR2LiteralGrammar(text);
    const bool automaticR2Selected =
        automaticR2LiteralGrammar && rewritten.replacementRegions == 2U &&
        rewritten.json.find(std::string(HybridSmrfStage)) != std::string::npos &&
        rewritten.json.find(std::string(HybridHagNnStage)) != std::string::npos;
    if (automaticR2GroundNormalize && !automaticR2Selected)
    {
        if (requireAutomaticR2)
            pdal::Utils::printError(
                "required automatic exact r2 ground-normalization stages "
                "were not selected");
        return requireAutomaticR2 ? std::optional<int>{1} : std::nullopt;
    }
    if (automaticR2Selected)
    {
        const std::optional<AutomaticLasInputFacts> facts =
            lasInputFacts(rewritten);
        const bool measuredInput =
            facts && facts->pointCount == 1'000'000U &&
            facts->measuredR4Input;
        if (!measuredInput)
        {
            if (requireAutomaticR2)
                pdal::Utils::printError(
                    "required automatic exact r2 ground-normalization input "
                    "fingerprint did not match");
            return requireAutomaticR2 ? std::optional<int>{1} : std::nullopt;
        }
    }
    if (!rewritten.linearPipeline || rewritten.hasUnstableInputOrderRegion)
        return std::nullopt;

    // Read only the fixed LAS header for count-aware metadata selection. Small
    // jobs delegate without initializing CUDA or preparing the pipeline.
    if (!enableExperimentalReplacements &&
        rewritten.hasPointCountDependentCudaCandidate)
    {
        try
        {
            const std::optional<AutomaticLasInputFacts> facts =
                lasInputFacts(rewritten);
            if (facts)
            {
                HybridPipelineRewrite candidate = rewriteHybridPipeline(
                    text, !metadataFile.empty(), false, facts->pointCount,
                    argc == 3 && facts->measuredR4Input,
                    argc == 3 && facts->measuredLabelNnDistanceInput);
                if (candidate.replacementRegions >
                        rewritten.replacementRegions &&
                    automaticCudaAvailable() &&
                    (!candidate.automaticApproximateCoplanarCuda ||
                     automaticApproximateCoplanarCudaDeviceQualified()) &&
                    (!candidate.automaticR4OutlierCuda ||
                     automaticR4OutlierCudaDeviceQualified()) &&
                    (!candidate.automaticLabelNnDistanceCuda ||
                     automaticLabelNnDistanceCudaDeviceQualified()))
                    rewritten = std::move(candidate);
            }
        }
        catch (const std::exception& exception)
        {
            if (std::getenv("PDG_DEBUG_HYBRID"))
                std::cerr << "gpupal: automatic CUDA count probe failed: "
                          << exception.what() << '\n';
            return std::nullopt;
        }
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA") &&
        !rewritten.automaticApproximateCoplanarCuda)
    {
        pdal::Utils::printError(
            "required automatic exact CUDA hybrid approximatecoplanar path "
            "was not selected");
        return 1;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA") &&
        !rewritten.automaticR4OutlierCuda)
    {
        pdal::Utils::printError(
            "required automatic exact CUDA r4 outlier path was not selected");
        return 1;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID") &&
        !rewritten.automaticLabelNnDistanceCuda)
    {
        pdal::Utils::printError(
            "required automatic exact CUDA label/NNDistance hybrid path "
            "was not selected");
        return 1;
    }
    if (!rewritten.replacementRegions)
        return std::nullopt;

    // Pipeline validation may be the first operation that discovers a dynamic
    // plugin. Capture those PluginManager diagnostics until the hybrid route is
    // committed so a delegated pipeline emits no duplicate output, then replay
    // them in the same position as the upstream application log.
    std::ostringstream pluginDiagnostics;
    pdal::LogPtr validationPluginLog =
        pdal::Log::makeLog("PDAL", &pluginDiagnostics);
    validationPluginLog->setLevel(verbosity.value_or(pdal::LogLevel::Warning));
    pdal::PluginManager<pdal::Stage>::setLog(validationPluginLog);
    pdal::PluginManager<pdal::Kernel>::setLog(validationPluginLog);

    bool originalStreamable = false;
    LayoutSignature originalLayout;
    try
    {
        // Validate the untouched pipeline without emitting warnings. If its
        // options, dimensions, or inputs are invalid, delegate so the pinned
        // CLI owns the exact diagnostic and failure boundary.
        pdal::PipelineManager original;
        original.setLog(pdal::Log::makeLog("PDAL", "devnull"));
        std::istringstream originalPipeline(text);
        original.readPipeline(originalPipeline);
        if (!original.hasReader())
            return std::nullopt;
        original.prepare();
        originalStreamable = original.pipelineStreamable();
        originalLayout = layoutSignature(original.pointTable().layout());
    }
    catch (const std::exception& exception)
    {
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupal: original hybrid validation failed: "
                      << exception.what() << '\n';
        return std::nullopt;
    }

    // Replacing a Streamable filter with a batch GPU filter changes PDAL's
    // execution mode on builds without the bounded batch-stream extension.
    // Retain the coverage flag as a conservative envelope check.
    if (originalStreamable && !rewritten.standardModeRewriteIsExact)
        return std::nullopt;
    if (!originalStreamable &&
        rewritten.standardModeRequiresPointCountValidation)
        return std::nullopt;
    if (std::getenv("PDG_REQUIRE_STREAMING_HYBRID") && !originalStreamable)
        return std::nullopt;
    if (forceStream && !originalStreamable)
        return std::nullopt;

    try
    {
        // Preparation is still side-effect free for point data and output.
        // Use it as the final envelope check so any PDG parser/layout gap
        // delegates before a reader, filter, or writer executes.
        pdal::PipelineManager prepared;
        prepared.setLog(pdal::Log::makeLog("PDAL", "devnull"));
        std::istringstream pipeline(rewritten.json);
        prepared.readPipeline(pipeline);
        if (!prepared.hasReader())
            return std::nullopt;
        prepared.prepare();
        if (rewritten.preparedLayoutOrderObservable &&
            layoutSignature(prepared.pointTable().layout()) != originalLayout)
        {
            if (std::getenv("PDG_DEBUG_HYBRID"))
                std::cerr << "gpupal: rewritten hybrid layout differs from the "
                             "original pipeline\n";
            return std::nullopt;
        }
    }
    catch (const std::exception& exception)
    {
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupal: rewritten hybrid validation failed: "
                      << exception.what() << '\n';
        return std::nullopt;
    }

    try
    {
        constexpr pdal::point_count_t StreamBatchPoints = 131072U;
        pdal::PipelineManager manager(StreamBatchPoints);
        // Match the upstream `pdal pipeline` kernel's observable log leader.
        pdal::LogPtr managerLog = pdal::Log::makeLog("pdal pipeline", "stderr");
        pdal::LogPtr appLog = pdal::Log::makeLog("PDAL", "stderr");
        if (verbosity)
        {
            managerLog->setLevel(*verbosity);
            appLog->setLevel(*verbosity);
        }
        appLog->get(pdal::LogLevel::Debug) << "Debugging..." << std::endl;
        std::cerr << pluginDiagnostics.str();
        pdal::PluginManager<pdal::Stage>::setLog(appLog);
        pdal::PluginManager<pdal::Kernel>::setLog(appLog);
        manager.setLog(managerLog);
        std::istringstream pipeline(rewritten.json);
        manager.readPipeline(pipeline);
        if (!manager.hasReader())
            throw pdal::pdal_error("Pipeline does not start with a reader.");
        pdal::ExecMode mode = originalStreamable ? pdal::ExecMode::PreferStream
                                                 : pdal::ExecMode::Standard;
        if (forceStream)
            mode = pdal::ExecMode::Stream;
        else if (forceStandard)
            mode = pdal::ExecMode::Standard;
        const pdal::PipelineManager::ExecResult result = manager.execute(mode);
        if (result.m_mode == pdal::ExecMode::None)
            throw pdal::pdal_error(
                "Couldn't run pipeline in requested execution mode.");
        if (!metadataFile.empty())
        {
            std::ostream* output = pdal::Utils::createFile(metadataFile, false);
            if (!output)
                throw pdal::pdal_error("Can't open file '" + metadataFile +
                                       "' for metadata output.");
            pdal::Utils::toJSON(manager.getMetadata(), *output);
            pdal::Utils::closeFile(output);
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        pdal::Utils::printError(exception.what());
        return 1;
    }
}

std::optional<int> tryHybridPipeline(int argc, char** argv)
{
    const std::optional<int> result = tryHybridPipelineImpl(argc, argv);
    if (!result &&
        std::getenv("PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA") &&
        argc >= 2 && std::string_view(argv[1]) == "pipeline")
    {
        pdal::Utils::printError(
            "required automatic exact CUDA hybrid approximatecoplanar path "
            "was not used");
        return 1;
    }
    if (!result && std::getenv("PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA") &&
        argc >= 2 && std::string_view(argv[1]) == "pipeline")
    {
        pdal::Utils::printError(
            "required automatic exact CUDA r4 outlier path was not used");
        return 1;
    }
    if (!result &&
        std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID") &&
        argc >= 2 && std::string_view(argv[1]) == "pipeline")
    {
        pdal::Utils::printError(
            "required automatic exact CUDA label/NNDistance hybrid path "
            "was not selected");
        return 1;
    }
    if (!result &&
        std::getenv("PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE") && argc >= 2 &&
        std::string_view(argv[1]) == "pipeline")
    {
        pdal::Utils::printError(
            "required automatic exact r2 ground-normalization hybrid path "
            "was not used");
        return 1;
    }
    return result;
}

} // namespace pdg::cli
