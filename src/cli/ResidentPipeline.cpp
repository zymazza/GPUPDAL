#include "ResidentPipeline.hpp"

#include "../pdal/PdgNeighborhood.hpp"
#include "../pdal/PdgResidentContext.hpp"
#include "DirectResidentLas.hpp"
#include "MappedFile.hpp"
#include "../io/las/LasChunkDecoder.hpp"

#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/LocalProfile.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>
#include <pdg/ResidentPipeline.hpp>
#include <pdg/RuntimePlacement.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <pdal/Log.hpp>
#include <pdal/PDALUtils.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/Stage.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace pdg::cli
{
namespace
{
using Json = nlohmann::json;

constexpr std::size_t ResidentTilePoints = 131072U;
constexpr std::size_t DefaultLasOutputRecordBytes = 36U;
constexpr std::size_t ExtraDoubleLasOutputRecordBytes = 48U;
constexpr std::uintmax_t MaximumPipelineBytes = 16U * 1024U * 1024U;
constexpr std::string_view ResidentValidationBudgetEnvironment =
    "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES";

using ResidentCommandClock = std::chrono::steady_clock;

struct ResidentCommandPhaseSeconds
{
    double commandBeforeStats = 0.0;
    double validationPlacementPreflight = 0.0;
    double planAndOriginalValidation = 0.0;
    double runtimePlacement = 0.0;
    double runtimeDeviceAndProfile = 0.0;
    double runtimeInitialPlacement = 0.0;
    double runtimeExecutorSelection = 0.0;
    double rewriteAndResidentPreflight = 0.0;
    double rewrittenManagerExecution = 0.0;
    double canonicalLasPublication = 0.0;
    double otherControl = 0.0;
};

class DirectResidentPointTable final : public pdal::BasePointTable
{
public:
    explicit DirectResidentPointTable(const pdg::las::FileView& input,
                                      bool includeClassification = false,
                                      bool includeReturnNumber = false)
        : pdal::BasePointTable(m_layout), m_input(input),
          m_encoding(input.header().coordinateEncoding()),
          m_userData(static_cast<std::size_t>(input.header().pointCount)),
          m_classification(includeClassification ? m_userData.size() : 0U),
          m_returnNumber(includeReturnNumber ? m_userData.size() : 0U)
    {
        const pdg::las::Header& header = input.header();
        const std::span<const std::byte> bytes = input.bytes();
        const std::size_t count = m_userData.size();
        if (count >
            std::numeric_limits<std::size_t>::max() / header.pointRecordLength)
            throw std::overflow_error(
                "direct resident LAS record extent overflows size_t");
        const std::size_t recordsBytes = count * header.pointRecordLength;
        if (header.pointDataOffset > bytes.size() ||
            recordsBytes > bytes.size() - header.pointDataOffset ||
            header.pointRecordLength <= 17U)
            throw pdg::las::Error(
                "direct resident LAS byte fields lie outside the input");
        const std::byte* records = bytes.data() + header.pointDataOffset;
        for (std::size_t point = 0; point < count; ++point)
        {
            const std::byte* record =
                records + point * header.pointRecordLength;
            if (!m_classification.empty())
            {
                std::uint8_t classification = static_cast<std::uint8_t>(
                    record[header.pointFormat <= 5U ? 15U : 16U]);
                if (header.pointFormat <= 5U)
                {
                    classification &= 0x1fU;
                    if (classification == 12U)
                        classification = 0U;
                }
                m_classification[point] = classification;
            }
            if (!m_returnNumber.empty())
                m_returnNumber[point] = input.returnNumber(point);
            m_userData[point] = static_cast<std::uint8_t>(record[17U]);
        }
    }

    bool supportsView() const override
    {
        return true;
    }

private:
    pdal::PointId addPoint() override
    {
        if (m_pointCount >= m_userData.size())
            throw pdal::pdal_error(
                "direct resident LAS reader exceeded its point count");
        return m_pointCount++;
    }

    char* getPoint(pdal::PointId) override
    {
        return nullptr;
    }

    void setFieldInternal(pdal::Dimension::Id id, pdal::PointId point,
                          const void* value) override
    {
        if (point >= m_userData.size())
            throw pdal::pdal_error(
                "direct resident LAS field write exceeds its point count");
        if (id == pdal::Dimension::Id::Classification)
        {
            if (m_classification.empty())
                throw pdal::pdal_error(
                    "direct resident LAS Classification was not requested");
            std::memcpy(m_classification.data() + point, value,
                        sizeof(std::uint8_t));
            return;
        }
        if (id == pdal::Dimension::Id::UserData)
        {
            std::memcpy(m_userData.data() + point, value, sizeof(std::uint8_t));
            return;
        }
        if (id == pdal::Dimension::Id::NNDistance)
        {
            if (m_nnDistance.empty())
                m_nnDistance.resize(m_userData.size());
            std::memcpy(m_nnDistance.data() + point, value, sizeof(double));
            return;
        }
        if (id == pdal::Dimension::Id::HeightAboveGround)
        {
            if (m_heightAboveGround.empty())
                m_heightAboveGround.resize(m_userData.size());
            std::memcpy(m_heightAboveGround.data() + point, value,
                        sizeof(double));
            return;
        }
        throw pdal::pdal_error(
            "direct resident LAS source received an unsupported field write");
    }

    void getFieldInternal(pdal::Dimension::Id id, pdal::PointId point,
                          void* value) const override
    {
        if (point >= m_userData.size())
            throw pdal::pdal_error(
                "direct resident LAS field read exceeds its point count");
        if (id == pdal::Dimension::Id::Classification)
        {
            if (m_classification.empty())
                throw pdal::pdal_error(
                    "direct resident LAS Classification was not requested");
            std::memcpy(value, m_classification.data() + point,
                        sizeof(std::uint8_t));
            return;
        }
        if (id == pdal::Dimension::Id::UserData)
        {
            std::memcpy(value, m_userData.data() + point, sizeof(std::uint8_t));
            return;
        }
        if (id == pdal::Dimension::Id::NNDistance)
        {
            const double nnDistance =
                m_nnDistance.empty() ? 0.0 : m_nnDistance[point];
            std::memcpy(value, &nnDistance, sizeof(nnDistance));
            return;
        }
        if (id == pdal::Dimension::Id::HeightAboveGround)
        {
            const double heightAboveGround =
                m_heightAboveGround.empty() ? 0.0 : m_heightAboveGround[point];
            std::memcpy(value, &heightAboveGround, sizeof(heightAboveGround));
            return;
        }
        if (id == pdal::Dimension::Id::ReturnNumber)
        {
            if (m_returnNumber.empty())
                throw pdal::pdal_error(
                    "direct resident LAS ReturnNumber was not requested");
            std::memcpy(value, m_returnNumber.data() + point,
                        sizeof(std::uint8_t));
            return;
        }
        std::size_t axis = 0U;
        if (id == pdal::Dimension::Id::X)
            axis = 0U;
        else if (id == pdal::Dimension::Id::Y)
            axis = 1U;
        else if (id == pdal::Dimension::Id::Z)
            axis = 2U;
        else
            throw pdal::pdal_error("direct resident LAS source received an "
                                   "unsupported field read");
        const double coordinate =
            m_encoding.decode(axis, m_input.rawCoordinate(point, axis));
        std::memcpy(value, &coordinate, sizeof(coordinate));
    }

    char* getDimension(const pdal::Dimension::Detail*, pdal::PointId) override
    {
        return nullptr;
    }

    pdal::PointLayout m_layout;
    const pdg::las::FileView& m_input;
    pdg::CoordinateEncoding m_encoding;
    std::vector<std::uint8_t> m_userData;
    std::vector<std::uint8_t> m_classification;
    std::vector<std::uint8_t> m_returnNumber;
    std::vector<double> m_nnDistance;
    std::vector<double> m_heightAboveGround;
    pdal::PointId m_pointCount = 0;
};

double elapsedSeconds(ResidentCommandClock::time_point started)
{
    return std::chrono::duration<double>(ResidentCommandClock::now() - started)
        .count();
}

double elapsedSeconds(ResidentCommandClock::time_point started,
                      ResidentCommandClock::time_point ended)
{
    return std::chrono::duration<double>(ended - started).count();
}

std::optional<std::string>
withoutTerminalLasWriter(std::string_view pipelineJson,
                         bool removeTerminalResidentSpill = false)
{
    Json document = Json::parse(pipelineJson, nullptr, false);
    if (document.is_discarded())
        return std::nullopt;
    Json* stages = nullptr;
    if (document.is_array())
        stages = &document;
    else if (document.is_object())
    {
        const auto pipeline = document.find("pipeline");
        if (pipeline != document.end() && pipeline->is_array())
            stages = &*pipeline;
    }
    if (!stages || stages->empty() || !stages->back().is_object())
        return std::nullopt;
    const auto type = stages->back().find("type");
    if (type == stages->back().end() || !type->is_string() ||
        type->get<std::string>() != "writers.las")
        return std::nullopt;
    stages->erase(stages->end() - 1);
    if (removeTerminalResidentSpill && !stages->empty() &&
        stages->back().is_object() &&
        stages->back().value("type", "") == pdg::HybridResidentBoundaryStage &&
        stages->back().value("pdg_boundary_kind", "") == "spill")
        stages->erase(stages->end() - 1);
    return document.dump();
}

std::optional<std::string>
withDirectResidentLasSourceReader(std::string_view pipelineJson,
                                  std::size_t pointCount,
                                  bool includeReturnNumber = false)
{
    Json document = Json::parse(pipelineJson, nullptr, false);
    if (document.is_discarded())
        return std::nullopt;
    Json* stages = nullptr;
    if (document.is_array())
        stages = &document;
    else if (document.is_object())
    {
        const auto pipeline = document.find("pipeline");
        if (pipeline != document.end() && pipeline->is_array())
            stages = &*pipeline;
    }
    if (!stages || stages->empty() || !stages->front().is_object() ||
        stages->front().value("type", "") != "readers.las")
        return std::nullopt;
    Json source = Json{{"type", std::string(pdg::HybridResidentLasSourceStage)},
                       {"count", pointCount}};
    if (includeReturnNumber)
        source["return_number"] = true;
    stages->front() = std::move(source);
    return document.dump();
}

bool automaticResidentLasOutputEnvelope(std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 4U)
            return false;
        const Json& reader = position->at(0U);
        const Json& producer = position->at(1U);
        const Json& assign = position->at(2U);
        const Json& writer = position->at(3U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !assign.is_object() ||
            assign.size() != 2U ||
            assign.value("type", "") != "filters.assign" ||
            !assign.contains("value") || !assign.at("value").is_string() ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        const std::string& assignment =
            assign.at("value").get_ref<const std::string&>();
        const bool lofEnvelope =
            producer.is_object() && producer.size() == 2U &&
            producer.value("type", "") == "filters.lof" &&
            producer.contains("minpts") &&
            producer.at("minpts").is_number_integer() &&
            producer.at("minpts").get<std::int64_t>() == 10 &&
            assignment == "UserData = 1 WHERE LocalOutlierFactor >= 1.2";
        const bool nnDistanceEnvelope =
            producer.is_object() && producer.size() == 2U &&
            producer.value("type", "") == "filters.nndistance" &&
            producer.contains("k") && producer.at("k").is_number_integer() &&
            producer.at("k").get<std::int64_t>() == 10 &&
            assignment == "UserData = 1 WHERE NNDistance >= 0.4";
        return lofEnvelope || nnDistanceEnvelope;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool directRadiusAssignEnvelope(std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& radius = position->at(1U);
        const Json& writer = position->at(2U);
        if (!(reader.is_object() && reader.size() == 2U &&
              reader.value("type", "") == "readers.las" &&
              reader.contains("filename") &&
              reader.at("filename").is_string() && radius.is_object() &&
              radius.size() == 6U &&
              radius.value("type", "") == "filters.radiusassign" &&
              radius.contains("radius") && radius.at("radius").is_number() &&
              radius.at("radius").get<double>() == 2.0 &&
              radius.value("src_domain", "") == "ReturnNumber[1:1]" &&
              radius.value("reference_domain", "") == "ReturnNumber[2:15]" &&
              radius.contains("is3d") && radius.at("is3d").is_boolean() &&
              radius.at("is3d").get<bool>() &&
              radius.value("update_expression", "") == "UserData = 9" &&
              writer.is_object() && writer.size() == 2U &&
              writer.value("type", "") == "writers.las" &&
              writer.contains("filename") && writer.at("filename").is_string()))
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalDirectSkewnessEnvelope(std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& skewness = position->at(1U);
        const Json& writer = position->at(2U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !skewness.is_object() ||
            skewness.size() != 1U ||
            skewness.value("type", "") != "filters.skewnessbalancing" ||
            !writer.is_object() || writer.size() != 3U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") ||
            !writer.at("filename").is_string() ||
            !writer.contains("extra_dims") ||
            !writer.at("extra_dims").is_string() ||
            writer.at("extra_dims").get<std::string>() != "all")
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticDirectSkewnessEnvelope(std::string_view pipelineJson) noexcept
{
    if (!experimentalDirectSkewnessEnvelope(pipelineJson))
        return false;
    try
    {
        const Json document = Json::parse(pipelineJson);
        const Json& reader = document.at("pipeline").at(0U);
        const Json& writer = document.at("pipeline").at(2U);
        return std::filesystem::path(
                   reader.at("filename").get_ref<const std::string&>())
                       .extension() == ".las" &&
               std::filesystem::path(
                   writer.at("filename").get_ref<const std::string&>())
                       .extension() == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalDirectSortEnvelope(std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& sort = position->at(1U);
        const Json& writer = position->at(2U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !sort.is_object() ||
            sort.size() != 4U || sort.value("type", "") != "filters.sort" ||
            sort.value("dimension", "") != "Z" ||
            sort.value("order", "") != "ASC" ||
            sort.value("algorithm", "") != "NORMAL" || !writer.is_object() ||
            writer.size() != 3U || writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") ||
            !writer.at("filename").is_string() ||
            writer.value("extra_dims", "") != "all")
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticResidentNnDistanceEnvelope(std::string_view pipelineJson) noexcept
{
    try
    {
        if (!automaticResidentLasOutputEnvelope(pipelineJson))
            return false;
        const Json document = Json::parse(pipelineJson);
        const Json& producer = document.at("pipeline").at(1U);
        return producer.value("type", "") == "filters.nndistance";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticResidentEigenFamilyEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 6U)
            return false;
        const Json& reader = position->at(0U);
        const Json& normal = position->at(1U);
        const Json& eigenvalues = position->at(2U);
        const Json& covariance = position->at(3U);
        const Json& assign = position->at(4U);
        const Json& writer = position->at(5U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !normal.is_object() ||
            normal.size() != 3U ||
            normal.value("type", "") != "filters.normal" ||
            !normal.contains("knn") || !normal.at("knn").is_number_integer() ||
            normal.at("knn").get<std::int64_t>() != 12 ||
            !normal.contains("always_up") ||
            !normal.at("always_up").is_boolean() ||
            normal.at("always_up").get<bool>() || !eigenvalues.is_object() ||
            eigenvalues.size() != 3U ||
            eigenvalues.value("type", "") != "filters.eigenvalues" ||
            !eigenvalues.contains("knn") ||
            !eigenvalues.at("knn").is_number_integer() ||
            eigenvalues.at("knn").get<std::int64_t>() != 12 ||
            !eigenvalues.contains("normalize") ||
            !eigenvalues.at("normalize").is_boolean() ||
            !eigenvalues.at("normalize").get<bool>() ||
            !covariance.is_object() || covariance.size() != 4U ||
            covariance.value("type", "") != "filters.covariancefeatures" ||
            !covariance.contains("knn") ||
            !covariance.at("knn").is_number_integer() ||
            covariance.at("knn").get<std::int64_t>() != 12 ||
            covariance.value("mode", "") != "raw" ||
            covariance.value("feature_set", "") != "dimensionality" ||
            !assign.is_object() || assign.size() != 2U ||
            assign.value("type", "") != "filters.assign" ||
            !assign.contains("value") || !assign.at("value").is_array() ||
            assign.at("value").size() != 3U ||
            !std::all_of(assign.at("value").begin(), assign.at("value").end(),
                         [](const Json& value) { return value.is_string(); }) ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool selectedAutomaticEigenFamilyPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model == "eigen-family-compose";
}

bool selectedAutomaticNormalCovariancePlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           (runtime.regionCalibrations.front().model ==
                "normal-covariancefeatures-compose" ||
            runtime.regionCalibrations.front().model ==
                "normal-covariancefeatures-compose-extradims");
}

bool automaticResidentRankOptimalEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 5U)
            return false;
        const Json& reader = position->at(0U);
        const Json& estimate = position->at(1U);
        const Json& optimal = position->at(2U);
        const Json& assign = position->at(3U);
        const Json& writer = position->at(4U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !estimate.is_object() ||
            estimate.size() != 3U ||
            estimate.value("type", "") != "filters.estimaterank" ||
            !estimate.contains("knn") ||
            !estimate.at("knn").is_number_integer() ||
            estimate.at("knn").get<std::int64_t>() != 14 ||
            !estimate.contains("thresh") ||
            !estimate.at("thresh").is_number() ||
            estimate.at("thresh").get<double>() != 0.01 ||
            !optimal.is_object() || optimal.size() != 3U ||
            optimal.value("type", "") != "filters.optimalneighborhood" ||
            !optimal.contains("min_k") ||
            !optimal.at("min_k").is_number_integer() ||
            optimal.at("min_k").get<std::int64_t>() != 10 ||
            !optimal.contains("max_k") ||
            !optimal.at("max_k").is_number_integer() ||
            optimal.at("max_k").get<std::int64_t>() != 14 ||
            !assign.is_object() || assign.size() != 2U ||
            assign.value("type", "") != "filters.assign" ||
            !assign.contains("value") || !assign.at("value").is_array() ||
            assign.at("value").size() != 3U ||
            !std::all_of(assign.at("value").begin(), assign.at("value").end(),
                         [](const Json& value) { return value.is_string(); }) ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool selectedAutomaticRankOptimalPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model == "rank-optimal-compose";
}

bool experimentalResidentOutlierNnDistanceEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 4U)
            return false;
        const Json& reader = position->at(0U);
        const Json& outlier = position->at(1U);
        const Json& nnDistance = position->at(2U);
        const Json& writer = position->at(3U);
        if (!(reader.is_object() && reader.size() == 2U &&
              reader.value("type", "") == "readers.las" &&
              reader.contains("filename") &&
              reader.at("filename").is_string() && outlier.is_object() &&
              outlier.size() == 5U &&
              outlier.value("type", "") == "filters.outlier" &&
              outlier.value("method", "") == "statistical" &&
              outlier.contains("mean_k") &&
              outlier.at("mean_k").is_number_integer() &&
              outlier.at("mean_k").get<std::int64_t>() == 8 &&
              outlier.contains("multiplier") &&
              outlier.at("multiplier").is_number() &&
              outlier.at("multiplier").get<double>() == 2.0 &&
              outlier.contains("class") &&
              outlier.at("class").is_number_integer() &&
              outlier.at("class").get<std::int64_t>() == 7 &&
              nnDistance.is_object() && nnDistance.size() == 3U &&
              nnDistance.value("type", "") == "filters.nndistance" &&
              nnDistance.value("mode", "") == "kth" &&
              nnDistance.contains("k") &&
              nnDistance.at("k").is_number_integer() &&
              nnDistance.at("k").get<std::int64_t>() == 10 &&
              writer.is_object() && writer.size() == 2U &&
              writer.value("type", "") == "writers.las" &&
              writer.contains("filename") && writer.at("filename").is_string()))
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalResidentStandaloneOutlierEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& outlier = position->at(1U);
        const Json& writer = position->at(2U);
        const bool readerMatches = reader.is_object() && reader.size() == 2U &&
                                   reader.value("type", "") == "readers.las" &&
                                   reader.contains("filename") &&
                                   reader.at("filename").is_string();
        const bool statisticalMatches =
            outlier.is_object() && outlier.size() == 5U &&
            outlier.value("type", "") == "filters.outlier" &&
            outlier.value("method", "") == "statistical" &&
            outlier.contains("mean_k") &&
            outlier.at("mean_k").is_number_integer() &&
            outlier.at("mean_k").get<std::int64_t>() == 8 &&
            outlier.contains("multiplier") &&
            outlier.at("multiplier").is_number() &&
            outlier.at("multiplier").get<double>() == 2.0 &&
            outlier.contains("class") &&
            outlier.at("class").is_number_integer() &&
            outlier.at("class").get<std::int64_t>() == 7;
        const bool radiusMatches =
            outlier.is_object() && outlier.size() == 4U &&
            outlier.value("type", "") == "filters.outlier" &&
            outlier.value("method", "") == "radius" &&
            outlier.contains("radius") && outlier.at("radius").is_number() &&
            outlier.at("radius").get<double>() == 1.0 &&
            outlier.contains("min_k") &&
            outlier.at("min_k").is_number_integer() &&
            outlier.at("min_k").get<std::int64_t>() == 2;
        const bool writerMatches = writer.is_object() && writer.size() == 2U &&
                                   writer.value("type", "") == "writers.las" &&
                                   writer.contains("filename") &&
                                   writer.at("filename").is_string();
        if (!readerMatches || (!statisticalMatches && !radiusMatches) ||
            !writerMatches)
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalResidentNeighborClassifierEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& classifier = position->at(1U);
        const Json& writer = position->at(2U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !classifier.is_object() ||
            classifier.size() != 2U ||
            classifier.value("type", "") != "filters.neighborclassifier" ||
            !classifier.contains("k") ||
            !classifier.at("k").is_number_integer() ||
            classifier.at("k").get<std::int64_t>() != 7 ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalResidentRadialDensityAssignEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 4U)
            return false;
        const Json& reader = position->at(0U);
        const Json& density = position->at(1U);
        const Json& assign = position->at(2U);
        const Json& writer = position->at(3U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !density.is_object() ||
            density.size() != 2U ||
            density.value("type", "") != "filters.radialdensity" ||
            !density.contains("radius") || !density.at("radius").is_number() ||
            density.at("radius").get<double>() != 1.01 || !assign.is_object() ||
            assign.size() != 2U ||
            assign.value("type", "") != "filters.assign" ||
            !assign.contains("value") || !assign.at("value").is_string() ||
            assign.at("value").get_ref<const std::string&>() !=
                "UserData = 1 WHERE RadialDensity >= 0.2" ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalResidentRadiusOutlierRadialDensityAssignEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 5U)
            return false;
        const Json& reader = position->at(0U);
        const Json& outlier = position->at(1U);
        const Json& density = position->at(2U);
        const Json& assign = position->at(3U);
        const Json& writer = position->at(4U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !outlier.is_object() ||
            outlier.size() != 5U ||
            outlier.value("type", "") != "filters.outlier" ||
            outlier.value("method", "") != "radius" ||
            !outlier.contains("radius") || !outlier.at("radius").is_number() ||
            outlier.at("radius").get<double>() != 1.01 ||
            !outlier.contains("min_k") ||
            !outlier.at("min_k").is_number_integer() ||
            outlier.at("min_k").get<std::int64_t>() != 2 ||
            !outlier.contains("class") ||
            !outlier.at("class").is_number_integer() ||
            outlier.at("class").get<std::int64_t>() != 7 ||
            !density.is_object() || density.size() != 2U ||
            density.value("type", "") != "filters.radialdensity" ||
            !density.contains("radius") || !density.at("radius").is_number() ||
            density.at("radius").get<double>() != 1.01 || !assign.is_object() ||
            assign.size() != 2U ||
            assign.value("type", "") != "filters.assign" ||
            !assign.contains("value") || !assign.at("value").is_string() ||
            assign.at("value").get_ref<const std::string&>() !=
                "UserData = 1 WHERE RadialDensity >= 0.2" ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool experimentalResidentExtraDoubleOutputEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 3U)
            return false;
        const Json& reader = position->at(0U);
        const Json& producer = position->at(1U);
        const Json& writer = position->at(2U);
        const bool supportedHagNnCount =
            producer.is_object() && producer.size() == 2U &&
            producer.value("type", "") == "filters.hag_nn" &&
            producer.contains("count") &&
            producer.at("count").is_number_unsigned() &&
            producer.at("count").get<std::uint64_t>() >= 1U &&
            producer.at("count").get<std::uint64_t>() <= 64U;
        const bool hagDelaunayCountThree =
            producer.is_object() && producer.size() == 2U &&
            producer.value("type", "") == "filters.hag_delaunay" &&
            producer.contains("count") &&
            producer.at("count").is_number_unsigned() &&
            producer.at("count").get<std::uint64_t>() == 3U;
        const bool nnDistanceKthTen =
            producer.is_object() && producer.size() == 3U &&
            producer.value("type", "") == "filters.nndistance" &&
            producer.contains("mode") && producer.at("mode").is_string() &&
            producer.at("mode").get_ref<const std::string&>() == "kth" &&
            producer.contains("k") && producer.at("k").is_number_unsigned() &&
            producer.at("k").get<std::uint64_t>() == 10U;
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() ||
            (!supportedHagNnCount && !hagDelaunayCountThree &&
             !nnDistanceKthTen) ||
            !writer.is_object() || writer.size() != 3U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") ||
            !writer.at("filename").is_string() ||
            !writer.contains("extra_dims") ||
            !writer.at("extra_dims").is_string() ||
            writer.at("extra_dims").get_ref<const std::string&>() != "all")
            return false;
        const auto lasExtension = [](const Json& stage)
        {
            std::string extension =
                std::filesystem::path(
                    stage.at("filename").get_ref<const std::string&>())
                    .extension()
                    .string();
            std::transform(extension.begin(), extension.end(),
                           extension.begin(), [](unsigned char value)
                           { return static_cast<char>(std::tolower(value)); });
            return extension == ".las";
        };
        return lasExtension(reader) && lasExtension(writer);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticDirectHagNnCountOneEnvelope(
    std::string_view pipelineJson) noexcept
{
    if (!experimentalResidentExtraDoubleOutputEnvelope(pipelineJson))
        return false;
    try
    {
        const Json document = Json::parse(pipelineJson);
        const Json& reader = document.at("pipeline").at(0U);
        const Json& hagNn = document.at("pipeline").at(1U);
        const Json& writer = document.at("pipeline").at(2U);
        return hagNn.at("type").get_ref<const std::string&>() ==
                   "filters.hag_nn" &&
               hagNn.at("count").is_number_unsigned() &&
               hagNn.at("count").get<std::uint64_t>() == 1U &&
               std::filesystem::path(
                   reader.at("filename").get_ref<const std::string&>())
                       .extension() == ".las" &&
               std::filesystem::path(
                   writer.at("filename").get_ref<const std::string&>())
                       .extension() == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticDirectHagDelaunayCountThreeEnvelope(
    std::string_view pipelineJson) noexcept
{
    if (!experimentalResidentExtraDoubleOutputEnvelope(pipelineJson))
        return false;
    try
    {
        const Json document = Json::parse(pipelineJson);
        const Json& reader = document.at("pipeline").at(0U);
        const Json& hagDelaunay = document.at("pipeline").at(1U);
        const Json& writer = document.at("pipeline").at(2U);
        return hagDelaunay.at("type").get_ref<const std::string&>() ==
                   "filters.hag_delaunay" &&
               hagDelaunay.at("count").is_number_unsigned() &&
               hagDelaunay.at("count").get<std::uint64_t>() == 3U &&
               std::filesystem::path(
                   reader.at("filename").get_ref<const std::string&>())
                       .extension() == ".las" &&
               std::filesystem::path(
                   writer.at("filename").get_ref<const std::string&>())
                       .extension() == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticResidentApproximateCoplanarEnvelope(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json document = Json::parse(pipelineJson, nullptr, false);
        if (document.is_discarded() || !document.is_object() ||
            document.size() != 1U)
            return false;
        const auto position = document.find("pipeline");
        if (position == document.end() || !position->is_array() ||
            position->size() != 4U)
            return false;
        const Json& reader = position->at(0U);
        const Json& approximate = position->at(1U);
        const Json& ferry = position->at(2U);
        const Json& writer = position->at(3U);
        if (!reader.is_object() || reader.size() != 2U ||
            reader.value("type", "") != "readers.las" ||
            !reader.contains("filename") ||
            !reader.at("filename").is_string() || !approximate.is_object() ||
            approximate.size() != 2U ||
            approximate.value("type", "") != "filters.approximatecoplanar" ||
            !approximate.contains("knn") ||
            !approximate.at("knn").is_number_integer() ||
            approximate.at("knn").get<std::int64_t>() != 8 ||
            !ferry.is_object() || ferry.size() != 2U ||
            ferry.value("type", "") != "filters.ferry" ||
            !ferry.contains("dimensions") ||
            !ferry.at("dimensions").is_string() ||
            ferry.at("dimensions").get_ref<const std::string&>() !=
                "Coplanar=>UserData" ||
            !writer.is_object() || writer.size() != 2U ||
            writer.value("type", "") != "writers.las" ||
            !writer.contains("filename") || !writer.at("filename").is_string())
            return false;
        std::string extension =
            std::filesystem::path(
                writer.at("filename").get_ref<const std::string&>())
                .extension()
                .string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char value)
                       { return static_cast<char>(std::tolower(value)); });
        return extension == ".las";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool selectedAutomaticOutlierNnDistancePlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "outlier-nndistance-direct-compose";
}

bool selectedAutomaticRadiusOutlierRadialDensityPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "radius-outlier-radialdensity-direct-compose";
}

bool selectedAutomaticRadiusAssignPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model == "radiusassign-direct";
}

bool selectedAutomaticNeighborClassifierPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "neighborclassifier-direct-compose";
}

bool selectedAutomaticHagNnCountOnePlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "hag-nn-count1-direct-compose";
}

bool selectedAutomaticHagDelaunayCountThreePlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "hag-delaunay-count3-direct-compose";
}

bool selectedAutomaticSkewnessPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "skewness-direct-compose";
}

bool selectedAutomaticSortPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model == "sort-direct-compose";
}

bool selectedAutomaticApproximateCoplanarPlacement(
    const RuntimePlacementResult& runtime) noexcept
{
    return runtime.available() &&
           runtime.estimate.choice == PlacementChoice::Device &&
           runtime.estimate.selectedRegionCount == 1U &&
           runtime.regionCalibrations.size() == 1U &&
           runtime.regionCalibrations.front().model ==
               "approximatecoplanar-direct-compose";
}

bool directApproximateCoplanarComposition(const Plan& plan) noexcept
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 4U || plan.summary().residentRegions != 1U ||
        stages.front().descriptor.type != "readers.las" ||
        stages.back().descriptor.type != "writers.las")
        return false;
    const auto* approximate =
        std::get_if<ApproximateCoplanarProgram>(&stages[1].payload);
    const auto* ferry = std::get_if<FerryProgram>(&stages[2].payload);
    if (!approximate || !ferry || approximate->neighbors != 8 ||
        approximate->threshold1 != 25.0 || approximate->threshold2 != 6.0 ||
        ferry->copies.size() != 1U)
        return false;
    const FerryCopy& copy = ferry->copies.front();
    return copy.hasSource && !copy.destinationCreated &&
           copy.source == DimensionId(StandardDimension::Coplanar) &&
           copy.destination == DimensionId(StandardDimension::UserData);
}

// Lowercased file extension, used to tell `.las` from `.laz` for both
// endpoints without pulling in the planner's copy of the same rule.
std::string lowercaseExtension(std::string_view filename)
{
    std::string extension =
        std::filesystem::path(filename).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value)
                   { return static_cast<char>(std::tolower(value)); });
    return extension;
}

struct LasHeaderFacts
{
    std::size_t pointCount = 0;
    std::uint8_t pointFormat = 0U;
    std::size_t recordBytes = 0;
    // B0151/D0213: a LAZ reader has a structurally identical public header, so
    // placement can use its facts, but its point records are chunk-compressed
    // and must never be memory-mapped. Every direct/mapped-source route checks
    // this flag before claiming the raw records.
    bool compressedReader = false;
    // Likewise for the sink: PDAL's writer compresses on write, so the
    // *record* size placement budgets is the uncompressed one either way. Only
    // the direct publisher, which emits raw LAS records itself, needs this.
    bool compressedWriter = false;
};

bool planDeclaresNeighborhoodRegion(const Plan& plan);
bool planDeclaresGridRegion(const Plan& plan);

std::optional<std::size_t>
residentValidationBudgetOverride(std::size_t defaultBudgetBytes)
{
    const char* configured =
        std::getenv(ResidentValidationBudgetEnvironment.data());
    if (!configured)
        return std::nullopt;
    const std::string_view text(configured);
    std::size_t value = 0U;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value ||
        value > defaultBudgetBytes)
        throw std::invalid_argument(
            std::string(ResidentValidationBudgetEnvironment) +
            " must be a positive integer no greater than the default "
            "resident VRAM budget");
    return value;
}

template <typename T>
T readLittleEndian(const std::array<char, 375>& bytes, std::size_t offset)
{
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

std::optional<LasHeaderFacts> lasHeaderFacts(const Plan& plan,
                                             bool allowExtraDimensionsAll)
{
    const FileStagePlan* writer =
        plan.stages().empty()
            ? nullptr
            : std::get_if<FileStagePlan>(&plan.stages().back().payload);
    // The reader need only have a readable LAS-family public header: a LAZ
    // file's header carries the same point count, format, and record length,
    // and those are all placement consumes. Record access stays gated on
    // `compressedReader` below.
    const auto* readerPlan =
        plan.stages().empty()
            ? nullptr
            : std::get_if<FileStagePlan>(&plan.stages().front().payload);
    const bool readerIsLasFamily =
        !plan.stages().empty() &&
        plan.stages().front().role == StageRole::Reader &&
        plan.stages().front().descriptor.type == "readers.las" && readerPlan &&
        readerPlan->optionFreeLasFamily;
    if (plan.stages().empty() ||
        plan.stages().front().role != StageRole::Reader ||
        plan.stages().front().descriptor.type != "readers.las" ||
        !readerIsLasFamily || plan.stages().back().role != StageRole::Writer ||
        plan.stages().back().descriptor.type != "writers.las" ||
        ((!writer || !writer->optionFreeLasFamily) &&
         (!allowExtraDimensionsAll ||
          (!writer || !writer->extraDimensionsAll))))
        return std::nullopt;
    const auto* reader =
        std::get_if<FileStagePlan>(&plan.stages().front().payload);
    if (!reader || reader->filename.empty())
        return std::nullopt;

    const auto writerExtension =
        writer ? lowercaseExtension(writer->filename) : std::string();
    const bool compressedWriter = writerExtension == ".laz";

    const std::filesystem::path path(reader->filename);
    const std::string extension = lowercaseExtension(reader->filename);
    const bool compressedReader = extension == ".laz";
    if (extension != ".las" && !compressedReader)
        return std::nullopt;

    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    std::array<char, 375> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::streamsize count = input.gcount();
    if (count < 227 || std::memcmp(bytes.data(), "LASF", 4) != 0)
        return std::nullopt;
    const std::uint8_t major = readLittleEndian<std::uint8_t>(bytes, 24U);
    const std::uint8_t minor = readLittleEndian<std::uint8_t>(bytes, 25U);
    if (major != 1U || minor > 4U)
        return std::nullopt;
    std::uint64_t points = readLittleEndian<std::uint32_t>(bytes, 107U);
    if (minor >= 4U)
    {
        if (count < static_cast<std::streamsize>(bytes.size()))
            return std::nullopt;
        points = readLittleEndian<std::uint64_t>(bytes, 247U);
    }
    if (points > (std::numeric_limits<std::size_t>::max)())
        return std::nullopt;
    const std::uint8_t rawPointFormat =
        readLittleEndian<std::uint8_t>(bytes, 104U);
    // Bit 7 is the LAZ compression flag. It must agree with the extension:
    // a mismatch means the file is not what it claims and placement must not
    // guess.
    if (((rawPointFormat & 0x80U) != 0U) != compressedReader)
        return std::nullopt;
    const std::uint8_t pointFormat = rawPointFormat & 0x3fU;
    const std::uint16_t recordBytes =
        readLittleEndian<std::uint16_t>(bytes, 105U);
    if (!recordBytes)
        return std::nullopt;
    return LasHeaderFacts{static_cast<std::size_t>(points), pointFormat,
                          recordBytes, compressedReader, compressedWriter};
}

void validateAutomaticLasPublication(const Plan& plan,
                                     std::size_t expectedPointCount)
{
    if (plan.stages().empty())
        throw pdal::pdal_error(
            "automatic resident LAS publication has no terminal stage");
    const auto* writer =
        std::get_if<FileStagePlan>(&plan.stages().back().payload);
    if (!writer || writer->filename.empty())
        throw pdal::pdal_error(
            "automatic resident LAS publication has no output path");

    const std::filesystem::path outputPath(writer->filename);
    // A deterministic process test truncates the just-published artifact at
    // this boundary. It models the failure observed when PDAL's unchanged
    // writer exhausted a tmpfs, left an unfinalized header, and nevertheless
    // returned success. The hook is internal and dispatches only to the engine.
    if (std::getenv(
            "PDG_TEST_AUTOMATIC_RESIDENT_PUBLICATION_TRUNCATION"))
    {
        std::error_code error;
        const std::uintmax_t bytes = std::filesystem::file_size(outputPath, error);
        if (error)
            throw std::system_error(
                error, "unable to inspect resident publication for truncation");
        std::ifstream header(outputPath, std::ios::binary);
        std::array<char, 375> prefix{};
        header.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        if (header.gcount() != static_cast<std::streamsize>(prefix.size()))
            throw pdal::pdal_error(
                "resident publication is too short for truncation injection");
        const std::uint32_t pointDataOffset =
            readLittleEndian<std::uint32_t>(prefix, 96U);
        if (pointDataOffset >= bytes || bytes - pointDataOffset < 2U)
            throw pdal::pdal_error(
                "resident publication has no payload for truncation injection");
        const bool compressed =
            (static_cast<unsigned char>(prefix[104U]) & 0x80U) != 0U;
        if (!compressed)
        {
            // Model the observed unfinalized plain-LAS header: the public
            // count remains zero while a partial raw payload is present.
            std::fill(prefix.begin() + 107U, prefix.begin() + 111U, 0);
            if (static_cast<unsigned char>(prefix[25U]) >= 4U)
                std::fill(prefix.begin() + 247U, prefix.begin() + 255U, 0);
            std::fstream output(outputPath,
                                std::ios::binary | std::ios::in | std::ios::out);
            output.write(prefix.data(),
                         static_cast<std::streamsize>(prefix.size()));
            output.flush();
            if (!output)
                throw pdal::pdal_error(
                    "unable to inject stale resident publication count");
        }
        const std::uintmax_t truncated =
            pointDataOffset + (bytes - pointDataOffset) / 2U;
        std::filesystem::resize_file(outputPath, truncated, error);
        if (error)
            throw std::system_error(
                error, "unable to inject resident publication truncation");
    }

    try
    {
        const MappedInput mapping(outputPath);
        const pdg::las::FileView output(mapping.bytes());
        const bool compressed = lowercaseExtension(writer->filename) == ".laz";
        if (output.header().compressed != compressed)
            throw pdg::las::Error(
                "published LAS compression does not match its filename");
        if (output.header().pointCount != expectedPointCount)
            throw pdg::las::Error(
                "published LAS point count does not match resident output");
        // FileView proves the complete uncompressed record extent and all
        // public header/VLR/EVLR extents. For LAZ, validate the LASzip chunk
        // table's declared point coverage and exact compressed-byte extent in
        // O(chunks). A retained header plus a partial payload is corrupt, not
        // a successful publication.
        if (output.header().compressed)
            pdg::las::validateCompressedPointRecords(outputPath, output);
    }
    catch (const std::exception& exception)
    {
        throw pdal::pdal_error(
            std::string("automatic resident LAS publication is invalid: ") +
            exception.what());
    }
}

const char* placementChoiceName(PlacementChoice choice) noexcept
{
    return choice == PlacementChoice::Device ? "device" : "host";
}

const char* placementReasonName(PlacementReason reason) noexcept
{
    switch (reason)
    {
    case PlacementReason::DeviceFaster:
        return "device_faster";
    case PlacementReason::HostFasterOrEqual:
        return "host_faster_or_equal";
    case PlacementReason::DeviceMemoryBudgetExceeded:
        return "device_memory_budget_exceeded";
    case PlacementReason::UncalibratedStage:
        return "uncalibrated_stage";
    case PlacementReason::OutsideCalibrationEnvelope:
        return "outside_calibration_envelope";
    case PlacementReason::MissingRecordLayout:
        return "missing_record_layout";
    case PlacementReason::SharedDeviceTollNotAmortized:
        return "shared_device_toll_not_amortized";
    case PlacementReason::UnsupportedPlanTopology:
        return "unsupported_plan_topology";
    case PlacementReason::NoDeviceStages:
        return "no_device_stages";
    }
    return "unknown";
}

const char*
unavailableReasonName(RuntimePlacementUnavailableReason reason) noexcept
{
    switch (reason)
    {
    case RuntimePlacementUnavailableReason::None:
        return "none";
    case RuntimePlacementUnavailableReason::ProfileNotExact:
        return "profile_not_exact";
    case RuntimePlacementUnavailableReason::InvalidRuntimeFacts:
        return "invalid_runtime_facts";
    case RuntimePlacementUnavailableReason::UnsupportedTopology:
        return "unsupported_topology";
    case RuntimePlacementUnavailableReason::UnsupportedStage:
        return "unsupported_stage";
    case RuntimePlacementUnavailableReason::NonCardinalityPreservingStage:
        return "non_cardinality_preserving_stage";
    case RuntimePlacementUnavailableReason::MissingCalibrationModel:
        return "missing_calibration_model";
    case RuntimePlacementUnavailableReason::MixedCalibrationModels:
        return "mixed_calibration_models";
    case RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope:
        return "outside_calibration_envelope";
    case RuntimePlacementUnavailableReason::UnknownCalibrationModel:
        return "unknown_calibration_model";
    case RuntimePlacementUnavailableReason::NoDeviceCandidate:
        return "no_device_candidate";
    case RuntimePlacementUnavailableReason::EvaluationFailed:
        return "evaluation_failed";
    }
    return "unknown";
}

const char* eventKindName(ExecutionEventKind kind) noexcept
{
    switch (kind)
    {
    case ExecutionEventKind::DeviceRegionBegin:
        return "device_region_begin";
    case ExecutionEventKind::DeviceRegionEnd:
        return "device_region_end";
    case ExecutionEventKind::HostToDevice:
        return "host_to_device";
    case ExecutionEventKind::DeviceToHost:
        return "device_to_host";
    case ExecutionEventKind::BoundaryUpload:
        return "boundary_upload";
    case ExecutionEventKind::BoundarySpill:
        return "boundary_spill";
    case ExecutionEventKind::FallbackBoundary:
        return "fallback_boundary";
    case ExecutionEventKind::IndexBuild:
        return "index_build";
    case ExecutionEventKind::IndexRebuild:
        return "index_rebuild";
    case ExecutionEventKind::GridBuild:
        return "grid_build";
    case ExecutionEventKind::RasterBuild:
        return "raster_build";
    case ExecutionEventKind::RasterUpload:
        return "raster_upload";
    case ExecutionEventKind::RasterDownload:
        return "raster_download";
    case ExecutionEventKind::Count:
        break;
    }
    return "unknown";
}

Json breakdownJson(const PlacementCostBreakdown& value)
{
    return {{"stage_ns", value.stageNanoseconds},
            {"startup_ns", value.startupNanoseconds},
            {"transfer_ns", value.transferNanoseconds},
            {"packing_ns", value.packingNanoseconds},
            {"index_build_ns", value.indexBuildNanoseconds},
            {"synchronization_ns", value.synchronizationNanoseconds},
            {"total_ns", value.totalNanoseconds}};
}

bool regionSelected(const PlanPlacementEstimate& estimate,
                    std::size_t regionId) noexcept
{
    return std::any_of(
        estimate.regions.begin(), estimate.regions.end(),
        [&](const PlacementRegionEstimate& region)
        { return region.residentRegion == regionId && region.selected; });
}

bool residentPointViewExecutorSupports(std::string_view pipelineJson,
                                       const Plan& plan)
{
    // The PointView row facts describe the tile executor; a shared-index
    // neighborhood plan executes through the whole-view attach machinery and
    // keeps calibration-default accounting.
    if (planDeclaresNeighborhoodRegion(plan) || planDeclaresGridRegion(plan))
        return false;
    PlanPlacementEstimate allRegions;
    allRegions.choice = PlacementChoice::Device;
    for (std::size_t regionId = 0; regionId < plan.summary().residentRegions;
         ++regionId)
    {
        PlacementRegionEstimate region;
        region.residentRegion = regionId;
        region.selected = true;
        for (const PlannedStage& stage : plan.stages())
            if (stage.residentRegion == regionId)
                region.stageIds.push_back(stage.id);
        allRegions.regions.push_back(std::move(region));
    }
    allRegions.selectedRegionCount = allRegions.regions.size();
    return !allRegions.regions.empty() &&
           rewriteResidentPlacement(pipelineJson, plan, allRegions).executable;
}

bool regionDeclaresCardinalityChange(const Plan& plan, std::size_t region)
{
    return region != NoResidentRegion &&
           std::any_of(
               plan.stages().begin(), plan.stages().end(),
               [&](const PlannedStage& stage)
               {
                   return stage.residentRegion == region &&
                          !stage.descriptor.fusion.cardinalityPreserving;
               });
}

bool planDeclaresResidentCardinalityChange(const Plan& plan)
{
    return std::any_of(
        plan.stages().begin(), plan.stages().end(),
        [](const PlannedStage& stage)
        {
            return stage.residentRegion != NoResidentRegion &&
                   !stage.descriptor.fusion.cardinalityPreserving;
        });
}

bool planDeclaresNeighborhoodRegion(const Plan& plan)
{
    return std::any_of(
        plan.stages().begin(), plan.stages().end(),
        [](const PlannedStage& stage)
        {
            return stage.residentRegion != NoResidentRegion &&
                   (std::holds_alternative<LabelDuplicatesProgram>(
                        stage.payload) ||
                    std::holds_alternative<ApproximateCoplanarProgram>(
                        stage.payload) ||
                    std::holds_alternative<LofProgram>(stage.payload) ||
                    std::holds_alternative<NnDistanceProgram>(stage.payload) ||
                    std::holds_alternative<OutlierProgram>(stage.payload) ||
                    std::holds_alternative<NormalProgram>(stage.payload) ||
                    std::holds_alternative<EigenvaluesProgram>(stage.payload) ||
                    std::holds_alternative<CovarianceFeaturesProgram>(
                        stage.payload) ||
                    std::holds_alternative<EstimateRankProgram>(
                        stage.payload) ||
                    std::holds_alternative<OptimalNeighborhoodProgram>(
                        stage.payload) ||
                    std::holds_alternative<NeighborClassifierProgram>(
                        stage.payload) ||
                    std::holds_alternative<HagNnProgram>(stage.payload) ||
                    std::holds_alternative<HagDelaunayProgram>(stage.payload) ||
                    std::holds_alternative<RadialDensityProgram>(
                        stage.payload) ||
                    std::holds_alternative<RadiusAssignProgram>(stage.payload));
        });
}

bool planDeclaresGridRegion(const Plan& plan)
{
    return std::any_of(
        plan.stages().begin(), plan.stages().end(),
        [](const PlannedStage& stage)
        {
            return stage.residentRegion != NoResidentRegion &&
                   (std::holds_alternative<SmrfProgram>(stage.payload) ||
                    std::holds_alternative<PmfProgram>(stage.payload) ||
                    std::holds_alternative<CsfProgram>(stage.payload));
        });
}

std::vector<PlacementBoundaryExecutionFact>
residentPointViewBoundaryFacts(const Plan& plan,
                               std::size_t physicalRecordBytes)
{
    if (!physicalRecordBytes)
        throw std::invalid_argument(
            "resident PointView boundary has an empty physical record");
    std::vector<PlacementBoundaryExecutionFact> facts;
    facts.reserve(plan.summary().residencyBoundaries.size());
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const bool spill = boundary.kind == ResidencyBoundaryKind::Spill;
        const std::size_t deviceStage =
            spill ? boundary.producer : boundary.consumer;
        // A declared cardinality change spills the complete input tile plus
        // one keep-mask byte per input point; the mask is produced directly
        // by the predicate kernel and is never packed.
        const std::size_t keepMaskBytes =
            spill && deviceStage < plan.stages().size() &&
                    regionDeclaresCardinalityChange(
                        plan, plan.stages()[deviceStage].residentRegion)
                ? 1U
                : 0U;
        facts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = physicalRecordBytes + keepMaskBytes,
             .packingBytesPerPoint =
                 spill ? boundary.repackBytesPerPoint : physicalRecordBytes,
             .deviceStagingBytesPerPoint =
                 physicalRecordBytes + keepMaskBytes});
    }
    return facts;
}

std::vector<PlacementBoundaryExecutionFact>
directResidentLasBoundaryFacts(const Plan& plan,
                               std::size_t physicalRecordBytes)
{
    if (!physicalRecordBytes)
        throw std::invalid_argument(
            "direct resident LAS boundary has an empty physical record");
    std::vector<PlacementBoundaryExecutionFact> facts;
    facts.reserve(plan.summary().residencyBoundaries.size());
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const PlannedStage& producer = plan.stages().at(boundary.producer);
        const bool permutationProducer =
            std::holds_alternative<OrderingProgram>(producer.payload) ||
            std::holds_alternative<SkewnessProgram>(producer.payload);
        const bool publishesPermutation =
            boundary.kind == ResidencyBoundaryKind::Spill &&
            permutationProducer;
        if (publishesPermutation &&
            (producer.descriptor.preservesOrder ||
             producer.descriptor.deviceToHostBytesPerInputPoint !=
                 sizeof(std::uint64_t) ||
             producer.descriptor.deviceToHostFixedBytes != 0U))
            throw std::invalid_argument(
                "direct resident LAS permutation has an invalid product");
        const std::size_t transferBytesPerPoint =
            publishesPermutation
                ? producer.descriptor.deviceToHostBytesPerInputPoint
                : boundary.bytesPerPoint;
        if (!transferBytesPerPoint)
            throw std::invalid_argument(
                "direct resident LAS boundary has no logical transfer");
        // The mapped source and sparse overlay publish ordinary logical
        // columns without PointView packing. A reorder-only terminal region
        // instead publishes the descriptor's output-position -> input-position
        // permutation; the mapped source supplies every record field in that
        // new order. The raw LAS record still bounds the single whole-view
        // staging allocation.
        facts.push_back({.boundaryId = boundaryId,
                         .transferBytesPerPoint = transferBytesPerPoint,
                         .packingBytesPerPoint = 0U,
                         .deviceStagingBytesPerPoint = physicalRecordBytes});
    }
    return facts;
}

Json plannedBoundariesJson(const Plan& plan,
                           const PlanPlacementEstimate& estimate,
                           bool directExtraDoubleOutput,
                           bool directPermutedClassificationOutput,
                           bool directPermutedSortOutput)
{
    Json boundaries = Json::array();
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const PlannedStage& deviceStage =
            boundary.kind == ResidencyBoundaryKind::Upload
                ? plan.stages()[boundary.consumer]
                : plan.stages()[boundary.producer];
        if (deviceStage.residentRegion == NoResidentRegion ||
            !regionSelected(estimate, deviceStage.residentRegion))
            continue;
        const auto accounting =
            std::find_if(estimate.boundaries.begin(), estimate.boundaries.end(),
                         [&](const PlacementBoundaryEstimate& candidate)
                         { return candidate.boundaryId == boundaryId; });
        if (accounting == estimate.boundaries.end())
            throw std::logic_error(
                "selected placement boundary has no byte accounting");
        const bool elidedTerminalSpill =
            (directExtraDoubleOutput || directPermutedClassificationOutput ||
             directPermutedSortOutput) &&
            boundary.kind == ResidencyBoundaryKind::Spill &&
            boundary.consumer + 1U == plan.stages().size();
        boundaries.push_back(
            {{"id", boundaryId},
             {"kind", boundary.kind == ResidencyBoundaryKind::Upload ? "upload"
                                                                     : "spill"},
             {"producer", boundary.producer},
             {"consumer", boundary.consumer},
             {"consumers", boundary.consumers},
             {"region", deviceStage.residentRegion},
             {"bytes_per_point", boundary.bytesPerPoint},
             {"repack_bytes_per_point", boundary.repackBytesPerPoint},
             {"point_count", accounting->pointCount},
             {"logical_column_bytes", accounting->logicalColumnBytes},
             {"full_record_bytes", accounting->fullRecordBytes},
             {"predicted_transfer_bytes", accounting->predictedTransferBytes},
             {"predicted_packing_bytes", accounting->predictedPackingBytes},
             {"fallback", boundary.fallback},
             {"elided_by_direct_extra_double_output",
              elidedTerminalSpill && directExtraDoubleOutput},
             {"elided_by_direct_permuted_classification_output",
              elidedTerminalSpill && directPermutedClassificationOutput},
             {"elided_by_direct_permuted_sort_output",
              elidedTerminalSpill && directPermutedSortOutput},
             {"elided_by_direct_permuted_output",
              elidedTerminalSpill && (directPermutedClassificationOutput ||
                                      directPermutedSortOutput)},
             {"requires_full_point_record", boundary.requiresFullPointRecord}});
    }
    return boundaries;
}

// B0185: the planner-derived attributes the compose matchers in
// `RuntimePlacement.cpp` test, reported unconditionally.
//
// A declining matcher previously surfaced only `missing_calibration_model`,
// which is a fallback artifact: when a compose matcher refuses, the per-stage
// path reports the first stage without a model and hides the predicate that
// actually failed. B0184 could not explain two refusals whose calibrated
// pipelines matched their envelopes field for field, because none of these
// attributes are settable or observable from a pipeline. Three slices inferred
// them by reading matcher source and two inferred wrong.
//
// This is diagnostics only: it reads the compiled plan, changes no placement
// decision, and is emitted even when placement is unavailable, because that is
// the case it exists to explain.
Json compiledPlanJson(const Plan& plan)
{
    Json stages = Json::array();
    for (const PlannedStage& stage : plan.stages())
        stages.push_back(
            {{"id", stage.id},
             {"type", stage.descriptor.type},
             {"placement_model", stage.descriptor.placementModel},
             {"native", stage.native},
             {"preferred_residency",
              stage.preferredResidency == MemoryKind::Device ? "device"
                                                             : "host"},
             {"resident_region", stage.residentRegion == NoResidentRegion
                                     ? Json(nullptr)
                                     : Json(stage.residentRegion)},
             {"device_knn_gather_neighbors", stage.deviceKnnGatherNeighbors},
             {"device_query_bytes_per_point", stage.deviceQueryBytesPerPoint},
             {"device_index_build_bytes_per_point",
              stage.deviceIndexBuildBytesPerPoint},
             {"device_to_host_bytes_per_input_point",
              stage.descriptor.deviceToHostBytesPerInputPoint}});

    Json boundaries = Json::array();
    for (const ResidencyBoundary& boundary : plan.summary().residencyBoundaries)
        boundaries.push_back(
            {{"kind", boundary.kind == ResidencyBoundaryKind::Upload ? "upload"
                                                                     : "spill"},
             {"producer", boundary.producer},
             {"consumer", boundary.consumer},
             {"bytes_per_point", boundary.bytesPerPoint},
             {"repack_bytes_per_point", boundary.repackBytesPerPoint}});

    return {{"stage_count", plan.stages().size()},
            {"resident_regions", plan.summary().residentRegions},
            {"index_builds", plan.summary().indexBuilds},
            {"grid_builds", plan.summary().gridBuilds},
            {"all_stages_native", plan.summary().allStagesNative},
            {"stages", std::move(stages)},
            {"residency_boundaries", std::move(boundaries)}};
}

// B0190/D0218: the LAS sink's real record size when it writes extra
// dimensions.
//
// `outputRecordBytes` was assigned from one of two compile-time constants, 36
// or 48. B0189 measured an `extra_dims=all` sink emitting **100** bytes per
// point, so admitting such a writer while still declaring 36 would understate
// the terminal spill boundary by 64 B/point -- about 64 MB at 1M points -- and
// the placement decision would be made against transfers that do not exist.
//
// Upstream's rule is: every layout dimension absent from
// `las::pdrfDims(pointFormat)` is an extra dimension, and the record grows by
// `Dimension::size` of each (`io/LasWriter.cpp:230-243`). `pdrfDims` itself is
// a hidden symbol and cannot be linked (B0190), so this asks pdg's own
// `formatCarriesField` -- the same per-format table the LAS ferry already
// uses -- rather than duplicating upstream's list into a third copy.
//
// A name the plan's registry does not know is treated as extra. That is the
// safe default and not a guess: every canonical LAS field is by construction
// known to pdg, so an unknown name cannot be one.
std::size_t lasExtraDimensionBytes(const Plan& plan,
                                   const DimensionRegistry& dimensions,
                                   const pdal::PointLayoutPtr layout,
                                   std::uint8_t pointFormat)
{
    if (!layout || plan.stages().empty())
        return 0U;
    const auto* sink =
        std::get_if<FileStagePlan>(&plan.stages().back().payload);
    if (!sink || !sink->extraDimensionsAll)
        return 0U;
    std::size_t extra = 0U;
    for (const pdal::DimType& type : layout->dimTypes())
    {
        const pdg::DimensionDefinition* known =
            dimensions.find(layout->dimName(type.m_id));
        if (known && pdg::las::formatCarriesField(pointFormat, known->id))
            continue;
        extra += pdal::Dimension::size(type.m_type);
    }
    return extra;
}

std::size_t selectedIndexBuildCount(const Plan& plan,
                                    const PlanPlacementEstimate& estimate)
{
    return static_cast<std::size_t>(std::count_if(
        plan.stages().begin(), plan.stages().end(),
        [&](const PlannedStage& stage)
        {
            return stage.residentRegion != NoResidentRegion &&
                   regionSelected(estimate, stage.residentRegion) &&
                   stage.deviceIndexBuildBytesPerPoint != 0U;
        }));
}

Json statsJson(
    const Plan& plan, const RuntimePlacementResult& runtime,
    const PlacementCalibrationProfile* profile,
    const ResidentPipelineRewrite& rewrite,
    const ExecutionStatsSnapshot& actual, const DirectResidentLasResult* direct,
    const TiledSchedule* residentSchedule,
    std::optional<std::size_t> residentOutputPointCount,
    const pdal::pdg_detail::ResidentPhaseSeconds* residentPhases,
    const ResidentCommandPhaseSeconds* commandPhases,
    const pdal::pdg_detail::ResidentManagerPhaseSeconds* managerPhases,
    bool directResidentLasOutput, bool directResidentLasSource,
    bool directExtraDoubleOutput, bool directPermutedClassificationOutput,
    bool directPermutedSortOutput, bool directLasRecordSummary,
    bool directLasHostXyzMirror, bool nnDistanceDeviceOnlyHandoff,
    bool nnDistanceHostRestore, bool nnDistanceAssignmentDeviceColumnReuse,
    bool knnGatherReuse, std::optional<std::size_t> validationBudgetOverride)
{
    Json placement{{"available", runtime.available()},
                   {"unavailable_reason",
                    unavailableReasonName(runtime.unavailableReason)},
                   {"profile", profile ? std::string(profile->id) : ""},
                   // D0215/D0218: a derived record size must be readable
                   // rather than inferred from a constant in the source.
                   {"input_record_bytes", runtime.request.inputRecordBytes},
                   {"output_record_bytes", runtime.request.outputRecordBytes}};
    if (runtime.available())
    {
        const PlanPlacementEstimate& estimate = runtime.estimate;
        placement["boundary_accounting_model"] =
            runtime.request.boundaryExecutionFacts.empty()
                ? "calibration_default"
                : "executor_declared";
        placement["validation_budget_override_bytes"] =
            validationBudgetOverride ? Json(*validationBudgetOverride)
                                     : Json(nullptr);
        placement["choice"] = placementChoiceName(estimate.choice);
        placement["reason"] = placementReasonName(estimate.reason);
        placement["selected_region_count"] = estimate.selectedRegionCount;
        placement["all_host_placement"] =
            breakdownJson(estimate.allHostPlacement);
        placement["selected_placement"] =
            breakdownJson(estimate.selectedPlacement);
        placement["predicted"] = {
            {"host_to_device_bytes", estimate.hostToDeviceBytes},
            {"device_to_host_bytes", estimate.deviceToHostBytes},
            {"packing_bytes", estimate.packingBytes},
            {"index_build_bytes", estimate.indexBuildBytes},
            {"stage_result_bytes", estimate.stageResultBytes},
            {"synchronization_count", estimate.synchronizationCount},
            {"peak_device_bytes", estimate.peakDeviceBytes},
            {"untiled_device_bytes", estimate.untiledDeviceBytes},
            {"configured_device_lane_count",
             estimate.configuredDeviceLaneCount},
            {"active_device_lane_count", estimate.activeDeviceLaneCount},
            {"index_build_count", selectedIndexBuildCount(plan, estimate)},
            {"grid_build_count", selectedGridBuildCount(plan, estimate)}};
        placement["planned_boundaries"] = plannedBoundariesJson(
            plan, estimate, directExtraDoubleOutput,
            directPermutedClassificationOutput, directPermutedSortOutput);
        placement["regions"] = Json::array();
        for (const PlacementRegionEstimate& region : estimate.regions)
            placement["regions"].push_back(
                {{"id", region.residentRegion},
                 {"stage_ids", region.stageIds},
                 {"selected", region.selected},
                 {"choice", placementChoiceName(region.estimate.choice)},
                 {"reason", placementReasonName(region.estimate.reason)},
                 {"host", breakdownJson(region.estimate.host)},
                 {"device", breakdownJson(region.estimate.device)}});
    }

    Json events = Json::array();
    for (const ExecutionEvent& event : actual.events)
    {
        Json encoded{{"kind", eventKindName(event.kind)},
                     {"bytes", event.bytes},
                     {"packing_bytes", event.packingBytes}};
        if (event.kind == ExecutionEventKind::BoundaryUpload ||
            event.kind == ExecutionEventKind::BoundarySpill ||
            event.kind == ExecutionEventKind::FallbackBoundary)
            encoded["boundary_id"] = event.regionId;
        else
            encoded["region"] = event.regionId;
        events.push_back(std::move(encoded));
    }

    struct ObservedRegion
    {
        std::size_t id = 0;
        std::size_t hostToDeviceBytes = 0;
        std::size_t deviceToHostBytes = 0;
        std::size_t hostToDevicePackingBytes = 0;
        std::size_t deviceToHostPackingBytes = 0;
        bool hostToDeviceObserved = false;
        bool deviceToHostObserved = false;
    };
    std::vector<ObservedRegion> observedRegions;
    std::optional<ObservedRegion> activeRegion;
    for (const ExecutionEvent& event : actual.events)
    {
        if (event.kind == ExecutionEventKind::DeviceRegionBegin)
        {
            if (activeRegion)
                observedRegions.push_back(*activeRegion);
            activeRegion = ObservedRegion{.id = event.regionId};
        }
        else if (activeRegion && event.regionId == activeRegion->id)
        {
            if (event.kind == ExecutionEventKind::HostToDevice)
            {
                activeRegion->hostToDeviceBytes += event.bytes;
                activeRegion->hostToDevicePackingBytes += event.packingBytes;
                activeRegion->hostToDeviceObserved = true;
            }
            else if (event.kind == ExecutionEventKind::DeviceToHost)
            {
                activeRegion->deviceToHostBytes += event.bytes;
                activeRegion->deviceToHostPackingBytes += event.packingBytes;
                activeRegion->deviceToHostObserved = true;
            }
            else if (event.kind == ExecutionEventKind::DeviceRegionEnd)
            {
                observedRegions.push_back(*activeRegion);
                activeRegion.reset();
            }
        }
    }
    if (activeRegion)
        observedRegions.push_back(*activeRegion);

    Json observedCrossings = Json::array();
    for (const ExecutionEvent& event : actual.events)
        if (event.kind == ExecutionEventKind::BoundaryUpload ||
            event.kind == ExecutionEventKind::BoundarySpill)
        {
            std::size_t packingBytes = 0U;
            bool packingObserved = false;
            if (event.regionId < plan.summary().residencyBoundaries.size())
            {
                const ResidencyBoundary& boundary =
                    plan.summary().residencyBoundaries[event.regionId];
                const std::size_t deviceStage =
                    event.kind == ExecutionEventKind::BoundaryUpload
                        ? boundary.consumer
                        : boundary.producer;
                if (deviceStage < plan.stages().size())
                {
                    const std::size_t residentRegion =
                        plan.stages()[deviceStage].residentRegion;
                    const auto region = std::find_if(
                        observedRegions.begin(), observedRegions.end(),
                        [&](const ObservedRegion& candidate)
                        { return candidate.id == residentRegion; });
                    if (region != observedRegions.end())
                    {
                        if (event.kind == ExecutionEventKind::BoundaryUpload)
                        {
                            packingBytes = region->hostToDevicePackingBytes;
                            packingObserved = region->hostToDeviceObserved;
                        }
                        else
                        {
                            packingBytes = region->deviceToHostPackingBytes;
                            packingObserved = region->deviceToHostObserved;
                        }
                    }
                }
            }
            observedCrossings.push_back(
                {{"kind", event.kind == ExecutionEventKind::BoundaryUpload
                              ? "upload"
                              : "spill"},
                 {"boundary_id", event.regionId},
                 {"bytes", event.bytes},
                 {"transfer_bytes", event.bytes},
                 {"packing_bytes", packingBytes},
                 {"packing_observed", packingObserved},
                 {"evidence", "materialized_full_record_with_observed_pack"}});
        }
    for (std::size_t index = observedCrossings.empty() ? 1U
                                                       : observedRegions.size();
         index < observedRegions.size(); ++index)
    {
        const ObservedRegion& previous = observedRegions[index - 1U];
        const ObservedRegion& current = observedRegions[index];
        if (previous.id == current.id)
            continue;
        observedCrossings.push_back(
            {{"kind", "spill"},
             {"region", previous.id},
             {"bytes", previous.deviceToHostBytes},
             {"transfer_bytes", previous.deviceToHostBytes},
             {"packing_bytes", previous.deviceToHostPackingBytes},
             {"packing_observed", previous.deviceToHostObserved},
             {"evidence", "inferred_from_region_transfers"}});
        observedCrossings.push_back(
            {{"kind", "upload"},
             {"region", current.id},
             {"bytes", current.hostToDeviceBytes},
             {"transfer_bytes", current.hostToDeviceBytes},
             {"packing_bytes", current.hostToDevicePackingBytes},
             {"packing_observed", current.hostToDeviceObserved},
             {"evidence", "inferred_from_region_transfers"}});
    }
    std::optional<bool> boundaryAccountingMatches;
    if (runtime.available() && !observedCrossings.empty())
    {
        const auto elidedBoundary = [&](std::size_t boundaryId)
        {
            if ((!directExtraDoubleOutput &&
                 !directPermutedClassificationOutput &&
                 !directPermutedSortOutput) ||
                boundaryId >= plan.summary().residencyBoundaries.size())
                return false;
            const ResidencyBoundary& boundary =
                plan.summary().residencyBoundaries[boundaryId];
            return boundary.kind == ResidencyBoundaryKind::Spill &&
                   boundary.consumer + 1U == plan.stages().size();
        };
        const std::size_t predictedCrossings = static_cast<std::size_t>(
            std::count_if(runtime.estimate.boundaries.begin(),
                          runtime.estimate.boundaries.end(),
                          [&](const PlacementBoundaryEstimate& predicted)
                          { return !elidedBoundary(predicted.boundaryId); }));
        boundaryAccountingMatches =
            observedCrossings.size() == predictedCrossings;
        std::vector<std::size_t> matchedBoundaryIds;
        matchedBoundaryIds.reserve(observedCrossings.size());
        for (const Json& crossing : observedCrossings)
        {
            if (!*boundaryAccountingMatches ||
                !crossing.contains("boundary_id") ||
                !crossing.value("packing_observed", false))
            {
                boundaryAccountingMatches = false;
                break;
            }
            const std::size_t boundaryId =
                crossing.at("boundary_id").get<std::size_t>();
            if (std::find(matchedBoundaryIds.begin(), matchedBoundaryIds.end(),
                          boundaryId) != matchedBoundaryIds.end())
            {
                boundaryAccountingMatches = false;
                break;
            }
            matchedBoundaryIds.push_back(boundaryId);
            const auto predicted =
                std::find_if(runtime.estimate.boundaries.begin(),
                             runtime.estimate.boundaries.end(),
                             [&](const PlacementBoundaryEstimate& candidate)
                             { return candidate.boundaryId == boundaryId; });
            if (predicted == runtime.estimate.boundaries.end() ||
                predicted->predictedTransferBytes !=
                    crossing.at("transfer_bytes").get<std::size_t>() ||
                predicted->predictedPackingBytes !=
                    crossing.at("packing_bytes").get<std::size_t>())
            {
                boundaryAccountingMatches = false;
                break;
            }
        }
        if (*boundaryAccountingMatches)
            for (const PlacementBoundaryEstimate& predicted :
                 runtime.estimate.boundaries)
                if (!elidedBoundary(predicted.boundaryId) &&
                    std::find(matchedBoundaryIds.begin(),
                              matchedBoundaryIds.end(),
                              predicted.boundaryId) == matchedBoundaryIds.end())
                {
                    boundaryAccountingMatches = false;
                    break;
                }
    }
    Json totals = Json::object();
    for (std::size_t index = 0; index < ExecutionEventKindCount; ++index)
    {
        const auto kind = static_cast<ExecutionEventKind>(index);
        totals[eventKindName(kind)] = {
            {"count", actual.totals[index].count},
            {"bytes", actual.totals[index].bytes},
            {"packing_bytes", actual.totals[index].packingBytes}};
    }

    Json execution;
    if (direct)
    {
        const TiledSchedule& schedule = direct->schedule;
        // The fused executor is B0005's calibrated class; the ordered
        // executor matches its calibration once the selected region resolved
        // to the measured ordered-point-program residual (D0064).
        const bool calibrationMatchesExecutor =
            !direct->orderedExecutor ||
            std::any_of(runtime.regionCalibrations.begin(),
                        runtime.regionCalibrations.end(),
                        [&](const PlacementRegionCalibration& calibration)
                        {
                            return calibration.residentRegion ==
                                       direct->residentRegion &&
                                   calibration.model == "ordered-point-program";
                        });
        execution = {
            {"executor", direct->orderedExecutor ? "direct_ordered_las"
                                                 : "direct_fused_las"},
            {"direct_las_output", true},
            {"selected_device_calibration_matches_executor",
             calibrationMatchesExecutor},
            {"boundary_accounting_matches_prediction",
             boundaryAccountingMatches ? Json(*boundaryAccountingMatches)
                                       : Json(nullptr)},
            {"rewrite_executable", true},
            {"rewrite_reason", "direct executor; no pipeline rewrite"},
            {"resident_preflight",
             {{"attempted", rewrite.preflightAttempted},
              {"accepted", rewrite.preflightAccepted},
              {"reason", rewrite.preflightReason}}},
            {"selected_regions", {direct->residentRegion}},
            {"selected_stage_ids", direct->stageIds},
            {"schedule",
             {{"pipeline_class", direct->orderedExecutor
                                     ? "ordered_point_program"
                                     : "fused_point_program"},
              {"observed_output_item_count", direct->outputPointCount},
              {"tile_count", schedule.tileCount},
              {"item_count", schedule.itemCount},
              {"tile_item_capacity", schedule.tileItemCapacity},
              {"configured_lane_count", schedule.configuredLaneCount},
              {"active_lane_count", schedule.activeLaneCount},
              {"lane_reuse_count", schedule.laneReuseCount},
              {"peak_lane_bytes", schedule.peakLaneBytes},
              {"memory_budget_bytes", runtime.request.deviceMemoryBudgetBytes},
              {"memory_limited", schedule.memoryLimited},
              {"serial_dependency", schedule.serialDependency}}},
            {"exact_host_repair", Json(nullptr)},
            {"exact_device_repair", Json(nullptr)},
            {"observed_crossings", Json::array()},
            {"events", std::move(events)},
            {"totals", std::move(totals)}};
    }
    else if (residentSchedule)
    {
        // The shared-index neighborhood models were re-measured against the
        // planner-selected resident executor itself (D0077/D0081 ladders,
        // status
        // clean-resident-shared-index), so a whole-view execution whose
        // every applied region calibration is one of those models matches
        // its calibration honestly. The boundary_batch executor still has
        // no calibration of its own and keeps reporting false.
        static constexpr std::string_view ResidentCalibratedModels[] = {
            "approximatecoplanar",
            "lof",
            "normal",
            "eigenvalues",
            "nndistance",
            "covariancefeatures",
            "eigen-family-compose",
            "estimaterank",
            "optimalneighborhood",
            "rank-optimal-compose",
            "neighborclassifier",
            "radiusassign"};
        const bool sharedIndex = planDeclaresNeighborhoodRegion(plan);
        const bool globalGrid = planDeclaresGridRegion(plan);
        const bool directCalibrationMatches =
            directResidentLasOutput &&
            runtime.estimate.reason != PlacementReason::UncalibratedStage &&
            !runtime.regionCalibrations.empty() &&
            ((directResidentLasSource && directLasRecordSummary &&
              !directLasHostXyzMirror &&
              std::all_of(runtime.regionCalibrations.begin(),
                          runtime.regionCalibrations.end(),
                          [](const PlacementRegionCalibration& calibration)
                          {
                              return calibration.model ==
                                         "radiusassign-direct" ||
                                     calibration.model ==
                                         "outlier-nndistance-direct-compose" ||
                                     calibration.model ==
                                         "neighborclassifier-direct-compose" ||
                                     calibration.model ==
                                         "hag-nn-count1-direct-compose" ||
                                     calibration.model ==
                                         "hag-delaunay-count3-direct-compose";
                          })) ||
             (!directResidentLasSource && !directLasRecordSummary &&
              !directLasHostXyzMirror &&
              std::all_of(runtime.regionCalibrations.begin(),
                          runtime.regionCalibrations.end(),
                          [](const PlacementRegionCalibration& calibration)
                          {
                              return calibration.model ==
                                     "approximatecoplanar-direct-compose";
                          })));
        const bool calibrationMatchesExecutor =
            sharedIndex &&
            (directResidentLasOutput
                 ? directCalibrationMatches
                 : !runtime.regionCalibrations.empty() &&
                       std::all_of(
                           runtime.regionCalibrations.begin(),
                           runtime.regionCalibrations.end(),
                           [](const PlacementRegionCalibration& calibration)
                           {
                               return std::find(
                                          std::begin(ResidentCalibratedModels),
                                          std::end(ResidentCalibratedModels),
                                          calibration.model) !=
                                      std::end(ResidentCalibratedModels);
                           }));
        execution = {
            {"executor",
             directResidentLasOutput
                 ? ((directPermutedClassificationOutput ||
                     directPermutedSortOutput)
                        ? "planner_resident_global_order_direct_las"
                        : "planner_resident_shared_index_direct_las")
             : sharedIndex ? "planner_resident_shared_index"
             : globalGrid  ? "planner_resident_global_grid"
                           : "planner_resident_boundary_batch"},
            {"direct_las_output", directResidentLasOutput},
            {"direct_extra_double_output", directExtraDoubleOutput},
            {"direct_permuted_classification_output",
             directPermutedClassificationOutput},
            {"direct_permuted_sort_output", directPermutedSortOutput},
            {"direct_permuted_output",
             directPermutedClassificationOutput || directPermutedSortOutput},
            {"terminal_spill_elided", directExtraDoubleOutput ||
                                          directPermutedClassificationOutput ||
                                          directPermutedSortOutput},
            {"selected_device_calibration_matches_executor",
             calibrationMatchesExecutor},
            {"boundary_accounting_matches_prediction",
             boundaryAccountingMatches.value_or(false)},
            {"rewrite_executable", rewrite.executable},
            {"rewrite_reason", rewrite.reason},
            {"resident_preflight",
             {{"attempted", rewrite.preflightAttempted},
              {"accepted", rewrite.preflightAccepted},
              {"reason", rewrite.preflightReason}}},
            {"selected_regions", rewrite.selectedRegions},
            {"selected_stage_ids", rewrite.selectedStageIds},
            {"schedule",
             {{"pipeline_class",
               (directPermutedClassificationOutput || directPermutedSortOutput)
                   ? "whole_view_global_order"
               : planDeclaresNeighborhoodRegion(plan)
                   ? "whole_view_neighborhood"
               : planDeclaresGridRegion(plan) ? "whole_view_grid"
               : planDeclaresResidentCardinalityChange(plan)
                   ? "ordered_point_program"
                   : "fused_point_program"},
              {"tile_count", residentSchedule->tileCount},
              {"item_count", residentSchedule->itemCount},
              {"observed_output_item_count",
               residentOutputPointCount ? Json(*residentOutputPointCount)
                                        : Json(nullptr)},
              {"tile_item_capacity", residentSchedule->tileItemCapacity},
              {"configured_lane_count", residentSchedule->configuredLaneCount},
              {"active_lane_count", residentSchedule->activeLaneCount},
              {"lane_reuse_count", residentSchedule->laneReuseCount},
              {"peak_lane_bytes", residentSchedule->peakLaneBytes},
              {"observed_peak_lane_bytes",
               residentSchedule->observedPeakLaneBytes},
              {"memory_budget_bytes", runtime.request.deviceMemoryBudgetBytes},
              {"memory_limited", residentSchedule->memoryLimited},
              {"serial_dependency", residentSchedule->serialDependency}}},
            {"host_boundary_phase_seconds",
             residentPhases
                 ? Json{{"upload_pack", residentPhases->uploadPack},
                        {"spill_wait", residentPhases->spillWait},
                        {"spill_publish", residentPhases->spillPublish}}
                 : Json(nullptr)},
            {"exact_host_repair",
             residentPhases
                 ? Json{{"seconds", residentPhases->exactHostRepair},
                        {"ambiguous_rows", residentPhases->ambiguousRepairRows},
                        {"incomplete_rows",
                         residentPhases->incompleteRepairRows},
                        {"repaired_rows", residentPhases->repairedRows},
                        {"lof",
                         {{"workers", residentPhases->lofRepairWorkers},
                          {"parallel_repaired_rows",
                           residentPhases->lofParallelRepairRows},
                          {"coordinate_cache_rows",
                           residentPhases->lofCoordinateCacheRows}}},
                        {"nndistance",
                         {{"seconds",
                           residentPhases->nnDistanceExactHostRepair},
                          {"incomplete_rows",
                           residentPhases->nnDistanceHostIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->nnDistanceHostRepairRows}}},
                        {"statistical_outlier",
                         {{"seconds", residentPhases->outlierExactHostRepair},
                          {"incomplete_rows",
                           residentPhases->outlierHostIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->outlierHostRepairRows}}},
                        {"approximate_coplanar",
                         {{"seconds",
                           residentPhases->approximateCoplanarExactHostRepair},
                          {"triggered",
                           residentPhases->approximateCoplanarRepairTriggers !=
                               0U},
                          {"trigger_count",
                           residentPhases->approximateCoplanarRepairTriggers},
                          {"ambiguous_rows",
                           residentPhases
                               ->approximateCoplanarAmbiguousRepairRows},
                          {"incomplete_rows",
                           residentPhases
                               ->approximateCoplanarIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->approximateCoplanarRepairRows},
                          {"kd3_used",
                           residentPhases->approximateCoplanarKd3Uses != 0U},
                          {"kd3_uses",
                           residentPhases->approximateCoplanarKd3Uses},
                          {"device_to_host_bytes",
                           residentPhases
                               ->approximateCoplanarDeviceToHostRepairBytes},
                          {"host_to_device_bytes",
                           residentPhases
                               ->approximateCoplanarHostToDeviceRepairBytes}}},
                        {"neighborclassifier",
                         {{"seconds",
                           residentPhases->neighborClassifierExactHostRepair},
                          {"ambiguous_rows",
                           residentPhases
                               ->neighborClassifierAmbiguousRepairRows},
                          {"incomplete_rows",
                           residentPhases
                               ->neighborClassifierIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->neighborClassifierRepairRows},
                          {"kd3_used",
                           residentPhases->neighborClassifierKd3Uses != 0U},
                          {"kd3_uses",
                           residentPhases->neighborClassifierKd3Uses}}}}
                 : Json(nullptr)},
            {"exact_device_repair",
             residentPhases
                 ? Json{{"seconds", residentPhases->exactDeviceRepair},
                        {"incomplete_rows",
                         residentPhases->deviceIncompleteRepairRows},
                        {"repaired_rows", residentPhases->deviceRepairRows},
                        {"nndistance",
                         {{"seconds",
                           residentPhases->nnDistanceExactDeviceRepair},
                          {"incomplete_rows",
                           residentPhases
                               ->nnDistanceDeviceIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->nnDistanceDeviceRepairRows},
                          {"parallel_repaired_rows",
                           residentPhases
                               ->nnDistanceParallelDeviceRepairRows}}},
                        {"statistical_outlier",
                         {{"seconds", residentPhases->outlierExactDeviceRepair},
                          {"incomplete_rows",
                           residentPhases->outlierDeviceIncompleteRepairRows},
                          {"repaired_rows",
                           residentPhases->outlierDeviceRepairRows},
                          {"parallel_repaired_rows",
                           residentPhases->outlierParallelDeviceRepairRows}}}}
                 : Json(nullptr)},
            {"index_builds",
             {{"predicted", selectedIndexBuildCount(plan, runtime.estimate)},
              {"observed", actual
                               .totals[static_cast<std::size_t>(
                                   ExecutionEventKind::IndexBuild)]
                               .count},
              {"matches_prediction",
               selectedIndexBuildCount(plan, runtime.estimate) ==
                   actual
                       .totals[static_cast<std::size_t>(
                           ExecutionEventKind::IndexBuild)]
                       .count}}},
            {"grid_builds",
             {{"predicted", selectedGridBuildCount(plan, runtime.estimate)},
              {"observed", actual
                               .totals[static_cast<std::size_t>(
                                   ExecutionEventKind::GridBuild)]
                               .count},
              {"matches_prediction",
               selectedGridBuildCount(plan, runtime.estimate) ==
                   actual
                       .totals[static_cast<std::size_t>(
                           ExecutionEventKind::GridBuild)]
                       .count}}},
            {"observed_crossings", std::move(observedCrossings)},
            {"events", std::move(events)},
            {"totals", std::move(totals)}};
    }
    else
    {
        execution = {
            {"executor", rewrite.selectedRegions.empty()
                             ? "pdal_standard_host"
                             : "pdal_point_program_wrapper"},
            {"direct_las_output", false},
            {"selected_device_calibration_matches_executor",
             rewrite.selectedRegions.empty() ? Json(nullptr) : Json(false)},
            {"boundary_accounting_matches_prediction",
             boundaryAccountingMatches ? Json(*boundaryAccountingMatches)
                                       : Json(nullptr)},
            {"rewrite_executable", rewrite.executable},
            {"rewrite_reason", rewrite.reason},
            {"resident_preflight",
             {{"attempted", rewrite.preflightAttempted},
              {"accepted", rewrite.preflightAccepted},
              {"reason", rewrite.preflightReason}}},
            {"selected_regions", rewrite.selectedRegions},
            {"selected_stage_ids", rewrite.selectedStageIds},
            {"exact_host_repair", Json(nullptr)},
            {"exact_device_repair", Json(nullptr)},
            {"observed_crossings", std::move(observedCrossings)},
            {"events", std::move(events)},
            {"totals", std::move(totals)}};
    }

    execution["direct_las_resident_source"] = directResidentLasSource;
    execution["direct_las_record_summary"] = directLasRecordSummary;
    execution["direct_las_host_xyz_mirror"] = directLasHostXyzMirror;
    execution["nndistance_device_only_handoff"] = nnDistanceDeviceOnlyHandoff;
    execution["nndistance_host_restore"] = nnDistanceHostRestore;
    execution["nndistance_assignment_device_column_reuse"] =
        nnDistanceAssignmentDeviceColumnReuse;
    execution["knn_gather_reuse"] =
        directResidentLasSource ? Json(knnGatherReuse) : Json(nullptr);
    execution["pipeline_phase_seconds"] =
        commandPhases
            ? Json{{"command_before_stats", commandPhases->commandBeforeStats},
                   {"validation_placement_preflight",
                    commandPhases->validationPlacementPreflight},
                   {"rewritten_manager_execution",
                    commandPhases->rewrittenManagerExecution},
                   {"canonical_las_publication",
                    commandPhases->canonicalLasPublication},
                   {"other_control", commandPhases->otherControl}}
            : Json(nullptr);
    execution["validation_placement_preflight_breakdown_seconds"] =
        commandPhases
            ? Json{{"plan_and_original_validation",
                    commandPhases->planAndOriginalValidation},
                   {"runtime_placement", commandPhases->runtimePlacement},
                   {"runtime_device_and_profile",
                    commandPhases->runtimeDeviceAndProfile},
                   {"runtime_initial_placement",
                    commandPhases->runtimeInitialPlacement},
                   {"runtime_executor_selection",
                    commandPhases->runtimeExecutorSelection},
                   {"runtime_subphases_match",
                    std::abs(commandPhases->runtimeDeviceAndProfile +
                             commandPhases->runtimeInitialPlacement +
                             commandPhases->runtimeExecutorSelection -
                             commandPhases->runtimePlacement) <= 1.0e-9},
                   {"rewrite_and_resident_preflight",
                    commandPhases->rewriteAndResidentPreflight},
                   {"total", commandPhases->validationPlacementPreflight},
                   {"matches_validation_placement_preflight",
                    std::abs(commandPhases->planAndOriginalValidation +
                             commandPhases->runtimePlacement +
                             commandPhases->rewriteAndResidentPreflight -
                             commandPhases->validationPlacementPreflight) <=
                        1.0e-9}}
            : Json(nullptr);
    execution["rewritten_manager_execution_breakdown_seconds"] =
        managerPhases
            ? Json{{"manager_graph_and_prepare",
                    managerPhases->managerGraphAndPrepare},
                   {"reader_row_table_materialization",
                    managerPhases->readerRowTableMaterialization},
                   {"resident_wrapper_index_filter",
                    managerPhases->residentWrapperIndexFilter},
                   {"post_spill_stage_control",
                    managerPhases->postSpillStageControl},
                   {"total", managerPhases->total},
                   {"matches_rewritten_manager_execution",
                    commandPhases &&
                        std::abs(managerPhases->total -
                                 commandPhases->rewrittenManagerExecution) <=
                            1.0e-9}}
            : Json(nullptr);
    if (managerPhases)
    {
        const pdal::pdg_detail::ResidentManagerDetailSeconds& detail =
            managerPhases->detail;
        const double profiled = detail.residentProductSetup +
                                detail.directLasCoordinateHydration +
                                detail.indexConfiguration + detail.indexBuild +
                                detail.neighborhoodQueryProjection +
                                detail.adjacentPointProgramBridge;
        const double unattributed =
            std::max(0.0, managerPhases->residentWrapperIndexFilter - profiled);
        execution["resident_work_breakdown_seconds"] = {
            {"resident_product_setup", detail.residentProductSetup},
            {"direct_las_coordinate_hydration",
             detail.directLasCoordinateHydration},
            {"index_configuration", detail.indexConfiguration},
            {"index_build", detail.indexBuild},
            {"neighborhood_query_projection",
             detail.neighborhoodQueryProjection},
            {"adjacent_point_program_bridge",
             detail.adjacentPointProgramBridge},
            {"profiled_subphases_total", profiled},
            {"resident_wrapper_unattributed", unattributed},
            {"resident_wrapper_index_filter_total",
             managerPhases->residentWrapperIndexFilter},
            {"fits_within_resident_wrapper_interval",
             profiled <= managerPhases->residentWrapperIndexFilter + 1.0e-9}};
        const double indexProfiled =
            detail.indexConfigSelection + detail.indexEnvelopeValidation;
        execution["index_configuration_breakdown_seconds"] = {
            {"config_selection", detail.indexConfigSelection},
            {"exact_envelope_validation", detail.indexEnvelopeValidation},
            {"profiled_subphases_total", indexProfiled},
            {"unattributed",
             std::max(0.0, detail.indexConfiguration - indexProfiled)},
            {"total", detail.indexConfiguration},
            {"fits_within_index_configuration",
             indexProfiled <= detail.indexConfiguration + 1.0e-9}};
        const double hydrationProfiled = detail.directLasHydrationAllocation +
                                         detail.directLasHydrationSubmission +
                                         detail.directLasHydrationWait;
        execution["direct_las_hydration_breakdown_seconds"] = {
            {"validation_and_allocation", detail.directLasHydrationAllocation},
            {"transfer_and_kernel_submission",
             detail.directLasHydrationSubmission},
            {"final_stream_wait", detail.directLasHydrationWait},
            {"profiled_subphases_total", hydrationProfiled},
            {"unattributed", std::max(0.0, detail.directLasCoordinateHydration -
                                               hydrationProfiled)},
            {"total", detail.directLasCoordinateHydration},
            {"fits_within_direct_las_hydration",
             hydrationProfiled <=
                 detail.directLasCoordinateHydration + 1.0e-9}};
    }
    else
    {
        execution["resident_work_breakdown_seconds"] = Json(nullptr);
        execution["index_configuration_breakdown_seconds"] = Json(nullptr);
        execution["direct_las_hydration_breakdown_seconds"] = Json(nullptr);
    }

    if (managerPhases)
    {
        const pdal::pdg_detail::ResidentEigenFamilyDetailSeconds& detail =
            managerPhases->detail.eigenFamily;
        const double profiled =
            detail.eigenSystemsSubmission + detail.ambiguousRepair +
            detail.outputPreparation + detail.projectionAndCopy +
            detail.statusScan + detail.transcendentalFeatures +
            detail.columnPublication + detail.hostColumnUpload;
        if (profiled > 0.0)
            execution["eigen_family_breakdown_seconds"] = {
                {"eigen_systems_submission", detail.eigenSystemsSubmission},
                {"ambiguous_repair", detail.ambiguousRepair},
                {"ambiguous_repair_status_wait", detail.ambiguousRepairStatusWait},
                {"ambiguous_repair_system_download",
                 detail.ambiguousRepairSystemDownload},
                {"ambiguous_repair_index_build", detail.ambiguousRepairIndexBuild},
                {"ambiguous_repair_rows", detail.ambiguousRepairRows},
                {"ambiguous_repair_upload", detail.ambiguousRepairUpload},
                {"ambiguous_rows", detail.ambiguousRows},
                {"incomplete_rows", detail.incompleteRows},
                {"repaired_rows", detail.repairedRows},
                {"output_preparation", detail.outputPreparation},
                {"projection_and_copy", detail.projectionAndCopy},
                {"status_scan", detail.statusScan},
                {"transcendental_features", detail.transcendentalFeatures},
                {"column_publication", detail.columnPublication},
                {"host_column_upload", detail.hostColumnUpload},
                {"profiled_subphases_total", profiled},
                {"neighborhood_query_projection_total",
                 managerPhases->detail.neighborhoodQueryProjection},
                {"unattributed",
                 std::max(0.0, managerPhases->detail.neighborhoodQueryProjection -
                                   profiled)}};
        else
            execution["eigen_family_breakdown_seconds"] = Json(nullptr);
    }
    else
        execution["eigen_family_breakdown_seconds"] = Json(nullptr);

    if (managerPhases)
    {
        const pdal::pdg_detail::ResidentNnDistanceDetailSeconds& detail =
            managerPhases->detail.nnDistance;
        const double profiled =
            detail.outputPreparation + detail.querySubmission +
            detail.statusAllocation + detail.resultTransferCall +
            detail.statusTransferCall + detail.explicitStreamWait +
            detail.statusScanAndRepair + detail.outputPublication;
        if (profiled > 0.0)
        {
            const double broad =
                managerPhases->detail.neighborhoodQueryProjection;
            execution["nndistance_query_breakdown_seconds"] = {
                {"output_preparation", detail.outputPreparation},
                {"query_submission", detail.querySubmission},
                {"status_allocation", detail.statusAllocation},
                {"result_transfer_call", detail.resultTransferCall},
                {"status_transfer_call", detail.statusTransferCall},
                {"explicit_stream_wait", detail.explicitStreamWait},
                {"status_scan_and_repair", detail.statusScanAndRepair},
                {"output_publication", detail.outputPublication},
                {"profiled_subphases_total", profiled},
                {"broad_query_unattributed", std::max(0.0, broad - profiled)},
                {"broad_query_total", broad},
                {"fits_within_broad_query", profiled <= broad + 1.0e-9}};
        }
        else
            execution["nndistance_query_breakdown_seconds"] = Json(nullptr);
    }
    else
        execution["nndistance_query_breakdown_seconds"] = Json(nullptr);

    return {{"schema", "pdg-resident-stats-v1"},
            {"placement", std::move(placement)},
            {"plan", compiledPlanJson(plan)},
            {"execution", std::move(execution)}};
}

// B0196/D0215: report where an admission attempt gave up.
//
// `gpupdal pipeline` is `runResidentPipelineImpl(..., automaticAdmission=true)`;
// when it declines it returns `std::nullopt` and the caller re-reads, prepares
// and executes the pipeline by another route. B0194 measured that discarded
// attempt at a fixed ~19 ms and could not attribute ~13 ms of it, because
// stats are written only on the success path and every decline returns before
// reaching them. B0195 narrowed the question to *where* a decline becomes
// determined, so the attempt could stop there instead of not starting — a
// blanket early exit would forfeit measured wins such as B0188's 6.351x.
//
// Diagnostics only, and inert unless `PDG_DEBUG_ADMISSION_PHASES` is set: it
// records a checkpoint label and reports it on destruction. It changes no
// decision and no output.
class AdmissionTrace
{
public:
    explicit AdmissionTrace(bool automatic) noexcept
        : m_enabled(automatic &&
                    std::getenv("PDG_DEBUG_ADMISSION_PHASES") != nullptr),
          m_started(ResidentCommandClock::now())
    {
    }
    AdmissionTrace(const AdmissionTrace&) = delete;
    AdmissionTrace& operator=(const AdmissionTrace&) = delete;

    void at(const char* checkpoint) noexcept
    {
        if (m_enabled)
        {
            const double now = std::chrono::duration<double>(
                                   ResidentCommandClock::now() - m_started)
                                   .count();
            std::clog << "pdg-admission: reached " << checkpoint << " at "
                      << (now * 1.0e3) << " ms\n";
            m_checkpoint = checkpoint;
        }
    }
    void accepted() noexcept
    {
        m_accepted = true;
    }

    ~AdmissionTrace()
    {
        if (!m_enabled)
            return;
        const double total = std::chrono::duration<double>(
                                 ResidentCommandClock::now() - m_started)
                                 .count();
        std::clog << "pdg-admission: " << (m_accepted ? "accepted" : "declined")
                  << " after " << (total * 1.0e3) << " ms; last checkpoint "
                  << m_checkpoint << '\n';
    }

private:
    bool m_enabled = false;
    bool m_accepted = false;
    const char* m_checkpoint = "entry";
    ResidentCommandClock::time_point m_started;
};

bool writeStats(std::string_view destination, const Json& document)
{
    std::ofstream output(std::filesystem::path(destination),
                         std::ios::binary | std::ios::trunc);
    if (!output)
        return false;
    output << document.dump(2) << '\n';
    return static_cast<bool>(output);
}

std::optional<std::string> readPipelineFile(const std::filesystem::path& path)
{
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error ||
        std::filesystem::file_size(path, error) > MaximumPipelineBytes || error)
        return std::nullopt;
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    std::string text((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    return input.bad() ? std::nullopt
                       : std::optional<std::string>(std::move(text));
}

std::filesystem::path normalizedPath(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::path absolute = std::filesystem::absolute(path, error);
    if (error)
        return path.lexically_normal();
    std::filesystem::path canonical =
        std::filesystem::weakly_canonical(absolute, error);
    return error ? absolute.lexically_normal() : canonical;
}

bool sameFilesystemPath(const std::filesystem::path& first,
                        const std::filesystem::path& second)
{
    std::error_code firstError;
    const bool firstExists = std::filesystem::exists(first, firstError);
    std::error_code secondError;
    const bool secondExists = std::filesystem::exists(second, secondError);
    if (!firstError && !secondError && firstExists && secondExists)
    {
        std::error_code equivalentError;
        if (std::filesystem::equivalent(first, second, equivalentError) &&
            !equivalentError)
            return true;
    }
    return normalizedPath(first) == normalizedPath(second);
}

bool statsAliasesPipelineFile(const Plan& plan,
                              const std::filesystem::path& statsPath)
{
    for (const PlannedStage& stage : plan.stages())
    {
        if (stage.role != StageRole::Reader && stage.role != StageRole::Writer)
            continue;
        const auto* file = std::get_if<FileStagePlan>(&stage.payload);
        if (file && !file->filename.empty() && file->filename != "-" &&
            sameFilesystemPath(statsPath, file->filename))
            return true;
    }
    return false;
}
} // unnamed namespace

std::optional<int> runResidentPipelineImpl(int argc, char** argv,
                                           bool automaticAdmission)
{
    if (argc != 3 && argc != 5)
    {
        if (automaticAdmission)
            return std::nullopt;
        std::cerr << "Usage: gpupdal resident PIPELINE [--stats FILE]\n";
        return 1;
    }
    std::string statsDestination;
    if (argc == 5)
    {
        if (std::string_view(argv[3]) != "--stats")
        {
            if (automaticAdmission)
                return std::nullopt;
            std::cerr << "Usage: gpupdal resident PIPELINE [--stats FILE]\n";
            return 1;
        }
        statsDestination = argv[4];
        if (statsDestination == "-")
        {
            if (automaticAdmission)
                return std::nullopt;
            std::cerr << "gpupdal: --stats requires a file path\n";
            return 1;
        }
    }

    AdmissionTrace admissionTrace(automaticAdmission);

    const std::optional<ResidentCommandClock::time_point> commandStarted =
        statsDestination.empty()
            ? std::nullopt
            : std::optional<ResidentCommandClock::time_point>(
                  ResidentCommandClock::now());

    const std::optional<std::string> pipelineText = readPipelineFile(argv[2]);
    if (!pipelineText)
    {
        if (automaticAdmission)
            return std::nullopt;
        std::cerr << "gpupdal: unable to read resident pipeline " << argv[2]
                  << '\n';
        return 1;
    }
    admissionTrace.at("pipeline-text-read");
    // B0233 measured the exact label/NNDistance/assignment hybrid composition
    // as the faster executor.  Decline its strict literal grammar before plan,
    // device, and profile discovery so the resident selector neither preempts
    // that winner nor charges it the old ~170 ms failed-placement probe.
    if (automaticAdmission &&
        automaticLabelNnDistanceHybridCandidate(*pipelineText))
        return std::nullopt;
    const bool requireAutomaticEigenFamily =
        std::getenv("PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT") != nullptr;
    const bool requireAutomaticNormalCovariance =
        std::getenv("PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT") !=
        nullptr;
    const bool automaticEigenFamily =
        automaticAdmission &&
        automaticResidentEigenFamilyEnvelope(*pipelineText);
    const bool requireAutomaticRankOptimal =
        std::getenv("PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT") != nullptr;
    const bool automaticRankOptimal =
        automaticAdmission &&
        automaticResidentRankOptimalEnvelope(*pipelineText);
    const bool outlierNnDistanceEnvelope =
        experimentalResidentOutlierNnDistanceEnvelope(*pipelineText);
    const bool standaloneOutlierEnvelope =
        experimentalResidentStandaloneOutlierEnvelope(*pipelineText);
    const bool neighborClassifierEnvelope =
        experimentalResidentNeighborClassifierEnvelope(*pipelineText);
    const bool requireAutomaticNeighborClassifier =
        std::getenv("PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT") !=
        nullptr;
    const bool automaticNeighborClassifier =
        automaticAdmission && neighborClassifierEnvelope;
    const bool radialDensityAssignEnvelope =
        experimentalResidentRadialDensityAssignEnvelope(*pipelineText);
    const bool radiusOutlierRadialDensityAssignEnvelope =
        experimentalResidentRadiusOutlierRadialDensityAssignEnvelope(
            *pipelineText);
    const bool requireRadiusOutlierRadialDensityComposition =
        std::getenv("PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION") !=
        nullptr;
    const bool requireAutomaticRadiusOutlierRadialDensity =
        std::getenv(
            "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT") !=
        nullptr;
    const bool automaticRadiusOutlierRadialDensity =
        automaticAdmission && radiusOutlierRadialDensityAssignEnvelope;
    const bool experimentalDirectRadialDensityAssign =
        !automaticAdmission && radialDensityAssignEnvelope &&
        ((std::getenv("PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT") !=
              nullptr &&
          std::getenv("PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE") !=
              nullptr) ||
         (std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr &&
          std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr));
    const bool extraDoubleOutputEnvelope =
        experimentalResidentExtraDoubleOutputEnvelope(*pipelineText);
    const bool requireAutomaticHagNn =
        std::getenv("PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT") != nullptr;
    const bool automaticHagNnCountOne =
        automaticAdmission &&
        automaticDirectHagNnCountOneEnvelope(*pipelineText);
    const bool requireAutomaticHagDelaunay =
        std::getenv("PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT") != nullptr;
    const bool automaticHagDelaunayCountThree =
        (automaticAdmission || requireAutomaticHagDelaunay) &&
        automaticDirectHagDelaunayCountThreeEnvelope(*pipelineText);
    const bool requireDirectExtraDoubleOutput =
        std::getenv("PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT") != nullptr;
    const bool experimentalDirectExtraDoubleOutputRequested =
        !automaticAdmission && extraDoubleOutputEnvelope &&
        (std::getenv("PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT") !=
             nullptr ||
         requireDirectExtraDoubleOutput);
    const bool requireAutomaticOutlierNnDistance =
        std::getenv("PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT") !=
        nullptr;
    const bool automaticOutlierNnDistance =
        automaticAdmission && outlierNnDistanceEnvelope;
    const bool directRadiusAssign = directRadiusAssignEnvelope(*pipelineText);
    const bool directSkewnessEnvelope =
        experimentalDirectSkewnessEnvelope(*pipelineText);
    const bool requireDirectSkewnessComposition =
        std::getenv("PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION") != nullptr;
    const bool requireAutomaticSkewness =
        std::getenv("PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT") != nullptr;
    const bool automaticSkewness =
        automaticAdmission && automaticDirectSkewnessEnvelope(*pipelineText);
    const bool experimentalDirectSkewnessComposition =
        automaticSkewness ||
        (!automaticAdmission && directSkewnessEnvelope &&
         (requireDirectSkewnessComposition ||
          std::getenv("PDG_EXPERIMENTAL_DIRECT_SKEWNESS_COMPOSITION") !=
              nullptr));
    const bool directSortEnvelope =
        experimentalDirectSortEnvelope(*pipelineText);
    const bool requireDirectSortComposition =
        std::getenv("PDG_REQUIRE_DIRECT_SORT_COMPOSITION") != nullptr;
    const bool requireAutomaticSort =
        std::getenv("PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT") != nullptr;
    const bool automaticSort = automaticAdmission && directSortEnvelope;
    const bool experimentalDirectSortComposition =
        !automaticAdmission && directSortEnvelope &&
        (requireDirectSortComposition ||
         std::getenv("PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION") != nullptr);
    const bool requireAutomaticRadiusAssign =
        std::getenv("PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT") != nullptr;
    const bool automaticRadiusAssign = automaticAdmission && directRadiusAssign;
    const bool approximateCoplanarEnvelope =
        automaticResidentApproximateCoplanarEnvelope(*pipelineText);
    const bool requireAutomaticApproximateCoplanar =
        std::getenv("PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT") !=
        nullptr;
    const bool automaticApproximateCoplanar =
        automaticAdmission && approximateCoplanarEnvelope;
    const bool requireDirectResidentLasOutput =
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr ||
        std::getenv("PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT") != nullptr ||
        requireDirectExtraDoubleOutput ||
        requireRadiusOutlierRadialDensityComposition ||
        requireDirectSkewnessComposition || requireDirectSortComposition;
    const bool requireDirectResidentLasSource =
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr ||
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") != nullptr ||
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND") !=
            nullptr ||
        std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") != nullptr ||
        std::getenv("PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF") != nullptr ||
        std::getenv("PDG_REQUIRE_NND_HOST_RESTORE") != nullptr ||
        std::getenv("PDG_REQUIRE_NND_PARALLEL_REPAIR") != nullptr ||
        requireRadiusOutlierRadialDensityComposition ||
        requireDirectSkewnessComposition || requireDirectSortComposition;
    const bool requireNnDistanceDeviceRepair =
        std::getenv("PDG_REQUIRE_NND_DEVICE_REPAIR") != nullptr;
    const bool requireNnDistanceParallelRepair =
        std::getenv("PDG_REQUIRE_NND_PARALLEL_REPAIR") != nullptr;
    const bool requireOutlierDeviceRepair =
        std::getenv("PDG_REQUIRE_OUTLIER_DEVICE_REPAIR") != nullptr;
    const bool requireOutlierParallelRepair =
        std::getenv("PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR") != nullptr;
    const bool requireLofParallelRepair =
        std::getenv("PDG_REQUIRE_LOF_PARALLEL_REPAIR") != nullptr;
    const bool requireLofCoordinateCache =
        std::getenv("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE") != nullptr;
    const bool requiredDirectRadiusAssignExecutor =
        !automaticAdmission && directRadiusAssign &&
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") != nullptr &&
        std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") != nullptr &&
        std::getenv("PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE") == nullptr;
    const bool exactDirectRadiusAssignExecutor =
        automaticRadiusAssign || requiredDirectRadiusAssignExecutor;
    const bool experimentalDirectClassificationOutput =
        std::getenv("PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT") != nullptr;
    bool directNeighborClassifierOutput =
        neighborClassifierEnvelope && experimentalDirectClassificationOutput;
    const bool directStandaloneOutlierOutput =
        !automaticAdmission && standaloneOutlierEnvelope &&
        experimentalDirectClassificationOutput;
    const bool directRadiusOutlierRadialDensityAssignOutput =
        radiusOutlierRadialDensityAssignEnvelope &&
        (automaticRadiusOutlierRadialDensity ||
         (!automaticAdmission &&
          (experimentalDirectClassificationOutput ||
           requireRadiusOutlierRadialDensityComposition)));
    bool directClassificationOutput =
        (outlierNnDistanceEnvelope &&
         (automaticOutlierNnDistance ||
          experimentalDirectClassificationOutput)) ||
        directNeighborClassifierOutput || directStandaloneOutlierOutput ||
        directRadiusOutlierRadialDensityAssignOutput;
    const bool requiredDirectStandaloneOutlierExecutor =
        directStandaloneOutlierOutput &&
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") != nullptr &&
        std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") != nullptr &&
        std::getenv("PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE") == nullptr;
    const bool requiredDirectOutlierNnDistanceExecutor =
        !automaticAdmission && outlierNnDistanceEnvelope &&
        directClassificationOutput &&
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") != nullptr &&
        std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") != nullptr &&
        std::getenv("PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE") == nullptr;
    const bool exactDirectOutlierNnDistanceExecutor =
        automaticOutlierNnDistance || requiredDirectOutlierNnDistanceExecutor;
    const bool requiredDirectRadiusOutlierRadialDensityExecutor =
        !automaticAdmission && directRadiusOutlierRadialDensityAssignOutput &&
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE") != nullptr &&
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") != nullptr &&
        std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") != nullptr &&
        std::getenv("PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE") == nullptr;
    const bool exactDirectRadiusOutlierRadialDensityExecutor =
        automaticRadiusOutlierRadialDensity ||
        requiredDirectRadiusOutlierRadialDensityExecutor;
    const bool requiredDirectApproximateCoplanarExecutor =
        !automaticAdmission &&
        std::getenv("PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT") != nullptr;
    if (automaticAdmission && requireAutomaticEigenFamily &&
        !automaticEigenFamily)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticRankOptimal &&
        !automaticRankOptimal)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticOutlierNnDistance &&
        !automaticOutlierNnDistance)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticRadiusOutlierRadialDensity &&
        !automaticRadiusOutlierRadialDensity)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticRadiusAssign &&
        !automaticRadiusAssign)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticNeighborClassifier &&
        !automaticNeighborClassifier)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticHagNn && !automaticHagNnCountOne)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticHagDelaunay &&
        !automaticHagDelaunayCountThree)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticSkewness && !automaticSkewness)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticSort && !automaticSort)
        return std::nullopt;
    if (automaticAdmission && requireAutomaticApproximateCoplanar &&
        !automaticApproximateCoplanar)
        return std::nullopt;
    if (requireDirectExtraDoubleOutput &&
        !experimentalDirectExtraDoubleOutputRequested)
        throw pdal::pdal_error(
            "required direct Extra Bytes binary64 output envelope was not "
            "used");
    if (requireRadiusOutlierRadialDensityComposition &&
        (automaticAdmission || !radiusOutlierRadialDensityAssignEnvelope))
        throw pdal::pdal_error(
            "required radius-outlier/radial-density resident composition "
            "was not used");
    if (requireDirectSkewnessComposition &&
        (automaticAdmission || !directSkewnessEnvelope))
    {
        if (automaticAdmission)
            return std::nullopt;
        throw pdal::pdal_error(
            "required direct skewness composition envelope was not used");
    }
    if (requireDirectSortComposition &&
        (automaticAdmission || !directSortEnvelope))
    {
        if (automaticAdmission)
            return std::nullopt;
        throw pdal::pdal_error(
            "required direct sort composition envelope was not used");
    }
    // B0181: automatic admission is no longer gated on a whitelist of exact
    // graph shapes, but it *is* ranked against the other accelerated path.
    //
    // B0180 measured that removing the whitelist outright recovers
    // `nndistance` (0.997x -> 7.338x) and `normal` + assign (0.996x -> 4.001x)
    // while destroying `assign` + `ferry`, which fell from 7.492x to 1.041x,
    // and `mortonorder`, which fell 28%. Those pipelines were already served
    // by the faster fused point-program lane, and an unranked generalization
    // lets the resident executor preempt it. Nothing became incorrect; the
    // planner simply chose a worse path.
    //
    // A pure point program -- no neighborhood query, no grid product -- is
    // therefore left to the fused lane, and the resident executor is admitted
    // only where that lane cannot serve the graph. Named envelopes still
    // matter below because they additionally enable direct source and
    // publication routes.
    //
    // Everything downstream stays fail-closed: placement unavailability,
    // preflight refusal, and any exception before commitment all return
    // nullopt so the unchanged public selector runs stock PDAL. B0156 made
    // that decline free by refusing on plan structure before device work, and
    // B0179 discharged the exactness burden for the class this newly reaches.
    static_cast<void>(&automaticResidentLasOutputEnvelope);

    bool committed = false;
    try
    {
        const std::optional<ResidentCommandClock::time_point>
            validationStarted =
                commandStarted
                    ? std::optional<ResidentCommandClock::time_point>(
                          ResidentCommandClock::now())
                    : std::nullopt;
        DimensionRegistry dimensions;
        const Plan plan = compilePipeline(*pipelineText, dimensions);
        admissionTrace.at("plan-compiled");
        const bool supportsDirectExtraDoubleOutput =
            supportsDirectResidentExtraDoubleOutput(plan);
        const bool experimentalDirectExtraDoubleOutput =
            (experimentalDirectExtraDoubleOutputRequested ||
             automaticHagNnCountOne || automaticHagDelaunayCountThree) &&
            supportsDirectExtraDoubleOutput;
        const bool experimentalDirectSkewness =
            experimentalDirectSkewnessComposition &&
            plan.stages().size() == 3U &&
            std::holds_alternative<SkewnessProgram>(
                plan.stages()[1U].payload) &&
            supportsDirectResidentLasOutput(plan, false, true);
        const bool experimentalDirectSort =
            (experimentalDirectSortComposition || automaticSort) &&
            plan.stages().size() == 3U &&
            std::holds_alternative<OrderingProgram>(
                plan.stages()[1U].payload) &&
            supportsDirectResidentLasOutput(plan, false, false, true);
        const bool experimentalDirectExtraDoubleNnDistance =
            experimentalDirectExtraDoubleOutput && plan.stages().size() == 3U &&
            std::holds_alternative<NnDistanceProgram>(
                plan.stages()[1U].payload);
        const bool experimentalDirectExtraDoubleHagNn =
            experimentalDirectExtraDoubleOutput && plan.stages().size() == 3U &&
            std::holds_alternative<HagNnProgram>(plan.stages()[1U].payload);
        const bool experimentalDirectExtraDoubleHagDelaunay =
            experimentalDirectExtraDoubleOutput && plan.stages().size() == 3U &&
            std::holds_alternative<HagDelaunayProgram>(
                plan.stages()[1U].payload);
        if (requireDirectExtraDoubleOutput && !supportsDirectExtraDoubleOutput)
            throw pdal::pdal_error(
                "required direct Extra Bytes binary64 output plan was not "
                "supported");
        if (requireDirectSkewnessComposition && !experimentalDirectSkewness)
            throw pdal::pdal_error(
                "required direct skewness composition plan was not "
                "supported");
        if (requireDirectSortComposition && !experimentalDirectSort)
            throw pdal::pdal_error(
                "required direct sort composition plan was not supported");
        // B0181: rank against the fused point-program lane. A plan with no
        // neighborhood query and no grid product is one the fused lane already
        // serves faster, so the resident executor declines it rather than
        // preempting it (B0180).
        if (automaticAdmission && !automaticHagNnCountOne &&
            !automaticHagDelaunayCountThree && !automaticSkewness &&
            !automaticSort && !planDeclaresNeighborhoodRegion(plan) &&
            !planDeclaresGridRegion(plan))
            return std::nullopt;
        static_cast<void>(automaticRadiusAssign);
        if (!statsDestination.empty())
        {
            const std::filesystem::path statsPath(statsDestination);
            std::error_code error;
            if (std::filesystem::is_symlink(
                    std::filesystem::symlink_status(statsPath, error)))
                throw pdal::pdal_error(
                    "Resident stats output must not be a symbolic link.");
            if (sameFilesystemPath(statsPath, argv[2]))
                throw pdal::pdal_error(
                    "Resident stats output aliases the pipeline definition.");
            if (statsAliasesPipelineFile(plan, statsPath))
                throw pdal::pdal_error(
                    "Resident stats output aliases pipeline input or output.");
        }

        pdal::PipelineManager validation;
        validation.setLog(pdal::Log::makeLog("PDAL", "devnull"));
        std::istringstream originalPipeline(*pipelineText);
        validation.readPipeline(originalPipeline);
        if (!validation.hasReader())
            throw pdal::pdal_error("Pipeline does not start with a reader.");
        validation.prepare();
        const std::size_t pointRecordBytes =
            validation.pointTable().layout()->pointSize();
        admissionTrace.at("original-pipeline-prepared");
        const std::optional<ResidentCommandClock::time_point>
            originalValidationEnded =
                validationStarted
                    ? std::optional<ResidentCommandClock::time_point>(
                          ResidentCommandClock::now())
                    : std::nullopt;

        RuntimePlacementResult runtime;
        std::optional<RuntimePlacementFacts> runtimeFacts;
        std::optional<std::size_t> validationBudgetOverride;
        const PlacementCalibrationProfile* profile = nullptr;
        std::optional<ResidentCommandClock::time_point> deviceAndProfileEnded;
        // D0218 derives the actual writer layout before placement.  The exact
        // `extra_dims=all` sink is therefore admissible here; direct-output
        // envelopes remain independently fail-closed on their source, layout,
        // and publication proofs below.
        const std::optional<LasHeaderFacts> las = lasHeaderFacts(plan, true);
        // B0156/D0214: refuse on plan structure before any device work. A
        // pipeline whose topology or cardinality can never be placed used to
        // pay full CUDA device and profile discovery, ~0.175 s, only to
        // decline afterwards. The reason reported is identical to the one
        // `buildRuntimePlacement` would have produced.
        admissionTrace.at("las-header-facts");
        const RuntimePlacementUnavailableReason structuralRefusal =
            planStructureRefusal(plan);
        admissionTrace.at("plan-structure-refusal");
        if (!las)
            runtime.unavailableReason =
                RuntimePlacementUnavailableReason::InvalidRuntimeFacts;
        else if (structuralRefusal != RuntimePlacementUnavailableReason::None)
            runtime.unavailableReason = structuralRefusal;
        else
        {
            try
            {
                const std::vector<CudaDeviceSummary> devices = cudaDevices();
                const std::string driver = nvidiaKernelDriverVersion();
                if (!devices.empty() && !driver.empty())
                {
                    const CudaDeviceSummary& device = devices.front();
                    const std::string capability =
                        std::to_string(device.computeMajor) + "." +
                        std::to_string(device.computeMinor);
                    const std::string toolkit =
                        formatCudaToolkitVersion(cudaCompiledToolkitVersion());
                    profile = placementCalibrationFor(
                        {.name = device.name,
                         .computeCapability = capability,
                         .driverVersion = driver,
                         .cudaToolkitVersion = toolkit});
                }
            }
            catch (const std::exception&)
            {
                profile = nullptr;
            }
            if (validationStarted)
                deviceAndProfileEnded = ResidentCommandClock::now();
            if (!profile)
                runtime.unavailableReason =
                    RuntimePlacementUnavailableReason::ProfileNotExact;
            else
            {
                const std::size_t freeDeviceBytes =
                    cudaCurrentDeviceFreeMemoryBytes();
                runtimeFacts.emplace();
                runtimeFacts->inputPointCount = las->pointCount;
                runtimeFacts->inputPointFormat = las->pointFormat;
                runtimeFacts->inputCompressed = las->compressedReader;
                runtimeFacts->outputCompressed = las->compressedWriter;
                runtimeFacts->inputRecordBytes = las->recordBytes;
                runtimeFacts->outputRecordBytes =
                    experimentalDirectExtraDoubleOutput
                        ? ExtraDoubleLasOutputRecordBytes
                        : DefaultLasOutputRecordBytes +
                              lasExtraDimensionBytes(
                                  plan, dimensions,
                                  validation.pointTable().layout(),
                                  las->pointFormat);
                runtimeFacts->fallbackRecordBytes = pointRecordBytes;
                runtimeFacts->tilePointCapacity = ResidentTilePoints;
                runtimeFacts->stageScratchBytes.assign(plan.stages().size(),
                                                       0U);
                runtimeFacts->stageAdditionalSynchronizations.assign(
                    plan.stages().size(), 0U);
                runtimeFacts->deviceMemoryBudgetBytes =
                    freeDeviceBytes - freeDeviceBytes / 4U;
                runtimeFacts->directApproximateCoplanarOutputExecutor =
                    (requiredDirectApproximateCoplanarExecutor ||
                     automaticApproximateCoplanar) &&
                    directApproximateCoplanarComposition(plan);
                const bool mappableRecords = !las->compressedReader;
                if (exactDirectRadiusAssignExecutor && mappableRecords)
                {
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directRadiusAssignBoundaryExecutor = true;
                    runtimeFacts->exactDirectRadiusAssignExecutor =
                        las->recordBytes == DefaultLasOutputRecordBytes;
                }
                else if (automaticHagNnCountOne && mappableRecords &&
                         las->pointFormat == 7U && las->recordBytes == 40U &&
                         !las->compressedWriter &&
                         runtimeFacts->outputRecordBytes ==
                             ExtraDoubleLasOutputRecordBytes)
                {
                    if (las->pointCount >
                            (std::numeric_limits<std::size_t>::max)() /
                                pdg::
                                    HagNnCountOneExactDeviceScratchBytesPerPoint ||
                        las->pointCount >
                            (std::numeric_limits<std::size_t>::max)() /
                                pdg::HagNnCountOneExactDevicePeakBytesPerPoint)
                        throw std::overflow_error(
                            "automatic HAG-NN count-one bytes overflow");
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directHagNnBoundaryExecutor = true;
                    runtimeFacts->stageScratchBytes[1U] =
                        las->pointCount *
                        pdg::HagNnCountOneExactDeviceScratchBytesPerPoint;
                }
                else if (automaticHagDelaunayCountThree && mappableRecords &&
                         las->pointFormat == 7U && las->recordBytes == 40U &&
                         !las->compressedWriter &&
                         runtimeFacts->outputRecordBytes ==
                             ExtraDoubleLasOutputRecordBytes)
                {
                    if (las->pointCount >
                            (std::numeric_limits<std::size_t>::max)() /
                                pdg::
                                    HagDelaunayCountThreeExactDeviceScratchBytesPerPoint ||
                        las->pointCount >
                            (std::numeric_limits<std::size_t>::max)() /
                                pdg::
                                    HagDelaunayCountThreeExactDevicePeakBytesPerPoint)
                        throw std::overflow_error(
                            "automatic HAG-Delaunay count-three bytes "
                            "overflow");
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directHagDelaunayBoundaryExecutor = true;
                    runtimeFacts->stageScratchBytes[1U] =
                        las->pointCount *
                        pdg::
                            HagDelaunayCountThreeExactDeviceScratchBytesPerPoint;
                }
                else if (automaticSkewness && mappableRecords &&
                         las->pointFormat == 7U &&
                         las->recordBytes == DefaultLasOutputRecordBytes &&
                         !las->compressedWriter &&
                         runtimeFacts->outputRecordBytes ==
                             DefaultLasOutputRecordBytes)
                {
                    if (las->pointCount >
                        (std::numeric_limits<std::size_t>::max)() /
                            pdg::SkewnessExactDeviceScratchBytesPerPoint)
                        throw std::overflow_error(
                            "automatic skewness scratch bytes overflow");
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    // Classification remains in the pinned host recurrence;
                    // only binary64 Z crosses the mapped-source upload.
                    runtimeFacts->boundaryExecutionFacts.front()
                        .transferBytesPerPoint = sizeof(double);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directSkewnessBoundaryExecutor = true;
                    runtimeFacts->stageScratchBytes[1U] =
                        las->pointCount *
                        pdg::SkewnessExactDeviceScratchBytesPerPoint;
                }
                else if (automaticSort && mappableRecords &&
                         las->pointFormat == 7U &&
                         las->recordBytes == DefaultLasOutputRecordBytes &&
                         !las->compressedWriter &&
                         runtimeFacts->outputRecordBytes ==
                             DefaultLasOutputRecordBytes)
                {
                    if (las->pointCount >
                        (std::numeric_limits<std::size_t>::max)() /
                            pdg::OrderingExactDeviceScratchBytesPerPoint)
                        throw std::overflow_error(
                            "automatic sort scratch bytes overflow");
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directSortBoundaryExecutor = true;
                    runtimeFacts->stageScratchBytes[1U] =
                        las->pointCount *
                        pdg::OrderingExactDeviceScratchBytesPerPoint;
                }
                else if (automaticNeighborClassifier && mappableRecords &&
                         las->pointFormat == 7U &&
                         las->recordBytes == DefaultLasOutputRecordBytes &&
                         !las->compressedWriter &&
                         runtimeFacts->outputRecordBytes ==
                             DefaultLasOutputRecordBytes)
                {
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directNeighborClassifierBoundaryExecutor =
                        true;
                }
                else if (exactDirectOutlierNnDistanceExecutor &&
                         mappableRecords)
                {
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts->directOutlierNnDistanceBoundaryExecutor =
                        true;
                }
                else if (exactDirectRadiusOutlierRadialDensityExecutor &&
                         mappableRecords)
                {
                    runtimeFacts->tilePointCapacity = las->pointCount;
                    runtimeFacts->boundaryExecutionFacts =
                        directResidentLasBoundaryFacts(plan, las->recordBytes);
                    runtimeFacts->executorLaneCount = 1U;
                    runtimeFacts
                        ->directRadiusOutlierRadialDensityBoundaryExecutor =
                        true;
                }
                // The automatic direct-sort route has a measured whole-view
                // allocation high-water and is selected only after the
                // placement budget accepts that reservation.  Apply the
                // validation-only budget before its first placement pass so
                // the process gate can freeze both sides of that boundary.
                if (runtimeFacts->directHagNnBoundaryExecutor ||
                    runtimeFacts->directHagDelaunayBoundaryExecutor ||
                    runtimeFacts->directSkewnessBoundaryExecutor ||
                    runtimeFacts->directSortBoundaryExecutor)
                {
                    validationBudgetOverride = residentValidationBudgetOverride(
                        runtimeFacts->deviceMemoryBudgetBytes);
                    if (validationBudgetOverride)
                        runtimeFacts->deviceMemoryBudgetBytes =
                            *validationBudgetOverride;
                }
                runtime = buildRuntimePlacement(plan, *runtimeFacts, *profile);
            }
        }
        if (validationStarted && !deviceAndProfileEnded)
            deviceAndProfileEnded = ResidentCommandClock::now();
        const std::optional<ResidentCommandClock::time_point>
            initialPlacementEnded =
                validationStarted
                    ? std::optional<ResidentCommandClock::time_point>(
                          ResidentCommandClock::now())
                    : std::nullopt;

        // B0107 is an explicit feasibility force, not a calibrated selector.
        // Build only the bounded facts needed to exercise its one resident
        // region when this machine has no exact placement profile. Ordinary
        // and automatic execution continue to require the profile above.
        if ((experimentalDirectExtraDoubleOutput ||
             (experimentalDirectSkewness && !automaticAdmission) ||
             (experimentalDirectSort && !automaticAdmission)) &&
            las && !runtimeFacts)
        {
            const std::size_t freeDeviceBytes =
                cudaCurrentDeviceFreeMemoryBytes();
            runtimeFacts.emplace();
            runtimeFacts->inputPointCount = las->pointCount;
            runtimeFacts->inputPointFormat = las->pointFormat;
            runtimeFacts->inputCompressed = las->compressedReader;
            runtimeFacts->outputCompressed = las->compressedWriter;
            runtimeFacts->inputRecordBytes = las->recordBytes;
            runtimeFacts->outputRecordBytes =
                experimentalDirectExtraDoubleOutput
                    ? ExtraDoubleLasOutputRecordBytes
                    : DefaultLasOutputRecordBytes +
                          lasExtraDimensionBytes(
                              plan, dimensions,
                              validation.pointTable().layout(),
                              las->pointFormat);
            runtimeFacts->fallbackRecordBytes = pointRecordBytes;
            runtimeFacts->tilePointCapacity = ResidentTilePoints;
            runtimeFacts->stageScratchBytes.assign(plan.stages().size(), 0U);
            runtimeFacts->stageAdditionalSynchronizations.assign(
                plan.stages().size(), 0U);
            runtimeFacts->deviceMemoryBudgetBytes =
                freeDeviceBytes - freeDeviceBytes / 4U;
        }

        ExecutionObservationScope observation;
        std::optional<DirectResidentLasResult> direct;
        if (runtime.available())
            direct =
                tryDirectResidentLas(plan, dimensions, runtime.estimate,
                                     runtime.request.deviceMemoryBudgetBytes);
        // B0043 qualifies the planner-owned resident LOF path, not the
        // independent fused LAS executor. If that executor's envelope grows
        // later, require separate end-to-end evidence before selecting it
        // through this automatic admission point.
        if (automaticAdmission && direct)
            return std::nullopt;
        if (!direct && runtime.available() && runtimeFacts && profile &&
            !exactDirectRadiusAssignExecutor &&
            !runtimeFacts->directNeighborClassifierBoundaryExecutor &&
            !runtimeFacts->directHagNnBoundaryExecutor &&
            !runtimeFacts->directHagDelaunayBoundaryExecutor &&
            !runtimeFacts->directSkewnessBoundaryExecutor &&
            !runtimeFacts->directSortBoundaryExecutor &&
            !exactDirectOutlierNnDistanceExecutor &&
            !exactDirectRadiusOutlierRadialDensityExecutor &&
            residentPointViewExecutorSupports(*pipelineText, plan))
        {
            validationBudgetOverride = residentValidationBudgetOverride(
                runtimeFacts->deviceMemoryBudgetBytes);
            if (validationBudgetOverride)
                runtimeFacts->deviceMemoryBudgetBytes =
                    *validationBudgetOverride;
            runtimeFacts->boundaryExecutionFacts =
                residentPointViewBoundaryFacts(plan, pointRecordBytes);
            runtimeFacts->executorLaneCount =
                fixedLaneCount(planDeclaresResidentCardinalityChange(plan)
                                   ? PipelineClass::OrderedPointProgram
                                   : PipelineClass::FusedPointProgram);
            runtime = buildRuntimePlacement(plan, *runtimeFacts, *profile);
        }
        if (automaticEigenFamily &&
            !selectedAutomaticEigenFamilyPlacement(runtime))
            return std::nullopt;
        if (automaticAdmission && requireAutomaticNormalCovariance &&
            !selectedAutomaticNormalCovariancePlacement(runtime))
            return std::nullopt;
        if (automaticRankOptimal &&
            !selectedAutomaticRankOptimalPlacement(runtime))
            return std::nullopt;
        if (automaticOutlierNnDistance &&
            !selectedAutomaticOutlierNnDistancePlacement(runtime))
            return std::nullopt;
        if (automaticRadiusOutlierRadialDensity &&
            !selectedAutomaticRadiusOutlierRadialDensityPlacement(runtime))
            return std::nullopt;
        if (automaticRadiusAssign &&
            !selectedAutomaticRadiusAssignPlacement(runtime))
            return std::nullopt;
        const bool selectedAutomaticNeighborClassifier =
            automaticNeighborClassifier && runtimeFacts &&
            runtimeFacts->directNeighborClassifierBoundaryExecutor &&
            selectedAutomaticNeighborClassifierPlacement(runtime);
        if (requireAutomaticNeighborClassifier &&
            !selectedAutomaticNeighborClassifier)
            return std::nullopt;
        if (selectedAutomaticNeighborClassifier)
        {
            directNeighborClassifierOutput = true;
            directClassificationOutput = true;
        }
        const bool selectedAutomaticHagNnCountOne =
            automaticHagNnCountOne && runtimeFacts &&
            runtimeFacts->directHagNnBoundaryExecutor &&
            selectedAutomaticHagNnCountOnePlacement(runtime);
        if (automaticHagNnCountOne && !selectedAutomaticHagNnCountOne)
            return std::nullopt;
        if (requireAutomaticHagNn && !selectedAutomaticHagNnCountOne)
            return std::nullopt;
        const bool selectedAutomaticHagDelaunayCountThree =
            automaticHagDelaunayCountThree && runtimeFacts &&
            runtimeFacts->directHagDelaunayBoundaryExecutor &&
            selectedAutomaticHagDelaunayCountThreePlacement(runtime);
        if (automaticHagDelaunayCountThree &&
            !selectedAutomaticHagDelaunayCountThree)
            return std::nullopt;
        if (requireAutomaticHagDelaunay &&
            !selectedAutomaticHagDelaunayCountThree)
            return std::nullopt;
        const bool selectedAutomaticSkewness =
            automaticSkewness && runtimeFacts &&
            runtimeFacts->directSkewnessBoundaryExecutor &&
            selectedAutomaticSkewnessPlacement(runtime);
        if (automaticSkewness && !selectedAutomaticSkewness)
            return std::nullopt;
        if (requireAutomaticSkewness && !selectedAutomaticSkewness)
            return std::nullopt;
        const bool selectedAutomaticSort =
            automaticSort && runtimeFacts &&
            runtimeFacts->directSortBoundaryExecutor &&
            selectedAutomaticSortPlacement(runtime);
        if (automaticSort && !selectedAutomaticSort)
            return std::nullopt;
        if (requireAutomaticSort && !selectedAutomaticSort)
            return std::nullopt;
        if (automaticApproximateCoplanar &&
            !selectedAutomaticApproximateCoplanarPlacement(runtime))
            return std::nullopt;
        // B0068's exact-shape opt-in deliberately bypasses the uncalibrated
        // placement result so its reusable source/output boundary can be
        // measured and exercised. Automatic admission never reaches this
        // branch, and the public planner remains authoritative otherwise.
        const bool exactDirectCalibratedClassificationExecutor =
            exactDirectOutlierNnDistanceExecutor ||
            exactDirectRadiusOutlierRadialDensityExecutor ||
            selectedAutomaticNeighborClassifier;
        const bool forceDirectClassificationOutput =
            directClassificationOutput &&
            (!exactDirectCalibratedClassificationExecutor ||
             (!automaticAdmission &&
              (!runtime.available() ||
               runtime.estimate.choice != PlacementChoice::Device)));
        const bool forceDirectResidentComposition =
            forceDirectClassificationOutput ||
            experimentalDirectRadialDensityAssign ||
            (experimentalDirectExtraDoubleOutput &&
             !selectedAutomaticHagNnCountOne &&
             !selectedAutomaticHagDelaunayCountThree) ||
            (experimentalDirectSkewness && !selectedAutomaticSkewness) ||
            (experimentalDirectSort && !selectedAutomaticSort);
        if (forceDirectResidentComposition && runtimeFacts &&
            !validationBudgetOverride)
        {
            validationBudgetOverride = residentValidationBudgetOverride(
                runtimeFacts->deviceMemoryBudgetBytes);
            if (validationBudgetOverride)
                runtimeFacts->deviceMemoryBudgetBytes =
                    *validationBudgetOverride;
        }
        if (forceDirectResidentComposition && runtimeFacts &&
            plan.summary().residentRegions == 1U)
        {
            if (experimentalDirectRadialDensityAssign ||
                experimentalDirectExtraDoubleOutput ||
                experimentalDirectSkewness || experimentalDirectSort ||
                requiredDirectStandaloneOutlierExecutor ||
                directRadiusOutlierRadialDensityAssignOutput)
            {
                runtimeFacts->boundaryExecutionFacts =
                    directResidentLasBoundaryFacts(
                        plan, runtimeFacts->inputRecordBytes);
                runtimeFacts->executorLaneCount = 1U;
            }
            else if (!exactDirectOutlierNnDistanceExecutor)
            {
                runtimeFacts->boundaryExecutionFacts =
                    residentPointViewBoundaryFacts(plan, pointRecordBytes);
                if (!runtimeFacts->executorLaneCount)
                    runtimeFacts->executorLaneCount =
                        fixedLaneCount(PipelineClass::FusedPointProgram);
            }
            PlacementRegionEstimate forcedRegion;
            forcedRegion.residentRegion = 0U;
            for (const PlannedStage& stage : plan.stages())
                if (stage.native &&
                    stage.preferredResidency == MemoryKind::Device &&
                    stage.residentRegion == 0U)
                    forcedRegion.stageIds.push_back(stage.id);
            if (!forcedRegion.stageIds.empty())
            {
                forcedRegion.selected = true;
                PlanPlacementEstimate forced;
                forced.choice = PlacementChoice::Device;
                forced.reason = PlacementReason::UncalibratedStage;
                forced.selectedRegionCount = 1U;
                const auto bytesFor =
                    [](std::size_t bytesPerPoint, std::size_t points)
                {
                    if (bytesPerPoint &&
                        points > (std::numeric_limits<std::size_t>::max)() /
                                     bytesPerPoint)
                        throw std::overflow_error(
                            "experimental resident boundary bytes overflow");
                    return bytesPerPoint * points;
                };
                const auto addBytes = [](std::size_t& total, std::size_t bytes)
                {
                    if (bytes >
                        (std::numeric_limits<std::size_t>::max)() - total)
                        throw std::overflow_error(
                            "experimental resident boundary sum overflows");
                    total += bytes;
                };
                const std::size_t points = runtimeFacts->inputPointCount;
                if (experimentalDirectSkewness || experimentalDirectSort)
                {
                    const std::size_t planned =
                        plan.estimatedDeviceBytes(points);
                    const std::size_t scratch = bytesFor(
                        experimentalDirectSkewness
                            ? pdg::SkewnessExactDeviceScratchBytesPerPoint
                            : pdg::OrderingExactDeviceScratchBytesPerPoint,
                        points);
                    forced.untiledDeviceBytes = planned;
                    forced.peakDeviceBytes = planned;
                    addBytes(forced.peakDeviceBytes, scratch);
                    forced.configuredDeviceLaneCount = 1U;
                    forced.activeDeviceLaneCount = points ? 1U : 0U;
                }
                for (const std::size_t stageId : forcedRegion.stageIds)
                {
                    const PlannedStage& selected = plan.stages().at(stageId);
                    addBytes(
                        forced.stageResultBytes,
                        bytesFor(
                            selected.descriptor.deviceToHostBytesPerInputPoint,
                            points));
                    addBytes(forced.stageResultBytes,
                             selected.descriptor.deviceToHostFixedBytes);
                }
                for (std::size_t boundaryId = 0;
                     boundaryId < plan.summary().residencyBoundaries.size();
                     ++boundaryId)
                {
                    const ResidencyBoundary& boundary =
                        plan.summary().residencyBoundaries[boundaryId];
                    const PlacementBoundaryExecutionFact& fact =
                        runtimeFacts->boundaryExecutionFacts.at(boundaryId);
                    forced.boundaries.push_back(
                        {.boundaryId = boundaryId,
                         .pointCount = points,
                         .logicalColumnBytes =
                             bytesFor(boundary.bytesPerPoint, points),
                         .fullRecordBytes =
                             bytesFor(fact.deviceStagingBytesPerPoint, points),
                         .predictedTransferBytes =
                             bytesFor(fact.transferBytesPerPoint, points),
                         .predictedPackingBytes =
                             bytesFor(fact.packingBytesPerPoint, points)});
                    const std::size_t transferBytes =
                        bytesFor(fact.transferBytesPerPoint, points);
                    if (boundary.kind == ResidencyBoundaryKind::Upload)
                        addBytes(forced.hostToDeviceBytes, transferBytes);
                    else
                        addBytes(forced.deviceToHostBytes, transferBytes);
                    addBytes(forced.packingBytes,
                             bytesFor(fact.packingBytesPerPoint, points));
                    ++forced.synchronizationCount;
                }
                forced.regions.push_back(std::move(forcedRegion));
                runtime.estimate = std::move(forced);
                runtime.request.deviceMemoryBudgetBytes =
                    runtimeFacts->deviceMemoryBudgetBytes;
                runtime.request.boundaryExecutionFacts =
                    runtimeFacts->boundaryExecutionFacts;
                runtime.request.executorLaneCount =
                    runtimeFacts->executorLaneCount;
                runtime.unavailableReason =
                    RuntimePlacementUnavailableReason::None;
            }
        }
        if (requireDirectExtraDoubleOutput &&
            (!runtime.available() ||
             runtime.estimate.choice != PlacementChoice::Device ||
             runtime.estimate.selectedRegionCount != 1U))
            throw pdal::pdal_error(
                "required direct Extra Bytes binary64 output placement was "
                "not selected");
        const std::optional<ResidentCommandClock::time_point>
            runtimePlacementEnded =
                validationStarted
                    ? std::optional<ResidentCommandClock::time_point>(
                          ResidentCommandClock::now())
                    : std::nullopt;

        ResidentPipelineRewrite rewrite;
        ExecutionStatsSnapshot actual;
        std::optional<TiledSchedule> residentSchedule;
        std::optional<std::size_t> residentOutputPointCount;
        std::optional<pdal::pdg_detail::ResidentPhaseSeconds> residentPhases;
        std::optional<pdal::pdg_detail::ResidentManagerPhaseSeconds>
            managerPhases;
        std::optional<ResidentCommandPhaseSeconds> commandPhases;
        bool directResidentLasOutput = false;
        bool directResidentLasSource = false;
        bool directExtraDoubleOutput = false;
        bool directPermutedClassificationOutput = false;
        bool directPermutedSortOutput = false;
        bool directLasRecordSummary = false;
        bool directLasHostXyzMirror = false;
        bool nnDistanceDeviceOnlyHandoff = false;
        bool nnDistanceHostRestore = false;
        bool nnDistanceAssignmentDeviceColumnReuse = false;
        bool knnGatherReuse = false;
        bool residentAssignmentExecuted = false;
        bool hagDelaunayCudaUsed = false;
        std::unique_ptr<MappedInput> directSourceMapping;
        std::unique_ptr<pdg::las::FileView> directSourceFile;
        if (direct)
        {
            rewrite.executable = true;
            rewrite.reason = "direct executor; no pipeline rewrite";
            rewrite.preflightReason =
                "direct executor uses its independent fail-closed envelope";
            actual = observation.snapshot();
        }
        else
        {
            if (runtime.available())
                rewrite = rewriteResidentPlacement(*pipelineText, plan,
                                                   runtime.estimate);
            else
            {
                rewrite.json = *pipelineText;
                rewrite.executable = true;
                rewrite.reason =
                    "placement unavailable; original host pipeline";
            }
            if (!rewrite.executable)
            {
                rewrite.json = *pipelineText;
                rewrite.selectedRegions.clear();
                rewrite.selectedStageIds.clear();
            }

            const bool tryDirectResidentLasOutput =
                (automaticAdmission && !automaticEigenFamily &&
                 !automaticRankOptimal) ||
                requireDirectResidentLasOutput ||
                experimentalDirectExtraDoubleOutput ||
                experimentalDirectSkewness || experimentalDirectSort ||
                std::getenv("PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT") !=
                    nullptr;
            const bool planContainsSkewness =
                std::any_of(plan.stages().begin(), plan.stages().end(),
                            [](const PlannedStage& stage)
                            {
                                return std::holds_alternative<SkewnessProgram>(
                                    stage.payload);
                            });
            const bool planContainsOrdering =
                std::any_of(plan.stages().begin(), plan.stages().end(),
                            [](const PlannedStage& stage)
                            {
                                return std::holds_alternative<OrderingProgram>(
                                    stage.payload);
                            });
            // B0152/D0213: the direct publisher emits raw LAS records, so it
            // cannot serve a compressed sink. A LAZ output still accelerates —
            // it just publishes through PDAL's unchanged writer instead.
            if (tryDirectResidentLasOutput && las && !las->compressedWriter &&
                (!planContainsSkewness || experimentalDirectSkewness) &&
                (!planContainsOrdering || experimentalDirectSort) &&
                !rewrite.selectedRegions.empty() &&
                supportsDirectResidentLasOutput(
                    plan,
                    directRadiusAssign || directNeighborClassifierOutput ||
                        directStandaloneOutlierOutput,
                    experimentalDirectSkewness, experimentalDirectSort))
            {
                directExtraDoubleOutput =
                    supportsDirectResidentExtraDoubleOutput(plan);
                const std::optional<std::string> withoutWriter =
                    withoutTerminalLasWriter(rewrite.json,
                                             directExtraDoubleOutput ||
                                                 experimentalDirectSkewness ||
                                                 experimentalDirectSort);
                if (withoutWriter)
                {
                    rewrite.json = *withoutWriter;
                    directResidentLasOutput = true;
                    directPermutedClassificationOutput =
                        experimentalDirectSkewness;
                    directPermutedSortOutput = experimentalDirectSort;
                }
            }
            // B0181: a plan without a direct output route publishes
            // canonically and remains admissible.
            static_cast<void>(directResidentLasOutput);
            if (requireDirectExtraDoubleOutput &&
                (!directResidentLasOutput ||
                 !supportsDirectResidentExtraDoubleOutput(plan)))
                throw pdal::pdal_error(
                    "required direct Extra Bytes binary64 output path was "
                    "not used");

            // B0151/D0213: the direct source memory-maps raw point records, so
            // it can never run against a compressed reader however the route
            // was requested.
            const bool tryDirectResidentLasSource =
                !std::getenv("PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE") && las &&
                !las->compressedReader &&
                (automaticAdmission || automaticHagNnCountOne ||
                 automaticHagDelaunayCountThree ||
                 requireDirectResidentLasSource || experimentalDirectSkewness ||
                 experimentalDirectSort ||
                 std::getenv("PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE") !=
                     nullptr);
            if (tryDirectResidentLasSource && directResidentLasOutput && las &&
                (automaticResidentNnDistanceEnvelope(*pipelineText) ||
                 directClassificationOutput || directRadiusAssign ||
                 experimentalDirectRadialDensityAssign ||
                 experimentalDirectExtraDoubleNnDistance ||
                 experimentalDirectExtraDoubleHagNn ||
                 experimentalDirectExtraDoubleHagDelaunay ||
                 directPermutedClassificationOutput ||
                 directPermutedSortOutput))
            {
                try
                {
                    const auto* reader = std::get_if<FileStagePlan>(
                        &plan.stages().front().payload);
                    if (!reader)
                        throw pdal::pdal_error(
                            "direct resident LAS source has no input plan");
                    directSourceMapping = std::make_unique<MappedInput>(
                        std::filesystem::path(reader->filename));
                    directSourceFile = std::make_unique<pdg::las::FileView>(
                        directSourceMapping->bytes());
                    const bool supportedSource =
                        (experimentalDirectExtraDoubleNnDistance ||
                         experimentalDirectExtraDoubleHagNn ||
                         experimentalDirectExtraDoubleHagDelaunay)
                            ? pdg::las::supportsExtraDoubleTranslation(
                                  *directSourceFile)
                            : pdg::las::supportsDefaultTranslation(
                                  *directSourceFile);
                    if (!supportedSource ||
                        directSourceFile->header().pointCount !=
                            las->pointCount)
                        throw pdal::pdal_error(
                            "direct resident LAS source is outside the exact "
                            "translation envelope");
                    const std::optional<std::string> withSource =
                        withDirectResidentLasSourceReader(
                            rewrite.json, las->pointCount, directRadiusAssign);
                    if (!withSource)
                        throw pdal::pdal_error(
                            "direct resident LAS source rewrite failed");
                    rewrite.json = *withSource;
                    directResidentLasSource = true;
                }
                catch (const std::exception&)
                {
                    directSourceFile.reset();
                    directSourceMapping.reset();
                    if (requireDirectResidentLasSource)
                        throw;
                }
            }
            if ((experimentalDirectSkewness || experimentalDirectSort) &&
                !directResidentLasSource && !requireDirectResidentLasSource)
            {
                rewrite.json = *pipelineText;
                rewrite.selectedRegions.clear();
                rewrite.selectedStageIds.clear();
                directResidentLasOutput = false;
                directExtraDoubleOutput = false;
                directPermutedClassificationOutput = false;
                directPermutedSortOutput = false;
                directSourceFile.reset();
                directSourceMapping.reset();
            }
            if (requireDirectResidentLasSource && !directResidentLasSource)
                throw pdal::pdal_error(
                    "required direct LAS resident source path was not used");
            if (automaticOutlierNnDistance && !directResidentLasSource)
                return std::nullopt;
            if (automaticRadiusOutlierRadialDensity && !directResidentLasSource)
                return std::nullopt;
            if (automaticRadiusAssign && !directResidentLasSource)
                return std::nullopt;
            if (selectedAutomaticNeighborClassifier && !directResidentLasSource)
                return std::nullopt;
            if (selectedAutomaticHagNnCountOne && !directResidentLasSource)
                return std::nullopt;
            if (selectedAutomaticHagDelaunayCountThree &&
                !directResidentLasSource)
                return std::nullopt;
            if (selectedAutomaticSkewness && !directResidentLasSource)
                return std::nullopt;
            if (selectedAutomaticSort && !directResidentLasSource)
                return std::nullopt;

            std::optional<pdal::pdg_detail::ResidentExecutionScope>
                residentScope;
            if (!rewrite.selectedRegions.empty())
            {
                rewrite.preflightAttempted = true;
                try
                {
                    pdal::PipelineManager preflight(ResidentTilePoints);
                    preflight.setLog(pdal::Log::makeLog(
                        "gpupdal resident preflight", "devnull"));
                    std::istringstream candidatePipeline(rewrite.json);
                    preflight.readPipeline(candidatePipeline);
                    if (!preflight.hasReader())
                        throw pdal::pdal_error(
                            "Pipeline does not start with a reader.");
                    preflight.prepare();
                    if (automaticAdmission &&
                        std::getenv(
                            "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"))
                        throw pdal::pdal_error(
                            "injected automatic resident preflight failure");
                    if (experimentalDirectSkewness &&
                        std::getenv(
                            "PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE"))
                        throw pdal::pdal_error(
                            "injected direct skewness preflight failure");
                    if (experimentalDirectSort &&
                        std::getenv("PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE"))
                        throw pdal::pdal_error(
                            "injected direct sort preflight failure");
                    if (!las)
                        throw pdal::pdal_error(
                            "resident preflight has no bounded point count");
                    residentScope.emplace(
                        plan, dimensions,
                        runtime.request.deviceMemoryBudgetBytes,
                        ResidentTilePoints);
                    residentScope->preflight(*preflight.pointTable().layout(),
                                             las->pointCount,
                                             rewrite.selectedRegions);
                    rewrite.preflightAccepted = true;
                    rewrite.preflightReason =
                        "rewritten graph, exact envelope, layout, schedule, "
                        "and lane allocation probe accepted";
                }
                catch (const std::exception& exception)
                {
                    residentScope.reset();
                    rewrite.preflightAccepted = false;
                    rewrite.preflightReason = exception.what();
                    rewrite.reason =
                        "resident preflight rejected; original host pipeline";
                    rewrite.json = *pipelineText;
                    rewrite.selectedRegions.clear();
                    rewrite.selectedStageIds.clear();
                    directResidentLasOutput = false;
                    directResidentLasSource = false;
                    directExtraDoubleOutput = false;
                    directPermutedClassificationOutput = false;
                    directPermutedSortOutput = false;
                    directSourceFile.reset();
                    directSourceMapping.reset();
                }
            }
            else
                rewrite.preflightReason =
                    "placement selected no executable resident region";

            if (requireDirectResidentLasOutput &&
                (!directResidentLasOutput || !residentScope))
                throw pdal::pdal_error(
                    "required direct resident LAS output path was not used");
            if (requireDirectResidentLasSource &&
                (!directResidentLasSource || !residentScope))
                throw pdal::pdal_error(
                    "required direct LAS resident source path was not used");
            const bool selectedStatisticalOutlier = std::any_of(
                rewrite.selectedStageIds.begin(),
                rewrite.selectedStageIds.end(),
                [&](std::size_t stageId)
                {
                    if (stageId >= plan.stages().size())
                        return false;
                    const auto* outlier = std::get_if<OutlierProgram>(
                        &plan.stages()[stageId].payload);
                    return outlier &&
                           outlier->method == OutlierMethod::Statistical;
                });
            const bool selectedNnDistance = std::any_of(
                rewrite.selectedStageIds.begin(),
                rewrite.selectedStageIds.end(),
                [&](std::size_t stageId)
                {
                    return stageId < plan.stages().size() &&
                           std::holds_alternative<NnDistanceProgram>(
                               plan.stages()[stageId].payload);
                });
            const bool selectedLof = std::any_of(
                rewrite.selectedStageIds.begin(),
                rewrite.selectedStageIds.end(),
                [&](std::size_t stageId)
                {
                    return stageId < plan.stages().size() &&
                           std::holds_alternative<LofProgram>(
                               plan.stages()[stageId].payload);
                });
            if ((requireNnDistanceDeviceRepair ||
                 requireNnDistanceParallelRepair) &&
                (!residentScope || !selectedNnDistance))
                throw pdal::pdal_error(
                    "required NNDistance device repair path was not "
                    "selected");
            if ((requireOutlierDeviceRepair || requireOutlierParallelRepair) &&
                (!residentScope || !selectedStatisticalOutlier))
                throw pdal::pdal_error(
                    "required statistical-outlier device repair path was "
                    "not selected");
            if ((requireLofParallelRepair || requireLofCoordinateCache) &&
                (!residentScope || !selectedLof))
                throw pdal::pdal_error(
                    "required LOF repair path was not selected");
            if (!residentScope)
            {
                if (automaticAdmission)
                    return std::nullopt;
            }
            if (requireAutomaticEigenFamily &&
                (!automaticEigenFamily || directResidentLasOutput ||
                 !residentScope))
                return std::nullopt;
            if (requireAutomaticNormalCovariance &&
                (!automaticAdmission || directResidentLasOutput ||
                 !selectedAutomaticNormalCovariancePlacement(runtime) ||
                 !residentScope || rewrite.selectedRegions.size() != 1U))
                return std::nullopt;
            if (requireAutomaticRankOptimal &&
                (!automaticRankOptimal || directResidentLasOutput ||
                 !residentScope))
                return std::nullopt;
            if (requireAutomaticOutlierNnDistance &&
                (!automaticOutlierNnDistance || !directResidentLasOutput ||
                 !residentScope))
                return std::nullopt;
            if (requireAutomaticRadiusOutlierRadialDensity &&
                (!automaticRadiusOutlierRadialDensity ||
                 !directResidentLasOutput || !residentScope))
                return std::nullopt;
            if (requireAutomaticRadiusAssign &&
                (!automaticRadiusAssign || !directResidentLasOutput ||
                 !residentScope))
                return std::nullopt;
            if (requireAutomaticNeighborClassifier &&
                (!selectedAutomaticNeighborClassifier ||
                 !directResidentLasOutput || !directResidentLasSource ||
                 !residentScope || rewrite.selectedRegions.size() != 1U))
                return std::nullopt;
            if (requireAutomaticHagNn &&
                (!selectedAutomaticHagNnCountOne || !directResidentLasOutput ||
                 !directExtraDoubleOutput || !directResidentLasSource ||
                 !residentScope || rewrite.selectedRegions.size() != 1U ||
                 !rewrite.executable))
                return std::nullopt;
            if (requireAutomaticHagDelaunay &&
                (!selectedAutomaticHagDelaunayCountThree ||
                 !directResidentLasOutput || !directExtraDoubleOutput ||
                 !directResidentLasSource || !residentScope ||
                 rewrite.selectedRegions.size() != 1U || !rewrite.executable))
                return std::nullopt;
            if (requireAutomaticSkewness &&
                (!selectedAutomaticSkewness || !directResidentLasOutput ||
                 !directPermutedClassificationOutput ||
                 !directResidentLasSource || !residentScope ||
                 rewrite.selectedRegions.size() != 1U || !rewrite.executable))
                return std::nullopt;
            if (requireAutomaticSort &&
                (!selectedAutomaticSort || !directResidentLasOutput ||
                 !directPermutedSortOutput || !directResidentLasSource ||
                 !residentScope || rewrite.selectedRegions.size() != 1U ||
                 !rewrite.executable))
                return std::nullopt;
            if (requireAutomaticApproximateCoplanar &&
                (!automaticApproximateCoplanar || !directResidentLasOutput ||
                 !residentScope))
                return std::nullopt;

            if (validationStarted)
            {
                const ResidentCommandClock::time_point preflightEnded =
                    ResidentCommandClock::now();
                commandPhases.emplace();
                commandPhases->validationPlacementPreflight =
                    elapsedSeconds(*validationStarted, preflightEnded);
                commandPhases->planAndOriginalValidation = elapsedSeconds(
                    *validationStarted, *originalValidationEnded);
                commandPhases->runtimePlacement = elapsedSeconds(
                    *originalValidationEnded, *runtimePlacementEnded);
                commandPhases->runtimeDeviceAndProfile = elapsedSeconds(
                    *originalValidationEnded, *deviceAndProfileEnded);
                commandPhases->runtimeInitialPlacement = elapsedSeconds(
                    *deviceAndProfileEnded, *initialPlacementEnded);
                commandPhases->runtimeExecutorSelection = elapsedSeconds(
                    *initialPlacementEnded, *runtimePlacementEnded);
                commandPhases->rewriteAndResidentPreflight =
                    elapsedSeconds(*runtimePlacementEnded, preflightEnded);
            }

            const std::optional<ResidentCommandClock::time_point>
                managerStarted =
                    commandPhases
                        ? std::optional<ResidentCommandClock::time_point>(
                              ResidentCommandClock::now())
                        : std::nullopt;
            pdal::PipelineManager manager;
            // Stage diagnostics are part of the byte-exact contract, and
            // upstream's log leader names the pdal pipeline application. A
            // resident-selected execution must reproduce a warning-emitting
            // stage's stderr bytes exactly, leader included.
            manager.setLog(pdal::Log::makeLog("pdal pipeline", "stderr"));
            std::istringstream executionPipeline(rewrite.json);
            manager.readPipeline(executionPipeline);
            if (!manager.hasReader())
                throw pdal::pdal_error(
                    "Pipeline does not start with a reader.");
            if (!selectedAutomaticHagNnCountOne &&
                !selectedAutomaticHagDelaunayCountThree &&
                !selectedAutomaticSkewness && !selectedAutomaticSort)
                committed = true;
            if (residentScope)
            {
                // Ordinary resident execution uses physical rows. The bounded
                // direct-LAS source instead supplies stable point identity,
                // original UserData, and on-demand XYZ while its mapped bytes
                // hydrate the planner-owned neighborhood product.
                manager.validateStageOptions();
                pdal::Stage* terminal = manager.getStage();
                if (!terminal)
                    throw pdal::pdal_error("Couldn't run resident pipeline.");
                std::unique_ptr<pdal::BasePointTable> executionTable;
                if (directResidentLasSource)
                {
                    if (!directSourceFile)
                        throw pdal::pdal_error(
                            "direct resident LAS source lost its mapping");
                    executionTable = std::make_unique<DirectResidentPointTable>(
                        *directSourceFile,
                        directClassificationOutput ||
                            directPermutedClassificationOutput ||
                            experimentalDirectExtraDoubleHagNn ||
                            experimentalDirectExtraDoubleHagDelaunay,
                        directRadiusAssign);
                }
                else
                    executionTable = std::make_unique<pdal::PointTable>();
                terminal->prepare(*executionTable);
                // Manager phase timing is a diagnostic (D0215): it needs one
                // selected neighborhood region whose single upload/spill pair
                // brackets the resident work, and it is valid for both direct
                // and ordinary (upstream writer) terminal routes.
                const bool collectManagerPhases =
                    commandPhases && rewrite.selectedRegions.size() == 1U &&
                    planDeclaresNeighborhoodRegion(plan);
                std::optional<ResidentCommandClock::time_point> executeStarted;
                if (collectManagerPhases)
                {
                    executeStarted = ResidentCommandClock::now();
                    residentScope->context().beginManagerPhaseTiming(
                        *managerStarted, *executeStarted);
                }
                std::optional<pdal::pdg_detail::CudaNeighborhoodLasSourceScope>
                    directSourceScope;
                if (directResidentLasSource)
                    directSourceScope.emplace(
                        directSourceFile->bytes(),
                        automaticResidentNnDistanceEnvelope(*pipelineText));
                const pdal::PointViewSet outputViews =
                    terminal->execute(*executionTable);
                if (directSourceScope)
                {
                    directLasRecordSummary =
                        directSourceScope->recordSummaryUsed();
                    directLasHostXyzMirror =
                        directSourceScope->hostXyzMirrored();
                    nnDistanceDeviceOnlyHandoff =
                        directSourceScope->nnDistanceDeviceOnlyHandoffUsed();
                    nnDistanceHostRestore =
                        directSourceScope->nnDistanceHostRestoreUsed();
                    nnDistanceAssignmentDeviceColumnReuse =
                        directSourceScope
                            ->nnDistanceAssignmentDeviceColumnReused();
                    knnGatherReuse = directSourceScope->knnGatherReuseUsed();
                    residentAssignmentExecuted =
                        directSourceScope->residentAssignmentExecuted();
                    hagDelaunayCudaUsed =
                        directSourceScope->hagDelaunayCudaUsed();
                }
                if (directResidentLasSource &&
                    (!directSourceScope || !directSourceScope->used()))
                    throw pdal::pdal_error(
                        "required direct LAS resident source path was not "
                        "used");
                if (std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") &&
                    !directLasRecordSummary)
                    throw pdal::pdal_error(
                        "required direct LAS record summary was not used");
                if (std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ") &&
                    (!directResidentLasSource || !directLasRecordSummary ||
                     directLasHostXyzMirror))
                    throw pdal::pdal_error(
                        "required direct LAS no-host-XYZ path was not used");
                if (std::getenv("PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF") &&
                    (!nnDistanceDeviceOnlyHandoff ||
                     !nnDistanceAssignmentDeviceColumnReuse))
                    throw pdal::pdal_error(
                        "required NNDistance device-only assignment handoff "
                        "was not used");
                if (std::getenv("PDG_REQUIRE_NND_HOST_RESTORE") &&
                    (!nnDistanceHostRestore ||
                     !nnDistanceAssignmentDeviceColumnReuse))
                    throw pdal::pdal_error(
                        "required NNDistance host restoration path was not "
                        "used");
                if (managerStarted)
                {
                    const auto managerEnded = ResidentCommandClock::now();
                    commandPhases->rewrittenManagerExecution =
                        elapsedSeconds(*managerStarted, managerEnded);
                    if (executeStarted)
                    {
                        pdal::pdg_detail::ResidentManagerPhaseSeconds phases =
                            residentScope->context().finishManagerPhaseTiming(
                                managerEnded);
                        if (phases.complete)
                            managerPhases = phases;
                    }
                }
                residentSchedule = residentScope->context().schedule();
                residentOutputPointCount =
                    residentScope->context().observedOutputPointCount();
                residentPhases = residentScope->context().phaseSeconds();
                if (automaticOutlierNnDistance)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || !directLasRecordSummary ||
                        directLasHostXyzMirror || !knnGatherReuse ||
                        observedIndexBuilds != 1U ||
                        std::getenv("PDG_TEST_AUTOMATIC_OUTLIER_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic outlier/NNDistance resident execution "
                            "proof failed");
                }
                if (automaticRadiusOutlierRadialDensity)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasOutput || !directResidentLasSource ||
                        !directSourceScope || !directSourceScope->used() ||
                        !directLasRecordSummary || directLasHostXyzMirror ||
                        observedIndexBuilds != 1U ||
                        !residentAssignmentExecuted ||
                        std::getenv("PDG_TEST_AUTOMATIC_RADIUS_COMPOSITION_"
                                    "PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic radius-outlier/radial-density resident "
                            "execution proof failed");
                }
                if (automaticRadiusAssign)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || !directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 1U ||
                        std::getenv(
                            "PDG_TEST_AUTOMATIC_RADIUSASSIGN_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic radiusassign resident execution "
                            "proof failed");
                }
                if (selectedAutomaticNeighborClassifier)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasOutput || !directResidentLasSource ||
                        !directSourceScope || !directSourceScope->used() ||
                        !directLasRecordSummary || directLasHostXyzMirror ||
                        observedIndexBuilds != 1U ||
                        std::getenv("PDG_TEST_AUTOMATIC_NEIGHBORCLASSIFIER_"
                                    "PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic neighborclassifier resident execution "
                            "proof failed");
                }
                if (selectedAutomaticHagNnCountOne)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    const bool selectedOnlyHagNn =
                        rewrite.selectedRegions.size() == 1U &&
                        rewrite.selectedStageIds.size() == 1U &&
                        rewrite.selectedStageIds.front() == 1U;
                    if (!rewrite.executable || !selectedOnlyHagNn ||
                        !directResidentLasOutput || !directExtraDoubleOutput ||
                        !experimentalDirectExtraDoubleHagNn ||
                        !directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || !directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 1U ||
                        !residentOutputPointCount || !las ||
                        *residentOutputPointCount != las->pointCount ||
                        !residentSchedule ||
                        residentSchedule->itemCount != las->pointCount ||
                        residentSchedule->configuredLaneCount != 1U ||
                        residentSchedule->activeLaneCount != 1U ||
                        std::getenv("PDG_TEST_AUTOMATIC_HAG_NN_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic HAG-NN count-one resident execution "
                            "proof failed");
                }
                if (selectedAutomaticHagDelaunayCountThree)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    const bool selectedOnlyHagDelaunay =
                        rewrite.selectedRegions.size() == 1U &&
                        rewrite.selectedStageIds.size() == 1U &&
                        rewrite.selectedStageIds.front() == 1U;
                    if (!rewrite.executable || !selectedOnlyHagDelaunay ||
                        !directResidentLasOutput || !directExtraDoubleOutput ||
                        !experimentalDirectExtraDoubleHagDelaunay ||
                        !directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || !directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 1U ||
                        !hagDelaunayCudaUsed || !residentOutputPointCount ||
                        !las || *residentOutputPointCount != las->pointCount ||
                        !residentSchedule ||
                        residentSchedule->itemCount != las->pointCount ||
                        residentSchedule->configuredLaneCount != 1U ||
                        residentSchedule->activeLaneCount != 1U ||
                        std::getenv(
                            "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic HAG-Delaunay count-three resident "
                            "execution proof failed");
                }
                if (automaticApproximateCoplanar)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    const bool requiredRepairObserved =
                        !requireAutomaticApproximateCoplanar ||
                        (residentPhases &&
                         residentPhases->approximateCoplanarRepairTriggers ==
                             1U &&
                         residentPhases->approximateCoplanarRepairRows != 0U &&
                         residentPhases->approximateCoplanarKd3Uses == 1U &&
                         residentPhases
                                 ->approximateCoplanarDeviceToHostRepairBytes !=
                             0U &&
                         residentPhases
                                 ->approximateCoplanarDeviceToHostRepairBytes ==
                             residentPhases
                                 ->approximateCoplanarHostToDeviceRepairBytes);
                    if (!directResidentLasOutput || directResidentLasSource ||
                        directSourceScope || directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 1U ||
                        !requiredRepairObserved ||
                        std::getenv(
                            "PDG_TEST_AUTOMATIC_COPLANAR_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "automatic approximate-coplanar resident "
                            "execution proof failed");
                }
                if (experimentalDirectSkewness)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasOutput ||
                        !directPermutedClassificationOutput ||
                        !directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 0U ||
                        !residentOutputPointCount || !las ||
                        *residentOutputPointCount != las->pointCount ||
                        std::getenv("PDG_TEST_DIRECT_SKEWNESS_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "direct skewness resident execution proof failed");
                }
                if (experimentalDirectSort)
                {
                    const ExecutionStatsSnapshot proof = observation.snapshot();
                    const std::size_t observedIndexBuilds =
                        proof
                            .totals[static_cast<std::size_t>(
                                ExecutionEventKind::IndexBuild)]
                            .count;
                    if (!directResidentLasOutput || !directPermutedSortOutput ||
                        directPermutedClassificationOutput ||
                        !directResidentLasSource || !directSourceScope ||
                        !directSourceScope->used() || directLasRecordSummary ||
                        directLasHostXyzMirror || observedIndexBuilds != 0U ||
                        !residentOutputPointCount || !las ||
                        *residentOutputPointCount != las->pointCount ||
                        std::getenv("PDG_TEST_DIRECT_SORT_PROOF_FAILURE"))
                        throw pdal::pdal_error(
                            "direct sort resident execution proof failed");
                }
                if (directResidentLasOutput)
                {
                    if (outputViews.size() != 1U || !*outputViews.begin())
                        throw pdal::pdal_error(
                            "direct resident LAS output requires one view");
                    const std::optional<ResidentCommandClock::time_point>
                        publicationStarted =
                            commandPhases
                                ? std::optional<
                                      ResidentCommandClock::time_point>(
                                      ResidentCommandClock::now())
                                : std::nullopt;
                    // The automatic global-order exactness proofs are
                    // data-dependent.
                    // Its rewritten graph has no writer and the direct
                    // publisher is atomic, so ties, non-finite keys, or a
                    // rejected destination can still return to the unchanged
                    // host pipeline. Commit only after execution, route
                    // proofs, and successful publication.
                    publishDirectResidentLasOutput(
                        plan, **outputViews.begin(),
                        directRadiusAssign || directNeighborClassifierOutput ||
                            directStandaloneOutlierOutput,
                        directPermutedClassificationOutput,
                        directPermutedSortOutput);
                    if (selectedAutomaticHagNnCountOne ||
                        selectedAutomaticHagDelaunayCountThree ||
                        selectedAutomaticSkewness || selectedAutomaticSort)
                        committed = true;
                    if (publicationStarted)
                        commandPhases->canonicalLasPublication =
                            elapsedSeconds(*publicationStarted);
                }

                // PDAL's writer can leave an unfinalized artifact yet return
                // success after an asynchronous/storage failure. Automatic
                // resident execution has already committed output side effects
                // at this point, so validate the public file and fail nonzero;
                // never report the accelerated route as successful merely
                // because the unchanged writer returned zero.
                if (automaticAdmission)
                {
                    if (!residentOutputPointCount)
                        throw pdal::pdal_error(
                            "automatic resident output point count is unavailable");
                    validateAutomaticLasPublication(
                        plan, *residentOutputPointCount);
                }
            }
            else
            {
                const pdal::PipelineManager::ExecResult execution =
                    manager.execute(pdal::ExecMode::PreferStream);
                if (execution.m_mode == pdal::ExecMode::None)
                    throw pdal::pdal_error("Couldn't run resident pipeline.");
                if (managerStarted)
                    commandPhases->rewrittenManagerExecution =
                        elapsedSeconds(*managerStarted);
            }
            actual = observation.snapshot();
            if (commandPhases)
            {
                commandPhases->commandBeforeStats =
                    elapsedSeconds(*commandStarted);
                const double attributed =
                    commandPhases->validationPlacementPreflight +
                    commandPhases->rewrittenManagerExecution +
                    commandPhases->canonicalLasPublication;
                commandPhases->otherControl = std::max(
                    0.0, commandPhases->commandBeforeStats - attributed);
            }
        }

        admissionTrace.accepted();
        if (!statsDestination.empty() &&
            !writeStats(
                statsDestination,
                statsJson(plan, runtime, profile, rewrite, actual,
                          direct ? &*direct : nullptr,
                          residentSchedule ? &*residentSchedule : nullptr,
                          residentOutputPointCount,
                          residentPhases ? &*residentPhases : nullptr,
                          commandPhases ? &*commandPhases : nullptr,
                          managerPhases ? &*managerPhases : nullptr,
                          directResidentLasOutput, directResidentLasSource,
                          directExtraDoubleOutput,
                          directPermutedClassificationOutput,
                          directPermutedSortOutput, directLasRecordSummary,
                          directLasHostXyzMirror, nnDistanceDeviceOnlyHandoff,
                          nnDistanceHostRestore,
                          nnDistanceAssignmentDeviceColumnReuse, knnGatherReuse,
                          validationBudgetOverride)))
            throw pdal::pdal_error("Can't write resident stats output.");
        return 0;
    }
    catch (const std::exception& exception)
    {
        if (automaticAdmission && !committed)
            return std::nullopt;
        pdal::Utils::printError(exception.what());
        return 1;
    }
}

int runResidentPipeline(int argc, char** argv)
{
    return runResidentPipelineImpl(argc, argv, false).value_or(1);
}

std::optional<int> tryAutomaticResidentLasPipeline(int argc, char** argv)
{
    if (argc != 3 || std::string_view(argv[1]) != "pipeline")
        return std::nullopt;
    // B0239's exact r2 selector belongs to the complete measured hybrid
    // workflow. The generic resident probe otherwise prepares and executes
    // the unchanged host pipeline before the hybrid selector is reached,
    // hiding both the route proof and the measured win.
    if (std::getenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID"))
    {
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupdal: automatic resident probe declined for r2 "
                         "hybrid\n";
        return std::nullopt;
    }
    // D0277: the calibrate command's forced-device override prices every
    // stage as a device win; it exists only so the explicit `resident`
    // command can time the device executor.  It must never widen the
    // automatic performance promise, so the automatic route declines here.
    if (std::getenv(CalibrationForceDeviceEnvironment.data()))
    {
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupdal: automatic resident probe declined under the "
                         "calibration placement override\n";
        return std::nullopt;
    }
    char residentCommand[] = "resident";
    std::array<char*, 4> residentArguments{argv[0], residentCommand, argv[2],
                                           nullptr};
    return runResidentPipelineImpl(
        static_cast<int>(residentArguments.size() - 1U),
        residentArguments.data(), true);
}

} // namespace pdg::cli
