#include <pdg/Hybrid.hpp>

#include <cstdlib>
#include <pdg/Memory.hpp>
#include <pdg/stages/Csf.hpp>
#include <pdg/stages/Elm.hpp>
#include <pdg/stages/Histogram.hpp>
#include <pdg/stages/Pmf.hpp>
#include <pdg/stages/Skewness.hpp>
#include <pdg/stages/Smrf.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pdg
{

namespace
{
using Json = nlohmann::json;

std::string lowercase(std::string value);

bool hasOnlyKeys(const Json& stage,
                 std::initializer_list<std::string_view> supported)
{
    for (const auto& [name, value] : stage.items())
    {
        (void)value;
        if (std::find(supported.begin(), supported.end(), name) ==
            supported.end())
            return false;
    }
    return true;
}

bool stringOrStringArray(const Json& value)
{
    if (value.is_string())
        return true;
    if (!value.is_array())
        return false;
    return std::all_of(value.begin(), value.end(),
                       [](const Json& item) { return item.is_string(); });
}

bool isUnsignedInteger(const Json& value)
{
    return value.is_number_unsigned() ||
           (value.is_number_integer() && value.get<std::int64_t>() >= 0);
}

bool isSigned32Integer(const Json& value)
{
    if (!value.is_number_integer())
        return false;
    const std::int64_t number = value.get<std::int64_t>();
    return number >= std::numeric_limits<std::int32_t>::min() &&
           number <= std::numeric_limits<std::int32_t>::max();
}

bool isExactDecimationStep(const Json& value)
{
    if (!value.is_number())
        return false;
    const double step = value.get<double>();
    return std::isfinite(step) && step >= 1.0;
}

bool eligiblePointStage(const Json& stage)
{
    if (!stage.is_object())
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string())
        return false;

    const std::string stageType = type->get<std::string>();
    if (stageType == "filters.assign")
    {
        if (!hasOnlyKeys(stage, {"type", "value"}))
            return false;
        const auto value = stage.find("value");
        return value == stage.end() || stringOrStringArray(*value);
    }
    if (stageType == "filters.ferry")
    {
        if (!hasOnlyKeys(stage, {"type", "dimensions"}))
            return false;
        const auto dimensions = stage.find("dimensions");
        return dimensions == stage.end() || stringOrStringArray(*dimensions);
    }
    if (stageType == "filters.expression")
    {
        if (!hasOnlyKeys(stage, {"type", "expression"}))
            return false;
        const auto expression = stage.find("expression");
        return expression != stage.end() && expression->is_string();
    }
    if (stageType == "filters.range")
    {
        if (!hasOnlyKeys(stage, {"type", "limits"}))
            return false;
        const auto limits = stage.find("limits");
        return limits != stage.end() && stringOrStringArray(*limits);
    }
    if (stageType == "filters.crop")
    {
        if (!hasOnlyKeys(stage, {"type", "bounds", "outside"}))
            return false;
        const auto bounds = stage.find("bounds");
        const auto outside = stage.find("outside");
        return bounds != stage.end() && bounds->is_string() &&
               (outside == stage.end() || outside->is_boolean());
    }
    if (stageType == "filters.decimation")
    {
        if (!hasOnlyKeys(stage, {"type", "step", "offset", "limit"}))
            return false;
        const auto step = stage.find("step");
        const auto offset = stage.find("offset");
        const auto limit = stage.find("limit");
        return (step == stage.end() || isExactDecimationStep(*step)) &&
               (offset == stage.end() || isUnsignedInteger(*offset)) &&
               (limit == stage.end() || isUnsignedInteger(*limit));
    }
    if (stageType == "filters.head" || stageType == "filters.tail")
    {
        if (!hasOnlyKeys(stage, {"type", "count", "invert"}))
            return false;
        const auto count = stage.find("count");
        const auto invert = stage.find("invert");
        return (count == stage.end() || isUnsignedInteger(*count)) &&
               (invert == stage.end() || invert->is_boolean());
    }
    if (stageType == "filters.transformation")
    {
        if (!hasOnlyKeys(stage, {"type", "matrix", "invert"}))
            return false;
        const auto matrix = stage.find("matrix");
        const auto invert = stage.find("invert");
        return matrix != stage.end() && matrix->is_string() &&
               (invert == stage.end() ||
                (invert->is_boolean() && !invert->get<bool>()));
    }
    return false;
}

bool eligibleResidentPointStage(const Json& stage)
{
    if (!eligiblePointStage(stage))
        return false;
    const std::string type = stage.at("type").get<std::string>();
    return type == "filters.assign" || type == "filters.ferry";
}

std::string_view trimWhitespace(std::string_view text)
{
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())))
        text.remove_prefix(1U);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
        text.remove_suffix(1U);
    return text;
}

bool coordinateDestination(std::string_view name)
{
    name = trimWhitespace(name);
    return name.size() == 1U &&
           (name.front() == 'X' || name.front() == 'x' || name.front() == 'Y' ||
            name.front() == 'y' || name.front() == 'Z' || name.front() == 'z');
}

bool validDestinationName(std::string_view name)
{
    name = trimWhitespace(name);
    if (name.empty())
        return false;
    const auto alpha = [](char character)
    {
        return (character >= 'A' && character <= 'Z') ||
               (character >= 'a' && character <= 'z');
    };
    const auto alphaNumeric = [&](char character)
    {
        return alpha(character) || (character >= '0' && character <= '9') ||
               character == '_';
    };
    return alpha(name.front()) &&
           std::all_of(name.begin() + 1, name.end(), alphaNumeric);
}

// Region planning has no PointLayout, so it cannot compile an assign/ferry
// program to discover its writes. Parse only its destination syntax and make
// anything ambiguous a boundary. Read-only XYZ references remain eligible;
// the runtime repeats the definitive compiled write-set check before launch.
bool assignPreservesCoordinates(std::string_view specification)
{
    const std::size_t separator = specification.find('=');
    if (separator == std::string_view::npos)
        return false;
    const std::string_view destination =
        trimWhitespace(specification.substr(0, separator));
    if (!validDestinationName(destination) ||
        (separator + 1U < specification.size() &&
         specification[separator + 1U] == '='))
        return false;
    return !coordinateDestination(destination);
}

bool ferryPreservesCoordinates(std::string_view specification)
{
    const std::size_t separator = specification.find('=');
    if (separator == std::string_view::npos)
        return false;
    const std::string_view source =
        trimWhitespace(specification.substr(0, separator));
    std::string_view destination =
        trimWhitespace(specification.substr(separator + 1U));
    if (!destination.empty() && destination.front() == '>')
        destination = trimWhitespace(destination.substr(1U));
    if ((!source.empty() && !validDestinationName(source)) ||
        !validDestinationName(destination))
        return false;
    return !coordinateDestination(destination);
}

bool allFerryDestinationsPreserveCoordinates(std::string_view specifications)
{
    std::size_t begin = 0;
    while (begin <= specifications.size())
    {
        const std::size_t comma = specifications.find(',', begin);
        const std::size_t end =
            comma == std::string_view::npos ? specifications.size() : comma;
        if (!ferryPreservesCoordinates(
                trimWhitespace(specifications.substr(begin, end - begin))))
            return false;
        if (comma == std::string_view::npos)
            return true;
        begin = comma + 1U;
    }
    return false;
}

bool residentPointStagePreservesCoordinates(const Json& stage)
{
    if (!eligibleResidentPointStage(stage))
        return false;
    const bool assign = stage.at("type").get<std::string>() == "filters.assign";
    const auto option = assign ? stage.find("value") : stage.find("dimensions");
    if (option == stage.end())
        return true;
    if (option->is_string())
        return assign ? assignPreservesCoordinates(option->get<std::string>())
                      : allFerryDestinationsPreserveCoordinates(
                            option->get<std::string>());
    for (const Json& value : *option)
        if (!(assign ? assignPreservesCoordinates(value.get<std::string>())
                     : allFerryDestinationsPreserveCoordinates(
                           value.get<std::string>())))
            return false;
    return true;
}

bool eligibleLocateStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "dimension", "minmax"}))
        return false;
    const auto type = stage.find("type");
    const auto dimension = stage.find("dimension");
    const auto minmax = stage.find("minmax");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.locate" &&
           dimension != stage.end() && dimension->is_string() &&
           (minmax == stage.end() || minmax->is_string());
}

bool eligibleRobustStage(const Json& stage)
{
    if (!stage.is_object())
        return false;
    const auto type = stage.find("type");
    const auto dimension = stage.find("dimension");
    if (type == stage.end() || !type->is_string() || dimension == stage.end() ||
        !dimension->is_string())
        return false;
    const std::string name = type->get<std::string>();
    if (name == "filters.iqr")
    {
        return hasOnlyKeys(stage, {"type", "dimension", "k"}) &&
               (!stage.contains("k") || stage.at("k").is_number());
    }
    if (name == "filters.mad")
    {
        return hasOnlyKeys(stage,
                           {"type", "dimension", "k", "mad_multiplier"}) &&
               (!stage.contains("k") || stage.at("k").is_number()) &&
               (!stage.contains("mad_multiplier") ||
                stage.at("mad_multiplier").is_number());
    }
    return false;
}

bool eligibleOrderStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage,
                     {"type", "dimension", "dimensions", "order", "algorithm"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.sort")
        return false;
    const auto dimension = stage.find("dimension");
    const auto dimensions = stage.find("dimensions");
    if ((dimension == stage.end()) == (dimensions == stage.end()))
        return false;
    if ((dimension != stage.end() && !stringOrStringArray(*dimension)) ||
        (dimensions != stage.end() && !stringOrStringArray(*dimensions)))
        return false;
    return (!stage.contains("order") || stage.at("order").is_string()) &&
           (!stage.contains("algorithm") || stage.at("algorithm").is_string());
}

bool eligibleRandomizeBridge(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "seed"}))
        return false;
    const auto type = stage.find("type");
    const auto seed = stage.find("seed");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.randomize" &&
           (seed == stage.end() || isUnsignedInteger(*seed));
}

bool eligibleMortonStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "reverse"}))
        return false;
    const auto type = stage.find("type");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.mortonorder" &&
           (!stage.contains("reverse") || stage.at("reverse").is_boolean());
}

bool eligibleGroupByStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "dimension"}))
        return false;
    const auto type = stage.find("type");
    const auto dimension = stage.find("dimension");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.groupby" &&
           dimension != stage.end() && dimension->is_string();
}

bool eligibleReturnsStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "groups"}))
        return false;
    const auto type = stage.find("type");
    const auto groups = stage.find("groups");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.returns" &&
           (groups == stage.end() || stringOrStringArray(*groups));
}

bool eligibleMergeStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type"}))
        return false;
    const auto type = stage.find("type");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.merge";
}

bool eligibleDividerStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "mode", "count"}))
        return false;
    const auto type = stage.find("type");
    const auto count = stage.find("count");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.divider" || count == stage.end() ||
        !isUnsignedInteger(*count))
        return false;
    const std::uint64_t value = count->get<std::uint64_t>();
    if (value < 2U || value > 1000U)
        return false;
    const auto mode = stage.find("mode");
    if (mode == stage.end())
        return true;
    if (!mode->is_string())
        return false;
    const std::string name = lowercase(mode->get<std::string>());
    return name == "partition" || name == "round_robin";
}

bool eligibleSplitterStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "length", "origin_x",
                                                   "origin_y", "buffer"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.splitter")
        return false;
    const auto finiteOption = [&](std::string_view name, double fallback)
    {
        const auto position = stage.find(std::string(name));
        if (position == stage.end())
            return std::pair<bool, double>{true, fallback};
        if (!position->is_number())
            return std::pair<bool, double>{false, fallback};
        const double value = position->get<double>();
        return std::pair<bool, double>{std::isfinite(value), value};
    };
    const auto [validLength, length] = finiteOption("length", 1000.0);
    const auto [validBuffer, buffer] = finiteOption("buffer", 0.0);
    const auto [validX, x] = finiteOption("origin_x", 0.0);
    const auto [validY, y] = finiteOption("origin_y", 0.0);
    (void)x;
    (void)y;
    return validLength && validBuffer && validX && validY && length > 0.0 &&
           buffer < length / 2.0;
}

bool eligibleColorinterpStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "dimension", "minimum", "maximum", "clamp",
                             "ramp", "invert", "mad", "mad_multiplier", "k"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.colorinterp")
        return false;
    const auto numeric = [](const Json& value) { return value.is_number(); };
    if ((stage.contains("dimension") && !stage.at("dimension").is_string()) ||
        (stage.contains("minimum") && !numeric(stage.at("minimum"))) ||
        (stage.contains("maximum") && !numeric(stage.at("maximum"))) ||
        (stage.contains("clamp") && !stage.at("clamp").is_boolean()) ||
        (stage.contains("ramp") && !stage.at("ramp").is_string()) ||
        (stage.contains("invert") && !stage.at("invert").is_boolean()) ||
        (stage.contains("mad") && !stage.at("mad").is_boolean()) ||
        (stage.contains("mad_multiplier") &&
         !numeric(stage.at("mad_multiplier"))) ||
        (stage.contains("k") && !numeric(stage.at("k"))))
        return false;
    return !stage.contains("minimum") || !stage.contains("maximum") ||
           stage.at("maximum").get<double>() >
               stage.at("minimum").get<double>();
}

bool eligibleStatsStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "dimensions", "enumerate", "global",
                             "count", "advanced", "commonsrs"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.stats")
        return false;
    for (const std::string_view option :
         {"dimensions", "enumerate", "global", "count"})
    {
        const std::string name(option);
        if (stage.contains(name) && !stringOrStringArray(stage.at(name)))
            return false;
    }
    return (!stage.contains("advanced") || stage.at("advanced").is_boolean()) &&
           (!stage.contains("commonsrs") || stage.at("commonsrs").is_string());
}

bool eligibleInfoStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "point", "query"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.info")
        return false;
    return (!stage.contains("point") || stage.at("point").is_string()) &&
           (!stage.contains("query") || stage.at("query").is_string());
}

bool eligibleExpressionStatsStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "dimension", "expressions"}))
        return false;
    const auto type = stage.find("type");
    const auto dimension = stage.find("dimension");
    const auto expressions = stage.find("expressions");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.expressionstats" &&
           dimension != stage.end() && dimension->is_string() &&
           expressions != stage.end() && stringOrStringArray(*expressions);
}

std::size_t expressionStatsWork(const Json& stage)
{
    const Json& expressions = stage.at("expressions");
    return expressions.is_string() ? 1U : expressions.size();
}

bool eligibleAutomaticPointCountReader(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "filename", "count", "override_srs",
                             "spatialreference", "default_srs", "extra_dims",
                             "compression", "use_eb_vlr", "ignore_vlr",
                             "ignore_missing_vlrs", "fix_dims", "nosrs",
                             "threads", "srs_vlr_order"}))
        return false;
    const auto type = stage.find("type");
    const auto filename = stage.find("filename");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "readers.las" &&
           filename != stage.end() && filename->is_string() &&
           (!stage.contains("count") || isUnsignedInteger(stage.at("count")));
}

bool pointStagePreservesCardinality(const Json& stage)
{
    if (!eligiblePointStage(stage))
        return false;
    const std::string type = stage.at("type").get<std::string>();
    return type == "filters.assign" || type == "filters.ferry" ||
           type == "filters.transformation";
}

bool eligibleOutlierStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "method", "min_k", "radius", "mean_k",
                             "multiplier", "class"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.outlier")
        return false;
    return (!stage.contains("method") || stage.at("method").is_string()) &&
           (!stage.contains("min_k") || isSigned32Integer(stage.at("min_k"))) &&
           (!stage.contains("radius") || stage.at("radius").is_number()) &&
           (!stage.contains("mean_k") ||
            isSigned32Integer(stage.at("mean_k"))) &&
           (!stage.contains("multiplier") ||
            stage.at("multiplier").is_number()) &&
           (!stage.contains("class") ||
            (isUnsignedInteger(stage.at("class")) &&
             stage.at("class").get<std::uint64_t>() <= 255U));
}

bool exactString(const Json& object, std::string_view name,
                 std::string_view value)
{
    const auto option = object.find(std::string(name));
    return option != object.end() && option->is_string() &&
           option->get_ref<const std::string&>() == value;
}

bool lazFilename(const Json& stage)
{
    const auto filename = stage.find("filename");
    if (filename == stage.end() || !filename->is_string())
        return false;
    const std::string& value = filename->get_ref<const std::string&>();
    return value.size() >= 4U &&
           value.compare(value.size() - 4U, 4U, ".laz") == 0 &&
           !(value.size() >= 9U &&
             value.compare(value.size() - 9U, 9U, ".copc.laz") == 0);
}

bool lasFilename(const Json& stage)
{
    const auto filename = stage.find("filename");
    if (filename == stage.end() || !filename->is_string())
        return false;
    const std::string& value = filename->get_ref<const std::string&>();
    return value.size() >= 4U &&
           value.compare(value.size() - 4U, 4U, ".las") == 0;
}

bool measuredR4PipelineGrammar(const Json& root, const Json& pipeline)
{
    if (!root.is_object() || root.size() != 1U || pipeline.size() != 5U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& outlier = pipeline.at(1U);
    const Json& range = pipeline.at(2U);
    const Json& sample = pipeline.at(3U);
    const Json& writer = pipeline.at(4U);
    return reader.is_object() && hasOnlyKeys(reader, {"type", "filename"}) &&
           exactString(reader, "type", "readers.las") && lazFilename(reader) &&
           outlier.is_object() &&
           hasOnlyKeys(outlier, {"type", "method", "mean_k", "multiplier"}) &&
           exactString(outlier, "type", "filters.outlier") &&
           exactString(outlier, "method", "statistical") &&
           outlier.contains("mean_k") && outlier.contains("multiplier") &&
           isSigned32Integer(outlier.at("mean_k")) &&
           outlier.at("mean_k").get<std::int64_t>() == 8 &&
           outlier.at("multiplier").is_number() &&
           outlier.at("multiplier").get<double>() == 2.0 && range.is_object() &&
           hasOnlyKeys(range, {"type", "limits"}) &&
           exactString(range, "type", "filters.range") &&
           exactString(range, "limits", "Classification![7:7]") &&
           sample.is_object() && hasOnlyKeys(sample, {"type", "radius"}) &&
           exactString(sample, "type", "filters.sample") &&
           sample.contains("radius") && sample.at("radius").is_number() &&
           sample.at("radius").get<double>() == 1.0 && writer.is_object() &&
           hasOnlyKeys(writer, {"type", "filename", "compression"}) &&
           exactString(writer, "type", "writers.las") && lazFilename(writer) &&
           exactString(writer, "compression", "true");
}

bool measuredLabelNnDistancePipelineGrammar(const Json& root,
                                            const Json& pipeline)
{
    if (!root.is_object() || root.size() != 1U || pipeline.size() != 5U)
        return false;
    const Json& reader = pipeline.at(0U);
    const Json& label = pipeline.at(1U);
    const Json& nndistance = pipeline.at(2U);
    const Json& assign = pipeline.at(3U);
    const Json& writer = pipeline.at(4U);
    return reader.is_object() && hasOnlyKeys(reader, {"type", "filename"}) &&
           exactString(reader, "type", "readers.las") && lasFilename(reader) &&
           label.is_object() && hasOnlyKeys(label, {"type", "dimensions"}) &&
           exactString(label, "type", "filters.label_duplicates") &&
           exactString(label, "dimensions", "Classification") &&
           nndistance.is_object() && hasOnlyKeys(nndistance, {"type", "k"}) &&
           exactString(nndistance, "type", "filters.nndistance") &&
           nndistance.contains("k") && isUnsignedInteger(nndistance.at("k")) &&
           nndistance.at("k").get<std::uint64_t>() == 10U &&
           assign.is_object() && hasOnlyKeys(assign, {"type", "value"}) &&
           exactString(assign, "type", "filters.assign") &&
           exactString(assign, "value", "UserData = Duplicate") &&
           writer.is_object() && hasOnlyKeys(writer, {"type", "filename"}) &&
           exactString(writer, "type", "writers.las") && lasFilename(writer);
}

bool eligibleRadialDensityStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "radius"}))
        return false;
    const auto type = stage.find("type");
    return type != stage.end() && type->is_string() &&
           type->get<std::string>() == "filters.radialdensity" &&
           (!stage.contains("radius") || stage.at("radius").is_number());
}

bool eligibleNormalStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "knn", "always_up", "refine"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.normal")
        return false;
    if (stage.contains("knn"))
    {
        if (!isSigned32Integer(stage.at("knn")))
            return false;
        const std::int64_t neighbors = stage.at("knn").get<std::int64_t>();
        if (neighbors < 2 || neighbors >= 64)
            return false;
    }
    return (!stage.contains("always_up") ||
            stage.at("always_up").is_boolean()) &&
           (!stage.contains("refine") || (stage.at("refine").is_boolean() &&
                                          !stage.at("refine").get<bool>()));
}

bool eligibleApproximateCoplanarStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "knn", "thresh1", "thresh2"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.approximatecoplanar")
        return false;
    const std::int64_t neighbors =
        stage.contains("knn") && isSigned32Integer(stage.at("knn"))
            ? stage.at("knn").get<std::int64_t>()
            : 8;
    return (!stage.contains("knn") || isSigned32Integer(stage.at("knn"))) &&
           neighbors >= 3 && neighbors <= 64 &&
           (!stage.contains("thresh1") || stage.at("thresh1").is_number()) &&
           (!stage.contains("thresh2") || stage.at("thresh2").is_number());
}

#if PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR
bool eligibleAutomaticApproximateCoplanarStage(const Json& stage)
{
    if (!eligibleApproximateCoplanarStage(stage))
        return false;
    return !stage.contains("knn") || stage.at("knn").get<std::int64_t>() == 8;
}

bool eligibleAutomaticApproximateCoplanarWriter(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "filename", "extra_dims"}))
        return false;
    const auto type = stage.find("type");
    const auto filename = stage.find("filename");
    const auto extraDimensions = stage.find("extra_dims");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "writers.las" || filename == stage.end() ||
        !filename->is_string() || extraDimensions == stage.end() ||
        !extraDimensions->is_string() ||
        extraDimensions->get<std::string>() != "all")
        return false;
    const std::string output = lowercase(filename->get<std::string>());
    return output.size() >= 4U &&
           output.compare(output.size() - 4U, 4U, ".las") == 0;
}
#endif

bool eligibleNnDistanceStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "mode", "k"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.nndistance")
        return false;
    const std::uint64_t neighbors =
        stage.contains("k") && isUnsignedInteger(stage.at("k"))
            ? stage.at("k").get<std::uint64_t>()
            : 10U;
    if ((!stage.contains("k") || isUnsignedInteger(stage.at("k"))) &&
        neighbors >= 1U && neighbors < 64U)
    {
        if (!stage.contains("mode"))
            return true;
        if (!stage.at("mode").is_string())
            return false;
        const std::string mode = lowercase(stage.at("mode").get<std::string>());
        return mode == "kth" || mode == "avg";
    }
    return false;
}

bool eligibleEigenvaluesStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "knn", "normalize", "stride", "min_k"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.eigenvalues")
        return false;
    const std::int64_t neighbors =
        stage.contains("knn") && isSigned32Integer(stage.at("knn"))
            ? stage.at("knn").get<std::int64_t>()
            : 8;
    return (!stage.contains("knn") || isSigned32Integer(stage.at("knn"))) &&
           neighbors >= 2 && neighbors < 64 &&
           (!stage.contains("normalize") ||
            stage.at("normalize").is_boolean()) &&
           (!stage.contains("stride") ||
            (isUnsignedInteger(stage.at("stride")) &&
             stage.at("stride").get<std::uint64_t>() == 1U)) &&
           (!stage.contains("min_k") || isSigned32Integer(stage.at("min_k")));
}

bool supportedCovarianceFeatureSet(const Json& value)
{
    if (!stringOrStringArray(value))
        return false;
    bool found = false;
    const auto inspect = [&](const Json& item)
    {
        const std::string text = item.get<std::string>();
        std::size_t begin = 0;
        while (begin <= text.size())
        {
            const std::size_t comma = text.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? text.size() : comma;
            std::size_t first = begin;
            while (first < end &&
                   std::isspace(static_cast<unsigned char>(text[first])))
                ++first;
            std::size_t last = end;
            while (last > first &&
                   std::isspace(static_cast<unsigned char>(text[last - 1U])))
                --last;
            const std::string feature =
                lowercase(text.substr(first, last - first));
            if (feature != "dimensionality" && feature != "linearity" &&
                feature != "planarity" && feature != "scattering" &&
                feature != "verticality" && feature != "omnivariance" &&
                feature != "anisotropy" && feature != "eigenentropy" &&
                feature != "eigenvaluesum" && feature != "surfacevariation" &&
                feature != "demantkeverticality")
                return false;
            found = true;
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
        return true;
    };
    if (value.is_array())
    {
        for (const Json& item : value)
            if (!inspect(item))
                return false;
    }
    else if (!inspect(value))
        return false;
    return found;
}

bool eligibleCovarianceFeaturesStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "knn", "threads", "feature_set", "stride",
                             "min_k", "mode", "optimized"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.covariancefeatures")
        return false;
    const std::int64_t neighbors =
        stage.contains("knn") && isSigned32Integer(stage.at("knn"))
            ? stage.at("knn").get<std::int64_t>()
            : 10;
    if ((!stage.contains("knn") || isSigned32Integer(stage.at("knn"))) &&
        neighbors >= 2 && neighbors < 64 &&
        (!stage.contains("threads") ||
         (isSigned32Integer(stage.at("threads")) &&
          stage.at("threads").get<std::int64_t>() == 1)) &&
        (!stage.contains("feature_set") ||
         supportedCovarianceFeatureSet(stage.at("feature_set"))) &&
        (!stage.contains("stride") ||
         (isUnsignedInteger(stage.at("stride")) &&
          stage.at("stride").get<std::uint64_t>() == 1U)) &&
        (!stage.contains("min_k") || isSigned32Integer(stage.at("min_k"))) &&
        (!stage.contains("optimized") || (stage.at("optimized").is_boolean() &&
                                          !stage.at("optimized").get<bool>())))
    {
        if (!stage.contains("mode"))
            return true;
        if (!stage.at("mode").is_string())
            return false;
        const std::string mode = lowercase(stage.at("mode").get<std::string>());
        return mode == "raw" || mode == "sqrt" || mode == "normalized";
    }
    return false;
}

bool eligibleLofStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "minpts"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.lof")
        return false;
    // The stage's self-inclusive increment must stay within the 64-neighbor
    // exact envelope.
    const std::int64_t minimumPoints =
        stage.contains("minpts") && isSigned32Integer(stage.at("minpts"))
            ? stage.at("minpts").get<std::int64_t>()
            : 10;
    return (!stage.contains("minpts") ||
            isSigned32Integer(stage.at("minpts"))) &&
           minimumPoints >= 1 && minimumPoints <= 63;
}

bool eligibleEstimateRankStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "knn", "thresh"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.estimaterank")
        return false;
    const std::int64_t neighbors =
        stage.contains("knn") && isSigned32Integer(stage.at("knn"))
            ? stage.at("knn").get<std::int64_t>()
            : 8;
    return (!stage.contains("knn") || isSigned32Integer(stage.at("knn"))) &&
           (!stage.contains("thresh") || stage.at("thresh").is_number()) &&
           neighbors >= 3 && neighbors <= 64;
}

bool eligibleOptimalNeighborhoodStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "min_k", "max_k"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.optimalneighborhood")
        return false;
    const auto unsignedOption = [&](const char* name, std::int64_t fallback)
    {
        return stage.contains(name) && stage.at(name).is_number_unsigned()
                   ? stage.at(name).get<std::int64_t>()
                   : fallback;
    };
    const std::int64_t minimumK = unsignedOption("min_k", 10);
    const std::int64_t maximumK = unsignedOption("max_k", 14);
    return (!stage.contains("min_k") ||
            stage.at("min_k").is_number_unsigned()) &&
           (!stage.contains("max_k") ||
            stage.at("max_k").is_number_unsigned()) &&
           minimumK >= 1 && minimumK <= maximumK && maximumK <= 64;
}

bool eligibleNeighborClassifierStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "k", "dimension"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.neighborclassifier")
        return false;
    if (stage.contains("dimension") &&
        (!stage.at("dimension").is_string() ||
         stage.at("dimension").get<std::string>() != "Classification"))
        return false;
    if (!stage.contains("k") || !isSigned32Integer(stage.at("k")))
        return false;
    const std::int64_t neighbors = stage.at("k").get<std::int64_t>();
    return neighbors >= 1 && neighbors <= 64;
}

bool eligibleRadiusAssignStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "src_domain", "reference_domain", "radius",
                             "update_expression", "is3d", "max2d_above",
                             "max2d_below"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.radiusassign")
        return false;
    if (!stage.contains("radius") || !stage.at("radius").is_number() ||
        stage.at("radius").get<double>() <= 0.0 ||
        !stage.contains("update_expression") ||
        !stringOrStringArray(stage.at("update_expression")))
        return false;
    for (const char* domain : {"src_domain", "reference_domain"})
    {
        if (stage.contains(domain) && !stringOrStringArray(stage.at(domain)))
            return false;
    }
    return (!stage.contains("is3d") || stage.at("is3d").is_boolean()) &&
           (!stage.contains("max2d_above") ||
            stage.at("max2d_above").is_number()) &&
           (!stage.contains("max2d_below") ||
            stage.at("max2d_below").is_number());
}

bool eligibleLabelDuplicatesStage(const Json& stage)
{
    if (!stage.is_object() || !hasOnlyKeys(stage, {"type", "dimensions"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.label_duplicates")
        return false;
    const auto dimensions = stage.find("dimensions");
    return dimensions == stage.end() || stringOrStringArray(*dimensions);
}

bool eligibleSmrfStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "cell", "slope", "scalar", "threshold",
                             "cut", "returns", "window", "ground_class",
                             "other_class", "only_ground"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.smrf")
        return false;

    const auto finiteOption = [&](const char* name, double fallback)
    {
        if (!stage.contains(name))
            return fallback;
        if (!stage.at(name).is_number())
            return std::numeric_limits<double>::quiet_NaN();
        return stage.at(name).get<double>();
    };
    const double cell = finiteOption("cell", 1.0);
    const double slope = finiteOption("slope", 0.15);
    const double scalar = finiteOption("scalar", 1.25);
    const double threshold = finiteOption("threshold", 0.5);
    const double cut = finiteOption("cut", 0.0);
    const double window = finiteOption("window", 18.0 * cell);
    if (!std::isfinite(cell) || cell <= 0.0 || !std::isfinite(slope) ||
        !std::isfinite(scalar) || !std::isfinite(threshold) ||
        !std::isfinite(cut) || cut < 0.0 || !std::isfinite(window) ||
        window < 0.0 ||
        std::ceil(window / cell) > SmrfExactDeviceMaximumMorphologyRadius ||
        (cut > 0.0 &&
         2.0 * std::ceil(cut / cell) > SmrfExactDeviceMaximumMorphologyRadius))
        return false;

    const auto validClass = [&](const char* name)
    {
        return !stage.contains(name) ||
               (isUnsignedInteger(stage.at(name)) &&
                stage.at(name).get<std::uint64_t>() <= 255U);
    };
    if (!validClass("ground_class") || !validClass("other_class") ||
        (stage.contains("only_ground") &&
         !stage.at("only_ground").is_boolean()))
        return false;
    const std::uint64_t ground =
        stage.contains("ground_class")
            ? stage.at("ground_class").get<std::uint64_t>()
            : 2U;
    const std::uint64_t other =
        stage.contains("other_class")
            ? stage.at("other_class").get<std::uint64_t>()
            : 1U;
    const bool onlyGround =
        stage.contains("only_ground") && stage.at("only_ground").get<bool>();
    if (ground == other && !onlyGround)
        return false;

    const auto returns = stage.find("returns");
    if (returns == stage.end())
        return true;
    const auto validReturn = [](const Json& value)
    {
        if (!value.is_string())
            return false;
        std::string text = value.get<std::string>();
        text.erase(text.begin(),
                   std::find_if(text.begin(), text.end(), [](unsigned char c)
                                { return !std::isspace(c); }));
        text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char c)
                                { return !std::isspace(c); })
                       .base(),
                   text.end());
        return text == "first" || text == "intermediate" || text == "last" ||
               text == "only";
    };
    if (returns->is_string())
        return validReturn(*returns);
    return returns->is_array() &&
           std::all_of(returns->begin(), returns->end(), validReturn);
}

bool eligiblePmfStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage,
                     {"type", "cell_size", "exponential", "initial_distance",
                      "returns", "max_distance", "max_window_size", "slope",
                      "ground_class", "other_class", "only_ground"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.pmf")
        return false;

    const auto finiteOption = [&](const char* name, double fallback)
    {
        if (!stage.contains(name))
            return fallback;
        if (!stage.at(name).is_number())
            return std::numeric_limits<double>::quiet_NaN();
        return stage.at(name).get<double>();
    };
    PmfProgram program;
    program.cellSize = finiteOption("cell_size", 1.0);
    if (stage.contains("exponential"))
    {
        if (!stage.at("exponential").is_boolean())
            return false;
        program.exponential = stage.at("exponential").get<bool>();
    }
    program.initialDistance = finiteOption("initial_distance", 0.15);
    program.maxDistance = finiteOption("max_distance", 2.5);
    program.maxWindowSize = finiteOption("max_window_size", 33.0);
    program.slope = finiteOption("slope", 1.0);
    const auto validClass = [&](const char* name)
    {
        return !stage.contains(name) ||
               (isUnsignedInteger(stage.at(name)) &&
                stage.at(name).get<std::uint64_t>() <= 255U);
    };
    if (!validClass("ground_class") || !validClass("other_class") ||
        (stage.contains("only_ground") &&
         !stage.at("only_ground").is_boolean()))
        return false;
    program.groundClass =
        stage.contains("ground_class")
            ? static_cast<std::uint8_t>(
                  stage.at("ground_class").get<std::uint64_t>())
            : std::uint8_t{2U};
    program.otherClass = stage.contains("other_class")
                             ? static_cast<std::uint8_t>(
                                   stage.at("other_class").get<std::uint64_t>())
                             : std::uint8_t{1U};
    program.onlyGround =
        stage.contains("only_ground") && stage.at("only_ground").get<bool>();
    if (!pmfProgramWithinExactDeviceEnvelope(program))
        return false;

    const auto returns = stage.find("returns");
    if (returns == stage.end())
        return true;
    const auto validReturn = [](const Json& value)
    {
        if (!value.is_string())
            return false;
        std::string text = value.get<std::string>();
        text.erase(text.begin(),
                   std::find_if(text.begin(), text.end(), [](unsigned char c)
                                { return !std::isspace(c); }));
        text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char c)
                                { return !std::isspace(c); })
                       .base(),
                   text.end());
        return text == "first" || text == "intermediate" || text == "last" ||
               text == "only";
    };
    if (returns->is_string())
        return validReturn(*returns);
    return returns->is_array() &&
           std::all_of(returns->begin(), returns->end(), validReturn);
}

bool eligibleCsfStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "smooth", "step", "threshold", "hdiff",
                             "resolution", "rigidness", "iterations", "returns",
                             "ground_class", "other_class", "only_ground"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.csf")
        return false;

    const auto finiteOption = [&](const char* name, double fallback)
    {
        if (!stage.contains(name))
            return fallback;
        if (!stage.at(name).is_number())
            return std::numeric_limits<double>::quiet_NaN();
        return stage.at(name).get<double>();
    };
    const auto integerOption = [&](const char* name,
                                   int fallback) -> std::optional<int>
    {
        if (!stage.contains(name))
            return fallback;
        if (!stage.at(name).is_number_integer())
            return std::nullopt;
        const std::int64_t value = stage.at(name).get<std::int64_t>();
        if (value < (std::numeric_limits<int>::min)() ||
            value > (std::numeric_limits<int>::max)())
            return std::nullopt;
        return static_cast<int>(value);
    };
    CsfProgram program;
    if (stage.contains("smooth"))
    {
        if (!stage.at("smooth").is_boolean())
            return false;
        program.smooth = stage.at("smooth").get<bool>();
    }
    program.timeStep = finiteOption("step", 0.65);
    program.classThreshold = finiteOption("threshold", 0.5);
    program.heightThreshold = finiteOption("hdiff", 0.3);
    program.resolution = finiteOption("resolution", 1.0);
    const std::optional<int> rigidness = integerOption("rigidness", 3);
    const std::optional<int> iterations = integerOption("iterations", 500);
    if (!rigidness || !iterations)
        return false;
    program.rigidness = *rigidness;
    program.iterations = *iterations;
    const auto validClass = [&](const char* name)
    {
        return !stage.contains(name) ||
               (isUnsignedInteger(stage.at(name)) &&
                stage.at(name).get<std::uint64_t>() <= 255U);
    };
    if (!validClass("ground_class") || !validClass("other_class") ||
        (stage.contains("only_ground") &&
         !stage.at("only_ground").is_boolean()))
        return false;
    program.groundClass =
        stage.contains("ground_class")
            ? static_cast<std::uint8_t>(
                  stage.at("ground_class").get<std::uint64_t>())
            : std::uint8_t{2U};
    program.otherClass = stage.contains("other_class")
                             ? static_cast<std::uint8_t>(
                                   stage.at("other_class").get<std::uint64_t>())
                             : std::uint8_t{1U};
    program.onlyGround =
        stage.contains("only_ground") && stage.at("only_ground").get<bool>();
    if (!csfProgramWithinExactDeviceEnvelope(program))
        return false;

    const auto returns = stage.find("returns");
    if (returns == stage.end())
        return true;
    const auto validReturn = [](const Json& value)
    {
        if (!value.is_string())
            return false;
        std::string text = value.get<std::string>();
        text.erase(text.begin(),
                   std::find_if(text.begin(), text.end(), [](unsigned char c)
                                { return !std::isspace(c); }));
        text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char c)
                                { return !std::isspace(c); })
                       .base(),
                   text.end());
        return text == "first" || text == "intermediate" || text == "last" ||
               text == "only";
    };
    if (returns->is_string())
        return validReturn(*returns);
    return returns->is_array() &&
           std::all_of(returns->begin(), returns->end(), validReturn);
}

bool eligibleElmStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "cell", "class", "threshold"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.elm")
        return false;

    ElmProgram program;
    if (stage.contains("cell"))
    {
        if (!stage.at("cell").is_number())
            return false;
        program.cell = stage.at("cell").get<double>();
    }
    if (stage.contains("threshold"))
    {
        if (!stage.at("threshold").is_number())
            return false;
        program.threshold = stage.at("threshold").get<double>();
    }
    if (stage.contains("class"))
    {
        if (!isUnsignedInteger(stage.at("class")) ||
            stage.at("class").get<std::uint64_t>() > 255U)
            return false;
        program.classification =
            static_cast<std::uint8_t>(stage.at("class").get<std::uint64_t>());
    }
    return elmProgramWithinExactDeviceEnvelope(program);
}

bool eligibleSkewnessStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage,
                     {"type", "ground_class", "other_class", "only_ground"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.skewnessbalancing")
        return false;
    const auto validClass = [&](const char* name)
    {
        return !stage.contains(name) ||
               (isUnsignedInteger(stage.at(name)) &&
                stage.at(name).get<std::uint64_t>() <= 255U);
    };
    if (!validClass("ground_class") || !validClass("other_class") ||
        (stage.contains("only_ground") &&
         !stage.at("only_ground").is_boolean()))
        return false;

    SkewnessProgram program;
    if (stage.contains("ground_class"))
        program.groundClass = static_cast<std::uint8_t>(
            stage.at("ground_class").get<std::uint64_t>());
    if (stage.contains("other_class"))
        program.otherClass = static_cast<std::uint8_t>(
            stage.at("other_class").get<std::uint64_t>());
    if (stage.contains("only_ground"))
        program.onlyGround = stage.at("only_ground").get<bool>();
    return skewnessProgramValid(program);
}

bool eligibleHagNnStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "count", "max_distance",
                             "allow_extrapolation", "class"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.hag_nn")
        return false;
    if (stage.contains("count") &&
        (!isUnsignedInteger(stage.at("count")) ||
         stage.at("count").get<std::uint64_t>() < 1U ||
         stage.at("count").get<std::uint64_t>() > 64U))
        return false;
    if (stage.contains("max_distance") &&
        (!stage.at("max_distance").is_number() ||
         !std::isfinite(stage.at("max_distance").get<double>())))
        return false;
    if (stage.contains("allow_extrapolation") &&
        !stage.at("allow_extrapolation").is_boolean())
        return false;
    if (stage.contains("class") &&
        (!isUnsignedInteger(stage.at("class")) ||
         stage.at("class").get<std::uint64_t>() > 255U))
        return false;
    return true;
}

bool eligibleHagDelaunayStage(const Json& stage)
{
    if (!stage.is_object() ||
        !hasOnlyKeys(stage, {"type", "count", "allow_extrapolation", "class"}))
        return false;
    const auto type = stage.find("type");
    if (type == stage.end() || !type->is_string() ||
        type->get<std::string>() != "filters.hag_delaunay")
        return false;
    if (!stage.contains("count") || !isUnsignedInteger(stage.at("count")) ||
        stage.at("count").get<std::uint64_t>() != 3U)
        return false;
    if (stage.contains("allow_extrapolation") &&
        !stage.at("allow_extrapolation").is_boolean())
        return false;
    if (stage.contains("class") &&
        (!isUnsignedInteger(stage.at("class")) ||
         stage.at("class").get<std::uint64_t>() > 255U))
        return false;
    return true;
}

bool eligibleNeighborhoodStage(const Json& stage)
{
    return eligibleNnDistanceStage(stage) || eligibleHagNnStage(stage) ||
           eligibleHagDelaunayStage(stage) || eligibleNormalStage(stage) ||
           eligibleApproximateCoplanarStage(stage) || eligibleLofStage(stage) ||
           eligibleEigenvaluesStage(stage) ||
           eligibleCovarianceFeaturesStage(stage) ||
           eligibleEstimateRankStage(stage) ||
           eligibleOptimalNeighborhoodStage(stage) ||
           eligibleNeighborClassifierStage(stage);
}

std::uint32_t neighborhoodRequest(const Json& stage)
{
    const std::string type = stage.at("type").get<std::string>();
    if (type == "filters.hag_nn")
        return static_cast<std::uint32_t>(
            stage.contains("count") ? stage.at("count").get<std::uint64_t>()
                                    : 1U);
    if (type == "filters.hag_delaunay")
        return 3U;
    if (type == "filters.nndistance")
    {
        const std::uint64_t neighbors =
            stage.contains("k") ? stage.at("k").get<std::uint64_t>() : 10U;
        return static_cast<std::uint32_t>(neighbors) + 1U;
    }
    if (type == "filters.approximatecoplanar")
    {
        const std::int64_t neighbors =
            stage.contains("knn") ? stage.at("knn").get<std::int64_t>() : 8;
        return static_cast<std::uint32_t>(neighbors);
    }
    if (type == "filters.lof")
    {
        const std::int64_t minimumPoints =
            stage.contains("minpts") ? stage.at("minpts").get<std::int64_t>()
                                     : 10;
        return static_cast<std::uint32_t>(minimumPoints) + 1U;
    }
    if (type == "filters.neighborclassifier")
        return static_cast<std::uint32_t>(stage.at("k").get<std::int64_t>());
    if (type == "filters.optimalneighborhood")
    {
        const std::int64_t maximumK =
            stage.contains("max_k") ? stage.at("max_k").get<std::int64_t>()
                                    : 14;
        return static_cast<std::uint32_t>(maximumK);
    }
    const std::int64_t defaultNeighbors =
        type == "filters.covariancefeatures" ? 10 : 8;
    const std::int64_t neighbors = stage.contains("knn")
                                       ? stage.at("knn").get<std::int64_t>()
                                       : defaultNeighbors;
    return static_cast<std::uint32_t>(neighbors) + 1U;
}

std::uint32_t neighborhoodDimensions(const Json& stage)
{
    const std::string type = stage.at("type").get<std::string>();
    return type == "filters.hag_nn" || type == "filters.hag_delaunay" ? 2U : 3U;
}

std::string_view neighborhoodReplacement(const Json& stage)
{
    const std::string type = stage.at("type").get<std::string>();
    if (type == "filters.nndistance")
        return HybridNnDistanceStage;
    if (type == "filters.hag_nn")
        return HybridHagNnStage;
    if (type == "filters.hag_delaunay")
        return HybridHagDelaunayStage;
    if (type == "filters.normal")
        return HybridNormalStage;
    if (type == "filters.eigenvalues")
        return HybridEigenvaluesStage;
    if (type == "filters.approximatecoplanar")
        return HybridApproximateCoplanarStage;
    if (type == "filters.lof")
        return HybridLofStage;
    if (type == "filters.estimaterank")
        return HybridEstimateRankStage;
    if (type == "filters.optimalneighborhood")
        return HybridOptimalNeighborhoodStage;
    if (type == "filters.neighborclassifier")
        return HybridNeighborClassifierStage;
    return HybridCovarianceFeaturesStage;
}

bool stageIsOrdinal(const Json& stage)
{
    const std::string type = stage.at("type").get<std::string>();
    return type == "filters.decimation" || type == "filters.head" ||
           type == "filters.tail";
}

bool stageIsValuePredicate(const Json& stage)
{
    const std::string type = stage.at("type").get<std::string>();
    return type == "filters.expression" || type == "filters.range" ||
           type == "filters.crop";
}

bool decimationNeedsStandardCountValidation(const Json& stage)
{
    if (stage.at("type").get<std::string>() != "filters.decimation")
        return false;
    const auto offset = stage.find("offset");
    return offset != stage.end() && offset->get<std::uint64_t>() != 0;
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.starts_with(prefix);
}

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character)
                   { return static_cast<char>(std::tolower(character)); });
    return value;
}

bool endsWith(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() && value.ends_with(suffix);
}

bool knownStableReader(std::string_view type)
{
    return type == "readers.las" || type == "readers.bpf" ||
           type == "readers.ply" || type == "readers.pcd" ||
           type == "readers.text";
}

bool knownStableReaderFilename(std::string filename)
{
    filename = lowercase(std::move(filename));
    if (endsWith(filename, ".copc.laz"))
        return false;
    return endsWith(filename, ".las") || endsWith(filename, ".laz") ||
           endsWith(filename, ".bpf") || endsWith(filename, ".ply") ||
           endsWith(filename, ".pcd") || endsWith(filename, ".txt") ||
           endsWith(filename, ".csv");
}

bool exactModeWriter(std::string_view type)
{
    return type == "writers.las";
}

bool preparedLayoutOrderObservable(const Json& stage)
{
    const auto option = stage.find("extra_dims");
    if (option == stage.end())
        return false;
    const auto isAll = [](const Json& value)
    {
        return value.is_string() &&
               trimWhitespace(value.get_ref<const std::string&>()) == "all";
    };
    if (isAll(*option))
        return true;
    if (!option->is_array())
        return false;
    return std::any_of(option->begin(), option->end(), isAll);
}

bool exactModeWriterFilename(std::string filename)
{
    filename = lowercase(std::move(filename));
    return endsWith(filename, ".las");
}

} // unnamed namespace

bool automaticApproximateCoplanarCudaDeviceQualified() noexcept
{
#if !PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR
    // Automatic selection is a product performance promise.  Ordinary builds
    // stay host-selected until the proposed model/SM/toolkit profile passes the
    // complete cross-host acceptance protocol.  A separate clean qualification
    // artifact enables this code without a runtime selector environment.
    return false;
#else
    if (!cudaBackendCompiled())
        return false;
    try
    {
        const std::vector<CudaDeviceSummary> devices = cudaDevices();
        if (devices.empty())
            return false;
        // A fresh CLI process uses CUDA ordinal zero.  The reference 4090 is
        // the only provisional profile with a physical exactness lane and an
        // at-threshold signal so far.  The qualification key includes the CUDA
        // toolkit major/minor; Vast.ai profiles enter a production allowlist
        // only after the complete cross-host gates pass.
        const CudaDeviceSummary& device = devices.front();
        const int toolkitVersion = cudaCompiledToolkitVersion();
        return device.computeMajor == 8 && device.computeMinor == 9 &&
               device.name == "NVIDIA GeForce RTX 4090" &&
               toolkitVersion / 1000 == 13 && (toolkitVersion % 1000) / 10 == 3;
    }
    catch (const std::exception&)
    {
        return false;
    }
#endif
}

bool automaticR4OutlierCudaDeviceQualified() noexcept
{
    if (!cudaBackendCompiled())
        return false;
    try
    {
        const std::vector<CudaDeviceSummary> devices = cudaDevices();
        if (devices.empty())
            return false;
        const CudaDeviceSummary& device = devices.front();
        const int toolkitVersion = cudaCompiledToolkitVersion();
        return device.computeMajor == 8 && device.computeMinor == 9 &&
               device.name == "NVIDIA GeForce RTX 4090" &&
               toolkitVersion / 1000 == 13 &&
               (toolkitVersion % 1000) / 10 == 3 &&
               nvidiaKernelDriverVersion() == "610.43.03";
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool automaticLabelNnDistanceCudaDeviceQualified() noexcept
{
    // The B0233 model is measured only on the same exact physical profile as
    // the B0227 public hybrid selector.  Functional CUDA support on another
    // device does not establish this complete-process performance promise.
    return automaticR4OutlierCudaDeviceQualified();
}

bool automaticR2GroundNormalizeDeviceQualified() noexcept
{
    // The B0239 result is a complete-process measurement on this one
    // SM/toolkit/driver profile. Functional exactness on another CUDA device
    // is not evidence that replacing the pinned host stage is profitable.
    return automaticR4OutlierCudaDeviceQualified();
}

bool preferAutomaticLabelNnDistanceCuda(std::uint64_t pointCount) noexcept
{
    if (pointCount < AutomaticLabelNnDistanceCudaMinimumPoints ||
        pointCount > AutomaticLabelNnDistanceCudaMaximumPoints)
        return false;

    // Least-squares complete-process fit over the exact 250K, 1M, 2M, 4M,
    // 8M, and 16M B0233 rows.  The independent 50K row is a measured loss and
    // remains below the hard floor rather than being extrapolated into the
    // device envelope.
    constexpr double HostFixedNanoseconds = -117'762'758.71262614;
    constexpr double HostNanosecondsPerPoint = 4'449.516262888824;
    constexpr double DeviceFixedNanoseconds = 256'286'291.3658352;
    constexpr double DeviceNanosecondsPerPoint = 350.92388364175963;
    const double points = static_cast<double>(pointCount);
    return DeviceFixedNanoseconds + DeviceNanosecondsPerPoint * points <
           HostFixedNanoseconds + HostNanosecondsPerPoint * points;
}

bool automaticLabelNnDistanceHybridCandidate(
    std::string_view pipelineJson) noexcept
{
    try
    {
        const Json root = Json::parse(pipelineJson, nullptr, true, true);
        const Json* pipeline = nullptr;
        if (root.is_array())
            pipeline = &root;
        else if (root.is_object())
        {
            const auto position = root.find("pipeline");
            if (position != root.end())
                pipeline = &*position;
        }
        return pipeline && pipeline->is_array() &&
               measuredLabelNnDistancePipelineGrammar(root, *pipeline);
    }
    catch (const Json::exception&)
    {
        return false;
    }
}

HybridPipelineRewrite rewriteHybridPipeline(std::string_view text,
                                            bool preserveStageBoundaries,
                                            bool enableExperimentalReplacements,
                                            std::uint64_t automaticPointCount,
                                            bool measuredR4Input,
                                            bool measuredLabelNnDistanceInput)
{
    Json root;
    try
    {
        root = Json::parse(text, nullptr, true, true);
    }
    catch (const Json::parse_error& error)
    {
        throw std::invalid_argument(std::string("invalid pipeline JSON: ") +
                                    error.what());
    }

    Json* pipeline = nullptr;
    if (root.is_array())
        pipeline = &root;
    else if (root.is_object())
    {
        const auto position = root.find("pipeline");
        if (position != root.end())
            pipeline = &*position;
    }
    if (!pipeline || !pipeline->is_array())
        throw std::invalid_argument(
            "pipeline root must be an array or contain a pipeline array");

    const bool automaticR4Candidate =
        measuredR4PipelineGrammar(root, *pipeline);
    // D0272/B0272: the literal 1M r4 CUDA outlier selector (B0227) is
    // retired from automatic selection. Since B0258's host worker kNN
    // passes and B0267's hashed sample table, the exact host path measures
    // faster than this route at 1M (0.58 s vs 0.72 s) and at 4M (2.55 s vs
    // 2.78 s forced): the route pays the fixed CUDA startup and a serial
    // host coordinate gather that the host path no longer has to beat. The
    // exact route stays available for its differential lanes behind an
    // explicit experimental opt-in; nothing else about it changes.
    const bool automaticR4Selected =
        automaticR4Candidate && !enableExperimentalReplacements &&
        automaticPointCount == 1'000'000U && measuredR4Input &&
        std::getenv("PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA") != nullptr;
    const bool automaticLabelNnDistanceCandidate =
        measuredLabelNnDistancePipelineGrammar(root, *pipeline);
    const bool automaticLabelNnDistanceSelected =
        automaticLabelNnDistanceCandidate && !enableExperimentalReplacements &&
        measuredLabelNnDistanceInput &&
        preferAutomaticLabelNnDistanceCuda(automaticPointCount);

    HybridPipelineRewrite result;
    result.hasPointCountDependentCudaCandidate =
        automaticLabelNnDistanceCandidate;
    Json rewritten = Json::array();
    bool inputOrderProven = false;
    bool singleKnownStableReader = false;
    bool exactWriter = false;
    bool linear = true;
    bool multipleViewsPossible = false;
    bool automaticInputCardinalityKnown = false;
    std::size_t readerCount = 0;
    std::size_t writerCount = 0;
    std::size_t index = 0;
    while (index < pipeline->size())
    {
        if (enableExperimentalReplacements &&
            eligibleStatsStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridStatsStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleInfoStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridInfoStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (eligibleExpressionStatsStage(pipeline->at(index)))
        {
            const std::size_t expressions =
                expressionStatsWork(pipeline->at(index));
            const bool countDependentCandidate =
                automaticInputCardinalityKnown && expressions != 0U;
            result.hasPointCountDependentCudaCandidate =
                result.hasPointCountDependentCudaCandidate ||
                countDependentCandidate;
            const bool automaticCuda =
                countDependentCandidate &&
                preferDefaultCudaExpressionStats(
                    static_cast<std::size_t>(automaticPointCount), expressions);
            if (enableExperimentalReplacements || automaticCuda)
            {
                Json replacement = pipeline->at(index);
                replacement["type"] = HybridExpressionStatsStage;
                if (automaticCuda && !enableExperimentalReplacements)
                    replacement["pdg_auto_cuda"] = true;
                rewritten.push_back(std::move(replacement));
                ++result.replacementRegions;
                ++result.fusedStages;
                ++index;
                continue;
            }
        }

        if (!enableExperimentalReplacements && automaticR4Candidate &&
            index == 1U)
        {
            result.hasPointCountDependentCudaCandidate = true;
            if (automaticR4Selected)
            {
                Json replacement = pipeline->at(index);
                replacement["type"] = HybridOutlierStage;
                replacement["pdg_auto_cuda"] = true;
                rewritten.push_back(std::move(replacement));
                ++result.replacementRegions;
                ++result.fusedStages;
                result.automaticR4OutlierCuda = true;
            }
            else
                rewritten.push_back(pipeline->at(index));
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        // The first, fact-free pass exists only to discover the input file.
        // Do not accidentally admit the adjacent range point program on its
        // own: B0227 measured the complete outlier/range/sample composition.
        if (!enableExperimentalReplacements && automaticR4Candidate &&
            !automaticR4Selected && index == 2U)
        {
            rewritten.push_back(pipeline->at(index));
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleOutlierStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridOutlierStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleRadialDensityStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridRadialDensityStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleRadiusAssignStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridRadiusAssignStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (!enableExperimentalReplacements &&
            automaticLabelNnDistanceCandidate && index == 1U)
        {
            if (automaticLabelNnDistanceSelected)
            {
                Json replacement = pipeline->at(index);
                replacement["type"] = HybridLabelDuplicatesStage;
                replacement["pdg_auto_cuda"] = true;
                rewritten.push_back(std::move(replacement));
                ++result.replacementRegions;
                ++result.fusedStages;
                result.automaticLabelNnDistanceCuda = true;
            }
            else
                rewritten.push_back(pipeline->at(index));
            ++index;
            continue;
        }

        // Preserve the exact candidate graph during the fact-free pass.  In
        // particular, do not rewrite its terminal assignment on its own: the
        // measured winner is the complete label/NNDistance/device-bridge
        // composition, not any one of its stages in isolation.
        if (!enableExperimentalReplacements &&
            automaticLabelNnDistanceCandidate &&
            !automaticLabelNnDistanceSelected && (index == 2U || index == 3U))
        {
            rewritten.push_back(pipeline->at(index));
            ++index;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleLabelDuplicatesStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridLabelDuplicatesStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleSmrfStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridSmrfStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligiblePmfStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridPmfStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleCsfStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridCsfStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleElmStage(pipeline->at(index)))
        {
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridElmStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

#if PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR
        if (!enableExperimentalReplacements && index == 1U &&
            pipeline->size() == 3U &&
            eligibleAutomaticApproximateCoplanarWriter(pipeline->at(2U)) &&
            eligibleAutomaticApproximateCoplanarStage(pipeline->at(index)))
        {
            const bool countDependentCandidate = automaticInputCardinalityKnown;
            result.hasPointCountDependentCudaCandidate =
                result.hasPointCountDependentCudaCandidate ||
                countDependentCandidate;
            const bool automaticCuda =
                countDependentCandidate &&
                automaticPointCount >=
                    AutomaticApproximateCoplanarCudaMinimumPoints;
            if (automaticCuda)
            {
                Json replacement = pipeline->at(index);
                replacement["type"] = HybridApproximateCoplanarStage;
                replacement["pdg_auto_cuda"] = true;
                replacement["pdg_region_id"] =
                    static_cast<std::uint64_t>(index) + 1U;
                replacement["pdg_region_neighbors"] = 8U;
                replacement["pdg_region_reuse"] = false;
                replacement["pdg_region_last"] = true;
                rewritten.push_back(std::move(replacement));
                ++result.replacementRegions;
                ++result.fusedStages;
                result.automaticApproximateCoplanarCuda = true;
                ++index;
                // Preserve order/cardinality, but do not extend the automatic
                // ownership region or infer reuse across this standalone gate.
                automaticInputCardinalityKnown = false;
                continue;
            }
        }
#endif

        if ((enableExperimentalReplacements ||
             (automaticLabelNnDistanceSelected && index == 2U)) &&
            eligibleNeighborhoodStage(pipeline->at(index)))
        {
            // An assign/ferry point program whose destination syntax proves
            // that it preserves membership, order, and XYZ is a bridge rather
            // than an ownership boundary. A later compatible neighborhood
            // client can then reuse the same device XYZ/index and resident
            // result columns. Predicates, ordinal stages, coordinate writes,
            // transformations, and every other stage remain hard boundaries.
            std::size_t regionEnd = index;
            std::uint32_t maximumNeighbors = 0;
            const std::uint32_t spatialDimensions =
                neighborhoodDimensions(pipeline->at(index));
            for (;;)
            {
                const std::size_t neighborhoodStart = regionEnd;
                while (regionEnd < pipeline->size() &&
                       eligibleNeighborhoodStage(pipeline->at(regionEnd)) &&
                       neighborhoodDimensions(pipeline->at(regionEnd)) ==
                           spatialDimensions)
                {
                    maximumNeighbors = (std::max)(maximumNeighbors,
                                                  neighborhoodRequest(
                                                      pipeline->at(regionEnd)));
                    ++regionEnd;
                }
                if (regionEnd == neighborhoodStart ||
                    regionEnd == pipeline->size() ||
                    !residentPointStagePreservesCoordinates(
                        pipeline->at(regionEnd)))
                    break;

                while (regionEnd < pipeline->size() &&
                       residentPointStagePreservesCoordinates(
                           pipeline->at(regionEnd)))
                    ++regionEnd;

                // A terminal bridge is part of the region.  Otherwise the
                // next loop consumes the following neighborhood run and
                // extends the maximum kNN envelope before anything is
                // emitted.
                if (regionEnd == pipeline->size() ||
                    !eligibleNeighborhoodStage(pipeline->at(regionEnd)) ||
                    neighborhoodDimensions(pipeline->at(regionEnd)) !=
                        spatialDimensions)
                    break;
            }
            // Zero is reserved for an unplanned one-stage invocation.  The
            // source position makes this identifier deterministic while only
            // needing to be unique within one rewritten pipeline execution.
            const std::uint64_t region = static_cast<std::uint64_t>(index) + 1U;
            bool firstNeighborhood = true;
            while (index < regionEnd)
            {
                if (eligibleNeighborhoodStage(pipeline->at(index)))
                {
                    Json replacement = pipeline->at(index);
                    replacement["type"] =
                        neighborhoodReplacement(pipeline->at(index));
                    replacement["pdg_region_id"] = region;
                    replacement["pdg_region_neighbors"] = maximumNeighbors;
                    if (spatialDimensions != 3U)
                        replacement["pdg_region_dimensions"] =
                            spatialDimensions;
                    replacement["pdg_region_reuse"] = !firstNeighborhood;
                    replacement["pdg_region_last"] = index + 1U == regionEnd;
                    if (automaticLabelNnDistanceSelected)
                        replacement["pdg_auto_cuda"] = true;
                    rewritten.push_back(std::move(replacement));
                    firstNeighborhood = false;
                    ++index;
                    ++result.replacementRegions;
                    ++result.fusedStages;
                    continue;
                }

                const std::size_t bridgeStart = index;
                Json bridge = Json::array();
                while (
                    index < regionEnd &&
                    residentPointStagePreservesCoordinates(pipeline->at(index)))
                {
                    bridge.push_back(pipeline->at(index));
                    ++index;
                }
                if (bridge.empty())
                    throw std::logic_error(
                        "resident neighborhood region contains an "
                        "unsupported bridge");
                Json replacement{
                    {"type", HybridPointProgramStage},
                    {"program", bridge.dump()},
                    {"pdg_neighborhood_region_id", region},
                    {"pdg_neighborhood_region_last", index == regionEnd}};
                if (automaticLabelNnDistanceSelected)
                    replacement["pdg_auto_cuda"] = true;
                rewritten.push_back(std::move(replacement));
                ++result.replacementRegions;
                ++result.pointProgramRegions;
                result.fusedStages += index - bridgeStart;
            }
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleColorinterpStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridColorinterpStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (eligibleDividerStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridDividerStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            inputOrderProven = true;
            multipleViewsPossible = true;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleSplitterStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridSplitterStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            inputOrderProven = true;
            multipleViewsPossible = true;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleReturnsStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridReturnsStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;

            // Return groups are stable within each input PointView and the
            // implementation is explicitly multi-view aware.
            inputOrderProven = true;
            multipleViewsPossible = true;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleMergeStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridMergeStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;

            // Upstream merge emits its one persistent PointView after
            // appending each incoming view in execution order.
            multipleViewsPossible = false;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleGroupByStage(pipeline->at(index)))
        {
            if (!inputOrderProven)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridGroupByStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;

            // This implementation is explicitly multi-view aware and retains
            // source-first view identity. Other native candidates remain
            // closed after this boundary until independently proved.
            inputOrderProven = true;
            multipleViewsPossible = true;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleMortonStage(pipeline->at(index)))
        {
            if (multipleViewsPossible)
                result.hasUnstableInputOrderRegion = true;
            if (!inputOrderProven)
            {
                rewritten.push_back(pipeline->at(index));
                inputOrderProven = true;
                ++index;
                continue;
            }
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridMortonOrderStage;
            rewritten.push_back(std::move(replacement));
            inputOrderProven = true;
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (eligibleOrderStage(pipeline->at(index)))
        {
            if (multipleViewsPossible)
                result.hasUnstableInputOrderRegion = true;
            if (!inputOrderProven)
            {
                // Preserve the upstream sort as the ordering barrier when the
                // reader's source order is not proven. This still permits
                // later exact regions without claiming the sort itself ran on
                // PDG.
                rewritten.push_back(pipeline->at(index));
                inputOrderProven = true;
                ++index;
                continue;
            }
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridOrderStage;
            rewritten.push_back(std::move(replacement));
            inputOrderProven = true;
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            continue;
        }

        if (enableExperimentalReplacements &&
            eligibleSkewnessStage(pipeline->at(index)))
        {
            if (multipleViewsPossible)
            {
                result.hasUnstableInputOrderRegion = true;
                rewritten.push_back(pipeline->at(index));
                inputOrderProven = false;
                ++index;
                continue;
            }
            if (!inputOrderProven)
            {
                // Preserve the upstream stage as an ordering barrier when a
                // prior native region has made source order unprovable. The
                // data-dependent CUDA tie gate cannot repair that boundary.
                rewritten.push_back(pipeline->at(index));
                inputOrderProven = true;
                ++index;
                continue;
            }
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridSkewnessStage;
            rewritten.push_back(std::move(replacement));
            inputOrderProven = true;
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleRobustStage(pipeline->at(index)))
        {
            if (!inputOrderProven || multipleViewsPossible)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            const std::string originalType =
                replacement.at("type").get<std::string>();
            replacement["type"] = HybridRobustStage;
            replacement["method"] =
                originalType == "filters.iqr" ? "iqr" : "mad";
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (eligibleLocateStage(pipeline->at(index)))
        {
            if (!inputOrderProven || multipleViewsPossible)
                result.hasUnstableInputOrderRegion = true;
            Json replacement = pipeline->at(index);
            replacement["type"] = HybridLocateStage;
            rewritten.push_back(std::move(replacement));
            ++result.replacementRegions;
            ++result.fusedStages;
            ++index;
            automaticInputCardinalityKnown = false;
            continue;
        }

        if (!eligiblePointStage(pipeline->at(index)))
        {
            const Json& stage = pipeline->at(index);
            if (stage.is_object())
            {
                if (stage.contains("tag") || stage.contains("inputs"))
                    linear = false;
                const auto type = stage.find("type");
                if (type != stage.end() && type->is_string())
                {
                    const std::string name = type->get<std::string>();
                    if (startsWith(name, "readers."))
                    {
                        ++readerCount;
                        const bool stable = knownStableReader(name);
                        singleKnownStableReader = stable && readerCount == 1U;
                        inputOrderProven = singleKnownStableReader;
                        automaticInputCardinalityKnown =
                            readerCount == 1U &&
                            eligibleAutomaticPointCountReader(stage);
                        if (automaticInputCardinalityKnown)
                        {
                            result.automaticPointCountFilename =
                                stage.at("filename").get<std::string>();
                            if (stage.contains("count"))
                                result.automaticPointCountLimit =
                                    stage.at("count").get<std::uint64_t>();
                        }
                        else
                            result.automaticPointCountFilename.clear();
                    }
                    else if (eligibleRandomizeBridge(stage))
                    {
                        // Upstream randomize mutates only the current
                        // PointView's index vector. It neither creates nor
                        // combines views, so an already-proven input remains
                        // safe for a later native region. Keep the original
                        // std::mt19937/std::shuffle implementation because its
                        // exact permutation is library-version dependent.
                    }
                    else if (eligibleStatsStage(stage) ||
                             eligibleInfoStage(stage) ||
                             eligibleExpressionStatsStage(stage))
                    {
                        // These upstream metadata stages observe each input
                        // PointView but neither mutate point order nor change
                        // view topology. Keep them as exact host bridges until
                        // native replacements have positive end-to-end gates.
                    }
                    else if (name == "filters.sort" ||
                             name == "filters.mortonorder")
                        inputOrderProven = true;
                    // An unchanged filter can preserve, split, merge, or
                    // synthesize PointViews. Until that individual stage's
                    // output-view contract is proved, later single-view
                    // native stages must not rewrite the graph. Ordering
                    // stages above establish point order but do not collapse
                    // existing views.
                    else if (startsWith(name, "filters."))
                    {
                        multipleViewsPossible = true;
                        automaticInputCardinalityKnown = false;
                    }
                    else if (startsWith(name, "writers."))
                    {
                        ++writerCount;
                        exactWriter =
                            writerCount == 1U && exactModeWriter(name);
                        result.preparedLayoutOrderObservable =
                            exactWriter && preparedLayoutOrderObservable(stage);
                    }
                }
                else
                    linear = false;
            }
            else if (stage.is_string())
            {
                if (index == 0U)
                {
                    ++readerCount;
                    singleKnownStableReader =
                        knownStableReaderFilename(stage.get<std::string>());
                    inputOrderProven = singleKnownStableReader;
                    const std::string filename = stage.get<std::string>();
                    const std::string normalized = lowercase(filename);
                    automaticInputCardinalityKnown =
                        singleKnownStableReader &&
                        (endsWith(normalized, ".las") ||
                         endsWith(normalized, ".laz"));
                    if (automaticInputCardinalityKnown)
                        result.automaticPointCountFilename = filename;
                    else
                        result.automaticPointCountFilename.clear();
                }
                else if (index + 1U == pipeline->size())
                {
                    ++writerCount;
                    exactWriter =
                        exactModeWriterFilename(stage.get<std::string>());
                }
                else
                {
                    linear = false;
                    inputOrderProven = false;
                }
            }
            else
                linear = false;
            rewritten.push_back(pipeline->at(index));
            ++index;
            continue;
        }

        if (!inputOrderProven || multipleViewsPossible)
            result.hasUnstableInputOrderRegion = true;
        Json region = Json::array();
        bool regionHasValuePredicate = false;
        while (index < pipeline->size() &&
               eligiblePointStage(pipeline->at(index)))
        {
            if (preserveStageBoundaries && !region.empty())
                break;
            const Json& candidate = pipeline->at(index);
            automaticInputCardinalityKnown =
                automaticInputCardinalityKnown &&
                pointStagePreservesCardinality(candidate);
            if (!region.empty() && regionHasValuePredicate &&
                stageIsOrdinal(candidate))
                break;
            result.standardModeRequiresPointCountValidation =
                result.standardModeRequiresPointCountValidation ||
                decimationNeedsStandardCountValidation(candidate);
            regionHasValuePredicate =
                regionHasValuePredicate || stageIsValuePredicate(candidate);
            region.push_back(candidate);
            ++index;
        }
        result.fusedStages += region.size();
        ++result.replacementRegions;
        ++result.pointProgramRegions;
        Json replacement{{"type", HybridPointProgramStage},
                         {"program", region.dump()}};
        rewritten.push_back(std::move(replacement));
    }

    *pipeline = std::move(rewritten);
    result.json = root.dump();
    result.linearPipeline = linear && readerCount == 1U && writerCount <= 1U;
    result.standardModeRewriteIsExact =
        result.linearPipeline && singleKnownStableReader && writerCount == 1U &&
        exactWriter && !result.hasUnstableInputOrderRegion;
    return result;
}

} // namespace pdg
