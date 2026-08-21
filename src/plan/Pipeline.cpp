#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>
#include <pdg/stages/Crop.hpp>
#include <pdg/stages/Csf.hpp>
#include <pdg/stages/Elm.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Information.hpp>
#include <pdg/stages/Locate.hpp>
#include <pdg/stages/Morton.hpp>
#include <pdg/stages/Ordering.hpp>
#include <pdg/stages/Ordinal.hpp>
#include <pdg/stages/Partition.hpp>
#include <pdg/stages/Pmf.hpp>
#include <pdg/stages/Range.hpp>
#include <pdg/stages/Skewness.hpp>
#include <pdg/stages/Smrf.hpp>
#include <pdg/stages/Summary.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace pdg
{

namespace
{
using Json = nlohmann::json;

bool contains(const std::vector<DimensionId>& values, DimensionId value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

void appendUnique(std::vector<DimensionId>& values, DimensionId value)
{
    if (!contains(values, value))
        values.push_back(value);
}

void appendUnique(std::vector<DimensionId>& values,
                  const std::vector<DimensionId>& additions)
{
    for (DimensionId value : additions)
        appendUnique(values, value);
}

void eraseDimensions(std::vector<DimensionId>& values,
                     const std::vector<DimensionId>& removed)
{
    std::erase_if(values,
                  [&](DimensionId value) { return contains(removed, value); });
}

std::string trim(std::string_view value)
{
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])))
        ++begin;
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return std::string(value.substr(begin, end - begin));
}

bool validTag(std::string_view tag)
{
    if (tag.empty() || !std::isalpha(static_cast<unsigned char>(tag.front())))
        return false;
    return std::all_of(tag.begin() + 1, tag.end(),
                       [](char character)
                       {
                           const unsigned char value =
                               static_cast<unsigned char>(character);
                           return std::isalnum(value) || character == '_';
                       });
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.starts_with(prefix);
}

std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](char character)
                   {
                       return static_cast<char>(
                           std::tolower(static_cast<unsigned char>(character)));
                   });
    return text;
}

std::string inferredLasType(std::string_view filename, StageRole role)
{
    const std::string extension =
        lowercase(std::filesystem::path(filename).extension().string());
    // Upstream PDAL infers readers.las/writers.las for `.laz` as well, so the
    // shorthand form must agree. Whether the engine may *map* the records is a
    // separate question answered by `native`/`isUncompressedLasFilename`.
    if (extension == ".las" || extension == ".laz")
        return role == StageRole::Reader ? "readers.las" : "writers.las";
    return role == StageRole::Reader ? "readers.unknown" : "writers.unknown";
}

bool isUncompressedLasFilename(std::string_view filename)
{
    return lowercase(std::filesystem::path(filename).extension().string()) ==
           ".las";
}

// `.las` or `.laz`: both carry the same public header, which is all
// header-derived placement facts need. Record access stays uncompressed-only.
bool isLasFamilyFilename(std::string_view filename)
{
    const std::string extension =
        lowercase(std::filesystem::path(filename).extension().string());
    return extension == ".las" || extension == ".laz";
}

bool hasOnlyOptions(const Json& node,
                    std::initializer_list<std::string_view> supported)
{
    for (const auto& [name, value] : node.items())
    {
        (void)value;
        if (std::find(supported.begin(), supported.end(), name) ==
            supported.end())
            return false;
    }
    return true;
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

std::int32_t signed32Option(const Json& node, std::string_view name,
                            std::int32_t fallback)
{
    const auto position = node.find(std::string(name));
    if (position == node.end())
        return fallback;
    if (!isSigned32Integer(*position))
        throw PlanError("pipeline field '" + std::string(name) +
                        "' must be a 32-bit integer");
    return position->get<std::int32_t>();
}

bool isExactDecimationStep(const Json& value)
{
    if (!value.is_number())
        return false;
    const double step = value.get<double>();
    return std::isfinite(step) && step >= 1.0;
}

std::uint64_t unsignedOption(const Json& node, std::string_view name,
                             std::uint64_t fallback)
{
    const auto position = node.find(std::string(name));
    if (position == node.end())
        return fallback;
    if (!isUnsignedInteger(*position))
        throw PlanError("pipeline field '" + std::string(name) +
                        "' must be a nonnegative integer");
    return position->get<std::uint64_t>();
}

bool booleanOption(const Json& node, std::string_view name, bool fallback)
{
    const auto position = node.find(std::string(name));
    if (position == node.end())
        return fallback;
    if (!position->is_boolean())
        throw PlanError("pipeline field '" + std::string(name) +
                        "' must be a boolean");
    return position->get<bool>();
}

void declarePureStage(StageDescriptor& descriptor,
                      bool cardinalityPreserving) noexcept
{
    descriptor.fusion.pure = true;
    descriptor.fusion.cardinalityPreserving = cardinalityPreserving;
    descriptor.fusion.deterministicSafe = true;
}

void declareFusablePointStage(StageDescriptor& descriptor, bool prologue,
                              bool epilogue) noexcept
{
    descriptor.fusion.fusableAsPrologue = prologue;
    descriptor.fusion.fusableAsEpilogue = epilogue;
}

void declareFusionAnchor(StageDescriptor& descriptor, bool prologue,
                         bool epilogue,
                         bool prologueConsumesPointWrites = false,
                         bool compactingPrologue = false) noexcept
{
    descriptor.fusion.acceptsFusedPrologue = prologue;
    descriptor.fusion.acceptsFusedEpilogue = epilogue;
    descriptor.fusion.prologueConsumesPointWrites = prologueConsumesPointWrites;
    descriptor.fusion.acceptsCompactingPrologue = compactingPrologue;
    descriptor.fusion.deterministicSafe = true;
}

WhereMergeMode parseWhereMerge(const Json& node)
{
    const bool hasWhere = node.contains("where");
    const auto position = node.find("where_merge");
    if (position == node.end())
        return hasWhere ? WhereMergeMode::Auto : WhereMergeMode::NotApplicable;
    if (position->is_boolean())
        return position->get<bool>() ? WhereMergeMode::MergeSkipped
                                     : WhereMergeMode::SeparateSkipped;
    if (position->is_string())
    {
        const std::string value = lowercase(position->get<std::string>());
        if (value == "auto")
            return WhereMergeMode::Auto;
        if (value == "true")
            return WhereMergeMode::MergeSkipped;
        if (value == "false")
            return WhereMergeMode::SeparateSkipped;
    }
    return WhereMergeMode::Invalid;
}

void finalizeFusionSemantics(StageDescriptor& descriptor, const Json& node)
{
    descriptor.fusion.dimsRead = descriptor.reads;
    descriptor.fusion.dimsWritten = descriptor.writes;
    descriptor.fusion.hasWhere = node.contains("where");
    descriptor.fusion.whereMerge = parseWhereMerge(node);
}

StageRole roleFor(std::string_view type, std::size_t index, std::size_t last)
{
    if (startsWith(type, "readers."))
        return StageRole::Reader;
    if (startsWith(type, "writers."))
        return StageRole::Writer;
    if (!type.empty())
        return StageRole::Filter;
    return (index == 0 || index != last) ? StageRole::Reader
                                         : StageRole::Writer;
}

std::string requireString(const Json& value, std::string_view field)
{
    if (!value.is_string())
        throw PlanError("pipeline field '" + std::string(field) +
                        "' must be a string");
    return value.get<std::string>();
}

std::vector<std::string> ferrySpecifications(const Json& node)
{
    const auto position = node.find("dimensions");
    if (position == node.end() || position->is_null())
        return {};

    std::vector<std::string> specifications;
    const auto append = [&](const Json& value)
    {
        const std::string text = requireString(value, "dimensions");
        std::size_t begin = 0;
        while (begin <= text.size())
        {
            const std::size_t comma = text.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? text.size() : comma;
            const std::string item =
                trim(std::string_view(text).substr(begin, end - begin));
            if (item.empty())
                throw PlanError("filters.ferry contains an empty dimension "
                                "mapping");
            specifications.push_back(item);
            if (comma == std::string::npos)
                break;
            begin = comma + 1;
        }
    };

    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return specifications;
}

std::vector<std::string> assignSpecifications(const Json& node)
{
    const auto position = node.find("value");
    if (position == node.end() || position->is_null())
        return {};

    std::vector<std::string> specifications;
    const auto append = [&](const Json& value)
    { specifications.push_back(requireString(value, "value")); };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return specifications;
}

std::vector<std::string> rangeSpecifications(const Json& node)
{
    const auto position = node.find("limits");
    if (position == node.end() || position->is_null())
        return {};

    std::vector<std::string> specifications;
    const auto append = [&](const Json& value)
    { specifications.push_back(requireString(value, "limits")); };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return specifications;
}

std::vector<std::string> stringSpecifications(const Json& node,
                                              std::string_view key)
{
    const auto position = node.find(std::string(key));
    if (position == node.end() || position->is_null())
        return {};
    std::vector<std::string> specifications;
    const auto append = [&](const Json& value)
    { specifications.push_back(requireString(value, key)); };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return specifications;
}

std::vector<std::string> labelDuplicateDimensions(const Json& node)
{
    const auto position = node.find("dimensions");
    if (position == node.end() || position->is_null())
        return {};

    std::vector<std::string> names;
    const auto append = [&](const Json& value)
    {
        const std::string text = requireString(value, "dimensions");
        std::size_t begin = 0U;
        while (begin < text.size())
        {
            const std::size_t comma = text.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? text.size() : comma;
            if (begin != end)
                names.push_back(
                    trim(std::string_view(text).substr(begin, end - begin)));
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
    };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return names;
}

bool appendRadiusDomainReads(const Json& node, std::string_view key,
                             const DimensionRegistry& dimensions,
                             StageDescriptor& descriptor)
{
    for (const std::string& specification : stringSpecifications(node, key))
    {
        std::size_t begin = 0U;
        while (begin < specification.size())
        {
            const std::size_t end = specification.find(',', begin);
            std::string range = trim(std::string_view(specification)
                                         .substr(begin, end == std::string::npos
                                                            ? std::string::npos
                                                            : end - begin));
            const std::size_t opener = range.find_first_of("[(");
            if (opener == std::string::npos)
                return false;
            std::string name = trim(std::string_view(range).substr(0, opener));
            if (!name.empty() && name.back() == '!')
            {
                name.pop_back();
                name = trim(name);
            }
            const DimensionDefinition* definition = dimensions.find(name);
            if (!definition)
                return false;
            appendUnique(descriptor.reads, definition->id);
            if (end == std::string::npos)
                break;
            begin = end + 1U;
        }
    }
    return true;
}

std::string cropBounds(const Json& node)
{
    const auto position = node.find("bounds");
    if (position == node.end() || !position->is_string())
        throw PlanError(
            "filters.crop native execution requires one bounds string");
    return position->get<std::string>();
}

bool cropOutside(const Json& node)
{
    const auto position = node.find("outside");
    if (position == node.end())
        return false;
    if (!position->is_boolean())
        throw PlanError("filters.crop outside must be a boolean");
    return position->get<bool>();
}

FerryProgram compileFerry(const Json& node, DimensionRegistry& dimensions,
                          StageDescriptor& descriptor)
{
    FerryProgram program;
    std::set<std::uint32_t> destinations;
    for (const std::string& specification : ferrySpecifications(node))
    {
        const std::size_t separator = specification.find('=');
        if (separator == std::string::npos)
            throw PlanError("invalid filters.ferry mapping '" + specification +
                            "'; expected source=>destination");
        std::string sourceName =
            trim(std::string_view(specification).substr(0, separator));
        std::string destinationName =
            trim(std::string_view(specification).substr(separator + 1));
        if (!destinationName.empty() && destinationName.front() == '>')
        {
            destinationName.erase(destinationName.begin());
            destinationName = trim(destinationName);
        }
        if (destinationName.empty() ||
            destinationName.find('=') != std::string::npos)
            throw PlanError("invalid filters.ferry mapping '" + specification +
                            "'");

        const DimensionDefinition* source = nullptr;
        if (!sourceName.empty())
        {
            source = dimensions.find(sourceName);
            if (!source)
                throw PlanError("filters.ferry source dimension does not "
                                "exist: " +
                                sourceName);
        }

        const DimensionDefinition* destination =
            dimensions.find(destinationName);
        const bool destinationCreated = !destination;
        if (!destination)
            destination = &dimensions.registerCustom(
                destinationName, source ? source->type : DimensionType::Double);
        if (source && source->id == destination->id)
            throw PlanError(
                "filters.ferry cannot copy a dimension to itself: " +
                sourceName);
        if (!destinations.insert(destination->id.value()).second)
            throw PlanError("filters.ferry has duplicate destination: " +
                            destinationName);

        program.copies.push_back({source != nullptr,
                                  source ? source->id : DimensionId{},
                                  destination->id, destinationCreated});
        if (source)
            appendUnique(descriptor.reads, source->id);
        if (!destinationCreated)
            appendUnique(descriptor.reads, destination->id);
        appendUnique(descriptor.writes, destination->id);
    }
    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    descriptor.mutatesCoordinates = contains(descriptor.writes, x) ||
                                    contains(descriptor.writes, y) ||
                                    contains(descriptor.writes, z);
    declarePureStage(descriptor, true);
    declareFusablePointStage(descriptor, true, true);
    return program;
}

AssignProgram compileAssign(const Json& node, DimensionRegistry& dimensions,
                            StageDescriptor& descriptor)
{
    AssignProgram program;
    try
    {
        const std::vector<std::string> specifications =
            assignSpecifications(node);
        program = compileAssignments(specifications, dimensions);
    }
    catch (const ExpressionError& error)
    {
        throw PlanError(std::string("invalid filters.assign expression: ") +
                        error.what());
    }

    descriptor.reads = program.reads;
    descriptor.writes = program.writes;
    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    descriptor.mutatesCoordinates = contains(descriptor.writes, x) ||
                                    contains(descriptor.writes, y) ||
                                    contains(descriptor.writes, z);
    declarePureStage(descriptor, true);
    declareFusablePointStage(descriptor, true, true);
    return program;
}

PredicateProgram compileExpression(const Json& node,
                                   DimensionRegistry& dimensions,
                                   StageDescriptor& descriptor)
{
    const auto position = node.find("expression");
    if (position == node.end() || !position->is_string())
        throw PlanError(
            "filters.expression native execution requires one expression");
    PredicateProgram program;
    try
    {
        program = compilePredicate(position->get<std::string>(), dimensions);
    }
    catch (const ExpressionError& error)
    {
        throw PlanError(std::string("invalid filters.expression predicate: ") +
                        error.what());
    }
    descriptor.reads = program.reads;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    // A declared stable predicate may be embedded in a consumer prologue
    // whose anchor declares compaction support; it is never a producer
    // epilogue.
    declareFusablePointStage(descriptor, true, false);
    return program;
}

PredicateProgram compileRange(const Json& node, DimensionRegistry& dimensions,
                              StageDescriptor& descriptor)
{
    PredicateProgram program;
    try
    {
        program = compileRangePredicate(rangeSpecifications(node), dimensions);
    }
    catch (const ExpressionError& error)
    {
        throw PlanError(std::string("invalid filters.range limits: ") +
                        error.what());
    }
    descriptor.reads = program.reads;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

PredicateProgram compileCrop(const Json& node, DimensionRegistry& dimensions,
                             StageDescriptor& descriptor)
{
    PredicateProgram program;
    try
    {
        program = compileCropPredicate(cropBounds(node), cropOutside(node),
                                       dimensions);
    }
    catch (const ExpressionError& error)
    {
        throw PlanError(std::string("invalid filters.crop bounds: ") +
                        error.what());
    }
    descriptor.reads = program.reads;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

OrdinalProgram compileOrdinal(const Json& node, std::string_view type)
{
    OrdinalProgram program;
    if (type == "filters.decimation")
    {
        program.kind = OrdinalKind::Decimation;
        const auto step = node.find("step");
        if (step != node.end())
        {
            if (!step->is_number())
                throw PlanError("pipeline field 'step' must be a number");
            program.step = step->get<double>();
        }
        program.offset = unsignedOption(node, "offset", 0);
        program.limit = unsignedOption(
            node, "limit", (std::numeric_limits<std::uint64_t>::max)());
    }
    else
    {
        program.kind =
            type == "filters.head" ? OrdinalKind::Head : OrdinalKind::Tail;
        program.count = unsignedOption(node, "count", 10);
        program.invert = booleanOption(node, "invert", false);
    }
    if (!ordinalSupportsMode(program, OrdinalMode::Standard))
        throw PlanError(std::string(type) +
                        " options are outside the exact ordinal envelope");
    return program;
}

LocateProgram compileLocate(const Json& node, DimensionRegistry& dimensions,
                            StageDescriptor& descriptor)
{
    const std::string name = requireString(node.at("dimension"), "dimension");
    const DimensionDefinition* definition = dimensions.find(name);
    if (!definition)
        throw PlanError("filters.locate dimension does not exist: " + name);

    LocateProgram program;
    program.dimension = definition->id;
    const auto minmax = node.find("minmax");
    const std::string kind =
        lowercase(minmax == node.end() ? std::string("max")
                                       : requireString(*minmax, "minmax"));
    if (kind == "min")
        program.kind = LocateKind::Minimum;
    else if (kind == "max")
        program.kind = LocateKind::Maximum;
    else
        program.kind = LocateKind::None;
    descriptor.reads.push_back(program.dimension);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

TransformationProgram compileTransformationStage(const Json& node,
                                                 StageDescriptor& descriptor)
{
    TransformationProgram program;
    try
    {
        program =
            compileTransformation(requireString(node.at("matrix"), "matrix"));
    }
    catch (const std::exception& error)
    {
        throw PlanError(std::string("invalid filters.transformation matrix: ") +
                        error.what());
    }
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = descriptor.reads;
    descriptor.mutatesCoordinates = true;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

RobustProgram compileRobustStage(const Json& node,
                                 DimensionRegistry& dimensions,
                                 StageDescriptor& descriptor, RobustKind kind)
{
    const std::string name = requireString(node.at("dimension"), "dimension");
    const DimensionDefinition* definition = dimensions.find(name);
    if (!definition)
        throw PlanError("robust filter dimension does not exist: " + name);
    RobustProgram program;
    program.dimension = definition->id;
    program.kind = kind;
    program.multiplier = node.contains("k")
                             ? node.at("k").get<double>()
                             : (kind == RobustKind::Iqr ? 1.5 : 2.0);
    if (kind == RobustKind::Mad && node.contains("mad_multiplier"))
        program.madMultiplier = node.at("mad_multiplier").get<double>();
    descriptor.reads.push_back(program.dimension);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

OutlierProgram compileOutlierStage(const Json& node,
                                   StageDescriptor& descriptor)
{
    OutlierProgram program;
    const std::string method = lowercase(
        node.contains("method") ? requireString(node.at("method"), "method")
                                : std::string("statistical"));
    if (method == "statistical")
        program.method = OutlierMethod::Statistical;
    else if (method == "radius")
        program.method = OutlierMethod::Radius;
    else
        program.method = OutlierMethod::Unknown;
    program.minimumNeighbors = signed32Option(node, "min_k", 2);
    program.radius =
        node.contains("radius") ? node.at("radius").get<double>() : 1.0;
    program.meanNeighbors = signed32Option(node, "mean_k", 8);
    program.multiplier =
        node.contains("multiplier") ? node.at("multiplier").get<double>() : 2.0;
    program.classification = static_cast<std::uint8_t>(
        node.contains("class") ? node.at("class").get<std::uint32_t>() : 7U);

    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.preservesOrder = true;
    if (program.method == OutlierMethod::Radius &&
        std::isfinite(program.radius) && program.radius > 0.0)
    {
        descriptor.index.kind = IndexKind::Radius;
        descriptor.index.radius = program.radius;
        descriptor.maximumRadius = program.radius;
    }
    else if (program.method == OutlierMethod::Statistical &&
             program.meanNeighbors >= 0 &&
             static_cast<std::uint64_t>(program.meanNeighbors) + 1U <=
                 std::numeric_limits<std::uint32_t>::max())
    {
        descriptor.index.kind = IndexKind::Knn;
        descriptor.index.neighbors =
            static_cast<std::uint32_t>(program.meanNeighbors) + 1U;
    }
    declarePureStage(descriptor, true);
    return program;
}

RadialDensityProgram compileRadialDensityStage(const Json& node,
                                               StageDescriptor& descriptor)
{
    RadialDensityProgram program;
    program.radius =
        node.contains("radius") ? node.at("radius").get<double>() : 1.0;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::RadialDensity)};
    descriptor.preservesOrder = true;
    if (std::isfinite(program.radius) && program.radius > 0.0)
    {
        descriptor.index.kind = IndexKind::Radius;
        descriptor.index.radius = program.radius;
        descriptor.maximumRadius = program.radius;
    }
    declarePureStage(descriptor, true);
    return program;
}

RadiusAssignProgram compileRadiusAssignStage(const Json& node,
                                             DimensionRegistry& dimensions,
                                             StageDescriptor& descriptor)
{
    RadiusAssignProgram program;
    program.radius = node.at("radius").get<double>();
    program.search3d =
        node.contains("is3d") ? node.at("is3d").get<bool>() : false;
    program.maximumAbove = node.contains("max2d_above")
                               ? node.at("max2d_above").get<double>()
                               : -1.0;
    program.maximumBelow = node.contains("max2d_below")
                               ? node.at("max2d_below").get<double>()
                               : -1.0;
    program.updates = compileAssignments(
        stringSpecifications(node, "update_expression"), dimensions);
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    if (!appendRadiusDomainReads(node, "src_domain", dimensions, descriptor) ||
        !appendRadiusDomainReads(node, "reference_domain", dimensions,
                                 descriptor))
        throw PlanError("filters.radiusassign domain dimension does not exist");
    appendUnique(descriptor.reads, program.updates.reads);
    descriptor.writes = program.updates.writes;
    descriptor.index.kind = IndexKind::Radius;
    descriptor.index.radius = program.radius;
    descriptor.index.dimensions = program.search3d ? 3U : 2U;
    descriptor.maximumRadius = program.radius;
    descriptor.preservesOrder = true;
    descriptor.mutatesCoordinates =
        contains(descriptor.writes, DimensionId(StandardDimension::X)) ||
        contains(descriptor.writes, DimensionId(StandardDimension::Y)) ||
        contains(descriptor.writes, DimensionId(StandardDimension::Z));
    declarePureStage(descriptor, true);
    return program;
}

NormalProgram compileNormalStage(const Json& node, StageDescriptor& descriptor)
{
    NormalProgram program;
    program.neighbors = signed32Option(node, "knn", 8);
    program.alwaysUp =
        node.contains("always_up") ? node.at("always_up").get<bool>() : true;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::NormalX),
                         DimensionId(StandardDimension::NormalY),
                         DimensionId(StandardDimension::NormalZ),
                         DimensionId(StandardDimension::Curvature)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors =
        static_cast<std::uint32_t>(program.neighbors) + 1U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

ApproximateCoplanarProgram
compileApproximateCoplanarStage(const Json& node, StageDescriptor& descriptor)
{
    ApproximateCoplanarProgram program;
    program.neighbors = signed32Option(node, "knn", 8);
    program.threshold1 =
        node.contains("thresh1") ? node.at("thresh1").get<double>() : 25.0;
    program.threshold2 =
        node.contains("thresh2") ? node.at("thresh2").get<double>() : 6.0;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::Coplanar)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = static_cast<std::uint32_t>(program.neighbors);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

NeighborClassifierProgram
compileNeighborClassifierStage(const Json& node, StageDescriptor& descriptor)
{
    NeighborClassifierProgram program;
    program.neighbors = signed32Option(node, "k", 0);
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = static_cast<std::uint32_t>(program.neighbors);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

OptimalNeighborhoodProgram
compileOptimalNeighborhoodStage(const Json& node, StageDescriptor& descriptor)
{
    OptimalNeighborhoodProgram program;
    program.minimumK =
        static_cast<std::uint32_t>(unsignedOption(node, "min_k", 10U));
    program.maximumK =
        static_cast<std::uint32_t>(unsignedOption(node, "max_k", 14U));
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::OptimalKNN),
                         DimensionId(StandardDimension::OptimalRadius)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = program.maximumK;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

EstimateRankProgram compileEstimateRankStage(const Json& node,
                                             StageDescriptor& descriptor)
{
    EstimateRankProgram program;
    program.neighbors = signed32Option(node, "knn", 8);
    program.threshold =
        node.contains("thresh") ? node.at("thresh").get<double>() : 0.01;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::Rank)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = static_cast<std::uint32_t>(program.neighbors);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

LofProgram compileLofStage(const Json& node, StageDescriptor& descriptor)
{
    LofProgram program;
    program.minimumPoints = signed32Option(node, "minpts", 10);
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {
        DimensionId(StandardDimension::NNDistance),
        DimensionId(StandardDimension::LocalReachabilityDistance),
        DimensionId(StandardDimension::LocalOutlierFactor)};
    descriptor.index.kind = IndexKind::Knn;
    // The executing stage replicates upstream's self-inclusive increment.
    descriptor.index.neighbors =
        static_cast<std::uint32_t>(program.minimumPoints) + 1U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

NnDistanceProgram compileNnDistanceStage(const Json& node,
                                         StageDescriptor& descriptor)
{
    NnDistanceProgram program;
    program.k = static_cast<std::uint32_t>(unsignedOption(node, "k", 10U));
    if (node.contains("mode") &&
        lowercase(requireString(node.at("mode"), "mode")) == "avg")
        program.mode = KnnDistanceMode::Average;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::NNDistance)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = program.k + 1U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

HagNnProgram compileHagNnStage(const Json& node, StageDescriptor& descriptor)
{
    HagNnProgram program;
    program.count =
        static_cast<std::uint32_t>(unsignedOption(node, "count", 1U));
    program.maximumDistance = node.contains("max_distance")
                                  ? node.at("max_distance").get<double>()
                                  : 0.0;
    program.allowExtrapolation =
        node.contains("allow_extrapolation")
            ? node.at("allow_extrapolation").get<bool>()
            : true;
    program.groundClass =
        static_cast<std::uint8_t>(unsignedOption(node, "class", 2U));
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification)};
    descriptor.writes = {DimensionId(StandardDimension::HeightAboveGround)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = program.count;
    descriptor.index.dimensions = 2U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

HagDelaunayProgram compileHagDelaunayStage(const Json& node,
                                           StageDescriptor& descriptor)
{
    HagDelaunayProgram program;
    program.count = 3U;
    program.allowExtrapolation =
        node.contains("allow_extrapolation")
            ? node.at("allow_extrapolation").get<bool>()
            : true;
    program.groundClass =
        static_cast<std::uint8_t>(unsignedOption(node, "class", 2U));
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification)};
    descriptor.writes = {DimensionId(StandardDimension::HeightAboveGround)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors = program.count;
    descriptor.index.dimensions = 2U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

EigenvaluesProgram compileEigenvaluesStage(const Json& node,
                                           StageDescriptor& descriptor)
{
    EigenvaluesProgram program;
    program.neighbors = signed32Option(node, "knn", 8);
    program.normalize =
        node.contains("normalize") ? node.at("normalize").get<bool>() : false;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = {DimensionId(StandardDimension::Eigenvalue0),
                         DimensionId(StandardDimension::Eigenvalue1),
                         DimensionId(StandardDimension::Eigenvalue2)};
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors =
        static_cast<std::uint32_t>(program.neighbors) + 1U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

bool parseCovarianceFeatureSet(const Json& node, std::uint32_t& mask,
                               std::vector<DimensionId>& dimensions)
{
    mask = 0U;
    dimensions.clear();
    const auto add = [&](std::string feature)
    {
        feature = lowercase(trim(feature));
        const auto append =
            [&](CovarianceFeature bit, StandardDimension dimension)
        {
            mask |= static_cast<std::uint32_t>(bit);
            appendUnique(dimensions, DimensionId(dimension));
        };
        if (feature == "dimensionality")
        {
            append(CovarianceLinearity, StandardDimension::Linearity);
            append(CovariancePlanarity, StandardDimension::Planarity);
            append(CovarianceScattering, StandardDimension::Scattering);
            append(CovarianceVerticality, StandardDimension::Verticality);
        }
        else if (feature == "linearity")
            append(CovarianceLinearity, StandardDimension::Linearity);
        else if (feature == "planarity")
            append(CovariancePlanarity, StandardDimension::Planarity);
        else if (feature == "scattering")
            append(CovarianceScattering, StandardDimension::Scattering);
        else if (feature == "verticality")
            append(CovarianceVerticality, StandardDimension::Verticality);
        else if (feature == "omnivariance")
            append(CovarianceOmnivariance, StandardDimension::Omnivariance);
        else if (feature == "anisotropy")
            append(CovarianceAnisotropy, StandardDimension::Anisotropy);
        else if (feature == "eigenentropy")
            append(CovarianceEigenentropy, StandardDimension::Eigenentropy);
        else if (feature == "eigenvaluesum")
            append(CovarianceEigenvalueSum, StandardDimension::EigenvalueSum);
        else if (feature == "surfacevariation")
            append(CovarianceSurfaceVariation,
                   StandardDimension::SurfaceVariation);
        else if (feature == "demantkeverticality")
            append(CovarianceDemantkeVerticality,
                   StandardDimension::DemantkeVerticality);
        else
            return false;
        return true;
    };
    const auto appendText = [&](const Json& value)
    {
        if (!value.is_string())
            return false;
        const std::string text = value.get<std::string>();
        std::size_t begin = 0;
        while (begin <= text.size())
        {
            const std::size_t comma = text.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? text.size() : comma;
            if (!add(text.substr(begin, end - begin)))
                return false;
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
        return true;
    };

    const auto position = node.find("feature_set");
    if (position == node.end())
        return add("dimensionality");
    if (position->is_array())
    {
        if (position->empty())
            return false;
        for (const Json& value : *position)
            if (!appendText(value))
                return false;
        return mask != 0U;
    }
    return appendText(*position) && mask != 0U;
}

CovarianceFeaturesProgram
compileCovarianceFeaturesStage(const Json& node, StageDescriptor& descriptor)
{
    CovarianceFeaturesProgram program;
    program.neighbors = signed32Option(node, "knn", 10);
    std::vector<DimensionId> writes;
    if (!parseCovarianceFeatureSet(node, program.features, writes))
        throw PlanError("unsupported covariance feature set");
    if (node.contains("mode"))
    {
        const std::string mode =
            lowercase(requireString(node.at("mode"), "mode"));
        if (mode == "raw")
            program.mode = EigenvalueMode::Raw;
        else if (mode == "normalized")
            program.mode = EigenvalueMode::Normalized;
        else
            program.mode = EigenvalueMode::Sqrt;
    }
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.writes = std::move(writes);
    descriptor.index.kind = IndexKind::Knn;
    descriptor.index.neighbors =
        static_cast<std::uint32_t>(program.neighbors) + 1U;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

OrderingProgram compileOrderingStage(const Json& node,
                                     DimensionRegistry& dimensions,
                                     StageDescriptor& descriptor)
{
    const Json* values = nullptr;
    if (node.contains("dimension"))
        values = &node.at("dimension");
    else if (node.contains("dimensions"))
        values = &node.at("dimensions");
    if (!values)
        throw PlanError("filters.sort requires at least one dimension");

    std::vector<std::string> names;
    if (values->is_array())
    {
        for (const Json& value : *values)
            names.push_back(requireString(value, "dimensions"));
    }
    else
        names.push_back(requireString(*values, "dimensions"));
    if (names.empty())
        throw PlanError("filters.sort requires at least one dimension");

    OrderingProgram program;
    for (const std::string& name : names)
    {
        const DimensionDefinition* definition = dimensions.find(name);
        if (!definition)
            throw PlanError("filters.sort dimension does not exist: " + name);
        program.dimensions.push_back(definition->id);
        appendUnique(descriptor.reads, definition->id);
    }

    const std::string order = lowercase(
        node.contains("order") ? requireString(node.at("order"), "order")
                               : std::string("asc"));
    if (order == "asc")
        program.direction = OrderingDirection::Ascending;
    else if (order == "desc")
        program.direction = OrderingDirection::Descending;
    else
        throw PlanError("filters.sort order must be ASC or DESC");

    const std::string algorithm =
        lowercase(node.contains("algorithm")
                      ? requireString(node.at("algorithm"), "algorithm")
                      : std::string("normal"));
    if (algorithm == "normal")
        program.algorithm = OrderingAlgorithm::Normal;
    else if (algorithm == "stable")
        program.algorithm = OrderingAlgorithm::Stable;
    else
        throw PlanError("filters.sort algorithm must be NORMAL or STABLE");
    descriptor.preservesOrder = false;
    declarePureStage(descriptor, true);
    return program;
}

LabelDuplicatesProgram
compileLabelDuplicatesStage(const Json& node, DimensionRegistry& dimensions,
                            StageDescriptor& descriptor)
{
    LabelDuplicatesProgram program;
    for (const std::string& name : labelDuplicateDimensions(node))
    {
        const DimensionDefinition* definition = dimensions.find(name);
        if (!definition)
            throw PlanError(
                "filters.label_duplicates dimension does not exist: " + name);
        program.dimensions.push_back(definition->id);
        appendUnique(descriptor.reads, definition->id);
    }
    const DimensionId duplicate(StandardDimension::Duplicate);
    appendUnique(descriptor.reads, duplicate);
    descriptor.writes = {duplicate};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

SmrfProgram compileSmrfStage(const Json& node, StageDescriptor& descriptor)
{
    SmrfProgram program;
    program.cell = node.contains("cell") ? node.at("cell").get<double>() : 1.0;
    program.slope =
        node.contains("slope") ? node.at("slope").get<double>() : 0.15;
    program.scalar =
        node.contains("scalar") ? node.at("scalar").get<double>() : 1.25;
    program.threshold =
        node.contains("threshold") ? node.at("threshold").get<double>() : 0.5;
    program.cut = node.contains("cut") ? node.at("cut").get<double>() : 0.0;
    program.window = node.contains("window") ? node.at("window").get<double>()
                                             : 18.0 * program.cell;
    program.groundClass = node.contains("ground_class")
                              ? node.at("ground_class").get<std::uint8_t>()
                              : std::uint8_t(2);
    program.otherClass = node.contains("other_class")
                             ? node.at("other_class").get<std::uint8_t>()
                             : std::uint8_t(1);
    program.onlyGround = node.contains("only_ground")
                             ? node.at("only_ground").get<bool>()
                             : false;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification),
                        DimensionId(StandardDimension::ReturnNumber),
                        DimensionId(StandardDimension::NumberOfReturns)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

PmfProgram compilePmfStage(const Json& node, StageDescriptor& descriptor)
{
    PmfProgram program;
    program.cellSize =
        node.contains("cell_size") ? node.at("cell_size").get<double>() : 1.0;
    program.exponential = node.contains("exponential")
                              ? node.at("exponential").get<bool>()
                              : true;
    program.initialDistance = node.contains("initial_distance")
                                  ? node.at("initial_distance").get<double>()
                                  : 0.15;
    program.maxDistance = node.contains("max_distance")
                              ? node.at("max_distance").get<double>()
                              : 2.5;
    program.maxWindowSize = node.contains("max_window_size")
                                ? node.at("max_window_size").get<double>()
                                : 33.0;
    program.slope =
        node.contains("slope") ? node.at("slope").get<double>() : 1.0;
    program.groundClass = node.contains("ground_class")
                              ? node.at("ground_class").get<std::uint8_t>()
                              : std::uint8_t(2);
    program.otherClass = node.contains("other_class")
                             ? node.at("other_class").get<std::uint8_t>()
                             : std::uint8_t(1);
    program.onlyGround = node.contains("only_ground")
                             ? node.at("only_ground").get<bool>()
                             : false;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification),
                        DimensionId(StandardDimension::ReturnNumber),
                        DimensionId(StandardDimension::NumberOfReturns)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.grid.framePolicy = GridFramePolicy::PmfInitialLookupV1;
    descriptor.grid.cellSize = program.cellSize;
    descriptor.grid.deviceBytesPerCell = PmfTiledDeviceBytesPerCell;
    descriptor.grid.deviceBackingCount = 2U;
    descriptor.grid.deviceProofBytesPerCell = PmfTiledDeviceProofBytesPerCell;
    descriptor.grid.deviceFixedBytes = PmfTiledDeviceFixedScratchBytes;
    descriptor.grid.hostBytesPerPoint = PmfTiledHostStagingBytesPerPoint;
    descriptor.grid.hostBytesPerCell = PmfTiledHostBytesPerCell;
    descriptor.grid.hostTileBytesPerExpandedCell = sizeof(double);
    descriptor.grid.maximumHaloCells = 1U;
    descriptor.grid.phaseSynchronized = true;
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

CsfProgram compileCsfStage(const Json& node, StageDescriptor& descriptor)
{
    CsfProgram program;
    program.smooth =
        node.contains("smooth") ? node.at("smooth").get<bool>() : true;
    program.timeStep =
        node.contains("step") ? node.at("step").get<double>() : 0.65;
    program.classThreshold =
        node.contains("threshold") ? node.at("threshold").get<double>() : 0.5;
    program.heightThreshold =
        node.contains("hdiff") ? node.at("hdiff").get<double>() : 0.3;
    program.resolution =
        node.contains("resolution") ? node.at("resolution").get<double>() : 1.0;
    program.rigidness =
        node.contains("rigidness") ? node.at("rigidness").get<int>() : 3;
    program.iterations =
        node.contains("iterations") ? node.at("iterations").get<int>() : 500;
    program.groundClass = node.contains("ground_class")
                              ? node.at("ground_class").get<std::uint8_t>()
                              : std::uint8_t(2);
    program.otherClass = node.contains("other_class")
                             ? node.at("other_class").get<std::uint8_t>()
                             : std::uint8_t(1);
    program.onlyGround = node.contains("only_ground")
                             ? node.at("only_ground").get<bool>()
                             : false;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification),
                        DimensionId(StandardDimension::ReturnNumber),
                        DimensionId(StandardDimension::NumberOfReturns)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

ElmProgram compileElmStage(const Json& node, StageDescriptor& descriptor)
{
    ElmProgram program;
    program.cell = node.contains("cell") ? node.at("cell").get<double>() : 10.0;
    program.classification = node.contains("class")
                                 ? node.at("class").get<std::uint8_t>()
                                 : std::uint8_t(7);
    program.threshold =
        node.contains("threshold") ? node.at("threshold").get<double>() : 1.0;
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification)};
    descriptor.writes = {DimensionId(StandardDimension::Classification)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

SkewnessProgram compileSkewnessStage(const Json& node,
                                     StageDescriptor& descriptor)
{
    SkewnessProgram program;
    program.groundClass = node.contains("ground_class")
                              ? node.at("ground_class").get<std::uint8_t>()
                              : std::uint8_t(2U);
    program.otherClass = node.contains("other_class")
                             ? node.at("other_class").get<std::uint8_t>()
                             : std::uint8_t(1U);
    program.onlyGround = node.contains("only_ground")
                             ? node.at("only_ground").get<bool>()
                             : false;
    descriptor.reads = {DimensionId(StandardDimension::Z),
                        DimensionId(StandardDimension::Classification)};
    descriptor.writes = {
        DimensionId(StandardDimension::Classification)};
    descriptor.preservesOrder = false;
    declarePureStage(descriptor, true);
    return program;
}

MortonProgram compileMortonStage(const Json& node, StageDescriptor& descriptor)
{
    MortonProgram program;
    if (node.contains("reverse"))
        program.reverse = node.at("reverse").get<bool>();
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y)};
    descriptor.preservesOrder = false;
    declarePureStage(descriptor, true);
    return program;
}

GroupByProgram compileGroupByStage(const Json& node,
                                   DimensionRegistry& dimensions,
                                   StageDescriptor& descriptor)
{
    const std::string name = requireString(node.at("dimension"), "dimension");
    const DimensionDefinition* definition = dimensions.find(name);
    if (!definition)
        throw PlanError("filters.groupby dimension does not exist: " + name);
    GroupByProgram program;
    program.dimension = definition->id;
    descriptor.reads.push_back(program.dimension);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

ReturnsProgram compileReturnsStage(const Json& node,
                                   StageDescriptor& descriptor)
{
    ReturnsProgram program;
    program.groups = 0U;
    std::vector<std::string> values;
    if (!node.contains("groups"))
        values.push_back("last");
    else if (node.at("groups").is_array())
    {
        for (const Json& value : node.at("groups"))
            values.push_back(requireString(value, "groups"));
    }
    else
        values.push_back(requireString(node.at("groups"), "groups"));

    for (const std::string& value : values)
    {
        std::size_t begin = 0;
        while (begin <= value.size())
        {
            const std::size_t comma = value.find(',', begin);
            const std::string group = trim(value.substr(
                begin, comma == std::string::npos ? std::string::npos
                                                  : comma - begin));
            if (group == "first")
                program.groups |= ReturnFirst;
            else if (group == "intermediate")
                program.groups |= ReturnIntermediate;
            else if (group == "last")
                program.groups |= ReturnLast;
            else if (group == "only")
                program.groups |= ReturnOnly;
            else
                throw PlanError("filters.returns invalid group: " + group);
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
    }
    descriptor.reads = {DimensionId(StandardDimension::ReturnNumber),
                        DimensionId(StandardDimension::NumberOfReturns)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

DividerProgram compileDividerStage(const Json& node,
                                   StageDescriptor& descriptor)
{
    DividerProgram program;
    const std::string mode =
        lowercase(node.contains("mode") ? requireString(node.at("mode"), "mode")
                                        : std::string("partition"));
    if (mode == "partition")
        program.mode = DividerMode::Partition;
    else if (mode == "round_robin")
        program.mode = DividerMode::RoundRobin;
    else
        throw PlanError("filters.divider unsupported mode: " + mode);

    const std::uint64_t count = unsignedOption(node, "count", 0U);
    if (count < 2U || count > 1000U)
        throw PlanError("filters.divider count must be in [2, 1000]");
    program.count = static_cast<std::uint32_t>(count);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

SplitterProgram compileSplitterStage(const Json& node,
                                     StageDescriptor& descriptor)
{
    SplitterProgram program;
    if (node.contains("length"))
        program.length = node.at("length").get<double>();
    if (node.contains("origin_x"))
        program.originX = node.at("origin_x").get<double>();
    if (node.contains("origin_y"))
        program.originY = node.at("origin_y").get<double>();
    if (node.contains("buffer"))
        program.buffer = node.at("buffer").get<double>();
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, false);
    return program;
}

ColorinterpProgram compileColorinterpStage(const Json& node,
                                           DimensionRegistry& dimensions,
                                           StageDescriptor& descriptor)
{
    ColorinterpProgram program;
    const std::string dimensionName =
        node.contains("dimension")
            ? requireString(node.at("dimension"), "dimension")
            : std::string("Z");
    const DimensionDefinition* definition = dimensions.find(dimensionName);
    if (!definition)
        throw PlanError("filters.colorinterp dimension does not exist: " +
                        dimensionName);
    program.dimension = definition->id;
    if (node.contains("minimum"))
        program.minimum = node.at("minimum").get<double>();
    if (node.contains("maximum"))
        program.maximum = node.at("maximum").get<double>();
    if (node.contains("clamp"))
        program.clamp = node.at("clamp").get<bool>();
    if (node.contains("ramp"))
        program.ramp = requireString(node.at("ramp"), "ramp");
    if (node.contains("invert"))
        program.invert = node.at("invert").get<bool>();
    if (node.contains("mad"))
        program.mad = node.at("mad").get<bool>();
    if (node.contains("mad_multiplier"))
        program.madMultiplier = node.at("mad_multiplier").get<double>();
    if (node.contains("k"))
        program.k = node.at("k").get<double>();

    const DimensionId red(StandardDimension::Red);
    const DimensionId green(StandardDimension::Green);
    const DimensionId blue(StandardDimension::Blue);
    descriptor.reads = {program.dimension, red, green, blue};
    descriptor.writes = {red, green, blue};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

std::vector<std::string> statsNames(const Json& node, std::string_view option)
{
    const auto position = node.find(std::string(option));
    if (position == node.end())
        return {};
    std::vector<std::string> names;
    const auto append = [&](const Json& value)
    {
        const std::string text = requireString(value, option);
        std::size_t begin = 0;
        while (begin <= text.size())
        {
            const std::size_t comma = text.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? text.size() : comma;
            const std::string name =
                trim(std::string_view(text).substr(begin, end - begin));
            if (!name.empty())
                names.push_back(name);
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
    };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return names;
}

StatsProgram compileStatsStage(const Json& node, DimensionRegistry& dimensions,
                               StageDescriptor& descriptor)
{
    StatsProgram program;
    program.advanced = booleanOption(node, "advanced", false);
    if (node.contains("commonsrs"))
        program.commonSrs = requireString(node.at("commonsrs"), "commonsrs");

    std::map<DimensionId, SummaryMode> selected;
    for (const std::string& name : statsNames(node, "dimensions"))
        if (const DimensionDefinition* definition = dimensions.find(name))
            selected[definition->id] = SummaryMode::None;

    const auto setModes = [&](std::string_view option, SummaryMode mode)
    {
        for (const std::string& name : statsNames(node, option))
            if (const DimensionDefinition* definition = dimensions.find(name);
                definition && selected.contains(definition->id))
                selected[definition->id] = mode;
    };
    setModes("enumerate", SummaryMode::Enumerate);
    setModes("count", SummaryMode::Count);
    setModes("global", SummaryMode::Global);

    for (const auto& [dimension, mode] : selected)
    {
        program.dimensions.push_back(dimension);
        program.modes.push_back(mode);
        descriptor.reads.push_back(dimension);
    }
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

InfoProgram compileInfoStage(const Json& node, StageDescriptor& descriptor)
{
    InfoProgram program;
    if (node.contains("point"))
        program.pointSpec = requireString(node.at("point"), "point");
    if (node.contains("query"))
        program.querySpec = requireString(node.at("query"), "query");
    descriptor.reads = {DimensionId(StandardDimension::X),
                        DimensionId(StandardDimension::Y),
                        DimensionId(StandardDimension::Z)};
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

ExpressionStatsProgram
compileExpressionStatsStage(const Json& node, DimensionRegistry& dimensions,
                            StageDescriptor& descriptor)
{
    const std::string dimensionName =
        requireString(node.at("dimension"), "dimension");
    const DimensionDefinition* definition = dimensions.find(dimensionName);
    if (!definition)
        throw PlanError("filters.expressionstats dimension does not exist: " +
                        dimensionName);

    ExpressionStatsProgram program;
    program.dimension = definition->id;
    const Json& configured = node.at("expressions");
    const auto append = [&](const Json& value)
    {
        CompiledExpression expression;
        try
        {
            expression = compileConditionalExpression(
                requireString(value, "expressions"), dimensions);
        }
        catch (const std::exception& error)
        {
            throw PlanError(
                std::string("invalid filters.expressionstats expression: ") +
                error.what());
        }
        for (DimensionId read : expression.reads)
            appendUnique(descriptor.reads, read);
        program.expressions.push_back(std::move(expression));
    };
    if (configured.is_array())
        for (const Json& expression : configured)
            append(expression);
    else
        append(configured);
    appendUnique(descriptor.reads, program.dimension);
    descriptor.preservesOrder = true;
    declarePureStage(descriptor, true);
    return program;
}

std::vector<std::size_t>
specifiedInputs(const Json& node,
                const std::unordered_map<std::string, std::size_t>& tags)
{
    const auto position = node.find("inputs");
    if (position == node.end() || position->is_null())
        return {};
    std::vector<std::size_t> inputs;
    const auto append = [&](const Json& value)
    {
        const std::string tag = requireString(value, "inputs");
        const auto found = tags.find(tag);
        if (found == tags.end())
            throw PlanError("pipeline input references an undefined tag: " +
                            tag);
        inputs.push_back(found->second);
    };
    if (position->is_array())
        for (const Json& value : *position)
            append(value);
    else
        append(*position);
    return inputs;
}

std::string stageTag(const Json& node)
{
    const auto position = node.find("tag");
    if (position == node.end() || position->is_null())
        return {};
    const std::string tag = requireString(*position, "tag");
    if (!validTag(tag))
        throw PlanError("invalid pipeline stage tag: " + tag);
    return tag;
}

std::size_t physicalBytes(const DimensionDefinition& definition)
{
    // Resident stage kernels consume the logical PDAL coordinate values as
    // doubles. Packed LAS integers are a reader/writer boundary resource and
    // are costed separately by placement, not as resident XYZ columns.
    return dimensionTypeSize(definition.type);
}

std::size_t columnBytesPerPoint(const std::vector<DimensionId>& columns,
                                const DimensionRegistry& dimensions)
{
    std::size_t bytes = 0U;
    for (DimensionId id : columns)
    {
        const std::size_t addition = physicalBytes(dimensions.require(id));
        if (bytes > (std::numeric_limits<std::size_t>::max)() - addition)
            throw std::overflow_error(
                "resident column byte estimate overflows");
        bytes += addition;
    }
    return bytes;
}

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* message)
{
    if (left > (std::numeric_limits<std::size_t>::max)() - right)
        throw std::overflow_error(message);
    return left + right;
}

void finishPlan(std::vector<PlannedStage>& stages,
                DimensionRegistry& dimensions, PlanSummary& summary,
                PlannerOptions options)
{
    constexpr std::size_t UniformGridPersistentBytesPerPoint =
        2U * sizeof(std::uint64_t) + 3U * sizeof(std::uint32_t);
    // The implicit BVH has fewer than two internal nodes per capacity point;
    // each stores an outward-rounded min/max float bound for three axes.
    constexpr std::size_t MortonBvhBoundsBytesPerPoint =
        2U * 6U * sizeof(float);
    // The kNN gather kernels additionally keep the coordinates gathered in
    // sorted order: three doubles for the exact contiguous candidate loads
    // and three origin-shifted floats for the certified distance prefilter.
    constexpr std::size_t SortedCoordinateBytesPerPoint =
        3U * sizeof(double) + 3U * sizeof(float);
    summary.deterministic = options.deterministic;
    std::vector<DimensionId> filterWrites;
    for (const PlannedStage& stage : stages)
    {
        appendUnique(summary.touchedDimensions, stage.descriptor.reads);
        appendUnique(summary.touchedDimensions, stage.descriptor.writes);
        if (stage.role == StageRole::Filter)
            appendUnique(filterWrites, stage.descriptor.writes);
        summary.maximumRadius =
            std::max(summary.maximumRadius, stage.descriptor.maximumRadius);
        if (stage.descriptor.index.kind != IndexKind::None &&
            stage.preferredResidency == MemoryKind::Device)
        {
            const std::size_t bytes =
                stage.descriptor.index.kind == IndexKind::Knn
                    ? UniformGridPersistentBytesPerPoint +
                          MortonBvhBoundsBytesPerPoint +
                          SortedCoordinateBytesPerPoint
                    : UniformGridPersistentBytesPerPoint;
            summary.indexBytesPerPoint =
                (std::max)(summary.indexBytesPerPoint, bytes);
        }
    }

    for (PlannedStage& stage : stages)
    {
        const GridRequest& grid = stage.descriptor.grid;
        if (stage.preferredResidency != MemoryKind::Device ||
            grid.framePolicy == GridFramePolicy::None)
            continue;
        if (!grid.deviceBytesPerCell || !grid.deviceBackingCount ||
            !grid.deviceProofBytesPerCell || !grid.hostBytesPerPoint ||
            !grid.hostBytesPerCell || !grid.hostTileBytesPerExpandedCell ||
            !grid.maximumHaloCells || !grid.phaseSynchronized)
            throw PlanError("native grid request has an incomplete product "
                            "contract");
        ++summary.gridBuilds;
        stage.deviceGridBuildBytesPerCell = grid.deviceBytesPerCell;
        stage.deviceGridProofBytesPerCell = grid.deviceProofBytesPerCell;
        stage.deviceGridFixedBytes = grid.deviceFixedBytes;
        summary.peakDeviceGridBytesPerCell =
            (std::max)(summary.peakDeviceGridBytesPerCell,
                       grid.deviceBytesPerCell);
        summary.peakDeviceGridProofBytesPerCell =
            (std::max)(summary.peakDeviceGridProofBytesPerCell,
                       grid.deviceProofBytesPerCell);
        summary.peakDeviceGridFixedBytes =
            (std::max)(summary.peakDeviceGridFixedBytes,
                       stage.deviceGridFixedBytes);
    }

    for (PlannedStage& stage : stages)
    {
        if (stage.role == StageRole::Reader)
            stage.descriptor.writes = summary.touchedDimensions;
        else if (stage.role == StageRole::Writer)
        {
            // The native LAS writer preserves untouched source-record fields
            // without materializing them as SoA columns. Standard dimensions
            // written by filters remain observable; temporary custom columns
            // are not writer inputs unless an option-rich writer delegates to
            // PDAL, in which case conservatively retain every touched column.
            stage.descriptor.reads.clear();
            const std::vector<DimensionId>& observable =
                stage.native ? filterWrites : summary.touchedDimensions;
            for (DimensionId id : observable)
            {
                const DimensionDefinition& definition = dimensions.require(id);
                if (!stage.native || definition.standard)
                    appendUnique(stage.descriptor.reads, id);
            }
        }
        else if (!stage.native)
        {
            // Pipeline JSON cannot describe the dimensions observed or
            // changed by an unknown PDAL stage. Preserve correctness for the
            // known resident set and mark the boundary as a full-record pack
            // below; the runtime layout supplies any additional fields.
            stage.descriptor.reads = summary.touchedDimensions;
            stage.descriptor.writes = summary.touchedDimensions;
        }
        stage.descriptor.fusion.dimsRead = stage.descriptor.reads;
        stage.descriptor.fusion.dimsWritten = stage.descriptor.writes;
    }

    std::vector<std::vector<std::size_t>> consumers(stages.size());
    for (const PlannedStage& stage : stages)
        for (std::size_t input : stage.inputs)
            consumers.at(input).push_back(stage.id);

    for (std::size_t reverse = stages.size(); reverse-- > 0;)
    {
        PlannedStage& stage = stages[reverse];
        for (std::size_t consumer : consumers[stage.id])
            appendUnique(stage.liveAfter, stages[consumer].liveBefore);
        stage.liveBefore = stage.liveAfter;
        eraseDimensions(stage.liveBefore, stage.descriptor.writes);
        appendUnique(stage.liveBefore, stage.descriptor.reads);
    }

    // Build maximal device regions. Device siblings of one host product share
    // the same uploaded resident batch, which is required for shared-index
    // reuse. A host stage between device stages always separates regions.
    std::vector<std::size_t> parent(stages.size());
    for (std::size_t id = 0; id < parent.size(); ++id)
        parent[id] = id;
    const auto findRoot = [&](std::size_t value)
    {
        std::size_t root = value;
        while (parent[root] != root)
            root = parent[root];
        while (parent[value] != value)
        {
            const std::size_t next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    };
    const auto residentProductsCompose =
        [](const PlannedStage& left, const PlannedStage& right)
    {
        // Grid regions own a RasterGridProduct and are emitted only as one
        // standalone grid stage or a proved compatible PMF chain. They cannot
        // absorb a neighborhood/point region merely because both stages prefer
        // device residency: that would keep the raster product alive across an
        // unsupported cross-kind bridge and erase its required spill boundary.
        const bool leftGrid = left.descriptor.kind == StageKind::Grid;
        const bool rightGrid = right.descriptor.kind == StageKind::Grid;
        return leftGrid == rightGrid;
    };
    const auto unite = [&](std::size_t left, std::size_t right)
    {
        const std::size_t leftRoot = findRoot(left);
        const std::size_t rightRoot = findRoot(right);
        if (leftRoot != rightRoot)
            parent[rightRoot] = leftRoot;
    };
    for (const PlannedStage& stage : stages)
        if (stage.preferredResidency == MemoryKind::Device)
            for (std::size_t input : stage.inputs)
                if (stages[input].preferredResidency == MemoryKind::Device &&
                    residentProductsCompose(stage, stages[input]))
                    unite(stage.id, input);
    for (const PlannedStage& producer : stages)
    {
        if (producer.preferredResidency == MemoryKind::Device)
            continue;
        std::size_t firstDeviceConsumer = NoResidentRegion;
        for (std::size_t consumer : consumers[producer.id])
        {
            if (stages[consumer].preferredResidency != MemoryKind::Device)
                continue;
            if (firstDeviceConsumer == NoResidentRegion)
                firstDeviceConsumer = consumer;
            else if (residentProductsCompose(stages[firstDeviceConsumer],
                                             stages[consumer]))
                unite(firstDeviceConsumer, consumer);
        }
    }
    std::unordered_map<std::size_t, std::size_t> regions;
    for (PlannedStage& stage : stages)
    {
        if (stage.preferredResidency != MemoryKind::Device)
            continue;
        const std::size_t root = findRoot(stage.id);
        const auto [position, inserted] = regions.emplace(root, regions.size());
        (void)inserted;
        stage.residentRegion = position->second;
    }
    summary.residentRegions = regions.size();

    // Track each shared index allocation to its final actual neighborhood
    // consumer. Coordinate/order/cardinality changes and host boundaries end
    // propagation. Intermediate compatible stages retain the allocation only
    // when a later request reuses it.
    struct IndexLifetime
    {
        std::size_t region = NoResidentRegion;
        std::size_t firstStage = 0;
        std::size_t lastStage = 0;
        std::size_t bytesPerPoint = 0;
        IndexKind kind = IndexKind::None;
        std::uint8_t dimensions = 0U;
    };
    std::vector<IndexLifetime> indexLifetimes;
    std::vector<std::size_t> outputIndex(stages.size(), NoResidentRegion);
    for (PlannedStage& stage : stages)
    {
        std::size_t available = NoResidentRegion;
        if (stage.preferredResidency == MemoryKind::Device &&
            stage.inputs.size() == 1U)
        {
            const std::size_t candidate = outputIndex.at(stage.inputs.front());
            if (candidate != NoResidentRegion &&
                indexLifetimes[candidate].region == stage.residentRegion &&
                indexLifetimes[candidate].kind == stage.descriptor.index.kind &&
                indexLifetimes[candidate].dimensions ==
                    stage.descriptor.index.dimensions)
                available = candidate;
        }
        const std::size_t requested =
            stage.preferredResidency != MemoryKind::Device ? 0U
            : stage.descriptor.index.kind == IndexKind::Knn
                ? UniformGridPersistentBytesPerPoint +
                      MortonBvhBoundsBytesPerPoint +
                      SortedCoordinateBytesPerPoint
            : stage.descriptor.index.kind == IndexKind::Radius
                ? UniformGridPersistentBytesPerPoint
                : 0U;
        if (requested && (available == NoResidentRegion ||
                          indexLifetimes[available].bytesPerPoint < requested))
        {
            ++summary.indexBuilds;
            available = indexLifetimes.size();
            indexLifetimes.push_back({stage.residentRegion, stage.id, stage.id,
                                      requested, stage.descriptor.index.kind,
                                      stage.descriptor.index.dimensions});
            stage.deviceIndexBuildBytesPerPoint = requested;
            if (stage.inputs.size() == 1U)
                outputIndex.at(stage.inputs.front()) = available;
        }
        if (requested && available != NoResidentRegion)
            indexLifetimes[available].lastStage = stage.id;

        const bool invalidates =
            stage.role == StageRole::Reader ||
            stage.preferredResidency != MemoryKind::Device ||
            stage.descriptor.mutatesCoordinates ||
            !stage.descriptor.fusion.cardinalityPreserving ||
            !stage.descriptor.preservesOrder;
        outputIndex.at(stage.id) = invalidates ? NoResidentRegion : available;
    }

    // A linear statistical-outlier -> NNDistance pair can project both exact
    // distance-valued results from one ordered max-k rowset. Keep the contract
    // explicit in the plan: index compatibility alone does not authorize a
    // retained query product, and branches or intervening stages decline it.
    for (std::size_t index = 0U; index + 1U < stages.size(); ++index)
    {
        PlannedStage& outlierStage = stages[index];
        PlannedStage& nnDistanceStage = stages[index + 1U];
        const auto* outlier =
            std::get_if<OutlierProgram>(&outlierStage.payload);
        const auto* nnDistance =
            std::get_if<NnDistanceProgram>(&nnDistanceStage.payload);
        if (!outlier || !nnDistance ||
            outlier->method != OutlierMethod::Statistical ||
            outlierStage.preferredResidency != MemoryKind::Device ||
            nnDistanceStage.preferredResidency != MemoryKind::Device ||
            outlierStage.residentRegion == NoResidentRegion ||
            outlierStage.residentRegion != nnDistanceStage.residentRegion ||
            outlierStage.inputs.size() != 1U ||
            nnDistanceStage.inputs.size() != 1U ||
            nnDistanceStage.inputs.front() != outlierStage.id ||
            consumers[outlierStage.id].size() != 1U ||
            outlierStage.descriptor.index.kind != IndexKind::Knn ||
            nnDistanceStage.descriptor.index.kind != IndexKind::Knn ||
            outlierStage.descriptor.index.dimensions !=
                nnDistanceStage.descriptor.index.dimensions ||
            outlierStage.descriptor.index.neighbors == 0U ||
            nnDistanceStage.descriptor.index.neighbors < 2U)
            continue;
        const std::uint32_t gatherNeighbors =
            (std::max)(outlierStage.descriptor.index.neighbors,
                       nnDistanceStage.descriptor.index.neighbors);
        if (gatherNeighbors > 64U)
            continue;
        const std::size_t queryBytesPerPoint =
            static_cast<std::size_t>(gatherNeighbors) *
                (sizeof(std::uint32_t) + sizeof(double)) +
            sizeof(std::uint8_t);
        outlierStage.deviceKnnGatherNeighbors = gatherNeighbors;
        nnDistanceStage.deviceKnnGatherNeighbors = gatherNeighbors;
        outlierStage.deviceQueryBytesPerPoint = queryBytesPerPoint;
        nnDistanceStage.deviceQueryBytesPerPoint = queryBytesPerPoint;
    }

    // Assemble maximal point-program chains from descriptor contracts. This
    // code deliberately knows no stage names: descriptors own both the
    // semantic legality of embedding a point operation and the implementation
    // capability of the producer/consumer kernel that would host it.
    const auto isPointProgram = [](const PlannedStage& stage)
    {
        const FusionSemantics& fusion = stage.descriptor.fusion;
        // A declared cardinality change may join a chain only when its
        // contract also declares purity, determinism, and stable order; the
        // folded chain semantics keep cardinalityPreserving=false so anchor
        // legality decides whether any consumer can host it.
        const bool declaredCardinalityContract =
            fusion.cardinalityPreserving ||
            (stage.descriptor.preservesOrder && !fusion.hasWhere &&
             fusion.whereMerge == WhereMergeMode::NotApplicable);
        return stage.native && stage.preferredResidency == MemoryKind::Device &&
               (stage.descriptor.kind == StageKind::Pointwise ||
                stage.descriptor.kind == StageKind::Split) &&
               fusion.pure && declaredCardinalityContract &&
               (fusion.fusableAsPrologue || fusion.fusableAsEpilogue);
    };
    std::vector<bool> pointStageVisited(stages.size(), false);
    for (const PlannedStage& first : stages)
    {
        if (!isPointProgram(first) || pointStageVisited[first.id])
            continue;
        if (first.inputs.size() == 1U)
        {
            const std::size_t predecessor = first.inputs.front();
            if (isPointProgram(stages[predecessor]) &&
                consumers[predecessor].size() == 1U)
                continue;
        }

        std::vector<std::size_t> chain;
        std::size_t current = first.id;
        while (isPointProgram(stages[current]) && !pointStageVisited[current])
        {
            pointStageVisited[current] = true;
            chain.push_back(current);
            if (consumers[current].size() != 1U)
                break;
            const std::size_t next = consumers[current].front();
            if (stages[next].inputs.size() != 1U ||
                stages[next].inputs.front() != current ||
                !isPointProgram(stages[next]))
                break;
            current = next;
        }

        FusionSemantics pointProgram;
        pointProgram.pure = true;
        pointProgram.cardinalityPreserving = true;
        pointProgram.fusableAsPrologue = true;
        pointProgram.fusableAsEpilogue = true;
        pointProgram.deterministicSafe = true;
        for (std::size_t stageId : chain)
        {
            const FusionSemantics& stageFusion =
                stages[stageId].descriptor.fusion;
            pointProgram.pure = pointProgram.pure && stageFusion.pure;
            pointProgram.cardinalityPreserving =
                pointProgram.cardinalityPreserving &&
                stageFusion.cardinalityPreserving;
            pointProgram.fusableAsPrologue =
                pointProgram.fusableAsPrologue && stageFusion.fusableAsPrologue;
            pointProgram.fusableAsEpilogue =
                pointProgram.fusableAsEpilogue && stageFusion.fusableAsEpilogue;
            pointProgram.deterministicSafe =
                pointProgram.deterministicSafe && stageFusion.deterministicSafe;
            pointProgram.hasWhere =
                pointProgram.hasWhere || stageFusion.hasWhere;
            if (stageFusion.whereMerge != WhereMergeMode::NotApplicable)
                pointProgram.whereMerge = stageFusion.whereMerge;
            appendUnique(pointProgram.dimsRead, stageFusion.dimsRead);
            appendUnique(pointProgram.dimsWritten, stageFusion.dimsWritten);
        }

        const auto addCandidate =
            [&](std::size_t anchor, FusionPlacement placement)
        {
            if (!pointFusionLegal(pointProgram,
                                  stages[anchor].descriptor.fusion, placement,
                                  options.deterministic))
                return;
            summary.fusionCandidates.push_back(
                {anchor, placement, chain, pointProgram.dimsRead,
                 pointProgram.dimsWritten,
                 pointProgram.deterministicSafe &&
                     stages[anchor].descriptor.fusion.deterministicSafe});
        };
        if (stages[chain.front()].inputs.size() == 1U)
            addCandidate(stages[chain.front()].inputs.front(),
                         FusionPlacement::ProducerEpilogue);
        if (consumers[chain.back()].size() == 1U)
            addCandidate(consumers[chain.back()].front(),
                         FusionPlacement::ConsumerPrologue);
    }

    // Derive region-wide column lifetimes. This is deliberately based on the
    // complete region rather than only an edge: it prevents an in-place branch
    // from retiring a source column before a later sibling consumes it.
    struct ColumnLifetime
    {
        std::size_t region = NoResidentRegion;
        DimensionId dimension;
        std::size_t firstStage = 0;
        std::size_t lastStage = 0;
        std::size_t bytesPerPoint = 0;
    };
    std::vector<ColumnLifetime> columnLifetimes;
    std::map<std::pair<std::size_t, std::uint32_t>, std::size_t>
        columnLifetimeByKey;
    for (PlannedStage& stage : stages)
    {
        if (stage.preferredResidency != MemoryKind::Device)
            continue;

        stage.deviceLiveBefore = stage.liveBefore;
        stage.deviceLiveAfter = stage.liveAfter;
        std::vector<DimensionId> working = stage.deviceLiveBefore;
        appendUnique(working, stage.descriptor.writes);
        appendUnique(working, stage.deviceLiveAfter);
        for (DimensionId id : working)
        {
            const auto key = std::pair(stage.residentRegion, id.value());
            const auto position = columnLifetimeByKey.find(key);
            if (position == columnLifetimeByKey.end())
            {
                const std::size_t lifetime = columnLifetimes.size();
                columnLifetimeByKey.emplace(key, lifetime);
                columnLifetimes.push_back(
                    {stage.residentRegion, id, stage.id, stage.id,
                     physicalBytes(dimensions.require(id))});
            }
            else
                columnLifetimes[position->second].lastStage = stage.id;
        }
    }

    for (const ColumnLifetime& lifetime : columnLifetimes)
    {
        appendUnique(stages[lifetime.firstStage].deviceMaterialize,
                     lifetime.dimension);
        appendUnique(stages[lifetime.lastStage].deviceRelease,
                     lifetime.dimension);
        for (std::size_t id = lifetime.firstStage; id <= lifetime.lastStage;
             ++id)
        {
            PlannedStage& stage = stages[id];
            if (stage.residentRegion != lifetime.region)
                continue;
            stage.deviceColumnBytesPerPoint = checkedAdd(
                stage.deviceColumnBytesPerPoint, lifetime.bytesPerPoint,
                "resident column byte estimate overflows");
        }
    }

    for (const IndexLifetime& lifetime : indexLifetimes)
    {
        stages[lifetime.lastStage].deviceIndexReleaseBytesPerPoint = checkedAdd(
            stages[lifetime.lastStage].deviceIndexReleaseBytesPerPoint,
            lifetime.bytesPerPoint, "resident index byte estimate overflows");
        for (std::size_t id = lifetime.firstStage; id <= lifetime.lastStage;
             ++id)
        {
            PlannedStage& stage = stages[id];
            if (stage.residentRegion != lifetime.region)
                continue;
            stage.deviceIndexBytesPerPoint = checkedAdd(
                stage.deviceIndexBytesPerPoint, lifetime.bytesPerPoint,
                "resident index byte estimate overflows");
        }
    }

    for (const PlannedStage& stage : stages)
    {
        summary.peakDeviceColumnBytesPerPoint =
            (std::max)(summary.peakDeviceColumnBytesPerPoint,
                       stage.deviceColumnBytesPerPoint);
        summary.peakDeviceQueryBytesPerPoint =
            (std::max)(summary.peakDeviceQueryBytesPerPoint,
                       stage.deviceQueryBytesPerPoint);
        const std::size_t stageBytes = checkedAdd(
            checkedAdd(stage.deviceColumnBytesPerPoint,
                       stage.deviceIndexBytesPerPoint,
                       "pipeline device-memory estimate overflows"),
            stage.deviceQueryBytesPerPoint,
            "pipeline device-memory estimate overflows");
        summary.peakDeviceBytesPerPoint =
            (std::max)(summary.peakDeviceBytesPerPoint, stageBytes);
    }

    std::map<std::pair<std::size_t, std::size_t>, std::size_t>
        uploadByProductAndRegion;
    std::unordered_map<std::size_t, std::size_t> spillByProduct;
    const auto addResidencyBoundary = [&](const PlannedStage& stage,
                                          std::size_t input,
                                          ResidencyBoundaryKind kind)
    {
        const PlannedStage& producer = stages[input];
        std::size_t boundaryIndex = NoResidentRegion;
        if (kind == ResidencyBoundaryKind::Upload)
        {
            const auto key = std::pair(input, stage.residentRegion);
            const auto position = uploadByProductAndRegion.find(key);
            if (position != uploadByProductAndRegion.end())
                boundaryIndex = position->second;
            else
            {
                boundaryIndex = summary.residencyBoundaries.size();
                uploadByProductAndRegion.emplace(key, boundaryIndex);
            }
        }
        else
        {
            const auto position = spillByProduct.find(input);
            if (position != spillByProduct.end())
                boundaryIndex = position->second;
            else
            {
                boundaryIndex = summary.residencyBoundaries.size();
                spillByProduct.emplace(input, boundaryIndex);
            }
        }

        if (boundaryIndex == summary.residencyBoundaries.size())
        {
            ResidencyBoundary boundary;
            boundary.producer = input;
            boundary.consumer = stage.id;
            boundary.consumers.push_back(stage.id);
            boundary.kind = kind;
            boundary.dimensions = kind == ResidencyBoundaryKind::Spill
                                      ? producer.liveAfter
                                      : stage.liveBefore;
            boundary.fallback = !producer.native || !stage.native;
            boundary.requiresFullPointRecord = boundary.fallback;
            summary.residencyBoundaries.push_back(std::move(boundary));
        }
        else
        {
            ResidencyBoundary& boundary =
                summary.residencyBoundaries.at(boundaryIndex);
            appendUnique(boundary.dimensions,
                         kind == ResidencyBoundaryKind::Spill
                             ? producer.liveAfter
                             : stage.liveBefore);
            boundary.consumers.push_back(stage.id);
            boundary.fallback =
                boundary.fallback || !producer.native || !stage.native;
            boundary.requiresFullPointRecord = boundary.fallback;
        }
    };
    for (const PlannedStage& stage : stages)
        for (std::size_t input : stage.inputs)
        {
            const PlannedStage& producer = stages[input];
            const bool changesDeviceRegion =
                producer.preferredResidency == MemoryKind::Device &&
                stage.preferredResidency == MemoryKind::Device &&
                producer.residentRegion != stage.residentRegion;
            if (changesDeviceRegion)
            {
                // Planner-separated Grid/non-Grid products cross a real host
                // boundary: publish the live columns from the old region, then
                // upload the next region's declared inputs. Point-batch
                // residency alone is not permission to retain the old Grid or
                // index product.
                addResidencyBoundary(stage, input,
                                     ResidencyBoundaryKind::Spill);
                addResidencyBoundary(stage, input,
                                     ResidencyBoundaryKind::Upload);
            }
            else if (producer.preferredResidency != stage.preferredResidency)
                addResidencyBoundary(stage, input,
                                     producer.preferredResidency ==
                                             MemoryKind::Device
                                         ? ResidencyBoundaryKind::Spill
                                         : ResidencyBoundaryKind::Upload);
        }

    for (ResidencyBoundary& boundary : summary.residencyBoundaries)
    {
        boundary.bytesPerPoint =
            columnBytesPerPoint(boundary.dimensions, dimensions);
        summary.hostDeviceTransferBytesPerPoint = checkedAdd(
            summary.hostDeviceTransferBytesPerPoint, boundary.bytesPerPoint,
            "pipeline transfer byte estimate overflows");
        ++summary.hostDeviceTransfers;
        if (boundary.kind == ResidencyBoundaryKind::Spill)
            ++summary.spillBoundaries;
        if (boundary.fallback)
            ++summary.fallbackBoundaries;
    }

    // A final output column cannot be returned to the allocator until its D2H
    // spill event is complete. Move those retirements from the producer-stage
    // hook to the last matching spill boundary.
    for (PlannedStage& stage : stages)
    {
        std::vector<DimensionId> afterStage;
        for (DimensionId dimension : stage.deviceRelease)
        {
            ResidencyBoundary* retirement = nullptr;
            for (ResidencyBoundary& boundary : summary.residencyBoundaries)
            {
                if (boundary.kind != ResidencyBoundaryKind::Spill ||
                    boundary.producer != stage.id ||
                    !contains(boundary.dimensions, dimension))
                    continue;
                if (!retirement || retirement->consumer < boundary.consumer)
                    retirement = &boundary;
            }
            if (retirement)
                appendUnique(retirement->releaseDimensions, dimension);
            else
                appendUnique(afterStage, dimension);
        }
        stage.deviceRelease = std::move(afterStage);
    }

    for (ResidencyBoundary& boundary : summary.residencyBoundaries)
    {
        if (boundary.kind != ResidencyBoundaryKind::Spill ||
            boundary.producer >= stages.size())
            continue;
        const std::size_t region = stages[boundary.producer].residentRegion;
        for (DimensionId dimension : boundary.releaseDimensions)
        {
            const bool written = std::any_of(
                stages.begin(), stages.end(),
                [&](const PlannedStage& stage)
                {
                    return stage.residentRegion == region &&
                           contains(stage.descriptor.writes, dimension);
                });
            if (!written)
                continue;
            appendUnique(boundary.repackDimensions, dimension);
            boundary.repackBytesPerPoint =
                checkedAdd(boundary.repackBytesPerPoint,
                           physicalBytes(dimensions.require(dimension)),
                           "resident spill repack byte estimate overflows");
        }
    }

    for (DimensionId id : summary.touchedDimensions)
        summary.bytesPerPoint = checkedAdd(
            summary.bytesPerPoint, physicalBytes(dimensions.require(id)),
            "pipeline touched-column byte estimate overflows");
}
} // unnamed namespace

bool pointFusionLegal(const FusionSemantics& pointProgram,
                      const FusionSemantics& anchor, FusionPlacement placement,
                      bool deterministic) noexcept
{
    if (!pointProgram.pure || pointProgram.hasWhere ||
        pointProgram.whereMerge != WhereMergeMode::NotApplicable)
        return false;
    // A cardinality-changing chain is legal only as the consumer prologue of
    // an anchor that declares stable compaction support.
    if (!pointProgram.cardinalityPreserving &&
        (placement != FusionPlacement::ConsumerPrologue ||
         !anchor.acceptsCompactingPrologue))
        return false;
    if (deterministic &&
        (!pointProgram.deterministicSafe || !anchor.deterministicSafe))
        return false;

    if (placement == FusionPlacement::ProducerEpilogue)
        return pointProgram.fusableAsEpilogue && anchor.acceptsFusedEpilogue;
    if (!pointProgram.fusableAsPrologue || !anchor.acceptsFusedPrologue)
        return false;
    if (anchor.prologueConsumesPointWrites)
        return true;
    return std::none_of(pointProgram.dimsWritten.begin(),
                        pointProgram.dimsWritten.end(),
                        [&](DimensionId dimension)
                        { return contains(anchor.dimsRead, dimension); });
}

Plan::Plan(std::vector<PlannedStage> stages, PlanSummary summary)
    : m_stages(std::move(stages)), m_summary(std::move(summary))
{
}

const std::vector<PlannedStage>& Plan::stages() const noexcept
{
    return m_stages;
}

const PlanSummary& Plan::summary() const noexcept
{
    return m_summary;
}

std::size_t Plan::estimatedDeviceBytes(std::size_t pointCapacity) const
{
    const std::size_t bytesPerPoint = m_summary.peakDeviceBytesPerPoint;
    if (bytesPerPoint &&
        pointCapacity > std::numeric_limits<std::size_t>::max() / bytesPerPoint)
        throw std::overflow_error("pipeline device-memory estimate overflows");
    const std::size_t pointBytes = pointCapacity * bytesPerPoint;
    if (m_summary.peakDeviceGridFixedBytes >
        (std::numeric_limits<std::size_t>::max)() - pointBytes)
        throw std::overflow_error("pipeline device-memory estimate overflows");
    return pointBytes + m_summary.peakDeviceGridFixedBytes;
}

void preparePlannedDeviceColumns(PointBatch& batch, const PlannedStage& stage)
{
    if (stage.residentRegion == NoResidentRegion)
        throw std::invalid_argument(
            "cannot prepare device columns for a host-resident stage");
    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    for (DimensionId dimension : stage.deviceMaterialize)
        if (!batch.has(dimension))
        {
            if (dimension == x || dimension == y || dimension == z)
                batch.materialize(dimension, DimensionType::Double);
            else
                batch.materialize(dimension);
        }
}

void releasePlannedDeviceColumns(PointBatch& batch, const PlannedStage& stage)
{
    if (stage.residentRegion == NoResidentRegion)
        throw std::invalid_argument(
            "cannot release device columns for a host-resident stage");
    for (DimensionId dimension : stage.deviceRelease)
        if (batch.has(dimension))
            batch.erase(dimension);
}

void releaseSpilledDeviceColumns(PointBatch& batch,
                                 const ResidencyBoundary& boundary)
{
    if (boundary.kind != ResidencyBoundaryKind::Spill)
        throw std::invalid_argument(
            "device columns can be retired only after a spill boundary");
    for (DimensionId dimension : boundary.releaseDimensions)
        if (batch.has(dimension))
            batch.erase(dimension);
}

Plan compilePipeline(std::string_view text, DimensionRegistry& dimensions,
                     PlannerOptions options)
{
    Json root;
    try
    {
        root = Json::parse(text, nullptr, true, true);
    }
    catch (const Json::parse_error& error)
    {
        throw PlanError(std::string("invalid pipeline JSON: ") + error.what());
    }

    const Json* pipeline = nullptr;
    if (root.is_array())
        pipeline = &root;
    else if (root.is_object())
    {
        const auto position = root.find("pipeline");
        if (position != root.end())
            pipeline = &*position;
    }
    if (!pipeline || !pipeline->is_array())
        throw PlanError("pipeline root must be an array or contain a pipeline "
                        "array");
    if (pipeline->empty())
        throw PlanError("pipeline must contain at least one stage");

    std::vector<PlannedStage> stages;
    std::vector<std::size_t> frontier;
    std::unordered_map<std::string, std::size_t> tags;
    PlanSummary summary;
    summary.allStagesNative = true;
    const std::size_t last = pipeline->size() - 1;

    for (std::size_t index = 0; index < pipeline->size(); ++index)
    {
        const Json& sourceNode = pipeline->at(index);
        Json node;
        std::string filename;
        std::string type;
        if (sourceNode.is_string())
            filename = sourceNode.get<std::string>();
        else if (sourceNode.is_object())
        {
            node = sourceNode;
            if (const auto position = node.find("type");
                position != node.end() && !position->is_null())
                type = requireString(*position, "type");
            if (const auto position = node.find("filename");
                position != node.end() && !position->is_null())
                filename = requireString(*position, "filename");
        }
        else
            throw PlanError("pipeline stages must be strings or objects");

        PlannedStage stage;
        stage.id = stages.size();
        stage.tag = sourceNode.is_object() ? stageTag(node) : std::string{};
        if (!stage.tag.empty() && tags.contains(stage.tag))
            throw PlanError("duplicate pipeline stage tag: " + stage.tag);
        stage.role = roleFor(type, index, last);
        if (type.empty())
            type = inferredLasType(filename, stage.role);
        stage.descriptor.type = type;

        const std::vector<std::size_t> explicitInputs =
            sourceNode.is_object() ? specifiedInputs(node, tags)
                                   : std::vector<std::size_t>{};
        if (stage.role == StageRole::Reader)
        {
            if (!explicitInputs.empty())
                throw PlanError("reader stages cannot specify inputs");
        }
        else
        {
            stage.inputs = explicitInputs.empty() ? frontier : explicitInputs;
            if (stage.inputs.empty())
                throw PlanError("non-reader stage has no input");
        }

        if (stage.role == StageRole::Reader || stage.role == StageRole::Writer)
        {
            stage.descriptor.kind = StageKind::Cpu;
            stage.preferredResidency = MemoryKind::Host;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "filename", "tag", "inputs"});
            // B0188/D0217: a `.laz` *reader* is native too. Nativeness here
            // means the engine can obtain the stage's records exactly, not
            // that it maps them: the direct source, which really does
            // memory-map raw records, is separately and independently gated on
            // `!compressedReader` in ResidentPipeline, so it stays off for
            // compressed input however this route is reached. The ordinary
            // resident path takes its points from the reader itself.
            //
            // The cost model is unaffected because decode is not a new cost.
            // B0188 measured pinned `laz -> las` translate at 0.309 s against
            // `las -> las` at 0.326 s: the smaller file offsets the
            // decompression, so admitting `.laz` input does not spend
            // unmeasured time in a model fitted on uncompressed input (D0216).
            //
            // Ordinary writers are deliberately not included. A `.laz` sink
            // must encode, which B0188 measured at +0.112 s. The literal
            // `extra_dims=all` exception below is functional admission only;
            // D0223 separately restricts automatic placement to B0224's one
            // measured compressed publication row.
            const bool nativeLasFilename =
                stage.role == StageRole::Reader
                    ? isLasFamilyFilename(filename)
                    : isUncompressedLasFilename(filename);
            const bool lazWriterFilename =
                stage.role == StageRole::Writer && !filename.empty() &&
                lowercase(std::filesystem::path(filename).extension().string())
                        == ".laz";
            const bool compressionTrue =
                node.contains("compression") &&
                ((node.at("compression").is_boolean() &&
                  node.at("compression").get<bool>()) ||
                 (node.at("compression").is_string() &&
                  lowercase(node.at("compression").get<std::string>()) ==
                      "true"));
            const bool extraDimensionsAll =
                stage.role == StageRole::Writer &&
                !filename.empty() &&
                ((isUncompressedLasFilename(filename) &&
                  hasOnlyOptions(node, {"type", "filename", "tag", "inputs",
                                        "extra_dims"})) ||
                 (lazWriterFilename &&
                  ((!node.contains("compression") &&
                    hasOnlyOptions(node, {"type", "filename", "tag", "inputs",
                                          "extra_dims"})) ||
                   (compressionTrue &&
                    hasOnlyOptions(
                        node, {"type", "filename", "tag", "inputs",
                               "compression", "extra_dims"}))))) &&
                node.contains("extra_dims") &&
                node.at("extra_dims").is_string() &&
                node.at("extra_dims").get<std::string>() == "all";
            // `extra_dims=all` preserves every standard filter output and the
            // untouched source record.  Its physical layout is derived from
            // the prepared writer (D0218), so it is a native sink even when a
            // reorder-only producer contributes no SoA column: that producer
            // publishes its separately declared non-column result instead.
            stage.native = (type == "readers.las" || type == "writers.las") &&
                           !filename.empty() &&
                           ((nativeLasFilename && supportedOptions) ||
                            extraDimensionsAll);

            // B0153/D0214: `compression:true` on a `.laz` sink is admitted as
            // well. It is redundant with the extension -- writing the same
            // points to `out.laz` with and without it produces byte-identical
            // files (B0152) -- so it cannot change the record layout placement
            // budgets. Any other combination, including `compression:false` or
            // compression on a `.las` name, stays unadmitted because there the
            // option genuinely decides the encoding.
            const bool redundantLazCompression =
                lazWriterFilename && compressionTrue &&
                hasOnlyOptions(node, {"type", "filename", "tag", "inputs",
                                      "compression"});
            const bool optionFreeLasFamily =
                (type == "readers.las" || type == "writers.las") &&
                !filename.empty() && isLasFamilyFilename(filename) &&
                (supportedOptions || redundantLazCompression);
            stage.payload =
                FileStagePlan{filename, extraDimensionsAll, optionFreeLasFamily};
            if (stage.native)
            {
                if (stage.role == StageRole::Reader)
                    declareFusionAnchor(stage.descriptor, false, true);
                else
                    // The ordered LAS sink performs declared stable
                    // compaction and final-size truncation inside the
                    // writer-side pack/summarize machinery.
                    declareFusionAnchor(stage.descriptor, true, false, true,
                                        true);
            }
        }
        else if (type == "filters.ferry")
        {
            stage.descriptor.kind = StageKind::Pointwise;
            stage.preferredResidency = MemoryKind::Device;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimensions"});
            if (supportedOptions)
            {
                stage.payload =
                    compileFerry(node, dimensions, stage.descriptor);
                if (ferrySupportsExactPointProgram(
                        std::get<FerryProgram>(stage.payload), dimensions))
                    stage.descriptor.placementModel = "point-program";
                stage.native = true;
            }
            else
                stage.payload = FallbackStagePlan{sourceNode.dump()};
        }
        else if (type == "filters.assign")
        {
            stage.descriptor.kind = StageKind::Pointwise;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "value"});
            if (supportedOptions)
            {
                stage.payload =
                    compileAssign(node, dimensions, stage.descriptor);
                stage.preferredResidency =
                    assignSupportsExactDevice(
                        std::get<AssignProgram>(stage.payload))
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.descriptor.placementModel = "point-program";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.expression")
        {
            stage.descriptor.kind = StageKind::Split;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "expression"}) &&
                node.contains("expression") &&
                node.at("expression").is_string();
            if (supportedOptions)
            {
                stage.payload =
                    compileExpression(node, dimensions, stage.descriptor);
                stage.preferredResidency =
                    predicateSupportsExactDevice(
                        std::get<PredicateProgram>(stage.payload))
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.descriptor.placementModel = "point-program";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.range")
        {
            stage.descriptor.kind = StageKind::Split;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "limits"}) &&
                node.contains("limits") &&
                (node.at("limits").is_string() ||
                 (node.at("limits").is_array() &&
                  std::all_of(node.at("limits").begin(),
                              node.at("limits").end(), [](const Json& value)
                              { return value.is_string(); })));
            if (supportedOptions)
            {
                stage.payload =
                    compileRange(node, dimensions, stage.descriptor);
                // B0162: coordinate loads are admitted here and re-checked
                // against a real batch by the preflight and the device
                // evaluator, which both fail closed.
                stage.preferredResidency =
                    predicateMaySupportExactDevice(
                        std::get<PredicateProgram>(stage.payload))
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.crop")
        {
            stage.descriptor.kind = StageKind::Split;
            const bool supportedOptions =
                hasOnlyOptions(
                    node, {"type", "tag", "inputs", "bounds", "outside"}) &&
                node.contains("bounds") && node.at("bounds").is_string() &&
                (!node.contains("outside") || node.at("outside").is_boolean());
            if (supportedOptions)
            {
                stage.payload = compileCrop(node, dimensions, stage.descriptor);
                // B0162: as for `filters.range` above; `filters.crop` is
                // inherently a coordinate predicate and was unplaceable.
                stage.preferredResidency =
                    predicateMaySupportExactDevice(
                        std::get<PredicateProgram>(stage.payload))
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.decimation" || type == "filters.head" ||
                 type == "filters.tail")
        {
            stage.descriptor.kind = StageKind::Split;
            const bool decimation = type == "filters.decimation";
            const bool supportedOptions =
                decimation
                    ? hasOnlyOptions(node, {"type", "tag", "inputs", "step",
                                            "offset", "limit"}) &&
                          (!node.contains("step") ||
                           isExactDecimationStep(node.at("step"))) &&
                          (!node.contains("offset") ||
                           isUnsignedInteger(node.at("offset"))) &&
                          (!node.contains("limit") ||
                           isUnsignedInteger(node.at("limit")))
                    : hasOnlyOptions(
                          node, {"type", "tag", "inputs", "count", "invert"}) &&
                          (!node.contains("count") ||
                           isUnsignedInteger(node.at("count"))) &&
                          (!node.contains("invert") ||
                           node.at("invert").is_boolean());
            if (supportedOptions)
            {
                stage.payload = compileOrdinal(node, type);
                declarePureStage(stage.descriptor, false);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.locate")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool supportedOptions =
                hasOnlyOptions(
                    node, {"type", "tag", "inputs", "dimension", "minmax"}) &&
                node.contains("dimension") &&
                node.at("dimension").is_string() &&
                (!node.contains("minmax") || node.at("minmax").is_string());
            if (supportedOptions)
            {
                stage.payload =
                    compileLocate(node, dimensions, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.transformation")
        {
            stage.descriptor.kind = StageKind::Pointwise;
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "matrix", "invert"}) &&
                node.contains("matrix") && node.at("matrix").is_string() &&
                (!node.contains("invert") || (node.at("invert").is_boolean() &&
                                              !node.at("invert").get<bool>()));
            if (supportedOptions)
            {
                stage.payload =
                    compileTransformationStage(node, stage.descriptor);
                stage.preferredResidency =
                    transformationSupportsExactDevice(
                        std::get<TransformationProgram>(stage.payload))
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.iqr" || type == "filters.mad")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool mad = type == "filters.mad";
            const bool supportedOptions =
                hasOnlyOptions(
                    node,
                    mad ? std::initializer_list<
                              std::string_view>{"type", "tag", "inputs",
                                                "dimension", "k",
                                                "mad_multiplier"}
                        : std::initializer_list<std::string_view>{"type", "tag",
                                                                  "inputs",
                                                                  "dimension",
                                                                  "k"}) &&
                node.contains("dimension") &&
                node.at("dimension").is_string() &&
                (!node.contains("k") || node.at("k").is_number()) &&
                (!mad || !node.contains("mad_multiplier") ||
                 node.at("mad_multiplier").is_number());
            if (supportedOptions)
            {
                stage.payload =
                    compileRobustStage(node, dimensions, stage.descriptor,
                                       mad ? RobustKind::Mad : RobustKind::Iqr);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint8_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.outlier")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("method") || node.at("method").is_string()) &&
                (!node.contains("min_k") ||
                 isSigned32Integer(node.at("min_k"))) &&
                (!node.contains("radius") || node.at("radius").is_number()) &&
                (!node.contains("mean_k") ||
                 isSigned32Integer(node.at("mean_k"))) &&
                (!node.contains("multiplier") ||
                 node.at("multiplier").is_number()) &&
                (!node.contains("class") ||
                 (isUnsignedInteger(node.at("class")) &&
                  node.at("class").get<std::uint64_t>() <= 255U));
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "method", "min_k",
                                "radius", "mean_k", "multiplier", "class"}) &&
                optionTypes;
            if (supportedOptions)
            {
                stage.payload = compileOutlierStage(node, stage.descriptor);
                const OutlierProgram& program =
                    std::get<OutlierProgram>(stage.payload);
                const bool radiusDevice =
                    program.method == OutlierMethod::Radius &&
                    std::isfinite(program.radius) && program.radius > 0.0;
                const bool statisticalDevice =
                    program.method == OutlierMethod::Statistical &&
                    program.meanNeighbors >= 0 && program.meanNeighbors < 64;
                stage.preferredResidency = radiusDevice || statisticalDevice
                                               ? MemoryKind::Device
                                               : MemoryKind::Host;
                if (radiusDevice)
                {
                    stage.deviceQueryBytesPerPoint = sizeof(std::uint32_t);
                    stage.descriptor.deviceToHostBytesPerInputPoint =
                        sizeof(std::uint32_t);
                }
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.radialdensity")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "radius"}) &&
                (!node.contains("radius") || node.at("radius").is_number());
            if (supportedOptions)
            {
                stage.payload =
                    compileRadialDensityStage(node, stage.descriptor);
                const RadialDensityProgram& program =
                    std::get<RadialDensityProgram>(stage.payload);
                stage.preferredResidency =
                    std::isfinite(program.radius) && program.radius > 0.0
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.radiusassign")
        {
            stage.descriptor.kind = StageKind::Knn;
            const auto stringOrStringArray = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() && !value.empty() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "src_domain",
                                      "reference_domain", "radius",
                                      "update_expression", "is3d",
                                      "max2d_above", "max2d_below"}) &&
                node.contains("radius") && node.at("radius").is_number() &&
                std::isfinite(node.at("radius").get<double>()) &&
                node.at("radius").get<double>() > 0.0 &&
                node.contains("update_expression") &&
                stringOrStringArray(node.at("update_expression")) &&
                (!node.contains("src_domain") ||
                 stringOrStringArray(node.at("src_domain"))) &&
                (!node.contains("reference_domain") ||
                 stringOrStringArray(node.at("reference_domain"))) &&
                (!node.contains("is3d") || node.at("is3d").is_boolean()) &&
                (!node.contains("max2d_above") ||
                 node.at("max2d_above").is_number()) &&
                (!node.contains("max2d_below") ||
                 node.at("max2d_below").is_number());
            bool compiled = false;
            if (supportedOptions)
            {
                try
                {
                    stage.payload = compileRadiusAssignStage(node, dimensions,
                                                             stage.descriptor);
                    compiled = !stage.descriptor.mutatesCoordinates &&
                               !stage.descriptor.writes.empty();
                }
                catch (const ExpressionError&)
                {
                }
                catch (const PlanError&)
                {
                }
            }
            if (compiled)
            {
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "radiusassign";
                stage.native = true;
            }
            else
            {
                stage.descriptor = StageDescriptor{};
                stage.descriptor.type = type;
                stage.descriptor.kind = StageKind::Knn;
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.sort")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool hasDimension = node.contains("dimension");
            const bool hasDimensions = node.contains("dimensions");
            const auto validDimensionValue = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimension",
                                      "dimensions", "order", "algorithm"}) &&
                (hasDimension != hasDimensions) &&
                validDimensionValue(
                    node.at(hasDimension ? "dimension" : "dimensions")) &&
                (!node.contains("order") || node.at("order").is_string()) &&
                (!node.contains("algorithm") ||
                 node.at("algorithm").is_string());
            if (supportedOptions)
            {
                stage.payload =
                    compileOrderingStage(node, dimensions, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint64_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.label_duplicates")
        {
            stage.descriptor.kind = StageKind::Global;
            const auto validDimensionValue = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimensions"}) &&
                (!node.contains("dimensions") ||
                 validDimensionValue(node.at("dimensions")));
            bool compiled = false;
            if (supportedOptions)
            {
                try
                {
                    stage.payload = compileLabelDuplicatesStage(
                        node, dimensions, stage.descriptor);
                    const auto& program =
                        std::get<LabelDuplicatesProgram>(stage.payload);
                    compiled =
                        std::find(program.dimensions.begin(),
                                  program.dimensions.end(),
                                  DimensionId(StandardDimension::Duplicate)) ==
                        program.dimensions.end();
                }
                catch (const PlanError&)
                {
                }
            }
            if (compiled)
            {
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.descriptor = StageDescriptor{};
                stage.descriptor.type = type;
                stage.descriptor.kind = StageKind::Global;
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.pmf")
        {
            stage.descriptor.kind = StageKind::Grid;
            const auto validClass = [&](const char* name)
            {
                return !node.contains(name) ||
                       (isUnsignedInteger(node.at(name)) &&
                        node.at(name).get<std::uint64_t>() <= 255U);
            };
            const auto finiteNumber = [&](const char* name)
            {
                return !node.contains(name) ||
                       (node.at(name).is_number() &&
                        std::isfinite(node.at(name).get<double>()));
            };
            const auto validReturn = [](const Json& value)
            {
                if (!value.is_string())
                    return false;
                const std::string returnName = trim(value.get<std::string>());
                return returnName == "first" || returnName == "intermediate" ||
                       returnName == "last" || returnName == "only";
            };
            const auto validReturns = [&]
            {
                if (!node.contains("returns"))
                    return true;
                const Json& values = node.at("returns");
                if (values.is_string())
                    return validReturn(values);
                return values.is_array() &&
                       std::all_of(values.begin(), values.end(), validReturn);
            };
            const bool optionTypes =
                finiteNumber("cell_size") &&
                (!node.contains("exponential") ||
                 node.at("exponential").is_boolean()) &&
                finiteNumber("initial_distance") &&
                finiteNumber("max_distance") &&
                finiteNumber("max_window_size") && finiteNumber("slope") &&
                validClass("ground_class") && validClass("other_class") &&
                (!node.contains("only_ground") ||
                 node.at("only_ground").is_boolean()) &&
                validReturns();
            const bool supportedOptions =
                hasOnlyOptions(
                    node, {"type", "tag", "inputs", "cell_size", "exponential",
                           "initial_distance", "returns", "max_distance",
                           "max_window_size", "slope", "ground_class",
                           "other_class", "only_ground"}) &&
                optionTypes;
            if (supportedOptions)
            {
                PmfProgram program = compilePmfStage(node, stage.descriptor);
                if (pmfProgramWithinExactDeviceEnvelope(program))
                {
                    stage.payload = program;
                    stage.preferredResidency = MemoryKind::Device;
                    stage.native = true;
                }
                else
                {
                    stage.descriptor = StageDescriptor{};
                    stage.descriptor.type = type;
                    stage.descriptor.kind = StageKind::Grid;
                    stage.preferredResidency = MemoryKind::Host;
                    stage.payload = FallbackStagePlan{sourceNode.dump()};
                }
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.csf")
        {
            stage.descriptor.kind = StageKind::Grid;
            const auto validClass = [&](const char* name)
            {
                return !node.contains(name) ||
                       (isUnsignedInteger(node.at(name)) &&
                        node.at(name).get<std::uint64_t>() <= 255U);
            };
            const auto finiteNumber = [&](const char* name)
            {
                return !node.contains(name) ||
                       (node.at(name).is_number() &&
                        std::isfinite(node.at(name).get<double>()));
            };
            const auto validInteger = [&](const char* name)
            {
                if (!node.contains(name))
                    return true;
                if (!node.at(name).is_number_integer())
                    return false;
                const std::int64_t value = node.at(name).get<std::int64_t>();
                return value >= (std::numeric_limits<int>::min)() &&
                       value <= (std::numeric_limits<int>::max)();
            };
            const auto validReturn = [](const Json& value)
            {
                if (!value.is_string())
                    return false;
                const std::string returnName = trim(value.get<std::string>());
                return returnName == "first" || returnName == "intermediate" ||
                       returnName == "last" || returnName == "only";
            };
            const auto validReturns = [&]
            {
                if (!node.contains("returns"))
                    return true;
                const Json& values = node.at("returns");
                if (values.is_string())
                    return validReturn(values);
                return values.is_array() &&
                       std::all_of(values.begin(), values.end(), validReturn);
            };
            const bool optionTypes =
                (!node.contains("smooth") || node.at("smooth").is_boolean()) &&
                finiteNumber("step") && finiteNumber("threshold") &&
                finiteNumber("hdiff") && finiteNumber("resolution") &&
                validInteger("rigidness") && validInteger("iterations") &&
                validClass("ground_class") && validClass("other_class") &&
                (!node.contains("only_ground") ||
                 node.at("only_ground").is_boolean()) &&
                validReturns();
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "smooth", "step",
                                "threshold", "hdiff", "resolution", "rigidness",
                                "iterations", "returns", "ground_class",
                                "other_class", "only_ground"}) &&
                optionTypes;
            if (supportedOptions)
            {
                CsfProgram program = compileCsfStage(node, stage.descriptor);
                if (csfProgramWithinExactDeviceEnvelope(program))
                {
                    stage.payload = program;
                    stage.preferredResidency = MemoryKind::Device;
                    stage.native = true;
                }
                else
                {
                    stage.descriptor = StageDescriptor{};
                    stage.descriptor.type = type;
                    stage.descriptor.kind = StageKind::Grid;
                    stage.preferredResidency = MemoryKind::Host;
                    stage.payload = FallbackStagePlan{sourceNode.dump()};
                }
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.elm")
        {
            stage.descriptor.kind = StageKind::Grid;
            const bool validCell =
                !node.contains("cell") ||
                (node.at("cell").is_number() &&
                 std::isfinite(node.at("cell").get<double>()));
            const bool validThreshold =
                !node.contains("threshold") ||
                (node.at("threshold").is_number() &&
                 std::isfinite(node.at("threshold").get<double>()));
            const bool validClass =
                !node.contains("class") ||
                (isUnsignedInteger(node.at("class")) &&
                 node.at("class").get<std::uint64_t>() <= 255U);
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "cell", "class",
                                      "threshold"}) &&
                validCell && validThreshold && validClass;
            if (supportedOptions)
            {
                ElmProgram program = compileElmStage(node, stage.descriptor);
                if (elmProgramWithinExactDeviceEnvelope(program))
                {
                    stage.payload = program;
                    stage.preferredResidency = MemoryKind::Device;
                    stage.native = true;
                }
                else
                {
                    stage.descriptor = StageDescriptor{};
                    stage.descriptor.type = type;
                    stage.descriptor.kind = StageKind::Grid;
                    stage.preferredResidency = MemoryKind::Host;
                    stage.payload = FallbackStagePlan{sourceNode.dump()};
                }
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.smrf")
        {
            stage.descriptor.kind = StageKind::Grid;
            const auto validClass = [&](const char* name)
            {
                return !node.contains(name) ||
                       (isUnsignedInteger(node.at(name)) &&
                        node.at(name).get<std::uint64_t>() <= 255U);
            };
            const auto finiteNumber = [&](const char* name)
            {
                return !node.contains(name) ||
                       (node.at(name).is_number() &&
                        std::isfinite(node.at(name).get<double>()));
            };
            const auto validReturn = [](const Json& value)
            {
                if (!value.is_string())
                    return false;
                const std::string returnName = trim(value.get<std::string>());
                return returnName == "first" || returnName == "intermediate" ||
                       returnName == "last" || returnName == "only";
            };
            const auto validReturns = [&]
            {
                if (!node.contains("returns"))
                    return true;
                const Json& values = node.at("returns");
                if (values.is_string())
                    return validReturn(values);
                return values.is_array() &&
                       std::all_of(values.begin(), values.end(), validReturn);
            };
            const bool optionTypes =
                finiteNumber("cell") && finiteNumber("slope") &&
                finiteNumber("scalar") && finiteNumber("threshold") &&
                finiteNumber("cut") && finiteNumber("window") &&
                validClass("ground_class") && validClass("other_class") &&
                (!node.contains("only_ground") ||
                 node.at("only_ground").is_boolean()) &&
                validReturns();
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "cell", "slope",
                                      "scalar", "threshold", "cut", "returns",
                                      "window", "ground_class", "other_class",
                                      "only_ground"}) &&
                optionTypes;
            bool supportedEnvelope = supportedOptions;
            if (supportedEnvelope)
            {
                const double cell =
                    node.contains("cell") ? node.at("cell").get<double>() : 1.0;
                const double cut =
                    node.contains("cut") ? node.at("cut").get<double>() : 0.0;
                const double window = node.contains("window")
                                          ? node.at("window").get<double>()
                                          : 18.0 * cell;
                const std::uint64_t ground =
                    node.contains("ground_class")
                        ? node.at("ground_class").get<std::uint64_t>()
                        : 2U;
                const std::uint64_t other =
                    node.contains("other_class")
                        ? node.at("other_class").get<std::uint64_t>()
                        : 1U;
                const bool onlyGround = node.contains("only_ground") &&
                                        node.at("only_ground").get<bool>();
                const double objectRadius =
                    cell > 0.0 ? std::ceil(window / cell)
                               : std::numeric_limits<double>::infinity();
                const double netRadius =
                    cell > 0.0 && cut > 0.0 ? 2.0 * std::ceil(cut / cell) : 0.0;
                supportedEnvelope =
                    cell > 0.0 && cut >= 0.0 && window >= 0.0 &&
                    objectRadius <= SmrfExactDeviceMaximumMorphologyRadius &&
                    netRadius <= SmrfExactDeviceMaximumMorphologyRadius &&
                    (ground != other || onlyGround);
            }
            if (supportedEnvelope)
            {
                stage.payload = compileSmrfStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                // Automatic placement remains fail-closed until a physical
                // exactness/profile lane publishes a measured residual.
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.mortonorder")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "reverse"}) &&
                (!node.contains("reverse") || node.at("reverse").is_boolean());
            if (supportedOptions)
            {
                stage.payload = compileMortonStage(node, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint64_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.skewnessbalancing")
        {
            stage.descriptor.kind = StageKind::Global;
            const auto validClass = [&](const char* name)
            {
                return !node.contains(name) ||
                       (isUnsignedInteger(node.at(name)) &&
                        node.at(name).get<std::uint64_t>() <= 255U);
            };
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs",
                                      "ground_class", "other_class",
                                      "only_ground"}) &&
                validClass("ground_class") && validClass("other_class") &&
                (!node.contains("only_ground") ||
                 node.at("only_ground").is_boolean());
            if (supportedOptions)
            {
                const SkewnessProgram program =
                    compileSkewnessStage(node, stage.descriptor);
                if (skewnessProgramValid(program))
                {
                    // The device returns only the exact unique-Z source
                    // permutation. The pinned recurrence and Classification
                    // publication remain host work behind the wrapper.
                    stage.payload = program;
                    stage.descriptor.deviceToHostBytesPerInputPoint =
                        sizeof(std::uint64_t);
                    stage.preferredResidency = MemoryKind::Device;
                    stage.native = true;
                }
                else
                {
                    stage.preferredResidency = MemoryKind::Host;
                    stage.payload = FallbackStagePlan{sourceNode.dump()};
                }
            }
            else
            {
                // Preserve the true ordering contract even when option-rich
                // forms delegate to the pinned implementation.
                stage.descriptor.reads = {
                    DimensionId(StandardDimension::Z),
                    DimensionId(StandardDimension::Classification)};
                stage.descriptor.writes = {
                    DimensionId(StandardDimension::Classification)};
                stage.descriptor.preservesOrder = false;
                declarePureStage(stage.descriptor, true);
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.groupby")
        {
            stage.descriptor.kind = StageKind::Split;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimension"}) &&
                node.contains("dimension") && node.at("dimension").is_string();
            if (supportedOptions)
            {
                stage.payload =
                    compileGroupByStage(node, dimensions, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint64_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.returns")
        {
            stage.descriptor.kind = StageKind::Split;
            const auto validGroups = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "groups"}) &&
                (!node.contains("groups") || validGroups(node.at("groups")));
            if (supportedOptions)
            {
                stage.payload = compileReturnsStage(node, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint64_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.divider")
        {
            stage.descriptor.kind = StageKind::Split;
            const std::string mode =
                node.contains("mode") && node.at("mode").is_string()
                    ? lowercase(node.at("mode").get<std::string>())
                    : std::string("partition");
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "mode", "count"}) &&
                node.contains("count") && isUnsignedInteger(node.at("count")) &&
                node.at("count").get<std::uint64_t>() >= 2U &&
                node.at("count").get<std::uint64_t>() <= 1000U &&
                (!node.contains("mode") || node.at("mode").is_string()) &&
                (mode == "partition" || mode == "round_robin");
            if (supportedOptions)
            {
                stage.payload = compileDividerStage(node, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    sizeof(std::uint64_t);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.splitter")
        {
            stage.descriptor.kind = StageKind::Split;
            const auto finiteNumber = [](const Json& value)
            { return value.is_number() && std::isfinite(value.get<double>()); };
            const bool numericOptions =
                (!node.contains("length") || finiteNumber(node.at("length"))) &&
                (!node.contains("origin_x") ||
                 finiteNumber(node.at("origin_x"))) &&
                (!node.contains("origin_y") ||
                 finiteNumber(node.at("origin_y"))) &&
                (!node.contains("buffer") || finiteNumber(node.at("buffer")));
            const double length =
                node.contains("length") && node.at("length").is_number()
                    ? node.at("length").get<double>()
                    : 1000.0;
            const double buffer =
                node.contains("buffer") && node.at("buffer").is_number()
                    ? node.at("buffer").get<double>()
                    : 0.0;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "length",
                                      "origin_x", "origin_y", "buffer"}) &&
                numericOptions && length > 0.0 && buffer < length / 2.0;
            if (supportedOptions)
            {
                stage.payload = compileSplitterStage(node, stage.descriptor);
                stage.descriptor.deviceToHostBytesPerInputPoint =
                    2U * sizeof(std::int32_t);
                stage.preferredResidency =
                    buffer <= 0.0 ? MemoryKind::Device : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.colorinterp")
        {
            stage.descriptor.kind = StageKind::Pointwise;
            const auto numeric = [](const Json& value)
            { return value.is_number(); };
            const bool optionTypes =
                (!node.contains("dimension") ||
                 node.at("dimension").is_string()) &&
                (!node.contains("minimum") || numeric(node.at("minimum"))) &&
                (!node.contains("maximum") || numeric(node.at("maximum"))) &&
                (!node.contains("clamp") || node.at("clamp").is_boolean()) &&
                (!node.contains("ramp") || node.at("ramp").is_string()) &&
                (!node.contains("invert") || node.at("invert").is_boolean()) &&
                (!node.contains("mad") || node.at("mad").is_boolean()) &&
                (!node.contains("mad_multiplier") ||
                 numeric(node.at("mad_multiplier"))) &&
                (!node.contains("k") || numeric(node.at("k")));
            const bool rangeValid =
                !node.contains("minimum") || !node.contains("maximum") ||
                !numeric(node.at("minimum")) || !numeric(node.at("maximum")) ||
                node.at("maximum").get<double>() >
                    node.at("minimum").get<double>();
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "dimension", "minimum",
                                "maximum", "clamp", "ramp", "invert", "mad",
                                "mad_multiplier", "k"}) &&
                optionTypes && rangeValid;
            if (supportedOptions)
            {
                stage.payload =
                    compileColorinterpStage(node, dimensions, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.nndistance")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("mode") || node.at("mode").is_string()) &&
                (!node.contains("k") || isUnsignedInteger(node.at("k")));
            const std::uint64_t neighbors =
                node.contains("k") && isUnsignedInteger(node.at("k"))
                    ? node.at("k").get<std::uint64_t>()
                    : 10U;
            const std::string mode =
                node.contains("mode") && node.at("mode").is_string()
                    ? lowercase(node.at("mode").get<std::string>())
                    : "kth";
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "mode", "k"}) &&
                optionTypes && neighbors >= 1U && neighbors < 64U &&
                (mode == "kth" || mode == "avg");
            if (supportedOptions)
            {
                stage.payload = compileNnDistanceStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "nndistance";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.hag_nn")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("count") ||
                 isUnsignedInteger(node.at("count"))) &&
                (!node.contains("max_distance") ||
                 node.at("max_distance").is_number()) &&
                (!node.contains("allow_extrapolation") ||
                 node.at("allow_extrapolation").is_boolean()) &&
                (!node.contains("class") ||
                 isUnsignedInteger(node.at("class")));
            const std::uint64_t count =
                node.contains("count") && isUnsignedInteger(node.at("count"))
                    ? node.at("count").get<std::uint64_t>()
                    : 1U;
            const std::uint64_t groundClass =
                node.contains("class") && isUnsignedInteger(node.at("class"))
                    ? node.at("class").get<std::uint64_t>()
                    : 2U;
            const bool maximumDistanceValid =
                !node.contains("max_distance") ||
                (node.at("max_distance").is_number() &&
                 std::isfinite(node.at("max_distance").get<double>()));
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "count",
                                      "max_distance", "allow_extrapolation",
                                      "class"}) &&
                optionTypes && count >= 1U && count <= 64U &&
                groundClass <= 255U && maximumDistanceValid;
            if (supportedOptions)
            {
                stage.payload = compileHagNnStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.hag_delaunay")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("count") ||
                 isUnsignedInteger(node.at("count"))) &&
                (!node.contains("allow_extrapolation") ||
                 node.at("allow_extrapolation").is_boolean()) &&
                (!node.contains("class") ||
                 isUnsignedInteger(node.at("class")));
            const std::uint64_t count =
                node.contains("count") && isUnsignedInteger(node.at("count"))
                    ? node.at("count").get<std::uint64_t>()
                    : 10U;
            const std::uint64_t groundClass =
                node.contains("class") && isUnsignedInteger(node.at("class"))
                    ? node.at("class").get<std::uint64_t>()
                    : 2U;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "count",
                                      "allow_extrapolation", "class"}) &&
                optionTypes && count == 3U && groundClass <= 255U;
            if (supportedOptions)
            {
                stage.payload = compileHagDelaunayStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.normal")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("knn") || isSigned32Integer(node.at("knn"))) &&
                (!node.contains("always_up") ||
                 node.at("always_up").is_boolean()) &&
                (!node.contains("refine") || node.at("refine").is_boolean());
            const std::int64_t neighbors =
                node.contains("knn") && isSigned32Integer(node.at("knn"))
                    ? node.at("knn").get<std::int64_t>()
                    : 8;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "knn",
                                      "always_up", "refine"}) &&
                optionTypes && neighbors >= 2 && neighbors < 64 &&
                (!node.contains("refine") || !node.at("refine").get<bool>());
            if (supportedOptions)
            {
                stage.payload = compileNormalStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "normal";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.approximatecoplanar")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("knn") || isSigned32Integer(node.at("knn"))) &&
                (!node.contains("thresh1") || node.at("thresh1").is_number()) &&
                (!node.contains("thresh2") || node.at("thresh2").is_number());
            const std::int64_t neighbors =
                node.contains("knn") && isSigned32Integer(node.at("knn"))
                    ? node.at("knn").get<std::int64_t>()
                    : 8;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "knn", "thresh1",
                                      "thresh2"}) &&
                optionTypes && neighbors >= 3 && neighbors <= 64;
            if (supportedOptions)
            {
                stage.payload =
                    compileApproximateCoplanarStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "approximatecoplanar";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.neighborclassifier")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                node.contains("k") && isSigned32Integer(node.at("k")) &&
                (!node.contains("dimension") ||
                 (node.at("dimension").is_string() &&
                  node.at("dimension").get<std::string>() == "Classification"));
            const std::int64_t neighbors =
                node.contains("k") && isSigned32Integer(node.at("k"))
                    ? node.at("k").get<std::int64_t>()
                    : 0;
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "k", "dimension"}) &&
                optionTypes && neighbors >= 1 && neighbors <= 64;
            if (supportedOptions)
            {
                stage.payload =
                    compileNeighborClassifierStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "neighborclassifier";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.optimalneighborhood")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes = (!node.contains("min_k") ||
                                      node.at("min_k").is_number_unsigned()) &&
                                     (!node.contains("max_k") ||
                                      node.at("max_k").is_number_unsigned());
            const std::int64_t minimumK =
                node.contains("min_k") && node.at("min_k").is_number_unsigned()
                    ? node.at("min_k").get<std::int64_t>()
                    : 10;
            const std::int64_t maximumK =
                node.contains("max_k") && node.at("max_k").is_number_unsigned()
                    ? node.at("max_k").get<std::int64_t>()
                    : 14;
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "min_k", "max_k"}) &&
                optionTypes && minimumK >= 1 && minimumK <= maximumK &&
                maximumK <= 64;
            if (supportedOptions)
            {
                stage.payload =
                    compileOptimalNeighborhoodStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "optimalneighborhood";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.estimaterank")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("knn") || isSigned32Integer(node.at("knn"))) &&
                (!node.contains("thresh") || node.at("thresh").is_number());
            const std::int64_t neighbors =
                node.contains("knn") && isSigned32Integer(node.at("knn"))
                    ? node.at("knn").get<std::int64_t>()
                    : 8;
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "knn", "thresh"}) &&
                optionTypes && neighbors >= 3 && neighbors <= 64;
            if (supportedOptions)
            {
                stage.payload =
                    compileEstimateRankStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "estimaterank";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.lof")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes = !node.contains("minpts") ||
                                     isSigned32Integer(node.at("minpts"));
            const std::int64_t minimumPoints =
                node.contains("minpts") && isSigned32Integer(node.at("minpts"))
                    ? node.at("minpts").get<std::int64_t>()
                    : 10;
            // The self-inclusive increment must stay inside the 64-neighbor
            // exact envelope.
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "minpts"}) &&
                optionTypes && minimumPoints >= 1 && minimumPoints <= 63;
            if (supportedOptions)
            {
                stage.payload = compileLofStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "lof";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.eigenvalues")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("knn") || isSigned32Integer(node.at("knn"))) &&
                (!node.contains("normalize") ||
                 node.at("normalize").is_boolean()) &&
                (!node.contains("stride") ||
                 isUnsignedInteger(node.at("stride"))) &&
                (!node.contains("min_k") ||
                 isSigned32Integer(node.at("min_k")));
            const std::int64_t neighbors =
                node.contains("knn") && isSigned32Integer(node.at("knn"))
                    ? node.at("knn").get<std::int64_t>()
                    : 8;
            const std::uint64_t stride =
                node.contains("stride") && isUnsignedInteger(node.at("stride"))
                    ? node.at("stride").get<std::uint64_t>()
                    : 1U;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "knn",
                                      "normalize", "stride", "min_k"}) &&
                optionTypes && neighbors >= 2 && neighbors < 64 && stride == 1U;
            if (supportedOptions)
            {
                stage.payload = compileEigenvaluesStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "eigenvalues";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.covariancefeatures")
        {
            stage.descriptor.kind = StageKind::Knn;
            const bool optionTypes =
                (!node.contains("knn") || isSigned32Integer(node.at("knn"))) &&
                (!node.contains("threads") ||
                 isSigned32Integer(node.at("threads"))) &&
                (!node.contains("stride") ||
                 isUnsignedInteger(node.at("stride"))) &&
                (!node.contains("min_k") ||
                 isSigned32Integer(node.at("min_k"))) &&
                (!node.contains("mode") || node.at("mode").is_string()) &&
                (!node.contains("optimized") ||
                 node.at("optimized").is_boolean());
            const std::int64_t neighbors =
                node.contains("knn") && isSigned32Integer(node.at("knn"))
                    ? node.at("knn").get<std::int64_t>()
                    : 10;
            const std::int64_t threads =
                node.contains("threads") &&
                        isSigned32Integer(node.at("threads"))
                    ? node.at("threads").get<std::int64_t>()
                    : 1;
            const std::uint64_t stride =
                node.contains("stride") && isUnsignedInteger(node.at("stride"))
                    ? node.at("stride").get<std::uint64_t>()
                    : 1U;
            const std::string mode =
                node.contains("mode") && node.at("mode").is_string()
                    ? lowercase(node.at("mode").get<std::string>())
                    : "sqrt";
            std::uint32_t featureMask = 0U;
            std::vector<DimensionId> featureDimensions;
            const bool featureSetSupported =
                parseCovarianceFeatureSet(node, featureMask, featureDimensions);
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "knn", "threads",
                                      "feature_set", "stride", "min_k", "mode",
                                      "optimized"}) &&
                optionTypes && neighbors >= 2 && neighbors < 64 &&
                threads == 1 && stride == 1U && featureSetSupported &&
                (mode == "raw" || mode == "sqrt" || mode == "normalized") &&
                (!node.contains("optimized") ||
                 !node.at("optimized").get<bool>());
            if (supportedOptions)
            {
                stage.payload =
                    compileCovarianceFeaturesStage(node, stage.descriptor);
                stage.preferredResidency = MemoryKind::Device;
                stage.descriptor.placementModel = "covariancefeatures";
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.stats")
        {
            stage.descriptor.kind = StageKind::Global;
            const auto validNames = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            bool optionTypes = true;
            for (const char* option :
                 {"dimensions", "enumerate", "global", "count"})
                optionTypes = optionTypes && (!node.contains(option) ||
                                              validNames(node.at(option)));
            optionTypes = optionTypes &&
                          (!node.contains("advanced") ||
                           node.at("advanced").is_boolean()) &&
                          (!node.contains("commonsrs") ||
                           node.at("commonsrs").is_string());
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimensions",
                                      "enumerate", "global", "count",
                                      "advanced", "commonsrs"}) &&
                optionTypes;
            if (supportedOptions)
            {
                stage.payload =
                    compileStatsStage(node, dimensions, stage.descriptor);
                const StatsProgram& program =
                    std::get<StatsProgram>(stage.payload);
                const bool deviceModes = std::all_of(
                    program.modes.begin(), program.modes.end(),
                    [](SummaryMode mode) { return mode == SummaryMode::None; });
                stage.preferredResidency = !program.advanced && deviceModes
                                               ? MemoryKind::Device
                                               : MemoryKind::Host;
                if (stage.preferredResidency == MemoryKind::Device)
                    stage.descriptor.deviceToHostFixedBytes =
                        program.dimensions.size() * sizeof(SummaryState);
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.info")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool optionTypes =
                (!node.contains("point") || node.at("point").is_string()) &&
                (!node.contains("query") || node.at("query").is_string());
            const bool supportedOptions =
                hasOnlyOptions(node,
                               {"type", "tag", "inputs", "point", "query"}) &&
                optionTypes;
            if (supportedOptions)
            {
                stage.payload = compileInfoStage(node, stage.descriptor);
                const InfoProgram& program =
                    std::get<InfoProgram>(stage.payload);
                stage.preferredResidency =
                    program.pointSpec.empty() && program.querySpec.empty()
                        ? MemoryKind::Device
                        : MemoryKind::Host;
                if (stage.preferredResidency == MemoryKind::Device)
                    stage.descriptor.deviceToHostFixedBytes =
                        sizeof(BoundsResult);
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.expressionstats")
        {
            stage.descriptor.kind = StageKind::Global;
            const auto validExpressions = [](const Json& value)
            {
                return value.is_string() ||
                       (value.is_array() &&
                        std::all_of(value.begin(), value.end(),
                                    [](const Json& item)
                                    { return item.is_string(); }));
            };
            const bool optionTypes = node.contains("dimension") &&
                                     node.at("dimension").is_string() &&
                                     node.contains("expressions") &&
                                     validExpressions(node.at("expressions"));
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs", "dimension",
                                      "expressions"}) &&
                optionTypes;
            if (supportedOptions)
            {
                stage.payload = compileExpressionStatsStage(node, dimensions,
                                                            stage.descriptor);
                const ExpressionStatsProgram& program =
                    std::get<ExpressionStatsProgram>(stage.payload);
                const bool deviceExpressions =
                    !program.expressions.empty() &&
                    std::all_of(
                        program.expressions.begin(), program.expressions.end(),
                        [](const CompiledExpression& expression)
                        {
                            return predicateSupportsExactDevice(
                                PredicateProgram{expression, expression.reads});
                        });
                stage.preferredResidency =
                    deviceExpressions ? MemoryKind::Device : MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.merge")
        {
            stage.descriptor.kind = StageKind::Global;
            const bool supportedOptions =
                hasOnlyOptions(node, {"type", "tag", "inputs"});
            if (supportedOptions)
            {
                stage.payload = MergeProgram{};
                stage.preferredResidency = MemoryKind::Host;
                stage.native = true;
            }
            else
            {
                stage.preferredResidency = MemoryKind::Host;
                stage.payload = FallbackStagePlan{sourceNode.dump()};
            }
        }
        else if (type == "filters.reprojection")
        {
            // Keep the original PDAL host implementation. Reprojection
            // rewrites coordinates, which is an explicit residency boundary
            // and a declared spatial-index invalidator. Upstream silently
            // drops points whose transform fails, so the one-view
            // cardinality contract is declarable only under
            // error_on_failure, where every point either reprojects or the
            // stage throws.
            stage.descriptor.kind = StageKind::Cpu;
            stage.descriptor.mutatesCoordinates = true;
            stage.descriptor.fusion.cardinalityPreserving =
                node.contains("error_on_failure") &&
                node.at("error_on_failure").is_boolean() &&
                node.at("error_on_failure").get<bool>();
            stage.preferredResidency = MemoryKind::Host;
            stage.payload = FallbackStagePlan{sourceNode.dump()};
        }
        else if (type == "filters.randomize")
        {
            // Keep the original PDAL host implementation. Its point-order
            // mutation is an explicit residency boundary, while its known
            // one-view cardinality contract lets the runtime placement model
            // propagate exact counts to a later independent device region.
            stage.descriptor.kind = StageKind::Cpu;
            stage.descriptor.preservesOrder = false;
            stage.descriptor.fusion.cardinalityPreserving = true;
            stage.preferredResidency = MemoryKind::Host;
            stage.payload = FallbackStagePlan{sourceNode.dump()};
        }
        else
        {
            stage.descriptor.kind = StageKind::Cpu;
            stage.preferredResidency = MemoryKind::Host;
            stage.payload = FallbackStagePlan{sourceNode.dump()};
        }

        finalizeFusionSemantics(stage.descriptor, node);

        if (!stage.native)
        {
            summary.allStagesNative = false;
            summary.fallbackReasons.push_back("stage " +
                                              std::to_string(stage.id) +
                                              " is not native: " + type);
            if (options.strict)
                throw PlanError(summary.fallbackReasons.back());
        }

        stages.push_back(std::move(stage));
        if (!stages.back().tag.empty())
            tags.emplace(stages.back().tag, stages.back().id);
        if (stages.back().role == StageRole::Reader)
            frontier.push_back(stages.back().id);
        else
            frontier = {stages.back().id};
    }

    finishPlan(stages, dimensions, summary, options);
    return Plan(std::move(stages), std::move(summary));
}

} // namespace pdg
