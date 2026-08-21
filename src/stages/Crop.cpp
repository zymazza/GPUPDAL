/******************************************************************************
 * Bounds grammar and containment semantics are derived from PDAL's
 * BSD-3-Clause pdal/util/Bounds implementation.
 * Copyright (c) 2011, Michael P. Gerlek (mpg@flaxen.com)
 ******************************************************************************/

#include <pdg/stages/Crop.hpp>

#include <array>
#include <cstddef>
#include <locale>
#include <sstream>
#include <string>

namespace pdg
{

namespace
{
struct Interval
{
    double minimum = 0.0;
    double maximum = 0.0;
};

void expect(std::istringstream& stream, char expected,
            std::string_view description)
{
    stream >> std::ws;
    char actual = 0;
    if (!stream.get(actual) || actual != expected)
        throw ExpressionError("filters.crop " + std::string(description));
}

double parseNumber(std::istringstream& stream)
{
    stream >> std::ws;
    double value = 0.0;
    if (!(stream >> value))
        throw ExpressionError("filters.crop has an invalid bounds value");
    return value;
}

Interval parseInterval(std::istringstream& stream)
{
    expect(stream, '[', "bounds interval is missing '['");
    const double minimum = parseNumber(stream);
    expect(stream, ',', "bounds interval is missing ','");
    const double maximum = parseNumber(stream);
    expect(stream, ']', "bounds interval is missing ']'");
    return {minimum, maximum};
}

struct ParsedBounds
{
    std::array<Interval, 3> axes{};
    std::size_t dimensions = 0;
};

ParsedBounds parseBounds(std::string_view text)
{
    std::istringstream stream{std::string(text)};
    stream.imbue(std::locale::classic());
    ParsedBounds result;
    expect(stream, '(', "bounds are missing '('");
    result.axes[0] = parseInterval(stream);
    expect(stream, ',', "bounds are missing the X/Y separator");
    result.axes[1] = parseInterval(stream);
    result.dimensions = 2;

    stream >> std::ws;
    if (stream.peek() == ',')
    {
        static_cast<void>(stream.get());
        result.axes[2] = parseInterval(stream);
        result.dimensions = 3;
    }
    expect(stream, ')', "bounds are missing ')'");
    stream >> std::ws;
    if (stream.peek() != std::char_traits<char>::eof())
        throw ExpressionError("filters.crop has characters after valid bounds");
    return result;
}

void appendAxis(CompiledExpression& expression, DimensionId dimension,
                const Interval& interval, bool combine)
{
    auto& instructions = expression.instructions;
    instructions.push_back({ExpressionOp::LoadDimension, dimension, 0.0});
    instructions.push_back({ExpressionOp::PushConstant, {}, interval.minimum});
    instructions.push_back({ExpressionOp::GreaterEqual, {}, 0.0});
    instructions.push_back({ExpressionOp::LoadDimension, dimension, 0.0});
    instructions.push_back({ExpressionOp::PushConstant, {}, interval.maximum});
    instructions.push_back({ExpressionOp::LessEqual, {}, 0.0});
    instructions.push_back({ExpressionOp::LogicalAnd, {}, 0.0});
    if (combine)
        instructions.push_back({ExpressionOp::LogicalAnd, {}, 0.0});
}
} // unnamed namespace

PredicateProgram compileCropPredicate(std::string_view bounds, bool outside,
                                      DimensionRegistry& dimensions)
{
    const ParsedBounds parsed = parseBounds(bounds);
    const std::array<DimensionId, 3> ids = {DimensionId(StandardDimension::X),
                                            DimensionId(StandardDimension::Y),
                                            DimensionId(StandardDimension::Z)};

    PredicateProgram program;
    program.reads.assign(ids.begin(), ids.begin() + parsed.dimensions);
    program.expression.reads = program.reads;
    program.expression.maximumStackDepth = 4;
    program.expression.boolean = true;
    program.expression.instructions.reserve(parsed.dimensions * 8U);
    for (std::size_t axis = 0; axis < parsed.dimensions; ++axis)
    {
        static_cast<void>(dimensions.require(ids[axis]));
        appendAxis(program.expression, ids[axis], parsed.axes[axis], axis != 0);
    }
    if (outside)
        program.expression.instructions.push_back(
            {ExpressionOp::LogicalNot, {}, 0.0});
    return program;
}

} // namespace pdg
