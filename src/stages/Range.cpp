/******************************************************************************
 * Range grammar and semantics are derived from PDAL's BSD-3-Clause
 * filters/private/DimRange implementation.
 * Copyright (c) 2015, Bradley J Chambers (brad.chambers@gmail.com)
 ******************************************************************************/

#include <pdg/stages/Range.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pdg
{

namespace
{
struct ParsedRange
{
    std::string name;
    DimensionId dimension;
    double lower = std::numeric_limits<double>::lowest();
    double upper = std::numeric_limits<double>::max();
    bool inclusiveLower = true;
    bool inclusiveUpper = true;
    bool negate = false;
};

std::string trim(std::string_view value)
{
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])))
        ++begin;
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1U])))
        --end;
    return std::string(value.substr(begin, end - begin));
}

void skipSpaces(std::string_view text, std::size_t& position)
{
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])))
        ++position;
}

double parseBound(std::string_view text, std::size_t& position, double missing,
                  std::string_view missingDelimiter)
{
    std::istringstream stream{std::string(text.substr(position))};
    stream.imbue(std::locale::classic());
    double value = 0.0;
    stream >> value;
    if (stream.fail())
        return missing;
    if (stream.eof())
        throw ExpressionError(std::string(missingDelimiter));
    const std::streampos consumed = stream.tellg();
    if (consumed < std::streampos(0))
        throw ExpressionError(std::string(missingDelimiter));
    position += static_cast<std::size_t>(consumed);
    return value;
}

ParsedRange parseRange(std::string text, DimensionRegistry& dimensions)
{
    text = trim(text);
    std::size_t position = 0;
    skipSpaces(text, position);
    if (position >= text.size() ||
        !std::isalpha(static_cast<unsigned char>(text[position])))
        throw ExpressionError("filters.range has no dimension name");
    const std::size_t nameBegin = position++;
    while (position < text.size())
    {
        const unsigned char character =
            static_cast<unsigned char>(text[position]);
        if (!std::isalpha(character) && !std::isdigit(character) &&
            character != '_' && character != ' ')
            break;
        ++position;
    }

    ParsedRange range;
    range.name = text.substr(nameBegin, position - nameBegin);
    const DimensionDefinition* definition = dimensions.find(range.name);
    if (!definition)
        throw ExpressionError("filters.range dimension does not exist: " +
                              range.name);
    range.dimension = definition->id;

    if (position < text.size() && text[position] == '!')
    {
        range.negate = true;
        ++position;
    }
    if (position >= text.size() ||
        (text[position] != '(' && text[position] != '['))
        throw ExpressionError("filters.range is missing '(' or '['");
    range.inclusiveLower = text[position] == '[';
    ++position;

    range.lower =
        parseBound(text, position, range.lower, "filters.range is missing ':'");
    skipSpaces(text, position);
    if (position >= text.size() || text[position] != ':')
        throw ExpressionError("filters.range is missing ':'");
    ++position;

    range.upper = parseBound(text, position, range.upper,
                             "filters.range is missing ')' or ']'");
    skipSpaces(text, position);
    if (position >= text.size() ||
        (text[position] != ')' && text[position] != ']'))
        throw ExpressionError("filters.range is missing ')' or ']'");
    range.inclusiveUpper = text[position] == ']';
    ++position;
    skipSpaces(text, position);
    if (position != text.size())
        throw ExpressionError(
            "filters.range has characters after a valid range");
    return range;
}

std::vector<ParsedRange>
parseRanges(std::span<const std::string> specifications,
            DimensionRegistry& dimensions)
{
    std::vector<ParsedRange> ranges;
    for (const std::string& specification : specifications)
    {
        std::size_t begin = 0;
        while (begin <= specification.size())
        {
            const std::size_t comma = specification.find(',', begin);
            const std::size_t end =
                comma == std::string::npos ? specification.size() : comma;
            const std::string item = trim(
                std::string_view(specification).substr(begin, end - begin));
            if (item.empty())
                throw ExpressionError("filters.range contains an empty range");
            ranges.push_back(parseRange(item, dimensions));
            if (comma == std::string::npos)
                break;
            begin = comma + 1U;
        }
    }
    if (ranges.empty())
        throw ExpressionError("filters.range requires at least one limit");
    std::sort(ranges.begin(), ranges.end(),
              [](const ParsedRange& left, const ParsedRange& right)
              { return left.name < right.name; });
    return ranges;
}

class PredicateBuilder
{
public:
    void push(ExpressionOp op, DimensionId dimension = {},
              double immediate = 0.0)
    {
        m_expression.instructions.push_back({op, dimension, immediate});
        switch (op)
        {
        case ExpressionOp::PushConstant:
        case ExpressionOp::PushFalse:
        case ExpressionOp::PushTrue:
        case ExpressionOp::LoadDimension:
            ++m_stackDepth;
            m_expression.maximumStackDepth =
                std::max(m_expression.maximumStackDepth, m_stackDepth);
            break;
        case ExpressionOp::LogicalNot:
        case ExpressionOp::IsNan:
        case ExpressionOp::IsMaximum:
        case ExpressionOp::IsMinimum:
        case ExpressionOp::Negative:
        case ExpressionOp::Floor:
        case ExpressionOp::Ceil:
        case ExpressionOp::Round:
        case ExpressionOp::Absolute:
        case ExpressionOp::SquareRoot:
        case ExpressionOp::Sine:
        case ExpressionOp::Cosine:
        case ExpressionOp::Tangent:
        case ExpressionOp::ArcSine:
        case ExpressionOp::ArcCosine:
        case ExpressionOp::ArcTangent:
        case ExpressionOp::HyperbolicSine:
        case ExpressionOp::HyperbolicCosine:
        case ExpressionOp::HyperbolicTangent:
        case ExpressionOp::InverseHyperbolicSine:
        case ExpressionOp::InverseHyperbolicCosine:
        case ExpressionOp::NaturalLog:
        case ExpressionOp::Log2:
        case ExpressionOp::Log10:
        case ExpressionOp::Exponential:
        case ExpressionOp::Exponential2:
            break;
        case ExpressionOp::Add:
        case ExpressionOp::Subtract:
        case ExpressionOp::Multiply:
        case ExpressionOp::Divide:
        case ExpressionOp::Equal:
        case ExpressionOp::NotEqual:
        case ExpressionOp::Greater:
        case ExpressionOp::GreaterEqual:
        case ExpressionOp::Less:
        case ExpressionOp::LessEqual:
        case ExpressionOp::LogicalAnd:
        case ExpressionOp::LogicalOr:
            if (m_stackDepth < 2U)
                throw ExpressionError("invalid generated range predicate");
            --m_stackDepth;
            break;
        }
    }

    void range(const ParsedRange& range)
    {
        if (range.negate)
        {
            push(ExpressionOp::LoadDimension, range.dimension);
            push(ExpressionOp::IsNan);
            push(ExpressionOp::LoadDimension, range.dimension);
            push(ExpressionOp::PushConstant, {}, range.lower);
            push(range.inclusiveLower ? ExpressionOp::Less
                                      : ExpressionOp::LessEqual);
            push(ExpressionOp::LogicalOr);
            push(ExpressionOp::LoadDimension, range.dimension);
            push(ExpressionOp::PushConstant, {}, range.upper);
            push(range.inclusiveUpper ? ExpressionOp::Greater
                                      : ExpressionOp::GreaterEqual);
            push(ExpressionOp::LogicalOr);
            return;
        }
        push(ExpressionOp::LoadDimension, range.dimension);
        push(ExpressionOp::PushConstant, {}, range.lower);
        push(range.inclusiveLower ? ExpressionOp::GreaterEqual
                                  : ExpressionOp::Greater);
        push(ExpressionOp::LoadDimension, range.dimension);
        push(ExpressionOp::PushConstant, {}, range.upper);
        push(range.inclusiveUpper ? ExpressionOp::LessEqual
                                  : ExpressionOp::Less);
        push(ExpressionOp::LogicalAnd);
    }

    CompiledExpression finish()
    {
        if (m_stackDepth != 1U)
            throw ExpressionError("invalid generated range predicate stack");
        m_expression.boolean = true;
        return std::move(m_expression);
    }

private:
    CompiledExpression m_expression;
    std::size_t m_stackDepth = 0;
};
} // unnamed namespace

PredicateProgram
compileRangePredicate(std::span<const std::string> specifications,
                      DimensionRegistry& dimensions)
{
    const std::vector<ParsedRange> ranges =
        parseRanges(specifications, dimensions);
    PredicateBuilder builder;
    std::vector<DimensionId> reads;
    std::string_view currentName;
    bool hasRangeInGroup = false;
    bool hasCompletedGroup = false;
    for (const ParsedRange& range : ranges)
    {
        if (!hasRangeInGroup || range.name != currentName)
        {
            if (hasRangeInGroup)
            {
                hasCompletedGroup = true;
                hasRangeInGroup = false;
            }
            currentName = range.name;
            reads.push_back(range.dimension);
            builder.range(range);
            if (hasCompletedGroup)
                builder.push(ExpressionOp::LogicalAnd);
            hasRangeInGroup = true;
        }
        else
        {
            builder.range(range);
            builder.push(ExpressionOp::LogicalOr);
        }
    }

    PredicateProgram program;
    program.expression = builder.finish();
    program.expression.reads = reads;
    program.reads = std::move(reads);
    return program;
}

} // namespace pdg
