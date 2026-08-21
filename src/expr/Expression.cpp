#include <pdg/Expression.hpp>
#include <pdg/stages/Assign.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace pdg
{

namespace
{
enum class TokenKind
{
    Eof,
    Error,
    Assign,
    Plus,
    Dash,
    Slash,
    Asterisk,
    Lparen,
    Rparen,
    Not,
    Or,
    And,
    Greater,
    Less,
    Equal,
    NotEqual,
    LessEqual,
    GreaterEqual,
    Number,
    Identifier
};

struct Token
{
    TokenKind kind = TokenKind::Error;
    std::string text;
    double number = 0.0;
};

class Lexer
{
public:
    explicit Lexer(std::string_view text) : m_text(text) {}

    Token next()
    {
        while (m_position < m_text.size() &&
               std::isspace(static_cast<unsigned char>(m_text[m_position])))
            ++m_position;
        if (m_position == m_text.size())
            return {TokenKind::Eof, {}};

        const char character = m_text[m_position++];
        switch (character)
        {
        case '+':
            return {TokenKind::Plus, "+"};
        case '*':
            return {TokenKind::Asterisk, "*"};
        case '/':
            return {TokenKind::Slash, "/"};
        case '(':
            return {TokenKind::Lparen, "("};
        case ')':
            return {TokenKind::Rparen, ")"};
        case '-':
            if (m_position < m_text.size() && m_text[m_position] == '-')
            {
                ++m_position;
                return {TokenKind::Error,
                        "Found disallowed consecutive dashes: '--'"};
            }
            return {TokenKind::Dash, "-"};
        case '&':
            if (take('&'))
                return {TokenKind::And, "&&"};
            return {TokenKind::Error, "'&' invalid in this context."};
        case '|':
            if (take('|'))
                return {TokenKind::Or, "||"};
            return {TokenKind::Error, "'|' invalid in this context."};
        case '!':
            if (take('='))
                return {TokenKind::NotEqual, "!="};
            return {TokenKind::Not, "!"};
        case '=':
            if (take('='))
                return {TokenKind::Equal, "=="};
            return {TokenKind::Assign, "="};
        case '<':
            if (take('='))
                return {TokenKind::LessEqual, "<="};
            return {TokenKind::Less, "<"};
        case '>':
            if (take('='))
                return {TokenKind::GreaterEqual, ">="};
            return {TokenKind::Greater, ">"};
        default:
            break;
        }

        if (character >= '0' && character <= '9')
            return number(m_position - 1U);
        if ((character >= 'A' && character <= 'Z') ||
            (character >= 'a' && character <= 'z'))
            return identifier(m_position - 1U);
        return {TokenKind::Error, std::string(1, character)};
    }

private:
    bool take(char expected)
    {
        if (m_position < m_text.size() && m_text[m_position] == expected)
        {
            ++m_position;
            return true;
        }
        return false;
    }

    Token number(std::size_t begin)
    {
        std::istringstream input(std::string(m_text.substr(begin)));
        input.imbue(std::locale::classic());
        double value = 0.0;
        input >> value;
        if (input.fail())
            return {TokenKind::Error, std::string(m_text.substr(begin, 1))};
        std::size_t length = m_text.size() - begin;
        if (!input.eof())
        {
            const std::streampos position = input.tellg();
            if (position < 0)
                return {TokenKind::Error, std::string(m_text.substr(begin, 1))};
            length = static_cast<std::size_t>(position);
        }
        m_position = begin + length;
        return {TokenKind::Number, std::string(m_text.substr(begin, length)),
                value};
    }

    Token identifier(std::size_t begin)
    {
        while (m_position < m_text.size())
        {
            const char character = m_text[m_position];
            const bool alphaNumeric = (character >= 'A' && character <= 'Z') ||
                                      (character >= 'a' && character <= 'z') ||
                                      (character >= '0' && character <= '9');
            if (!alphaNumeric && character != '_')
                break;
            ++m_position;
        }
        return {TokenKind::Identifier,
                std::string(m_text.substr(begin, m_position - begin))};
    }

    std::string_view m_text;
    std::size_t m_position = 0;
};

enum class ResultKind
{
    Numeric,
    Boolean
};

struct PartialExpression
{
    std::vector<ExpressionInstruction> instructions;
    std::vector<DimensionId> reads;
    std::size_t maximumStackDepth = 0;
    ResultKind kind = ResultKind::Numeric;
    std::optional<double> numericConstant;
    std::optional<bool> directBooleanConstant;
};

void appendUnique(std::vector<DimensionId>& values, DimensionId value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

void appendUnique(std::vector<DimensionId>& values,
                  const std::vector<DimensionId>& additions)
{
    for (DimensionId value : additions)
        appendUnique(values, value);
}

bool iequals(std::string_view first, std::string_view second)
{
    if (first.size() != second.size())
        return false;
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        const unsigned char a = static_cast<unsigned char>(first[index]);
        const unsigned char b = static_cast<unsigned char>(second[index]);
        if (std::toupper(a) != std::toupper(b))
            return false;
    }
    return true;
}

PartialExpression numericConstant(double value)
{
    PartialExpression expression;
    expression.instructions.push_back({ExpressionOp::PushConstant, {}, value});
    expression.maximumStackDepth = 1;
    expression.numericConstant = value;
    return expression;
}

PartialExpression booleanConstant(bool value)
{
    PartialExpression expression;
    expression.instructions.push_back(
        {value ? ExpressionOp::PushTrue : ExpressionOp::PushFalse, {}, 0.0});
    expression.maximumStackDepth = 1;
    expression.kind = ResultKind::Boolean;
    expression.directBooleanConstant = value;
    return expression;
}

PartialExpression dimensionValue(DimensionId id)
{
    PartialExpression expression;
    expression.instructions.push_back({ExpressionOp::LoadDimension, id, 0.0});
    expression.reads.push_back(id);
    expression.maximumStackDepth = 1;
    return expression;
}

PartialExpression combine(PartialExpression left, PartialExpression right,
                          ExpressionOp op, ResultKind resultKind)
{
    PartialExpression expression;
    expression.maximumStackDepth =
        std::max(left.maximumStackDepth, 1U + right.maximumStackDepth);
    expression.instructions = std::move(left.instructions);
    expression.instructions.insert(
        expression.instructions.end(),
        std::make_move_iterator(right.instructions.begin()),
        std::make_move_iterator(right.instructions.end()));
    expression.instructions.push_back({op, {}, 0.0});
    expression.reads = std::move(left.reads);
    appendUnique(expression.reads, right.reads);
    expression.kind = resultKind;
    return expression;
}

PartialExpression unary(PartialExpression value, ExpressionOp op,
                        ResultKind resultKind)
{
    value.instructions.push_back({op, {}, 0.0});
    value.kind = resultKind;
    value.numericConstant.reset();
    value.directBooleanConstant.reset();
    return value;
}

class Parser
{
public:
    Parser(std::string_view text, DimensionRegistry& dimensions)
        : m_lexer(text), m_dimensions(dimensions)
    {
        advance();
    }

    CompiledExpression valueExpression()
    {
        PartialExpression expression = add(false);
        requireKind(expression, ResultKind::Numeric,
                    "expression doesn't evaluate to a value");
        expect(TokenKind::Eof, "invalid token following value expression");
        return finish(std::move(expression));
    }

    CompiledExpression conditionalExpression()
    {
        PartialExpression expression = logicalOr();
        requireKind(expression, ResultKind::Boolean,
                    "expression evaluates to a value, not a boolean");
        expect(TokenKind::Eof,
               "invalid token following conditional expression");
        rejectConstantCondition(expression);
        return finish(std::move(expression));
    }

    PointAssignment assignment()
    {
        if (m_current.kind != TokenKind::Identifier)
            fail("expected dimension name for assignment");
        const std::string destinationName = m_current.text;
        advance();
        expectAndAdvance(TokenKind::Assign,
                         "expected '=' after dimension name in assignment");

        const DimensionDefinition* destination =
            m_dimensions.find(destinationName);
        const bool destinationCreated = !destination;
        if (!destination)
            destination = &m_dimensions.registerCustom(destinationName,
                                                       DimensionType::Double);

        if (m_current.kind == TokenKind::Eof || isWhere())
            fail("expected value expression after '='");
        PartialExpression value = add(false);
        requireKind(value, ResultKind::Numeric,
                    "assignment value doesn't evaluate to a number");

        CompiledExpression condition;
        if (isWhere())
        {
            advance();
            PartialExpression parsed = logicalOr();
            requireKind(parsed, ResultKind::Boolean,
                        "assignment condition evaluates to a value, not a "
                        "boolean");
            rejectConstantCondition(parsed);
            condition = finish(std::move(parsed));
        }
        expect(TokenKind::Eof, "invalid token following assignment expression");
        return {destination->id, finish(std::move(value)), std::move(condition),
                destinationCreated};
    }

private:
    [[noreturn]] void fail(std::string message) const
    {
        if (!m_current.text.empty())
            message += ": '" + m_current.text + "'";
        throw ExpressionError(std::move(message));
    }

    void advance()
    {
        m_current = m_lexer.next();
        if (m_current.kind == TokenKind::Error)
            fail("invalid expression token");
    }

    bool match(TokenKind kind)
    {
        if (m_current.kind != kind)
            return false;
        advance();
        return true;
    }

    void expect(TokenKind kind, std::string_view message) const
    {
        if (m_current.kind != kind)
            fail(std::string(message));
    }

    void expectAndAdvance(TokenKind kind, std::string_view message)
    {
        expect(kind, message);
        advance();
    }

    bool isWhere() const
    {
        return m_current.kind == TokenKind::Identifier &&
               iequals(m_current.text, "WHERE");
    }

    static void requireKind(const PartialExpression& expression,
                            ResultKind kind, std::string_view message)
    {
        if (expression.kind != kind)
            throw ExpressionError(std::string(message));
    }

    static void rejectConstantCondition(const PartialExpression& expression)
    {
        if (expression.directBooleanConstant)
            throw ExpressionError(*expression.directBooleanConstant
                                      ? "expression is always true"
                                      : "expression is always false");
    }

    static CompiledExpression finish(PartialExpression expression)
    {
        return {std::move(expression.instructions), std::move(expression.reads),
                expression.maximumStackDepth,
                expression.kind == ResultKind::Boolean};
    }

    PartialExpression add(bool conditionalContext)
    {
        PartialExpression left = multiply(conditionalContext);
        while (m_current.kind == TokenKind::Plus ||
               m_current.kind == TokenKind::Dash)
        {
            const TokenKind token = m_current.kind;
            advance();
            PartialExpression right = multiply(conditionalContext);
            requireKind(left, ResultKind::Numeric,
                        "can't apply math operator to logical expression");
            requireKind(right, ResultKind::Numeric,
                        "can't apply math operator to logical expression");
            if (left.numericConstant && right.numericConstant)
            {
                left = numericConstant(
                    token == TokenKind::Plus
                        ? *left.numericConstant + *right.numericConstant
                        : *left.numericConstant - *right.numericConstant);
            }
            else
                left =
                    combine(std::move(left), std::move(right),
                            token == TokenKind::Plus ? ExpressionOp::Add
                                                     : ExpressionOp::Subtract,
                            ResultKind::Numeric);
        }
        return left;
    }

    PartialExpression multiply(bool conditionalContext)
    {
        PartialExpression left = negative(conditionalContext);
        while (m_current.kind == TokenKind::Asterisk ||
               m_current.kind == TokenKind::Slash)
        {
            const TokenKind token = m_current.kind;
            advance();
            PartialExpression right = negative(conditionalContext);
            requireKind(left, ResultKind::Numeric,
                        "can't apply math operator to logical expression");
            requireKind(right, ResultKind::Numeric,
                        "can't apply math operator to logical expression");
            if (left.numericConstant && right.numericConstant)
            {
                if (token == TokenKind::Slash && *right.numericConstant == 0.0)
                    fail("divide by 0");
                left = numericConstant(
                    token == TokenKind::Asterisk
                        ? *left.numericConstant * *right.numericConstant
                        : *left.numericConstant / *right.numericConstant);
            }
            else
                left = combine(std::move(left), std::move(right),
                               token == TokenKind::Asterisk
                                   ? ExpressionOp::Multiply
                                   : ExpressionOp::Divide,
                               ResultKind::Numeric);
        }
        return left;
    }

    PartialExpression negative(bool conditionalContext)
    {
        if (!match(TokenKind::Dash))
            return primary(conditionalContext);
        PartialExpression value = primary(conditionalContext);
        requireKind(value, ResultKind::Numeric,
                    "can't negate logical expression");
        if (value.numericConstant)
            return numericConstant(-*value.numericConstant);
        return unary(std::move(value), ExpressionOp::Negative,
                     ResultKind::Numeric);
    }

    static std::optional<ExpressionOp> mathFunction(std::string_view name)
    {
        static const std::pair<std::string_view, ExpressionOp> Functions[] = {
            {"floor", ExpressionOp::Floor},
            {"ceil", ExpressionOp::Ceil},
            {"round", ExpressionOp::Round},
            {"abs", ExpressionOp::Absolute},
            {"fabs", ExpressionOp::Absolute},
            {"sqrt", ExpressionOp::SquareRoot},
            {"sin", ExpressionOp::Sine},
            {"cos", ExpressionOp::Cosine},
            {"tan", ExpressionOp::Tangent},
            {"asin", ExpressionOp::ArcSine},
            {"acos", ExpressionOp::ArcCosine},
            {"atan", ExpressionOp::ArcTangent},
            {"sinh", ExpressionOp::HyperbolicSine},
            {"cosh", ExpressionOp::HyperbolicCosine},
            {"tanh", ExpressionOp::HyperbolicTangent},
            {"asinh", ExpressionOp::InverseHyperbolicSine},
            {"acosh", ExpressionOp::InverseHyperbolicCosine},
            {"log", ExpressionOp::NaturalLog},
            {"log2", ExpressionOp::Log2},
            {"log10", ExpressionOp::Log10},
            {"exp", ExpressionOp::Exponential},
            {"exp2", ExpressionOp::Exponential2},
        };
        const auto position = std::find_if(
            std::begin(Functions), std::end(Functions),
            [&](const auto& function) { return function.first == name; });
        if (position == std::end(Functions))
            return std::nullopt;
        return position->second;
    }

    PartialExpression primary(bool conditionalContext)
    {
        if (m_current.kind == TokenKind::Number)
        {
            const double value = m_current.number;
            advance();
            return numericConstant(value);
        }
        if (m_current.kind == TokenKind::Identifier)
        {
            const std::string name = m_current.text;
            advance();
            if (name == "nan" || name == "lowest" || name == "highest")
            {
                expectAndAdvance(TokenKind::Lparen,
                                 "expected '(' to open function invocation");
                expectAndAdvance(TokenKind::Rparen,
                                 "expected ')' to close function invocation");
                if (name == "nan")
                    return numericConstant(
                        std::numeric_limits<double>::quiet_NaN());
                if (name == "lowest")
                    return numericConstant(
                        std::numeric_limits<double>::lowest());
                return numericConstant(std::numeric_limits<double>::max());
            }
            if (const std::optional<ExpressionOp> function = mathFunction(name))
            {
                expectAndAdvance(TokenKind::Lparen,
                                 "expected '(' to open function invocation");
                PartialExpression argument = add(conditionalContext);
                requireKind(argument, ResultKind::Numeric,
                            "math function requires a numeric argument");
                expectAndAdvance(TokenKind::Rparen,
                                 "expected ')' following function argument");
                return unary(std::move(argument), *function,
                             ResultKind::Numeric);
            }
            if (m_current.kind == TokenKind::Lparen)
                fail("invalid function name '" + name + "'");
            const DimensionDefinition* definition = m_dimensions.find(name);
            if (!definition)
                fail("unknown dimension '" + name + "' in assignment");
            return dimensionValue(definition->id);
        }
        if (match(TokenKind::Lparen))
        {
            PartialExpression expression =
                conditionalContext ? logicalOr() : add(false);
            expectAndAdvance(TokenKind::Rparen,
                             "expected ')' following expression");
            return expression;
        }
        fail("expected value expression");
    }

    PartialExpression compare()
    {
        PartialExpression left = add(true);
        while (m_current.kind == TokenKind::Equal ||
               m_current.kind == TokenKind::NotEqual ||
               m_current.kind == TokenKind::Greater ||
               m_current.kind == TokenKind::GreaterEqual ||
               m_current.kind == TokenKind::Less ||
               m_current.kind == TokenKind::LessEqual)
        {
            const TokenKind token = m_current.kind;
            advance();
            PartialExpression right = add(true);
            requireKind(left, ResultKind::Numeric,
                        "comparison requires numeric expressions");
            requireKind(right, ResultKind::Numeric,
                        "comparison requires numeric expressions");
            if (left.numericConstant && right.numericConstant)
            {
                bool result = false;
                switch (token)
                {
                case TokenKind::Equal:
                    result = *left.numericConstant == *right.numericConstant;
                    break;
                case TokenKind::NotEqual:
                    result = *left.numericConstant != *right.numericConstant;
                    break;
                case TokenKind::Greater:
                    result = *left.numericConstant > *right.numericConstant;
                    break;
                case TokenKind::GreaterEqual:
                    result = *left.numericConstant >= *right.numericConstant;
                    break;
                case TokenKind::Less:
                    result = *left.numericConstant < *right.numericConstant;
                    break;
                case TokenKind::LessEqual:
                    result = *left.numericConstant <= *right.numericConstant;
                    break;
                default:
                    break;
                }
                left = booleanConstant(result);
            }
            else
            {
                ExpressionOp op = ExpressionOp::Equal;
                switch (token)
                {
                case TokenKind::Equal:
                    op = ExpressionOp::Equal;
                    break;
                case TokenKind::NotEqual:
                    op = ExpressionOp::NotEqual;
                    break;
                case TokenKind::Greater:
                    op = ExpressionOp::Greater;
                    break;
                case TokenKind::GreaterEqual:
                    op = ExpressionOp::GreaterEqual;
                    break;
                case TokenKind::Less:
                    op = ExpressionOp::Less;
                    break;
                case TokenKind::LessEqual:
                    op = ExpressionOp::LessEqual;
                    break;
                default:
                    break;
                }
                left = combine(std::move(left), std::move(right), op,
                               ResultKind::Boolean);
            }
        }
        return left;
    }

    static std::optional<ExpressionOp> booleanFunction(std::string_view name)
    {
        if (name == "isnan")
            return ExpressionOp::IsNan;
        if (name == "ismax")
            return ExpressionOp::IsMaximum;
        if (name == "ismin")
            return ExpressionOp::IsMinimum;
        return std::nullopt;
    }

    PartialExpression primaryLogical()
    {
        if (m_current.kind == TokenKind::Identifier)
        {
            if (const std::optional<ExpressionOp> function =
                    booleanFunction(m_current.text))
            {
                advance();
                expectAndAdvance(TokenKind::Lparen,
                                 "expected '(' to open function invocation");
                PartialExpression argument = add(true);
                requireKind(argument, ResultKind::Numeric,
                            "logical function requires a numeric argument");
                expectAndAdvance(TokenKind::Rparen,
                                 "expected ')' following function argument");
                return unary(std::move(argument), *function,
                             ResultKind::Boolean);
            }
        }
        return compare();
    }

    PartialExpression logicalNot()
    {
        if (!match(TokenKind::Not))
            return primaryLogical();
        PartialExpression expression = primaryLogical();
        requireKind(expression, ResultKind::Boolean,
                    "can't apply '!' to numeric value");
        return unary(std::move(expression), ExpressionOp::LogicalNot,
                     ResultKind::Boolean);
    }

    PartialExpression logicalAnd()
    {
        PartialExpression left = logicalNot();
        while (match(TokenKind::And))
        {
            PartialExpression right = logicalNot();
            requireKind(left, ResultKind::Boolean,
                        "can't apply '&&' to numeric expression");
            requireKind(right, ResultKind::Boolean,
                        "can't apply '&&' to numeric expression");
            left = combine(std::move(left), std::move(right),
                           ExpressionOp::LogicalAnd, ResultKind::Boolean);
        }
        return left;
    }

    PartialExpression logicalOr()
    {
        PartialExpression left = logicalAnd();
        while (match(TokenKind::Or))
        {
            PartialExpression right = logicalAnd();
            requireKind(left, ResultKind::Boolean,
                        "can't apply '||' to numeric expression");
            requireKind(right, ResultKind::Boolean,
                        "can't apply '||' to numeric expression");
            left = combine(std::move(left), std::move(right),
                           ExpressionOp::LogicalOr, ResultKind::Boolean);
        }
        return left;
    }

    Lexer m_lexer;
    DimensionRegistry& m_dimensions;
    Token m_current;
};
} // unnamed namespace

CompiledExpression compileValueExpression(std::string_view text,
                                          DimensionRegistry& dimensions)
{
    return Parser(text, dimensions).valueExpression();
}

CompiledExpression compileConditionalExpression(std::string_view text,
                                                DimensionRegistry& dimensions)
{
    return Parser(text, dimensions).conditionalExpression();
}

AssignProgram compileAssignments(std::span<const std::string> specifications,
                                 DimensionRegistry& dimensions)
{
    AssignProgram program;
    for (const std::string& specification : specifications)
    {
        PointAssignment assignment =
            Parser(specification, dimensions).assignment();
        appendUnique(program.reads, assignment.value.reads);
        appendUnique(program.reads, assignment.condition.reads);
        appendUnique(program.writes, assignment.destination);
        program.assignments.push_back(std::move(assignment));
    }
    return program;
}

void appendAssignments(AssignProgram& destination, const AssignProgram& source)
{
    appendUnique(destination.reads, source.reads);
    appendUnique(destination.writes, source.writes);
    destination.assignments.insert(destination.assignments.end(),
                                   source.assignments.begin(),
                                   source.assignments.end());
}

void appendFerry(AssignProgram& destination, const FerryProgram& source)
{
    for (const FerryCopy& copy : source.copies)
    {
        PointAssignment assignment;
        assignment.destination = copy.destination;
        assignment.destinationCreated = copy.destinationCreated;
        if (copy.hasSource)
        {
            assignment.value.instructions.push_back(
                {ExpressionOp::LoadDimension, copy.source, 0.0});
            assignment.value.reads.push_back(copy.source);
            assignment.value.maximumStackDepth = 1;
            appendUnique(destination.reads, copy.source);
        }
        appendUnique(destination.writes, copy.destination);
        destination.assignments.push_back(std::move(assignment));
    }
}

} // namespace pdg
