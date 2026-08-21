#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace pdg
{

class ExpressionError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum class ExpressionOp : std::uint8_t
{
    PushConstant,
    PushFalse,
    PushTrue,
    LoadDimension,
    Add,
    Subtract,
    Multiply,
    Divide,
    Negative,
    Floor,
    Ceil,
    Round,
    Absolute,
    SquareRoot,
    Sine,
    Cosine,
    Tangent,
    ArcSine,
    ArcCosine,
    ArcTangent,
    HyperbolicSine,
    HyperbolicCosine,
    HyperbolicTangent,
    InverseHyperbolicSine,
    InverseHyperbolicCosine,
    NaturalLog,
    Log2,
    Log10,
    Exponential,
    Exponential2,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    LogicalNot,
    LogicalAnd,
    LogicalOr,
    IsNan,
    IsMaximum,
    IsMinimum
};

struct ExpressionInstruction
{
    ExpressionOp op = ExpressionOp::PushConstant;
    DimensionId dimension;
    double immediate = 0.0;
};

struct CompiledExpression
{
    std::vector<ExpressionInstruction> instructions;
    std::vector<DimensionId> reads;
    std::size_t maximumStackDepth = 0;
    bool boolean = false;

    [[nodiscard]] bool empty() const noexcept
    {
        return instructions.empty();
    }
};

[[nodiscard]] CompiledExpression
compileValueExpression(std::string_view text, DimensionRegistry& dimensions);

[[nodiscard]] CompiledExpression
compileConditionalExpression(std::string_view text,
                             DimensionRegistry& dimensions);

} // namespace pdg
