#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/Expression.hpp>

#include <string>
#include <vector>

namespace pdg
{

struct InfoProgram
{
    std::string pointSpec;
    std::string querySpec;
};

struct ExpressionStatsProgram
{
    DimensionId dimension;
    std::vector<CompiledExpression> expressions;
};

} // namespace pdg
