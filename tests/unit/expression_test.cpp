#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Range.hpp>

#include <pdal/util/Utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace
{

TEST(ExpressionPredicate, HostMaskUsesCompiledConditionalSemantics)
{
    constexpr std::size_t Count = 257;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    const pdg::PredicateProgram predicate = pdg::compilePredicate(
        "(Intensity >= 100 && Intensity < 200) || X == 17", dimensions);
    EXPECT_TRUE(predicate.expression.boolean);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(intensity);
    batch.setSize(Count);
    for (std::size_t index = 0; index < Count; ++index)
    {
        batch.data<double>(x)[index] = static_cast<double>(index);
        batch.data<std::uint16_t>(intensity)[index] =
            static_cast<std::uint16_t>(index);
    }
    std::vector<std::uint8_t> keep(Count, 0xffU);
    pdg::evaluatePredicate(batch, predicate, keep.data(), 3);
    for (std::size_t index = 0; index < Count; ++index)
        EXPECT_EQ(keep[index],
                  static_cast<std::uint8_t>((index >= 100U && index < 200U) ||
                                            index == 17U))
            << index;
}

TEST(RangePredicate, MatchesPdalGroupedBoundsNegationAndNanSemantics)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const pdg::DimensionId other =
        dimensions.registerCustom("Other", pdg::DimensionType::Unsigned16).id;
    const std::vector<std::string> limits = {"Source(0:2], Source![4:5]",
                                             "Other[10:20)"};
    const pdg::PredicateProgram predicate =
        pdg::compileRangePredicate(limits, dimensions);
    EXPECT_TRUE(predicate.expression.boolean);
    EXPECT_EQ(predicate.reads.size(), 2U);

    const std::array<double, 9> values = {
        std::numeric_limits<double>::quiet_NaN(),
        -1.0,
        0.0,
        0.5,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0};
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        values.size(),
        pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}), dimensions,
        memory);
    batch.materialize(source);
    batch.materialize(other);
    batch.setSize(values.size());
    std::copy(values.begin(), values.end(), batch.data<double>(source));
    std::fill_n(batch.data<std::uint16_t>(other), values.size(), 10U);

    std::vector<std::uint8_t> keep(values.size());
    pdg::evaluatePredicate(batch, predicate, keep.data(), 1);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const double value = values[index];
        const bool sourcePass = (value > 0.0 && value <= 2.0) ||
                                std::isnan(value) || value < 4.0 || value > 5.0;
        EXPECT_EQ(keep[index], static_cast<std::uint8_t>(sourcePass)) << index;
    }

    EXPECT_THROW(static_cast<void>(pdg::compileRangePredicate(
                     std::array<std::string, 1>{"Missing[0:1]"}, dimensions)),
                 pdg::ExpressionError);
}
pdg::CoordinateEncoding expressionCoordinates()
{
    return {{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}};
}

pdg::AssignProgram compile(std::initializer_list<std::string> values,
                           pdg::DimensionRegistry& dimensions)
{
    const std::vector<std::string> specifications(values);
    return pdg::compileAssignments(specifications, dimensions);
}
} // unnamed namespace

TEST(ExpressionVm, PreservesOrderedAssignmentAndConditionSemantics)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto middle =
        dimensions.registerCustom("Middle", pdg::DimensionType::Signed16).id;
    const auto flag =
        dimensions.registerCustom("AssignFlag", pdg::DimensionType::Unsigned8)
            .id;
    const pdg::AssignProgram program =
        compile({"Middle = round(Source * 2 + 0.5) WHERE "
                 "Source >= -0.5 && Source < 4",
                 "Source = Middle / 2", "AssignFlag = 9 WHERE Source == 0.5"},
                dimensions);

    EXPECT_EQ(program.assignments.size(), 3U);
    EXPECT_NE(std::find(program.reads.begin(), program.reads.end(), source),
              program.reads.end());
    EXPECT_NE(std::find(program.writes.begin(), program.writes.end(), middle),
              program.writes.end());

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(5, expressionCoordinates(), dimensions, memory);
    for (pdg::DimensionId id : {source, middle, flag})
        batch.materialize(id);
    batch.setSize(5);
    const std::array<double, 5> input = {-2.0, -0.5, 0.0, 2.0, 4.0};
    std::copy(input.begin(), input.end(), batch.data<double>(source));
    std::fill_n(batch.data<std::int16_t>(middle), 5, 7);
    std::fill_n(batch.data<std::uint8_t>(flag), 5, 0);

    pdg::executeAssign(batch, program, 1);

    const std::array<std::int16_t, 5> expectedMiddle = {7, -1, 1, 5, 7};
    const std::array<double, 5> expectedSource = {3.5, -0.5, 0.5, 2.5, 3.5};
    const std::array<std::uint8_t, 5> expectedFlag = {0, 0, 9, 0, 0};
    EXPECT_TRUE(std::equal(expectedMiddle.begin(), expectedMiddle.end(),
                           batch.data<std::int16_t>(middle)));
    EXPECT_TRUE(std::equal(expectedSource.begin(), expectedSource.end(),
                           batch.data<double>(source)));
    EXPECT_TRUE(std::equal(expectedFlag.begin(), expectedFlag.end(),
                           batch.data<std::uint8_t>(flag)));
}

TEST(ExpressionVm, MatchesPdalMathAndLogicalFunctionFamilies)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto result =
        dimensions.registerCustom("Result", pdg::DimensionType::Double).id;
    const auto flag =
        dimensions.registerCustom("AssignFlag", pdg::DimensionType::Unsigned8)
            .id;
    const pdg::AssignProgram program =
        compile({"Result = floor(Source) + ceil(Source) + round(Source) + "
                 "abs(Source) + fabs(Source) + sqrt(Source) + sin(Source) + "
                 "cos(Source) + tan(Source) + asin(Source) + acos(Source) + "
                 "atan(Source) + sinh(Source) + cosh(Source) + tanh(Source) + "
                 "asinh(Source) + acosh(Source + 1) + log(Source) + "
                 "log2(Source) + log10(Source) + exp(Source) + exp2(Source)",
                 "AssignFlag = 1 WHERE ismin(Source)",
                 "AssignFlag = 2 WHERE ismax(Source)",
                 "AssignFlag = 3 WHERE isnan(Source)"},
                dimensions);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(4, expressionCoordinates(), dimensions, memory);
    for (pdg::DimensionId id : {source, result, flag})
        batch.materialize(id);
    batch.setSize(4);
    const double low = std::numeric_limits<double>::lowest();
    const double high = std::numeric_limits<double>::max();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<double, 4> input = {0.5, low, high, nan};
    std::copy(input.begin(), input.end(), batch.data<double>(source));
    std::fill_n(batch.data<double>(result), 4, 0.0);
    std::fill_n(batch.data<std::uint8_t>(flag), 4, 0);

    pdg::executeAssign(batch, program, 1);

    const double value = 0.5;
    const double expected =
        std::floor(value) + std::ceil(value) + std::round(value) +
        std::fabs(value) + std::fabs(value) + std::sqrt(value) +
        std::sin(value) + std::cos(value) + std::tan(value) + std::asin(value) +
        std::acos(value) + std::atan(value) + std::sinh(value) +
        std::cosh(value) + std::tanh(value) + std::asinh(value) +
        std::acosh(value + 1.0) + std::log(value) + std::log2(value) +
        std::log10(value) + std::exp(value) + std::exp2(value);
    EXPECT_DOUBLE_EQ(batch.data<double>(result)[0], expected);
    EXPECT_EQ(batch.data<std::uint8_t>(flag)[0], 0);
    EXPECT_EQ(batch.data<std::uint8_t>(flag)[1], 1);
    EXPECT_EQ(batch.data<std::uint8_t>(flag)[2], 2);
    EXPECT_EQ(batch.data<std::uint8_t>(flag)[3], 3);
}

TEST(ExpressionVm, PreservesDestinationWhenPdalNumericCastFails)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto destination =
        dimensions.registerCustom("Destination", pdg::DimensionType::Unsigned8)
            .id;
    const pdg::AssignProgram program =
        compile({"Destination = Source"}, dimensions);
    const std::array<double, 7> input = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -1.0,
        0.49,
        0.5,
        255.49,
        255.5};
    std::array<std::uint8_t, input.size()> expected;
    expected.fill(42);
    for (std::size_t index = 0; index < input.size(); ++index)
        static_cast<void>(
            pdal::Utils::numericCast(input[index], expected[index]));

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(input.size(), expressionCoordinates(), dimensions,
                          memory);
    batch.materialize(source);
    batch.materialize(destination);
    batch.setSize(input.size());
    std::copy(input.begin(), input.end(), batch.data<double>(source));
    std::fill_n(batch.data<std::uint8_t>(destination), input.size(), 42);

    pdg::executeAssign(batch, program, 1);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                           batch.data<std::uint8_t>(destination)));
}

TEST(ExpressionVm, InitializesAndReusesNewDimensionsInProgramOrder)
{
    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program =
        compile({"Scratch = Scratch + 1", "Result = Scratch * 2"}, dimensions);
    const pdg::DimensionId scratch = dimensions.require("Scratch").id;
    const pdg::DimensionId result = dimensions.require("Result").id;
    ASSERT_TRUE(program.assignments[0].destinationCreated);
    ASSERT_TRUE(program.assignments[1].destinationCreated);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(8, expressionCoordinates(), dimensions, memory);
    batch.setSize(8);
    pdg::executeAssign(batch, program, 1);

    ASSERT_TRUE(batch.has(scratch));
    ASSERT_TRUE(batch.has(result));
    EXPECT_TRUE(std::all_of(batch.data<double>(scratch),
                            batch.data<double>(scratch) + batch.size(),
                            [](double value) { return value == 1.0; }));
    EXPECT_TRUE(std::all_of(batch.data<double>(result),
                            batch.data<double>(result) + batch.size(),
                            [](double value) { return value == 2.0; }));
}

TEST(ExpressionVm, SupportsLogicalAndEncodedCoordinateColumns)
{
    pdg::DimensionRegistry dimensions;
    const auto input =
        dimensions.registerCustom("Input", pdg::DimensionType::Double).id;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    const pdg::AssignProgram program =
        compile({"X = Input", "Z = X + 0.01 WHERE !(X < 0)"}, dimensions);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(3, expressionCoordinates(), dimensions, memory);
    batch.materialize(input);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(z);
    batch.setSize(3);
    const std::array<double, 3> values = {-1.005, 0.004, 1.015};
    std::copy(values.begin(), values.end(), batch.data<double>(input));
    std::fill_n(batch.data<double>(x), 3, 0.0);
    std::fill_n(batch.data<std::int32_t>(z), 3, -99);

    pdg::executeAssign(batch, program, 1);

    EXPECT_TRUE(
        std::equal(values.begin(), values.end(), batch.data<double>(x)));
    const std::array<std::int32_t, 3> expectedZ = {
        -99, expressionCoordinates().encode(2, values[1] + 0.01),
        expressionCoordinates().encode(2, values[2] + 0.01)};
    EXPECT_TRUE(std::equal(expectedZ.begin(), expectedZ.end(),
                           batch.data<std::int32_t>(z)));
}

TEST(ExpressionVm, IsBitIdenticalAcrossHostWorkerCounts)
{
    constexpr std::size_t Count = 131073;
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto destination =
        dimensions.registerCustom("Destination", pdg::DimensionType::Signed32)
            .id;
    const pdg::AssignProgram program = compile(
        {"Destination = round(Source * 7 - 3) WHERE Source != 17"}, dimensions);
    pdg::HostMemoryResource memory;
    pdg::PointBatch serial(Count, expressionCoordinates(), dimensions, memory);
    pdg::PointBatch parallel(Count, expressionCoordinates(), dimensions,
                             memory);
    for (pdg::PointBatch* batch : {&serial, &parallel})
    {
        batch->materialize(source);
        batch->materialize(destination);
        batch->setSize(Count);
        for (std::size_t index = 0; index < Count; ++index)
        {
            batch->data<double>(source)[index] =
                static_cast<double>((index * 7919U) % 100000U) / 13.0;
            batch->data<std::int32_t>(destination)[index] = -123;
        }
    }

    pdg::executeAssign(serial, program, 1);
    pdg::executeAssign(parallel, program, 7);
    EXPECT_EQ(std::memcmp(serial.rawData(destination),
                          parallel.rawData(destination),
                          Count * sizeof(std::int32_t)),
              0);
}

TEST(ExpressionCompiler, RejectsPinnedGrammarAndPreparationErrors)
{
    const std::vector<std::string> invalid = {
        "",
        "X",
        "X =",
        "X = .5",
        "X = 1 / 0",
        "X = Missing",
        "X = 1 FOO",
        "X = sqrt",
        "X = bogus(1)",
        "X = 1 WHERE X",
        "X = 1 WHERE 1 == 1",
        "X = 1 WHERE 1 == 2",
        "X = 1 WHERE !1",
        "X = 1 WHERE X > 0 && X",
        "X = 1 WHERE --X > 0",
    };
    for (const std::string& specification : invalid)
    {
        pdg::DimensionRegistry dimensions;
        EXPECT_THROW(static_cast<void>(compile({specification}, dimensions)),
                     pdg::ExpressionError)
            << specification;
    }
}

TEST(ExpressionCompiler, LowersFerryIntoTheSameOrderedPointProgram)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto middle =
        dimensions.registerCustom("Middle", pdg::DimensionType::Signed16).id;
    const auto result =
        dimensions.registerCustom("Result", pdg::DimensionType::Double).id;
    pdg::AssignProgram program;
    pdg::appendFerry(program, {{{true, source, middle, false},
                                {true, middle, result, false}}});
    ASSERT_EQ(program.assignments.size(), 2U);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(3, expressionCoordinates(), dimensions, memory);
    for (pdg::DimensionId id : {source, middle, result})
        batch.materialize(id);
    batch.setSize(3);
    const std::array<double, 3> values = {-1.5, 0.5, 2.4};
    std::copy(values.begin(), values.end(), batch.data<double>(source));
    pdg::executeAssign(batch, program, 1);
    const std::array<double, 3> expected = {-2.0, 1.0, 2.0};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                           batch.data<double>(result)));
}
