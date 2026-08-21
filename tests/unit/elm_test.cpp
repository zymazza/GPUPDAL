#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace
{
struct BatchFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch;

    explicit BatchFixture(std::size_t capacity)
        : batch(capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::X),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Y),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Z),
                          pdg::DimensionType::Double);
        batch.materialize(
            pdg::DimensionId(pdg::StandardDimension::Classification),
            pdg::DimensionType::Unsigned8);
        batch.setSize(capacity);
    }

    void set(std::size_t point, double x, double y, double z,
             std::uint8_t classification = 3U)
    {
        batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X))[point] =
            x;
        batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y))[point] =
            y;
        batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z))[point] =
            z;
        batch.data<std::uint8_t>(pdg::DimensionId(
            pdg::StandardDimension::Classification))[point] = classification;
    }

    [[nodiscard]] const std::uint8_t* classifications() const
    {
        return batch.data<std::uint8_t>(
            pdg::DimensionId(pdg::StandardDimension::Classification));
    }
};
} // unnamed namespace

TEST(Elm, EmptyInputIsAnExactNoop)
{
    BatchFixture fixture(0U);
    const pdg::ElmResult result = pdg::classifyElm(fixture.batch, {});
    EXPECT_EQ(result.rows, 0U);
    EXPECT_EQ(result.columns, 0U);
    EXPECT_EQ(result.classifiedPoints, 0U);
}

TEST(Elm, PreservesThePinnedFractionalCellExpression)
{
    BatchFixture fixture(4U);
    fixture.set(0U, 0.00, 0.0, 0.0);
    fixture.set(1U, 1.24, 0.0, 2.0);
    fixture.set(2U, 1.25, 0.0, 4.0);
    fixture.set(3U, 2.49, 0.0, 8.0);
    pdg::ElmProgram program;
    program.cell = 1.25;
    const pdg::ElmResult result = pdg::classifyElm(fixture.batch, program);

    EXPECT_EQ(result.rows, 1U);
    EXPECT_EQ(result.columns, 2U);
    EXPECT_EQ(result.classifiedPoints, 2U);
    EXPECT_EQ(fixture.classifications()[0], 7U);
    EXPECT_EQ(fixture.classifications()[1], 7U);
    EXPECT_EQ(fixture.classifications()[2], 3U);
    EXPECT_EQ(fixture.classifications()[3], 3U);
}

TEST(Elm, EqualZPointsRetainSourceOrder)
{
    BatchFixture fixture(2U);
    fixture.set(0U, 0.0, 0.0, 1.0, 11U);
    fixture.set(1U, 0.1, 0.0, 1.0, 12U);
    pdg::ElmProgram program;
    program.threshold = -1.0;
    const pdg::ElmResult result = pdg::classifyElm(fixture.batch, program);

    EXPECT_EQ(result.classifiedPoints, 1U);
    EXPECT_EQ(fixture.classifications()[0], 7U);
    EXPECT_EQ(fixture.classifications()[1], 12U);
}

TEST(Elm, ThresholdComparisonIsStrict)
{
    BatchFixture equal(3U);
    equal.set(0U, 0.0, 0.0, 0.0);
    equal.set(1U, 0.1, 0.0, 1.0);
    equal.set(2U, 0.2, 0.0, 3.0);
    pdg::ElmProgram program;
    program.threshold = 1.0;
    EXPECT_EQ(pdg::classifyElm(equal.batch, program).classifiedPoints, 2U);

    BatchFixture above(3U);
    above.set(0U, 0.0, 0.0, 0.0);
    above.set(1U, 0.1, 0.0, 1.0);
    above.set(2U, 0.2, 0.0, 3.0);
    program.threshold = std::nextafter(1.0, 2.0);
    EXPECT_EQ(pdg::classifyElm(above.batch, program).classifiedPoints, 0U);
}

TEST(Elm, UsesTheConfiguredOutputClassWithoutReordering)
{
    BatchFixture fixture(3U);
    fixture.set(0U, 0.0, 0.0, -2.0, 20U);
    fixture.set(1U, 0.1, 0.0, 0.0, 21U);
    fixture.set(2U, 0.2, 0.0, 0.1, 22U);
    const std::array<double, 3> sourceX{0.0, 0.1, 0.2};
    pdg::ElmProgram program;
    program.classification = 18U;
    const pdg::ElmResult result = pdg::classifyElm(fixture.batch, program);

    EXPECT_EQ(result.classifiedPoints, 1U);
    EXPECT_EQ(fixture.classifications()[0], 18U);
    EXPECT_EQ(fixture.classifications()[1], 21U);
    EXPECT_EQ(fixture.classifications()[2], 22U);
    const double* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    for (std::size_t point = 0U; point < sourceX.size(); ++point)
        EXPECT_DOUBLE_EQ(x[point], sourceX[point]);
}

TEST(Elm, ExactDeviceEnvelopeRejectsInvalidProgramsAndFrames)
{
    pdg::ElmProgram program;
    EXPECT_TRUE(pdg::elmProgramWithinExactDeviceEnvelope(program));
    program.cell = 0.0;
    EXPECT_FALSE(pdg::elmProgramWithinExactDeviceEnvelope(program));
    program = {};
    program.threshold = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::elmProgramWithinExactDeviceEnvelope(program));

    BatchFixture nonfinite(2U);
    nonfinite.set(0U, 0.0, 0.0, 0.0);
    nonfinite.set(1U, 1.0, 0.0, std::numeric_limits<double>::infinity());
    EXPECT_FALSE(pdg::elmSupportsExactDevice(nonfinite.batch, {}));

    BatchFixture tooWide(2U);
    tooWide.set(0U, 0.0, 0.0, 0.0);
    tooWide.set(1U, static_cast<double>(pdg::ElmExactDeviceMaximumGridCells),
                0.0, 1.0);
    pdg::ElmProgram unitCell;
    unitCell.cell = 1.0;
    EXPECT_FALSE(pdg::elmSupportsExactDevice(tooWide.batch, unitCell));

    BatchFixture maximumFrame(2U);
    maximumFrame.set(0U, 0.0, 0.0, 0.0);
    maximumFrame.set(
        1U, static_cast<double>(pdg::ElmExactDeviceMaximumGridCells - 1U), 0.0,
        1.0);
    EXPECT_TRUE(pdg::elmSupportsExactDevice(maximumFrame.batch, unitCell));
}
