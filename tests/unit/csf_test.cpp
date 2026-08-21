#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

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
};

pdg::CsfProgram boundedProgram()
{
    pdg::CsfProgram program;
    program.smooth = false;
    program.timeStep = 1.0;
    program.iterations = 3;
    return program;
}

void fillCompactObject(BatchFixture& fixture, std::uint8_t initialClass)
{
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const std::size_t point = column * 5U + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            z[point] = point == 12U ? 5.0 : 0.0;
            classification[point] = initialClass;
        }
}
} // unnamed namespace

TEST(Csf, ClassifiesACompactObjectWithoutChangingPointOrder)
{
    BatchFixture fixture(25U);
    fillCompactObject(fixture, 7U);
    const pdg::CsfResult result =
        pdg::classifyCsf(fixture.batch, boundedProgram());
    const auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));

    EXPECT_EQ(result.width, 8U);
    EXPECT_EQ(result.height, 8U);
    EXPECT_EQ(result.groundPoints, 24U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 12U ? 1U : 2U);
}

TEST(Csf, OnlyGroundPreservesRejectedClassifications)
{
    BatchFixture fixture(25U);
    fillCompactObject(fixture, 12U);
    pdg::CsfProgram program = boundedProgram();
    program.groundClass = 9U;
    program.otherClass = 9U;
    program.onlyGround = true;
    const pdg::CsfResult result = pdg::classifyCsf(fixture.batch, program);
    const auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));

    EXPECT_EQ(result.groundPoints, 24U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 12U ? 12U : 9U);
}

TEST(Csf, ExactDeviceEnvelopeRejectsSmoothingAndUnboundedPrograms)
{
    pdg::CsfProgram program;
    EXPECT_FALSE(pdg::csfProgramWithinExactDeviceEnvelope(program));

    program.smooth = false;
    EXPECT_FALSE(pdg::csfProgramWithinExactDeviceEnvelope(program));

    program.iterations = 32;
    EXPECT_EQ(pdg::csfProgramWithinExactDeviceEnvelope(program),
              pdg::CsfPinnedOracleHasSerialExecution);

    program.resolution = 0.0;
    EXPECT_FALSE(pdg::csfProgramWithinExactDeviceEnvelope(program));

    program = boundedProgram();
    program.iterations = 65;
    EXPECT_FALSE(pdg::csfProgramWithinExactDeviceEnvelope(program));

    program = boundedProgram();
    program.rigidness = -1;
    EXPECT_FALSE(pdg::csfProgramWithinExactDeviceEnvelope(program));
}
