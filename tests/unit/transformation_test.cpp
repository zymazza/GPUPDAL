#include <pdg/PointBatch.hpp>
#include <pdg/stages/Transformation.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{
pdg::CoordinateEncoding identityCoordinates()
{
    return pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
}

TEST(Transformation, ParsesExactlySixteenClassicLocaleEntries)
{
    const pdg::TransformationProgram program =
        pdg::compileTransformation("1 2 3 4\n5 6 7 8\n9 10 11 12\n13 14 15 16");
    for (std::size_t index = 0; index < program.matrix.size(); ++index)
        EXPECT_DOUBLE_EQ(program.matrix[index],
                         static_cast<double>(index + 1U));

    EXPECT_THROW(static_cast<void>(pdg::compileTransformation(
                     "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0")),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::compileTransformation(
                     "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 9")),
                 std::invalid_argument);
}

TEST(Transformation, UsesOriginalCoordinatesAndFullHomogeneousDivision)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(2, identityCoordinates(), dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.materialize(z, pdg::DimensionType::Double);
    batch.setSize(2);
    batch.hostSpan<double>(x)[0] = 2.0;
    batch.hostSpan<double>(y)[0] = 3.0;
    batch.hostSpan<double>(z)[0] = 4.0;
    batch.hostSpan<double>(x)[1] = -5.0;
    batch.hostSpan<double>(y)[1] = 7.0;
    batch.hostSpan<double>(z)[1] = 11.0;

    const pdg::TransformationProgram program = pdg::compileTransformation(
        "2 3 5 7 11 13 17 19 23 29 31 37 0.5 0.25 0 1");
    pdg::executeTransformation(batch, program);

    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(x)[0], 40.0 / 2.75);
    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(y)[0], 148.0 / 2.75);
    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(z)[0], 294.0 / 2.75);
    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(x)[1], 73.0 / 0.25);
    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(y)[1], 242.0 / 0.25);
    EXPECT_DOUBLE_EQ(batch.hostSpan<double>(z)[1], 466.0 / 0.25);
}

TEST(Transformation, QualifiesOnlyAffineDoubleCoordinateBatchesForCuda)
{
    const pdg::TransformationProgram affine =
        pdg::compileTransformation("1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1");
    const pdg::TransformationProgram projective =
        pdg::compileTransformation("1 0 0 4 0 1 0 5 0 0 1 6 0.01 0 0 1");
    EXPECT_TRUE(pdg::transformationSupportsExactDevice(affine));
    EXPECT_FALSE(pdg::transformationSupportsExactDevice(projective));

    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch doubles(1, identityCoordinates(), dimensions, memory);
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::X, pdg::StandardDimension::Y,
          pdg::StandardDimension::Z})
        doubles.materialize(pdg::DimensionId(dimension),
                            pdg::DimensionType::Double);
    EXPECT_TRUE(pdg::transformationSupportsExactDevice(doubles, affine));
    EXPECT_FALSE(pdg::transformationSupportsExactDevice(doubles, projective));

    pdg::PointBatch packed(1, identityCoordinates(), dimensions, memory);
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::X, pdg::StandardDimension::Y,
          pdg::StandardDimension::Z})
        packed.materialize(pdg::DimensionId(dimension),
                           pdg::DimensionType::Signed32);
    EXPECT_FALSE(pdg::transformationSupportsExactDevice(packed, affine));
    EXPECT_THROW(pdg::executeTransformation(packed, affine),
                 std::invalid_argument);
}
} // unnamed namespace
