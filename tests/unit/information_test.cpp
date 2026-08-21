#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>

namespace
{
std::uint64_t bits(double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

TEST(Information, PreservesOrderedBoundsSpecialValuesAndCount)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::PointBatch batch(4, coordinates, dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.materialize(z, pdg::DimensionType::Double);
    batch.setSize(4);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double highest = (std::numeric_limits<double>::max)();
    const double lowest = (std::numeric_limits<double>::lowest)();
    const double infinity = std::numeric_limits<double>::infinity();
    auto xs = batch.hostSpan<double>(x);
    auto ys = batch.hostSpan<double>(y);
    auto zs = batch.hostSpan<double>(z);
    xs[0] = -0.0;
    xs[1] = +0.0;
    xs[2] = nan;
    xs[3] = highest;
    ys[0] = nan;
    ys[1] = 2.0;
    ys[2] = -3.0;
    ys[3] = highest;
    zs[0] = lowest;
    zs[1] = -infinity;
    zs[2] = infinity;
    zs[3] = 5.0;

    const pdg::BoundsResult result = pdg::summarizeBounds(batch, 40);
    EXPECT_EQ(result.count, 4U);
    EXPECT_EQ(bits(result.minimum[0].value), bits(-0.0));
    EXPECT_EQ(result.minimum[0].index, 40U);
    EXPECT_DOUBLE_EQ(result.maximum[0].value, highest);
    EXPECT_EQ(result.maximum[0].index, 43U);
    EXPECT_DOUBLE_EQ(result.minimum[1].value, -3.0);
    EXPECT_DOUBLE_EQ(result.maximum[1].value, highest);
    EXPECT_DOUBLE_EQ(result.minimum[2].value, -infinity);
    EXPECT_DOUBLE_EQ(result.maximum[2].value, infinity);
    EXPECT_EQ(result.minimum[2].index, 41U);
    EXPECT_EQ(result.maximum[2].index, 42U);
}

TEST(Information, MergesChunksInGlobalOrderAndDecodesCoordinates)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    const pdg::CoordinateEncoding coordinates({0.25, 0.5, 2.0},
                                              {100.0, -50.0, 7.0});
    pdg::PointBatch batch(2, coordinates, dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    for (pdg::DimensionId dimension : {x, y, z})
        batch.materialize(dimension, pdg::DimensionType::Signed32);
    batch.setSize(1);

    batch.hostSpan<std::int32_t>(x)[0] = -4;
    batch.hostSpan<std::int32_t>(y)[0] = 10;
    batch.hostSpan<std::int32_t>(z)[0] = 3;
    const pdg::BoundsResult first = pdg::summarizeBounds(batch, 10);
    batch.hostSpan<std::int32_t>(x)[0] = -4;
    batch.hostSpan<std::int32_t>(y)[0] = -10;
    batch.hostSpan<std::int32_t>(z)[0] = 9;
    const pdg::BoundsResult second = pdg::summarizeBounds(batch, 11);
    const pdg::BoundsResult merged = pdg::mergeBoundsResults(first, second);

    EXPECT_EQ(merged.count, 2U);
    EXPECT_DOUBLE_EQ(merged.minimum[0].value, 99.0);
    EXPECT_DOUBLE_EQ(merged.maximum[0].value, 99.0);
    EXPECT_EQ(merged.minimum[0].index, 10U);
    EXPECT_EQ(merged.maximum[0].index, 10U);
    EXPECT_DOUBLE_EQ(merged.minimum[1].value, -55.0);
    EXPECT_DOUBLE_EQ(merged.maximum[1].value, -45.0);
    EXPECT_DOUBLE_EQ(merged.minimum[2].value, 13.0);
    EXPECT_DOUBLE_EQ(merged.maximum[2].value, 25.0);

    batch.setSize(0);
    const pdg::BoundsResult empty = pdg::summarizeBounds(batch, 12);
    EXPECT_EQ(empty.count, 0U);
    EXPECT_EQ(pdg::mergeBoundsResults(merged, empty).count, 2U);
}
} // unnamed namespace
