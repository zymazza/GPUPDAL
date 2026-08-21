#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{
constexpr std::size_t Count = 6;

struct ColorFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::DimensionId value;
    pdg::PointBatch batch;

    ColorFixture()
        : value(dimensions
                    .registerCustom("PdgColorValue", pdg::DimensionType::Double)
                    .id),
          batch(Count,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        batch.materialize(value, pdg::DimensionType::Double);
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
            batch.materialize(pdg::DimensionId(dimension),
                              pdg::DimensionType::Unsigned16);
        batch.setSize(Count);
        const std::array<double, Count> values = {0.0,  24.999, 25.0,
                                                  99.0, 100.0,  -1.0};
        std::copy(values.begin(), values.end(), batch.data<double>(value));
        resetColors();
    }

    void resetColors()
    {
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
            std::fill_n(batch.data<std::uint16_t>(pdg::DimensionId(dimension)),
                        Count, std::uint16_t{999});
    }
};

constexpr std::array<std::uint8_t, 4> RampRed = {1, 2, 3, 4};
constexpr std::array<std::uint8_t, 4> RampGreen = {11, 12, 13, 14};
constexpr std::array<std::uint8_t, 4> RampBlue = {21, 22, 23, 24};

pdg::ColorRampView ramp()
{
    return {RampRed.data(), RampGreen.data(), RampBlue.data(), RampRed.size()};
}

pdg::ColorMapProgram program(const ColorFixture& fixture)
{
    pdg::ColorMapProgram result;
    result.value = fixture.value;
    result.minimum = 0.0;
    result.maximum = 100.0;
    return result;
}

TEST(ColorMap, MapsExactBinsAndPreservesUnclampedOutsideColors)
{
    ColorFixture fixture;
    const pdg::ColorMapProgram map = program(fixture);
    ASSERT_TRUE(pdg::colorMapMaySupportExactDevice(fixture.batch, map, ramp()));
    pdg::applyColorMap(fixture.batch, map, ramp());

    const std::array<std::uint16_t, Count> expectedRed = {1, 1, 2, 4, 4, 999};
    const std::array<std::uint16_t, Count> expectedGreen = {11, 11, 12,
                                                            14, 14, 999};
    const std::array<std::uint16_t, Count> expectedBlue = {21, 21, 22,
                                                           24, 24, 999};
    EXPECT_TRUE(std::equal(expectedRed.begin(), expectedRed.end(),
                           fixture.batch.data<std::uint16_t>(
                               pdg::DimensionId(pdg::StandardDimension::Red))));
    EXPECT_TRUE(std::equal(expectedGreen.begin(), expectedGreen.end(),
                           fixture.batch.data<std::uint16_t>(pdg::DimensionId(
                               pdg::StandardDimension::Green))));
    EXPECT_TRUE(std::equal(expectedBlue.begin(), expectedBlue.end(),
                           fixture.batch.data<std::uint16_t>(pdg::DimensionId(
                               pdg::StandardDimension::Blue))));
}

TEST(ColorMap, ClampsAndInvertsUsingUpstreamRampIndexing)
{
    ColorFixture fixture;
    pdg::ColorMapProgram map = program(fixture);
    map.clamp = true;
    pdg::applyColorMap(fixture.batch, map, ramp());
    EXPECT_EQ(fixture.batch.data<std::uint16_t>(
                  pdg::DimensionId(pdg::StandardDimension::Red))[Count - 1U],
              1U);

    fixture.resetColors();
    map.clamp = false;
    map.invert = true;
    pdg::applyColorMap(fixture.batch, map, ramp());
    const std::array<std::uint16_t, Count> expected = {4, 4, 3, 1, 1, 999};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                           fixture.batch.data<std::uint16_t>(
                               pdg::DimensionId(pdg::StandardDimension::Red))));
}

TEST(ColorMap, ValidatesEnvelopeAndRejectsNonfiniteCudaInputs)
{
    ColorFixture fixture;
    pdg::ColorMapProgram map = program(fixture);
    EXPECT_THROW(pdg::applyColorMap(fixture.batch, map, {}),
                 std::invalid_argument);

    map.maximum = map.minimum;
    EXPECT_FALSE(
        pdg::colorMapMaySupportExactDevice(fixture.batch, map, ramp()));
    EXPECT_THROW(pdg::applyColorMap(fixture.batch, map, ramp()),
                 std::invalid_argument);

    map = program(fixture);
    fixture.batch.data<double>(fixture.value)[0] =
        (std::numeric_limits<double>::quiet_NaN)();
    EXPECT_FALSE(
        pdg::colorMapMaySupportExactDevice(fixture.batch, map, ramp()));

    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch missing(
        1, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    missing.setSize(1);
    EXPECT_FALSE(pdg::colorMapMaySupportExactDevice(missing, map, ramp()));
    EXPECT_THROW(pdg::applyColorMap(missing, map, ramp()),
                 std::invalid_argument);
}
} // unnamed namespace
