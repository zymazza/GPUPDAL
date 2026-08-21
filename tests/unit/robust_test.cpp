#include <pdg/PointBatch.hpp>
#include <pdg/stages/Robust.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{
pdg::CoordinateEncoding identityCoordinates()
{
    return pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
}

TEST(RobustStatistics, MatchesIqrOrderStatisticsAndStrictFences)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionDefinition& value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(8, identityCoordinates(), dimensions, memory);
    batch.materialize(value.id, pdg::DimensionType::Double);
    batch.setSize(8);
    const std::vector<double> source = {1, 2, 3, 4, 5, 6, 100, 101};
    std::copy(source.begin(), source.end(),
              batch.hostSpan<double>(value.id).begin());

    pdg::RobustProgram program;
    program.dimension = value.id;
    program.kind = pdg::RobustKind::Iqr;
    program.multiplier = 0.0;
    std::vector<std::uint8_t> keep(source.size());
    const pdg::RobustResult result =
        pdg::evaluateRobust(batch, program, keep.data());
    EXPECT_DOUBLE_EQ(result.first, 3.0);
    EXPECT_DOUBLE_EQ(result.second, 100.0);
    EXPECT_DOUBLE_EQ(result.scale, 97.0);
    EXPECT_EQ(keep, std::vector<std::uint8_t>({0, 0, 0, 1, 1, 1, 0, 0}));
}

TEST(RobustStatistics, MatchesMadDeviationDivisionAndMultiplier)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(5, identityCoordinates(), dimensions, memory);
    batch.materialize(intensity, pdg::DimensionType::Unsigned16);
    batch.setSize(5);
    const std::vector<std::uint16_t> source = {1, 2, 3, 4, 100};
    std::copy(source.begin(), source.end(),
              batch.hostSpan<std::uint16_t>(intensity).begin());

    pdg::RobustProgram program;
    program.dimension = intensity;
    program.kind = pdg::RobustKind::Mad;
    program.multiplier = 2.0;
    program.madMultiplier = 1.4862;
    std::vector<std::uint8_t> keep(source.size());
    const pdg::RobustResult result =
        pdg::evaluateRobust(batch, program, keep.data());
    EXPECT_DOUBLE_EQ(result.first, 3.0);
    EXPECT_DOUBLE_EQ(result.scale, 1.4862);
    EXPECT_EQ(keep, std::vector<std::uint8_t>({1, 1, 1, 1, 0}));
}

TEST(RobustStatistics, PreservesEncodedCoordinateConversion)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        5, pdg::CoordinateEncoding({0.5, 1.0, 1.0}, {10.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(x, pdg::DimensionType::Signed32);
    batch.setSize(5);
    const std::vector<std::int32_t> source = {0, 2, 4, 6, 100};
    std::copy(source.begin(), source.end(),
              batch.hostSpan<std::int32_t>(x).begin());
    pdg::RobustProgram program;
    program.dimension = x;
    program.kind = pdg::RobustKind::Mad;
    std::vector<std::uint8_t> keep(source.size());
    const pdg::RobustResult result =
        pdg::evaluateRobust(batch, program, keep.data());
    EXPECT_DOUBLE_EQ(result.first, 12.0);
    EXPECT_EQ(keep, std::vector<std::uint8_t>({1, 1, 1, 1, 0}));
}

TEST(RobustStatistics, DeviceEnvelopeRejectsNonfiniteAndNegativeZeroKeys)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionDefinition& value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(3, identityCoordinates(), dimensions, memory);
    batch.materialize(value.id, pdg::DimensionType::Double);
    batch.setSize(3);
    auto values = batch.hostSpan<double>(value.id);
    values[0] = 1.0;
    values[1] = 2.0;
    values[2] = 3.0;
    pdg::RobustProgram program;
    program.dimension = value.id;
    EXPECT_TRUE(pdg::robustSupportsExactDevice(batch, program));
    values[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::robustSupportsExactDevice(batch, program));
    values[1] = -0.0;
    EXPECT_TRUE(std::bit_cast<std::uint64_t>(values[1]) != 0U);
    EXPECT_FALSE(pdg::robustSupportsExactDevice(batch, program));
}
} // unnamed namespace
