#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

namespace
{
pdg::CoordinateEncoding coordinates()
{
    return {{0.001, 0.001, 0.001}, {500000.0, 4800000.0, 100.0}};
}
} // unnamed namespace

TEST(PointBatch, MaterializesOnlyTouchedSoAColumns)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(16, coordinates(), dimensions, memory);

    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    EXPECT_FALSE(batch.has(x));
    EXPECT_EQ(batch.allocatedBytes(), 0U);

    const auto& xInfo = batch.materialize(x);
    EXPECT_EQ(xInfo.logicalType, pdg::DimensionType::Double);
    EXPECT_EQ(xInfo.physicalType, pdg::DimensionType::Signed32);
    EXPECT_EQ(batch.allocatedBytes(), 16U * sizeof(std::int32_t));

    const auto& intensityInfo = batch.materialize(intensity);
    EXPECT_EQ(intensityInfo.physicalType, pdg::DimensionType::Unsigned16);
    EXPECT_EQ(batch.allocatedBytes(),
              16U * (sizeof(std::int32_t) + sizeof(std::uint16_t)));
    EXPECT_TRUE(batch.has("INTENSITY"));
}

TEST(PointBatch, EnforcesCapacityTypeAndGhostInvariants)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(8, coordinates(), dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    batch.materialize(x);
    batch.setSize(3);

    auto values = batch.hostSpan<std::int32_t>(x);
    ASSERT_EQ(values.size(), 3U);
    values[0] = 42;
    EXPECT_EQ(batch.data<std::int32_t>(x)[0], 42);
    EXPECT_THROW(static_cast<void>(batch.data<double>(x)),
                 std::invalid_argument);
    EXPECT_THROW(batch.setSize(9), std::out_of_range);
    EXPECT_THROW(static_cast<void>(batch.ghostData()), std::logic_error);

    batch.materializeGhostMask();
    batch.ghostData()[0] = 1;
    EXPECT_TRUE(batch.hasGhostMask());
    EXPECT_EQ(batch.allocatedBytes(),
              8U * (sizeof(std::int32_t) + sizeof(std::uint8_t)));
}
