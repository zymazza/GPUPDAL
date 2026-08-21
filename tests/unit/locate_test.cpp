#include <pdg/PointBatch.hpp>
#include <pdg/stages/Locate.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace
{
pdg::CoordinateEncoding identityCoordinates()
{
    return pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
}

TEST(Locate, RetainsFirstTieAndMergesChunkResultsByGlobalIndex)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(4, identityCoordinates(), dimensions, memory);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    batch.materialize(intensity, pdg::DimensionType::Unsigned16);
    batch.setSize(4);
    auto values = batch.hostSpan<std::uint16_t>(intensity);
    values[0] = 3;
    values[1] = 9;
    values[2] = 9;
    values[3] = 1;

    pdg::LocateProgram maximum{intensity, pdg::LocateKind::Maximum};
    const pdg::LocateResult first = pdg::locateExtreme(batch, maximum, 20);
    EXPECT_TRUE(first.hasPoints);
    EXPECT_TRUE(first.comparable);
    EXPECT_EQ(first.index, 21U);
    EXPECT_DOUBLE_EQ(first.value, 9.0);

    batch.setSize(2);
    values = batch.hostSpan<std::uint16_t>(intensity);
    values[0] = 9;
    values[1] = 10;
    const pdg::LocateResult second = pdg::locateExtreme(batch, maximum, 24);
    const pdg::LocateResult merged =
        pdg::mergeLocateResults(maximum, first, second);
    EXPECT_EQ(merged.index, 25U);
    EXPECT_DOUBLE_EQ(merged.value, 10.0);

    pdg::LocateProgram minimum{intensity, pdg::LocateKind::Minimum};
    batch.setSize(4);
    values = batch.hostSpan<std::uint16_t>(intensity);
    values[0] = 3;
    values[1] = 1;
    values[2] = 1;
    values[3] = 2;
    const pdg::LocateResult low = pdg::locateExtreme(batch, minimum, 100);
    EXPECT_EQ(low.index, 101U);
    EXPECT_DOUBLE_EQ(low.value, 1.0);
}

TEST(Locate, PreservesPdalSentinelAndNonComparableBehavior)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionDefinition& scratch =
        dimensions.registerCustom("Scratch", pdg::DimensionType::Double);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(4, identityCoordinates(), dimensions, memory);
    batch.materialize(scratch.id, pdg::DimensionType::Double);
    batch.setSize(3);
    auto values = batch.hostSpan<double>(scratch.id);
    values[0] = std::numeric_limits<double>::quiet_NaN();
    values[1] = std::numeric_limits<double>::lowest();
    values[2] = -std::numeric_limits<double>::infinity();

    pdg::LocateProgram maximum{scratch.id, pdg::LocateKind::Maximum};
    const pdg::LocateResult high = pdg::locateExtreme(batch, maximum, 17);
    EXPECT_TRUE(high.hasPoints);
    EXPECT_FALSE(high.comparable);
    EXPECT_EQ(high.index, 17U);
    EXPECT_DOUBLE_EQ(high.value, std::numeric_limits<double>::lowest());

    values[2] = -4.0;
    const pdg::LocateResult comparable = pdg::locateExtreme(batch, maximum, 17);
    EXPECT_TRUE(comparable.comparable);
    EXPECT_EQ(comparable.index, 19U);
    EXPECT_DOUBLE_EQ(comparable.value, -4.0);

    pdg::LocateProgram minimum{scratch.id, pdg::LocateKind::Minimum};
    values[0] = std::numeric_limits<double>::quiet_NaN();
    values[1] = std::numeric_limits<double>::max();
    values[2] = std::numeric_limits<double>::infinity();
    const pdg::LocateResult low = pdg::locateExtreme(batch, minimum, 31);
    EXPECT_TRUE(low.hasPoints);
    EXPECT_FALSE(low.comparable);
    EXPECT_EQ(low.index, 31U);
    EXPECT_DOUBLE_EQ(low.value, std::numeric_limits<double>::max());
}

TEST(Locate, DecodesPhysicalLasCoordinatesBeforeComparison)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        3,
        pdg::CoordinateEncoding({0.01, 0.02, 0.03}, {1000.0, 2000.0, 3000.0}),
        dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    batch.materialize(x, pdg::DimensionType::Signed32);
    batch.setSize(3);
    auto values = batch.hostSpan<std::int32_t>(x);
    values[0] = -50;
    values[1] = 400;
    values[2] = 399;

    const pdg::LocateProgram program{x, pdg::LocateKind::Maximum};
    const pdg::LocateResult result = pdg::locateExtreme(batch, program, 5);
    EXPECT_EQ(result.index, 6U);
    EXPECT_DOUBLE_EQ(result.value, 1004.0);
}

TEST(Locate, HandlesEmptyNoneAndUnmaterializedInputs)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(2, identityCoordinates(), dimensions, memory);
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    batch.materialize(intensity, pdg::DimensionType::Unsigned16);
    const pdg::LocateProgram maximum{intensity, pdg::LocateKind::Maximum};
    const pdg::LocateResult empty = pdg::locateExtreme(batch, maximum, 9);
    EXPECT_FALSE(empty.hasPoints);
    EXPECT_EQ(empty.index, 9U);

    batch.setSize(1);
    const pdg::LocateProgram none{intensity, pdg::LocateKind::None};
    EXPECT_FALSE(pdg::locateExtreme(batch, none).hasPoints);

    const pdg::LocateProgram missing{
        pdg::DimensionId(pdg::StandardDimension::GpsTime),
        pdg::LocateKind::Maximum};
    EXPECT_THROW(static_cast<void>(pdg::locateExtreme(batch, missing)),
                 std::invalid_argument);
}
} // unnamed namespace
