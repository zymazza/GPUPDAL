#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
TEST(Ordering, PreservesStableSingleDimensionTiesInBothDirections)
{
    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("Key", pdg::DimensionType::Signed32);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        5, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(key.id, pdg::DimensionType::Signed32);
    batch.setSize(5);
    const std::int32_t source[] = {2, 1, 2, 1, 1};
    std::copy(std::begin(source), std::end(source),
              batch.data<std::int32_t>(key.id));

    pdg::OrderingProgram program;
    program.dimensions = {key.id};
    program.algorithm = pdg::OrderingAlgorithm::Stable;
    std::vector<std::uint64_t> permutation(batch.size());
    EXPECT_TRUE(pdg::orderPoints(batch, program, permutation.data()).exact);
    EXPECT_EQ(permutation, (std::vector<std::uint64_t>{1, 3, 4, 0, 2}));

    program.direction = pdg::OrderingDirection::Descending;
    EXPECT_TRUE(pdg::orderPoints(batch, program, permutation.data()).exact);
    EXPECT_EQ(permutation, (std::vector<std::uint64_t>{0, 2, 1, 3, 4}));
}

TEST(Ordering, ReproducesPdalMultiPassDimensionPriority)
{
    pdg::DimensionRegistry dimensions;
    const auto& first =
        dimensions.registerCustom("First", pdg::DimensionType::Signed32);
    const auto& last =
        dimensions.registerCustom("Last", pdg::DimensionType::Unsigned16);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        4, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(first.id, pdg::DimensionType::Signed32);
    batch.materialize(last.id, pdg::DimensionType::Unsigned16);
    batch.setSize(4);
    const std::int32_t firstValues[] = {2, 1, 0, 3};
    const std::uint16_t lastValues[] = {1, 1, 0, 0};
    std::copy(std::begin(firstValues), std::end(firstValues),
              batch.data<std::int32_t>(first.id));
    std::copy(std::begin(lastValues), std::end(lastValues),
              batch.data<std::uint16_t>(last.id));

    pdg::OrderingProgram program;
    program.dimensions = {first.id, last.id};
    std::vector<std::uint64_t> permutation(batch.size());
    static_cast<void>(pdg::orderPoints(batch, program, permutation.data()));
    EXPECT_EQ(permutation, (std::vector<std::uint64_t>{2, 3, 1, 0}));
}

TEST(Ordering, SortsEncodedCoordinatesByLogicalDoubleValue)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId x(pdg::StandardDimension::X);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        3, pdg::CoordinateEncoding({0.5, 1.0, 1.0}, {1000.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(x, pdg::DimensionType::Signed32);
    batch.setSize(3);
    const std::int32_t raw[] = {2, 0, 1};
    std::copy(std::begin(raw), std::end(raw), batch.data<std::int32_t>(x));
    pdg::OrderingProgram program;
    program.dimensions = {x};
    std::vector<std::uint64_t> permutation(batch.size());
    EXPECT_TRUE(pdg::orderingMaySupportExactDevice(batch, program));
    static_cast<void>(pdg::orderPoints(batch, program, permutation.data()));
    EXPECT_EQ(permutation, (std::vector<std::uint64_t>{1, 2, 0}));
}

TEST(Ordering, DeviceEnvelopeRejectsNanAndInvalidPrograms)
{
    pdg::DimensionRegistry dimensions;
    const auto& key =
        dimensions.registerCustom("Key", pdg::DimensionType::Double);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        4, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(key.id, pdg::DimensionType::Double);
    batch.setSize(4);
    auto values = batch.hostSpan<double>(key.id);
    values[0] = -std::numeric_limits<double>::infinity();
    values[1] = -0.0;
    values[2] = 0.0;
    values[3] = std::numeric_limits<double>::infinity();
    pdg::OrderingProgram program;
    program.dimensions = {key.id};
    program.algorithm = pdg::OrderingAlgorithm::Stable;
    EXPECT_TRUE(pdg::orderingMaySupportExactDevice(batch, program));
    values[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::orderingMaySupportExactDevice(batch, program));

    pdg::OrderingProgram empty;
    EXPECT_FALSE(pdg::orderingMaySupportExactDevice(batch, empty));
    std::vector<std::uint64_t> permutation(batch.size());
    EXPECT_THROW(
        static_cast<void>(pdg::orderPoints(batch, empty, permutation.data())),
        std::invalid_argument);
}
} // unnamed namespace
