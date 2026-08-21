#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
struct Fixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch;

    Fixture()
        : batch(9, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        const pdg::DimensionId returnNumber(
            pdg::StandardDimension::ReturnNumber);
        const pdg::DimensionId numberOfReturns(
            pdg::StandardDimension::NumberOfReturns);
        batch.materialize(returnNumber, pdg::DimensionType::Unsigned8);
        batch.materialize(numberOfReturns, pdg::DimensionType::Unsigned8);
        batch.setSize(9);
        const std::uint8_t returnNumbers[] = {1, 2, 3, 1, 7, 0, 2, 4, 1};
        const std::uint8_t numbersOfReturns[] = {3, 3, 3, 1, 1, 0, 4, 4, 2};
        std::copy(std::begin(returnNumbers), std::end(returnNumbers),
                  batch.data<std::uint8_t>(returnNumber));
        std::copy(std::begin(numbersOfReturns), std::end(numbersOfReturns),
                  batch.data<std::uint8_t>(numberOfReturns));
    }
};

TEST(ReturnPartition, PreservesFixedGroupAndSourceOrder)
{
    Fixture fixture;
    pdg::ReturnsProgram program;
    program.groups = pdg::AllReturnGroups;
    std::vector<std::uint64_t> permutation(fixture.batch.size());
    const pdg::ReturnsPartitionResult result =
        pdg::partitionReturns(fixture.batch, program, permutation.data());
    EXPECT_EQ(result.counts, (std::array<std::uint64_t, 4>{2, 2, 2, 2}));
    EXPECT_EQ(result.selectedCount(), 8U);
    permutation.resize(static_cast<std::size_t>(result.selectedCount()));
    EXPECT_EQ(permutation,
              (std::vector<std::uint64_t>{0, 8, 1, 6, 2, 7, 3, 4}));
}

TEST(ReturnPartition, HonorsSubsetsAndMalformedOnlySemantics)
{
    Fixture fixture;
    pdg::ReturnsProgram program;
    program.groups = pdg::ReturnLast | pdg::ReturnOnly;
    std::vector<std::uint64_t> permutation(fixture.batch.size());
    const pdg::ReturnsPartitionResult result =
        pdg::partitionReturns(fixture.batch, program, permutation.data());
    EXPECT_EQ(result.counts, (std::array<std::uint64_t, 4>{0, 0, 2, 2}));
    permutation.resize(static_cast<std::size_t>(result.selectedCount()));
    EXPECT_EQ(permutation, (std::vector<std::uint64_t>{2, 7, 3, 4}));

    program.groups = 0U;
    const pdg::ReturnsPartitionResult empty =
        pdg::partitionReturns(fixture.batch, program, permutation.data());
    EXPECT_EQ(empty.selectedCount(), 0U);
}

TEST(ReturnPartition, ValidatesColumnsProgramsAndPointers)
{
    Fixture fixture;
    pdg::ReturnsProgram program;
    EXPECT_TRUE(pdg::returnsMaySupportExactDevice(fixture.batch, program));
    EXPECT_THROW(static_cast<void>(
                     pdg::partitionReturns(fixture.batch, program, nullptr)),
                 std::invalid_argument);

    program.groups = 0x80U;
    std::vector<std::uint64_t> permutation(fixture.batch.size());
    EXPECT_FALSE(pdg::returnsMaySupportExactDevice(fixture.batch, program));
    EXPECT_THROW(static_cast<void>(pdg::partitionReturns(fixture.batch, program,
                                                         permutation.data())),
                 std::invalid_argument);

    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch missing(
        1, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    missing.setSize(1);
    pdg::ReturnsProgram valid;
    EXPECT_FALSE(pdg::returnsMaySupportExactDevice(missing, valid));
    EXPECT_THROW(static_cast<void>(
                     pdg::partitionReturns(missing, valid, permutation.data())),
                 std::invalid_argument);
}

TEST(DividerPartition, MatchesPartitionAndRoundRobinViewOrder)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        10, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.setSize(10);
    std::vector<std::uint64_t> permutation(batch.size());

    pdg::DividerProgram program;
    program.count = 3;
    pdg::DividerPartitionResult result =
        pdg::partitionDivider(batch, program, permutation.data());
    EXPECT_EQ(result.counts, (std::vector<std::uint64_t>{4, 4, 2}));
    EXPECT_EQ(result.selectedCount(), 10U);
    EXPECT_EQ(permutation,
              (std::vector<std::uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    program.mode = pdg::DividerMode::RoundRobin;
    result = pdg::partitionDivider(batch, program, permutation.data());
    EXPECT_EQ(result.counts, (std::vector<std::uint64_t>{4, 3, 3}));
    EXPECT_EQ(permutation,
              (std::vector<std::uint64_t>{0, 3, 6, 9, 1, 4, 7, 2, 5, 8}));
}

TEST(DividerPartition, PreservesEmptyViewsAndValidatesCount)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        1, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.setSize(1);
    std::uint64_t permutation = 99;
    pdg::DividerProgram program;
    program.count = 3;
    const pdg::DividerPartitionResult result =
        pdg::partitionDivider(batch, program, &permutation);
    EXPECT_EQ(result.counts, (std::vector<std::uint64_t>{1, 0, 0}));
    EXPECT_EQ(permutation, 0U);
    EXPECT_TRUE(pdg::dividerMaySupportExactDevice(batch, program));

    batch.setSize(0);
    const pdg::DividerPartitionResult empty =
        pdg::partitionDivider(batch, program, nullptr);
    EXPECT_EQ(empty.counts, (std::vector<std::uint64_t>{0, 0, 0}));

    program.count = 1;
    EXPECT_FALSE(pdg::dividerMaySupportExactDevice(batch, program));
    EXPECT_THROW(
        static_cast<void>(pdg::partitionDivider(batch, program, nullptr)),
        std::invalid_argument);
}

TEST(SplitterCells, MatchesUpstreamNegativeBoundarySemantics)
{
    constexpr std::size_t Count = 7;
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.setSize(Count);
    const double coordinates[Count] = {0.0,  9.999, 10.0, -0.0,
                                       -0.1, -10.0, -10.1};
    std::copy(std::begin(coordinates), std::end(coordinates),
              batch.data<double>(x));
    std::copy(std::rbegin(coordinates), std::rend(coordinates),
              batch.data<double>(y));

    pdg::SplitterProgram program;
    program.length = 10.0;
    program.originX = 0.0;
    program.originY = 0.0;
    std::array<std::int32_t, Count> xCells{};
    std::array<std::int32_t, Count> yCells{};
    pdg::computeSplitterCells(batch, program, xCells.data(), yCells.data());
    const std::array<std::int32_t, Count> expected = {0, 0, 1, 0, -1, -2, -2};
    EXPECT_EQ(xCells, expected);
    EXPECT_TRUE(std::equal(expected.rbegin(), expected.rend(), yCells.begin()));
    EXPECT_TRUE(pdg::splitterCellsMaySupportExactDevice(batch, program));

    program.buffer = 1.0;
    EXPECT_FALSE(pdg::splitterCellsMaySupportExactDevice(batch, program));
    pdg::computeSplitterCells(batch, program, xCells.data(), yCells.data());
}

TEST(SplitterCells, RejectsInvalidProgramsColumnsAndCoordinates)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        1, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.setSize(1);
    pdg::SplitterProgram program;
    program.originX = 0.0;
    program.originY = 0.0;
    std::int32_t cell = 0;
    EXPECT_FALSE(pdg::splitterCellsMaySupportExactDevice(batch, program));
    EXPECT_THROW(pdg::computeSplitterCells(batch, program, &cell, &cell),
                 std::invalid_argument);

    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.data<double>(x)[0] = (std::numeric_limits<double>::quiet_NaN)();
    batch.data<double>(y)[0] = 0.0;
    EXPECT_FALSE(pdg::splitterCellsMaySupportExactDevice(batch, program));
    EXPECT_THROW(pdg::computeSplitterCells(batch, program, &cell, &cell),
                 std::invalid_argument);

    batch.data<double>(x)[0] = 0.0;
    program.length = 0.0;
    EXPECT_THROW(pdg::computeSplitterCells(batch, program, &cell, &cell),
                 std::invalid_argument);
}
} // unnamed namespace
