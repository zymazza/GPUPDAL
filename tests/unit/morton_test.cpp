#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Morton.hpp>
#include <pdg/stages/Ordering.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
struct MortonFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::DimensionId key =
        dimensions.registerCustom("MortonKey", pdg::DimensionType::Unsigned64)
            .id;
    pdg::HostMemoryResource memory;
};

bool lessMostSignificantBit(const int& left, const int& right)
{
    return left < right && left < (left ^ right);
}

bool pdalZOrderLess(std::pair<double, double> left,
                    std::pair<double, double> right)
{
    const int a[2] = {static_cast<int>(left.first * INT_MAX),
                      static_cast<int>(left.second * INT_MAX)};
    const int b[2] = {static_cast<int>(right.first * INT_MAX),
                      static_cast<int>(right.second * INT_MAX)};
    int selected = 0;
    int difference = 0;
    for (int axis = 0; axis < 2; ++axis)
    {
        const int candidate = a[axis] ^ b[axis];
        if (lessMostSignificantBit(difference, candidate))
        {
            selected = axis;
            difference = candidate;
        }
    }
    return (a[selected] - b[selected]) < 0;
}

TEST(Morton, DefaultCudaSelectionKeepsMarginAboveMeasuredCrossover)
{
    pdg::MortonProgram program;
    program.bounds = {0.0, 0.0, 1.0, 1.0};
    EXPECT_FALSE(pdg::preferDefaultCudaMorton(0, program));
    EXPECT_FALSE(pdg::preferDefaultCudaMorton(1'999'999, program));
    EXPECT_TRUE(pdg::preferDefaultCudaMorton(2'000'000, program));
    EXPECT_TRUE(pdg::preferDefaultCudaMorton(22'000'000, program));

    program.bounds.maxX = program.bounds.minX;
    EXPECT_FALSE(pdg::preferDefaultCudaMorton(22'000'000, program));
    program.bounds.maxX = 1.0;
    program.bounds.maxY = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(pdg::preferDefaultCudaMorton(22'000'000, program));
}

TEST(Morton, ForwardKeysReproducePdalComparatorPriority)
{
    MortonFixture fixture;
    constexpr std::size_t Count = 35;
    pdg::PointBatch batch(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        fixture.dimensions, fixture.memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.materialize(fixture.key, pdg::DimensionType::Unsigned64);
    batch.setSize(Count);
    auto xValues = batch.hostSpan<double>(x);
    auto yValues = batch.hostSpan<double>(y);
    for (std::size_t point = 0; point < Count; ++point)
    {
        xValues[point] = static_cast<double>((point * 11U) % 7U) - 3.0;
        yValues[point] = static_cast<double>((point * 13U) % 5U) + 10.0;
    }
    pdg::MortonProgram program;
    program.bounds = {-3.0, 10.0, 3.0, 14.0};
    ASSERT_TRUE(pdg::mortonMaySupportExactDevice(batch, program));
    pdg::generateMortonKeys(
        batch, program,
        static_cast<std::uint64_t*>(batch.rawData(fixture.key)));

    std::vector<std::uint64_t> actual(Count);
    std::iota(actual.begin(), actual.end(), std::uint64_t{0});
    const auto keys = batch.hostSpan<std::uint64_t>(fixture.key);
    std::stable_sort(actual.begin(), actual.end(),
                     [&keys](std::uint64_t left, std::uint64_t right)
                     { return keys[left] < keys[right]; });

    std::vector<std::uint64_t> expected(Count);
    std::iota(expected.begin(), expected.end(), std::uint64_t{0});
    std::stable_sort(
        expected.begin(), expected.end(),
        [&xValues, &yValues](std::uint64_t left, std::uint64_t right)
        {
            return pdalZOrderLess(
                {(xValues[left] + 3.0) / 6.0, (yValues[left] - 10.0) / 4.0},
                {(xValues[right] + 3.0) / 6.0, (yValues[right] - 10.0) / 4.0});
        });
    EXPECT_EQ(actual, expected);
}

TEST(Morton, ReverseKeysMatchPinnedFourByFourTraversal)
{
    MortonFixture fixture;
    constexpr std::size_t Count = 16;
    pdg::PointBatch batch(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        fixture.dimensions, fixture.memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.materialize(fixture.key, pdg::DimensionType::Unsigned64);
    batch.setSize(Count);
    for (std::size_t point = 0; point < Count; ++point)
    {
        batch.hostSpan<double>(x)[point] = static_cast<double>(point / 4U);
        batch.hostSpan<double>(y)[point] = static_cast<double>(point % 4U);
    }
    pdg::MortonProgram program;
    program.bounds = {0.0, 0.0, 3.0, 3.0};
    program.reverse = true;
    pdg::generateMortonKeys(
        batch, program,
        static_cast<std::uint64_t*>(batch.rawData(fixture.key)));
    pdg::OrderingProgram ordering;
    ordering.dimensions = {fixture.key};
    ordering.algorithm = pdg::OrderingAlgorithm::Stable;
    std::vector<std::uint64_t> permutation(Count);
    static_cast<void>(pdg::orderPoints(batch, ordering, permutation.data()));
    EXPECT_EQ(permutation,
              (std::vector<std::uint64_t>{0, 3, 12, 15, 2, 14, 8, 11, 10, 1, 13,
                                          9, 4, 7, 6, 5}));
}

TEST(Morton, DeviceEnvelopeRejectsDegenerateAndNonfiniteCoordinates)
{
    MortonFixture fixture;
    pdg::PointBatch batch(
        2, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        fixture.dimensions, fixture.memory);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    batch.materialize(x, pdg::DimensionType::Double);
    batch.materialize(y, pdg::DimensionType::Double);
    batch.setSize(2);
    batch.hostSpan<double>(x)[0] = 0.0;
    batch.hostSpan<double>(x)[1] = 1.0;
    batch.hostSpan<double>(y)[0] = 2.0;
    batch.hostSpan<double>(y)[1] = 3.0;
    pdg::MortonProgram program;
    program.bounds = {0.0, 2.0, 1.0, 3.0};
    EXPECT_TRUE(pdg::mortonMaySupportExactDevice(batch, program));

    program.bounds.maxX = program.bounds.minX;
    EXPECT_FALSE(pdg::mortonMaySupportExactDevice(batch, program));
    EXPECT_THROW(pdg::generateMortonKeys(batch, program, nullptr),
                 std::invalid_argument);

    program.bounds.maxX = 1.0;
    batch.hostSpan<double>(y)[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::mortonMaySupportExactDevice(batch, program));
}
} // unnamed namespace
