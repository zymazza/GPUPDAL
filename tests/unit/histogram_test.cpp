#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>

namespace
{
TEST(Histogram, DefaultCudaSelectionRequiresMeasuredWork)
{
    EXPECT_FALSE(pdg::preferDefaultCudaExpressionStats(22'000'000, 0));
    EXPECT_FALSE(pdg::preferDefaultCudaExpressionStats(3'999'999, 1));
    EXPECT_TRUE(pdg::preferDefaultCudaExpressionStats(4'000'000, 1));
    EXPECT_FALSE(pdg::preferDefaultCudaExpressionStats(1'999'999, 2));
    EXPECT_TRUE(pdg::preferDefaultCudaExpressionStats(2'000'000, 2));
    EXPECT_FALSE(pdg::preferDefaultCudaExpressionStats(999'999, 3));
    EXPECT_TRUE(pdg::preferDefaultCudaExpressionStats(1'000'000, 3));
    EXPECT_TRUE(pdg::preferDefaultCudaExpressionStats(1'000'000, 8));
}

TEST(Histogram, PreservesMapOrderCountsAndFirstEquivalentBits)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double).id;
    const pdg::DimensionId flag =
        dimensions.registerCustom("PredicateFlag", pdg::DimensionType::Double)
            .id;
    const pdg::PredicateProgram predicate = pdg::compilePredicate(
        "PredicateFlag >= 1 && PredicateFlag < 3", dimensions);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        8, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(value);
    batch.materialize(flag);
    batch.setSize(8);
    const double values[8] = {+0.0, -0.0, 2.0, -1.0, 2.0, 3.0, 7.0, 9.0};
    const double flags[8] = {1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0, 3.0};
    std::copy_n(values, 8, batch.data<double>(value));
    std::copy_n(flags, 8, batch.data<double>(flag));

    const std::vector<pdg::HistogramBin> bins =
        pdg::selectedHistogram(batch, value, predicate, 100, 1);
    ASSERT_EQ(bins.size(), 4U);
    EXPECT_DOUBLE_EQ(bins[0].value, -1.0);
    EXPECT_EQ(bins[0].count, 1U);
    EXPECT_EQ(bins[0].firstIndex, 103U);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(bins[1].value),
              std::bit_cast<std::uint64_t>(+0.0));
    EXPECT_EQ(bins[1].count, 2U);
    EXPECT_EQ(bins[1].firstIndex, 100U);
    EXPECT_DOUBLE_EQ(bins[2].value, 2.0);
    EXPECT_EQ(bins[2].count, 2U);
    EXPECT_EQ(bins[2].firstIndex, 102U);
    EXPECT_DOUBLE_EQ(bins[3].value, 3.0);
    EXPECT_EQ(bins[3].count, 1U);
}

TEST(Histogram, HandlesEmptyMatchesAndValidatesExactDeviceEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double).id;
    const pdg::PredicateProgram predicate =
        pdg::compilePredicate("Value > 100", dimensions);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        3, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(value);
    batch.setSize(3);
    batch.data<double>(value)[0] = 1.0;
    batch.data<double>(value)[1] = 2.0;
    batch.data<double>(value)[2] = 3.0;
    EXPECT_TRUE(pdg::selectedHistogram(batch, value, predicate).empty());
    EXPECT_TRUE(pdg::histogramMaySupportExactDevice(batch, value, predicate));

    batch.data<double>(value)[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(pdg::histogramMaySupportExactDevice(batch, value, predicate));
    const pdg::PredicateProgram unsupported =
        pdg::compilePredicate("floor(Value) > 0", dimensions);
    batch.data<double>(value)[1] = 2.0;
    EXPECT_FALSE(
        pdg::histogramMaySupportExactDevice(batch, value, unsupported));
}
} // unnamed namespace
