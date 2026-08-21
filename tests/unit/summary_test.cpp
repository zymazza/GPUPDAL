#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Summary.hpp>

#include <filters/StatsFilter.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace
{
void expectSameBits(double actual, double expected)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual),
              std::bit_cast<std::uint64_t>(expected));
}

void compareWithUpstream(std::span<const double> values, bool advanced)
{
    pdal::stats::Summary upstream("Value", pdal::stats::Summary::NoEnum,
                                  advanced);
    pdg::SummaryState state;
    for (double value : values)
    {
        upstream.insert(value);
        pdg::insertSummary(state, value, advanced);
    }

    EXPECT_EQ(state.count, upstream.count());
    expectSameBits(state.minimum, upstream.minimum());
    expectSameBits(state.maximum, upstream.maximum());
    expectSameBits(pdg::summaryAverage(state), upstream.average());
    expectSameBits(pdg::summaryPopulationVariance(state),
                   upstream.populationVariance());
    expectSameBits(pdg::summarySampleVariance(state),
                   upstream.sampleVariance());
    expectSameBits(pdg::summaryPopulationStddev(state),
                   upstream.populationStddev());
    expectSameBits(pdg::summarySampleStddev(state), upstream.sampleStddev());
    expectSameBits(pdg::summaryPopulationSkewness(state, advanced),
                   upstream.populationSkewness());
    expectSameBits(pdg::summarySampleSkewness(state, advanced),
                   upstream.sampleSkewness());
    expectSameBits(pdg::summaryPopulationKurtosis(state, advanced),
                   upstream.populationKurtosis());
    expectSameBits(pdg::summarySampleKurtosis(state, advanced),
                   upstream.sampleKurtosis());
    expectSameBits(pdg::summarySampleExcessKurtosis(state, advanced),
                   upstream.sampleExcessKurtosis());
}

TEST(Summary, MatchesPinnedPdalOnlineMomentsBitForBit)
{
    const std::array<double, 13> values = {1.0,
                                           -17.25,
                                           1.0e12,
                                           -1.0e-12,
                                           42.5,
                                           42.5,
                                           -0.0,
                                           3.141592653589793,
                                           -9007199254740992.0,
                                           0.125,
                                           65535.0,
                                           -2048.75,
                                           7.0};
    compareWithUpstream(values, false);
    compareWithUpstream(values, true);
}

TEST(Summary, PreservesExactStateAcrossBatchesAndDimensions)
{
    constexpr std::size_t Count = 4099;
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId first =
        dimensions.registerCustom("First", pdg::DimensionType::Double).id;
    const pdg::DimensionId second =
        dimensions.registerCustom("Second", pdg::DimensionType::Double).id;
    const std::array<pdg::DimensionId, 2> ids = {first, second};
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    for (pdg::DimensionId id : ids)
        batch.materialize(id, pdg::DimensionType::Double);
    batch.setSize(Count);
    for (std::size_t point = 0; point < Count; ++point)
    {
        batch.data<double>(first)[point] =
            static_cast<double>(static_cast<std::int64_t>(point % 991U) - 495) /
            11.0;
        batch.data<double>(second)[point] =
            static_cast<double>((point * 104729U) % 1000003U) / 13.0;
    }

    std::array<pdg::SummaryState, 2> expected{};
    for (std::size_t dimension = 0; dimension < ids.size(); ++dimension)
        for (std::size_t point = 0; point < Count; ++point)
            pdg::insertSummary(expected[dimension],
                               batch.data<double>(ids[dimension])[point], true);

    std::array<pdg::SummaryState, 2> actual{};
    batch.setSize(2001);
    pdg::updateSummaries(batch, ids, actual.data(), true);
    for (std::size_t dimension = 0; dimension < ids.size(); ++dimension)
    {
        const double* values = batch.data<double>(ids[dimension]);
        std::move(values + 2001, values + Count,
                  batch.data<double>(ids[dimension]));
    }
    batch.setSize(Count - 2001U);
    pdg::updateSummaries(batch, ids, actual.data(), true);

    for (std::size_t dimension = 0; dimension < ids.size(); ++dimension)
    {
        EXPECT_EQ(actual[dimension].count, expected[dimension].count);
        expectSameBits(actual[dimension].minimum, expected[dimension].minimum);
        expectSameBits(actual[dimension].maximum, expected[dimension].maximum);
        expectSameBits(actual[dimension].m1, expected[dimension].m1);
        expectSameBits(actual[dimension].m2, expected[dimension].m2);
        expectSameBits(actual[dimension].m3, expected[dimension].m3);
        expectSameBits(actual[dimension].m4, expected[dimension].m4);
    }
}

TEST(Summary, ValidatesExactCudaEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const pdg::DimensionId value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double).id;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(
        3, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.materialize(value, pdg::DimensionType::Double);
    batch.setSize(3);
    const std::array<pdg::DimensionId, 1> ids = {value};
    const std::array<double, 3> values = {1.0, 2.0, 3.0};
    std::copy(values.begin(), values.end(), batch.data<double>(value));
    EXPECT_TRUE(pdg::summariesMaySupportExactDevice(batch, ids, false));
    EXPECT_FALSE(pdg::summariesMaySupportExactDevice(batch, ids, true));

    batch.data<double>(value)[1] = (std::numeric_limits<double>::quiet_NaN)();
    EXPECT_FALSE(pdg::summariesMaySupportExactDevice(batch, ids, false));

    const std::array<pdg::DimensionId, 1> missing = {
        pdg::DimensionId(pdg::StandardDimension::Z)};
    EXPECT_FALSE(pdg::summariesMaySupportExactDevice(batch, missing, false));
    pdg::SummaryState state;
    EXPECT_THROW(pdg::updateSummaries(batch, missing, &state, false),
                 std::invalid_argument);
    EXPECT_THROW(pdg::updateSummaries(batch, ids, nullptr, false),
                 std::invalid_argument);
}
} // unnamed namespace
