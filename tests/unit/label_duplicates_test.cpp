#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/LabelDuplicates.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
pdg::PointBatch makeBatch(std::size_t size, pdg::DimensionRegistry& dimensions,
                          pdg::MemoryResource& memory)
{
    pdg::PointBatch batch(
        size, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    batch.setSize(size);
    return batch;
}

TEST(LabelDuplicates, MatchesAdjacentPdalDoubleComparisons)
{
    pdg::DimensionRegistry dimensions;
    const auto& wide =
        dimensions.registerCustom("Wide", pdg::DimensionType::Unsigned64);
    const auto& value =
        dimensions.registerCustom("Value", pdg::DimensionType::Double);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch = makeBatch(8U, dimensions, memory);
    batch.materialize(wide.id, pdg::DimensionType::Unsigned64);
    batch.materialize(value.id, pdg::DimensionType::Double);

    const std::uint64_t firstRounded = UINT64_C(9007199254740992);
    const std::uint64_t wideValues[] = {firstRounded,
                                        firstRounded + 1U,
                                        firstRounded + 2U,
                                        firstRounded + 2U,
                                        7U,
                                        7U,
                                        8U,
                                        8U};
    const double valueValues[] = {
        1.0,
        1.0,
        1.0,
        2.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        -0.0,
        0.0,
    };
    std::copy(std::begin(wideValues), std::end(wideValues),
              batch.data<std::uint64_t>(wide.id));
    std::copy(std::begin(valueValues), std::end(valueValues),
              batch.data<double>(value.id));

    pdg::LabelDuplicatesProgram program;
    program.dimensions = {wide.id, value.id};
    std::vector<std::uint8_t> duplicate(batch.size(), 9U);
    pdg::labelDuplicates(batch, program, duplicate.data());

    EXPECT_EQ(duplicate,
              (std::vector<std::uint8_t>{9U, 1U, 0U, 0U, 0U, 0U, 0U, 1U}));
}

TEST(LabelDuplicates, EmptyDimensionsAreVacuouslyEqualAndPreserveFirstRow)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch = makeBatch(4U, dimensions, memory);
    pdg::LabelDuplicatesProgram program;
    std::vector<std::uint8_t> duplicate(batch.size(), 7U);

    pdg::labelDuplicates(batch, program, duplicate.data());

    EXPECT_EQ(duplicate, (std::vector<std::uint8_t>{7U, 1U, 1U, 1U}));
}

TEST(LabelDuplicates, RejectsMissingColumnsAndNullOutput)
{
    pdg::DimensionRegistry dimensions;
    const auto& missing =
        dimensions.registerCustom("Missing", pdg::DimensionType::Signed32);
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch = makeBatch(2U, dimensions, memory);
    pdg::LabelDuplicatesProgram program;
    program.dimensions = {missing.id};
    std::vector<std::uint8_t> duplicate(batch.size());

    EXPECT_THROW(pdg::labelDuplicates(batch, program, duplicate.data()),
                 std::invalid_argument);
    program.dimensions.clear();
    EXPECT_THROW(pdg::labelDuplicates(batch, program, nullptr),
                 std::invalid_argument);
}

TEST(LabelDuplicates, ExactDeviceEnvelopeDeclinesSelfReferentialOutput)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch = makeBatch(3U, dimensions, memory);
    const pdg::DimensionId duplicate(pdg::StandardDimension::Duplicate);
    batch.materialize(duplicate, pdg::DimensionType::Unsigned8);
    pdg::LabelDuplicatesProgram program;
    program.dimensions = {duplicate};

    EXPECT_FALSE(pdg::labelDuplicatesMaySupportExactDevice(batch, program));
}
} // unnamed namespace
