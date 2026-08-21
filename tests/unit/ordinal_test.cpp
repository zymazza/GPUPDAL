#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordinal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{
std::vector<std::uint64_t> selected(const pdg::OrdinalProgram& program,
                                    pdg::OrdinalMode mode, std::uint64_t total,
                                    const std::vector<std::size_t>& chunks)
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    const std::size_t capacity =
        *std::max_element(chunks.begin(), chunks.end());
    pdg::PointBatch batch(
        capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, memory);
    pdg::OrdinalState state = pdg::makeOrdinalState(program, mode, total);
    std::vector<std::uint8_t> keep(capacity);
    std::vector<std::uint64_t> result;
    std::uint64_t begin = 0;
    for (std::size_t size : chunks)
    {
        batch.setSize(size);
        pdg::evaluateOrdinal(batch, program, state, keep.data());
        for (std::size_t index = 0; index < size; ++index)
            if (keep[index])
                result.push_back(begin + index);
        begin += size;
    }
    EXPECT_EQ(begin, total);
    return result;
}

TEST(Ordinal, PreservesDistinctStandardAndStreamingDecimationSemantics)
{
    pdg::OrdinalProgram program;
    program.kind = pdg::OrdinalKind::Decimation;
    program.step = 4.2;

    EXPECT_EQ(selected(program, pdg::OrdinalMode::Standard, 30, {30}),
              std::vector<std::uint64_t>({0, 4, 8, 13, 17, 21, 25}));
    EXPECT_EQ(selected(program, pdg::OrdinalMode::Streaming, 30, {30}),
              std::vector<std::uint64_t>({0, 4, 8, 13, 17, 21, 25, 29}));
}

TEST(Ordinal, DecimationIsInvariantToChunkBoundaries)
{
    pdg::OrdinalProgram program;
    program.kind = pdg::OrdinalKind::Decimation;
    program.step = 2.6;
    program.offset = 10;
    program.limit = 90;

    const std::vector<std::uint64_t> whole =
        selected(program, pdg::OrdinalMode::Streaming, 100, {100});
    const std::vector<std::uint64_t> split = selected(
        program, pdg::OrdinalMode::Streaming, 100, {1, 8, 17, 3, 41, 30});
    EXPECT_EQ(split, whole);
    EXPECT_EQ(whole, std::vector<std::uint64_t>({10, 13, 15, 18, 20, 23, 26, 28,
                                                 31, 33, 36, 39, 41, 44, 46, 49,
                                                 52, 54, 57, 59, 62, 65, 67, 70,
                                                 72, 75, 78, 80, 83, 85, 88}));
}

TEST(Ordinal, MatchesHeadAndTailInversion)
{
    pdg::OrdinalProgram head;
    head.kind = pdg::OrdinalKind::Head;
    head.count = 3;
    EXPECT_EQ(selected(head, pdg::OrdinalMode::Standard, 8, {2, 3, 3}),
              std::vector<std::uint64_t>({0, 1, 2}));
    head.invert = true;
    EXPECT_EQ(selected(head, pdg::OrdinalMode::Streaming, 8, {1, 4, 3}),
              std::vector<std::uint64_t>({3, 4, 5, 6, 7}));

    pdg::OrdinalProgram tail;
    tail.kind = pdg::OrdinalKind::Tail;
    tail.count = 3;
    EXPECT_EQ(selected(tail, pdg::OrdinalMode::Standard, 8, {2, 3, 3}),
              std::vector<std::uint64_t>({5, 6, 7}));
    tail.invert = true;
    EXPECT_EQ(selected(tail, pdg::OrdinalMode::Standard, 8, {1, 4, 3}),
              std::vector<std::uint64_t>({0, 1, 2, 3, 4}));

    head.count = 20;
    head.invert = false;
    EXPECT_EQ(selected(head, pdg::OrdinalMode::Standard, 8, {3, 5}),
              std::vector<std::uint64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    tail.count = 20;
    tail.invert = false;
    EXPECT_EQ(selected(tail, pdg::OrdinalMode::Standard, 8, {3, 5}),
              std::vector<std::uint64_t>({0, 1, 2, 3, 4, 5, 6, 7}));

    head.count = 0;
    EXPECT_TRUE(selected(head, pdg::OrdinalMode::Streaming, 8, {4, 4}).empty());
    tail.count = 0;
    EXPECT_TRUE(selected(tail, pdg::OrdinalMode::Standard, 8, {4, 4}).empty());
}

TEST(Ordinal, ComputesStandardCountsAndRejectsUnsafeDomains)
{
    pdg::OrdinalProgram program;
    program.kind = pdg::OrdinalKind::Decimation;
    program.step = 4.2;
    EXPECT_EQ(pdg::ordinalStandardOutputCount(program, 30), 7U);
    program.offset = 31;
    EXPECT_THROW(
        static_cast<void>(pdg::ordinalStandardOutputCount(program, 30)),
        std::invalid_argument);
    program.offset = 0;
    program.step = 0.5;
    EXPECT_THROW(static_cast<void>(pdg::makeOrdinalState(
                     program, pdg::OrdinalMode::Standard, 30)),
                 std::invalid_argument);

    pdg::OrdinalProgram tail;
    tail.kind = pdg::OrdinalKind::Tail;
    EXPECT_FALSE(pdg::ordinalSupportsMode(tail, pdg::OrdinalMode::Streaming));
}
} // unnamed namespace
