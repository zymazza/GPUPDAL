#include <pdg/stages/Skewness.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

TEST(Skewness, EmptyInputIsAnExactNoop)
{
    const pdg::SkewnessResult result =
        pdg::classifySkewnessSorted(nullptr, nullptr, 0U, {});
    EXPECT_EQ(result.groundPoints, 0U);
    EXPECT_EQ(result.otherPoints, 0U);
}

TEST(Skewness, OnePointAndConstantSurfaceFollowPinnedNanTailBehavior)
{
    const std::array<double, 1U> oneZ{7.0};
    std::array<std::uint8_t, 1U> oneClass{7U};
    const pdg::SkewnessResult one = pdg::classifySkewnessSorted(
        oneZ.data(), oneClass.data(), oneZ.size(), {});
    EXPECT_EQ(oneClass[0], 1U);
    EXPECT_EQ(one.groundPoints, 0U);
    EXPECT_EQ(one.otherPoints, 1U);

    const std::array<double, 4U> flatZ{10.0, 10.0, 10.0, 10.0};
    std::array<std::uint8_t, 4U> flatClass{7U, 7U, 7U, 7U};
    const pdg::SkewnessResult flat = pdg::classifySkewnessSorted(
        flatZ.data(), flatClass.data(), flatZ.size(), {});
    EXPECT_EQ(flatClass, (std::array<std::uint8_t, 4U>{1U, 1U, 1U, 1U}));
    EXPECT_EQ(flat.otherPoints, flatZ.size());
}

TEST(Skewness, PreservesSequentialMomentCrossingSegments)
{
    // The pinned recurrence crosses from non-positive to positive at sorted
    // positions 2 and 4. Each crossing classifies the preceding segment.
    const std::array<double, 5U> z{0.0, 1.0, 3.0, 4.0, 6.0};
    std::array<std::uint8_t, 5U> classification{7U, 7U, 7U, 7U, 7U};
    const pdg::SkewnessResult result = pdg::classifySkewnessSorted(
        z.data(), classification.data(), z.size(), {});
    EXPECT_EQ(classification,
              (std::array<std::uint8_t, 5U>{2U, 2U, 2U, 2U, 1U}));
    EXPECT_EQ(result.groundPoints, 4U);
    EXPECT_EQ(result.otherPoints, 1U);
}

TEST(Skewness, OnlyGroundPreservesTheUnclassifiedSuffix)
{
    const std::array<double, 5U> z{0.0, 1.0, 3.0, 4.0, 6.0};
    std::array<std::uint8_t, 5U> classification{10U, 11U, 12U, 13U, 14U};
    pdg::SkewnessProgram program;
    program.groundClass = 3U;
    program.otherClass = 3U;
    program.onlyGround = true;
    const pdg::SkewnessResult result = pdg::classifySkewnessSorted(
        z.data(), classification.data(), z.size(), program);
    EXPECT_EQ(classification,
              (std::array<std::uint8_t, 5U>{3U, 3U, 3U, 3U, 14U}));
    EXPECT_EQ(result.groundPoints, 4U);
    EXPECT_EQ(result.otherPoints, 0U);
}

TEST(Skewness, RejectsInvalidProgramsAndNullColumns)
{
    pdg::SkewnessProgram invalid;
    invalid.otherClass = invalid.groundClass;
    EXPECT_FALSE(pdg::skewnessProgramValid(invalid));
    EXPECT_THROW(static_cast<void>(pdg::classifySkewnessSorted(nullptr, nullptr,
                                                               0U, invalid)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(
                     pdg::classifySkewnessSorted(nullptr, nullptr, 1U, {})),
                 std::invalid_argument);

    EXPECT_FALSE(pdg::skewnessOrderingSizeWithinDeviceEnvelope(0U));
    EXPECT_TRUE(pdg::skewnessOrderingSizeWithinDeviceEnvelope(
        static_cast<std::size_t>((std::numeric_limits<int>::max)())));
    EXPECT_FALSE(pdg::skewnessOrderingSizeWithinDeviceEnvelope(
        static_cast<std::size_t>((std::numeric_limits<int>::max)()) + 1U));
}
