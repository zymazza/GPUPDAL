#include <pdg/Coordinate.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

TEST(CoordinateEncoding, PreservesLasIntegerDomain)
{
    const pdg::CoordinateEncoding encoding({0.001, 0.01, 0.1},
                                           {500000.0, 4800000.0, -100.0});
    const std::array<std::int32_t, 5> rawValues{
        std::numeric_limits<std::int32_t>::min(), -1, 0, 1,
        std::numeric_limits<std::int32_t>::max()};

    for (std::int32_t raw : rawValues)
        EXPECT_EQ(encoding.encode(0, encoding.decode(0, raw)), raw);
}

TEST(CoordinateEncoding, RejectsInvalidAndOverflowingValues)
{
    EXPECT_THROW((pdg::CoordinateEncoding({0.0, 1.0, 1.0}, {0.0, 0.0, 0.0})),
                 std::invalid_argument);
    const pdg::CoordinateEncoding encoding({0.001, 0.001, 0.001},
                                           {0.0, 0.0, 0.0});
    EXPECT_THROW(static_cast<void>(encoding.encode(
                     0, std::numeric_limits<double>::infinity())),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(encoding.encode(0, 1.0e30)),
                 std::overflow_error);
    EXPECT_THROW(static_cast<void>(encoding.decode(3, 0)), std::out_of_range);
}

TEST(TileFrame, RebasesBeforeFp32Conversion)
{
    const pdg::CoordinateEncoding encoding({0.001, 0.001, 0.001},
                                           {500000.0, 4800000.0, 100.0});
    const pdg::TileFrame frame(encoding, {500100.0, 4800100.0, 100.0});
    EXPECT_FLOAT_EQ(frame.local(0, 100000), 0.0F);
    EXPECT_FLOAT_EQ(frame.local(1, 101000), 1.0F);
}
