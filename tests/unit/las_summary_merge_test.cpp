// D0268: merging per-slot writers.las summaries in slot order must
// reproduce the serial per-point fold bit-for-bit, including first-seen
// signed-zero extremes, NaN coordinates that never enter the bounds, empty
// slots, out-of-range return numbers, and the return histogram.

#include <gtest/gtest.h>

#include <io/private/las/Summary.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

namespace
{

struct Row
{
    double x;
    double y;
    double z;
    int returnNumber;
};

bool sameBits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void expectIdentical(const pdal::las::Summary& expected,
    const pdal::las::Summary& observed)
{
    EXPECT_EQ(expected.getTotalNumPoints(), observed.getTotalNumPoints());
    const pdal::BOX3D e = expected.getBounds();
    const pdal::BOX3D o = observed.getBounds();
    EXPECT_TRUE(sameBits(e.minx, o.minx));
    EXPECT_TRUE(sameBits(e.maxx, o.maxx));
    EXPECT_TRUE(sameBits(e.miny, o.miny));
    EXPECT_TRUE(sameBits(e.maxy, o.maxy));
    EXPECT_TRUE(sameBits(e.minz, o.minz));
    EXPECT_TRUE(sameBits(e.maxz, o.maxz));
    // getReturnCount() indexes the histogram from zero.
    for (int r = 0; r < static_cast<int>(pdal::las::Header::ReturnCount); ++r)
        EXPECT_EQ(expected.getReturnCount(r), observed.getReturnCount(r))
            << "return index " << r;
}

void checkSlotFolds(const std::vector<Row>& rows, std::size_t slots)
{
    pdal::las::Summary serial;
    for (const Row& row : rows)
        serial.addPoint(row.x, row.y, row.z, row.returnNumber);

    std::vector<pdal::las::Summary> partial(slots);
    const std::size_t per = rows.size() / slots;
    const std::size_t rem = rows.size() % slots;
    std::size_t begin = 0;
    for (std::size_t slot = 0; slot < slots; ++slot)
    {
        const std::size_t end = begin + per + (slot < rem ? 1U : 0U);
        for (std::size_t i = begin; i < end; ++i)
            partial[slot].addPoint(rows[i].x, rows[i].y, rows[i].z,
                rows[i].returnNumber);
        begin = end;
    }
    pdal::las::Summary merged;
    for (const pdal::las::Summary& s : partial)
        merged.merge(s);
    expectIdentical(serial, merged);
}

} // unnamed namespace

TEST(LasSummaryMerge, ReproducesSerialFoldAtEverySlotCount)
{
    std::vector<Row> rows;
    for (int k = 0; k < 997; ++k)
    {
        const double x = 184500.0 + ((k * 7919) % 97) * 0.25;
        const double y = 494920.0 + ((k * 104729) % 89) * 0.5;
        const double z = -2.0 + ((k * 1299709) % 101) * 0.125;
        rows.push_back({x, y, z, 1 + (k * 17) % 5});
    }
    // Out-of-range return numbers are ignored by both folds.
    rows.push_back({184501.0, 494921.0, 0.0, 0});
    rows.push_back({184501.0, 494921.0, 0.0, 99});
    // NaN never enters the bounds in either fold.
    rows.push_back({std::numeric_limits<double>::quiet_NaN(), 494921.0,
        0.0, 1});
    for (std::size_t slots = 1; slots <= 9; ++slots)
        checkSlotFolds(rows, slots);
    // More slots than rows: trailing slots are empty.
    checkSlotFolds(std::vector<Row>(rows.begin(), rows.begin() + 3), 8);
    checkSlotFolds({}, 4);
}

TEST(LasSummaryMerge, KeepsFirstSeenSignedZeroExtreme)
{
    // +0.0 first, then -0.0: strict comparisons keep the first-seen zero in
    // the serial fold; slot merges must agree whichever slot holds each.
    std::vector<Row> rows;
    for (int k = 0; k < 40; ++k)
        rows.push_back({k < 20 ? 0.0 : -0.0, k < 20 ? -0.0 : 0.0,
            k % 3 == 0 ? 0.0 : -0.0, 1});
    for (std::size_t slots = 1; slots <= 6; ++slots)
        checkSlotFolds(rows, slots);
    pdal::las::Summary serial;
    for (const Row& row : rows)
        serial.addPoint(row.x, row.y, row.z, row.returnNumber);
    EXPECT_FALSE(std::signbit(serial.getBounds().minx));
    EXPECT_TRUE(std::signbit(serial.getBounds().miny));
}

TEST(LasSummaryMerge, EmptyMergeLeavesSummaryUntouched)
{
    pdal::las::Summary summary;
    summary.addPoint(1.0, 2.0, 3.0, 2);
    pdal::las::Summary empty;
    summary.merge(empty);
    EXPECT_EQ(1U, summary.getTotalNumPoints());
    EXPECT_EQ(1U, summary.getReturnCount(1));  // zero-indexed histogram
    EXPECT_DOUBLE_EQ(1.0, summary.getBounds().minx);
    EXPECT_DOUBLE_EQ(3.0, summary.getBounds().maxz);
}
