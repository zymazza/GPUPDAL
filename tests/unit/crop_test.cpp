#include <pdg/PointBatch.hpp>
#include <pdg/stages/Crop.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace
{

struct CropFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch;

    explicit CropFixture(std::size_t size)
        : batch(size, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        for (pdg::StandardDimension dimension :
             {pdg::StandardDimension::X, pdg::StandardDimension::Y,
              pdg::StandardDimension::Z})
            batch.materialize(pdg::DimensionId(dimension),
                              pdg::DimensionType::Double);
        batch.setSize(size);
    }
};

std::vector<std::uint8_t> evaluate(CropFixture& fixture,
                                   std::string_view bounds, bool outside)
{
    const pdg::PredicateProgram predicate =
        pdg::compileCropPredicate(bounds, outside, fixture.dimensions);
    std::vector<std::uint8_t> keep(fixture.batch.size());
    pdg::evaluatePredicate(fixture.batch, predicate, keep.data(), 1);
    return keep;
}

TEST(CropPredicate, MatchesInclusiveTwoDimensionalBoundsAndOutside)
{
    CropFixture fixture(7);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<double, 7> xs = {-1.0, 0.0, 1.0, 2.0, 3.0, nan, 1.0};
    const std::array<double, 7> ys = {11.0, 10.0, 15.0, 20.0, 19.0, 15.0, nan};
    std::copy(xs.begin(), xs.end(), fixture.batch.data<double>(x));
    std::copy(ys.begin(), ys.end(), fixture.batch.data<double>(y));

    const std::vector<std::uint8_t> inside =
        evaluate(fixture, "([0, 2], [10, 20])", false);
    EXPECT_EQ(inside, std::vector<std::uint8_t>({0U, 1U, 1U, 1U, 0U, 0U, 0U}));

    const std::vector<std::uint8_t> outside =
        evaluate(fixture, "([0, 2], [10, 20])", true);
    EXPECT_EQ(outside, std::vector<std::uint8_t>({1U, 0U, 0U, 0U, 1U, 1U, 1U}));
}

TEST(CropPredicate, MatchesThreeDimensionalBounds)
{
    CropFixture fixture(6);
    const pdg::DimensionId x(pdg::StandardDimension::X);
    const pdg::DimensionId y(pdg::StandardDimension::Y);
    const pdg::DimensionId z(pdg::StandardDimension::Z);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::array<double, 6> xs = {0.0, 2.0, 1.0, 1.0, 1.0, 1.0};
    const std::array<double, 6> ys = {10.0, 20.0, 15.0, 15.0, 15.0, 15.0};
    const std::array<double, 6> zs = {-5.0, 5.0, 0.0, 6.0, -6.0, nan};
    std::copy(xs.begin(), xs.end(), fixture.batch.data<double>(x));
    std::copy(ys.begin(), ys.end(), fixture.batch.data<double>(y));
    std::copy(zs.begin(), zs.end(), fixture.batch.data<double>(z));

    const pdg::PredicateProgram predicate = pdg::compileCropPredicate(
        " ( [0,2] , [10,20] , [-5,5] ) ", false, fixture.dimensions);
    EXPECT_EQ(predicate.reads,
              std::vector<pdg::DimensionId>(
                  {pdg::DimensionId(pdg::StandardDimension::X),
                   pdg::DimensionId(pdg::StandardDimension::Y),
                   pdg::DimensionId(pdg::StandardDimension::Z)}));
    std::vector<std::uint8_t> keep(fixture.batch.size());
    pdg::evaluatePredicate(fixture.batch, predicate, keep.data(), 1);
    EXPECT_EQ(keep, std::vector<std::uint8_t>({1U, 1U, 1U, 0U, 0U, 0U}));
}

TEST(CropPredicate, RejectsFormsOutsideTheFirstExactEnvelope)
{
    CropFixture fixture(1);
    const std::array<std::string, 6> invalid = {
        "",
        "([0,1])",
        "([0,1],[0,1],[0,1],[0,1])",
        "([0,1],[0,1]) trailing",
        "[0,0,1,1]",
        R"({"minx":0,"miny":0,"maxx":1,"maxy":1})"};
    for (const std::string& bounds : invalid)
        EXPECT_THROW(static_cast<void>(pdg::compileCropPredicate(
                         bounds, false, fixture.dimensions)),
                     pdg::ExpressionError)
            << bounds;
}

} // unnamed namespace
