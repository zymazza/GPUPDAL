#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Smrf.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct BatchFixture
{
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch;

    explicit BatchFixture(std::size_t capacity)
        : batch(capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                dimensions, memory)
    {
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::X),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Y),
                          pdg::DimensionType::Double);
        batch.materialize(pdg::DimensionId(pdg::StandardDimension::Z),
                          pdg::DimensionType::Double);
        batch.materialize(
            pdg::DimensionId(pdg::StandardDimension::Classification),
            pdg::DimensionType::Unsigned8);
        batch.setSize(capacity);
    }
};
} // unnamed namespace

TEST(Smrf, ClassifiesACompactObjectWithoutChangingPointOrder)
{
    BatchFixture fixture(25U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < 5U; ++column)
        for (std::size_t row = 0U; row < 5U; ++row)
        {
            const std::size_t point = column * 5U + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            z[point] = point == 12U ? 5.0 : 0.0;
            classification[point] = 7U;
        }

    pdg::SmrfProgram program;
    program.window = 2.0;
    const pdg::SmrfResult result = pdg::classifySmrf(fixture.batch, program);

    EXPECT_EQ(result.rows, 5U);
    EXPECT_EQ(result.columns, 5U);
    EXPECT_EQ(result.groundPoints, 24U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 12U ? 1U : 2U);
}

TEST(Smrf, OnlyGroundPreservesRejectedClassifications)
{
    BatchFixture fixture(9U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < 3U; ++column)
        for (std::size_t row = 0U; row < 3U; ++row)
        {
            const std::size_t point = column * 3U + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            z[point] = point == 4U ? 6.0 : 0.0;
            classification[point] = 12U;
        }

    pdg::SmrfProgram program;
    program.window = 1.0;
    program.groundClass = 9U;
    program.otherClass = 9U;
    program.onlyGround = true;
    const pdg::SmrfResult result = pdg::classifySmrf(fixture.batch, program);

    EXPECT_EQ(result.groundPoints, 8U);
    EXPECT_EQ(result.nongroundPoints, 1U);
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
        EXPECT_EQ(classification[point], point == 4U ? 12U : 9U);
}

TEST(Smrf, UnqualifiedDeviceEnvelopeFailsClosed)
{
    BatchFixture fixture(4U);
    auto* x =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::X));
    auto* y =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z =
        fixture.batch.data<double>(pdg::DimensionId(pdg::StandardDimension::Z));
    for (std::size_t point = 0U; point < fixture.batch.size(); ++point)
    {
        x[point] = static_cast<double>(point % 2U);
        y[point] = static_cast<double>(point / 2U);
        z[point] = 0.0;
    }
    pdg::SmrfProgram program;
    EXPECT_FALSE(pdg::SmrfExactDeviceQualified);
    EXPECT_FALSE(pdg::smrfSupportsExactDevice(fixture.batch, program));

    program.cell = 0.0;
    EXPECT_FALSE(pdg::smrfSupportsExactDevice(fixture.batch, program));
    EXPECT_THROW(
        {
            const auto unused = pdg::classifySmrf(fixture.batch, program);
            (void)unused;
        },
        std::invalid_argument);

    program = {};
    program.groundClass = program.otherClass;
    EXPECT_FALSE(pdg::smrfSupportsExactDevice(fixture.batch, program));
}

// B0265/D0265: above 32,768 raster cells the port's diamond morphology runs
// on an internal pass pool. The pooled classification must equal the serial
// one (same-final-binary control) bit-for-bit, including with the cut/net
// morphology, on a frame large enough for every pass to fan out.
namespace
{
class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        ::setenv(name, value, 1);
    }
    ~ScopedEnvironment()
    {
        if (m_prior)
            ::setenv(m_name.c_str(), m_prior->c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
    }

private:
    std::string m_name;
    std::optional<std::string> m_prior;
};

std::vector<std::uint8_t> classifyLattice(const pdg::SmrfProgram& program,
                                          std::size_t side)
{
    BatchFixture fixture(side * side);
    auto* x = fixture.batch.data<double>(
        pdg::DimensionId(pdg::StandardDimension::X));
    auto* y = fixture.batch.data<double>(
        pdg::DimensionId(pdg::StandardDimension::Y));
    auto* z = fixture.batch.data<double>(
        pdg::DimensionId(pdg::StandardDimension::Z));
    auto* classification = fixture.batch.data<std::uint8_t>(
        pdg::DimensionId(pdg::StandardDimension::Classification));
    for (std::size_t column = 0U; column < side; ++column)
        for (std::size_t row = 0U; row < side; ++row)
        {
            const std::size_t point = column * side + row;
            x[point] = static_cast<double>(column);
            y[point] = static_cast<double>(row);
            double height = 0.05 * static_cast<double>(column) +
                            0.03 * static_cast<double>(row);
            if (column >= 60U && column <= 99U && row >= 60U && row <= 99U)
                height += 4.0;
            if (column >= 150U && column <= 155U && row >= 20U && row <= 200U)
                height += 12.0;
            if ((column == 2U && row == 2U) || (column == 200U && row == 30U))
                height -= 2.0;
            if ((column * 3U + row * 5U) % 17U == 0U)
                height += 0.4;
            z[point] = height;
            classification[point] = 7U;
        }
    const pdg::SmrfResult result = pdg::classifySmrf(fixture.batch, program);
    EXPECT_EQ(result.rows, side);
    EXPECT_EQ(result.columns, side);
    return std::vector<std::uint8_t>(classification,
                                     classification + side * side);
}
} // unnamed namespace

TEST(Smrf, PooledMorphologyMatchesSerialOnALargeFrame)
{
    for (const double cut : {0.0, 5.0})
    {
        pdg::SmrfProgram program;
        program.cut = cut;
        std::vector<std::uint8_t> serial;
        {
            ScopedEnvironment disable("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
                                      "1");
            serial = classifyLattice(program, 240U);
        }
        const std::vector<std::uint8_t> pooled = classifyLattice(program, 240U);
        EXPECT_EQ(serial, pooled) << "cut " << cut;
        EXPECT_NE(std::count(serial.begin(), serial.end(), 1U), 0);
        EXPECT_NE(std::count(serial.begin(), serial.end(), 2U), 0);
    }
}
