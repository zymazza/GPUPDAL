#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/stages/Skewness.hpp>

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace
{
struct ScopedEnvironment
{
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
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

bool cudaDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

struct PointFixture
{
    double x;
    double y;
    double z;
    std::uint8_t classification;
    std::uint16_t intensity;
    std::uint8_t returnNumber;
};
} // namespace

TEST(PdgSkewnessBalancingFilter, RequiredCudaReordersAllFieldsAndClassifies)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::array<PointFixture, 5U> Points = {{
        {10.0, 0.0, 12.0, 7U, 100U, 8U},
        {20.0, 1.0, -2.0, 8U, 200U, 7U},
        {30.0, 2.0, 5.0, 9U, 300U, 6U},
        {40.0, 3.0, 0.0, 10U, 400U, 5U},
        {50.0, 4.0, 7.0, 11U, 500U, 4U},
    }};
    std::vector<std::size_t> order(Points.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(),
              [&](std::size_t left, std::size_t right)
              { return Points[left].z < Points[right].z; });
    std::vector<double> sortedZ(Points.size());
    std::vector<std::uint8_t> expectedClass(Points.size());
    for (std::size_t index = 0U; index < Points.size(); ++index)
    {
        const std::size_t source = order[index];
        sortedZ[index] = Points[source].z;
        expectedClass[index] = Points[source].classification;
    }
    static_cast<void>(pdg::classifySkewnessSorted(
        sortedZ.data(), expectedClass.data(), sortedZ.size(), {}));

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::Intensity, Id::ReturnNumber});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (std::size_t index = 0U; index < Points.size(); ++index)
    {
        const auto& point = Points[index];
        view->setField(Id::X, static_cast<pdal::PointId>(index), point.x);
        view->setField(Id::Y, static_cast<pdal::PointId>(index), point.y);
        view->setField(Id::Z, static_cast<pdal::PointId>(index), point.z);
        view->setField(Id::Classification, static_cast<pdal::PointId>(index),
                       point.classification);
        view->setField(Id::Intensity, static_cast<pdal::PointId>(index),
                       point.intensity);
        view->setField(Id::ReturnNumber, static_cast<pdal::PointId>(index),
                       point.returnNumber);
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter =
        factory.createStage(std::string(pdg::HybridSkewnessStage));
    ASSERT_NE(filter, nullptr);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));

    for (std::size_t output = 0U; output < Points.size(); ++output)
    {
        const std::size_t source = order[output];
        const auto pointId = static_cast<pdal::PointId>(output);
        EXPECT_EQ(view->getFieldAs<double>(Id::X, pointId), Points[source].x);
        EXPECT_EQ(view->getFieldAs<double>(Id::Y, pointId), Points[source].y);
        EXPECT_EQ(view->getFieldAs<double>(Id::Z, pointId), Points[source].z);
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, pointId),
                  expectedClass[output]);
        EXPECT_EQ(view->getFieldAs<std::uint16_t>(Id::Intensity, pointId),
                  Points[source].intensity);
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::ReturnNumber, pointId),
                  Points[source].returnNumber);
    }
}

TEST(PdgSkewnessBalancingFilter,
     RequiredCudaRejectsSignedZeroTieBeforeMutatingClassifications)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::array<PointFixture, 4U> Points = {{
        {10.0, 0.0, 1.0, 11U, 110U, 5U},
        {20.0, 1.0, 0.0, 12U, 120U, 6U},
        {30.0, 2.0, -0.0, 13U, 130U, 7U},
        {40.0, 3.0, -1.0, 14U, 140U, 8U},
    }};
    std::vector<double> expectedX(Points.size());
    std::vector<double> expectedY(Points.size());
    std::vector<double> expectedZ(Points.size());
    std::vector<std::uint8_t> expectedClass(Points.size());
    std::vector<std::uint16_t> expectedIntensity(Points.size());
    std::vector<std::uint8_t> expectedReturn(Points.size());
    for (std::size_t index = 0U; index < Points.size(); ++index)
    {
        const auto& point = Points[index];
        expectedX[index] = point.x;
        expectedY[index] = point.y;
        expectedZ[index] = point.z;
        expectedClass[index] = point.classification;
        expectedIntensity[index] = point.intensity;
        expectedReturn[index] = point.returnNumber;
    }

    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::Intensity, Id::ReturnNumber});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (std::size_t index = 0U; index < Points.size(); ++index)
    {
        const auto& point = Points[index];
        view->setField(Id::X, static_cast<pdal::PointId>(index), point.x);
        view->setField(Id::Y, static_cast<pdal::PointId>(index), point.y);
        view->setField(Id::Z, static_cast<pdal::PointId>(index), point.z);
        view->setField(Id::Classification, static_cast<pdal::PointId>(index),
                       point.classification);
        view->setField(Id::Intensity, static_cast<pdal::PointId>(index),
                       point.intensity);
        view->setField(Id::ReturnNumber, static_cast<pdal::PointId>(index),
                       point.returnNumber);
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter =
        factory.createStage(std::string(pdg::HybridSkewnessStage));
    ASSERT_NE(filter, nullptr);
    filter->setInput(reader);
    filter->prepare(table);
    try
    {
        static_cast<void>(filter->execute(table));
        FAIL() << "required CUDA skewnessbalancing path unexpectedly executed";
    }
    catch (const std::exception& error)
    {
        EXPECT_STREQ(error.what(),
                     "filters.skewnessbalancing: required exact CUDA hybrid "
                     "skewnessbalancing path was not used");
    }

    for (std::size_t index = 0U; index < Points.size(); ++index)
    {
        const auto pointId = static_cast<pdal::PointId>(index);
        EXPECT_EQ(view->getFieldAs<double>(Id::X, pointId), expectedX[index]);
        EXPECT_EQ(view->getFieldAs<double>(Id::Y, pointId), expectedY[index]);
        EXPECT_EQ(view->getFieldAs<double>(Id::Z, pointId), expectedZ[index]);
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::Classification, pointId),
                  expectedClass[index]);
        EXPECT_EQ(view->getFieldAs<std::uint16_t>(Id::Intensity, pointId),
                  expectedIntensity[index]);
        EXPECT_EQ(view->getFieldAs<std::uint8_t>(Id::ReturnNumber, pointId),
                  expectedReturn[index]);
    }
}
