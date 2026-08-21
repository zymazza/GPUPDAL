#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
class ScopedEnvironment
{
public:
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

std::vector<std::uint8_t> runRadiusAssign(const std::string& stageName,
                                          bool search3d,
                                          bool orderedUpdates = false,
                                          bool nonfiniteReferenceZ = false)
{
    using pdal::Dimension::Id;
    constexpr std::array<std::array<double, 3>, 6> Points{{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 2.0},
        {10.0, 0.0, 0.0},
        {10.5, 0.0, 2.01},
        {20.0, 0.0, 0.0},
        {21.0, 0.0, 0.0},
    }};
    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::X, Id::Y, Id::Z, Id::Classification, Id::UserData});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < Points.size(); ++point)
    {
        const auto& xyz = Points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, xyz[0]);
        view->setField(Id::Y, point, xyz[1]);
        view->setField(Id::Z, point, xyz[2]);
        view->setField(Id::Classification, point,
                       static_cast<std::uint8_t>(point % 2U ? 2U : 1U));
        view->setField(Id::UserData, point, std::uint8_t{0});
    }
    if (nonfiniteReferenceZ)
        view->setField(Id::Z, 1, std::numeric_limits<double>::quiet_NaN());

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("radiusassign test stage is missing");
    pdal::Options options;
    options.add("src_domain", "Classification[1:1]");
    options.add("reference_domain", "Classification[2:2]");
    options.add("radius", search3d ? 3.0 : 1.0);
    options.add("is3d", search3d);
    options.add("max2d_above", 2.0);
    options.add("update_expression", "UserData = 9");
    if (orderedUpdates)
        options.add("update_expression",
                    "UserData = UserData + 1 WHERE Classification == 1");
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    (void)filter->execute(table);

    std::vector<std::uint8_t> result;
    result.reserve(Points.size());
    for (pdal::PointId point = 0; point < Points.size(); ++point)
        result.push_back(view->getFieldAs<std::uint8_t>(Id::UserData, point));
    return result;
}
} // unnamed namespace

TEST(PdgRadiusAssignFilter, HostFallbackMatchesUpstreamExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool search3d : {false, true})
        EXPECT_EQ(runRadiusAssign(std::string(pdg::HybridRadiusAssignStage),
                                  search3d),
                  runRadiusAssign("filters.radiusassign", search3d));
}

TEST(PdgRadiusAssignFilter, CudaSelectionMatchesUpstreamExactly)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    for (const bool search3d : {false, true})
        EXPECT_EQ(runRadiusAssign(std::string(pdg::HybridRadiusAssignStage),
                                  search3d),
                  runRadiusAssign("filters.radiusassign", search3d));
}

TEST(PdgRadiusAssignFilter,
     CudaPreservesOrderedUpdatesAndNonfiniteTwoDimensionalZ)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError&)
    {
        GTEST_SKIP() << "no CUDA device is available";
    }
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    EXPECT_EQ(runRadiusAssign(std::string(pdg::HybridRadiusAssignStage), false,
                              true, true),
              runRadiusAssign("filters.radiusassign", false, true, true));
}
