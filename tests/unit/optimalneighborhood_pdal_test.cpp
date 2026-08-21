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

std::vector<std::array<double, 3>> rankPoints(bool withDuplicates)
{
    // A plane, a line, a volumetric cluster, and an isolated point probe
    // ranks 1 through 3; the duplicate pair and exact midpoint force
    // distance ties through the host closure repair.
    std::vector<std::array<double, 3>> points{
        {0.0, 0.0, 0.0},    {1.0, 0.0, 0.0},    {2.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},    {4.0, 0.0, 0.0},    {10.0, 10.0, 0.0},
        {11.0, 10.0, 0.0},  {10.0, 11.0, 0.0},  {11.0, 11.0, 0.0},
        {10.5, 10.5, 0.0},  {20.0, 20.0, 20.0}, {21.1, 20.2, 20.3},
        {20.4, 21.5, 20.6}, {20.7, 20.8, 21.9}, {21.2, 21.3, 20.1},
        {40.0, 0.0, 5.0},
    };
    if (withDuplicates)
    {
        points.push_back({10.0, 10.0, 0.0});
        points.push_back({2.5, 0.0, 0.0});
    }
    return points;
}

std::vector<double> runOptimalNeighborhood(const std::string& stageName,
                                           bool withDuplicates,
                                           std::string* error = nullptr)
{
    using pdal::Dimension::Id;
    const std::vector<std::array<double, 3>> points =
        rankPoints(withDuplicates);

    pdal::PointTable table;
    table.layout()->registerDims(
        {Id::X, Id::Y, Id::Z, Id::OptimalKNN, Id::OptimalRadius});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < points.size(); ++point)
    {
        const auto& xyz = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, xyz[0]);
        view->setField(Id::Y, point, xyz[1]);
        view->setField(Id::Z, point, xyz[2]);
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("optimalneighborhood test stage is missing");
    pdal::Options options;
    options.add("min_k", 10);
    options.add("max_k", 14);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    try
    {
        (void)filter->execute(table);
    }
    catch (const std::exception& failure)
    {
        if (error)
            *error = failure.what();
    }
    std::vector<double> values;
    values.reserve(points.size() * 2U);
    for (pdal::PointId point = 0; point < points.size(); ++point)
    {
        values.push_back(static_cast<double>(
            view->getFieldAs<std::uint64_t>(Id::OptimalKNN, point)));
        values.push_back(view->getFieldAs<double>(Id::OptimalRadius, point));
    }
    return values;
}
} // unnamed namespace

TEST(PdgOptimalNeighborhoodFilter, HostFallbackMatchesUpstreamExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool withDuplicates : {false, true})
    {
        const std::vector<double> oracle = runOptimalNeighborhood(
            "filters.optimalneighborhood", withDuplicates);
        const std::vector<double> candidate = runOptimalNeighborhood(
            std::string(pdg::HybridOptimalNeighborhoodStage), withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}

TEST(PdgOptimalNeighborhoodFilter, CudaPathMatchesUpstreamIncludingTieRepair)
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
    for (const bool withDuplicates : {false, true})
    {
        const std::vector<double> oracle = runOptimalNeighborhood(
            "filters.optimalneighborhood", withDuplicates);
        ScopedEnvironment proveRepair("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
                                      withDuplicates ? "1" : nullptr);
        const std::vector<double> candidate = runOptimalNeighborhood(
            std::string(pdg::HybridOptimalNeighborhoodStage), withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}

TEST(PdgOptimalNeighborhoodFilter,
     DeviceShellBudgetRepairsIncompleteRowsExactly)
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
    ScopedEnvironment budget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    for (const bool withDuplicates : {false, true})
    {
        const std::vector<double> oracle = runOptimalNeighborhood(
            "filters.optimalneighborhood", withDuplicates);
        const std::vector<double> candidate = runOptimalNeighborhood(
            std::string(pdg::HybridOptimalNeighborhoodStage), withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}
