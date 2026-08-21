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

std::vector<std::uint8_t> runEstimateRank(const std::string& stageName, int knn,
                                          double thresh, bool withDuplicates,
                                          std::string* error = nullptr)
{
    using pdal::Dimension::Id;
    const std::vector<std::array<double, 3>> points =
        rankPoints(withDuplicates);

    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Rank});
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
        throw std::runtime_error("estimaterank test stage is missing");
    pdal::Options options;
    options.add("knn", knn);
    options.add("thresh", thresh);
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
    std::vector<std::uint8_t> ranks;
    ranks.reserve(points.size());
    for (pdal::PointId point = 0; point < points.size(); ++point)
        ranks.push_back(view->getFieldAs<std::uint8_t>(Id::Rank, point));
    return ranks;
}
} // unnamed namespace

TEST(PdgEstimateRankFilter, HostFallbackMatchesUpstreamExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool withDuplicates : {false, true})
        for (const double thresh : {0.01, 0.2})
        {
            const std::vector<std::uint8_t> oracle = runEstimateRank(
                "filters.estimaterank", 8, thresh, withDuplicates);
            const std::vector<std::uint8_t> candidate =
                runEstimateRank(std::string(pdg::HybridEstimateRankStage), 8,
                                thresh, withDuplicates);
            EXPECT_EQ(candidate, oracle);
        }
}

TEST(PdgEstimateRankFilter, CudaPathMatchesUpstreamIncludingTieRepair)
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
        const std::vector<std::uint8_t> oracle =
            runEstimateRank("filters.estimaterank", 8, 0.01, withDuplicates);
        ScopedEnvironment proveRepair("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
                                      withDuplicates ? "1" : nullptr);
        const std::vector<std::uint8_t> candidate = runEstimateRank(
            std::string(pdg::HybridEstimateRankStage), 8, 0.01, withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}

TEST(PdgEstimateRankFilter, DeviceShellBudgetRepairsIncompleteRowsExactly)
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
        const std::vector<std::uint8_t> oracle =
            runEstimateRank("filters.estimaterank", 8, 0.01, withDuplicates);
        const std::vector<std::uint8_t> candidate = runEstimateRank(
            std::string(pdg::HybridEstimateRankStage), 8, 0.01, withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}
