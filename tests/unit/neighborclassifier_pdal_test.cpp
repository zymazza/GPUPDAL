#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Plan.hpp>

#include "src/pdal/PdgResidentContext.hpp"

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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using ResidentPhaseSeconds = pdal::pdg_detail::ResidentPhaseSeconds;

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

std::vector<std::uint8_t> runNeighborClassifier(const std::string& stageName,
                                                bool withDuplicates,
                                                std::string* error = nullptr,
                                                ResidentPhaseSeconds* phases =
                                                    nullptr)
{
    using pdal::Dimension::Id;
    const std::vector<std::array<double, 3>> points =
        rankPoints(withDuplicates);

    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < points.size(); ++point)
    {
        const auto& xyz = points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, xyz[0]);
        view->setField(Id::Y, point, xyz[1]);
        view->setField(Id::Z, point, xyz[2]);
        // Alternate seed classes inside each cluster so majorities flip a
        // minority of points.
        view->setField(Id::Classification, point,
                       static_cast<std::uint8_t>(point % 3U == 0U ? 5U : 2U));
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("neighborclassifier test stage is missing");
    pdal::Options options;
    options.add("k", 7);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    pdg::DimensionRegistry planDimensions;
    std::unique_ptr<pdal::pdg_detail::ResidentExecutionScope> residentScope;
    if (phases)
    {
        const pdg::Plan plan = pdg::compilePipeline(
            "[\"in.las\",{\"type\":\"filters.neighborclassifier\","
            "\"k\":7},\"out.las\"]",
            planDimensions);
        residentScope =
            std::make_unique<pdal::pdg_detail::ResidentExecutionScope>(
                plan, planDimensions, 64U * 1024U * 1024U, points.size());
    }
    try
    {
        (void)filter->execute(table);
    }
    catch (const std::exception& failure)
    {
        if (error)
            *error = failure.what();
    }
    if (residentScope)
        *phases = residentScope->context().phaseSeconds();
    std::vector<std::uint8_t> ranks;
    ranks.reserve(points.size());
    for (pdal::PointId point = 0; point < points.size(); ++point)
        ranks.push_back(
            view->getFieldAs<std::uint8_t>(Id::Classification, point));
    return ranks;
}
} // unnamed namespace

TEST(PdgNeighborClassifierFilter, HostFallbackMatchesUpstreamExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool withDuplicates : {false, true})
    {
        const std::vector<std::uint8_t> oracle =
            runNeighborClassifier("filters.neighborclassifier", withDuplicates);
        const std::vector<std::uint8_t> candidate = runNeighborClassifier(
            std::string(pdg::HybridNeighborClassifierStage), withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}

TEST(PdgNeighborClassifierFilter, CudaPathMatchesUpstreamIncludingTieRepair)
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
            runNeighborClassifier("filters.neighborclassifier", withDuplicates);
        ScopedEnvironment proveRepair("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
                                      withDuplicates ? "1" : nullptr);
        pdal::pdg_detail::ResidentPhaseSeconds phases;
        const std::vector<std::uint8_t> candidate = runNeighborClassifier(
            std::string(pdg::HybridNeighborClassifierStage), withDuplicates,
            nullptr, withDuplicates ? &phases : nullptr);
        EXPECT_EQ(candidate, oracle);
        if (withDuplicates)
        {
            EXPECT_GT(phases.neighborClassifierExactHostRepair, 0.0);
            EXPECT_GT(phases.neighborClassifierAmbiguousRepairRows, 0U);
            EXPECT_EQ(phases.neighborClassifierIncompleteRepairRows, 0U);
            EXPECT_EQ(phases.neighborClassifierRepairRows,
                      phases.neighborClassifierAmbiguousRepairRows);
            EXPECT_EQ(phases.neighborClassifierKd3Uses, 1U);
            EXPECT_EQ(phases.repairedRows,
                      phases.neighborClassifierRepairRows);
        }
    }
}

// D0271: under the fast marker the classifier keeps the device vote for
// tie rows — no host tie repair runs — and every non-tie row still equals
// upstream. The number of rows that may differ is bounded by the exact
// run's ambiguous-row count.
TEST(PdgNeighborClassifierFilter, RelaxedTieOrderSkipsTheHostRepairUnderFast)
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
    ScopedEnvironment noProof("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR", nullptr);
    const std::vector<std::uint8_t> oracle =
        runNeighborClassifier("filters.neighborclassifier", true);
    ResidentPhaseSeconds exactPhases;
    const std::vector<std::uint8_t> exact = runNeighborClassifier(
        std::string(pdg::HybridNeighborClassifierStage), true, nullptr,
        &exactPhases);
    ASSERT_EQ(exact, oracle);
    ASSERT_GT(exactPhases.neighborClassifierAmbiguousRepairRows, 0U);

    ScopedEnvironment fast("PDG_INTERNAL_FAST_MODE", "1");
    ResidentPhaseSeconds fastPhases;
    std::string error;
    const std::vector<std::uint8_t> relaxed = runNeighborClassifier(
        std::string(pdg::HybridNeighborClassifierStage), true, &error,
        &fastPhases);
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_EQ(fastPhases.neighborClassifierAmbiguousRepairRows, 0U);
    EXPECT_EQ(fastPhases.neighborClassifierRepairRows, 0U);
    EXPECT_EQ(fastPhases.repairedRows, 0U);
    EXPECT_EQ(fastPhases.neighborClassifierExactHostRepair, 0.0);
    ASSERT_EQ(relaxed.size(), oracle.size());
    std::size_t differing = 0;
    for (std::size_t point = 0; point < oracle.size(); ++point)
        differing += relaxed[point] != oracle[point];
    EXPECT_LE(differing, exactPhases.neighborClassifierAmbiguousRepairRows);
    for (const std::uint8_t value : relaxed)
        EXPECT_TRUE(value == 2U || value == 5U) << int(value);
}

TEST(PdgNeighborClassifierFilter, DeviceShellBudgetRepairsIncompleteRowsExactly)
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
            runNeighborClassifier("filters.neighborclassifier", withDuplicates);
        const std::vector<std::uint8_t> candidate = runNeighborClassifier(
            std::string(pdg::HybridNeighborClassifierStage), withDuplicates);
        EXPECT_EQ(candidate, oracle);
    }
}
