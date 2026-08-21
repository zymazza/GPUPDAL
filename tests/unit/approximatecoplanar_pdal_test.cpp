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
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

struct FilterOutcome
{
    std::string error;
    std::vector<std::uint8_t> coplanar;
    pdal::pdg_detail::ResidentPhaseSeconds phases;
};

FilterOutcome runApproximateCoplanar(std::string stageName, int knn = 3,
                                     bool automaticCuda = false,
                                     bool observeRepair = false)
{
    using pdal::Dimension::Id;
    constexpr std::uint8_t Sentinel = 0xa5U;
    constexpr std::array<std::array<double, 3>, 12> Points{{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {100.0, 100.0, 100.0},
        {101.0, 100.0, 100.0},
        {100.0, 102.0, 100.0},
        {100.0, 100.0, 103.0},
        {200.0, 200.0, 200.0},
        {201.0, 200.0, 200.0},
        {200.0, 202.0, 200.0},
        {200.0, 200.0, 203.0},
    }};

    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Coplanar});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < Points.size(); ++point)
    {
        const auto& xyz = Points[static_cast<std::size_t>(point)];
        view->setField(Id::X, point, xyz[0]);
        view->setField(Id::Y, point, xyz[1]);
        view->setField(Id::Z, point, xyz[2]);
        view->setField(Id::Coplanar, point, Sentinel);
    }

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage(stageName);
    if (!filter)
        throw std::runtime_error("approximate-coplanar test stage is missing");
    pdal::Options options;
    options.add("knn", knn);
    if (automaticCuda)
        options.add("pdg_auto_cuda", true);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);

    pdg::DimensionRegistry planDimensions;
    std::unique_ptr<pdal::pdg_detail::ResidentExecutionScope> residentScope;
    if (observeRepair)
    {
        const pdg::Plan plan = pdg::compilePipeline(
            "[\"in.las\",{\"type\":\"filters.approximatecoplanar\","
            "\"knn\":8},\"out.las\"]",
            planDimensions);
        residentScope =
            std::make_unique<pdal::pdg_detail::ResidentExecutionScope>(
                plan, planDimensions, 64U * 1024U * 1024U, Points.size());
    }

    FilterOutcome outcome;
    try
    {
        (void)filter->execute(table);
    }
    catch (const std::exception& error)
    {
        outcome.error = error.what();
    }
    if (residentScope)
        outcome.phases = residentScope->context().phaseSeconds();
    outcome.coplanar.reserve(Points.size());
    for (pdal::PointId point = 0; point < Points.size(); ++point)
        outcome.coplanar.push_back(
            view->getFieldAs<std::uint8_t>(Id::Coplanar, point));
    return outcome;
}
} // unnamed namespace

TEST(PdgApproximateCoplanarFilter,
     AutomaticMarkerRechecksActualViewSizeAndFallsBackExactly)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireAutomatic(
        "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA", nullptr);

    const FilterOutcome oracle =
        runApproximateCoplanar("filters.approximatecoplanar", 8);
    const FilterOutcome fallback = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage), 8, true);
    EXPECT_TRUE(oracle.error.empty()) << oracle.error;
    EXPECT_TRUE(fallback.error.empty()) << fallback.error;
    EXPECT_EQ(fallback.coplanar, oracle.coplanar);

    ScopedEnvironment proveAutomatic(
        "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA", "1");
    const FilterOutcome rejected = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage), 8, true);
    EXPECT_EQ(rejected.error,
              "filters.approximatecoplanar: required automatic exact CUDA "
              "hybrid approximatecoplanar path was not used");
    for (const std::uint8_t value : rejected.coplanar)
        EXPECT_EQ(value, 0xa5U);
}

// D0271: under the fast marker the eigen family keeps the device systems for
// tie rows and triggers no host repair; incomplete rows would still be
// repaired (none here).
TEST(PdgApproximateCoplanarFilter, RelaxedTieOrderSkipsTheEigenRepairUnderFast)
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
    const FilterOutcome oracle =
        runApproximateCoplanar("filters.approximatecoplanar", 3);
    const FilterOutcome exact = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage), 3, false, true);
    ASSERT_TRUE(exact.error.empty()) << exact.error;
    ASSERT_EQ(exact.coplanar, oracle.coplanar);
    ASSERT_GT(exact.phases.approximateCoplanarAmbiguousRepairRows, 0U);

    ScopedEnvironment fast("PDG_INTERNAL_FAST_MODE", "1");
    const FilterOutcome relaxed = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage), 3, false, true);
    EXPECT_TRUE(relaxed.error.empty()) << relaxed.error;
    EXPECT_EQ(relaxed.phases.approximateCoplanarRepairTriggers, 0U);
    EXPECT_EQ(relaxed.phases.approximateCoplanarAmbiguousRepairRows, 0U);
    EXPECT_EQ(relaxed.phases.approximateCoplanarRepairRows, 0U);
    EXPECT_EQ(relaxed.phases.exactHostRepair, 0.0);
    ASSERT_EQ(relaxed.coplanar.size(), oracle.coplanar.size());
    std::size_t differing = 0;
    for (std::size_t point = 0; point < oracle.coplanar.size(); ++point)
    {
        EXPECT_TRUE(relaxed.coplanar[point] == 0U ||
                    relaxed.coplanar[point] == 1U);
        differing += relaxed.coplanar[point] != oracle.coplanar[point];
    }
    EXPECT_LE(differing, exact.phases.approximateCoplanarAmbiguousRepairRows);
}

TEST(PdgApproximateCoplanarFilter,
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

    const FilterOutcome oracle =
        runApproximateCoplanar("filters.approximatecoplanar", 8);
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    // A one-shell budget declares the widely separated clusters incomplete
    // on device; the eigen-family host repair must reproduce upstream
    // exactly.
    ScopedEnvironment budget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    const FilterOutcome candidate = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage), 8, false, true);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_TRUE(candidate.error.empty()) << candidate.error;
    EXPECT_EQ(candidate.coplanar, oracle.coplanar);
    EXPECT_EQ(candidate.phases.approximateCoplanarRepairTriggers, 1U);
    EXPECT_GT(candidate.phases.approximateCoplanarIncompleteRepairRows, 0U);
    EXPECT_GT(candidate.phases.approximateCoplanarRepairRows, 0U);
    EXPECT_EQ(candidate.phases.approximateCoplanarKd3Uses, 1U);
    EXPECT_EQ(candidate.phases.approximateCoplanarDeviceToHostRepairBytes,
              12U * (sizeof(std::uint8_t) + sizeof(pdg::EigenSystem3d)));
    EXPECT_EQ(candidate.phases.approximateCoplanarHostToDeviceRepairBytes,
              12U * (sizeof(pdg::EigenSystem3d) + sizeof(std::uint8_t)));
    EXPECT_GT(candidate.phases.approximateCoplanarExactHostRepair, 0.0);
    EXPECT_GT(candidate.phases.exactHostRepair, 0.0);
}

TEST(PdgApproximateCoplanarFilter,
     CudaSolverFailureMatchesUpstreamPrefixMutation)
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

    const FilterOutcome oracle =
        runApproximateCoplanar("filters.approximatecoplanar");
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", nullptr);
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", "1");
    ScopedEnvironment forceMorton("PDG_FORCE_MORTON_BVH", "1");
    ScopedEnvironment forceUniform("PDG_FORCE_UNIFORM_GRID", nullptr);
    ScopedEnvironment injectedFailure(
        "PDG_TEST_NEIGHBORHOOD_EIGEN_FAILURE_POINT", "4");
    const FilterOutcome candidate = runApproximateCoplanar(
        std::string(pdg::HybridApproximateCoplanarStage));

    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    EXPECT_EQ(candidate.error,
              "filters.approximatecoplanar: Cannot perform eigen "
              "decomposition.");
    EXPECT_NE(oracle.coplanar.front(), 0xa5U);
    for (std::size_t point = 0; point < 4; ++point)
        EXPECT_EQ(candidate.coplanar[point], oracle.coplanar[point]) << point;
    for (std::size_t point = 4; point < candidate.coplanar.size(); ++point)
        EXPECT_EQ(candidate.coplanar[point], 0xa5U) << point;
}
