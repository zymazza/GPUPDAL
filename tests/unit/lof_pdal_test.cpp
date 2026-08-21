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
#include <cstring>
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

struct LofOutcome
{
    std::string error;
    std::vector<double> kDistances;
    std::vector<double> densities;
    std::vector<double> factors;
};

std::vector<std::array<double, 3>> lofPoints(bool withDuplicates,
                                             std::size_t minimumPoints = 0U)
{
    // A quantized-looking cluster mix. The duplicate pair and the exact
    // midpoint force distance ties through the device path's host closure
    // repair.
    std::vector<std::array<double, 3>> points{
        {0.0, 0.0, 0.0},    {1.0, 0.1, 0.0},    {0.3, 1.1, 0.2},
        {1.3, 1.0, 0.1},    {0.7, 0.4, 1.2},    {5.0, 5.1, 5.2},
        {5.9, 5.0, 5.1},    {5.2, 6.0, 5.0},    {6.1, 6.2, 5.3},
        {5.5, 5.5, 6.4},    {20.0, 20.0, 20.0}, {21.2, 20.1, 20.0},
        {20.2, 21.3, 20.1}, {30.0, 0.0, 0.0},
    };
    if (withDuplicates)
    {
        points.push_back({5.0, 5.1, 5.2});
        points.push_back({2.5, 0.05, 0.0});
    }
    // A fixed, reviewable arithmetic sequence supplies enough incomplete
    // rows to cross the production repair scheduler's worker threshold.
    // This is not random input: every coordinate is an exact function of the
    // stable point ordinal.
    for (std::size_t point = points.size(); point < minimumPoints; ++point)
        points.push_back({double((point * 104729U) % 16381U) * 0.001,
                          double((point * 13007U) % 16369U) * 0.001,
                          double((point * 8191U) % 16363U) * 0.001});
    return points;
}

LofOutcome runLof(const std::string& stageName, int minpts, bool withDuplicates,
                  std::size_t minimumPoints = 0U)
{
    using pdal::Dimension::Id;
    const std::vector<std::array<double, 3>> points =
        lofPoints(withDuplicates, minimumPoints);

    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::NNDistance,
                                  Id::LocalReachabilityDistance,
                                  Id::LocalOutlierFactor});
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
        throw std::runtime_error("lof test stage is missing");
    pdal::Options options;
    options.add("minpts", minpts);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);

    LofOutcome outcome;
    try
    {
        (void)filter->execute(table);
    }
    catch (const std::exception& error)
    {
        outcome.error = error.what();
    }
    for (pdal::PointId point = 0; point < points.size(); ++point)
    {
        outcome.kDistances.push_back(
            view->getFieldAs<double>(Id::NNDistance, point));
        outcome.densities.push_back(
            view->getFieldAs<double>(Id::LocalReachabilityDistance, point));
        outcome.factors.push_back(
            view->getFieldAs<double>(Id::LocalOutlierFactor, point));
    }
    return outcome;
}

std::vector<std::uint64_t> bits(const std::vector<double>& values)
{
    std::vector<std::uint64_t> result(values.size());
    std::memcpy(result.data(), values.data(), values.size() * sizeof(double));
    return result;
}

void expectBitEqual(const LofOutcome& candidate, const LofOutcome& oracle)
{
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_TRUE(candidate.error.empty()) << candidate.error;
    EXPECT_EQ(bits(candidate.kDistances), bits(oracle.kDistances));
    EXPECT_EQ(bits(candidate.densities), bits(oracle.densities));
    EXPECT_EQ(bits(candidate.factors), bits(oracle.factors));
}
} // unnamed namespace

TEST(PdgLofFilter, HostFallbackMatchesUpstreamBits)
{
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    ScopedEnvironment experimentalCuda("PDG_EXPERIMENTAL_CUDA_HYBRID", nullptr);
    ScopedEnvironment requireCuda("PDG_REQUIRE_CUDA_HYBRID", nullptr);
    for (const bool withDuplicates : {false, true})
    {
        const LofOutcome oracle = runLof("filters.lof", 5, withDuplicates);
        const LofOutcome candidate =
            runLof(std::string(pdg::HybridLofStage), 5, withDuplicates);
        expectBitEqual(candidate, oracle);
    }
}

TEST(PdgLofFilter, CudaPathMatchesUpstreamBitsIncludingTieRepair)
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
        const LofOutcome oracle = runLof("filters.lof", 5, withDuplicates);
        ScopedEnvironment proveRepair("PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
                                      withDuplicates ? "1" : nullptr);
        const LofOutcome candidate =
            runLof(std::string(pdg::HybridLofStage), 5, withDuplicates);
        expectBitEqual(candidate, oracle);
    }
}

TEST(PdgLofFilter, RowBackedBoundariesMatchTheFieldPathBitForBit)
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
    // The semantic per-field boundary is the reference; the row-backed
    // fast path must engage on a row-backed PointTable (the proof hook
    // throws if it does not) and produce identical bits.
    LofOutcome fieldPath;
    {
        ScopedEnvironment disableRows("PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY",
                                      "1");
        fieldPath = runLof(std::string(pdg::HybridLofStage), 5, false);
    }
    ScopedEnvironment proveRows("PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY", "1");
    const LofOutcome rowPath =
        runLof(std::string(pdg::HybridLofStage), 5, false);
    expectBitEqual(rowPath, fieldPath);
}

TEST(PdgLofFilter, DeviceShellBudgetRepairsIncompleteRowsBitForBit)
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
        const LofOutcome oracle = runLof("filters.lof", 5, withDuplicates);
        // A one-shell budget declares nearly every row incomplete, forcing
        // the k-distance, density, and factor closure repair to produce
        // the entire result; it must still match upstream bit for bit.
        ScopedEnvironment budget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
        const LofOutcome candidate =
            runLof(std::string(pdg::HybridLofStage), 5, withDuplicates);
        expectBitEqual(candidate, oracle);
    }
}

TEST(PdgLofFilter, ParallelClosureRepairMatchesUpstreamBits)
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
    constexpr std::size_t Points = 16'384U;
    const LofOutcome oracle = runLof("filters.lof", 5, false, Points);
    LofOutcome uncached;
    {
        ScopedEnvironment workers("PDG_NATIVE_WORKERS", "4");
        ScopedEnvironment requireParallel("PDG_REQUIRE_LOF_PARALLEL_REPAIR",
                                          "1");
        ScopedEnvironment disableCache("PDG_DISABLE_LOF_KD3_COORDINATE_CACHE",
                                       "1");
        uncached = runLof(std::string(pdg::HybridLofStage), 5, false, Points);
    }
    expectBitEqual(uncached, oracle);
    for (const char* workerCount : {"1", "2", "4"})
    {
        ScopedEnvironment workers("PDG_NATIVE_WORKERS", workerCount);
        ScopedEnvironment requireParallel(
            "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
            std::string_view(workerCount) == "1" ? nullptr : "1");
        ScopedEnvironment requireCache("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
                                       "1");
        const LofOutcome candidate =
            runLof(std::string(pdg::HybridLofStage), 5, false, Points);
        expectBitEqual(candidate, oracle);
    }
}

TEST(PdgLofFilter, CachedBuildFailureFallsBackToUncachedExactRepair)
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
    ScopedEnvironment workers("PDG_NATIVE_WORKERS", "4");
    ScopedEnvironment requireParallel("PDG_REQUIRE_LOF_PARALLEL_REPAIR", "1");
    constexpr std::size_t Points = 16'384U;
    const LofOutcome oracle = runLof("filters.lof", 5, false, Points);
    ScopedEnvironment failure("PDG_TEST_KD3_CACHE_BUILD_FAILURE", "1");
    const LofOutcome recovered =
        runLof(std::string(pdg::HybridLofStage), 5, false, Points);
    expectBitEqual(recovered, oracle);

    ScopedEnvironment requireCache("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE", "1");
    const LofOutcome proof =
        runLof(std::string(pdg::HybridLofStage), 5, false, Points);
    EXPECT_EQ(proof.error, "injected cached KD3 build failure");
}

TEST(PdgLofFilter, SmallClosureRepairRemainsSerial)
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
    ScopedEnvironment workers("PDG_NATIVE_WORKERS", "4");
    ScopedEnvironment requireParallel("PDG_REQUIRE_LOF_PARALLEL_REPAIR", "1");
    const LofOutcome candidate =
        runLof(std::string(pdg::HybridLofStage), 5, false);
    EXPECT_EQ(candidate.error, "required parallel LOF repair did not occur");
}

TEST(PdgLofFilter, ReplicatesUpstreamStatefulMinptsIncrementAcrossViews)
{
    // Upstream increments its minpts member once per filter() call, so a
    // second view in the same execution queries one more neighbor. The
    // wrapper must replicate that quirk exactly.
    using pdal::Dimension::Id;
    ScopedEnvironment disableCuda("PDG_DISABLE_CUDA_HYBRID", "1");
    const auto runTwoViews = [](const std::string& stageName)
    {
        const std::vector<std::array<double, 3>> points = lofPoints(false);
        pdal::PointTable table;
        table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::NNDistance,
                                      Id::LocalReachabilityDistance,
                                      Id::LocalOutlierFactor});
        pdal::PointViewPtr first(new pdal::PointView(table));
        pdal::PointViewPtr second(new pdal::PointView(table));
        for (pdal::PointId point = 0; point < points.size(); ++point)
        {
            const auto& xyz = points[static_cast<std::size_t>(point)];
            for (const pdal::PointViewPtr& view : {first, second})
            {
                view->setField(Id::X, point, xyz[0]);
                view->setField(Id::Y, point, xyz[1]);
                view->setField(Id::Z, point, xyz[2]);
            }
        }
        pdal::BufferReader reader;
        reader.addView(first);
        reader.addView(second);
        pdal::StageFactory factory;
        pdal::Stage* filter = factory.createStage(stageName);
        pdal::Options options;
        options.add("minpts", 5);
        filter->setOptions(options);
        filter->setInput(reader);
        filter->prepare(table);
        (void)filter->execute(table);
        std::vector<double> factors;
        for (const pdal::PointViewPtr& view : {first, second})
            for (pdal::PointId point = 0; point < points.size(); ++point)
                factors.push_back(
                    view->getFieldAs<double>(Id::LocalOutlierFactor, point));
        return factors;
    };
    const std::vector<double> oracle = runTwoViews("filters.lof");
    const std::vector<double> candidate =
        runTwoViews(std::string(pdg::HybridLofStage));
    EXPECT_EQ(candidate, oracle);
    // The two views genuinely differ: the second used one more neighbor.
    const std::size_t half = oracle.size() / 2U;
    EXPECT_NE(std::vector<double>(oracle.begin(), oracle.begin() + half),
              std::vector<double>(oracle.begin() + half, oracle.end()));
}
