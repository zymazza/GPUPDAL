#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Plan.hpp>
#include <pdg/io/Las.hpp>

#include "src/pdal/PdgNeighborhood.hpp"
#include "src/pdal/PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

using Point = std::array<double, 3>;

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

std::size_t boundaryId(const pdg::Plan& plan,
                       pdg::ResidencyBoundaryKind kind)
{
    const auto position =
        std::find_if(plan.summary().residencyBoundaries.begin(),
                     plan.summary().residencyBoundaries.end(),
                     [&](const pdg::ResidencyBoundary& boundary)
                     { return boundary.kind == kind; });
    if (position == plan.summary().residencyBoundaries.end())
        throw std::runtime_error("nndistance test boundary is unavailable");
    return static_cast<std::size_t>(
        std::distance(plan.summary().residencyBoundaries.begin(), position));
}

std::vector<double> runNnDistanceOracle(const std::vector<Point>& points,
                                        std::size_t k)
{
    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::NNDistance});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const Point& value = points.at(static_cast<std::size_t>(point));
        view->setField(Id::X, point, value[0]);
        view->setField(Id::Y, point, value[1]);
        view->setField(Id::Z, point, value[2]);
    }
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage("filters.nndistance");
    if (!filter)
        throw std::runtime_error("nndistance oracle stage is unavailable");
    pdal::Options options;
    options.add("k", k);
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    static_cast<void>(filter->execute(table));

    std::vector<double> result;
    result.reserve(points.size());
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        result.push_back(view->getFieldAs<double>(Id::NNDistance, point));
    return result;
}

struct ResidentNnDistanceOutcome
{
    std::vector<double> values;
    pdal::pdg_detail::ResidentPhaseSeconds phases;
    pdal::pdg_detail::ResidentManagerPhaseSeconds manager;
    bool directLasSourceUsed = false;
    bool directLasRecordSummaryUsed = false;
    bool directLasHostXyzMirrored = false;
};

ResidentNnDistanceOutcome
runResidentNnDistance(const std::vector<Point>& points, bool enableStats,
                      std::size_t k = 3U,
                      const std::string& mode = "kth",
                      std::span<const std::byte> directLasSource = {},
                      bool* failedSummaryUsed = nullptr)
{
    pdg::DimensionRegistry dimensions;
    const std::string pipeline =
        "[\"in.las\",{\"type\":\"filters.nndistance\",\"mode\":\"" +
        mode + "\",\"k\":" + std::to_string(k) + "},\"out.las\"]";
    const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    if (region == pdg::NoResidentRegion)
        throw std::runtime_error("nndistance resident region is unavailable");
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    using pdal::Dimension::Id;
    pdal::PointTable table;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::NNDistance});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        const Point& value = points.at(static_cast<std::size_t>(point));
        view->setField(Id::X, point, value[0]);
        view->setField(Id::Y, point, value[1]);
        view->setField(Id::Z, point, value[2]);
    }

    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, 64U * 1024U * 1024U, 64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    const auto managerStarted = std::chrono::steady_clock::now();
    EXPECT_EQ(context.managerDetailAccumulator(), nullptr);
    if (enableStats)
    {
        context.beginManagerPhaseTiming(managerStarted, managerStarted);
        EXPECT_NE(context.managerDetailAccumulator(), nullptr);
    }
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter =
        factory.createStage(std::string(pdg::HybridNnDistanceStage));
    if (!filter)
        throw std::runtime_error("resident nndistance stage is unavailable");
    pdal::Options options;
    options.add("k", k);
    options.add("mode", mode);
    options.add("pdg_region_id", static_cast<std::uint64_t>(region + 1U));
    options.add("pdg_region_neighbors",
                static_cast<std::uint32_t>(k + 1U));
    options.add("pdg_region_last", true);
    options.add("pdg_resident_context", true);
    options.add("pdg_execution_region", static_cast<std::uint64_t>(region));
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    std::optional<pdal::pdg_detail::CudaNeighborhoodLasSourceScope>
        sourceScope;
    if (!directLasSource.empty())
        sourceScope.emplace(directLasSource);
    try
    {
        static_cast<void>(filter->execute(table));
    }
    catch (...)
    {
        if (failedSummaryUsed)
            *failedSummaryUsed = sourceScope && sourceScope->recordSummaryUsed();
        throw;
    }
    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    ResidentNnDistanceOutcome outcome;
    outcome.directLasSourceUsed = sourceScope && sourceScope->used();
    outcome.directLasRecordSummaryUsed =
        sourceScope && sourceScope->recordSummaryUsed();
    outcome.directLasHostXyzMirrored =
        sourceScope && sourceScope->hostXyzMirrored();
    outcome.phases = context.phaseSeconds();
    if (enableStats)
        outcome.manager =
            context.finishManagerPhaseTiming(std::chrono::steady_clock::now());
    outcome.values.reserve(points.size());
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        outcome.values.push_back(
            view->getFieldAs<double>(Id::NNDistance, point));
    return outcome;
}

TEST(PdgNnDistanceFilter, DirectLasSourceHydratesPlannerNeighborhood)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    const std::filesystem::path path =
        std::filesystem::path(PDG_TEST_DATA_DIR) / "las/simple.las";
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input);
    const std::vector<char> characters((std::istreambuf_iterator<char>(input)),
                                       std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(characters.size());
    std::transform(characters.begin(), characters.end(), bytes.begin(),
                   [](char value)
                   { return static_cast<std::byte>(value); });
    const pdg::las::FileView file(bytes);
    const pdg::CoordinateEncoding encoding =
        file.header().coordinateEncoding();
    std::vector<Point> sourcePoints(
        static_cast<std::size_t>(file.header().pointCount));
    for (std::size_t point = 0; point < sourcePoints.size(); ++point)
        for (std::size_t axis = 0; axis < 3U; ++axis)
            sourcePoints[point][axis] = encoding.decode(
                axis, file.rawCoordinate(point, axis));
    const std::vector<double> expected =
        runNnDistanceOracle(sourcePoints, 3U);
    std::vector<Point> deliberatelyWrong(sourcePoints.size(),
                                         Point{0.0, 0.0, 0.0});
    {
        ScopedEnvironment bvh("PDG_FORCE_MORTON_BVH", nullptr);
        ScopedEnvironment grid("PDG_FORCE_UNIFORM_GRID", nullptr);
        ScopedEnvironment requireBackend(
            "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND", "uniform_grid");
        const ResidentNnDistanceOutcome observed = runResidentNnDistance(
            deliberatelyWrong, true, 3U, "kth", bytes);
        EXPECT_TRUE(observed.directLasRecordSummaryUsed);
        EXPECT_FALSE(observed.directLasHostXyzMirrored);
        EXPECT_EQ(observed.values, expected);
    }
    for (const bool forceBvh : {false, true})
    {
        ScopedEnvironment bvh("PDG_FORCE_MORTON_BVH",
                              forceBvh ? "1" : nullptr);
        ScopedEnvironment grid("PDG_FORCE_UNIFORM_GRID",
                               forceBvh ? nullptr : "1");
        const ResidentNnDistanceOutcome observed = runResidentNnDistance(
            deliberatelyWrong, true, 3U, "kth", bytes);
        EXPECT_TRUE(observed.directLasSourceUsed);
        EXPECT_TRUE(observed.directLasRecordSummaryUsed);
        EXPECT_FALSE(observed.directLasHostXyzMirrored);
        EXPECT_EQ(observed.values, expected);
    }
    {
        ScopedEnvironment bvh("PDG_FORCE_MORTON_BVH", nullptr);
        ScopedEnvironment grid("PDG_FORCE_UNIFORM_GRID", nullptr);
        ScopedEnvironment requireBackend(
            "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND", "invalid");
        bool failedSummaryUsed = true;
        EXPECT_THROW(static_cast<void>(runResidentNnDistance(
                         deliberatelyWrong, true, 3U, "kth", bytes,
                         &failedSummaryUsed)),
                     std::runtime_error);
        EXPECT_FALSE(failedSummaryUsed);
    }

    constexpr std::uint32_t ClusterCount = 3000U;
    std::vector<std::byte> clustered(
        file.header().pointDataOffset +
        static_cast<std::size_t>(ClusterCount) *
            file.header().pointRecordLength);
    std::copy_n(bytes.begin(), file.header().pointDataOffset,
                clustered.begin());
    const std::span<const std::byte> sourceRecord = file.pointRecord(0U);
    for (std::size_t point = 0U; point < ClusterCount; ++point)
        std::copy(sourceRecord.begin(), sourceRecord.end(),
                  clustered.begin() + file.header().pointDataOffset +
                      point * file.header().pointRecordLength);
    std::memcpy(clustered.data() + 107U, &ClusterCount,
                sizeof(ClusterCount));
    std::memcpy(clustered.data() + 111U, &ClusterCount,
                sizeof(ClusterCount));
    for (std::size_t point = ClusterCount - 12U; point < ClusterCount;
         ++point)
    {
        const std::int32_t step =
            static_cast<std::int32_t>(point - (ClusterCount - 13U));
        const std::array<std::int32_t, 3> raw{
            100000000 + step * 10000000, -100000000 - step * 7000000,
            50000000 + step * 3000000};
        std::byte* record =
            clustered.data() + file.header().pointDataOffset +
            point * file.header().pointRecordLength;
        for (std::size_t axis = 0U; axis < raw.size(); ++axis)
            std::memcpy(record + axis * sizeof(std::int32_t), &raw[axis],
                        sizeof(std::int32_t));
    }
    const pdg::las::FileView clusteredFile(clustered);
    std::vector<Point> clusteredPoints(ClusterCount);
    for (std::size_t point = 0U; point < clusteredPoints.size(); ++point)
        for (std::size_t axis = 0U; axis < 3U; ++axis)
            clusteredPoints[point][axis] = encoding.decode(
                axis, clusteredFile.rawCoordinate(point, axis));
    const std::vector<double> clusteredExpected =
        runNnDistanceOracle(clusteredPoints, 3U);
    std::vector<Point> clusteredWrong(ClusterCount, Point{0.0, 0.0, 0.0});
    ScopedEnvironment bvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment grid("PDG_FORCE_UNIFORM_GRID", nullptr);
    ScopedEnvironment requireBackend(
        "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND", "morton_bvh");
    const ResidentNnDistanceOutcome clusteredObserved = runResidentNnDistance(
        clusteredWrong, true, 3U, "kth", clustered);
    EXPECT_TRUE(clusteredObserved.directLasRecordSummaryUsed);
    EXPECT_FALSE(clusteredObserved.directLasHostXyzMirrored);
    EXPECT_EQ(clusteredObserved.values, clusteredExpected);

}

TEST(PdgNnDistanceFilter, StatisticalOutlierReusesOneResidentIndexAndGather)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    std::vector<Point> points;
    for (std::size_t point = 0U; point < 14U; ++point)
        points.push_back(
            {static_cast<double>(point) * 0.3,
             static_cast<double>(point % 3U) * 0.2,
             static_cast<double>(point % 5U) * 0.1});
    points.push_back({20.0, 20.0, 20.0});

    using pdal::Dimension::Id;
    const auto populate = [&](pdal::PointTable& table)
    {
        table.layout()->registerDims(
            {Id::X, Id::Y, Id::Z, Id::Classification, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const Point& value = points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value[0]);
            view->setField(Id::Y, point, value[1]);
            view->setField(Id::Z, point, value[2]);
            view->setField(Id::Classification, point, std::uint8_t{2U});
        }
        return view;
    };

    pdal::PointTable oracleTable;
    pdal::PointViewPtr expected = populate(oracleTable);
    pdal::BufferReader oracleReader;
    oracleReader.addView(expected);
    pdal::StageFactory oracleFactory;
    pdal::Stage* outlier = oracleFactory.createStage("filters.outlier");
    pdal::Stage* nndistance =
        oracleFactory.createStage("filters.nndistance");
    ASSERT_NE(outlier, nullptr);
    ASSERT_NE(nndistance, nullptr);
    pdal::Options outlierOptions;
    outlierOptions.add("method", "statistical");
    outlierOptions.add("mean_k", 4);
    outlierOptions.add("multiplier", 1.25);
    outlierOptions.add("class", 18U);
    outlier->setOptions(outlierOptions);
    pdal::Options nnOptions;
    nnOptions.add("mode", "kth");
    nnOptions.add("k", 3U);
    nndistance->setOptions(nnOptions);
    outlier->setInput(oracleReader);
    nndistance->setInput(*outlier);
    nndistance->prepare(oracleTable);
    static_cast<void>(nndistance->execute(oracleTable));

    pdal::PointTable candidateTable;
    pdal::PointViewPtr actual = populate(candidateTable);

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":4,"multiplier":1.25,"class":18},{"type":"filters.nndistance","mode":"kth","k":3},"out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);
    ASSERT_EQ(plan.stages().at(2U).residentRegion, region);
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    pdg::ExecutionObservationScope observation;
    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, 64U * 1024U * 1024U, 64U);
    const std::array selectedRegions{region};
    scope.preflight(*candidateTable.layout(), actual->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    context.enterBoundary(
        *actual, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader candidateReader;
    candidateReader.addView(actual);
    pdal::StageFactory candidateFactory;
    pdal::Stage* candidateOutlier =
        candidateFactory.createStage(std::string(pdg::HybridOutlierStage));
    pdal::Stage* candidateNn =
        candidateFactory.createStage(std::string(pdg::HybridNnDistanceStage));
    ASSERT_NE(candidateOutlier, nullptr);
    ASSERT_NE(candidateNn, nullptr);
    pdal::Options candidateOutlierOptions;
    candidateOutlierOptions.add("method", "statistical");
    candidateOutlierOptions.add("mean_k", 4);
    candidateOutlierOptions.add("multiplier", 1.25);
    candidateOutlierOptions.add("class", 18U);
    candidateOutlierOptions.add("pdg_region_id",
                                static_cast<std::uint64_t>(region + 1U));
    candidateOutlierOptions.add("pdg_region_neighbors", 5U);
    candidateOutlierOptions.add("pdg_region_gather_neighbors", 5U);
    candidateOutlierOptions.add("pdg_region_last", false);
    candidateOutlierOptions.add("pdg_resident_context", true);
    candidateOutlierOptions.add("pdg_execution_region",
                                static_cast<std::uint64_t>(region));
    candidateOutlier->setOptions(candidateOutlierOptions);
    pdal::Options candidateNnOptions;
    candidateNnOptions.add("mode", "kth");
    candidateNnOptions.add("k", 3U);
    candidateNnOptions.add("pdg_region_id",
                           static_cast<std::uint64_t>(region + 1U));
    candidateNnOptions.add("pdg_region_neighbors", 5U);
    candidateNnOptions.add("pdg_region_gather_neighbors", 5U);
    candidateNnOptions.add("pdg_region_reuse", true);
    candidateNnOptions.add("pdg_region_last", true);
    candidateNnOptions.add("pdg_resident_context", true);
    candidateNnOptions.add("pdg_execution_region",
                           static_cast<std::uint64_t>(region));
    candidateNn->setOptions(candidateNnOptions);
    candidateOutlier->setInput(candidateReader);
    candidateNn->setInput(*candidateOutlier);
    candidateNn->prepare(candidateTable);
    ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_REUSE", "1");
    ScopedEnvironment requireGatherReuse("PDG_REQUIRE_KNN_GATHER_REUSE",
                                         "1");
    const pdal::PointViewSet output = candidateNn->execute(candidateTable);
    ASSERT_EQ(output.size(), 1U);
    context.enterBoundary(
        *actual, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    const pdg::ExecutionStatsSnapshot stats = observation.snapshot();
    EXPECT_EQ(stats
                  .totals[static_cast<std::size_t>(
                      pdg::ExecutionEventKind::IndexBuild)]
                  .count,
              1U);
    EXPECT_EQ(stats
                  .totals[static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionBegin)]
                  .count,
              1U);
    EXPECT_EQ(stats
                  .totals[static_cast<std::size_t>(
                      pdg::ExecutionEventKind::DeviceRegionEnd)]
                  .count,
              1U);
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        EXPECT_EQ(actual->getFieldAs<std::uint8_t>(Id::Classification, point),
                  expected->getFieldAs<std::uint8_t>(Id::Classification,
                                                     point));
        EXPECT_EQ(actual->getFieldAs<double>(Id::NNDistance, point),
                  expected->getFieldAs<double>(Id::NNDistance, point));
    }
}

TEST(PdgNnDistanceFilter, StatisticalOutlierPreflightRejectsTooFewRows)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8},{"type":"filters.nndistance","k":3},"out.las"])",
        dimensions);
    const std::size_t region = plan.stages().at(1U).residentRegion;
    ASSERT_NE(region, pdg::NoResidentRegion);

    pdal::PointTable table;
    table.layout()->registerDims(
        {pdal::Dimension::Id::X, pdal::Dimension::Id::Y,
         pdal::Dimension::Id::Z, pdal::Dimension::Id::Classification,
         pdal::Dimension::Id::NNDistance});
    pdal::pdg_detail::ResidentExecutionScope scope(
        plan, dimensions, 64U * 1024U * 1024U, 64U);
    const std::array selectedRegions{region};
    EXPECT_THROW(scope.preflight(*table.layout(), 8U, selectedRegions),
                 std::invalid_argument);
}

TEST(PdgNnDistanceFilter,
     StatisticalOutlierCachedGatherRepairsEachIncompleteConsumerExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    std::vector<Point> points{{0.0, 0.0, 30.0}};
    for (std::size_t point = 0U; point < 128U; ++point)
        points.push_back(
            {5001.0 + 0.03125 * static_cast<double>(point),
             0.0625 * static_cast<double>(point % 11U),
             10.25 + 0.125 * static_cast<double>(point % 17U)});
    using pdal::Dimension::Id;
    const auto populate = [&](pdal::PointTable& table)
    {
        table.layout()->registerDims({Id::X, Id::Y, Id::Z,
                                      Id::Classification, Id::NNDistance});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const Point& value = points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value[0]);
            view->setField(Id::Y, point, value[1]);
            view->setField(Id::Z, point, value[2]);
            view->setField(Id::Classification, point, std::uint8_t{2U});
        }
        return view;
    };

    pdal::PointTable oracleTable;
    pdal::PointViewPtr expected = populate(oracleTable);
    pdal::BufferReader oracleReader;
    oracleReader.addView(expected);
    pdal::StageFactory factory;
    pdal::Stage* oracle = factory.createStage("filters.outlier");
    pdal::Stage* oracleNn = factory.createStage("filters.nndistance");
    ASSERT_NE(oracle, nullptr);
    ASSERT_NE(oracleNn, nullptr);
    pdal::Options options;
    options.add("method", "statistical");
    options.add("mean_k", 3);
    options.add("multiplier", 1.25);
    options.add("class", 18U);
    oracle->setOptions(options);
    oracle->setInput(oracleReader);
    pdal::Options nnOptions;
    nnOptions.add("mode", "kth");
    nnOptions.add("k", 3U);
    oracleNn->setOptions(nnOptions);
    oracleNn->setInput(*oracle);
    oracleNn->prepare(oracleTable);
    static_cast<void>(oracleNn->execute(oracleTable));

    pdal::PointTable candidateTable;
    pdal::PointViewPtr actual = populate(candidateTable);
    pdal::pdg_detail::CudaNeighborhoodRegion region;
    region.id = 1U;
    region.maximumNeighbors = 4U;
    region.gatherNeighbors = 4U;
    region.last = false;
    pdal::pdg_detail::CudaStatisticalOutlierResult result;
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireGatherReuse("PDG_REQUIRE_KNN_GATHER_REUSE",
                                         "1");
    {
        ScopedEnvironment requireOutlierDeviceRepair(
            "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR", "1");
        ScopedEnvironment requireOutlierParallelRepair(
            "PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR", "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaStatisticalOutlier(
            *actual, 4U, 1.25, 18U, region, result, /*requireCuda=*/true));
    }
    EXPECT_GT(result.repairedRows, 0U);
    EXPECT_EQ(result.deviceRepairedRows, result.repairedRows);
    region.reuseExpected = true;
    region.last = true;
    ScopedEnvironment requireIndexReuse("PDG_REQUIRE_NEIGHBORHOOD_REUSE",
                                        "1");
    ScopedEnvironment disableDeviceRepair("PDG_DISABLE_NND_DEVICE_REPAIR",
                                          "1");
    ASSERT_TRUE(pdal::pdg_detail::tryCudaNnDistanceColumns(
        *actual, 4U, region, pdg::KnnDistanceMode::Kth,
        /*requireCuda=*/true));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
    {
        EXPECT_EQ(actual->getFieldAs<std::uint8_t>(Id::Classification, point),
                  expected->getFieldAs<std::uint8_t>(Id::Classification,
                                                     point));
        EXPECT_EQ(actual->getFieldAs<double>(Id::NNDistance, point),
                  expected->getFieldAs<double>(Id::NNDistance, point));
    }

    pdal::PointTable hostTable;
    pdal::PointViewPtr hostActual = populate(hostTable);
    pdal::pdg_detail::CudaNeighborhoodRegion hostRegion;
    hostRegion.id = 2U;
    hostRegion.maximumNeighbors = 4U;
    hostRegion.last = true;
    pdal::pdg_detail::CudaStatisticalOutlierResult hostResult;
    {
        ScopedEnvironment noGatherProof("PDG_REQUIRE_KNN_GATHER_REUSE",
                                        nullptr);
        ScopedEnvironment disableOutlierDeviceRepair(
            "PDG_DISABLE_OUTLIER_DEVICE_REPAIR", "1");
        ASSERT_TRUE(pdal::pdg_detail::tryCudaStatisticalOutlier(
            *hostActual, 4U, 1.25, 18U, hostRegion, hostResult,
            /*requireCuda=*/true));
    }
    EXPECT_GT(hostResult.hostRepairedRows, 0U);
    EXPECT_EQ(hostResult.deviceRepairedRows, 0U);
    EXPECT_EQ(hostResult.repairedRows, hostResult.hostRepairedRows);
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        EXPECT_EQ(hostActual->getFieldAs<std::uint8_t>(
                      Id::Classification, point),
                  expected->getFieldAs<std::uint8_t>(Id::Classification,
                                                     point));

    pdal::PointTable rejectedTable;
    pdal::PointViewPtr rejected = populate(rejectedTable);
    hostRegion.id = 3U;
    {
        ScopedEnvironment noGatherProof("PDG_REQUIRE_KNN_GATHER_REUSE",
                                        nullptr);
        ScopedEnvironment requireOutlierDeviceRepair(
            "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR", "1");
        ScopedEnvironment disableOutlierDeviceRepair(
            "PDG_DISABLE_OUTLIER_DEVICE_REPAIR", "1");
        EXPECT_THROW(static_cast<void>(
                         pdal::pdg_detail::tryCudaStatisticalOutlier(
                             *rejected, 4U, 1.25, 18U, hostRegion, hostResult,
                             /*requireCuda=*/true)),
                     std::runtime_error);
    }
}

TEST(PdgNnDistanceFilter,
     StatisticalOutlierDeclinesWiderGatherForSmallViewExactly)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    std::vector<Point> points;
    for (std::size_t point = 0U; point < 10U; ++point)
        points.push_back({static_cast<double>(point),
                          static_cast<double>(point % 3U),
                          static_cast<double>(point % 2U)});
    const auto populate = [&](pdal::PointTable& table)
    {
        using pdal::Dimension::Id;
        table.layout()->registerDims(
            {Id::X, Id::Y, Id::Z, Id::Classification});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const Point& value = points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value[0]);
            view->setField(Id::Y, point, value[1]);
            view->setField(Id::Z, point, value[2]);
            view->setField(Id::Classification, point, std::uint8_t{2U});
        }
        return view;
    };

    pdal::PointTable oracleTable;
    pdal::PointViewPtr expected = populate(oracleTable);
    pdal::BufferReader oracleReader;
    oracleReader.addView(expected);
    pdal::StageFactory factory;
    pdal::Stage* oracle = factory.createStage("filters.outlier");
    ASSERT_NE(oracle, nullptr);
    pdal::Options options;
    options.add("method", "statistical");
    options.add("mean_k", 8U);
    options.add("multiplier", 1.25);
    options.add("class", 18U);
    oracle->setOptions(options);
    oracle->setInput(oracleReader);
    oracle->prepare(oracleTable);
    static_cast<void>(oracle->execute(oracleTable));

    pdal::PointTable candidateTable;
    pdal::PointViewPtr actual = populate(candidateTable);
    pdal::pdg_detail::CudaNeighborhoodRegion region;
    region.id = 1U;
    region.maximumNeighbors = 11U;
    region.gatherNeighbors = 11U;
    region.last = false;
    pdal::pdg_detail::CudaStatisticalOutlierResult result;
    ASSERT_TRUE(pdal::pdg_detail::tryCudaStatisticalOutlier(
        *actual, 9U, 1.25, 18U, region, result, /*requireCuda=*/true));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        EXPECT_EQ(actual->getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point),
                  expected->getFieldAs<std::uint8_t>(
                      pdal::Dimension::Id::Classification, point));
    pdal::pdg_detail::clearCudaNeighborhood(*actual);
}

TEST(PdgNnDistanceFilter, StatisticalOutlierRefreshesResidentClassification)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    std::vector<Point> points;
    for (std::size_t point = 0U; point < 14U; ++point)
        points.push_back(
            {static_cast<double>(point) * 0.3,
             static_cast<double>(point % 3U) * 0.2,
             static_cast<double>(point % 5U) * 0.1});
    points.push_back({20.0, 20.0, 20.0});
    using pdal::Dimension::Id;
    const auto populate = [&](pdal::PointTable& table)
    {
        table.layout()->registerDims(
            {Id::X, Id::Y, Id::Z, Id::Classification});
        pdal::PointViewPtr view(new pdal::PointView(table));
        for (pdal::PointId point = 0U; point < points.size(); ++point)
        {
            const Point& value = points.at(static_cast<std::size_t>(point));
            view->setField(Id::X, point, value[0]);
            view->setField(Id::Y, point, value[1]);
            view->setField(Id::Z, point, value[2]);
            view->setField(Id::Classification, point, std::uint8_t{2U});
        }
        return view;
    };

    pdal::PointTable oracleTable;
    pdal::PointViewPtr expected = populate(oracleTable);
    pdal::BufferReader oracleReader;
    oracleReader.addView(expected);
    pdal::StageFactory factory;
    pdal::Stage* firstVote =
        factory.createStage("filters.neighborclassifier");
    pdal::Stage* outlier = factory.createStage("filters.outlier");
    pdal::Stage* secondVote =
        factory.createStage("filters.neighborclassifier");
    ASSERT_NE(firstVote, nullptr);
    ASSERT_NE(outlier, nullptr);
    ASSERT_NE(secondVote, nullptr);
    pdal::Options voteOptions;
    voteOptions.add("k", 1);
    firstVote->setOptions(voteOptions);
    secondVote->setOptions(voteOptions);
    pdal::Options outlierOptions;
    outlierOptions.add("method", "statistical");
    outlierOptions.add("mean_k", 4);
    outlierOptions.add("multiplier", 1.25);
    outlierOptions.add("class", 18U);
    outlier->setOptions(outlierOptions);
    firstVote->setInput(oracleReader);
    outlier->setInput(*firstVote);
    secondVote->setInput(*outlier);
    secondVote->prepare(oracleTable);
    static_cast<void>(secondVote->execute(oracleTable));
    bool labeledOutlier = false;
    for (pdal::PointId point = 0U; point < expected->size(); ++point)
        labeledOutlier =
            labeledOutlier ||
            expected->getFieldAs<std::uint8_t>(Id::Classification, point) ==
                18U;
    ASSERT_TRUE(labeledOutlier);

    pdal::PointTable candidateTable;
    pdal::PointViewPtr actual = populate(candidateTable);
    pdal::pdg_detail::CudaNeighborhoodRegion region;
    region.id = 1U;
    region.maximumNeighbors = 5U;
    region.last = false;
    ASSERT_TRUE(pdal::pdg_detail::tryCudaNeighborClassifierColumn(
        *actual, 1U, region, /*requireCuda=*/true));
    region.reuseExpected = true;
    pdal::pdg_detail::CudaStatisticalOutlierResult result;
    ASSERT_TRUE(pdal::pdg_detail::tryCudaStatisticalOutlier(
        *actual, 5U, 1.25, 18U, region, result, /*requireCuda=*/true));
    EXPECT_GT(result.inliers, 0U);
    EXPECT_GT(result.outliers, 0U);
    region.last = true;
    ScopedEnvironment requireReuse("PDG_REQUIRE_NEIGHBORHOOD_REUSE", "1");
    ASSERT_TRUE(pdal::pdg_detail::tryCudaNeighborClassifierColumn(
        *actual, 1U, region, /*requireCuda=*/true));
    for (pdal::PointId point = 0U; point < points.size(); ++point)
        EXPECT_EQ(actual->getFieldAs<std::uint8_t>(Id::Classification, point),
                  expected->getFieldAs<std::uint8_t>(Id::Classification,
                                                     point));
}

} // unnamed namespace

TEST(PdgNnDistanceFilter, RequiredGatherReuseFailsClosedWithoutProducer)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    const std::vector<Point> points{{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {2.0, 0.0, 0.0},
                                    {3.0, 0.0, 0.0}};
    ScopedEnvironment requireGatherReuse("PDG_REQUIRE_KNN_GATHER_REUSE",
                                         "1");
    EXPECT_THROW(static_cast<void>(runResidentNnDistance(points, false, 3U)),
                 std::runtime_error);
}

TEST(PdgNnDistanceFilter, ForcedIncompleteRowsRepairExactlyAndOnlyWithStats)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    // The distant point cannot prove three neighbors in a one-shell grid
    // search. The exact path must repair it with upstream's KD3 ordering.
    const std::vector<Point> points{{0.0, 0.0, 30.0},
                                    {5001.0, 0.0, 10.0},
                                    {5002.0, 0.0, 11.0},
                                    {5003.0, 0.0, 12.0}};
    const std::vector<double> expected = runNnDistanceOracle(points, 3U);
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment enableDeviceRepair("PDG_DISABLE_NND_DEVICE_REPAIR",
                                         nullptr);
    ScopedEnvironment serialDeviceRepair(
        "PDG_DISABLE_NND_PARALLEL_REPAIR", "1");

    const ResidentNnDistanceOutcome ordinary =
        runResidentNnDistance(points, false);
    EXPECT_EQ(ordinary.values, expected);
    EXPECT_EQ(ordinary.phases.incompleteRepairRows, 0U);
    EXPECT_EQ(ordinary.phases.repairedRows, 0U);
    EXPECT_EQ(ordinary.phases.exactHostRepair, 0.0);

    ResidentNnDistanceOutcome observed;
    {
        ScopedEnvironment disableDeviceRepair(
            "PDG_DISABLE_NND_DEVICE_REPAIR", "1");
        observed = runResidentNnDistance(points, true);
    }
    EXPECT_EQ(observed.values, expected);
    EXPECT_GT(observed.phases.incompleteRepairRows, 0U);
    EXPECT_EQ(observed.phases.repairedRows,
              observed.phases.incompleteRepairRows);
    EXPECT_GT(observed.phases.exactHostRepair, 0.0);
    ASSERT_TRUE(observed.manager.complete);
    EXPECT_GT(observed.manager.detail.nnDistance.statusScanAndRepair, 0.0);
    EXPECT_GE(observed.manager.detail.nnDistance.statusScanAndRepair,
              observed.phases.exactHostRepair);

    const ResidentNnDistanceOutcome automatic =
        runResidentNnDistance(points, true);
    EXPECT_EQ(automatic.values, expected);
    EXPECT_EQ(automatic.phases.incompleteRepairRows, 0U);
    EXPECT_EQ(automatic.phases.repairedRows, 0U);
    EXPECT_EQ(automatic.phases.exactHostRepair, 0.0);
    EXPECT_GT(automatic.phases.deviceIncompleteRepairRows, 0U);
    EXPECT_EQ(automatic.phases.deviceRepairRows,
              automatic.phases.deviceIncompleteRepairRows);
    EXPECT_GT(automatic.phases.exactDeviceRepair, 0.0);

    ScopedEnvironment requireDeviceRepair("PDG_REQUIRE_NND_DEVICE_REPAIR",
                                          "1");
    const ResidentNnDistanceOutcome deviceWithoutStats =
        runResidentNnDistance(points, false);
    EXPECT_EQ(deviceWithoutStats.values, expected);
    EXPECT_EQ(deviceWithoutStats.phases.deviceIncompleteRepairRows, 0U);
    EXPECT_EQ(deviceWithoutStats.phases.deviceRepairRows, 0U);
    EXPECT_EQ(deviceWithoutStats.phases.exactDeviceRepair, 0.0);

    const ResidentNnDistanceOutcome device =
        runResidentNnDistance(points, true);
    EXPECT_EQ(device.values, expected);
    EXPECT_EQ(device.phases.incompleteRepairRows, 0U);
    EXPECT_EQ(device.phases.repairedRows, 0U);
    EXPECT_EQ(device.phases.exactHostRepair, 0.0);
    EXPECT_EQ(device.phases.deviceRepairRows,
              device.phases.deviceIncompleteRepairRows);
    EXPECT_GT(device.phases.deviceIncompleteRepairRows, 0U);
    EXPECT_GT(device.phases.exactDeviceRepair, 0.0);
}

TEST(PdgNnDistanceFilter, SelectiveDeviceRepairMatchesExactBoundaries)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);
    ScopedEnvironment requireDeviceRepair("PDG_REQUIRE_NND_DEVICE_REPAIR",
                                          "1");
    ScopedEnvironment enableDeviceRepair("PDG_DISABLE_NND_DEVICE_REPAIR",
                                         nullptr);
    ScopedEnvironment parallelRepair("PDG_DISABLE_NND_PARALLEL_REPAIR",
                                     nullptr);
    ScopedEnvironment requireParallelRepair(
        "PDG_REQUIRE_NND_PARALLEL_REPAIR", "1");

    const auto expectExactDeviceRepair = [](const std::vector<Point>& points,
                                            std::size_t k)
    {
        const std::vector<double> expected = runNnDistanceOracle(points, k);
        const ResidentNnDistanceOutcome observed =
            runResidentNnDistance(points, true, k);
        EXPECT_EQ(observed.values, expected);
        EXPECT_EQ(observed.phases.incompleteRepairRows, 0U);
        EXPECT_EQ(observed.phases.repairedRows, 0U);
        EXPECT_EQ(observed.phases.exactHostRepair, 0.0);
        EXPECT_GT(observed.phases.deviceIncompleteRepairRows, 0U);
        EXPECT_EQ(observed.phases.deviceRepairRows,
                  observed.phases.deviceIncompleteRepairRows);
        EXPECT_EQ(observed.phases.nnDistanceDeviceRepairRows,
                  observed.phases.nnDistanceDeviceIncompleteRepairRows);
        EXPECT_EQ(observed.phases.nnDistanceParallelDeviceRepairRows,
                  observed.phases.nnDistanceDeviceRepairRows);
        EXPECT_GT(observed.phases.nnDistanceParallelDeviceRepairRows, 0U);
        EXPECT_GT(observed.phases.exactDeviceRepair, 0.0);
    };

    expectExactDeviceRepair({{0.125, -0.375, 30.1},
                             {5001.0625, 0.2, 10.3},
                             {5002.1875, -0.4, 11.7}},
                            1U);

    const double offset = 1000000000000.0;
    expectExactDeviceRepair({{offset + 0.125, -offset + 0.375, 30.1},
                             {offset + 5001.0625, -offset + 0.2, 10.3},
                             {offset + 5002.1875, -offset - 0.4, 11.7},
                             {offset + 5000.0, -offset + 1.1, 12.9},
                             {offset + 5000.0, -offset - 1.1, 12.9}},
                            3U);

    std::vector<Point> maximumK{{0.125, -0.375, 30.1}};
    for (std::size_t point = 0U; point < 15U; ++point)
        maximumK.push_back(
            {5001.0 + static_cast<double>(point),
             0.125 * static_cast<double>(point % 3U),
             10.25 + 0.5 * static_cast<double>(point)});
    expectExactDeviceRepair(maximumK, 15U);

    std::vector<Point> multiplePartitions{{0.125, -0.375, 30.1}};
    for (std::size_t point = 0U; point < 128U; ++point)
        multiplePartitions.push_back(
            {5001.0 + 0.03125 * static_cast<double>(point),
             0.0625 * static_cast<double>(point % 11U),
             10.25 + 0.125 * static_cast<double>(point % 17U)});
    expectExactDeviceRepair(multiplePartitions, 15U);

    std::vector<Point> cappedPartitions{{0.125, -0.375, 30.1}};
    for (std::size_t point = 0U; point < 16384U; ++point)
        cappedPartitions.push_back(
            {5001.0 + 0.00025 * static_cast<double>(point),
             0.0005 * static_cast<double>(point % 127U),
             10.25 + 0.000125 * static_cast<double>(point % 251U)});
    expectExactDeviceRepair(cappedPartitions, 15U);
}

TEST(PdgNnDistanceFilter, SelectiveDeviceRepairFailsClosedOutsideEnvelope)
{
    if (!cudaDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    const std::vector<Point> points{{0.0, 0.0, 30.0},
                                    {5001.0, 0.0, 10.0},
                                    {5002.0, 0.0, 11.0},
                                    {5003.0, 0.0, 12.0}};
    ScopedEnvironment shellBudget("PDG_KNN_DEVICE_SHELL_BUDGET", "1");
    ScopedEnvironment forceGrid("PDG_FORCE_UNIFORM_GRID", "1");
    ScopedEnvironment forceBvh("PDG_FORCE_MORTON_BVH", nullptr);

    {
        ScopedEnvironment requireParallelRepair(
            "PDG_REQUIRE_NND_PARALLEL_REPAIR", "1");
        ScopedEnvironment parallelRepair(
            "PDG_DISABLE_NND_PARALLEL_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(points, true),
                     std::runtime_error);
    }
    {
        std::vector<Point> completePoints;
        for (double z : {0.0, 1.125})
            for (double y : {0.0, 1.25})
                for (double x : {0.0, 1.5})
                    completePoints.push_back({x, y, z});
        ScopedEnvironment completeShellBudget(
            "PDG_KNN_DEVICE_SHELL_BUDGET", "4096");
        ScopedEnvironment requireParallelRepair(
            "PDG_REQUIRE_NND_PARALLEL_REPAIR", "1");
        ScopedEnvironment parallelRepair(
            "PDG_DISABLE_NND_PARALLEL_REPAIR", nullptr);
        EXPECT_THROW(runResidentNnDistance(completePoints, true, 3U),
                     std::runtime_error);
    }
    {
        ScopedEnvironment requireDeviceRepair(
            "PDG_REQUIRE_NND_DEVICE_REPAIR", "1");
        ScopedEnvironment disableDeviceRepair(
            "PDG_DISABLE_NND_DEVICE_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(points, true),
                     std::runtime_error);
    }
    {
        ScopedEnvironment requireDeviceRepair(
            "PDG_REQUIRE_NND_DEVICE_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(points, true, 3U, "avg"),
                     std::runtime_error);
    }

    std::vector<Point> tooManyNeighbors{{0.0, 0.0, 30.0}};
    for (std::size_t point = 0U; point < 16U; ++point)
        tooManyNeighbors.push_back(
            {5001.0 + static_cast<double>(point), 0.0,
             10.0 + static_cast<double>(point)});
    const std::vector<double> expected =
        runNnDistanceOracle(tooManyNeighbors, 16U);
    const ResidentNnDistanceOutcome host =
        runResidentNnDistance(tooManyNeighbors, true, 16U);
    EXPECT_EQ(host.values, expected);
    EXPECT_GT(host.phases.incompleteRepairRows, 0U);
    EXPECT_EQ(host.phases.repairedRows, host.phases.incompleteRepairRows);
    EXPECT_EQ(host.phases.deviceRepairRows, 0U);
    EXPECT_GT(host.phases.exactHostRepair, 0.0);
    {
        ScopedEnvironment requireDeviceRepair(
            "PDG_REQUIRE_NND_DEVICE_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(tooManyNeighbors, true, 16U),
                     std::runtime_error);
    }

    std::vector<Point> tooManyRows;
    for (double z : {0.0, 10000.0})
        for (double y : {0.0, 5000.0, 10000.0})
            for (double x : {0.0, 5000.0, 10000.0})
                tooManyRows.push_back({x, y, z});
    const std::vector<double> rowExpected =
        runNnDistanceOracle(tooManyRows, 1U);
    const ResidentNnDistanceOutcome rowHost =
        runResidentNnDistance(tooManyRows, true, 1U);
    EXPECT_EQ(rowHost.values, rowExpected);
    EXPECT_GT(rowHost.phases.incompleteRepairRows, 16U);
    EXPECT_EQ(rowHost.phases.repairedRows,
              rowHost.phases.incompleteRepairRows);
    EXPECT_EQ(rowHost.phases.deviceRepairRows, 0U);
    {
        ScopedEnvironment requireDeviceRepair(
            "PDG_REQUIRE_NND_DEVICE_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(tooManyRows, true, 1U),
                     std::runtime_error);
    }

    {
        ScopedEnvironment noGrid("PDG_FORCE_UNIFORM_GRID", nullptr);
        ScopedEnvironment requireBvh("PDG_FORCE_MORTON_BVH", "1");
        ScopedEnvironment requireDeviceRepair(
            "PDG_REQUIRE_NND_DEVICE_REPAIR", "1");
        EXPECT_THROW(runResidentNnDistance(points, true),
                     std::runtime_error);
    }
}
