#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Plan.hpp>
#include <pdg/stages/Pmf.hpp>

#include "src/pdal/PdgResidentContext.hpp"

#include <io/BufferReader.hpp>

#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr std::size_t FixtureWidth = 65U;
constexpr double CellSize = 1.0;
constexpr std::size_t DefaultWarmups = 1U;
constexpr std::size_t DefaultIterations = 5U;
constexpr double OriginX = -7.0;
constexpr double OriginY = 3.5;

struct Options
{
    std::size_t warmups = DefaultWarmups;
    std::size_t iterations = DefaultIterations;
    std::filesystem::path output;
};

struct PmfFixture
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<std::uint8_t> classification;
    std::vector<std::uint8_t> returnNumber;
    std::vector<std::uint8_t> numberOfReturns;
};

struct PmfSnapshot
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<std::uint8_t> classification;
    std::vector<std::uint8_t> returnNumber;
    std::vector<std::uint8_t> numberOfReturns;
};

struct CandidateRun
{
    PmfSnapshot snapshot;
    pdg::ExecutionStatsSnapshot stats;
};

struct Difference
{
    std::size_t point = (std::numeric_limits<std::size_t>::max)();
    double upstreamX = 0.0;
    double upstreamY = 0.0;
    double upstreamZ = 0.0;
    double candidateX = 0.0;
    double candidateY = 0.0;
    double candidateZ = 0.0;
    std::uint8_t upstreamClassification = 0U;
    std::uint8_t candidateClassification = 0U;
    std::uint8_t upstreamReturnNumber = 0U;
    std::uint8_t candidateReturnNumber = 0U;
    std::uint8_t upstreamNumberOfReturns = 0U;
    std::uint8_t candidateNumberOfReturns = 0U;
};

double milliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t parseSize(std::string_view value, std::string_view option)
{
    std::size_t parsed = 0U;
    std::size_t position = 0U;
    try
    {
        const unsigned long long wide =
            std::stoull(std::string(value), &position, 10);
        if (position != value.size())
            throw std::out_of_range("invalid");
        parsed = static_cast<std::size_t>(wide);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid value for " + std::string(option));
    }
    return parsed;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option = argv[index];
        if (option == "--help")
        {
            std::cout << "Usage: pdg_pmf_reuse_benchmark [options]\n"
                      << "  --warmups N --iterations N\n"
                      << "  --output FILE  (refuses to overwrite)\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " +
                                        std::string(option));
        const std::string_view value = argv[++index];
        if (option == "--warmups")
            options.warmups = parseSize(value, option);
        else if (option == "--iterations")
            options.iterations = parseSize(value, option);
        else if (option == "--output")
            options.output = value;
        else
            throw std::invalid_argument("unknown option " +
                                        std::string(option));
    }
    if (options.warmups == 0U || options.iterations == 0U)
        throw std::invalid_argument("warmups and iterations must be positive");
    return options;
}

PmfFixture makePmfMorphologyFixture()
{
    PmfFixture fixture;
    const std::size_t pointCount = FixtureWidth * FixtureWidth;
    fixture.x.reserve(pointCount);
    fixture.y.reserve(pointCount);
    fixture.z.reserve(pointCount);
    fixture.classification.reserve(pointCount);
    fixture.returnNumber.reserve(pointCount);
    fixture.numberOfReturns.reserve(pointCount);
    for (std::size_t column = 0U; column < FixtureWidth; ++column)
        for (std::size_t row = 0U; row < FixtureWidth; ++row)
        {
            const std::size_t hash = column * 13U + row * 7U;
            const double z = static_cast<double>(hash % 23U) * 0.125 +
                             ((column == 32U && row == 32U) ? 8.0 : 0.0);
            fixture.x.push_back(OriginX + static_cast<double>(column));
            fixture.y.push_back(OriginY + static_cast<double>(row));
            fixture.z.push_back(z);
            fixture.classification.push_back(7U);
            fixture.returnNumber.push_back(1U);
            fixture.numberOfReturns.push_back(1U);
        }
    return fixture;
}

pdal::PointViewPtr buildView(const PmfFixture& fixture, pdal::PointTable& table)
{
    using pdal::Dimension::Id;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0U; point < fixture.x.size(); ++point)
    {
        const std::size_t index = static_cast<std::size_t>(point);
        view->setField(Id::X, point, fixture.x[index]);
        view->setField(Id::Y, point, fixture.y[index]);
        view->setField(Id::Z, point, fixture.z[index]);
        view->setField(Id::Classification, point,
                       fixture.classification[index]);
        view->setField(Id::ReturnNumber, point, fixture.returnNumber[index]);
        view->setField(Id::NumberOfReturns, point,
                       fixture.numberOfReturns[index]);
    }
    return view;
}

PmfSnapshot snapshotFromView(const pdal::PointView& view)
{
    using pdal::Dimension::Id;
    PmfSnapshot snapshot;
    const pdal::PointId pointCount = view.size();
    snapshot.x.reserve(static_cast<std::size_t>(pointCount));
    snapshot.y.reserve(static_cast<std::size_t>(pointCount));
    snapshot.z.reserve(static_cast<std::size_t>(pointCount));
    snapshot.classification.reserve(static_cast<std::size_t>(pointCount));
    snapshot.returnNumber.reserve(static_cast<std::size_t>(pointCount));
    snapshot.numberOfReturns.reserve(static_cast<std::size_t>(pointCount));
    for (pdal::PointId point = 0U; point < pointCount; ++point)
    {
        snapshot.x.push_back(view.getFieldAs<double>(Id::X, point));
        snapshot.y.push_back(view.getFieldAs<double>(Id::Y, point));
        snapshot.z.push_back(view.getFieldAs<double>(Id::Z, point));
        snapshot.classification.push_back(
            view.getFieldAs<std::uint8_t>(Id::Classification, point));
        snapshot.returnNumber.push_back(
            view.getFieldAs<std::uint8_t>(Id::ReturnNumber, point));
        snapshot.numberOfReturns.push_back(
            view.getFieldAs<std::uint8_t>(Id::NumberOfReturns, point));
    }
    return snapshot;
}

pdal::Options firstPmfOptions()
{
    pdal::Options options;
    options.add("cell_size", CellSize);
    options.add("max_window_size", 3.0);
    options.add("returns", "only");
    return options;
}

pdal::Options secondPmfOptions()
{
    pdal::Options options;
    options.add("cell_size", CellSize);
    options.add("max_window_size", 5.0);
    options.add("returns", "only");
    options.add("only_ground", true);
    options.add("ground_class", 9);
    options.add("other_class", 9);
    return options;
}

PmfSnapshot runUpstream(const PmfFixture& fixture)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildView(fixture, table);
    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage("filters.pmf");
    pdal::Stage* second = factory.createStage("filters.pmf");
    if (!first || !second)
        throw std::runtime_error("filters.pmf stage is unavailable");
    first->setOptions(firstPmfOptions());
    second->setOptions(secondPmfOptions());
    first->setInput(reader);
    second->setInput(*first);
    second->prepare(table);
    const pdal::PointViewSet output = second->execute(table);
    if (output.size() != 1U)
        throw std::runtime_error("PMF differential chain changed view count");
    return snapshotFromView(**output.begin());
}

std::size_t boundaryId(const pdg::Plan& plan, pdg::ResidencyBoundaryKind kind)
{
    const auto position =
        std::find_if(plan.summary().residencyBoundaries.begin(),
                     plan.summary().residencyBoundaries.end(),
                     [&](const pdg::ResidencyBoundary& boundary)
                     { return boundary.kind == kind; });
    if (position == plan.summary().residencyBoundaries.end())
        throw std::runtime_error("resident PMF plan has no matching boundary");
    return static_cast<std::size_t>(
        std::distance(plan.summary().residencyBoundaries.begin(), position));
}

const pdg::ExecutionEventTotals&
eventTotal(const pdg::ExecutionStatsSnapshot& stats,
           pdg::ExecutionEventKind kind)
{
    return stats.totals.at(static_cast<std::size_t>(kind));
}

CandidateRun runCandidate(const PmfFixture& fixture, const pdg::Plan& plan,
                          const pdg::DimensionRegistry& dimensions,
                          std::size_t region, std::size_t upload,
                          std::size_t spill, std::size_t budget)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildView(fixture, table);
    pdal::pdg_detail::ResidentExecutionScope scope(plan, dimensions, budget,
                                                   64U);
    const std::array selectedRegions{region};
    scope.preflight(*table.layout(), view->size(), selectedRegions);
    pdal::pdg_detail::ResidentExecutionContext& context = scope.context();
    pdg::ExecutionObservationScope observation;
    context.enterBoundary(
        *view, upload, pdal::pdg_detail::ResidentBoundaryDirection::Upload,
        region,
        plan.summary().residencyBoundaries.at(upload).requiresFullPointRecord);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* first = factory.createStage(std::string(pdg::HybridPmfStage));
    pdal::Stage* second = factory.createStage(std::string(pdg::HybridPmfStage));
    if (!first || !second)
        throw std::runtime_error("resident PMF wrapper is unavailable");

    pdal::Options firstOptions = firstPmfOptions();
    firstOptions.add("pdg_resident_context", true);
    firstOptions.add("pdg_execution_region",
                     static_cast<std::uint64_t>(region));
    firstOptions.add("pdg_grid_reuse", false);
    firstOptions.add("pdg_grid_region_last", false);
    first->setOptions(firstOptions);
    first->setInput(reader);

    pdal::Options secondOptions = secondPmfOptions();
    secondOptions.add("pdg_resident_context", true);
    secondOptions.add("pdg_execution_region",
                      static_cast<std::uint64_t>(region));
    secondOptions.add("pdg_grid_reuse", true);
    secondOptions.add("pdg_grid_region_last", true);
    second->setOptions(secondOptions);
    second->setInput(*first);
    second->prepare(table);
    static_cast<void>(second->execute(table));

    context.enterBoundary(
        *view, spill, pdal::pdg_detail::ResidentBoundaryDirection::Spill,
        region,
        plan.summary().residencyBoundaries.at(spill).requiresFullPointRecord);

    CandidateRun run;
    run.snapshot = snapshotFromView(*view);
    run.stats = observation.snapshot();
    if (eventTotal(run.stats, pdg::ExecutionEventKind::GridBuild).count != 1U ||
        eventTotal(run.stats, pdg::ExecutionEventKind::RasterBuild).count !=
            2U ||
        eventTotal(run.stats, pdg::ExecutionEventKind::RasterUpload).count !=
            0U ||
        eventTotal(run.stats, pdg::ExecutionEventKind::RasterDownload).count !=
            0U ||
        eventTotal(run.stats, pdg::ExecutionEventKind::DeviceRegionBegin)
                .count != 1U ||
        eventTotal(run.stats, pdg::ExecutionEventKind::DeviceRegionEnd).count !=
            1U)
        throw std::runtime_error(
            "resident PMF wrapper reported an unexpected lifecycle");
    return run;
}

nlohmann::json summarize(const std::vector<double>& values)
{
    if (values.empty())
        return {{"count", 0U}};
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return {{"count", sorted.size()},
            {"median_ms", sorted[sorted.size() / 2U]},
            {"min_ms", sorted.front()},
            {"max_ms", sorted.back()}};
}

bool snapshotsMatch(const PmfSnapshot& left, const PmfSnapshot& right,
                    Difference& mismatch)
{
    if (left.x.size() != right.x.size() || left.y.size() != right.y.size() ||
        left.z.size() != right.z.size() ||
        left.classification.size() != right.classification.size() ||
        left.returnNumber.size() != right.returnNumber.size() ||
        left.numberOfReturns.size() != right.numberOfReturns.size())
    {
        mismatch.point = 0U;
        return false;
    }
    for (std::size_t point = 0U; point < left.classification.size(); ++point)
        if (std::bit_cast<std::uint64_t>(left.x[point]) !=
                std::bit_cast<std::uint64_t>(right.x[point]) ||
            std::bit_cast<std::uint64_t>(left.y[point]) !=
                std::bit_cast<std::uint64_t>(right.y[point]) ||
            std::bit_cast<std::uint64_t>(left.z[point]) !=
                std::bit_cast<std::uint64_t>(right.z[point]) ||
            left.classification[point] != right.classification[point] ||
            left.returnNumber[point] != right.returnNumber[point] ||
            left.numberOfReturns[point] != right.numberOfReturns[point])
        {
            mismatch.point = point;
            mismatch.upstreamX = left.x[point];
            mismatch.upstreamY = left.y[point];
            mismatch.upstreamZ = left.z[point];
            mismatch.candidateX = right.x[point];
            mismatch.candidateY = right.y[point];
            mismatch.candidateZ = right.z[point];
            mismatch.upstreamClassification = left.classification[point];
            mismatch.candidateClassification = right.classification[point];
            mismatch.upstreamReturnNumber = left.returnNumber[point];
            mismatch.candidateReturnNumber = right.returnNumber[point];
            mismatch.upstreamNumberOfReturns = left.numberOfReturns[point];
            mismatch.candidateNumberOfReturns = right.numberOfReturns[point];
            return false;
        }
    return true;
}

int execute(const Options& options)
{
    const std::vector<pdg::CudaDeviceSummary> devices = pdg::cudaDevices();
    if (devices.empty())
        throw std::runtime_error("no CUDA device is available");

    constexpr std::string_view Pipeline = R"([{
      "type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":3.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only","only_ground":true,"ground_class":9,
       "other_class":9},
      {"type":"writers.las","filename":"out.las"}])";
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(Pipeline, dimensions);
    if (plan.summary().residentRegions != 1U)
        throw std::runtime_error("unexpected PMF resident region count");
    const std::size_t region = plan.stages().at(1U).residentRegion;
    if (region == pdg::NoResidentRegion ||
        plan.stages().at(2U).residentRegion != region)
        throw std::runtime_error("PMF stages do not share a resident region");
    const std::size_t upload =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Upload);
    const std::size_t spill =
        boundaryId(plan, pdg::ResidencyBoundaryKind::Spill);

    const PmfFixture fixture = makePmfMorphologyFixture();
    const std::size_t pointCount = fixture.x.size();
    const std::size_t budget =
        plan.estimatedDeviceBytes(pointCount) +
        pointCount * pdg::PmfExactDeviceScratchBytesPerPoint +
        FixtureWidth * FixtureWidth * pdg::PmfTiledDeviceBytesPerCell;

    PmfSnapshot upstreamReference = runUpstream(fixture);
    Difference firstDifference{};
    bool exact = true;
    std::vector<double> upstreamSamples;
    std::vector<double> candidateSamples;
    upstreamSamples.reserve(options.iterations);
    candidateSamples.reserve(options.iterations);
    CandidateRun representative;

    const auto checkCandidate = [&](const CandidateRun& candidate)
    {
        if (!snapshotsMatch(upstreamReference, candidate.snapshot,
                            firstDifference))
            exact = false;
    };
    const auto runAndCheckCandidate = [&]()
    {
        CandidateRun candidate = runCandidate(fixture, plan, dimensions, region,
                                              upload, spill, budget);
        checkCandidate(candidate);
        return candidate;
    };

    for (std::size_t warmup = 0U; warmup < options.warmups; ++warmup)
        if ((warmup & 1U) == 0U)
        {
            upstreamReference = runUpstream(fixture);
            representative = runAndCheckCandidate();
        }
        else
        {
            representative = runAndCheckCandidate();
            upstreamReference = runUpstream(fixture);
        }

    for (std::size_t iteration = 0U; iteration < options.iterations;
         ++iteration)
        if ((iteration & 1U) == 0U)
        {
            const Clock::time_point upstreamStart = Clock::now();
            upstreamReference = runUpstream(fixture);
            upstreamSamples.push_back(
                milliseconds(upstreamStart, Clock::now()));
            const Clock::time_point candidateStart = Clock::now();
            representative = runCandidate(fixture, plan, dimensions, region,
                                          upload, spill, budget);
            candidateSamples.push_back(
                milliseconds(candidateStart, Clock::now()));
            checkCandidate(representative);
        }
        else
        {
            const Clock::time_point candidateStart = Clock::now();
            representative = runCandidate(fixture, plan, dimensions, region,
                                          upload, spill, budget);
            candidateSamples.push_back(
                milliseconds(candidateStart, Clock::now()));
            checkCandidate(representative);
            const Clock::time_point upstreamStart = Clock::now();
            upstreamReference = runUpstream(fixture);
            upstreamSamples.push_back(
                milliseconds(upstreamStart, Clock::now()));
        }

    int runtimeVersion = 0;
    int driverVersion = 0;
    PDG_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    PDG_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));

    const auto eventJson = [&](pdg::ExecutionEventKind kind)
    {
        const pdg::ExecutionEventTotals& total =
            eventTotal(representative.stats, kind);
        return nlohmann::json{{"count", total.count}, {"bytes", total.bytes}};
    };
    nlohmann::json report{
        {"schema", "pdg-pmf-reuse-benchmark-v4"},
        {"oracle",
         {{"commit", PDG_ORACLE_COMMIT},
          {"chain", nlohmann::json::array(
                        {"filters.pmf(max_window_size=3, returns=only)",
                         "filters.pmf(max_window_size=5, returns=only, "
                         "only_ground=true, ground_class=9, "
                         "other_class=9)"})}}},
        {"fixture",
         {{"name", "65x65_morphology_fixture"},
          {"points", pointCount},
          {"width", FixtureWidth},
          {"height", FixtureWidth},
          {"cell_size", CellSize}}},
        {"comparison",
         {{"exact", exact},
          {"alternating_order", true},
          {"warmups", options.warmups},
          {"iterations", options.iterations},
          {"upstream_wall_scope",
           "fixture-view+prepared-two-stage-upstream-chain+complete-snapshot+"
           "worker-raii-teardown"},
          {"candidate_wall_scope",
           "fixture-view+resident-scope/preflight+upload+prepared-two-wrapper-"
           "chain+spill+complete-snapshot+worker-raii-teardown"},
          {"candidate_path", "actual-resident-pmf-wrappers-and-context"},
          {"exact_comparison_outside_timed_scope", true},
          {"candidate_return_selection_executed", true},
          {"candidate_output_columns_read_back",
           {"X", "Y", "Z", "Classification", "ReturnNumber",
            "NumberOfReturns"}}}},
        {"timing",
         {{"upstream", summarize(upstreamSamples)},
          {"candidate", summarize(candidateSamples)}}},
        {"execution",
         {{"resident_region", region},
          {"grid_build", eventJson(pdg::ExecutionEventKind::GridBuild)},
          {"raster_build", eventJson(pdg::ExecutionEventKind::RasterBuild)},
          {"raster_upload", eventJson(pdg::ExecutionEventKind::RasterUpload)},
          {"raster_download",
           eventJson(pdg::ExecutionEventKind::RasterDownload)},
          {"device_region_begin",
           eventJson(pdg::ExecutionEventKind::DeviceRegionBegin)},
          {"device_region_end",
           eventJson(pdg::ExecutionEventKind::DeviceRegionEnd)},
          {"product_count", 1U},
          {"raster_generation_count", 2U},
          {"allocation_reuse_provenance",
           "RasterGrid.ReusesPromotedDeviceAllocationForNextRasterGeneration"},
          {"device_memory_budget_bytes", budget}}},
        {"device",
         {{"ordinal", devices.front().ordinal},
          {"name", devices.front().name},
          {"compute_capability",
           std::to_string(devices.front().computeMajor) + "." +
               std::to_string(devices.front().computeMinor)},
          {"memory_bytes", devices.front().totalMemory},
          {"runtime_version", runtimeVersion},
          {"driver_version", driverVersion}}}};

    if (!exact)
        report["first_difference"] = {
            {"point", firstDifference.point},
            {"upstream",
             {{"x", firstDifference.upstreamX},
              {"y", firstDifference.upstreamY},
              {"z", firstDifference.upstreamZ},
              {"classification", firstDifference.upstreamClassification},
              {"return_number", firstDifference.upstreamReturnNumber},
              {"number_of_returns", firstDifference.upstreamNumberOfReturns}}},
            {"candidate",
             {{"x", firstDifference.candidateX},
              {"y", firstDifference.candidateY},
              {"z", firstDifference.candidateZ},
              {"classification", firstDifference.candidateClassification},
              {"return_number", firstDifference.candidateReturnNumber},
              {"number_of_returns",
               firstDifference.candidateNumberOfReturns}}}};

    const std::string serialized = report.dump(2) + '\n';
    std::cout << serialized;
    if (!options.output.empty())
    {
        if (std::filesystem::exists(options.output))
            throw std::runtime_error("refusing to overwrite benchmark report " +
                                     options.output.string());
        std::ofstream output(options.output, std::ios::binary);
        if (!output)
            throw std::runtime_error("unable to open benchmark report " +
                                     options.output.string());
        output.write(serialized.data(),
                     static_cast<std::streamsize>(serialized.size()));
        if (!output)
            throw std::runtime_error("unable to write benchmark report " +
                                     options.output.string());
    }
    return exact ? 0 : 2;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        return execute(parseOptions(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << "pdg pmf reuse benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
