#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/RasterGrid.hpp>
#include <pdg/stages/Pmf.hpp>

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

constexpr std::size_t Width = 65U;
constexpr double CellSize = 1.0;
constexpr double MaxWindowSize = 6.0;
constexpr std::size_t DefaultWarmups = 1U;
constexpr std::size_t DefaultIterations = 5U;
constexpr std::size_t DefaultSparsePoints = 25U;

constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);
constexpr pdg::DimensionId
    Classification(pdg::StandardDimension::Classification);
constexpr pdg::DimensionId ReturnNumber(pdg::StandardDimension::ReturnNumber);
constexpr pdg::DimensionId
    NumberOfReturns(pdg::StandardDimension::NumberOfReturns);

struct Options
{
    std::size_t warmups = DefaultWarmups;
    std::size_t iterations = DefaultIterations;
    std::filesystem::path output;
    bool sparse = false;
    std::size_t width = Width;
    std::size_t sparsePoints = DefaultSparsePoints;
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

struct CandidateRun
{
    double wallMilliseconds = 0.0;
    double rasterBuildMilliseconds = 0.0;
    double deviceBackingMaterializeMilliseconds = 0.0;
    double h2dMilliseconds = 0.0;
    double sourceH2dMilliseconds = 0.0;
    double classificationH2dMilliseconds = 0.0;
    double classifyMilliseconds = 0.0;
    double d2hMilliseconds = 0.0;
    pdg::PmfRasterBuildFacts rasterBuildFacts;
    pdg::PmfTiledExecutionFacts executionFacts;
    PmfSnapshot snapshot;
};

struct CudaEvent
{
    CudaEvent()
    {
        PDG_CUDA_CHECK(cudaEventCreate(&value));
    }

    ~CudaEvent()
    {
        if (value)
            PDG_CUDA_CHECK_NOEXCEPT(cudaEventDestroy(value));
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t value = nullptr;
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
            std::cout << "Usage: pdg_pmf_tiled_benchmark [options]\n"
                      << "  --warmups N --iterations N --fixture dense|sparse\n"
                      << "  --width N --sparse-points N\n"
                      << "  --output FILE\n"
                      << "  (refuses to overwrite output file)\n";
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
        else if (option == "--width")
            options.width = parseSize(value, option);
        else if (option == "--sparse-points")
            options.sparsePoints = parseSize(value, option);
        else if (option == "--fixture")
        {
            if (value == "dense")
                options.sparse = false;
            else if (value == "sparse")
                options.sparse = true;
            else
                throw std::invalid_argument(
                    "--fixture must be dense or sparse");
        }
        else
            throw std::invalid_argument("unknown option " +
                                        std::string(option));
    }
    if (options.warmups == 0U || options.iterations == 0U ||
        options.width < 2U || options.sparsePoints < 4U)
        throw std::invalid_argument(
            "warmups/iterations must be positive, width at least two, and "
            "sparse-points at least four");
    if (options.width >
        (std::numeric_limits<std::size_t>::max)() / options.width)
        throw std::invalid_argument("fixture width overflows its cell count");
    if (options.sparse && options.sparsePoints > options.width * options.width)
        throw std::invalid_argument(
            "sparse-points cannot exceed the raster cell count");
    return options;
}

PmfFixture buildFixture(std::size_t width)
{
    PmfFixture fixture;
    const std::size_t pointCount = width * width;
    fixture.x.reserve(pointCount);
    fixture.y.reserve(pointCount);
    fixture.z.reserve(pointCount);
    fixture.classification.reserve(pointCount);
    fixture.returnNumber.reserve(pointCount);
    fixture.numberOfReturns.reserve(pointCount);

    constexpr double OriginX = -7.0;
    constexpr double OriginY = 3.5;
    for (std::size_t column = 0U; column < width; ++column)
        for (std::size_t row = 0U; row < width; ++row)
        {
            const std::size_t hash = column * 13U + row * 7U;
            fixture.x.push_back(OriginX + static_cast<double>(column));
            fixture.y.push_back(OriginY + static_cast<double>(row));
            fixture.z.push_back(
                static_cast<double>(hash % 23U) * 0.125 +
                ((column == width / 2U && row == width / 2U) ? 8.0 : 0.0));
            fixture.classification.push_back(7U);
            fixture.returnNumber.push_back(1U);
            fixture.numberOfReturns.push_back(1U);
        }
    return fixture;
}

PmfFixture buildSparseFixture(std::size_t width, std::size_t sparsePoints)
{
    PmfFixture fixture;
    fixture.x.reserve(sparsePoints);
    fixture.y.reserve(sparsePoints);
    fixture.z.reserve(sparsePoints);
    fixture.classification.reserve(sparsePoints);
    fixture.returnNumber.reserve(sparsePoints);
    fixture.numberOfReturns.reserve(sparsePoints);

    for (std::size_t point = 0U; point < sparsePoints; ++point)
    {
        std::size_t column = (point * 17U) % width;
        std::size_t row = (point * 29U) % width;
        if (point == 0U)
            column = row = 0U;
        else if (point == 1U)
            column = row = width - 1U;
        else if (point == 2U)
        {
            column = 0U;
            row = width - 1U;
        }
        else if (point == 3U)
        {
            column = width - 1U;
            row = 0U;
        }
        fixture.x.push_back(-7.0 + static_cast<double>(column));
        fixture.y.push_back(3.5 + static_cast<double>(row));
        // Lattice sites of opposite checkerboard parity cannot be exactly
        // equidistant from an integer raster cell.  Giving each parity a
        // distinct value therefore makes the fill heterogeneous while every
        // representable nearest tie still has identical source bits.
        fixture.z.push_back(((column + row) & 1U) != 0U ? 1.0 : -0.0);
        fixture.classification.push_back(7U);
        fixture.returnNumber.push_back(1U);
        fixture.numberOfReturns.push_back(1U);
    }
    return fixture;
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
    for (pdal::PointId point = 0; point < pointCount; ++point)
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

PmfSnapshot makeFixtureSnapshot(const PmfFixture& fixture)
{
    return PmfSnapshot{.x = fixture.x,
                       .y = fixture.y,
                       .z = fixture.z,
                       .classification = fixture.classification,
                       .returnNumber = fixture.returnNumber,
                       .numberOfReturns = fixture.numberOfReturns};
}

pdal::PointViewPtr buildUpstreamView(const PmfFixture& fixture,
                                     pdal::PointTable& table)
{
    using pdal::Dimension::Id;
    table.layout()->registerDims({Id::X, Id::Y, Id::Z, Id::Classification,
                                  Id::ReturnNumber, Id::NumberOfReturns});
    pdal::PointViewPtr view(new pdal::PointView(table));
    for (pdal::PointId point = 0; point < fixture.x.size(); ++point)
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

PmfSnapshot runUpstream(const pdal::Options& options, const PmfFixture& fixture)
{
    pdal::PointTable table;
    pdal::PointViewPtr view = buildUpstreamView(fixture, table);

    pdal::BufferReader reader;
    reader.addView(view);
    pdal::StageFactory factory;
    pdal::Stage* filter = factory.createStage("filters.pmf");
    if (!filter)
        throw std::runtime_error("filters.pmf stage is unavailable");
    filter->setOptions(options);
    filter->setInput(reader);
    filter->prepare(table);
    filter->execute(table);
    return snapshotFromView(*view);
}

CandidateRun runCandidate(const PmfFixture& fixture, pdg::PointBatch& host,
                          pdg::PointBatch& device,
                          const pdg::RasterGridFrame& frame,
                          const pdg::RasterGridProductConfig& productConfig,
                          const pdg::PmfProgram& program)
{
    const std::size_t pointCount = fixture.x.size();
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    auto* hX = host.data<double>(X);
    auto* hY = host.data<double>(Y);
    auto* hZ = host.data<double>(Z);
    auto* hClassification = host.data<std::uint8_t>(Classification);
    for (std::size_t point = 0U; point < pointCount; ++point)
    {
        hX[point] = fixture.x[point];
        hY[point] = fixture.y[point];
        hZ[point] = fixture.z[point];
        hClassification[point] = fixture.classification[point];
    }

    const Clock::time_point wallStart = Clock::now();
    pdg::RasterGridProduct product(frame, productConfig, host.memoryResource(),
                                   device.memoryResource());
    if (!product.canMaterializeResidentDeviceBackings())
        throw std::runtime_error(
            "planner-like raster budget did not admit device phase backings");
    if (!product.canMaterializeDeviceProofWorkspace())
        throw std::runtime_error(
            "planner-like raster budget did not admit device proof workspace");

    CandidateRun result;
    CudaEvent sourceH2dStart;
    CudaEvent sourceH2dStop;
    CudaEvent classificationH2dStart;
    CudaEvent classificationH2dStop;
    CudaEvent classifyStart;
    CudaEvent classifyStop;
    CudaEvent d2hStart;
    CudaEvent d2hStop;
    PDG_CUDA_CHECK(cudaEventRecord(sourceH2dStart.value, stream));
    for (const auto& item :
         std::array{std::pair{X, pointCount * sizeof(double)},
                    std::pair{Y, pointCount * sizeof(double)},
                    std::pair{Z, pointCount * sizeof(double)}})
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(item.first),
                                       host.rawData(item.first), item.second,
                                       cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaEventRecord(sourceH2dStop.value, stream));

    const Clock::time_point rasterBuildStart = Clock::now();
    pdg::buildPmfTiledRasterDevice(device, program, product,
                                   &result.rasterBuildFacts);
    if (!result.rasterBuildFacts.deviceNativeSourceBuild ||
        !result.rasterBuildFacts.usedDeviceTieProof ||
        !product.deviceRasterBuild())
        throw std::runtime_error(
            "candidate did not execute its device-native raster proof");
    result.rasterBuildMilliseconds =
        milliseconds(rasterBuildStart, Clock::now());
    const Clock::time_point materializeStart = Clock::now();
    if (!product.materializeResidentDeviceBackings())
        throw std::runtime_error(
            "planner-owned device phase backings were not materialized");
    result.deviceBackingMaterializeMilliseconds =
        milliseconds(materializeStart, Clock::now());

    PDG_CUDA_CHECK(cudaEventRecord(classificationH2dStart.value, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(Classification),
                                   host.rawData(Classification), pointCount,
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaEventRecord(classificationH2dStop.value, stream));

    PDG_CUDA_CHECK(cudaEventRecord(classifyStart.value, stream));
    static_cast<void>(pdg::classifyPmfTiledDevice(device, program, product,
                                                  &result.executionFacts));
    PDG_CUDA_CHECK(cudaEventRecord(classifyStop.value, stream));

    PDG_CUDA_CHECK(cudaEventRecord(d2hStart.value, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(
        host.rawData(Classification), device.rawData(Classification),
        pointCount * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaEventRecord(d2hStop.value, stream));
    PDG_CUDA_CHECK(cudaEventSynchronize(d2hStop.value));
    const Clock::time_point wallEnd = Clock::now();
    if (!result.executionFacts.deviceResidentPhases ||
        !result.executionFacts.deviceNativeRaster ||
        result.executionFacts.rasterHostToDeviceTransfers != 0U ||
        result.executionFacts.rasterHostToDeviceBytes != 0U ||
        result.executionFacts.rasterDeviceToHostTransfers != 0U ||
        result.executionFacts.rasterDeviceToHostBytes != 0U)
        throw std::runtime_error(
            "candidate did not retain its full raster phases on device");

    float elapsed = 0.0F;
    PDG_CUDA_CHECK(cudaEventElapsedTime(&elapsed, sourceH2dStart.value,
                                        sourceH2dStop.value));
    result.sourceH2dMilliseconds = static_cast<double>(elapsed);
    PDG_CUDA_CHECK(cudaEventElapsedTime(&elapsed, classificationH2dStart.value,
                                        classificationH2dStop.value));
    result.classificationH2dMilliseconds = static_cast<double>(elapsed);
    result.h2dMilliseconds =
        result.sourceH2dMilliseconds + result.classificationH2dMilliseconds;
    PDG_CUDA_CHECK(cudaEventElapsedTime(&elapsed, classifyStart.value,
                                        classifyStop.value));
    result.classifyMilliseconds = static_cast<double>(elapsed);
    PDG_CUDA_CHECK(
        cudaEventElapsedTime(&elapsed, d2hStart.value, d2hStop.value));
    result.d2hMilliseconds = static_cast<double>(elapsed);
    result.wallMilliseconds = milliseconds(wallStart, wallEnd);

    // PMF publishes only Classification. XYZ remain the host staging inputs,
    // while the return fields are fixture invariants used by the upstream
    // oracle but are not materialized in this direct primitive benchmark.
    result.snapshot = makeFixtureSnapshot(fixture);
    std::copy_n(host.data<std::uint8_t>(Classification), pointCount,
                result.snapshot.classification.data());
    return result;
}

nlohmann::json summarize(const std::vector<double>& samples)
{
    if (samples.empty())
        return {{"count", 0U}};
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return {{"count", sorted.size()},
            {"median_ms", sorted[sorted.size() / 2U]},
            {"min_ms", sorted.front()},
            {"max_ms", sorted.back()}};
}

nlohmann::json summarizeCandidate(const std::vector<CandidateRun>& samples)
{
    std::vector<double> wall;
    std::vector<double> rasterBuild;
    std::vector<double> deviceBackingMaterialize;
    std::vector<double> h2d;
    std::vector<double> sourceH2d;
    std::vector<double> classificationH2d;
    std::vector<double> classify;
    std::vector<double> d2h;
    wall.reserve(samples.size());
    rasterBuild.reserve(samples.size());
    deviceBackingMaterialize.reserve(samples.size());
    h2d.reserve(samples.size());
    sourceH2d.reserve(samples.size());
    classificationH2d.reserve(samples.size());
    classify.reserve(samples.size());
    d2h.reserve(samples.size());

    for (const CandidateRun& sample : samples)
    {
        wall.push_back(sample.wallMilliseconds);
        rasterBuild.push_back(sample.rasterBuildMilliseconds);
        deviceBackingMaterialize.push_back(
            sample.deviceBackingMaterializeMilliseconds);
        h2d.push_back(sample.h2dMilliseconds);
        sourceH2d.push_back(sample.sourceH2dMilliseconds);
        classificationH2d.push_back(sample.classificationH2dMilliseconds);
        classify.push_back(sample.classifyMilliseconds);
        d2h.push_back(sample.d2hMilliseconds);
    }

    return {
        {"wall_ms", summarize(wall)},
        {"raster_build_ms", summarize(rasterBuild)},
        {"device_backing_materialize_ms", summarize(deviceBackingMaterialize)},
        {"h2d_ms", summarize(h2d)},
        {"source_xyz_h2d_ms", summarize(sourceH2d)},
        {"classification_h2d_ms", summarize(classificationH2d)},
        {"classify_ms", summarize(classify)},
        {"d2h_ms", summarize(d2h)}};
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

    const std::size_t pointCount = left.classification.size();
    for (std::size_t point = 0U; point < pointCount; ++point)
    {
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
    }
    return true;
}

int execute(const Options& options)
{
    const std::vector<pdg::CudaDeviceSummary> devices = pdg::cudaDevices();
    if (devices.empty())
        throw std::runtime_error("no CUDA device is available");

    const PmfFixture fixture =
        options.sparse ? buildSparseFixture(options.width, options.sparsePoints)
                       : buildFixture(options.width);
    const std::size_t pointCount = fixture.x.size();
    const std::size_t deviceBaseBytes =
        pointCount * pdg::PmfTiledHostStagingBytesPerPoint +
        pdg::PmfTiledDeviceFixedScratchBytes;
    const std::size_t hostBaseBytes =
        pointCount * pdg::PmfTiledHostStagingBytesPerPoint;
    const std::size_t frameCells = options.width * options.width;
    const std::size_t rasterBudgetBytes =
        (std::max)(deviceBaseBytes +
                       frameCells * pdg::PmfTiledDeviceBytesPerCell,
                   hostBaseBytes + frameCells * (pdg::PmfTiledHostBytesPerCell +
                                                 sizeof(double)));

    constexpr double OriginX = -7.0;
    constexpr double OriginY = 3.5;
    const pdg::RasterGridFrame frame{
        OriginX,       OriginY,  options.width,
        options.width, CellSize, pdg::RasterGridFramePolicy::PmfV1};
    pdg::PmfProgram program;
    program.cellSize = CellSize;
    program.maxWindowSize = MaxWindowSize;

    pdal::Options upstreamOptions;
    upstreamOptions.add("cell_size", CellSize);
    upstreamOptions.add("max_window_size", MaxWindowSize);

    pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
    pdg::DimensionRegistry dimensions;
    auto stagingMemory = pdg::makeCudaPinnedMemoryResource();
    auto executionMemory = pdg::makeCudaMemoryResource();
    pdg::PointBatch host(pointCount, encoding, dimensions, *stagingMemory);
    pdg::PointBatch device(pointCount, encoding, dimensions, *executionMemory);
    for (const auto& item :
         std::array{std::pair{X, pdg::DimensionType::Double},
                    std::pair{Y, pdg::DimensionType::Double},
                    std::pair{Z, pdg::DimensionType::Double},
                    std::pair{Classification, pdg::DimensionType::Unsigned8}})
    {
        host.materialize(item.first, item.second);
        device.materialize(item.first, item.second);
    }
    host.setSize(pointCount);
    device.setSize(pointCount);

    const pdg::RasterGridProductConfig productConfig{
        .haloCells = 1U,
        .deviceBytesPerExpandedCell = pdg::PmfTiledDeviceBytesPerCell,
        .deviceBackingCount = 2U,
        .deviceProofBytesPerCell = pdg::PmfTiledDeviceProofBytesPerCell,
        .hostBytesPerCell = pdg::PmfTiledHostBytesPerCell,
        .hostTileBytesPerExpandedCell = sizeof(double),
        .hostBackingCount = 2U,
        .baseDeviceBytes = deviceBaseBytes,
        .baseHostBytes = hostBaseBytes,
        .deviceMemoryBudgetBytes = rasterBudgetBytes,
        .hostMemoryBudgetBytes = rasterBudgetBytes};
    pdg::TiledSchedule schedule;
    std::size_t backingBytes = 0U;
    std::size_t tileScratchBytes = 0U;
    std::size_t totalHostBytes = 0U;
    {
        pdg::RasterGridProduct product(frame, productConfig, *stagingMemory,
                                       *executionMemory);
        schedule = product.schedule();
        backingBytes = product.backingBytes();
        tileScratchBytes = product.tileScratchBytes();
        totalHostBytes = product.totalHostBytes();
        if (!product.canMaterializeResidentDeviceBackings())
            throw std::runtime_error(
                "planner-like raster budget did not fit full device phases");
    }
    if (schedule.tileCount != 1U)
        throw std::runtime_error(
            "planner-like raster budget did not produce one device phase");

    PmfSnapshot upstreamReference = runUpstream(upstreamOptions, fixture);
    std::vector<double> upstreamSamples;
    upstreamSamples.reserve(options.iterations);
    std::vector<CandidateRun> candidateSamples;
    candidateSamples.reserve(options.iterations);
    Difference firstDifference{};
    bool exactMatch = true;

    for (std::size_t warmup = 0U; warmup < options.warmups; ++warmup)
    {
        if ((warmup & 1U) == 0U)
        {
            upstreamReference = runUpstream(upstreamOptions, fixture);
            const CandidateRun candidate = runCandidate(
                fixture, host, device, frame, productConfig, program);
            if (exactMatch &&
                !snapshotsMatch(upstreamReference, candidate.snapshot,
                                firstDifference))
                exactMatch = false;
        }
        else
        {
            const CandidateRun candidate = runCandidate(
                fixture, host, device, frame, productConfig, program);
            if (exactMatch &&
                !snapshotsMatch(upstreamReference, candidate.snapshot,
                                firstDifference))
                exactMatch = false;
            upstreamReference = runUpstream(upstreamOptions, fixture);
        }
    }

    for (std::size_t iteration = 0U; iteration < options.iterations;
         ++iteration)
    {
        if ((iteration & 1U) == 0U)
        {
            const Clock::time_point start = Clock::now();
            upstreamReference = runUpstream(upstreamOptions, fixture);
            upstreamSamples.push_back(milliseconds(start, Clock::now()));
            const CandidateRun candidate = runCandidate(
                fixture, host, device, frame, productConfig, program);
            candidateSamples.push_back(candidate);
            if (exactMatch &&
                !snapshotsMatch(upstreamReference, candidate.snapshot,
                                firstDifference))
                exactMatch = false;
        }
        else
        {
            const CandidateRun candidate = runCandidate(
                fixture, host, device, frame, productConfig, program);
            candidateSamples.push_back(candidate);
            if (exactMatch &&
                !snapshotsMatch(upstreamReference, candidate.snapshot,
                                firstDifference))
                exactMatch = false;
            const Clock::time_point start = Clock::now();
            upstreamReference = runUpstream(upstreamOptions, fixture);
            upstreamSamples.push_back(milliseconds(start, Clock::now()));
        }
    }

    int runtimeVersion = 0;
    int driverVersion = 0;
    PDG_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    PDG_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));

    nlohmann::json report{
        {"schema", "pdg-pmf-tiled-benchmark-v7"},
        {"oracle_commit", PDG_ORACLE_COMMIT},
        {"fixture",
         {{"name", std::to_string(options.width) + "x" +
                       std::to_string(options.width) +
                       (options.sparse ? "_sparse_void" : "_nontrivial_seam")},
          {"points", pointCount},
          {"width", options.width},
          {"height", options.width},
          {"cell_size", CellSize},
          {"max_window_size", MaxWindowSize},
          {"program",
           {{"only_ground", program.onlyGround},
            {"cell_size", program.cellSize},
            {"max_window_size", program.maxWindowSize},
            {"exponential", program.exponential},
            {"slope", program.slope},
            {"ground_class", program.groundClass},
            {"other_class", program.otherClass}}},
          {"grid_semantics",
           {{"minimum_x", OriginX},
            {"minimum_y", OriginY},
            {"rows", options.width},
            {"columns", options.width},
            {"policy", "PmfV1"}}}}},
        {"comparison",
         {{"classification_exact_and_staged_snapshot_consistent", exactMatch},
          {"candidate_output_columns_read_back", {"Classification"}},
          {"candidate_staging_columns", {"X", "Y", "Z"}},
          {"fixture_invariant_columns", {"ReturnNumber", "NumberOfReturns"}},
          {"warmups", options.warmups},
          {"iterations", options.iterations},
          {"alternating_order", true},
          {"upstream_wall_scope", "fixture-to-complete-stage-snapshot"},
          {"candidate_wall_scope",
           "planner-product-source-XYZ-H2D-device-raster-proof-promotion-"
           "classification-H2D-device-phases-classification-D2H"},
          {"candidate_product_setup_timed", true},
          {"candidate_raster_build_timed", true},
          {"candidate_device_backing_materialization_timed", true}}},
        {"timing",
         {{"upstream", summarize(upstreamSamples)},
          {"candidate", summarizeCandidate(candidateSamples)}}},
        {"raster_grid",
         {{"budget_bytes", rasterBudgetBytes},
          {"device_base_bytes", deviceBaseBytes},
          {"host_base_bytes", hostBaseBytes},
          {"frame",
           {{"minimum_x", frame.minimumX},
            {"minimum_y", frame.minimumY},
            {"rows", frame.rows},
            {"columns", frame.columns},
            {"cell_size", frame.cellSize},
            {"policy", "PmfV1"}}},
          {"schedule",
           {{"item_count", schedule.itemCount},
            {"tile_count", schedule.tileCount},
            {"tile_item_capacity", schedule.tileItemCapacity},
            {"configured_lane_count", schedule.configuredLaneCount},
            {"active_lane_count", schedule.activeLaneCount},
            {"peak_lane_bytes", schedule.peakLaneBytes},
            {"lane_reuse_count", schedule.laneReuseCount},
            {"memory_limited", schedule.memoryLimited},
            {"serial_dependency", schedule.serialDependency}}},
          {"backing_bytes", backingBytes},
          {"device_phase_backing_bytes", 2U * backingBytes},
          {"device_phase_resident", true},
          {"device_proof_bytes_per_cell", pdg::PmfTiledDeviceProofBytesPerCell},
          {"device_native_source_build",
           candidateSamples.front().rasterBuildFacts.deviceNativeSourceBuild},
          {"used_device_tie_proof",
           candidateSamples.front().rasterBuildFacts.usedDeviceTieProof},
          {"device_native_raster",
           candidateSamples.front().executionFacts.deviceNativeRaster},
          {"populated_cells",
           candidateSamples.front().rasterBuildFacts.populatedCells},
          {"used_bounded_source_scan",
           candidateSamples.front().rasterBuildFacts.usedBoundedSourceScan},
          {"used_occupancy_hierarchy",
           candidateSamples.front().rasterBuildFacts.usedOccupancyHierarchy},
          {"source_slots_visited",
           candidateSamples.front().rasterBuildFacts.sourceSlotsVisited},
          {"hierarchy_nodes_visited",
           candidateSamples.front().rasterBuildFacts.hierarchyNodesVisited},
          {"raster_host_to_device_transfers",
           candidateSamples.front().executionFacts.rasterHostToDeviceTransfers},
          {"raster_host_to_device_bytes",
           candidateSamples.front().executionFacts.rasterHostToDeviceBytes},
          {"raster_device_to_host_transfers",
           candidateSamples.front().executionFacts.rasterDeviceToHostTransfers},
          {"raster_device_to_host_bytes",
           candidateSamples.front().executionFacts.rasterDeviceToHostBytes},
          {"tile_scratch_bytes", tileScratchBytes},
          {"total_host_bytes", totalHostBytes}}},
        {"device",
         {{"ordinal", devices.front().ordinal},
          {"name", devices.front().name},
          {"compute_capability",
           std::to_string(devices.front().computeMajor) + "." +
               std::to_string(devices.front().computeMinor)},
          {"memory_bytes", devices.front().totalMemory},
          {"runtime_version", runtimeVersion},
          {"driver_version", driverVersion}}}};
    if (!exactMatch)
    {
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
    }

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
    return exactMatch ? 0 : 2;
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
        std::cerr << "pdg pmf tiled benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
