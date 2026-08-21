#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <nlohmann/json.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

enum class Query
{
    MeanDistance,
    Covariance
};

struct Options
{
    std::size_t points = 262144U;
    std::uint32_t neighbors = 16U;
    std::size_t warmups = 2U;
    std::size_t iterations = 9U;
    std::string profile = "uniform-als";
    std::filesystem::path output;
    Query query = Query::MeanDistance;
};

struct Sample
{
    double buildWallMilliseconds = 0.0;
    double queryGpuMilliseconds = 0.0;
    double queryWallMilliseconds = 0.0;
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

std::uint64_t mix(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double unit(std::uint64_t value) noexcept
{
    return static_cast<double>(mix(value) >> 11U) * 0x1.0p-53;
}

std::size_t parseSize(std::string_view value, std::string_view option)
{
    std::size_t parsed = 0U;
    std::size_t position = 0U;
    try
    {
        const unsigned long long wide =
            std::stoull(std::string(value), &position, 10);
        if (position != value.size() ||
            wide > static_cast<unsigned long long>(
                       (std::numeric_limits<std::size_t>::max)()))
            throw std::out_of_range("size");
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
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string_view option(argv[argument]);
        if (option == "--help")
        {
            std::cout
                << "Usage: pdg_spatial_benchmark [options]\n"
                << "  --profile uniform-als|clustered-tls|mixed-density\n"
                << "  --query mean-distance|covariance\n"
                << "  --points N --neighbors K --warmups N --iterations N\n"
                << "  --output FILE  (refuses to overwrite)\n";
            std::exit(0);
        }
        if (argument + 1 >= argc)
            throw std::invalid_argument("missing value for " +
                                        std::string(option));
        const std::string_view value(argv[++argument]);
        if (option == "--profile")
            options.profile = value;
        else if (option == "--output")
            options.output = value;
        else if (option == "--query")
        {
            if (value == "mean-distance")
                options.query = Query::MeanDistance;
            else if (value == "covariance")
                options.query = Query::Covariance;
            else
                throw std::invalid_argument("invalid query");
        }
        else if (option == "--points")
            options.points = parseSize(value, option);
        else if (option == "--neighbors")
        {
            const std::size_t parsed = parseSize(value, option);
            if (parsed > static_cast<std::size_t>(
                             (std::numeric_limits<std::uint32_t>::max)()))
                throw std::invalid_argument("invalid neighbor count");
            options.neighbors = static_cast<std::uint32_t>(parsed);
        }
        else if (option == "--warmups")
            options.warmups = parseSize(value, option);
        else if (option == "--iterations")
            options.iterations = parseSize(value, option);
        else
            throw std::invalid_argument("unknown option " +
                                        std::string(option));
    }
    if (options.profile != "uniform-als" &&
        options.profile != "clustered-tls" &&
        options.profile != "mixed-density")
        throw std::invalid_argument("invalid profile");
    if (options.points == 0U ||
        options.points >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("point count must be in [1, INT_MAX]");
    if (options.neighbors == 0U || options.neighbors > 64U ||
        options.points < options.neighbors)
        throw std::invalid_argument(
            "neighbors must be in [1, 64] and no larger than points");
    if (options.iterations == 0U)
        throw std::invalid_argument("iterations must be positive");
    return options;
}

void fillUniformAls(pdg::PointBatch& batch)
{
    double* x = batch.data<double>(X);
    double* y = batch.data<double>(Y);
    double* z = batch.data<double>(Z);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::uint64_t id = static_cast<std::uint64_t>(point);
        x[point] = 500000.0 + 1000.0 * unit(id * 3U + 0U);
        y[point] = 4800000.0 + 1000.0 * unit(id * 3U + 1U);
        z[point] =
            100.0 + 35.0 * unit(id * 3U + 2U) + 0.0001 * (x[point] - 500000.0);
    }
}

void fillClusteredTls(pdg::PointBatch& batch)
{
    constexpr std::size_t ClusterCount = 256U;
    double* x = batch.data<double>(X);
    double* y = batch.data<double>(Y);
    double* z = batch.data<double>(Z);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::size_t cluster = point % ClusterCount;
        const std::uint64_t id = static_cast<std::uint64_t>(point);
        const double centerX =
            500000.0 + 60.0 * static_cast<double>(cluster % 16U);
        const double centerY =
            4800000.0 + 60.0 * static_cast<double>(cluster / 16U);
        const double centerZ =
            100.0 + 12.0 * static_cast<double>((cluster * 7U) % 16U);
        x[point] = centerX + 2.0 * unit(id * 3U + 0U) - 1.0;
        y[point] = centerY + 2.0 * unit(id * 3U + 1U) - 1.0;
        z[point] = centerZ + 4.0 * unit(id * 3U + 2U) - 2.0;
    }
}

void fillMixedDensity(pdg::PointBatch& batch)
{
    constexpr std::size_t ClusterCount = 64U;
    double* x = batch.data<double>(X);
    double* y = batch.data<double>(Y);
    double* z = batch.data<double>(Z);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::uint64_t id = static_cast<std::uint64_t>(point);
        if ((point & 3U) != 0U)
        {
            x[point] = 500000.0 + 1000.0 * unit(id * 3U + 0U);
            y[point] = 4800000.0 + 1000.0 * unit(id * 3U + 1U);
            z[point] = 100.0 + 250.0 * unit(id * 3U + 2U);
            continue;
        }
        const std::size_t cluster = (point / 4U) % ClusterCount;
        const double centerX =
            500040.0 + 130.0 * static_cast<double>(cluster % 8U);
        const double centerY =
            4800040.0 + 130.0 * static_cast<double>(cluster / 8U);
        const double centerZ =
            110.0 + 30.0 * static_cast<double>((cluster * 5U) % 8U);
        x[point] = centerX + 1.5 * unit(id * 3U + 0U) - 0.75;
        y[point] = centerY + 1.5 * unit(id * 3U + 1U) - 0.75;
        z[point] = centerZ + 3.0 * unit(id * 3U + 2U) - 1.5;
    }
}

void fillProfile(pdg::PointBatch& batch, const std::string& profile)
{
    if (profile == "uniform-als")
        fillUniformAls(batch);
    else if (profile == "clustered-tls")
        fillClusteredTls(batch);
    else
        fillMixedDensity(batch);
}

double milliseconds(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> values, double quantile)
{
    std::sort(values.begin(), values.end());
    if (values.size() == 1U)
        return values.front();
    const double position = quantile * static_cast<double>(values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

nlohmann::json timingSummary(const std::vector<Sample>& samples,
                             double Sample::* member)
{
    std::vector<double> values;
    values.reserve(samples.size());
    for (const Sample& sample : samples)
        values.push_back(sample.*member);
    return {{"median_ms", percentile(values, 0.5)},
            {"p05_ms", percentile(values, 0.05)},
            {"p95_ms", percentile(values, 0.95)}};
}

std::uint64_t fnv1a(std::span<const std::byte> bytes,
                    std::uint64_t state = 14695981039346656037ULL) noexcept
{
    for (const std::byte value : bytes)
    {
        state ^=
            static_cast<std::uint64_t>(std::to_integer<unsigned char>(value));
        state *= 1099511628211ULL;
    }
    return state;
}

std::string hexadecimal(std::uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

std::size_t firstDifference(std::span<const std::byte> left,
                            std::span<const std::byte> right) noexcept
{
    const std::size_t size = (std::min)(left.size(), right.size());
    for (std::size_t byte = 0U; byte < size; ++byte)
        if (left[byte] != right[byte])
            return byte;
    return left.size() == right.size()
               ? (std::numeric_limits<std::size_t>::max)()
               : size;
}

std::string backendName(pdg::SpatialIndexBackend backend)
{
    return backend == pdg::SpatialIndexBackend::MortonBvh ? "morton-bvh"
                                                          : "uniform-grid";
}

struct BackendRun
{
    BackendRun(pdg::PointBatch& batch, pdg::UniformGridConfig indexConfig,
               Query selectedQuery, std::uint32_t selectedNeighbors)
        : config(std::move(indexConfig)), query(selectedQuery),
          neighbors(selectedNeighbors), index(batch, config)
    {
        const std::size_t elementBytes = query == Query::MeanDistance
                                             ? sizeof(double)
                                             : sizeof(pdg::Covariance3d);
        outputBytes = batch.size() * elementBytes;
        output = batch.memoryResource().allocate(outputBytes,
                                                 alignof(std::max_align_t));
        status = batch.memoryResource().allocate(batch.size(),
                                                 alignof(std::uint8_t));
    }

    void executeQuery()
    {
        if (query == Query::MeanDistance)
            pdg::knnMeanDistances(index, neighbors,
                                  static_cast<double*>(output->data()),
                                  static_cast<std::uint8_t*>(status->data()));
        else
            pdg::knnCovariances(index, neighbors,
                                static_cast<pdg::Covariance3d*>(output->data()),
                                static_cast<std::uint8_t*>(status->data()));
    }

    Sample run(cudaStream_t stream)
    {
        index.invalidate();
        const Clock::time_point buildStart = Clock::now();
        index.build();
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        const Clock::time_point buildEnd = Clock::now();

        CudaEvent start;
        CudaEvent stop;
        const Clock::time_point queryStart = Clock::now();
        PDG_CUDA_CHECK(cudaEventRecord(start.value, stream));
        executeQuery();
        PDG_CUDA_CHECK(cudaEventRecord(stop.value, stream));
        PDG_CUDA_CHECK(cudaEventSynchronize(stop.value));
        const Clock::time_point queryEnd = Clock::now();
        float gpuMilliseconds = 0.0F;
        PDG_CUDA_CHECK(
            cudaEventElapsedTime(&gpuMilliseconds, start.value, stop.value));
        return {milliseconds(buildStart, buildEnd),
                static_cast<double>(gpuMilliseconds),
                milliseconds(queryStart, queryEnd)};
    }

    void copyResult(cudaStream_t stream)
    {
        hostOutput.resize(outputBytes);
        hostStatus.resize(index.batch().size());
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostOutput.data(), output->data(),
                                       outputBytes, cudaMemcpyDeviceToHost,
                                       stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus.data(), status->data(),
                                       hostStatus.size(),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    pdg::UniformGridConfig config;
    Query query;
    std::uint32_t neighbors;
    pdg::SpatialIndex index;
    std::size_t outputBytes = 0U;
    std::unique_ptr<pdg::Allocation> output;
    std::unique_ptr<pdg::Allocation> status;
    std::vector<std::byte> hostOutput;
    std::vector<std::byte> hostStatus;
    std::vector<Sample> samples;
};

nlohmann::json backendReport(const BackendRun& run)
{
    std::uint64_t fingerprint = fnv1a(run.hostOutput);
    fingerprint = fnv1a(run.hostStatus, fingerprint);
    return {{"backend", backendName(run.config.backend)},
            {"persistent_index_bytes", run.index.allocatedBytes()},
            {"cell_count", run.index.cellCount()},
            {"result_fnv1a64", hexadecimal(fingerprint)},
            {"build_wall",
             timingSummary(run.samples, &Sample::buildWallMilliseconds)},
            {"query_gpu",
             timingSummary(run.samples, &Sample::queryGpuMilliseconds)},
            {"query_wall",
             timingSummary(run.samples, &Sample::queryWallMilliseconds)}};
}

int execute(const Options& options)
{
    const std::vector<pdg::CudaDeviceSummary> devices = pdg::cudaDevices();
    if (devices.empty())
        throw std::runtime_error("no CUDA device is available");

    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> pinned =
        pdg::makeCudaPinnedMemoryResource();
    pdg::CoordinateEncoding encoding({0.001, 0.001, 0.001},
                                     {500000.0, 4800000.0, 100.0});
    pdg::PointBatch host(options.points, encoding, dimensions, *pinned);
    for (pdg::DimensionId dimension : {X, Y, Z})
        host.materialize(dimension, pdg::DimensionType::Double);
    host.setSize(options.points);
    fillProfile(host, options.profile);

    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(4ULL * 1024ULL * 1024ULL * 1024ULL);
    pdg::PointBatch device(options.points, encoding, dimensions, *deviceMemory);
    for (pdg::DimensionId dimension : {X, Y, Z})
        device.materialize(dimension, pdg::DimensionType::Double);
    device.setSize(options.points);
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            options.points * sizeof(double), cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    const pdg::UniformGridConfig adaptive =
        pdg::makeAdaptiveKnnConfig(host, 3U, options.neighbors);
    BackendRun grid(device, pdg::makeKnnGridConfig(host, 3U, options.neighbors),
                    options.query, options.neighbors);
    BackendRun bvh(device, pdg::makeMortonBvhConfig(host, 3U), options.query,
                   options.neighbors);
    BackendRun* runs[2] = {&grid, &bvh};

    for (std::size_t warmup = 0U; warmup < options.warmups; ++warmup)
    {
        runs[warmup & 1U]->run(stream);
        runs[(warmup & 1U) ^ 1U]->run(stream);
    }
    for (std::size_t iteration = 0U; iteration < options.iterations;
         ++iteration)
    {
        BackendRun* first = runs[iteration & 1U];
        BackendRun* second = runs[(iteration & 1U) ^ 1U];
        first->samples.push_back(first->run(stream));
        second->samples.push_back(second->run(stream));
    }
    grid.copyResult(stream);
    bvh.copyResult(stream);

    const std::size_t outputDifference =
        firstDifference(grid.hostOutput, bvh.hostOutput);
    const std::size_t statusDifference =
        firstDifference(grid.hostStatus, bvh.hostStatus);
    const bool exact =
        outputDifference == (std::numeric_limits<std::size_t>::max)() &&
        statusDifference == (std::numeric_limits<std::size_t>::max)();

    int runtimeVersion = 0;
    int driverVersion = 0;
    PDG_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    PDG_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    std::uint64_t coordinateFingerprint = 14695981039346656037ULL;
    for (pdg::DimensionId dimension : {X, Y, Z})
    {
        const auto bytes = std::span<const std::byte>(
            static_cast<const std::byte*>(host.rawData(dimension)),
            options.points * sizeof(double));
        coordinateFingerprint = fnv1a(bytes, coordinateFingerprint);
    }

    nlohmann::json report{
        {"schema", "pdg-spatial-benchmark-v1"},
        {"profile", options.profile},
        {"query",
         options.query == Query::MeanDistance ? "mean-distance" : "covariance"},
        {"points", options.points},
        {"neighbors", options.neighbors},
        {"warmups", options.warmups},
        {"iterations", options.iterations},
        {"coordinate_fnv1a64", hexadecimal(coordinateFingerprint)},
        {"adaptive_selection", backendName(adaptive.backend)},
        {"exact_backend_bytes", exact},
        {"device",
         {{"ordinal", devices.front().ordinal},
          {"name", devices.front().name},
          {"compute_capability",
           std::to_string(devices.front().computeMajor) + "." +
               std::to_string(devices.front().computeMinor)},
          {"memory_bytes", devices.front().totalMemory},
          {"cuda_runtime_version", runtimeVersion},
          {"cuda_driver_version", driverVersion}}},
        {"backends",
         nlohmann::json::array({backendReport(grid), backendReport(bvh)})}};
    if (!exact)
    {
        report["first_output_difference"] =
            outputDifference == (std::numeric_limits<std::size_t>::max)()
                ? nlohmann::json(nullptr)
                : nlohmann::json(outputDifference);
        report["first_status_difference"] =
            statusDifference == (std::numeric_limits<std::size_t>::max)()
                ? nlohmann::json(nullptr)
                : nlohmann::json(statusDifference);
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
    return exact ? 0 : 2;
}
} // unnamed namespace

int main(int argc, char** argv)
{
    try
    {
        return execute(parseOptions(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << "pdg spatial benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
