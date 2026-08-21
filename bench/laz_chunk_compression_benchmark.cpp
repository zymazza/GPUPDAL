#include <pdal/compression/LazPerfVlrCompression.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr std::size_t Format7PointBytes = 36U;
constexpr std::uint32_t ChunkPoints = 50'000U;

struct Options
{
    std::size_t points = 1'000'000U;
    std::size_t warmups = 1U;
    std::size_t iterations = 9U;
    std::filesystem::path output;
};

struct Run
{
    double milliseconds = 0.0;
    std::string payload;
};

std::size_t parseSize(std::string_view value, std::string_view option)
{
    std::size_t position = 0U;
    try
    {
        const unsigned long long parsed =
            std::stoull(std::string(value), &position, 10);
        if (position != value.size())
            throw std::out_of_range("invalid");
        return static_cast<std::size_t>(parsed);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid value for " +
                                    std::string(option));
    }
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option = argv[index];
        if (option == "--help")
        {
            std::cout << "Usage: pdg_laz_chunk_compression_benchmark [options]\n"
                      << "  --points N --warmups N --iterations N\n"
                      << "  --output FILE  (refuses to overwrite)\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " +
                                        std::string(option));
        const std::string_view value = argv[++index];
        if (option == "--points")
            options.points = parseSize(value, option);
        else if (option == "--warmups")
            options.warmups = parseSize(value, option);
        else if (option == "--iterations")
            options.iterations = parseSize(value, option);
        else if (option == "--output")
            options.output = value;
        else
            throw std::invalid_argument("unknown option " +
                                        std::string(option));
    }
    if (!options.points || !options.warmups || !options.iterations)
        throw std::invalid_argument(
            "points, warmups, and iterations must be positive");
    if (options.points >
            (std::numeric_limits<std::size_t>::max)() / Format7PointBytes)
        throw std::invalid_argument("point count is too large");
    return options;
}

template <typename T>
void store(std::vector<char>& records, std::size_t offset, T value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(records.data() + offset, &value, sizeof(value));
}

std::vector<char> format7Records(std::size_t count)
{
    std::vector<char> records(count * Format7PointBytes);
    for (std::size_t index = 0; index < count; ++index)
    {
        const std::size_t offset = index * Format7PointBytes;
        store<std::int32_t>(records, offset,
            static_cast<std::int32_t>(index * 3U));
        store<std::int32_t>(records, offset + 4U,
            static_cast<std::int32_t>(index * 5U + 1U));
        store<std::int32_t>(records, offset + 8U,
            static_cast<std::int32_t>(index * 7U + 2U));
        store<std::uint16_t>(records, offset + 12U,
            static_cast<std::uint16_t>((index * 11U) & 0xffffU));
        store(records, offset + 14U,
            static_cast<std::uint8_t>(1U | ((1U + (index % 3U)) << 4U)));
        store(records, offset + 15U,
            static_cast<std::uint8_t>((index % 4U) << 4U));
        store(records, offset + 16U,
            static_cast<std::uint8_t>(index % 32U));
        store(records, offset + 17U,
            static_cast<std::uint8_t>(index % 251U));
        store<std::int16_t>(records, offset + 18U,
            static_cast<std::int16_t>(static_cast<int>(index % 501U) - 250));
        store<std::uint16_t>(records, offset + 20U,
            static_cast<std::uint16_t>(index % 65'535U));
        store<double>(records, offset + 22U,
            static_cast<double>(index) * 0.125);
        store<std::uint16_t>(records, offset + 30U,
            static_cast<std::uint16_t>((index * 13U) & 0xffffU));
        store<std::uint16_t>(records, offset + 32U,
            static_cast<std::uint16_t>((index * 17U) & 0xffffU));
        store<std::uint16_t>(records, offset + 34U,
            static_cast<std::uint16_t>((index * 19U) & 0xffffU));
    }
    return records;
}

Run compress(const std::vector<char>& records, std::size_t workers)
{
    const Clock::time_point start = Clock::now();
    std::ostringstream stream(std::ios::binary | std::ios::out);
    // The production compressor starts after a nonzero LAS point offset. Its
    // implementation uses stream position zero as the no-points sentinel, so
    // retain one untimed-equivalent prefix byte in this isolated primitive.
    constexpr std::array<char, 9> PrefixAndChunkOffset{};
    stream.write(PrefixAndChunkOffset.data(),
                 static_cast<std::streamsize>(PrefixAndChunkOffset.size()));
    stream.seekp(1);
    {
        pdal::LazPerfVlrCompressor compressor(
            stream, 7, 0, ChunkPoints, workers);
        for (std::size_t offset = 0; offset < records.size();
             offset += Format7PointBytes)
            compressor.compress(records.data() + offset);
        compressor.done();
    }
    Run result;
    result.payload = stream.str();
    result.milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}

nlohmann::json summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const double median = values[values.size() / 2U];
    return {{"samples_ms", values},
            {"minimum_ms", values.front()},
            {"median_ms", median},
            {"maximum_ms", values.back()}};
}

int execute(const Options& options)
{
    const std::vector<char> records = format7Records(options.points);
    bool exact = true;
    Run sequential;
    Run parallel;
    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
    {
        sequential = compress(records, 1U);
        parallel = compress(records, 2U);
        exact = exact && sequential.payload == parallel.payload;
    }

    std::vector<double> sequentialSamples;
    std::vector<double> parallelSamples;
    sequentialSamples.reserve(options.iterations);
    parallelSamples.reserve(options.iterations);
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration)
    {
        if ((iteration & 1U) == 0U)
        {
            sequential = compress(records, 1U);
            parallel = compress(records, 2U);
        }
        else
        {
            parallel = compress(records, 2U);
            sequential = compress(records, 1U);
        }
        exact = exact && sequential.payload == parallel.payload;
        sequentialSamples.push_back(sequential.milliseconds);
        parallelSamples.push_back(parallel.milliseconds);
    }

    const nlohmann::json sequentialSummary = summarize(sequentialSamples);
    const nlohmann::json parallelSummary = summarize(parallelSamples);
    const double speedup =
        sequentialSummary.at("median_ms").get<double>() /
        parallelSummary.at("median_ms").get<double>();
    const nlohmann::json report{
        {"schema", "pdg-laz-chunk-compression-benchmark-v1"},
        {"fixture", {{"kind", "deterministic_format7_records"},
                     {"points", options.points},
                     {"record_bytes", Format7PointBytes},
                     {"chunk_points", ChunkPoints},
                     {"stream_prefix_bytes", 1}}},
        {"comparison", {{"exact_payload_bytes", exact},
                        {"alternating_order", true},
                        {"warmups", options.warmups},
                        {"iterations", options.iterations},
                        {"sequential_workers", 1},
                        {"parallel_workers", 2},
                        {"median_speedup", speedup}}},
        {"payload_bytes", sequential.payload.size()},
        {"timing", {{"sequential", sequentialSummary},
                    {"parallel", parallelSummary}}}};
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
        std::cerr << "pdg LAZ chunk compression benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
