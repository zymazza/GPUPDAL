#include <lazperf/lazperf.hpp>
#include <lazperf/writers.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr std::size_t Format7PointBytes = 36U;

struct Options
{
    std::size_t points = 50'000U;
    std::size_t chunks = 20U;
    std::size_t constructions = 20U;
    std::size_t warmups = 1U;
    std::size_t iterations = 9U;
    std::filesystem::path output;
};

std::size_t parseSize(std::string_view value, std::string_view option)
{
    std::size_t position = 0U;
    try
    {
        const unsigned long long parsed =
            std::stoull(std::string(value), &position, 10);
        if (position != value.size() ||
            parsed > (std::numeric_limits<std::size_t>::max)())
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
            std::cout << "Usage: pdg_laz_chunk_decode_benchmark [options]\n"
                      << "  --points N --chunks N --constructions N\n"
                      << "  --warmups N --iterations N --output FILE\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " +
                                        std::string(option));
        const std::string_view value = argv[++index];
        if (option == "--points")
            options.points = parseSize(value, option);
        else if (option == "--chunks")
            options.chunks = parseSize(value, option);
        else if (option == "--constructions")
            options.constructions = parseSize(value, option);
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
    if (!options.points || !options.chunks || !options.constructions ||
        !options.warmups || !options.iterations)
        throw std::invalid_argument("all numeric options must be positive");
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

std::vector<unsigned char> compressChunk(const std::vector<char>& records)
{
    lazperf::writer::chunk_compressor compressor(7, 0);
    for (std::size_t offset = 0; offset < records.size();
         offset += Format7PointBytes)
        compressor.compress(records.data() + offset);
    return compressor.done();
}

lazperf::las_decompressor::ptr makeDecompressor(
    const std::vector<unsigned char>& compressed, std::size_t& offset)
{
    const lazperf::InputCb input =
        [&compressed, &offset](unsigned char* destination, std::size_t size)
        {
            if (offset > compressed.size() ||
                size > compressed.size() - offset)
                throw std::runtime_error("truncated compressed chunk");
            std::memcpy(destination, compressed.data() + offset, size);
            offset += size;
        };
    lazperf::las_decompressor::ptr decompressor =
        lazperf::build_las_decompressor(input, 7, 0);
    if (!decompressor)
        throw std::runtime_error("unable to construct LAZ decompressor");
    return decompressor;
}

double measureConstruction(const std::vector<unsigned char>& compressed,
    std::size_t constructions)
{
    const Clock::time_point start = Clock::now();
    for (std::size_t index = 0; index < constructions; ++index)
    {
        std::size_t offset = 0U;
        const lazperf::las_decompressor::ptr decompressor =
            makeDecompressor(compressed, offset);
        if (!decompressor)
            throw std::runtime_error("unreachable null decompressor");
    }
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

struct DecodeRun
{
    double milliseconds = 0.0;
    bool exact = true;
};

DecodeRun measureDecode(const std::vector<unsigned char>& compressed,
    const std::vector<char>& expected, std::size_t chunks)
{
    std::vector<char> decoded(expected.size());
    DecodeRun result;
    const Clock::time_point start = Clock::now();
    for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    {
        std::size_t offset = 0U;
        lazperf::las_decompressor::ptr decompressor =
            makeDecompressor(compressed, offset);
        for (std::size_t point = 0; point < expected.size() / Format7PointBytes;
             ++point)
            decompressor->decompress(
                decoded.data() + point * Format7PointBytes);
        result.exact = result.exact && decoded == expected;
    }
    result.milliseconds = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return result;
}

nlohmann::json summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return {{"samples_ms", values},
            {"minimum_ms", values.front()},
            {"median_ms", values[values.size() / 2U]},
            {"maximum_ms", values.back()}};
}

int execute(const Options& options)
{
    const std::vector<char> expected = format7Records(options.points);
    const std::vector<unsigned char> compressed = compressChunk(expected);
    bool exact = true;
    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
    {
        (void)measureConstruction(compressed, options.constructions);
        exact = exact && measureDecode(
            compressed, expected, options.chunks).exact;
    }

    std::vector<double> constructionSamples;
    std::vector<double> decodeSamples;
    constructionSamples.reserve(options.iterations);
    decodeSamples.reserve(options.iterations);
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration)
    {
        constructionSamples.push_back(
            measureConstruction(compressed, options.constructions));
        const DecodeRun decode =
            measureDecode(compressed, expected, options.chunks);
        decodeSamples.push_back(decode.milliseconds);
        exact = exact && decode.exact;
    }

    const nlohmann::json report{
        {"schema", "pdg-laz-chunk-decode-benchmark-v1"},
        {"fixture", {{"kind", "deterministic_format7_chunk"},
                     {"points_per_chunk", options.points},
                     {"record_bytes", Format7PointBytes},
                     {"compressed_bytes", compressed.size()}}},
        {"comparison", {{"exact_decoded_records", exact},
                        {"warmups", options.warmups},
                        {"iterations", options.iterations},
                        {"constructions_per_sample", options.constructions},
                        {"decoded_chunks_per_sample", options.chunks}}},
        {"timing", {{"factory_construction", summarize(constructionSamples)},
                    {"construct_and_decode", summarize(decodeSamples)}}}};
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
        std::cerr << "pdg_laz_chunk_decode_benchmark: " << error.what()
                  << '\n';
        return 1;
    }
}
