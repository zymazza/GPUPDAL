#include <gtest/gtest.h>

#include <lazperf/filestream.hpp>
#include <lazperf/lazperf.hpp>
#include <lazperf/writers.hpp>

#include <pdal/util/OStream.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

constexpr std::size_t Format7PointBytes = 36U;
constexpr std::size_t ChunkPoints = 50'000U;

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
        const std::uint8_t returns = static_cast<std::uint8_t>(
            1U | ((1U + (index % 3U)) << 4U));
        store(records, offset + 14U, returns);
        store(records, offset + 15U,
            static_cast<std::uint8_t>((index % 4U) << 4U));
        store(records, offset + 16U,
            static_cast<std::uint8_t>(index % 32U));
        store(records, offset + 17U,
            static_cast<std::uint8_t>(index % 251U));
        store<std::int16_t>(records, offset + 18U,
            static_cast<std::int16_t>(
                static_cast<int>(index % 501U) - 250));
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

std::string sequentialPayload(const std::vector<char>& records)
{
    std::ostringstream stream(std::ios::binary | std::ios::out);
    lazperf::OutFileStream output(stream);
    constexpr std::array<char, sizeof(std::uint64_t)> Placeholder{};
    stream.write(Placeholder.data(),
                 static_cast<std::streamsize>(Placeholder.size()));
    std::streampos chunkStart = stream.tellp();
    std::vector<std::uint32_t> chunkBytes;
    lazperf::las_compressor::ptr compressor;
    std::size_t pointsInChunk = 0;
    const auto finishChunk = [&]()
    {
        compressor->done();
        compressor.reset();
        const std::streamoff bytes = stream.tellp() - chunkStart;
        if (bytes < 0 || static_cast<std::uint64_t>(bytes) >
                (std::numeric_limits<std::uint32_t>::max)())
            throw std::runtime_error("compressed LAZ chunk is too large");
        chunkBytes.push_back(static_cast<std::uint32_t>(bytes));
        chunkStart = stream.tellp();
        pointsInChunk = 0;
    };
    for (std::size_t offset = 0; offset < records.size();
         offset += Format7PointBytes)
    {
        if (!compressor)
            compressor = lazperf::build_las_compressor(output.cb(), 7, 0);
        else if (pointsInChunk == ChunkPoints)
        {
            finishChunk();
            compressor = lazperf::build_las_compressor(output.cb(), 7, 0);
        }
        compressor->compress(records.data() + offset);
        ++pointsInChunk;
    }
    if (compressor)
        finishChunk();

    const std::streampos tablePosition = stream.tellp();
    stream.seekp(0);
    pdal::OLeStream littleEndian(&stream);
    littleEndian << static_cast<std::uint64_t>(
        static_cast<std::streamoff>(tablePosition));
    stream.seekp(tablePosition);
    littleEndian << std::uint32_t{0};
    littleEndian << static_cast<std::uint32_t>(chunkBytes.size());
    lazperf::compress_chunk_table(output.cb(), chunkBytes);
    return stream.str();
}

std::string independentlyCompressedPayload(const std::vector<char>& records)
{
    std::vector<std::vector<unsigned char>> chunks;
    for (std::size_t first = 0; first < records.size();
         first += ChunkPoints * Format7PointBytes)
    {
        const std::size_t available = records.size() - first;
        const std::size_t bytes =
            (std::min)(available, ChunkPoints * Format7PointBytes);
        lazperf::writer::chunk_compressor compressor(7, 0);
        for (std::size_t offset = 0; offset < bytes;
             offset += Format7PointBytes)
            compressor.compress(records.data() + first + offset);
        chunks.push_back(compressor.done());
    }

    std::ostringstream stream(std::ios::binary | std::ios::out);
    constexpr std::array<char, sizeof(std::uint64_t)> Placeholder{};
    stream.write(Placeholder.data(),
                 static_cast<std::streamsize>(Placeholder.size()));
    std::vector<std::uint32_t> chunkBytes;
    chunkBytes.reserve(chunks.size());
    for (const std::vector<unsigned char>& chunk : chunks)
    {
        if (chunk.size() > static_cast<std::size_t>(
                (std::numeric_limits<std::uint32_t>::max)()))
            throw std::runtime_error("compressed LAZ chunk is too large");
        stream.write(reinterpret_cast<const char*>(chunk.data()),
                     static_cast<std::streamsize>(chunk.size()));
        chunkBytes.push_back(static_cast<std::uint32_t>(chunk.size()));
    }

    const std::streampos tablePosition = stream.tellp();
    stream.seekp(0);
    pdal::OLeStream littleEndian(&stream);
    littleEndian << static_cast<std::uint64_t>(
        static_cast<std::streamoff>(tablePosition));
    stream.seekp(tablePosition);
    littleEndian << std::uint32_t{0};
    littleEndian << static_cast<std::uint32_t>(chunkBytes.size());
    lazperf::OutFileStream output(stream);
    lazperf::compress_chunk_table(output.cb(), chunkBytes);
    return stream.str();
}

TEST(LazChunkCompression, IndependentFormatSevenChunksMatchSequentialPayload)
{
    for (const std::size_t count :
         {0U, 1U, 49'999U, 50'000U, 50'001U, 100'001U})
    {
        SCOPED_TRACE(count);
        const std::vector<char> records = format7Records(count);
        const std::string sequential = sequentialPayload(records);
        const std::string independent =
            independentlyCompressedPayload(records);
        EXPECT_EQ(independent, sequential);
        EXPECT_EQ(independentlyCompressedPayload(records), independent);
    }
}

} // unnamed namespace
