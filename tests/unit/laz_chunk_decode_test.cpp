#include <gtest/gtest.h>

#include <lazperf/lazperf.hpp>
#include <lazperf/writers.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

constexpr std::size_t ExtraBytes = 4U;

template <typename T>
void store(std::vector<char>& records, std::size_t offset, T value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(records.data() + offset, &value, sizeof(value));
}

std::size_t basePointBytes(int format)
{
    switch (format)
    {
    case 6:
        return 30U;
    case 7:
        return 36U;
    case 8:
        return 38U;
    default:
        throw std::invalid_argument("unsupported test point format");
    }
}

std::vector<char> records(int format, std::size_t count)
{
    const std::size_t pointBytes = basePointBytes(format) + ExtraBytes;
    std::vector<char> result(count * pointBytes);
    for (std::size_t index = 0; index < count; ++index)
    {
        const std::size_t offset = index * pointBytes;
        store<std::int32_t>(result, offset,
            static_cast<std::int32_t>(index * 3U));
        store<std::int32_t>(result, offset + 4U,
            static_cast<std::int32_t>(index * 5U + 1U));
        store<std::int32_t>(result, offset + 8U,
            static_cast<std::int32_t>(index * 7U + 2U));
        store<std::uint16_t>(result, offset + 12U,
            static_cast<std::uint16_t>((index * 11U) & 0xffffU));
        store(result, offset + 14U,
            static_cast<std::uint8_t>(1U | ((1U + (index % 3U)) << 4U)));
        store(result, offset + 15U,
            static_cast<std::uint8_t>((index % 4U) << 4U));
        store(result, offset + 16U,
            static_cast<std::uint8_t>(index % 32U));
        store(result, offset + 17U,
            static_cast<std::uint8_t>(index % 251U));
        store<std::int16_t>(result, offset + 18U,
            static_cast<std::int16_t>(static_cast<int>(index % 501U) - 250));
        store<std::uint16_t>(result, offset + 20U,
            static_cast<std::uint16_t>(index % 65'535U));
        store<double>(result, offset + 22U,
            static_cast<double>(index) * 0.125);
        if (format >= 7)
        {
            store<std::uint16_t>(result, offset + 30U,
                static_cast<std::uint16_t>((index * 13U) & 0xffffU));
            store<std::uint16_t>(result, offset + 32U,
                static_cast<std::uint16_t>((index * 17U) & 0xffffU));
            store<std::uint16_t>(result, offset + 34U,
                static_cast<std::uint16_t>((index * 19U) & 0xffffU));
        }
        if (format == 8)
            store<std::uint16_t>(result, offset + 36U,
                static_cast<std::uint16_t>((index * 23U) & 0xffffU));
        const std::size_t extraOffset = offset + basePointBytes(format);
        store<std::uint32_t>(result, extraOffset,
            static_cast<std::uint32_t>(index * 29U + 3U));
    }
    return result;
}

std::vector<unsigned char> compress(const std::vector<char>& input, int format)
{
    const std::size_t pointBytes = basePointBytes(format) + ExtraBytes;
    lazperf::writer::chunk_compressor compressor(
        format, static_cast<int>(ExtraBytes));
    for (std::size_t offset = 0; offset < input.size(); offset += pointBytes)
        compressor.compress(input.data() + offset);
    return compressor.done();
}

std::vector<char> decode(const std::vector<unsigned char>& compressed,
    int format, std::size_t count)
{
    std::size_t offset = 0U;
    const lazperf::InputCb input =
        [&compressed, &offset](unsigned char* destination, std::size_t size)
        {
            if (size > compressed.size() -
                    (std::min)(offset, compressed.size()))
                throw std::runtime_error("truncated compressed chunk");
            std::memcpy(destination, compressed.data() + offset, size);
            offset += size;
        };
    lazperf::las_decompressor::ptr decompressor =
        lazperf::build_las_decompressor(input, format, ExtraBytes);
    if (!decompressor)
        throw std::runtime_error("unable to construct LAZ decompressor");

    const std::size_t pointBytes = basePointBytes(format) + ExtraBytes;
    std::vector<char> output(count * pointBytes);
    for (std::size_t index = 0; index < count; ++index)
        decompressor->decompress(output.data() + index * pointBytes);
    return output;
}

TEST(LazChunkDecode, PointFormatsSixThroughEightAreByteExactAndDeterministic)
{
    for (const std::size_t pointCount : {1U, 2U, 50'000U})
    {
        for (const int format : {6, 7, 8})
        {
            SCOPED_TRACE(pointCount);
            SCOPED_TRACE(format);
            const std::vector<char> expected = records(format, pointCount);
            const std::vector<unsigned char> compressed =
                compress(expected, format);
            const std::vector<char> first =
                decode(compressed, format, pointCount);
            EXPECT_EQ(first, expected);
            EXPECT_EQ(decode(compressed, format, pointCount), first);
        }
    }
}

TEST(LazChunkDecode, BoundedInputRejectsTruncatedChunk)
{
    constexpr std::size_t PointCount = 257U;
    const std::vector<char> expected = records(7, PointCount);
    const std::vector<unsigned char> compressed = compress(expected, 7);
    ASSERT_GT(compressed.size(), 1U);
    const std::vector<unsigned char> truncated{compressed.front()};
    EXPECT_THROW((void)decode(truncated, 7, PointCount), std::runtime_error);
}

} // unnamed namespace
