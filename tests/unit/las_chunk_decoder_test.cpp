// B0150/D0212: the LAZ decode path must reproduce the uncompressed records
// exactly, because everything downstream — placement facts, the mapped source,
// and every exactness matrix — assumes the engine's own reader sees the same
// bytes PDAL does.

#include "../../src/io/las/LasChunkDecoder.hpp"

#include <pdg/io/Las.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{

std::vector<std::byte> readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};
    const std::vector<char> characters((std::istreambuf_iterator<char>(input)),
                                       std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(characters.size());
    for (std::size_t index = 0U; index < characters.size(); ++index)
        bytes[index] = static_cast<std::byte>(characters[index]);
    return bytes;
}

// The paired fixtures are large bench data rather than committed test data, so
// the case skips when they are absent instead of failing a clean checkout.
std::filesystem::path benchData(const std::string& name)
{
    const char* root = std::getenv("PDG_BENCH_DATA_DIR");
    return std::filesystem::path(root ? root : "build/bench-data") / name;
}

} // unnamed namespace

TEST(LasChunkDecoder, DecodesLazRecordsExactlyLikeTheUncompressedFile)
{
    const std::vector<std::byte> compressed =
        readFile(benchData("reference/ref-1m.laz"));
    const std::vector<std::byte> plain =
        readFile(benchData("download-3101-prefix-1m.las"));
    if (compressed.empty() || plain.empty())
        GTEST_SKIP() << "paired LAZ/LAS bench fixtures are unavailable";

    const pdg::las::FileView compressedView{compressed};
    const pdg::las::FileView plainView{plain};
    ASSERT_TRUE(compressedView.header().compressed);
    ASSERT_FALSE(plainView.header().compressed);
    ASSERT_EQ(compressedView.header().pointCount,
              plainView.header().pointCount);
    ASSERT_EQ(compressedView.header().pointRecordLength,
              plainView.header().pointRecordLength);

    const std::vector<std::byte> decoded =
        pdg::las::decodeCompressedPointRecords(compressedView);
    const std::size_t recordBytes = plainView.header().pointRecordLength;
    ASSERT_EQ(decoded.size(),
              static_cast<std::size_t>(plainView.header().pointCount) *
                  recordBytes);

    for (std::uint64_t point = 0U; point < plainView.header().pointCount;
         ++point)
    {
        const std::span<const std::byte> expected =
            plainView.pointRecord(point);
        const std::span<const std::byte> actual(
            decoded.data() + static_cast<std::size_t>(point) * recordBytes,
            recordBytes);
        ASSERT_EQ(expected.size(), actual.size()) << point;
        if (std::memcmp(expected.data(), actual.data(), recordBytes) != 0)
            FAIL() << "decoded record " << point << " differs";
    }
}

TEST(LasChunkDecoder, RejectsAnUncompressedFile)
{
    const std::vector<std::byte> plain =
        readFile(benchData("download-3101-prefix-25k.las"));
    if (plain.empty())
        GTEST_SKIP() << "bench fixture is unavailable";
    const pdg::las::FileView view{plain};
    EXPECT_THROW(pdg::las::decodeCompressedPointRecords(view),
                 pdg::las::Error);
}

TEST(LasChunkDecoder, ChunkTableValidationRejectsAPartialCompressedPayload)
{
    const std::filesystem::path source = benchData("reference/ref-1m.laz");
    const std::vector<std::byte> compressed = readFile(source);
    if (compressed.empty())
        GTEST_SKIP() << "LAZ bench fixture is unavailable";

    const pdg::las::FileView complete{compressed};
    ASSERT_TRUE(complete.header().compressed);
    EXPECT_NO_THROW(pdg::las::validateCompressedPointRecords(source, complete));

    const std::filesystem::path truncated =
        std::filesystem::temp_directory_path() /
        ("pdg-laz-validation-truncated-" + std::to_string(::getpid()) +
         ".laz");
    std::error_code error;
    std::filesystem::remove(truncated, error);
    error.clear();
    ASSERT_TRUE(std::filesystem::copy_file(source, truncated, error)) << error;
    const std::uintmax_t offset = complete.header().pointDataOffset;
    ASSERT_LT(offset + 1U, compressed.size());
    std::filesystem::resize_file(
        truncated, offset + (compressed.size() - offset) / 2U, error);
    ASSERT_FALSE(error) << error;

    const std::vector<std::byte> partial = readFile(truncated);
    ASSERT_FALSE(partial.empty());
    // Public LAS/VLR facts remain parseable; only the compressed payload and
    // chunk table are incomplete, which is the failure the structural
    // validator exists to detect.
    const pdg::las::FileView partialView{partial};
    EXPECT_ANY_THROW(
        pdg::las::validateCompressedPointRecords(truncated, partialView));
    std::filesystem::remove(truncated, error);
}
