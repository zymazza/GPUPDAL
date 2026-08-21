#include "LasChunkDecoder.hpp"

#include "pdg/io/Las.hpp"

#include <lazperf/readers.hpp>
#include <lazperf/lazperf.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace pdg::las
{
namespace
{

// lazperf's reader takes a mutable buffer even though it only reads from it, so
// the mapped view is copied once. That copy is exactly the cost this first
// correctness-first slice accepts; the chunk-parallel follow-up decodes from
// the mapped bytes directly.
std::vector<char> mutableCopy(std::span<const std::byte> bytes)
{
    std::vector<char> copy(bytes.size());
    if (!bytes.empty())
        std::memcpy(copy.data(), bytes.data(), bytes.size());
    return copy;
}

std::uint64_t readLittleEndian64(std::span<const std::byte> bytes,
                                 std::size_t offset,
                                 const char* description)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(std::uint64_t))
        throw Error(std::string(description) + " extends beyond the file");
    std::uint64_t value = 0U;
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte)
        value |= std::uint64_t(std::to_integer<unsigned char>(bytes[offset + byte]))
                 << (byte * 8U);
    return value;
}

std::uint32_t readLittleEndian32(std::span<const std::byte> bytes,
                                 std::size_t offset,
                                 const char* description)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(std::uint32_t))
        throw Error(std::string(description) + " extends beyond the file");
    std::uint32_t value = 0U;
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + byte]))
                 << (byte * 8U);
    return value;
}

} // unnamed namespace

std::vector<std::byte> decodeCompressedPointRecords(const FileView& view)
{
    const Header& header = view.header();
    if (!header.compressed)
        throw Error("LAZ decode requires a compressed LAS file");
    if (!header.pointRecordLength)
        throw Error("LAZ decode requires a nonzero point record length");

    std::vector<char> source = mutableCopy(view.bytes());
    lazperf::reader::mem_file file(source.data(), source.size());

    const std::uint64_t declared = header.pointCount;
    if (file.pointCount() != declared)
    {
        std::ostringstream message;
        message << "LAZ point count " << file.pointCount()
                << " does not match the LAS header's " << declared;
        throw Error(message.str());
    }

    const std::size_t recordBytes = header.pointRecordLength;
    if (declared > (std::numeric_limits<std::size_t>::max)() / recordBytes)
        throw Error("LAZ decode size overflows the address space");

    // lazperf writes one point at a time into a caller-owned record buffer, so
    // the destination is sized up front and filled in file order. Point order
    // is the file's own order and is never reordered here: downstream exactness
    // depends on it.
    std::vector<std::byte> decoded(static_cast<std::size_t>(declared) *
                                   recordBytes);
    char* out = reinterpret_cast<char*>(decoded.data());
    for (std::uint64_t point = 0U; point < declared; ++point)
    {
        file.readPoint(out);
        out += recordBytes;
    }

    if (static_cast<std::size_t>(out - reinterpret_cast<char*>(decoded.data()))
        != decoded.size())
        throw Error("LAZ decode produced an unexpected byte count");
    return decoded;
}

void validateCompressedPointRecords(const std::filesystem::path& path,
                                    const FileView& view)
{
    const Header& header = view.header();
    if (!header.compressed)
        throw Error("LAZ validation requires a compressed LAS file");
    if (!header.pointRecordLength)
        throw Error("LAZ validation requires a nonzero point record length");

    // Opening the file makes lazperf parse the LASzip VLR and compressed chunk
    // table. Do not decode all points here: publication validation is on the
    // command's timed path and a second O(points) decode erased most of the
    // acceleration on large LAZ outputs.
    lazperf::reader::named_file file(path.string());
    if (file.pointCount() != header.pointCount)
    {
        std::ostringstream message;
        message << "LAZ point count " << file.pointCount()
                << " does not match the LAS header's " << header.pointCount;
        throw Error(message.str());
    }

    const std::span<const std::byte> bytes = view.bytes();
    const std::size_t firstChunkOffset =
        static_cast<std::size_t>(header.pointDataOffset) + sizeof(std::uint64_t);
    const std::uint64_t chunkTableOffset64 = readLittleEndian64(
        bytes, header.pointDataOffset, "LAZ chunk-table pointer");
    if (chunkTableOffset64 == (std::numeric_limits<std::uint64_t>::max)())
        throw Error("LAZ publication has no finalized chunk table");
    if (chunkTableOffset64 > (std::numeric_limits<std::size_t>::max)())
        throw Error("LAZ chunk-table offset overflows the address space");
    const std::size_t chunkTableOffset =
        static_cast<std::size_t>(chunkTableOffset64);
    if (chunkTableOffset < firstChunkOffset)
        throw Error("LAZ chunk table overlaps the compressed point payload");
    if (chunkTableOffset > bytes.size() ||
        bytes.size() - chunkTableOffset < 2U * sizeof(std::uint32_t))
        throw Error("LAZ chunk table extends beyond the file");

    const std::uint32_t tableVersion =
        readLittleEndian32(bytes, chunkTableOffset, "LAZ chunk-table version");
    if (tableVersion != 0U)
        throw Error("LAZ chunk table has an unsupported version");
    const std::uint32_t chunkCount = readLittleEndian32(
        bytes, chunkTableOffset + sizeof(std::uint32_t),
        "LAZ chunk-table count");
    const std::size_t compressedBytes = chunkTableOffset - firstChunkOffset;
    if (chunkCount > compressedBytes)
        throw Error("LAZ chunk count cannot fit in the compressed point payload");

    const lazperf::laz_vlr laz = file.lazVlr();
    const bool variableChunks = laz.variableChunks();
    if (!variableChunks)
    {
        if (!laz.chunk_size)
            throw Error("LAZ fixed chunk size is zero");
        const std::uint64_t expectedChunks =
            header.pointCount / laz.chunk_size +
            (header.pointCount % laz.chunk_size != 0U ? 1U : 0U);
        if (expectedChunks != chunkCount)
            throw Error("LAZ chunk count does not cover the declared points");
    }
    else if (chunkCount > header.pointCount)
        throw Error("LAZ variable chunk count exceeds the declared points");

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw Error("unable to reopen LAZ publication for chunk validation");
    stream.seekg(static_cast<std::streamoff>(
        chunkTableOffset + 2U * sizeof(std::uint32_t)));
    if (!stream)
        throw Error("unable to seek to the LAZ chunk table");
    const std::vector<lazperf::chunk> chunks = lazperf::decompress_chunk_table(
        stream, chunkCount, variableChunks);

    std::uint64_t describedBytes = 0U;
    std::uint64_t describedPoints = 0U;
    for (const lazperf::chunk& chunk : chunks)
    {
        if (!chunk.offset)
            throw Error("LAZ chunk table contains an empty point payload");
        if (chunk.offset > (std::numeric_limits<std::uint64_t>::max)() -
                               describedBytes)
            throw Error("LAZ chunk byte extent overflows");
        describedBytes += chunk.offset;
        if (variableChunks)
        {
            if (!chunk.count)
                throw Error("LAZ chunk table contains an empty point chunk");
            if (chunk.count > (std::numeric_limits<std::uint64_t>::max)() -
                                  describedPoints)
                throw Error("LAZ chunk point extent overflows");
            describedPoints += chunk.count;
        }
    }
    if (describedBytes != compressedBytes)
        throw Error("LAZ chunk table does not cover the compressed point payload");
    if (variableChunks && describedPoints != header.pointCount)
        throw Error("LAZ chunk table does not cover the declared points");
}

} // namespace pdg::las
