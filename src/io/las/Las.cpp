#include <pdg/io/Las.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>

namespace pdg::las
{

namespace
{
static_assert(std::endian::native == std::endian::little,
              "PDG v1 supports little-endian x86_64 hosts only");

template <typename T>
T read(std::span<const std::byte> bytes, std::size_t offset,
       std::string_view field)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw Error("truncated LAS field: " + std::string(field));
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::string readString(std::span<const std::byte> bytes, std::size_t offset,
                       std::size_t length, std::string_view field)
{
    if (offset > bytes.size() || bytes.size() - offset < length)
        throw Error("truncated LAS field: " + std::string(field));
    std::string value(reinterpret_cast<const char*>(bytes.data() + offset),
                      length);
    if (const std::size_t end = value.find('\0'); end != std::string::npos)
        value.resize(end);
    while (!value.empty() && value.back() == ' ')
        value.pop_back();
    return value;
}

std::size_t checkedExtent(std::size_t offset, std::uint64_t count,
                          std::size_t stride, std::string_view field)
{
    if (count > std::numeric_limits<std::size_t>::max() / stride)
        throw Error(std::string(field) +
                    " size overflows the host address space");
    const std::size_t bytes = static_cast<std::size_t>(count) * stride;
    if (offset > std::numeric_limits<std::size_t>::max() - bytes)
        throw Error(std::string(field) + " end offset overflows");
    return offset + bytes;
}

std::uint16_t requiredHeaderSize(std::uint8_t minor)
{
    if (minor >= 4)
        return 375;
    if (minor == 3)
        return 235;
    return 227;
}

Header parseHeader(std::span<const std::byte> bytes)
{
    if (bytes.size() < 227)
        throw Error("LAS file is shorter than the 227-byte base header");
    if (readString(bytes, 0, 4, "file signature") != "LASF")
        throw Error("invalid LAS file signature; expected LASF");

    Header header;
    header.fileSourceId = read<std::uint16_t>(bytes, 4, "file source id");
    header.globalEncoding = read<std::uint16_t>(bytes, 6, "global encoding");
    header.versionMajor = read<std::uint8_t>(bytes, 24, "version major");
    header.versionMinor = read<std::uint8_t>(bytes, 25, "version minor");
    if (header.versionMajor != 1 || header.versionMinor > 4)
    {
        std::ostringstream message;
        message << "unsupported LAS version "
                << static_cast<int>(header.versionMajor) << '.'
                << static_cast<int>(header.versionMinor);
        throw Error(message.str());
    }

    header.systemIdentifier = readString(bytes, 26, 32, "system identifier");
    header.generatingSoftware =
        readString(bytes, 58, 32, "generating software");
    header.creationDayOfYear =
        read<std::uint16_t>(bytes, 90, "creation day of year");
    header.creationYear = read<std::uint16_t>(bytes, 92, "creation year");
    header.headerSize = read<std::uint16_t>(bytes, 94, "header size");
    header.pointDataOffset =
        read<std::uint32_t>(bytes, 96, "point data offset");
    header.variableLengthRecordCount =
        read<std::uint32_t>(bytes, 100, "VLR count");

    const std::uint8_t pointFormatBits =
        read<std::uint8_t>(bytes, 104, "point format");
    header.pointFormat = pointFormatBits & 0x0fU;
    header.compressed = (pointFormatBits & 0x80U) != 0;
    if (header.pointFormat > 10)
        throw Error("unsupported LAS point format " +
                    std::to_string(header.pointFormat));
    header.pointRecordLength =
        read<std::uint16_t>(bytes, 105, "point record length");
    const std::uint16_t minimumLength =
        minimumPointRecordLength(header.pointFormat);
    if (header.pointRecordLength < minimumLength)
        throw Error("LAS point record is shorter than its format requires");

    const std::uint32_t legacyPointCount =
        read<std::uint32_t>(bytes, 107, "legacy point count");
    for (std::size_t index = 0; index < 5; ++index)
        header.pointsByReturn[index] = read<std::uint32_t>(
            bytes, 111 + index * 4, "legacy points by return");

    header.scale = {read<double>(bytes, 131, "X scale"),
                    read<double>(bytes, 139, "Y scale"),
                    read<double>(bytes, 147, "Z scale")};
    header.offset = {read<double>(bytes, 155, "X offset"),
                     read<double>(bytes, 163, "Y offset"),
                     read<double>(bytes, 171, "Z offset")};
    header.maximum = {read<double>(bytes, 179, "maximum X"),
                      read<double>(bytes, 195, "maximum Y"),
                      read<double>(bytes, 211, "maximum Z")};
    header.minimum = {read<double>(bytes, 187, "minimum X"),
                      read<double>(bytes, 203, "minimum Y"),
                      read<double>(bytes, 219, "minimum Z")};

    if (header.versionMinor >= 3)
        header.waveformDataOffset =
            read<std::uint64_t>(bytes, 227, "waveform data offset");
    if (header.versionMinor >= 4)
    {
        header.extendedVariableLengthRecordOffset =
            read<std::uint64_t>(bytes, 235, "EVLR offset");
        header.extendedVariableLengthRecordCount =
            read<std::uint32_t>(bytes, 243, "EVLR count");
        header.pointCount =
            read<std::uint64_t>(bytes, 247, "extended point count");
        for (std::size_t index = 0; index < 15; ++index)
            header.pointsByReturn[index] = read<std::uint64_t>(
                bytes, 255 + index * 8, "extended points by return");
    }
    else
        header.pointCount = legacyPointCount;

    if (header.headerSize < requiredHeaderSize(header.versionMinor))
        throw Error("LAS header size is smaller than its version requires");
    if (header.headerSize > bytes.size())
        throw Error("LAS header extends beyond the file");
    if (header.pointDataOffset < header.headerSize)
        throw Error("LAS point data overlaps the header");
    if (header.pointDataOffset > bytes.size())
        throw Error("LAS point data offset extends beyond the file");

    // Constructor validation also rejects zero/NaN scales and offsets.
    static_cast<void>(header.coordinateEncoding());

    if (!header.compressed)
    {
        const std::size_t end =
            checkedExtent(header.pointDataOffset, header.pointCount,
                          header.pointRecordLength, "LAS point data");
        if (end > bytes.size())
            throw Error("LAS point count extends beyond the file");
    }
    return header;
}

std::vector<VariableLengthRecord> parseVlrs(std::span<const std::byte> bytes,
                                            const Header& header)
{
    std::vector<VariableLengthRecord> records;
    constexpr std::size_t HeaderBytes = 54;
    const std::size_t available = header.pointDataOffset - header.headerSize;
    if (header.variableLengthRecordCount > available / HeaderBytes)
        throw Error("LAS VLR count cannot fit before point data");
    records.reserve(header.variableLengthRecordCount);
    std::size_t offset = header.headerSize;
    for (std::uint32_t index = 0; index < header.variableLengthRecordCount;
         ++index)
    {
        if (offset > header.pointDataOffset ||
            header.pointDataOffset - offset < 54)
            throw Error("LAS VLR header overlaps point data");
        const std::uint16_t length =
            read<std::uint16_t>(bytes, offset + 20, "VLR payload length");
        const std::size_t payloadOffset = offset + 54;
        if (payloadOffset > header.pointDataOffset ||
            header.pointDataOffset - payloadOffset < length)
            throw Error("LAS VLR payload overlaps point data");
        records.push_back(
            {offset, payloadOffset, length,
             readString(bytes, offset + 2, 16, "VLR user id"),
             read<std::uint16_t>(bytes, offset + 18, "VLR record id"),
             readString(bytes, offset + 22, 32, "VLR description"), false});
        offset = payloadOffset + length;
    }
    return records;
}

std::vector<VariableLengthRecord> parseEvlrs(std::span<const std::byte> bytes,
                                             const Header& header)
{
    std::vector<VariableLengthRecord> records;
    if (!header.extendedVariableLengthRecordCount)
        return records;
    if (header.extendedVariableLengthRecordOffset > bytes.size())
        throw Error("LAS EVLR offset extends beyond the file");

    std::size_t offset =
        static_cast<std::size_t>(header.extendedVariableLengthRecordOffset);
    constexpr std::size_t HeaderBytes = 60;
    const std::size_t available = bytes.size() - offset;
    if (header.extendedVariableLengthRecordCount > available / HeaderBytes)
        throw Error("LAS EVLR count cannot fit in the file");
    records.reserve(header.extendedVariableLengthRecordCount);
    for (std::uint32_t index = 0;
         index < header.extendedVariableLengthRecordCount; ++index)
    {
        if (offset > bytes.size() || bytes.size() - offset < 60)
            throw Error("truncated LAS EVLR header");
        const std::uint64_t length =
            read<std::uint64_t>(bytes, offset + 20, "EVLR payload length");
        const std::size_t payloadOffset = offset + 60;
        if (length > bytes.size() - payloadOffset)
            throw Error("truncated LAS EVLR payload");
        records.push_back(
            {offset, payloadOffset, length,
             readString(bytes, offset + 2, 16, "EVLR user id"),
             read<std::uint16_t>(bytes, offset + 18, "EVLR record id"),
             readString(bytes, offset + 28, 32, "EVLR description"), true});
        offset = payloadOffset + static_cast<std::size_t>(length);
    }
    return records;
}

std::string_view headerFieldAt(std::size_t offset)
{
    struct Field
    {
        std::size_t begin;
        std::size_t end;
        std::string_view name;
    };
    static constexpr Field fields[] = {{0, 4, "file signature"},
                                       {4, 6, "file source id"},
                                       {6, 8, "global encoding"},
                                       {8, 24, "project id"},
                                       {24, 25, "version major"},
                                       {25, 26, "version minor"},
                                       {26, 58, "system identifier"},
                                       {58, 90, "generating software"},
                                       {90, 92, "creation day of year"},
                                       {92, 94, "creation year"},
                                       {94, 96, "header size"},
                                       {96, 100, "point data offset"},
                                       {100, 104, "VLR count"},
                                       {104, 105, "point format"},
                                       {105, 107, "point record length"},
                                       {107, 111, "legacy point count"},
                                       {111, 131, "legacy points by return"},
                                       {131, 139, "X scale"},
                                       {139, 147, "Y scale"},
                                       {147, 155, "Z scale"},
                                       {155, 163, "X offset"},
                                       {163, 171, "Y offset"},
                                       {171, 179, "Z offset"},
                                       {179, 187, "maximum X"},
                                       {187, 195, "minimum X"},
                                       {195, 203, "maximum Y"},
                                       {203, 211, "minimum Y"},
                                       {211, 219, "maximum Z"},
                                       {219, 227, "minimum Z"},
                                       {227, 235, "waveform data offset"},
                                       {235, 243, "EVLR offset"},
                                       {243, 247, "EVLR count"},
                                       {247, 255, "extended point count"},
                                       {255, 375, "extended points by return"}};
    for (const Field& field : fields)
        if (offset >= field.begin && offset < field.end)
            return field.name;
    return "header extension";
}
} // unnamed namespace

CoordinateEncoding Header::coordinateEncoding() const
{
    return {scale, offset};
}

FileView::FileView(std::span<const std::byte> bytes)
    : m_header(parseHeader(bytes)), m_bytes(bytes),
      m_vlrs(parseVlrs(bytes, m_header)), m_evlrs(parseEvlrs(bytes, m_header))
{
}

const Header& FileView::header() const noexcept
{
    return m_header;
}

const std::vector<VariableLengthRecord>& FileView::vlrs() const noexcept
{
    return m_vlrs;
}

const std::vector<VariableLengthRecord>& FileView::evlrs() const noexcept
{
    return m_evlrs;
}

std::span<const std::byte> FileView::bytes() const noexcept
{
    return m_bytes;
}

std::span<const std::byte> FileView::pointRecord(std::uint64_t index) const
{
    if (m_header.compressed)
        throw Error("compressed LAZ records require the chunk decoder");
    if (index >= m_header.pointCount)
        throw std::out_of_range("LAS point index is out of range");
    const std::size_t offset =
        checkedExtent(m_header.pointDataOffset, index,
                      m_header.pointRecordLength, "LAS point index");
    return m_bytes.subspan(offset, m_header.pointRecordLength);
}

std::int32_t FileView::rawCoordinate(std::uint64_t index,
                                     std::size_t axis) const
{
    if (axis >= 3)
        throw std::out_of_range("coordinate axis must be 0, 1, or 2");
    return read<std::int32_t>(pointRecord(index), axis * sizeof(std::int32_t),
                              "point coordinate");
}

std::uint8_t FileView::returnNumber(std::uint64_t index) const
{
    const std::uint8_t returns =
        static_cast<std::uint8_t>(pointRecord(index)[14U]);
    return static_cast<std::uint8_t>(
        returns & (m_header.pointFormat <= 5U ? 0x07U : 0x0fU));
}

std::string FileView::describeOffset(std::size_t offset) const
{
    if (offset >= m_bytes.size())
        return "past end of file";
    if (offset < m_header.headerSize)
        return "LAS header: " + std::string(headerFieldAt(offset));

    const auto describeRecord = [offset](const VariableLengthRecord& record,
                                         std::string_view kind,
                                         std::size_t index) -> std::string
    {
        const std::size_t payloadEnd =
            record.payloadOffset +
            static_cast<std::size_t>(record.payloadLength);
        if (offset >= record.headerOffset && offset < record.payloadOffset)
            return std::string(kind) + '[' + std::to_string(index) +
                   "] header " + record.userId + '/' +
                   std::to_string(record.recordId);
        if (offset >= record.payloadOffset && offset < payloadEnd)
            return std::string(kind) + '[' + std::to_string(index) +
                   "] payload " + record.userId + '/' +
                   std::to_string(record.recordId);
        return {};
    };
    for (std::size_t index = 0; index < m_vlrs.size(); ++index)
        if (std::string description =
                describeRecord(m_vlrs[index], "VLR", index);
            !description.empty())
            return description;
    for (std::size_t index = 0; index < m_evlrs.size(); ++index)
        if (std::string description =
                describeRecord(m_evlrs[index], "EVLR", index);
            !description.empty())
            return description;

    if (offset >= m_header.pointDataOffset)
    {
        if (m_header.compressed)
            return "compressed point data";
        const std::size_t relative = offset - m_header.pointDataOffset;
        const std::uint64_t index = relative / m_header.pointRecordLength;
        if (index < m_header.pointCount)
        {
            const std::size_t recordOffset =
                relative % m_header.pointRecordLength;
            return "point[" + std::to_string(index) + "]." +
                   std::string(
                       pointFieldAt(m_header.pointFormat, recordOffset));
        }
    }
    return "LAS padding or trailing data";
}

std::uint16_t minimumPointRecordLength(std::uint8_t pointFormat)
{
    static constexpr std::uint16_t lengths[] = {20, 28, 26, 34, 57, 63,
                                                30, 36, 38, 59, 67};
    if (pointFormat >= std::size(lengths))
        throw Error("unsupported LAS point format " +
                    std::to_string(pointFormat));
    return lengths[pointFormat];
}

std::string_view pointFieldAt(std::uint8_t pointFormat,
                              std::size_t recordOffset)
{
    if (recordOffset < 4)
        return "X";
    if (recordOffset < 8)
        return "Y";
    if (recordOffset < 12)
        return "Z";
    if (recordOffset < 14)
        return "Intensity";

    if (pointFormat <= 5)
    {
        if (recordOffset == 14)
            return "return flags";
        if (recordOffset == 15)
            return "Classification";
        if (recordOffset == 16)
            return "ScanAngleRank";
        if (recordOffset == 17)
            return "UserData";
        if (recordOffset < 20)
            return "PointSourceId";
        if (pointFormat == 0)
            return "ExtraBytes";
        if (pointFormat == 1)
            return recordOffset < 28 ? "GpsTime" : "ExtraBytes";
        if (pointFormat == 2)
            return recordOffset < 26 ? "RGB" : "ExtraBytes";
        if (recordOffset < 28)
            return "GpsTime";
        if (pointFormat == 3)
            return recordOffset < 34 ? "RGB" : "ExtraBytes";
        if (pointFormat == 4)
            return recordOffset < 57 ? "Waveform" : "ExtraBytes";
        if (recordOffset < 34)
            return "RGB";
        return recordOffset < 63 ? "Waveform" : "ExtraBytes";
    }

    if (recordOffset == 14)
        return "return flags";
    if (recordOffset == 15)
        return "classification flags";
    if (recordOffset == 16)
        return "Classification";
    if (recordOffset == 17)
        return "UserData";
    if (recordOffset < 20)
        return "ScanAngle";
    if (recordOffset < 22)
        return "PointSourceId";
    if (recordOffset < 30)
        return "GpsTime";
    if (pointFormat == 6)
        return "ExtraBytes";
    if (pointFormat == 7)
        return recordOffset < 36 ? "RGB" : "ExtraBytes";
    if (pointFormat == 8)
    {
        if (recordOffset < 36)
            return "RGB";
        return recordOffset < 38 ? "Infrared" : "ExtraBytes";
    }
    if (pointFormat == 9)
        return recordOffset < 59 ? "Waveform" : "ExtraBytes";
    if (recordOffset < 36)
        return "RGB";
    if (recordOffset < 38)
        return "Infrared";
    return recordOffset < 67 ? "Waveform" : "ExtraBytes";
}

} // namespace pdg::las
