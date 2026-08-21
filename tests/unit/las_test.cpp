#include <pdg/io/Las.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
std::vector<std::byte> readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("unable to open fixture: " + path.string());
    const std::vector<char> characters((std::istreambuf_iterator<char>(input)),
                                       std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(characters.size());
    for (std::size_t index = 0; index < characters.size(); ++index)
        bytes[index] = static_cast<std::byte>(characters[index]);
    return bytes;
}

std::vector<std::byte> readFixture(const std::string& relative)
{
    return readFile(std::filesystem::path(PDG_TEST_DATA_DIR) / relative);
}

template <typename T>
void writeValue(std::vector<std::byte>& bytes, std::size_t offset, T value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw std::out_of_range("synthetic LAS write exceeds buffer");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <typename T>
T readValue(const std::vector<std::byte>& bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw std::out_of_range("LAS test read exceeds buffer");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::vector<std::byte> syntheticLas(std::uint8_t pointFormat)
{
    const bool extended = pointFormat >= 6;
    const std::uint16_t headerSize = extended ? 375 : 227;
    const std::uint16_t recordLength =
        pdg::las::minimumPointRecordLength(pointFormat);
    std::vector<std::byte> bytes(headerSize + recordLength);
    bytes[0] = std::byte{'L'};
    bytes[1] = std::byte{'A'};
    bytes[2] = std::byte{'S'};
    bytes[3] = std::byte{'F'};
    if (extended)
        writeValue<std::uint16_t>(bytes, 6, 1U << 4U);
    writeValue<std::uint8_t>(bytes, 24, 1);
    writeValue<std::uint8_t>(bytes, 25, extended ? 4 : 2);
    writeValue(bytes, 94, headerSize);
    writeValue(bytes, 96, static_cast<std::uint32_t>(headerSize));
    writeValue(bytes, 104, pointFormat);
    writeValue(bytes, 105, recordLength);
    if (extended)
        writeValue<std::uint64_t>(bytes, 247, 1);
    else
        writeValue<std::uint32_t>(bytes, 107, 1);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        writeValue(bytes, 131 + axis * sizeof(double), 0.01);
        writeValue(bytes, 155 + axis * sizeof(double), 1000.0);
    }

    for (std::size_t index = 0; index < recordLength; ++index)
        bytes[headerSize + index] =
            static_cast<std::byte>((index * 37U + pointFormat) & 0xffU);
    writeValue(bytes, headerSize, std::numeric_limits<std::int32_t>::min());
    writeValue(bytes, headerSize + 4, std::numeric_limits<std::int32_t>::max());
    writeValue<std::int32_t>(bytes, headerSize + 8, -1);
    return bytes;
}

std::vector<std::byte> syntheticLasPoints(std::uint8_t pointFormat,
                                          std::size_t pointCount)
{
    const auto onePoint = syntheticLas(pointFormat);
    const std::size_t headerSize = pointFormat >= 6 ? 375 : 227;
    const std::size_t recordLength =
        pdg::las::minimumPointRecordLength(pointFormat);
    std::vector<std::byte> bytes(headerSize + pointCount * recordLength);
    std::copy_n(onePoint.begin(), headerSize, bytes.begin());
    if (pointFormat >= 6)
        writeValue<std::uint64_t>(bytes, 247, pointCount);
    else
        writeValue<std::uint32_t>(bytes, 107,
                                  static_cast<std::uint32_t>(pointCount));
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        const std::size_t offset = headerSize + index * recordLength;
        std::copy_n(onePoint.begin() + headerSize, recordLength,
                    bytes.begin() + offset);
        const std::int32_t coordinate =
            static_cast<std::int32_t>(index % 100001U) - 50000;
        writeValue(bytes, offset, coordinate);
        writeValue(bytes, offset + 4, -coordinate);
        writeValue(bytes, offset + 8,
                   static_cast<std::int32_t>((index * 17U) % 65537U) - 32768);
    }
    return bytes;
}

void writeFixedString(std::vector<std::byte>& bytes, std::size_t offset,
                      std::size_t length, std::string_view value)
{
    if (offset > bytes.size() || bytes.size() - offset < length)
        throw std::out_of_range("synthetic LAS string exceeds buffer");
    const std::size_t copied = std::min(length, value.size());
    std::memcpy(bytes.data() + offset, value.data(), copied);
}

std::vector<std::byte> syntheticLasWithOffsetTime(std::size_t pointCount)
{
    constexpr std::size_t HeaderBytes = 375U;
    constexpr std::size_t VlrHeaderBytes = 54U;
    constexpr std::size_t DescriptorBytes = 192U;
    constexpr std::size_t InputPointBytes = 40U;
    constexpr std::size_t InputPointOffset =
        HeaderBytes + VlrHeaderBytes + DescriptorBytes;
    const auto canonical = syntheticLasPoints(7U, pointCount);
    std::vector<std::byte> bytes(InputPointOffset +
                                 pointCount * InputPointBytes);
    std::copy_n(canonical.begin(), HeaderBytes, bytes.begin());
    writeFixedString(bytes, 26U, 32U, "PDAL");
    writeFixedString(bytes, 58U, 32U, pdg::las::oracleSoftwareId());
    writeValue<std::uint16_t>(bytes, 90U, 1U);
    writeValue<std::uint16_t>(bytes, 92U, 2024U);
    writeValue<std::uint32_t>(bytes, 96U, InputPointOffset);
    writeValue<std::uint32_t>(bytes, 100U, 1U);
    writeValue<std::uint16_t>(bytes, 105U, InputPointBytes);

    constexpr std::size_t VlrOffset = HeaderBytes;
    writeFixedString(bytes, VlrOffset + 2U, 16U, "LASF_Spec");
    writeValue<std::uint16_t>(bytes, VlrOffset + 18U, 4U);
    writeValue<std::uint16_t>(bytes, VlrOffset + 20U, DescriptorBytes);
    writeFixedString(bytes, VlrOffset + 22U, 32U, "Extra Bytes Record");

    constexpr std::size_t DescriptorOffset = VlrOffset + VlrHeaderBytes;
    writeValue<std::uint8_t>(bytes, DescriptorOffset + 2U, 5U);
    writeFixedString(bytes, DescriptorOffset + 4U, 32U, "OffsetTime");
    writeFixedString(bytes, DescriptorOffset + 160U, 32U,
                     "Milliseconds from first acquired");

    constexpr std::size_t CanonicalPointOffset = HeaderBytes;
    constexpr std::size_t CanonicalPointBytes = 36U;
    for (std::size_t point = 0; point < pointCount; ++point)
    {
        std::copy_n(canonical.begin() + CanonicalPointOffset +
                        point * CanonicalPointBytes,
                    CanonicalPointBytes,
                    bytes.begin() + InputPointOffset + point * InputPointBytes);
        writeValue<std::uint32_t>(bytes,
                                  InputPointOffset + point * InputPointBytes +
                                      CanonicalPointBytes,
                                  static_cast<std::uint32_t>(17U * point));
    }
    return bytes;
}

void expectCoordinateRoundTrip(const std::vector<std::byte>& bytes)
{
    const pdg::las::FileView file(bytes);
    ASSERT_FALSE(file.header().compressed);
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(static_cast<std::size_t>(file.header().pointCount),
                          file.header().coordinateEncoding(), dimensions,
                          memory);
    pdg::las::decodeCoordinates(
        file, 0, static_cast<std::size_t>(file.header().pointCount), batch);

    const std::size_t recordBytes =
        batch.size() * file.header().pointRecordLength;
    const auto original = bytes.begin() + file.header().pointDataOffset;
    std::vector<std::byte> records(original, original + recordBytes);
    const std::vector<std::byte> expected = records;
    pdg::las::packCoordinates(batch, records, file.header().pointRecordLength);
    EXPECT_EQ(records, expected);
}
} // unnamed namespace

TEST(LasFileView, ParsesLas12AndRawCoordinates)
{
    const auto bytes = readFixture("las/simple.las");
    const pdg::las::FileView file(bytes);
    const auto& header = file.header();

    EXPECT_EQ(header.versionMajor, 1);
    EXPECT_EQ(header.versionMinor, 2);
    EXPECT_EQ(header.headerSize, 227);
    EXPECT_EQ(header.pointDataOffset, 227U);
    EXPECT_EQ(header.pointFormat, 3);
    EXPECT_EQ(header.pointRecordLength, 34);
    EXPECT_EQ(header.pointCount, 1065U);
    EXPECT_FALSE(header.compressed);
    EXPECT_EQ(header.generatingSoftware, "TerraScan");
    EXPECT_TRUE(file.vlrs().empty());

    EXPECT_EQ(file.rawCoordinate(0, 0), 63701224);
    EXPECT_EQ(file.rawCoordinate(0, 1), 84902831);
    EXPECT_EQ(file.rawCoordinate(0, 2), 43166);
    const auto encoding = header.coordinateEncoding();
    EXPECT_DOUBLE_EQ(encoding.decode(0, file.rawCoordinate(0, 0)), 637012.24);
    EXPECT_DOUBLE_EQ(encoding.decode(1, file.rawCoordinate(0, 1)), 849028.31);
    EXPECT_DOUBLE_EQ(encoding.decode(2, file.rawCoordinate(0, 2)), 431.66);
}

TEST(LasFileView, ParsesLas14VlrAndExtendedCount)
{
    const auto bytes = readFixture("las/test1_4.las");
    const pdg::las::FileView file(bytes);
    const auto& header = file.header();

    EXPECT_EQ(header.versionMinor, 4);
    EXPECT_EQ(header.headerSize, 375);
    EXPECT_EQ(header.pointDataOffset, 2305U);
    EXPECT_EQ(header.pointFormat, 6);
    EXPECT_EQ(header.pointRecordLength, 30);
    EXPECT_EQ(header.pointCount, 1000U);
    ASSERT_EQ(file.vlrs().size(), 2U);
    EXPECT_EQ(file.vlrs()[0].userId, "LASF_Projection");
    EXPECT_EQ(file.vlrs()[0].recordId, 2112);
    EXPECT_EQ(file.vlrs()[1].userId, "liblas");
}

TEST(LasFileView, DecodesLegacyAndModernReturnNumberWidths)
{
    for (const std::uint8_t pointFormat : {std::uint8_t{3}, std::uint8_t{7}})
    {
        std::vector<std::byte> bytes = syntheticLasPoints(pointFormat, 2U);
        const std::size_t headerSize = pointFormat >= 6U ? 375U : 227U;
        const std::size_t recordLength =
            pdg::las::minimumPointRecordLength(pointFormat);
        bytes[headerSize + 14U] = std::byte{0x09U};
        bytes[headerSize + recordLength + 14U] = std::byte{0x0fU};

        const pdg::las::FileView file(bytes);
        EXPECT_EQ(file.returnNumber(0U), pointFormat <= 5U ? 1U : 9U);
        EXPECT_EQ(file.returnNumber(1U), pointFormat <= 5U ? 7U : 15U);
        EXPECT_THROW(static_cast<void>(file.returnNumber(2U)),
                     std::out_of_range);
    }
}

TEST(LasFileView, DescribesHeaderVlrAndPointDifferences)
{
    const auto bytes = readFixture("las/test1_4.las");
    const pdg::las::FileView file(bytes);
    EXPECT_EQ(file.describeOffset(104), "LAS header: point format");
    EXPECT_EQ(file.describeOffset(file.vlrs()[0].payloadOffset),
              "VLR[0] payload LASF_Projection/2112");
    EXPECT_EQ(file.describeOffset(file.header().pointDataOffset), "point[0].X");
    EXPECT_EQ(file.describeOffset(file.header().pointDataOffset + 16),
              "point[0].Classification");
}

TEST(LasFileView, RejectsTruncationAndInvalidSignature)
{
    std::vector<std::byte> shortFile(10);
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(shortFile)),
                 pdg::las::Error);

    auto bytes = readFixture("las/simple.las");
    bytes[0] = std::byte{'N'};
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(bytes)), pdg::las::Error);

    bytes = readFixture("las/simple.las");
    bytes.resize(bytes.size() - 1);
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(bytes)), pdg::las::Error);
}

TEST(LasFileView, RejectsImpossibleRecordCountsBeforeAllocating)
{
    auto legacy = syntheticLas(3);
    writeValue<std::uint32_t>(legacy, 100,
                              std::numeric_limits<std::uint32_t>::max());
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(legacy)),
                 pdg::las::Error);

    auto modern = syntheticLas(7);
    writeValue<std::uint64_t>(modern, 235, modern.size());
    writeValue<std::uint32_t>(modern, 243,
                              std::numeric_limits<std::uint32_t>::max());
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(modern)),
                 pdg::las::Error);
}

TEST(LasPointLayout, CoversFormatsZeroThroughTen)
{
    constexpr std::uint16_t expected[] = {20, 28, 26, 34, 57, 63,
                                          30, 36, 38, 59, 67};
    for (std::uint8_t format = 0; format <= 10; ++format)
        EXPECT_EQ(pdg::las::minimumPointRecordLength(format), expected[format]);
    EXPECT_EQ(pdg::las::pointFieldAt(3, 28), "RGB");
    EXPECT_EQ(pdg::las::pointFieldAt(8, 36), "Infrared");
    EXPECT_EQ(pdg::las::pointFieldAt(10, 38), "Waveform");
}

TEST(LasPointIo, CoordinateSoARoundTripPreservesEveryRecordByte)
{
    const auto bytes = readFixture("las/simple.las");
    const pdg::las::FileView file(bytes);
    pdg::DimensionRegistry dimensions;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(static_cast<std::size_t>(file.header().pointCount),
                          file.header().coordinateEncoding(), dimensions,
                          memory);

    pdg::las::decodeCoordinates(
        file, 0, static_cast<std::size_t>(file.header().pointCount), batch);
    ASSERT_EQ(batch.size(), 1065U);
    EXPECT_EQ(batch.data<std::int32_t>(
                  pdg::DimensionId(pdg::StandardDimension::X))[0],
              63701224);

    const auto original = bytes.begin() + file.header().pointDataOffset;
    std::vector<std::byte> records(original, bytes.end());
    const std::vector<std::byte> expected = records;
    pdg::las::packCoordinates(batch, records, file.header().pointRecordLength);
    EXPECT_EQ(records, expected);
}

TEST(LasPointIo, SyntheticFormatsZeroThroughTenRoundTripExactly)
{
    for (std::uint8_t pointFormat = 0; pointFormat <= 10; ++pointFormat)
    {
        SCOPED_TRACE(static_cast<int>(pointFormat));
        const auto bytes = syntheticLas(pointFormat);
        const pdg::las::FileView file(bytes);
        EXPECT_EQ(file.header().pointFormat, pointFormat);
        EXPECT_EQ(file.rawCoordinate(0, 0),
                  std::numeric_limits<std::int32_t>::min());
        EXPECT_EQ(file.rawCoordinate(0, 1),
                  std::numeric_limits<std::int32_t>::max());
        EXPECT_EQ(file.rawCoordinate(0, 2), -1);
        expectCoordinateRoundTrip(bytes);
    }
}

TEST(LasDefaultTranslation, SupportsOnlyTheProvenCompatibilityEnvelope)
{
    constexpr std::uint8_t supported[] = {0, 1, 2, 3, 6, 7, 8};
    for (const std::uint8_t pointFormat : supported)
    {
        const auto bytes = syntheticLas(pointFormat);
        EXPECT_TRUE(
            pdg::las::supportsDefaultTranslation(pdg::las::FileView(bytes)))
            << static_cast<int>(pointFormat);
    }
    constexpr std::uint8_t unsupported[] = {4, 5, 9, 10};
    for (const std::uint8_t pointFormat : unsupported)
    {
        const auto bytes = syntheticLas(pointFormat);
        EXPECT_FALSE(
            pdg::las::supportsDefaultTranslation(pdg::las::FileView(bytes)))
            << static_cast<int>(pointFormat);
    }

    const auto withVlrs = readFixture("las/test1_4.las");
    EXPECT_FALSE(
        pdg::las::supportsDefaultTranslation(pdg::las::FileView(withVlrs)));

    auto missingWktEncoding = syntheticLas(6);
    writeValue<std::uint16_t>(missingWktEncoding, 6, 0);
    EXPECT_FALSE(pdg::las::supportsDefaultTranslation(
        pdg::las::FileView(missingWktEncoding)));

    auto invalidFormatVersion = syntheticLas(2);
    writeValue<std::uint8_t>(invalidFormatVersion, 25, 1);
    EXPECT_FALSE(pdg::las::supportsDefaultTranslation(
        pdg::las::FileView(invalidFormatVersion)));
}

TEST(LasDefaultTranslation, ConvertsAllSupportedFormatsToPdalDefaults)
{
    constexpr std::uint8_t formats[] = {0, 1, 2, 3, 6, 7, 8};
    for (const std::uint8_t pointFormat : formats)
    {
        SCOPED_TRACE(static_cast<int>(pointFormat));
        auto bytes = syntheticLas(pointFormat);
        const std::size_t recordOffset = pointFormat >= 6 ? 375 : 227;
        writeValue<std::int32_t>(bytes, recordOffset, 12345);
        writeValue<std::int32_t>(bytes, recordOffset + 4, -6789);
        writeValue<std::int32_t>(bytes, recordOffset + 8, 42);
        writeValue<std::uint16_t>(bytes, recordOffset + 12, 0xabcd);
        if (pointFormat <= 3)
        {
            writeValue<std::uint8_t>(bytes, recordOffset + 14,
                                     3U | (5U << 3U) | (1U << 6U) | (1U << 7U));
            writeValue<std::uint8_t>(bytes, recordOffset + 15,
                                     12U | (1U << 5U) | (1U << 7U));
            writeValue<std::int8_t>(bytes, recordOffset + 16, -12);
            writeValue<std::uint8_t>(bytes, recordOffset + 17, 77);
            writeValue<std::uint16_t>(bytes, recordOffset + 18, 0x1234);
            if (pointFormat == 1 || pointFormat == 3)
                writeValue<std::uint64_t>(bytes, recordOffset + 20,
                                          0x7ff8000000001234ULL);
            const std::size_t colorOffset = pointFormat == 2   ? 20
                                            : pointFormat == 3 ? 28
                                                               : 0;
            if (colorOffset)
            {
                writeValue<std::uint16_t>(bytes, recordOffset + colorOffset,
                                          101);
                writeValue<std::uint16_t>(bytes, recordOffset + colorOffset + 2,
                                          202);
                writeValue<std::uint16_t>(bytes, recordOffset + colorOffset + 4,
                                          303);
            }
        }
        else
        {
            writeValue<std::uint8_t>(bytes, recordOffset + 14,
                                     9U | (10U << 4U));
            writeValue<std::uint8_t>(bytes, recordOffset + 15,
                                     1U | (1U << 2U) | (1U << 3U) | (2U << 4U) |
                                         (1U << 6U));
            writeValue<std::uint8_t>(bytes, recordOffset + 16, 42);
            writeValue<std::uint8_t>(bytes, recordOffset + 17, 77);
            writeValue<std::int16_t>(bytes, recordOffset + 18, -321);
            writeValue<std::uint16_t>(bytes, recordOffset + 20, 0x1234);
            writeValue<std::uint64_t>(bytes, recordOffset + 22,
                                      0x7ff8000000001234ULL);
            if (pointFormat == 7 || pointFormat == 8)
            {
                writeValue<std::uint16_t>(bytes, recordOffset + 30, 101);
                writeValue<std::uint16_t>(bytes, recordOffset + 32, 202);
                writeValue<std::uint16_t>(bytes, recordOffset + 34, 303);
            }
        }

        const std::string software = pdg::las::oracleSoftwareId();
        const auto output = pdg::las::translateDefault(
            pdg::las::FileView(bytes), {60, 2024, software});
        ASSERT_EQ(output.size(), 375U + 36U);
        const pdg::las::FileView translated(output);
        EXPECT_EQ(translated.header().versionMinor, 4);
        EXPECT_EQ(translated.header().pointFormat, 7);
        EXPECT_EQ(translated.header().pointRecordLength, 36);
        EXPECT_EQ(translated.header().pointCount, 1U);
        EXPECT_EQ(translated.header().globalEncoding, 1U << 4U);
        EXPECT_EQ(translated.header().creationDayOfYear, 60);
        EXPECT_EQ(translated.header().creationYear, 2024);
        EXPECT_EQ(translated.header().systemIdentifier, "PDAL");
        EXPECT_EQ(translated.header().generatingSoftware, software);
        EXPECT_EQ(translated.rawCoordinate(0, 0), 112345);
        EXPECT_EQ(translated.rawCoordinate(0, 1), 93211);
        EXPECT_EQ(translated.rawCoordinate(0, 2), 100042);

        constexpr std::size_t outputRecord = 375;
        EXPECT_EQ(readValue<std::uint16_t>(output, outputRecord + 12), 0xabcd);
        EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 17), 77);
        EXPECT_EQ(readValue<std::uint16_t>(output, outputRecord + 20), 0x1234);
        if (pointFormat <= 3)
        {
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 14),
                      3U | (5U << 4U));
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 16), 0);
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 15) &
                          0x0fU,
                      0x0dU);
        }
        else
        {
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 14),
                      9U | (10U << 4U));
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 16), 42);
            EXPECT_EQ(readValue<std::uint8_t>(output, outputRecord + 15),
                      0x6dU);
        }
        const bool hasGps =
            pointFormat == 1 || pointFormat == 3 || pointFormat >= 6;
        EXPECT_EQ(readValue<std::uint64_t>(output, outputRecord + 22),
                  hasGps ? 0x7ff8000000001234ULL : 0ULL);
        const bool hasColor = pointFormat == 2 || pointFormat == 3 ||
                              pointFormat == 7 || pointFormat == 8;
        EXPECT_EQ(readValue<std::uint16_t>(output, outputRecord + 30),
                  hasColor ? 101 : 0);
        EXPECT_EQ(readValue<std::uint16_t>(output, outputRecord + 32),
                  hasColor ? 202 : 0);
        EXPECT_EQ(readValue<std::uint16_t>(output, outputRecord + 34),
                  hasColor ? 303 : 0);
    }
}

TEST(LasDefaultTranslation, EmptyCloudHasCanonicalZeroBounds)
{
    const auto bytes = syntheticLasPoints(7, 0);
    auto output = pdg::las::translateDefault(
        pdg::las::FileView(bytes), {1, 2024, pdg::las::oracleSoftwareId()});
    ASSERT_EQ(output.size(), 375U);
    const pdg::las::FileView translated(output);
    EXPECT_EQ(translated.header().pointCount, 0U);
    EXPECT_EQ(translated.header().minimum,
              (std::array<double, 3>{0.0, 0.0, 0.0}));
    EXPECT_EQ(translated.header().maximum,
              (std::array<double, 3>{0.0, 0.0, 0.0}));
    EXPECT_EQ(translated.header().pointsByReturn,
              (std::array<std::uint64_t, 15>{}));
    EXPECT_NO_THROW(pdg::las::overlayDefaultUserData(output, {}));
    EXPECT_NO_THROW(pdg::las::overlayDefaultClassification(output, {}));
}

TEST(LasDefaultTranslation, CallerOwnedOutputMatchesVectorApiExactly)
{
    const auto bytes = syntheticLasPoints(7, 65537);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto expected = pdg::las::translateDefault(input, metadata, 3);
    ASSERT_EQ(pdg::las::defaultTranslationSize(input), expected.size());

    std::vector<std::byte> actual(expected.size(), std::byte{0xff});
    pdg::las::translateDefaultInto(input, metadata, actual, 3);
    EXPECT_EQ(actual, expected);

    actual.pop_back();
    EXPECT_THROW(pdg::las::translateDefaultInto(input, metadata, actual, 1),
                 pdg::las::Error);
}

TEST(LasDefaultTranslation, OverlaysCanonicalUserDataWithoutChangingOtherBytes)
{
    constexpr std::size_t PointCount = 3U;
    const auto bytes = syntheticLasPoints(7, PointCount);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    auto output = pdg::las::translateDefault(input, metadata, 1);
    auto expected = output;
    const std::array<std::uint8_t, PointCount> values{0U, 127U, 255U};
    for (std::size_t point = 0; point < PointCount; ++point)
        writeValue(expected, 375U + point * 36U + 17U, values[point]);

    pdg::las::overlayDefaultUserData(output, values);
    EXPECT_EQ(output, expected);

    EXPECT_THROW(pdg::las::overlayDefaultUserData(
                     output, std::span<const std::uint8_t>(values).first(2U)),
                 pdg::las::Error);
    output[104] = std::byte{6};
    EXPECT_THROW(pdg::las::overlayDefaultUserData(output, values),
                 pdg::las::Error);
}

TEST(LasDefaultTranslation,
     OverlaysCanonicalClassificationWithoutChangingOtherBytes)
{
    constexpr std::size_t PointCount = 3U;
    const auto bytes = syntheticLasPoints(3, PointCount);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    auto output = pdg::las::translateDefault(input, metadata, 1);
    auto expected = output;
    const std::array<std::uint8_t, PointCount> values{0U, 127U, 255U};
    std::array<std::byte, PointCount> flags{};
    for (std::size_t point = 0; point < PointCount; ++point)
    {
        const std::size_t record = 375U + point * 36U;
        flags[point] = output[record + 15U];
        writeValue(expected, record + 16U, values[point]);
    }

    pdg::las::overlayDefaultClassification(output, values);
    EXPECT_EQ(output, expected);
    for (std::size_t point = 0; point < PointCount; ++point)
        EXPECT_EQ(output[375U + point * 36U + 15U], flags[point]);

    EXPECT_THROW(pdg::las::overlayDefaultClassification(
                     output, std::span<const std::uint8_t>(values).first(2U)),
                 pdg::las::Error);
    output[104] = std::byte{6};
    EXPECT_THROW(pdg::las::overlayDefaultClassification(output, values),
                 pdg::las::Error);
}

TEST(LasDefaultTranslation,
     PermutedClassificationPublisherPreservesEveryOtherRecordByte)
{
    constexpr std::size_t PointCount = 3U;
    const auto bytes = syntheticLasPoints(7U, PointCount);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1U, 2024U, pdg::las::oracleSoftwareId()};
    const auto canonical = pdg::las::translateDefault(input, metadata, 1U);
    const std::array<std::uint64_t, PointCount> order{2U, 0U, 1U};
    const std::array<std::uint8_t, PointCount> classification{9U, 7U, 3U};
    auto expected = canonical;
    for (std::size_t destination = 0U; destination < PointCount; ++destination)
    {
        const std::size_t destinationOffset = 375U + destination * 36U;
        const std::size_t sourceOffset =
            375U + static_cast<std::size_t>(order[destination]) * 36U;
        std::copy_n(canonical.begin() + sourceOffset, 36U,
                    expected.begin() + destinationOffset);
        expected[destinationOffset + 16U] =
            static_cast<std::byte>(classification[destination]);
    }

    std::vector<std::byte> actual(expected.size(), std::byte{0xff});
    pdg::las::translateDefaultPermutedClassificationInto(
        input, metadata, order, classification, actual, 1U);
    EXPECT_EQ(actual, expected);

    const std::array<std::uint64_t, PointCount> duplicate{2U, 0U, 2U};
    EXPECT_THROW(pdg::las::translateDefaultPermutedClassificationInto(
                     input, metadata, duplicate, classification, actual, 1U),
                 pdg::las::Error);
    const std::array<std::uint64_t, PointCount> outside{2U, 0U, 3U};
    EXPECT_THROW(pdg::las::translateDefaultPermutedClassificationInto(
                     input, metadata, outside, classification, actual, 1U),
                 pdg::las::Error);
    EXPECT_THROW(pdg::las::translateDefaultPermutedClassificationInto(
                     input, metadata,
                     std::span<const std::uint64_t>(order).first(2U),
                     classification, actual, 1U),
                 pdg::las::Error);
    actual.pop_back();
    EXPECT_THROW(pdg::las::translateDefaultPermutedClassificationInto(
                     input, metadata, order, classification, actual, 1U),
                 pdg::las::Error);
}

TEST(LasDefaultTranslation, PermutedPublisherReordersCompleteRecordsExactly)
{
    constexpr std::size_t PointCount = 3U;
    const auto bytes = syntheticLasPoints(7U, PointCount);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1U, 2024U, pdg::las::oracleSoftwareId()};
    const auto canonical = pdg::las::translateDefault(input, metadata, 1U);
    const std::array<std::uint64_t, PointCount> order{2U, 0U, 1U};
    auto expected = canonical;
    for (std::size_t destination = 0U; destination < PointCount; ++destination)
    {
        const std::size_t destinationOffset = 375U + destination * 36U;
        const std::size_t sourceOffset =
            375U + static_cast<std::size_t>(order[destination]) * 36U;
        std::copy_n(canonical.begin() + sourceOffset, 36U,
                    expected.begin() + destinationOffset);
    }

    std::vector<std::byte> actual(expected.size(), std::byte{0xff});
    pdg::las::translateDefaultPermutedInto(input, metadata, order, actual, 1U);
    EXPECT_EQ(actual, expected);

    const std::array<std::uint64_t, PointCount> duplicate{2U, 0U, 2U};
    EXPECT_THROW(pdg::las::translateDefaultPermutedInto(input, metadata,
                                                        duplicate, actual, 1U),
                 pdg::las::Error);
    const std::array<std::uint64_t, PointCount> outside{2U, 0U, 3U};
    EXPECT_THROW(pdg::las::translateDefaultPermutedInto(input, metadata,
                                                        outside, actual, 1U),
                 pdg::las::Error);
}

TEST(LasDefaultTranslation, WorkerCountCannotChangeOutputBytes)
{
    constexpr std::size_t PointCount = 131073;
    const auto bytes = syntheticLasPoints(7, PointCount);
    const pdg::las::FileView input(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto serial = pdg::las::translateDefault(input, metadata, 1);
    const auto parallel = pdg::las::translateDefault(input, metadata, 17);
    const auto automatic = pdg::las::translateDefault(input, metadata);
    EXPECT_EQ(parallel, serial);
    EXPECT_EQ(automatic, serial);
}

TEST(LasExtraDoubleTranslation,
     AppendsOneDescriptorAndValueWithoutChangingSourceRecordBytes)
{
    constexpr std::size_t PointCount = 5U;
    constexpr std::size_t InputPointOffset = 621U;
    constexpr std::size_t OutputPointOffset = 813U;
    constexpr std::size_t InputPointBytes = 40U;
    constexpr std::size_t OutputPointBytes = 48U;
    const auto bytes = syntheticLasWithOffsetTime(PointCount);
    const pdg::las::FileView input(bytes);
    const std::array<std::uint64_t, PointCount> values{
        0x0000000000000000ULL, 0x8000000000000000ULL, 0x7ff0000000000000ULL,
        0x7ff8000000001234ULL, 0x7ff8000000005678ULL};
    const pdg::las::DefaultTranslationMetadata metadata{
        1U, 2024U, pdg::las::oracleSoftwareId()};

    ASSERT_TRUE(pdg::las::supportsExtraDoubleTranslation(input));
    auto output = pdg::las::translateExtraDouble(
        input, metadata,
        {.name = "HeightAboveGround", .description = "Height Above Ground"},
        values, 2U);
    ASSERT_EQ(output.size(), OutputPointOffset + PointCount * OutputPointBytes);
    const pdg::las::FileView translated(output);
    EXPECT_EQ(translated.header().pointDataOffset, OutputPointOffset);
    EXPECT_EQ(translated.header().pointRecordLength, OutputPointBytes);
    ASSERT_EQ(translated.vlrs().size(), 1U);
    EXPECT_EQ(translated.vlrs()[0].userId, "LASF_Spec");
    EXPECT_EQ(translated.vlrs()[0].recordId, 4U);
    EXPECT_EQ(translated.vlrs()[0].payloadLength, 384U);
    EXPECT_TRUE(translated.evlrs().empty());

    std::vector<std::byte> expectedHeader(bytes.begin(), bytes.begin() + 375U);
    writeValue<std::uint32_t>(expectedHeader, 96U, OutputPointOffset);
    writeValue<std::uint16_t>(expectedHeader, 105U, OutputPointBytes);
    writeValue<std::uint32_t>(expectedHeader, 107U, 0U);
    for (std::size_t index = 0; index < 5U; ++index)
        writeValue<std::uint32_t>(expectedHeader,
                                  111U + index * sizeof(std::uint32_t), 0U);
    for (std::size_t index = 0; index < 15U; ++index)
        writeValue<std::uint64_t>(expectedHeader,
                                  255U + index * sizeof(std::uint64_t),
                                  index == 12U ? PointCount : 0U);
    const auto decoded = [](std::int32_t raw)
    { return static_cast<double>(raw) * 0.01 + 1000.0; };
    writeValue<double>(expectedHeader, 179U, decoded(-49996));
    writeValue<double>(expectedHeader, 187U, decoded(-50000));
    writeValue<double>(expectedHeader, 195U, decoded(50000));
    writeValue<double>(expectedHeader, 203U, decoded(49996));
    writeValue<double>(expectedHeader, 211U, decoded(-32700));
    writeValue<double>(expectedHeader, 219U, decoded(-32768));
    const auto headerDifference = std::mismatch(
        expectedHeader.begin(), expectedHeader.end(), output.begin());
    ASSERT_EQ(headerDifference.first, expectedHeader.end())
        << "first rewritten header difference at byte "
        << std::distance(expectedHeader.begin(), headerDifference.first);
    EXPECT_TRUE(std::equal(bytes.begin() + 375U, bytes.begin() + 395U,
                           output.begin() + 375U));
    EXPECT_TRUE(std::equal(bytes.begin() + 397U, bytes.begin() + 621U,
                           output.begin() + 397U));
    EXPECT_EQ(readValue<std::uint16_t>(output, 395U), 384U);
    EXPECT_EQ(readValue<std::uint8_t>(output, 623U), 10U);
    EXPECT_EQ(
        std::string(reinterpret_cast<const char*>(output.data() + 625U), 17U),
        "HeightAboveGround");
    EXPECT_EQ(
        std::string(reinterpret_cast<const char*>(output.data() + 781U), 19U),
        "Height Above Ground");
    for (std::size_t point = 0; point < PointCount; ++point)
    {
        EXPECT_TRUE(std::equal(
            bytes.begin() + InputPointOffset + point * InputPointBytes,
            bytes.begin() + InputPointOffset + (point + 1U) * InputPointBytes,
            output.begin() + OutputPointOffset + point * OutputPointBytes));
        EXPECT_EQ(readValue<std::uint64_t>(
                      output, OutputPointOffset + point * OutputPointBytes +
                                  InputPointBytes),
                  values[point]);
    }

    std::vector<std::byte> into(output.size());
    pdg::las::translateExtraDoubleInto(
        input, metadata,
        {.name = "HeightAboveGround", .description = "Height Above Ground"},
        values, into, 1U);
    EXPECT_EQ(into, output);
    into.pop_back();
    EXPECT_THROW(
        pdg::las::translateExtraDoubleInto(
            input, metadata,
            {.name = "HeightAboveGround", .description = "Height Above Ground"},
            values, into, 1U),
        pdg::las::Error);
}

TEST(LasExtraDoubleTranslation, EmptyAndSinglePointHeadersAreSummarized)
{
    const pdg::las::DefaultTranslationMetadata metadata{
        1U, 2024U, pdg::las::oracleSoftwareId()};
    for (const std::size_t pointCount : {0U, 1U})
    {
        const auto bytes = syntheticLasWithOffsetTime(pointCount);
        const pdg::las::FileView input(bytes);
        const std::vector<std::uint64_t> values(pointCount,
                                                0x7ff0000000000000ULL);
        const auto output = pdg::las::translateExtraDouble(
            input, metadata,
            {.name = "HeightAboveGround", .description = "Height Above Ground"},
            values);
        const pdg::las::FileView translated(output);
        EXPECT_EQ(translated.header().pointCount, pointCount);
        EXPECT_DOUBLE_EQ(readValue<double>(output, 179U),
                         pointCount ? 500.0 : 0.0);
        EXPECT_DOUBLE_EQ(readValue<double>(output, 187U),
                         pointCount ? 500.0 : 0.0);
        EXPECT_EQ(readValue<std::uint64_t>(output, 255U + 12U * 8U),
                  pointCount);
    }
}

TEST(LasExtraDoubleTranslation, RejectsEveryUnprovenEnvelopeChange)
{
    const auto supported = syntheticLasWithOffsetTime(1U);
    const auto rejects = [&](std::vector<std::byte> bytes)
    {
        EXPECT_FALSE(pdg::las::supportsExtraDoubleTranslation(
            pdg::las::FileView(bytes)));
    };

    auto changed = supported;
    changed[104U] = std::byte{6U};
    rejects(std::move(changed));
    changed = supported;
    writeValue<std::uint16_t>(changed, 105U, 41U);
    changed.push_back(std::byte{});
    rejects(std::move(changed));
    changed = supported;
    writeValue<std::uint16_t>(changed, 395U, 191U);
    rejects(std::move(changed));
    changed = supported;
    changed[376U] = std::byte{1U};
    rejects(std::move(changed));
    changed = supported;
    changed[388U] = std::byte{1U};
    rejects(std::move(changed));
    changed = supported;
    changed[431U] = std::byte{1U};
    rejects(std::move(changed));
    changed = supported;
    changed.push_back(std::byte{});
    rejects(std::move(changed));
    changed = supported;
    writeValue<std::uint64_t>(changed, 247U,
                              (std::numeric_limits<std::uint64_t>::max)());
    EXPECT_THROW(static_cast<void>(pdg::las::FileView(changed)),
                 pdg::las::Error);

    const pdg::las::FileView input(supported);
    const std::array<std::uint64_t, 1U> value{0x3ff0000000000000ULL};
    const pdg::las::DefaultTranslationMetadata metadata{
        1U, 2024U, pdg::las::oracleSoftwareId()};
    EXPECT_THROW(
        static_cast<void>(pdg::las::translateExtraDouble(
            input, metadata, {.name = "", .description = "Height Above Ground"},
            value)),
        pdg::las::Error);
    EXPECT_THROW(
        static_cast<void>(pdg::las::translateExtraDouble(
            input, metadata,
            {.name = "HeightAboveGround", .description = "Height Above Ground"},
            std::span<const std::uint64_t>{})),
        pdg::las::Error);
}

TEST(LocalLasCorpus, OptionalUncompressedFileRoundTripsExactly)
{
    const char* configuredPath = std::getenv("PDG_LOCAL_LAS_FILE");
    if (!configuredPath || !*configuredPath)
        GTEST_SKIP() << "set PDG_LOCAL_LAS_FILE for a read-only corpus check";

    const std::filesystem::path path(configuredPath);
    constexpr std::uintmax_t MaximumFixtureBytes = 128U * 1024U * 1024U;
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    ASSERT_LE(std::filesystem::file_size(path), MaximumFixtureBytes)
        << "select a <=128 MiB stratified fixture for the smoke lane";
    const auto bytes = readFile(path);
    const pdg::las::FileView file(bytes);
    if (file.header().compressed)
        GTEST_SKIP() << "native LAZ decoding is a P4 deliverable";
    expectCoordinateRoundTrip(bytes);
}

TEST(LocalLasCorpus, OptionalOneExtraDoubleInputMatchesStrictEnvelope)
{
    const char* configuredPath = std::getenv("PDG_LOCAL_EXTRA_DOUBLE_LAS_FILE");
    if (!configuredPath || !*configuredPath)
        GTEST_SKIP() << "set PDG_LOCAL_EXTRA_DOUBLE_LAS_FILE for the strict "
                        "Extra Bytes envelope check";
    const auto bytes = readFile(configuredPath);
    const pdg::las::FileView file(bytes);
    EXPECT_TRUE(pdg::las::supportsExtraDoubleTranslation(file));
}
