#include <pdg/io/LasFerry.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
std::vector<std::byte> readFerryFixture(const std::string& relative)
{
    const std::filesystem::path path =
        std::filesystem::path(PDG_TEST_DATA_DIR) / relative;
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

template <typename T>
T readFerryValue(const std::vector<std::byte>& bytes, std::size_t offset)
{
    static_assert(std::is_trivially_copyable_v<T>);
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void writeFerryValue(std::vector<std::byte>& bytes, std::size_t offset, T value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

pdg::las::DefaultTranslationMetadata ferryMetadata()
{
    return {1, 2024, pdg::las::oracleSoftwareId()};
}
} // unnamed namespace

TEST(LasFerry, AppliesOrderedMappingsToCanonicalOutputExactly)
{
    const auto bytes = readFerryFixture("las/simple.las");
    const pdg::las::FileView input(bytes);
    const auto baseline = pdg::las::translateDefault(input, ferryMetadata(), 1);
    auto expected = baseline;
    auto actual = baseline;

    const auto intensity = pdg::DimensionId(pdg::StandardDimension::Intensity);
    const auto pointSource =
        pdg::DimensionId(pdg::StandardDimension::PointSourceId);
    const auto red = pdg::DimensionId(pdg::StandardDimension::Red);
    const auto returnNumber =
        pdg::DimensionId(pdg::StandardDimension::ReturnNumber);
    const auto userData = pdg::DimensionId(pdg::StandardDimension::UserData);
    const auto x = pdg::DimensionId(pdg::StandardDimension::X);
    const auto z = pdg::DimensionId(pdg::StandardDimension::Z);
    const pdg::FerryProgram program{{{true, intensity, pointSource, false},
                                     {true, pointSource, red, false},
                                     {true, returnNumber, userData, false},
                                     {true, x, z, false}}};
    pdg::DimensionRegistry dimensions;
    ASSERT_TRUE(pdg::las::supportsDefaultFerry(input, program, dimensions));

    constexpr std::size_t HeaderBytes = 375;
    constexpr std::size_t PointBytes = 36;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(input.header().pointCount); ++index)
    {
        const std::size_t record = HeaderBytes + index * PointBytes;
        const std::uint16_t intensityValue =
            readFerryValue<std::uint16_t>(baseline, record + 12);
        const std::uint8_t returnValue =
            readFerryValue<std::uint8_t>(baseline, record + 14) & 0x0fU;
        const std::int32_t xValue =
            readFerryValue<std::int32_t>(baseline, record);
        writeFerryValue(expected, record + 20, intensityValue);
        writeFerryValue(expected, record + 30, intensityValue);
        writeFerryValue(expected, record + 17, returnValue);
        writeFerryValue(expected, record + 8, xValue);
    }
    writeFerryValue(expected, 211, readFerryValue<double>(baseline, 179));
    writeFerryValue(expected, 219, readFerryValue<double>(baseline, 187));

    pdg::las::applyDefaultFerry(actual, input, program, dimensions, 3);
    EXPECT_EQ(actual, expected);
}

TEST(LasFerry, RecomputesLogicalReturnCountsExactly)
{
    const auto bytes = readFerryFixture("las/simple.las");
    const pdg::las::FileView input(bytes);
    const auto baseline = pdg::las::translateDefault(input, ferryMetadata(), 1);
    auto expected = baseline;
    auto actual = baseline;

    const auto classification =
        pdg::DimensionId(pdg::StandardDimension::Classification);
    const auto returnNumber =
        pdg::DimensionId(pdg::StandardDimension::ReturnNumber);
    const pdg::FerryProgram program{
        {{true, classification, returnNumber, false}}};
    pdg::DimensionRegistry dimensions;
    std::array<std::uint64_t, 15> counts{};
    constexpr std::size_t HeaderBytes = 375;
    constexpr std::size_t PointBytes = 36;
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(input.header().pointCount); ++index)
    {
        const std::size_t record = HeaderBytes + index * PointBytes;
        const std::uint8_t classificationValue =
            readFerryValue<std::uint8_t>(baseline, record + 16);
        const std::uint8_t numberOfReturns =
            (readFerryValue<std::uint8_t>(baseline, record + 14) >> 4U) & 0x0fU;
        writeFerryValue(expected, record + 14,
                        static_cast<std::uint8_t>(classificationValue |
                                                  (numberOfReturns << 4U)));
        if (classificationValue >= 1 && classificationValue <= counts.size())
            ++counts[classificationValue - 1U];
    }
    for (std::size_t index = 0; index < counts.size(); ++index)
        writeFerryValue(expected, 255 + index * sizeof(std::uint64_t),
                        counts[index]);

    pdg::las::applyDefaultFerry(actual, input, program, dimensions, 5);
    EXPECT_EQ(actual, expected);
}

TEST(LasFerry, UsesConservativeExactEnvelope)
{
    const auto formatThreeBytes = readFerryFixture("las/simple.las");
    const pdg::las::FileView formatThree(formatThreeBytes);
    pdg::DimensionRegistry dimensions;
    const auto intensity = pdg::DimensionId(pdg::StandardDimension::Intensity);
    const auto scanAngle =
        pdg::DimensionId(pdg::StandardDimension::ScanAngleRank);
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(
        formatThree, {{{true, scanAngle, intensity, false}}}, dimensions));
    const auto gpsTime = pdg::DimensionId(pdg::StandardDimension::GpsTime);
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(
        formatThree, {{{true, scanAngle, gpsTime, false}}}, dimensions));
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(
        formatThree, {{{true, intensity, scanAngle, false}}}, dimensions));

    const auto formatZeroBytes = readFerryFixture("las/hextest.las");
    const pdg::las::FileView formatZero(formatZeroBytes);
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(
        formatZero, {{{true, gpsTime, intensity, false}}}, dimensions));

    const auto scaledBytes = readFerryFixture("las/sample_c.las");
    const pdg::las::FileView scaled(scaledBytes);
    const auto x = pdg::DimensionId(pdg::StandardDimension::X);
    const auto z = pdg::DimensionId(pdg::StandardDimension::Z);
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(scaled, {{{true, x, z, false}}},
                                                dimensions));
    EXPECT_TRUE(pdg::las::supportsDefaultFerry(
        scaled, {{{true, intensity, z, false}}}, dimensions));

    const auto custom =
        dimensions.registerCustom("Custom", pdg::DimensionType::Unsigned16).id;
    EXPECT_FALSE(pdg::las::supportsDefaultFerry(
        formatThree, {{{true, intensity, custom, true}}}, dimensions));

    auto output = pdg::las::translateDefault(formatThree, ferryMetadata(), 1);
    EXPECT_THROW(pdg::las::applyDefaultFerry(
                     output, formatThree,
                     {{{true, intensity, scanAngle, false}}}, dimensions),
                 pdg::las::Error);
}

TEST(LasPointProgram, FusesOrderedAssignAndFerrySemanticsInOnePass)
{
    const auto bytes = readFerryFixture("las/simple.las");
    const pdg::las::FileView input(bytes);
    const auto baseline = pdg::las::translateDefault(input, ferryMetadata(), 1);
    auto expected = baseline;
    auto serial = baseline;
    auto parallel = baseline;

    pdg::DimensionRegistry dimensions;
    const std::vector<std::string> specifications = {
        "Scratch = Intensity * 2",
        "Classification = Scratch / 2 WHERE Intensity <= 255",
        "PointSourceId = Classification + 1", "X = X + 0.01",
        "ReturnNumber = 2 WHERE Classification >= 0"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(specifications, dimensions);
    ASSERT_TRUE(
        pdg::las::supportsDefaultPointProgram(input, program, dimensions));

    constexpr std::size_t HeaderBytes = 375;
    constexpr std::size_t PointBytes = 36;
    std::uint64_t pointCount = 0;
    const pdg::CoordinateEncoding coordinates =
        pdg::las::FileView(baseline).header().coordinateEncoding();
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(input.header().pointCount); ++index)
    {
        const std::size_t record = HeaderBytes + index * PointBytes;
        const std::uint16_t intensity =
            readFerryValue<std::uint16_t>(baseline, record + 12);
        std::uint8_t classification =
            readFerryValue<std::uint8_t>(baseline, record + 16);
        if (intensity <= 255)
            classification = static_cast<std::uint8_t>(intensity);
        writeFerryValue(expected, record + 16, classification);
        writeFerryValue(expected, record + 20,
                        static_cast<std::uint16_t>(classification + 1U));

        const std::int32_t rawX =
            readFerryValue<std::int32_t>(baseline, record);
        const std::int32_t assignedX =
            coordinates.encode(0, coordinates.decode(0, rawX) + 0.01);
        writeFerryValue(expected, record, assignedX);

        const std::uint8_t packedReturns =
            readFerryValue<std::uint8_t>(baseline, record + 14);
        writeFerryValue(
            expected, record + 14,
            static_cast<std::uint8_t>((packedReturns & 0xf0U) | 0x02U));
        ++pointCount;
    }
    writeFerryValue(expected, 179,
                    readFerryValue<double>(baseline, 179) + 0.01);
    writeFerryValue(expected, 187,
                    readFerryValue<double>(baseline, 187) + 0.01);
    for (std::size_t index = 0; index < 15; ++index)
        writeFerryValue(expected, 255 + index * sizeof(std::uint64_t),
                        index == 1 ? pointCount : 0U);

    pdg::las::applyDefaultPointProgram(serial, input, program, dimensions, 1);
    pdg::las::applyDefaultPointProgram(parallel, input, program, dimensions, 4);
    EXPECT_EQ(serial, expected);
    EXPECT_EQ(parallel, expected);
}

TEST(LasPointProgram, RejectsUnrepresentableInputSemanticsConservatively)
{
    const auto scaledBytes = readFerryFixture("las/sample_c.las");
    const pdg::las::FileView scaled(scaledBytes);
    pdg::DimensionRegistry coordinateDimensions;
    const std::vector<std::string> coordinateSpecification = {"Z = X + 1"};
    const pdg::AssignProgram coordinateProgram =
        pdg::compileAssignments(coordinateSpecification, coordinateDimensions);
    EXPECT_FALSE(pdg::las::supportsDefaultPointProgram(
        scaled, coordinateProgram, coordinateDimensions));

    const auto formatZeroBytes = readFerryFixture("las/hextest.las");
    const pdg::las::FileView formatZero(formatZeroBytes);
    pdg::DimensionRegistry gpsDimensions;
    const std::vector<std::string> gpsSpecification = {"Intensity = GpsTime"};
    const pdg::AssignProgram gpsProgram =
        pdg::compileAssignments(gpsSpecification, gpsDimensions);
    EXPECT_FALSE(pdg::las::supportsDefaultPointProgram(formatZero, gpsProgram,
                                                       gpsDimensions));

    const auto formatThreeBytes = readFerryFixture("las/simple.las");
    const pdg::las::FileView formatThree(formatThreeBytes);
    pdg::DimensionRegistry angleDimensions;
    const std::vector<std::string> angleSpecification = {
        "ScanAngleRank = Intensity"};
    const pdg::AssignProgram angleProgram =
        pdg::compileAssignments(angleSpecification, angleDimensions);
    EXPECT_FALSE(pdg::las::supportsDefaultPointProgram(
        formatThree, angleProgram, angleDimensions));
}
