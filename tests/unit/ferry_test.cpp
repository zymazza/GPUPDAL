#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ferry.hpp>

#include <pdal/util/Utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
pdg::CoordinateEncoding coordinates()
{
    return {{0.001, 0.002, 0.005}, {500000.0, 4800000.0, 100.0}};
}

template <typename T> std::vector<T> sourceValues()
{
    if constexpr (std::is_same_v<T, float>)
        return {-std::numeric_limits<float>::max(),
                -2.5F,
                -0.5F,
                0.0F,
                0.5F,
                2.5F,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN()};
    else if constexpr (std::is_same_v<T, double>)
        return {-std::numeric_limits<double>::infinity(),
                -9007199254740991.0,
                -2147483648.5,
                -2147483648.49,
                -255.5,
                -128.5,
                -127.5,
                -2.5,
                -1.5,
                -0.5,
                0.0,
                0.49,
                0.5,
                1.5,
                127.49,
                127.5,
                255.49,
                255.5,
                65535.49,
                65535.5,
                2147483647.49,
                2147483647.5,
                4294967295.49,
                4294967295.5,
                9007199254740991.0,
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()};
    else if constexpr (std::is_signed_v<T>)
    {
        const T high = []
        {
            if constexpr (sizeof(T) == sizeof(std::int64_t))
                return static_cast<T>(std::numeric_limits<T>::max() - 2048);
            else
                return std::numeric_limits<T>::max();
        }();
        return {std::numeric_limits<T>::lowest(),
                static_cast<T>(-2),
                static_cast<T>(-1),
                static_cast<T>(0),
                static_cast<T>(1),
                static_cast<T>(2),
                high};
    }
    else
    {
        const T high = []
        {
            if constexpr (sizeof(T) == sizeof(std::uint64_t))
                return static_cast<T>(std::numeric_limits<T>::max() - 4096);
            else
                return std::numeric_limits<T>::max();
        }();
        return {static_cast<T>(0), static_cast<T>(1), static_cast<T>(2), high};
    }
}

template <typename Source, typename Destination>
void compareConversionPair(pdg::DimensionType sourceType,
                           pdg::DimensionType destinationType)
{
    pdg::DimensionRegistry dimensions;
    const auto source = dimensions.registerCustom("Source", sourceType).id;
    const auto destination =
        dimensions.registerCustom("Destination", destinationType).id;
    const std::vector<Source> values = sourceValues<Source>();
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(values.size(), coordinates(), dimensions, memory);
    batch.materialize(source);
    batch.materialize(destination);
    batch.setSize(values.size());
    std::copy(values.begin(), values.end(), batch.data<Source>(source));

    std::vector<Destination> expected(values.size(),
                                      static_cast<Destination>(7));
    std::fill_n(batch.data<Destination>(destination), values.size(),
                static_cast<Destination>(7));
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const double value = static_cast<double>(values[index]);
        static_cast<void>(pdal::Utils::numericCast(value, expected[index]));
    }

    pdg::executeFerry(batch, {{{true, source, destination, false}}}, 1);
    EXPECT_EQ(std::memcmp(batch.data<Destination>(destination), expected.data(),
                          expected.size() * sizeof(Destination)),
              0)
        << "source type " << static_cast<unsigned>(sourceType)
        << ", destination type " << static_cast<unsigned>(destinationType);
}

template <typename Source>
void compareAllDestinations(pdg::DimensionType sourceType)
{
    compareConversionPair<Source, std::int8_t>(sourceType,
                                               pdg::DimensionType::Signed8);
    compareConversionPair<Source, std::int16_t>(sourceType,
                                                pdg::DimensionType::Signed16);
    compareConversionPair<Source, std::int32_t>(sourceType,
                                                pdg::DimensionType::Signed32);
    compareConversionPair<Source, std::int64_t>(sourceType,
                                                pdg::DimensionType::Signed64);
    compareConversionPair<Source, std::uint8_t>(sourceType,
                                                pdg::DimensionType::Unsigned8);
    compareConversionPair<Source, std::uint16_t>(
        sourceType, pdg::DimensionType::Unsigned16);
    compareConversionPair<Source, std::uint32_t>(
        sourceType, pdg::DimensionType::Unsigned32);
    compareConversionPair<Source, std::uint64_t>(
        sourceType, pdg::DimensionType::Unsigned64);
    compareConversionPair<Source, float>(sourceType, pdg::DimensionType::Float);
    compareConversionPair<Source, double>(sourceType,
                                          pdg::DimensionType::Double);
}
} // unnamed namespace

TEST(Ferry, MatchesPinnedPdalNumericCastAcrossAllPhysicalTypePairs)
{
    compareAllDestinations<std::int8_t>(pdg::DimensionType::Signed8);
    compareAllDestinations<std::int16_t>(pdg::DimensionType::Signed16);
    compareAllDestinations<std::int32_t>(pdg::DimensionType::Signed32);
    compareAllDestinations<std::int64_t>(pdg::DimensionType::Signed64);
    compareAllDestinations<std::uint8_t>(pdg::DimensionType::Unsigned8);
    compareAllDestinations<std::uint16_t>(pdg::DimensionType::Unsigned16);
    compareAllDestinations<std::uint32_t>(pdg::DimensionType::Unsigned32);
    compareAllDestinations<std::uint64_t>(pdg::DimensionType::Unsigned64);
    compareAllDestinations<float>(pdg::DimensionType::Float);
    compareAllDestinations<double>(pdg::DimensionType::Double);
}

TEST(Ferry, PreservesProgramOrderAndInitializesNewDestinations)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto middle =
        dimensions.registerCustom("Middle", pdg::DimensionType::Signed16).id;
    const auto result =
        dimensions.registerCustom("Result", pdg::DimensionType::Double).id;
    const auto empty =
        dimensions.registerCustom("Empty", pdg::DimensionType::Double).id;
    const auto unchanged =
        dimensions.registerCustom("Unchanged", pdg::DimensionType::Unsigned16)
            .id;

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(5, coordinates(), dimensions, memory);
    batch.materialize(source);
    batch.materialize(middle);
    batch.materialize(result);
    batch.materialize(unchanged);
    batch.setSize(5);
    const double input[] = {-1.6, -0.5, 0.49, 0.5, 1.6};
    std::copy(std::begin(input), std::end(input), batch.data<double>(source));
    std::fill_n(batch.data<std::int16_t>(middle), 5, 99);
    std::fill_n(batch.data<double>(result), 5, -99.0);
    std::fill_n(batch.data<std::uint16_t>(unchanged), 5, 42);

    pdg::FerryProgram program;
    program.copies = {{true, source, middle, false},
                      {true, middle, result, false},
                      {false, {}, empty, true},
                      {false, {}, unchanged, false}};
    pdg::executeFerry(batch, program, 1);

    const std::int16_t expectedMiddle[] = {-2, -1, 0, 1, 2};
    const double expectedResult[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    EXPECT_TRUE(std::equal(std::begin(expectedMiddle), std::end(expectedMiddle),
                           batch.data<std::int16_t>(middle)));
    EXPECT_TRUE(std::equal(std::begin(expectedResult), std::end(expectedResult),
                           batch.data<double>(result)));
    ASSERT_TRUE(batch.has(empty));
    EXPECT_TRUE(std::all_of(batch.data<double>(empty),
                            batch.data<double>(empty) + batch.size(),
                            [](double value) { return value == 0.0; }));
    EXPECT_TRUE(std::all_of(batch.data<std::uint16_t>(unchanged),
                            batch.data<std::uint16_t>(unchanged) + batch.size(),
                            [](std::uint16_t value) { return value == 42; }));
}

TEST(Ferry, ConvertsBetweenLogicalCoordinatesAndPhysicalLasIntegers)
{
    pdg::DimensionRegistry dimensions;
    const auto x = pdg::DimensionId(pdg::StandardDimension::X);
    const auto z = pdg::DimensionId(pdg::StandardDimension::Z);
    const auto decoded =
        dimensions.registerCustom("DecodedX", pdg::DimensionType::Double).id;
    const auto newZ =
        dimensions.registerCustom("NewZ", pdg::DimensionType::Double).id;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(4, coordinates(), dimensions, memory);
    batch.materialize(x);
    batch.materialize(z);
    batch.materialize(newZ);
    batch.setSize(4);
    const std::int32_t rawX[] = {-2, -1, 0, 1234};
    const double zValues[] = {99.995, 100.0, 100.005, 106.17};
    std::copy(std::begin(rawX), std::end(rawX), batch.data<std::int32_t>(x));
    std::copy(std::begin(zValues), std::end(zValues), batch.data<double>(newZ));

    pdg::executeFerry(batch,
                      {{{true, x, decoded, true}, {true, newZ, z, false}}}, 1);

    for (std::size_t index = 0; index < batch.size(); ++index)
        EXPECT_DOUBLE_EQ(batch.data<double>(decoded)[index],
                         coordinates().decode(0, rawX[index]));
    const std::int32_t expectedRawZ[] = {-1, 0, 1, 1234};
    EXPECT_TRUE(std::equal(std::begin(expectedRawZ), std::end(expectedRawZ),
                           batch.data<std::int32_t>(z)));
}

TEST(Ferry, IsBitIdenticalAcrossHostWorkerCounts)
{
    constexpr std::size_t Count = 131073;
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto destination =
        dimensions.registerCustom("Destination", pdg::DimensionType::Unsigned16)
            .id;
    pdg::HostMemoryResource memory;
    pdg::PointBatch serial(Count, coordinates(), dimensions, memory);
    pdg::PointBatch parallel(Count, coordinates(), dimensions, memory);
    for (pdg::PointBatch* batch : {&serial, &parallel})
    {
        batch->materialize(source);
        batch->materialize(destination);
        batch->setSize(Count);
        for (std::size_t index = 0; index < Count; ++index)
        {
            batch->data<double>(source)[index] =
                static_cast<double>((index * 37U) % 70000U) - 2000.5;
            batch->data<std::uint16_t>(destination)[index] =
                static_cast<std::uint16_t>(index % 65536U);
        }
    }

    const pdg::FerryProgram program{{{true, source, destination, false}}};
    pdg::executeFerry(serial, program, 1);
    pdg::executeFerry(parallel, program, 7);
    EXPECT_EQ(std::memcmp(serial.data<std::uint16_t>(destination),
                          parallel.data<std::uint16_t>(destination),
                          Count * sizeof(std::uint16_t)),
              0);
}

TEST(Ferry, RejectsAnUnmaterializedSourceColumn)
{
    pdg::DimensionRegistry dimensions;
    const auto source =
        dimensions.registerCustom("Source", pdg::DimensionType::Double).id;
    const auto destination =
        dimensions.registerCustom("Destination", pdg::DimensionType::Double).id;
    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(1, coordinates(), dimensions, memory);
    batch.materialize(destination);
    batch.setSize(1);
    EXPECT_THROW(
        pdg::executeFerry(batch, {{{true, source, destination, false}}}, 1),
        std::invalid_argument);
}
