#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/io/LasCuda.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslate.hpp>
#include <pdg/io/LasTranslateCuda.hpp>
#include <pdg/stages/Assign.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
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
std::vector<std::byte> readCudaFixture(const std::string& relative)
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
void writeCudaValue(std::vector<std::byte>& bytes, std::size_t offset, T value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw std::out_of_range("synthetic CUDA LAS write exceeds buffer");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <typename T>
T readCudaValue(std::span<const std::byte> bytes, std::size_t offset)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw std::out_of_range("synthetic CUDA LAS read exceeds buffer");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::vector<std::byte> syntheticCudaLas(std::uint8_t pointFormat,
                                        std::size_t pointCount)
{
    const bool extended = pointFormat >= 6;
    const std::size_t headerBytes = extended ? 375 : 227;
    const std::size_t basePointBytes =
        pdg::las::minimumPointRecordLength(pointFormat);
    constexpr std::size_t ExtraPointBytes = 3;
    const std::size_t pointBytes = basePointBytes + ExtraPointBytes;
    std::vector<std::byte> bytes(headerBytes + pointCount * pointBytes);
    bytes[0] = std::byte{'L'};
    bytes[1] = std::byte{'A'};
    bytes[2] = std::byte{'S'};
    bytes[3] = std::byte{'F'};
    if (extended)
        writeCudaValue<std::uint16_t>(bytes, 6, 1U << 4U);
    writeCudaValue<std::uint8_t>(bytes, 24, 1);
    writeCudaValue<std::uint8_t>(bytes, 25, extended ? 4 : 2);
    writeCudaValue<std::uint16_t>(bytes, 94,
                                  static_cast<std::uint16_t>(headerBytes));
    writeCudaValue<std::uint32_t>(bytes, 96,
                                  static_cast<std::uint32_t>(headerBytes));
    writeCudaValue(bytes, 104, pointFormat);
    writeCudaValue<std::uint16_t>(bytes, 105,
                                  static_cast<std::uint16_t>(pointBytes));
    if (extended)
        writeCudaValue<std::uint64_t>(bytes, 247, pointCount);
    else
        writeCudaValue<std::uint32_t>(bytes, 107,
                                      static_cast<std::uint32_t>(pointCount));
    for (std::size_t axis = 0; axis < 3; ++axis)
        writeCudaValue<double>(bytes, 131 + axis * sizeof(double), 0.01);

    for (std::size_t index = 0; index < pointCount; ++index)
    {
        const std::size_t record = headerBytes + index * pointBytes;
        const std::int32_t coordinate =
            static_cast<std::int32_t>(index % 100003U) - 50001;
        writeCudaValue(bytes, record, coordinate);
        writeCudaValue(bytes, record + 4, -coordinate);
        writeCudaValue(bytes, record + 8,
                       static_cast<std::int32_t>((index * 65521U) % 200003U) -
                           100001);
        writeCudaValue<std::uint16_t>(bytes, record + 12,
                                      static_cast<std::uint16_t>(index * 257U));
        if (pointFormat <= 3)
        {
            const std::uint8_t returnNumber =
                static_cast<std::uint8_t>(index % 5U + 1U);
            const std::uint8_t numberOfReturns = static_cast<std::uint8_t>(
                std::max<std::size_t>(returnNumber, index % 5U + 1U));
            writeCudaValue<std::uint8_t>(
                bytes, record + 14,
                static_cast<std::uint8_t>(returnNumber |
                                          (numberOfReturns << 3U) |
                                          (((index >> 1U) & 1U) << 6U) |
                                          (((index >> 2U) & 1U) << 7U)));
            const std::uint8_t classification =
                static_cast<std::uint8_t>(index % 97U == 0 ? 12 : index % 32U);
            writeCudaValue<std::uint8_t>(
                bytes, record + 15,
                static_cast<std::uint8_t>(classification |
                                          (((index >> 3U) & 1U) << 5U) |
                                          (((index >> 4U) & 1U) << 6U) |
                                          (((index >> 5U) & 1U) << 7U)));
            writeCudaValue<std::int8_t>(
                bytes, record + 16,
                static_cast<std::int8_t>(static_cast<int>(index % 256U) - 128));
            writeCudaValue<std::uint8_t>(
                bytes, record + 17, static_cast<std::uint8_t>(index * 29U));
            writeCudaValue<std::uint16_t>(
                bytes, record + 18, static_cast<std::uint16_t>(index * 313U));
            if (pointFormat == 1 || pointFormat == 3)
                writeCudaValue<std::uint64_t>(bytes, record + 20,
                                              0x7ff8000000000000ULL | index);
            const std::size_t colorOffset = pointFormat == 2   ? 20
                                            : pointFormat == 3 ? 28
                                                               : 0;
            if (colorOffset)
                for (std::size_t channel = 0; channel < 3; ++channel)
                    writeCudaValue<std::uint16_t>(
                        bytes, record + colorOffset + channel * 2U,
                        static_cast<std::uint16_t>(index * (channel + 11U)));
        }
        else
        {
            const std::uint8_t returnNumber =
                static_cast<std::uint8_t>(index % 15U + 1U);
            const std::uint8_t numberOfReturns = static_cast<std::uint8_t>(
                std::max<std::size_t>(returnNumber, index % 15U + 1U));
            writeCudaValue<std::uint8_t>(
                bytes, record + 14,
                static_cast<std::uint8_t>(returnNumber |
                                          (numberOfReturns << 4U)));
            writeCudaValue<std::uint8_t>(
                bytes, record + 15, static_cast<std::uint8_t>(index * 73U));
            writeCudaValue<std::uint8_t>(
                bytes, record + 16, static_cast<std::uint8_t>(index * 67U));
            writeCudaValue<std::uint8_t>(
                bytes, record + 17, static_cast<std::uint8_t>(index * 29U));
            writeCudaValue<std::int16_t>(
                bytes, record + 18,
                static_cast<std::int16_t>(
                    static_cast<int>((index * 40503U) & 0xffffU) - 32768));
            writeCudaValue<std::uint16_t>(
                bytes, record + 20, static_cast<std::uint16_t>(index * 313U));
            writeCudaValue<std::uint64_t>(bytes, record + 22,
                                          0x7ff8000000000000ULL | index);
            if (pointFormat == 7 || pointFormat == 8)
                for (std::size_t channel = 0; channel < 3; ++channel)
                    writeCudaValue<std::uint16_t>(
                        bytes, record + 30 + channel * 2U,
                        static_cast<std::uint16_t>(index * (channel + 11U)));
            if (pointFormat == 8)
                writeCudaValue<std::uint16_t>(
                    bytes, record + 36,
                    static_cast<std::uint16_t>(index * 719U));
        }
        for (std::size_t extra = 0; extra < ExtraPointBytes; ++extra)
            bytes[record + basePointBytes + extra] =
                static_cast<std::byte>((index * 31U + extra * 17U) & 0xffU);
    }
    return bytes;
}

pdg::AssignProgram exactCudaPointProgram(pdg::DimensionRegistry& dimensions)
{
    const std::vector<std::string> specifications = {
        "Scratch = Classification + Intensity * 2",
        "Classification = Scratch / 257 WHERE ReturnNumber > 0",
        "UserData = Classification + NumberOfReturns",
        "PointSourceId = UserData * 11 + Intensity",
        "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"};
    return pdg::compileAssignments(specifications, dimensions);
}
} // unnamed namespace

TEST(CudaLasPointIo, RejectsInvalidLaunchArgumentsWithoutADriver)
{
    EXPECT_THROW(pdg::las::decodeCoordinatesAsync(nullptr, 11, 0, nullptr,
                                                  nullptr, nullptr, nullptr),
                 std::invalid_argument);
    EXPECT_THROW(pdg::las::decodeCoordinatesAsync(nullptr, 12, 1, nullptr,
                                                  nullptr, nullptr, nullptr),
                 std::invalid_argument);
    EXPECT_NO_THROW(pdg::las::decodeCoordinatesAsync(
        nullptr, 12, 0, nullptr, nullptr, nullptr, nullptr));
    const pdg::CoordinateEncoding encoding({1.0, 1.0, 1.0},
                                           {0.0, 0.0, 0.0});
    EXPECT_THROW(pdg::las::expandCoordinatesAsync(
                     nullptr, nullptr, nullptr, 1, encoding, nullptr, nullptr,
                     nullptr, nullptr),
                 std::invalid_argument);
    EXPECT_NO_THROW(pdg::las::expandCoordinatesAsync(
        nullptr, nullptr, nullptr, 0, encoding, nullptr, nullptr, nullptr,
        nullptr));
}

TEST(CudaMemoryResource, ValidatesPinnedAllocationsWithoutADriver)
{
    auto memory = pdg::makeCudaPinnedMemoryResource();
    EXPECT_EQ(memory->kind(), pdg::MemoryKind::PinnedHost);
    EXPECT_EQ(memory->nativeStreamHandle(), nullptr);
    EXPECT_THROW(static_cast<void>(memory->allocate(1, 3)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(memory->allocate(1, 512)),
                 std::invalid_argument);
    const auto empty = memory->allocate(0, alignof(std::max_align_t));
    EXPECT_EQ(empty->data(), nullptr);
    EXPECT_EQ(empty->size(), 0U);
    EXPECT_EQ(empty->kind(), pdg::MemoryKind::PinnedHost);
}

TEST(CudaLasTranslation, EmptyOutputMatchesHostWithoutADriver)
{
    const auto bytes = syntheticCudaLas(7, 0);
    const pdg::las::FileView file(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto expected = pdg::las::translateDefault(file, metadata, 1);
    const auto actual = pdg::las::translateDefaultCuda(file, metadata, 257);
    EXPECT_EQ(actual, expected);
    std::vector<std::byte> callerOwned(expected.size(), std::byte{0xff});
    pdg::las::translateDefaultCudaInto(file, metadata, callerOwned, 257);
    EXPECT_EQ(callerOwned, expected);
    callerOwned.pop_back();
    EXPECT_THROW(
        pdg::las::translateDefaultCudaInto(file, metadata, callerOwned, 257),
        pdg::las::Error);
    EXPECT_THROW(
        static_cast<void>(pdg::las::translateDefaultCuda(file, metadata, 0)),
        std::invalid_argument);
}

TEST(CudaLasPointIo, CoordinateRoundTripIsByteExact)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    const auto bytes = readCudaFixture("las/simple.las");
    const pdg::las::FileView file(bytes);
    const std::size_t count =
        static_cast<std::size_t>(file.header().pointCount);
    const std::size_t stride = file.header().pointRecordLength;
    const std::size_t recordBytes = count * stride;
    const auto recordsBegin = bytes.begin() + file.header().pointDataOffset;
    const std::vector<std::byte> expected(recordsBegin,
                                          recordsBegin + recordBytes);
    std::vector<std::byte> actual(recordBytes);

    auto memory = pdg::makeCudaMemoryResource(64U * 1024U * 1024U);
    EXPECT_THROW(static_cast<void>(memory->allocate(1, 3)),
                 std::invalid_argument);
    auto records = memory->allocate(recordBytes, alignof(std::max_align_t));
    auto x =
        memory->allocate(count * sizeof(std::int32_t), alignof(std::int32_t));
    auto y =
        memory->allocate(count * sizeof(std::int32_t), alignof(std::int32_t));
    auto z =
        memory->allocate(count * sizeof(std::int32_t), alignof(std::int32_t));
    auto decodedX =
        memory->allocate(count * sizeof(double), alignof(double));
    auto decodedY =
        memory->allocate(count * sizeof(double), alignof(double));
    auto decodedZ =
        memory->allocate(count * sizeof(double), alignof(double));
    std::vector<double> logicalX(count);
    std::vector<double> logicalY(count);
    std::vector<double> logicalZ(count);
    const auto stream = static_cast<cudaStream_t>(memory->nativeStreamHandle());

    PDG_CUDA_CHECK(cudaMemcpyAsync(records->data(), expected.data(),
                                   recordBytes, cudaMemcpyHostToDevice,
                                   stream));
    pdg::las::decodeCoordinatesAsync(
        records->data(), stride, count, static_cast<std::int32_t*>(x->data()),
        static_cast<std::int32_t*>(y->data()),
        static_cast<std::int32_t*>(z->data()), stream);
    pdg::las::expandCoordinatesAsync(
        static_cast<const std::int32_t*>(x->data()),
        static_cast<const std::int32_t*>(y->data()),
        static_cast<const std::int32_t*>(z->data()), count,
        file.header().coordinateEncoding(),
        static_cast<double*>(decodedX->data()),
        static_cast<double*>(decodedY->data()),
        static_cast<double*>(decodedZ->data()), stream);
    pdg::las::packCoordinatesAsync(static_cast<const std::int32_t*>(x->data()),
                                   static_cast<const std::int32_t*>(y->data()),
                                   static_cast<const std::int32_t*>(z->data()),
                                   count, records->data(), stride, stream);
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), records->data(), recordBytes,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(logicalX.data(), decodedX->data(),
                                   count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(logicalY.data(), decodedY->data(),
                                   count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(logicalZ.data(), decodedZ->data(),
                                   count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(actual, expected);
    const pdg::CoordinateEncoding encoding =
        file.header().coordinateEncoding();
    for (std::size_t point = 0; point < count; ++point)
    {
        EXPECT_EQ(logicalX[point],
                  encoding.decode(0, file.rawCoordinate(point, 0)));
        EXPECT_EQ(logicalY[point],
                  encoding.decode(1, file.rawCoordinate(point, 1)));
        EXPECT_EQ(logicalZ[point],
                  encoding.decode(2, file.rawCoordinate(point, 2)));
    }
}

TEST(CudaLasTranslation, CompleteOutputMatchesExactHostPath)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    const auto bytes = readCudaFixture("las/simple.las");
    const pdg::las::FileView file(bytes);
    ASSERT_TRUE(pdg::las::supportsDefaultCudaTranslation(file));
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto expected = pdg::las::translateDefault(file, metadata, 1);
    const auto actual = pdg::las::translateDefaultCuda(file, metadata, 257);
    EXPECT_EQ(actual, expected);
}

TEST(CudaLasTranslation, SupportedFormatMatrixMatchesHostAcrossChunks)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    constexpr std::array<std::uint8_t, 7> Formats{0, 1, 2, 3, 6, 7, 8};
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    for (const std::uint8_t pointFormat : Formats)
    {
        SCOPED_TRACE(static_cast<int>(pointFormat));
        const auto bytes = syntheticCudaLas(pointFormat, 65537);
        const pdg::las::FileView file(bytes);
        ASSERT_TRUE(pdg::las::supportsDefaultCudaTranslation(file));
        const auto expected = pdg::las::translateDefault(file, metadata, 1);
        const auto actual =
            pdg::las::translateDefaultCuda(file, metadata, 4093);
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaLasPointIo, CanonicalColumnsRoundTripEverySupportedField)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    constexpr std::size_t Count = 8193;
    const auto source = syntheticCudaLas(7, Count);
    const pdg::las::FileView sourceFile(source);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto expected = pdg::las::translateDefault(sourceFile, metadata, 1);
    const pdg::las::FileView canonical(expected);
    const std::vector<pdg::DimensionId> fields = {
        pdg::DimensionId(pdg::StandardDimension::X),
        pdg::DimensionId(pdg::StandardDimension::Y),
        pdg::DimensionId(pdg::StandardDimension::Z),
        pdg::DimensionId(pdg::StandardDimension::Intensity),
        pdg::DimensionId(pdg::StandardDimension::ReturnNumber),
        pdg::DimensionId(pdg::StandardDimension::NumberOfReturns),
        pdg::DimensionId(pdg::StandardDimension::ScanDirectionFlag),
        pdg::DimensionId(pdg::StandardDimension::EdgeOfFlightLine),
        pdg::DimensionId(pdg::StandardDimension::Classification),
        pdg::DimensionId(pdg::StandardDimension::ScanAngleRank),
        pdg::DimensionId(pdg::StandardDimension::UserData),
        pdg::DimensionId(pdg::StandardDimension::PointSourceId),
        pdg::DimensionId(pdg::StandardDimension::Red),
        pdg::DimensionId(pdg::StandardDimension::Green),
        pdg::DimensionId(pdg::StandardDimension::Blue),
        pdg::DimensionId(pdg::StandardDimension::GpsTime),
        pdg::DimensionId(pdg::StandardDimension::ScanChannel),
        pdg::DimensionId(pdg::StandardDimension::Synthetic),
        pdg::DimensionId(pdg::StandardDimension::KeyPoint),
        pdg::DimensionId(pdg::StandardDimension::Withheld),
        pdg::DimensionId(pdg::StandardDimension::Overlap)};
    std::vector<pdg::DimensionId> writableFields = fields;
    std::erase(writableFields,
               pdg::DimensionId(pdg::StandardDimension::ScanAngleRank));

    const std::size_t recordsBytes = Count * 36U;
    auto memory = pdg::makeCudaMemoryResource(64U * 1024U * 1024U);
    auto records = memory->allocate(recordsBytes, alignof(std::max_align_t));
    pdg::DimensionRegistry dimensions;
    pdg::PointBatch batch(Count, canonical.header().coordinateEncoding(),
                          dimensions, *memory);
    for (const pdg::DimensionId field : fields)
        batch.materialize(field);
    batch.setSize(Count);
    const auto stream = static_cast<cudaStream_t>(memory->nativeStreamHandle());
    const std::byte* expectedRecords = expected.data() + 375;
    std::vector<std::byte> actual(recordsBytes);

    PDG_CUDA_CHECK(cudaMemcpyAsync(records->data(), expectedRecords,
                                   recordsBytes, cudaMemcpyHostToDevice,
                                   stream));
    pdg::las::decodeCanonicalColumnsAsync(records->data(), Count, batch, fields,
                                          stream);
    pdg::las::packCanonicalColumnsAsync(batch, writableFields, Count,
                                        records->data(), stream);
    PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), records->data(), recordsBytes,
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expectedRecords));
}

TEST(CudaLasPointProgram, FusedJitSpecializesEveryAdmittedFormat)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    // The requirement hook makes an interpreter fallback a hard failure,
    // so this matrix proves the NVRTC specialization engages and matches
    // the host oracle bit for bit on every admitted record format.
    struct ScopedRequireJit
    {
        ScopedRequireJit()
        {
            ::setenv("PDG_REQUIRE_FUSED_JIT", "1", 1);
        }
        ~ScopedRequireJit()
        {
            ::unsetenv("PDG_REQUIRE_FUSED_JIT");
        }
    } scopedRequireJit;

    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program = exactCudaPointProgram(dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    constexpr std::array<std::uint8_t, 7> Formats{0, 1, 2, 3, 6, 7, 8};
    for (const std::uint8_t pointFormat : Formats)
    {
        SCOPED_TRACE(static_cast<int>(pointFormat));
        const auto bytes = syntheticCudaLas(pointFormat, 65537);
        const pdg::las::FileView file(bytes);
        auto expected = pdg::las::translateDefault(file, metadata, 1);
        pdg::las::applyDefaultPointProgram(expected, file, program, dimensions,
                                           3);
        std::vector<std::byte> actual(expected.size(), std::byte{0xff});
        pdg::las::translateDefaultPointProgramCudaInto(
            file, metadata, program, dimensions, actual, 4093);
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaLasPointProgram, FusedProgramMatchesHostAcrossFormatsAndChunks)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program = exactCudaPointProgram(dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    constexpr std::array<std::uint8_t, 7> Formats{0, 1, 2, 3, 6, 7, 8};

    for (const std::uint8_t pointFormat : Formats)
    {
        SCOPED_TRACE(static_cast<int>(pointFormat));
        const auto bytes = syntheticCudaLas(pointFormat, 65537);
        const pdg::las::FileView file(bytes);
        ASSERT_TRUE(pdg::las::supportsDefaultCudaPointProgram(file, program,
                                                              dimensions));
        ASSERT_TRUE(pdg::las::supportsDefaultFusedCudaPointProgram(
            file, program, dimensions));
        auto expected = pdg::las::translateDefault(file, metadata, 1);
        pdg::las::applyDefaultPointProgram(expected, file, program, dimensions,
                                           3);
        std::vector<std::byte> actual(expected.size(), std::byte{0xff});
        pdg::las::translateDefaultPointProgramCudaInto(
            file, metadata, program, dimensions, actual, 4093);
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaLasPointProgram, FusedSingleKernelTwoLaneSanitizerSmoke)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program = exactCudaPointProgram(dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(program));
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto bytes = syntheticCudaLas(7, 1025);
    const pdg::las::FileView file(bytes);
    ASSERT_TRUE(pdg::las::supportsDefaultFusedCudaPointProgram(file, program,
                                                               dimensions));
    auto expected = pdg::las::translateDefault(file, metadata, 1);
    pdg::las::applyDefaultPointProgram(expected, file, program, dimensions, 3);
    std::vector<std::byte> actual(expected.size(), std::byte{0xff});
    std::vector<std::uint8_t> coverage(expected.size());

    // Four launches force both streams to drain and be reused.
    pdg::las::translateDefaultPointProgramCudaToSink(
        file, metadata, program, dimensions,
        [&](std::size_t offset, std::span<const std::byte> chunk)
        {
            if (offset > actual.size() || chunk.size() > actual.size() - offset)
                throw std::out_of_range("CUDA test sink extent");
            for (std::size_t index = 0; index < chunk.size(); ++index)
            {
                if (coverage[offset + index]++)
                    throw std::runtime_error("overlapping CUDA test chunks");
            }
            std::memcpy(actual.data() + offset, chunk.data(), chunk.size());
        },
        257);
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(std::ranges::all_of(coverage, [](std::uint8_t value)
                                    { return value == 1; }));
}

TEST(CudaLasPointProgram, BoundedLaneSweepIsExact)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program = exactCudaPointProgram(dimensions);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto bytes = syntheticCudaLas(7, 1543);
    const pdg::las::FileView file(bytes);
    auto expected = pdg::las::translateDefault(file, metadata, 1);
    pdg::las::applyDefaultPointProgram(expected, file, program, dimensions, 3);

    for (std::size_t lanes = pdg::MinimumSweptLaneCount;
         lanes <= pdg::MaximumSweptLaneCount; ++lanes)
    {
        SCOPED_TRACE(lanes);
        std::vector<std::byte> actual(expected.size(), std::byte{0xff});
        std::vector<std::uint8_t> coverage(expected.size());
        pdg::las::translateDefaultPointProgramCudaToSink(
            file, metadata, program, dimensions,
            [&](std::size_t offset, std::span<const std::byte> chunk)
            {
                ASSERT_LE(offset + chunk.size(), actual.size());
                for (std::size_t index = 0; index < chunk.size(); ++index)
                    EXPECT_EQ(coverage[offset + index]++, 0U);
                std::memcpy(actual.data() + offset, chunk.data(), chunk.size());
            },
            257, lanes);
        EXPECT_EQ(actual, expected);
        EXPECT_TRUE(std::ranges::all_of(coverage, [](std::uint8_t value)
                                        { return value == 1U; }));
    }
}

TEST(CudaLasPointProgram, ReportsPlannerBudgetedDirectSchedule)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram program = exactCudaPointProgram(dimensions);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    const auto bytes = syntheticCudaLas(7, 1543);
    const pdg::las::FileView file(bytes);
    auto expected = pdg::las::translateDefault(file, metadata, 1);
    pdg::las::applyDefaultPointProgram(expected, file, program, dimensions, 3);

    const auto execute = [&](std::size_t memoryBudgetBytes)
    {
        std::vector<std::byte> actual(expected.size(), std::byte{0xff});
        pdg::las::CudaTranslationMetrics metrics;
        const pdg::TiledSchedule schedule =
            pdg::las::translateDefaultPointProgramCudaToSink(
                file, metadata, program, dimensions,
                [&](std::size_t offset, std::span<const std::byte> chunk)
                {
                    ASSERT_LE(offset + chunk.size(), actual.size());
                    std::memcpy(actual.data() + offset, chunk.data(),
                                chunk.size());
                },
                257, 0, memoryBudgetBytes, &metrics);
        EXPECT_EQ(actual, expected);
        const std::size_t inputPointBytes =
            static_cast<std::size_t>(file.header().pointCount) *
            file.header().pointRecordLength;
        const std::size_t outputPointBytes =
            static_cast<std::size_t>(file.header().pointCount) * 36U;
        if (metrics.hostToDeviceBytes <= inputPointBytes ||
            metrics.deviceToHostBytes <= outputPointBytes ||
            !schedule.tileCount)
        {
            ADD_FAILURE() << "successful CUDA execution omitted transfer facts";
            return schedule;
        }
        const std::size_t inputSummaryBytes =
            metrics.hostToDeviceBytes - inputPointBytes;
        const std::size_t outputSummaryBytes =
            metrics.deviceToHostBytes - outputPointBytes;
        EXPECT_EQ(inputSummaryBytes, outputSummaryBytes);
        EXPECT_EQ(inputSummaryBytes % schedule.tileCount, 0U);
        return schedule;
    };

    const pdg::TiledSchedule defaultSchedule = execute(0);
    ASSERT_EQ(defaultSchedule.configuredLaneCount, 2U);
    ASSERT_EQ(defaultSchedule.activeLaneCount, 2U);
    ASSERT_EQ(defaultSchedule.peakLaneBytes % 2U, 0U);
    const std::size_t bytesPerLane = defaultSchedule.peakLaneBytes / 2U;
    ASSERT_NE(bytesPerLane, 0U);

    const pdg::TiledSchedule limitedSchedule = execute(bytesPerLane);
    EXPECT_EQ(limitedSchedule.configuredLaneCount, 2U);
    EXPECT_EQ(limitedSchedule.activeLaneCount, 1U);
    EXPECT_EQ(limitedSchedule.peakLaneBytes, bytesPerLane);
    EXPECT_TRUE(limitedSchedule.memoryLimited);

    std::size_t sinkCalls = 0;
    EXPECT_THROW(
        static_cast<void>(pdg::las::translateDefaultPointProgramCudaToSink(
            file, metadata, program, dimensions,
            [&](std::size_t, std::span<const std::byte>) { ++sinkCalls; }, 257,
            0, bytesPerLane - 1U)),
        std::runtime_error);
    EXPECT_EQ(sinkCalls, 0U);
}

TEST(CudaLasPointProgram, OrderedPredicateTwoLaneSanitizerSmoke)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    constexpr std::size_t PointCount = 1025;
    constexpr std::size_t HeaderBytes = 375;
    constexpr std::size_t PointBytes = 36;
    pdg::DimensionRegistry dimensions;
    const pdg::AssignProgram assignment = pdg::compileAssignments(
        std::vector<std::string>{"UserData = Classification / 2"}, dimensions);
    const pdg::PredicateProgram predicate =
        pdg::compilePredicate("Classification >= 128", dimensions);
    pdg::las::OrderedPointProgram program;
    program.operations.emplace_back(assignment);
    program.operations.emplace_back(predicate);
    program.reads = {pdg::DimensionId(pdg::StandardDimension::Classification)};
    program.writes = {pdg::DimensionId(pdg::StandardDimension::UserData)};
    program.filtersPoints = true;

    const auto bytes = syntheticCudaLas(7, PointCount);
    const pdg::las::FileView file(bytes);
    ASSERT_TRUE(
        pdg::las::supportsDefaultCudaPointProgram(file, program, dimensions));
    // The operation list is authoritative. Stale or omitted planner summaries
    // must not suppress a written field during canonical repack.
    program.reads.clear();
    program.writes.clear();
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};
    std::vector<std::byte> actual(HeaderBytes + PointCount * PointBytes,
                                  std::byte{0xff});
    std::vector<std::uint8_t> coverage(actual.size());

    const std::uint64_t survivorCount =
        pdg::las::translateDefaultOrderedPointProgramCudaToSink(
            file, metadata, program, dimensions,
            [&](std::size_t offset, std::span<const std::byte> chunk)
            {
                if (offset > actual.size() ||
                    chunk.size() > actual.size() - offset)
                    throw std::out_of_range("CUDA test sink extent");
                for (std::size_t index = 0; index < chunk.size(); ++index)
                {
                    if (coverage[offset + index]++)
                        throw std::runtime_error(
                            "overlapping ordered CUDA test chunks");
                }
                std::memcpy(actual.data() + offset, chunk.data(), chunk.size());
            },
            257);

    std::vector<std::size_t> survivors;
    for (std::size_t index = 0; index < PointCount; ++index)
        if (static_cast<std::uint8_t>(index * 67U) >= 128U)
            survivors.push_back(index);
    ASSERT_EQ(survivorCount, survivors.size());
    actual.resize(HeaderBytes + survivors.size() * PointBytes);
    coverage.resize(actual.size());
    EXPECT_TRUE(std::ranges::all_of(coverage, [](std::uint8_t value)
                                    { return value == 1; }));
    EXPECT_EQ(pdg::las::FileView(actual).header().pointCount, survivorCount);

    const std::span<const std::byte> output(actual);
    for (std::size_t point = 0; point < survivors.size(); ++point)
    {
        const std::uint8_t classification =
            static_cast<std::uint8_t>(survivors[point] * 67U);
        const std::size_t record = HeaderBytes + point * PointBytes;
        EXPECT_EQ(readCudaValue<std::uint8_t>(output, record + 16),
                  classification);
        EXPECT_EQ(readCudaValue<std::uint8_t>(output, record + 17),
                  static_cast<std::uint8_t>((classification + 1U) / 2U));
    }
}

TEST(CudaLasPointProgram, OrderedOrdinalModesPreserveGlobalSequence)
{
    try
    {
        if (pdg::cudaDevices().empty())
            GTEST_SKIP() << "no CUDA device is available";
    }
    catch (const pdg::CudaError& error)
    {
        GTEST_SKIP() << "CUDA runtime is unavailable: " << error.what();
    }

    constexpr std::size_t PointCount = 1025;
    constexpr std::size_t HeaderBytes = 375;
    constexpr std::size_t PointBytes = 36;
    const auto bytes = syntheticCudaLas(7, PointCount);
    const pdg::las::FileView file(bytes);
    const pdg::las::DefaultTranslationMetadata metadata{
        1, 2024, pdg::las::oracleSoftwareId()};

    const auto execute = [&](pdg::las::OrderedPointProgram program,
                             const std::vector<std::size_t>& survivors)
    {
        pdg::DimensionRegistry dimensions;
        ASSERT_TRUE(pdg::las::supportsDefaultCudaPointProgram(file, program,
                                                              dimensions));
        std::vector<std::byte> actual(HeaderBytes + PointCount * PointBytes,
                                      std::byte{0xff});
        std::vector<std::uint8_t> coverage(actual.size());
        const std::uint64_t survivorCount =
            pdg::las::translateDefaultOrderedPointProgramCudaToSink(
                file, metadata, program, dimensions,
                [&](std::size_t offset, std::span<const std::byte> chunk)
                {
                    ASSERT_LE(offset + chunk.size(), actual.size());
                    for (std::size_t index = 0; index < chunk.size(); ++index)
                        EXPECT_EQ(coverage[offset + index]++, 0U);
                    std::memcpy(actual.data() + offset, chunk.data(),
                                chunk.size());
                },
                257);
        ASSERT_EQ(survivorCount, survivors.size());
        actual.resize(HeaderBytes + survivors.size() * PointBytes);
        coverage.resize(actual.size());
        EXPECT_TRUE(std::ranges::all_of(coverage, [](std::uint8_t value)
                                        { return value == 1U; }));
        EXPECT_EQ(pdg::las::FileView(actual).header().pointCount,
                  survivors.size());
        const std::span<const std::byte> output(actual);
        for (std::size_t point = 0; point < survivors.size(); ++point)
            EXPECT_EQ(readCudaValue<std::uint8_t>(
                          output, HeaderBytes + point * PointBytes + 16U),
                      static_cast<std::uint8_t>(survivors[point] * 67U));
    };

    pdg::OrdinalProgram streamingDecimation;
    streamingDecimation.kind = pdg::OrdinalKind::Decimation;
    streamingDecimation.step = 4.2;
    streamingDecimation.offset = 10;
    streamingDecimation.limit = 900;
    pdg::OrdinalProgram streamingHead;
    streamingHead.kind = pdg::OrdinalKind::Head;
    streamingHead.count = 75;
    streamingHead.invert = true;
    pdg::las::OrderedPointProgram streaming;
    streaming.operations.emplace_back(streamingDecimation);
    streaming.operations.emplace_back(streamingHead);
    streaming.filtersPoints = true;
    streaming.ordinalMode = pdg::OrdinalMode::Streaming;
    std::vector<std::size_t> streamingSurvivors;
    std::uint64_t kept = 0;
    std::uint64_t decimated = 0;
    for (std::uint64_t point = 0; point < PointCount; ++point)
    {
        if (point < streamingDecimation.offset ||
            point >= streamingDecimation.limit ||
            point !=
                streamingDecimation.offset +
                    static_cast<std::uint64_t>(std::llround(
                        static_cast<double>(kept) * streamingDecimation.step)))
            continue;
        ++kept;
        if (decimated++ >= streamingHead.count)
            streamingSurvivors.push_back(static_cast<std::size_t>(point));
    }
    execute(streaming, streamingSurvivors);

    pdg::OrdinalProgram standardDecimation;
    standardDecimation.kind = pdg::OrdinalKind::Decimation;
    standardDecimation.step = 4.2;
    standardDecimation.offset = 10;
    standardDecimation.limit = 900;
    pdg::OrdinalProgram standardHead;
    standardHead.kind = pdg::OrdinalKind::Head;
    standardHead.count = 200;
    pdg::OrdinalProgram standardTail;
    standardTail.kind = pdg::OrdinalKind::Tail;
    standardTail.count = 50;
    standardTail.invert = true;
    pdg::las::OrderedPointProgram standard;
    standard.operations.emplace_back(standardDecimation);
    standard.operations.emplace_back(standardHead);
    standard.operations.emplace_back(standardTail);
    standard.filtersPoints = true;
    standard.ordinalMode = pdg::OrdinalMode::Standard;
    std::vector<std::size_t> standardSurvivors;
    const std::uint64_t standardCount = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(standardDecimation.limit -
                                         standardDecimation.offset) /
                     standardDecimation.step));
    for (std::uint64_t sequence = 0;
         sequence < std::min<std::uint64_t>(standardCount, 150U); ++sequence)
        standardSurvivors.push_back(
            standardDecimation.offset +
            static_cast<std::size_t>(std::llround(
                static_cast<double>(sequence) * standardDecimation.step)));
    execute(standard, standardSurvivors);
}

TEST(CudaLasPointProgram, RejectsUnprovenAndAcceptsExactSummaryRecount)
{
    const auto bytes = syntheticCudaLas(7, 1);
    const pdg::las::FileView file(bytes);
    pdg::DimensionRegistry dimensions;
    const std::vector<std::string> mathSpecification = {
        "Classification = sqrt(Intensity)"};
    const pdg::AssignProgram mathProgram =
        pdg::compileAssignments(mathSpecification, dimensions);
    EXPECT_FALSE(pdg::las::supportsDefaultCudaPointProgram(file, mathProgram,
                                                           dimensions));

    const std::vector<std::string> returnSpecification = {
        "ReturnNumber = NumberOfReturns"};
    const pdg::AssignProgram returnProgram =
        pdg::compileAssignments(returnSpecification, dimensions);
    ASSERT_TRUE(pdg::assignSupportsExactDevice(returnProgram));
    EXPECT_TRUE(pdg::las::supportsDefaultCudaPointProgram(file, returnProgram,
                                                          dimensions));

    pdg::OrdinalProgram decimation;
    decimation.kind = pdg::OrdinalKind::Decimation;
    decimation.offset = 2;
    pdg::las::OrderedPointProgram ordinal;
    ordinal.operations.emplace_back(decimation);
    ordinal.filtersPoints = true;
    ordinal.ordinalMode = pdg::OrdinalMode::Standard;
    EXPECT_FALSE(
        pdg::las::supportsDefaultCudaPointProgram(file, ordinal, dimensions));
    ordinal.ordinalMode = pdg::OrdinalMode::Streaming;
    EXPECT_TRUE(
        pdg::las::supportsDefaultCudaPointProgram(file, ordinal, dimensions));

    pdg::OrdinalProgram tail;
    tail.kind = pdg::OrdinalKind::Tail;
    ordinal.operations = {tail};
    EXPECT_FALSE(
        pdg::las::supportsDefaultCudaPointProgram(file, ordinal, dimensions));
    ordinal.ordinalMode = pdg::OrdinalMode::Standard;
    EXPECT_FALSE(
        pdg::las::supportsDefaultCudaPointProgram(file, ordinal, dimensions));
}

TEST(CudaLasPointProgram, AutomaticSelectorStaysInsideMeasuredEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const std::vector<std::string> measuredSpecifications = {
        "Scratch = Intensity * 2 - 1",
        "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1",
        "UserData = Classification",
        "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
        "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"};
    const pdg::AssignProgram program =
        pdg::compileAssignments(measuredSpecifications, dimensions);
    EXPECT_FALSE(pdg::las::preferDefaultCudaPointProgram(15'999'999, program));
    EXPECT_TRUE(pdg::las::preferDefaultCudaPointProgram(16'000'000, program));

    pdg::AssignProgram wider = program;
    wider.writes.push_back(pdg::DimensionId(pdg::StandardDimension::GpsTime));
    EXPECT_FALSE(pdg::las::preferDefaultCudaPointProgram(16'000'000, wider));

    const std::vector<std::string> simpleSpecification = {
        "Classification = Intensity"};
    const pdg::AssignProgram simple =
        pdg::compileAssignments(simpleSpecification, dimensions);
    EXPECT_FALSE(pdg::las::preferDefaultCudaPointProgram(100'000'000, simple));
}
