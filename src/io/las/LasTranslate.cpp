#include <pdg/Version.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace pdg::las
{

namespace
{
static_assert(std::endian::native == std::endian::little,
              "PDG v1 supports little-endian x86_64 hosts only");

constexpr std::size_t OutputHeaderBytes = 375;
constexpr std::size_t OutputPointBytes = 36;
constexpr std::size_t CanonicalCopyDefaultWorkers = 5;
constexpr double OutputScale = 0.01;
constexpr std::size_t VlrHeaderBytes = 54;
constexpr std::size_t ExtraBytesDescriptorBytes = 192;
constexpr std::size_t ExtraDoubleInputPointOffset =
    OutputHeaderBytes + VlrHeaderBytes + ExtraBytesDescriptorBytes;
constexpr std::size_t ExtraDoubleOutputPointOffset =
    ExtraDoubleInputPointOffset + ExtraBytesDescriptorBytes;
constexpr std::size_t ExtraDoubleInputPointBytes = 40;
constexpr std::size_t ExtraDoubleOutputPointBytes = 48;

template <typename T>
T read(std::span<const std::byte> bytes, std::size_t offset)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw Error("truncated LAS point field in native translation");
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T>
void write(std::span<std::byte> bytes, std::size_t offset, T value)
{
    if (offset > bytes.size() || bytes.size() - offset < sizeof(T))
        throw Error("native LAS translation exceeded its output buffer");
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void writeString(std::span<std::byte> bytes, std::size_t offset,
                 std::size_t length, std::string_view value)
{
    if (offset > bytes.size() || bytes.size() - offset < length)
        throw Error("native LAS translation exceeded its output header");
    const std::size_t copied = std::min(length, value.size());
    std::memcpy(bytes.data() + offset, value.data(), copied);
}

std::array<std::byte, ExtraBytesDescriptorBytes>
extraBytesDescriptor(std::uint8_t type, std::string_view name,
                     std::string_view description)
{
    if (name.empty() || name.size() > 32U || description.size() > 32U)
        throw Error("native Extra Bytes dimension strings must contain 1 to "
                    "32 bytes");
    std::array<std::byte, ExtraBytesDescriptorBytes> descriptor{};
    descriptor[2U] = static_cast<std::byte>(type);
    std::memcpy(descriptor.data() + 4U, name.data(), name.size());
    std::memcpy(descriptor.data() + 160U, description.data(),
                description.size());
    return descriptor;
}

const std::array<std::byte, ExtraBytesDescriptorBytes>& offsetTimeDescriptor()
{
    static const std::array<std::byte, ExtraBytesDescriptorBytes> descriptor =
        extraBytesDescriptor(5U, "OffsetTime",
                             "Milliseconds from first acquired");
    return descriptor;
}

const std::array<std::byte, VlrHeaderBytes>& offsetTimeVlrHeader()
{
    static const std::array<std::byte, VlrHeaderBytes> header = []
    {
        std::array<std::byte, VlrHeaderBytes> value{};
        constexpr std::string_view UserId = "LASF_Spec";
        constexpr std::string_view Description = "Extra Bytes Record";
        constexpr std::uint16_t RecordId = 4U;
        constexpr std::uint16_t PayloadLength = ExtraBytesDescriptorBytes;
        std::memcpy(value.data() + 2U, UserId.data(), UserId.size());
        std::memcpy(value.data() + 18U, &RecordId, sizeof(RecordId));
        std::memcpy(value.data() + 20U, &PayloadLength, sizeof(PayloadLength));
        std::memcpy(value.data() + 22U, Description.data(), Description.size());
        return value;
    }();
    return header;
}

bool exactExtraDoubleInputExtent(const FileView& input) noexcept
{
    const Header& header = input.header();
    if (header.pointCount > (std::numeric_limits<std::size_t>::max() -
                             ExtraDoubleInputPointOffset) /
                                ExtraDoubleInputPointBytes)
        return false;
    return input.bytes().size() ==
           ExtraDoubleInputPointOffset +
               static_cast<std::size_t>(header.pointCount) *
                   ExtraDoubleInputPointBytes;
}

void writePdalIdentity(std::span<std::byte> output,
                       const DefaultTranslationMetadata& metadata)
{
    std::fill_n(output.begin() + 26U, 64U, std::byte{});
    writeString(output, 26U, 32U, "PDAL");
    writeString(output, 58U, 32U, metadata.softwareId);
    write(output, 90U, metadata.creationDayOfYear);
    write(output, 92U, metadata.creationYear);
}

std::int32_t outputCoordinate(std::int32_t raw, double inputScale,
                              double inputOffset)
{
    const double decoded = static_cast<double>(raw) * inputScale + inputOffset;
    const double scaled = decoded / OutputScale;
    const double rounded =
        scaled > 0.0 ? std::floor(scaled + 0.5) : std::ceil(scaled - 0.5);
    if (!std::isfinite(rounded) ||
        rounded >
            static_cast<double>(std::numeric_limits<std::int32_t>::max()) ||
        rounded <
            static_cast<double>(std::numeric_limits<std::int32_t>::lowest()))
        throw Error("coordinate cannot be represented by the default LAS "
                    "writer scale and offset");
    return static_cast<std::int32_t>(rounded);
}

std::int16_t legacyScanAngle(std::int8_t value)
{
    const float rank = static_cast<float>(value);
    return static_cast<std::int16_t>(std::round(rank / 0.006F));
}

std::int16_t modernScanAngle(std::int16_t value)
{
    const float rank = static_cast<float>(static_cast<double>(value) * 0.006);
    return static_cast<std::int16_t>(std::round(rank / 0.006F));
}

struct Point
{
    std::array<std::int32_t, 3> coordinate{};
    std::uint16_t intensity = 0;
    std::uint8_t returnNumber = 0;
    std::uint8_t numberOfReturns = 0;
    std::uint8_t synthetic = 0;
    std::uint8_t keyPoint = 0;
    std::uint8_t withheld = 0;
    std::uint8_t overlap = 0;
    std::uint8_t scanChannel = 0;
    std::uint8_t scanDirection = 0;
    std::uint8_t edgeOfFlightLine = 0;
    std::uint8_t classification = 0;
    std::uint8_t userData = 0;
    std::int16_t scanAngle = 0;
    std::uint16_t pointSourceId = 0;
    std::uint64_t gpsTimeBits = 0;
    std::array<std::uint16_t, 3> color{};
};

Point readPoint(std::span<const std::byte> record, const Header& header)
{
    Point point;
    for (std::size_t axis = 0; axis < 3; ++axis)
        point.coordinate[axis] = outputCoordinate(
            read<std::int32_t>(record, axis * sizeof(std::int32_t)),
            header.scale[axis], header.offset[axis]);
    point.intensity = read<std::uint16_t>(record, 12);

    if (header.pointFormat <= 3)
    {
        const std::uint8_t returns = read<std::uint8_t>(record, 14);
        const std::uint8_t classification = read<std::uint8_t>(record, 15);
        point.returnNumber = returns & 0x07U;
        point.numberOfReturns = (returns >> 3U) & 0x07U;
        point.scanDirection = (returns >> 6U) & 0x01U;
        point.edgeOfFlightLine = (returns >> 7U) & 0x01U;
        point.classification = classification & 0x1fU;
        point.synthetic = (classification >> 5U) & 0x01U;
        point.keyPoint = (classification >> 6U) & 0x01U;
        point.withheld = (classification >> 7U) & 0x01U;
        if (point.classification == 12)
        {
            point.classification = 0;
            point.overlap = 1;
        }
        point.scanAngle = legacyScanAngle(read<std::int8_t>(record, 16));
        point.userData = read<std::uint8_t>(record, 17);
        point.pointSourceId = read<std::uint16_t>(record, 18);

        if (header.pointFormat == 1 || header.pointFormat == 3)
            point.gpsTimeBits = read<std::uint64_t>(record, 20);
        const std::size_t colorOffset = header.pointFormat == 2   ? 20
                                        : header.pointFormat == 3 ? 28
                                                                  : 0;
        if (colorOffset)
            for (std::size_t channel = 0; channel < 3; ++channel)
                point.color[channel] = read<std::uint16_t>(
                    record, colorOffset + channel * sizeof(std::uint16_t));
    }
    else
    {
        const std::uint8_t returns = read<std::uint8_t>(record, 14);
        const std::uint8_t flags = read<std::uint8_t>(record, 15);
        point.returnNumber = returns & 0x0fU;
        point.numberOfReturns = (returns >> 4U) & 0x0fU;
        point.synthetic = flags & 0x01U;
        point.keyPoint = (flags >> 1U) & 0x01U;
        point.withheld = (flags >> 2U) & 0x01U;
        point.overlap = (flags >> 3U) & 0x01U;
        point.scanChannel = (flags >> 4U) & 0x03U;
        point.scanDirection = (flags >> 6U) & 0x01U;
        point.edgeOfFlightLine = (flags >> 7U) & 0x01U;
        point.classification = read<std::uint8_t>(record, 16);
        point.userData = read<std::uint8_t>(record, 17);
        point.scanAngle = modernScanAngle(read<std::int16_t>(record, 18));
        point.pointSourceId = read<std::uint16_t>(record, 20);
        point.gpsTimeBits = read<std::uint64_t>(record, 22);
        if (header.pointFormat == 7 || header.pointFormat == 8)
            for (std::size_t channel = 0; channel < 3; ++channel)
                point.color[channel] = read<std::uint16_t>(
                    record, 30 + channel * sizeof(std::uint16_t));
    }
    return point;
}

void writePoint(std::span<std::byte> record, const Point& point)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
        write(record, axis * sizeof(std::int32_t), point.coordinate[axis]);
    write(record, 12, point.intensity);
    write<std::uint8_t>(
        record, 14,
        static_cast<std::uint8_t>(point.returnNumber |
                                  (point.numberOfReturns << 4U)));
    write<std::uint8_t>(
        record, 15,
        static_cast<std::uint8_t>(
            (point.synthetic & 0x01U) | ((point.keyPoint & 0x01U) << 1U) |
            ((point.withheld & 0x01U) << 2U) | ((point.overlap & 0x01U) << 3U) |
            ((point.scanChannel & 0x03U) << 4U) |
            ((point.scanDirection & 0x01U) << 6U) |
            ((point.edgeOfFlightLine & 0x01U) << 7U)));
    write(record, 16, point.classification);
    write(record, 17, point.userData);
    write(record, 18, point.scanAngle);
    write(record, 20, point.pointSourceId);
    write(record, 22, point.gpsTimeBits);
    for (std::size_t channel = 0; channel < 3; ++channel)
        write(record, 30 + channel * sizeof(std::uint16_t),
              point.color[channel]);
}

struct alignas(64) Summary
{
    bool populated = false;
    std::array<std::int32_t, 3> minimum{};
    std::array<std::int32_t, 3> maximum{};
    std::array<std::uint64_t, 15> returns{};

    void addRaw(const std::array<std::int32_t, 3>& coordinate,
                std::uint8_t returnNumber)
    {
        if (!populated)
        {
            minimum = coordinate;
            maximum = coordinate;
            populated = true;
        }
        else
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                minimum[axis] = std::min(minimum[axis], coordinate[axis]);
                maximum[axis] = std::max(maximum[axis], coordinate[axis]);
            }
        }
        if (returnNumber >= 1 && returnNumber <= returns.size())
            ++returns[returnNumber - 1U];
    }

    void add(const Point& point)
    {
        addRaw(point.coordinate, point.returnNumber);
    }

    void merge(const Summary& other)
    {
        if (other.populated && !populated)
        {
            minimum = other.minimum;
            maximum = other.maximum;
            populated = true;
        }
        else if (other.populated)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                minimum[axis] = std::min(minimum[axis], other.minimum[axis]);
                maximum[axis] = std::max(maximum[axis], other.maximum[axis]);
            }
        }
        for (std::size_t index = 0; index < returns.size(); ++index)
            returns[index] += other.returns[index];
    }
};

void copyExtraDoubleRange(const FileView& input,
                          std::span<const std::uint64_t> valueBits,
                          std::span<std::byte> output, std::size_t begin,
                          std::size_t end, Summary& summary)
{
    const std::byte* source = input.bytes().data() +
                              ExtraDoubleInputPointOffset +
                              begin * ExtraDoubleInputPointBytes;
    std::byte* destination = output.data() + ExtraDoubleOutputPointOffset +
                             begin * ExtraDoubleOutputPointBytes;
    for (std::size_t point = begin; point < end; ++point)
    {
        std::memcpy(destination, source, ExtraDoubleInputPointBytes);
        std::memcpy(destination + ExtraDoubleInputPointBytes, &valueBits[point],
                    sizeof(valueBits[point]));
        std::array<std::int32_t, 3> coordinate;
        for (std::size_t axis = 0; axis < coordinate.size(); ++axis)
            std::memcpy(&coordinate[axis], source + axis * sizeof(std::int32_t),
                        sizeof(std::int32_t));
        summary.addRaw(coordinate,
                       std::to_integer<std::uint8_t>(source[14U]) & 0x0fU);
        source += ExtraDoubleInputPointBytes;
        destination += ExtraDoubleOutputPointBytes;
    }
}

bool canCopyCanonicalModernRecords(const Header& header) noexcept
{
    if (header.pointFormat != 7 && header.pointFormat != 8)
        return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (header.scale[axis] != OutputScale || header.offset[axis] != 0.0)
            return false;
    return true;
}

void copyCanonicalModernRange(const FileView& input,
                              std::span<std::byte> output, std::size_t begin,
                              std::size_t end, Summary& summary)
{
    const Header& header = input.header();
    const std::size_t stride = header.pointRecordLength;
    const std::byte* source =
        input.bytes().data() + header.pointDataOffset + begin * stride;
    std::byte* destination =
        output.data() + OutputHeaderBytes + begin * OutputPointBytes;
    for (std::size_t index = begin; index < end; ++index)
    {
        std::memcpy(destination, source, OutputPointBytes);
        std::array<std::int32_t, 3> coordinate;
        for (std::size_t axis = 0; axis < 3; ++axis)
            std::memcpy(&coordinate[axis], source + axis * sizeof(std::int32_t),
                        sizeof(std::int32_t));
        summary.addRaw(coordinate,
                       std::to_integer<std::uint8_t>(source[14]) & 0x0fU);
        source += stride;
        destination += OutputPointBytes;
    }
}

void translateRange(const FileView& input, std::span<std::byte> output,
                    std::size_t begin, std::size_t end, Summary& summary)
{
    for (std::size_t index = begin; index < end; ++index)
    {
        const Point point = readPoint(input.pointRecord(index), input.header());
        writePoint(output.subspan(OutputHeaderBytes + index * OutputPointBytes,
                                  OutputPointBytes),
                   point);
        summary.add(point);
    }
}

void writeHeader(std::span<std::byte> output, std::uint64_t pointCount,
                 const Summary& summary,
                 const DefaultTranslationMetadata& metadata)
{
    writeString(output, 0, 4, "LASF");
    write<std::uint16_t>(output, 6, 1U << 4U);
    write<std::uint8_t>(output, 24, 1);
    write<std::uint8_t>(output, 25, 4);
    writeString(output, 26, 32, "PDAL");
    writeString(output, 58, 32, metadata.softwareId);
    write(output, 90, metadata.creationDayOfYear);
    write(output, 92, metadata.creationYear);
    write<std::uint16_t>(output, 94, OutputHeaderBytes);
    write<std::uint32_t>(output, 96, OutputHeaderBytes);
    write<std::uint8_t>(output, 104, 7);
    write<std::uint16_t>(output, 105, OutputPointBytes);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        write(output, 131 + axis * sizeof(double), OutputScale);
        write<double>(output, 155 + axis * sizeof(double), 0.0);
        const double minimum =
            summary.populated
                ? static_cast<double>(summary.minimum[axis]) * OutputScale
                : 0.0;
        const double maximum =
            summary.populated
                ? static_cast<double>(summary.maximum[axis]) * OutputScale
                : 0.0;
        write(output, 179 + axis * 16U, maximum);
        write(output, 187 + axis * 16U, minimum);
    }
    write(output, 247, pointCount);
    for (std::size_t index = 0; index < summary.returns.size(); ++index)
        write(output, 255 + index * sizeof(std::uint64_t),
              summary.returns[index]);
}

void writeExtraDoubleSummary(std::span<std::byte> output,
                             const Header& inputHeader,
                             std::uint64_t pointCount, const Summary& summary)
{
    write<std::uint32_t>(output, 107U, 0U);
    for (std::size_t index = 0; index < 5U; ++index)
        write<std::uint32_t>(output, 111U + index * sizeof(std::uint32_t), 0U);
    write(output, 247U, pointCount);
    for (std::size_t index = 0; index < summary.returns.size(); ++index)
        write(output, 255U + index * sizeof(std::uint64_t),
              summary.returns[index]);
    for (std::size_t axis = 0; axis < 3U; ++axis)
    {
        const double minimum =
            summary.populated ? static_cast<double>(summary.minimum[axis]) *
                                        inputHeader.scale[axis] +
                                    inputHeader.offset[axis]
                              : 0.0;
        const double maximum =
            summary.populated ? static_cast<double>(summary.maximum[axis]) *
                                        inputHeader.scale[axis] +
                                    inputHeader.offset[axis]
                              : 0.0;
        write(output, 179U + axis * 16U, maximum);
        write(output, 187U + axis * 16U, minimum);
    }
}
} // unnamed namespace

std::string oracleSoftwareId()
{
    const std::string commit(OracleCommit);
    return "PDAL " + std::string(OracleVersion) + " (" +
           commit.substr(0, std::min<std::size_t>(6, commit.size())) + ')';
}

bool supportsDefaultTranslation(const FileView& input) noexcept
{
    const Header& header = input.header();
    const bool supportedFormat =
        header.pointFormat <= 3 ||
        (header.pointFormat >= 6 && header.pointFormat <= 8);
    const bool validFormatVersion =
        header.pointFormat <= 1 ||
        (header.pointFormat <= 3 && header.versionMinor >= 2) ||
        (header.pointFormat >= 6 && header.versionMinor == 4);
    const bool modernEncodingIsValid =
        header.pointFormat < 6 || (header.globalEncoding & (1U << 4U));
    return !header.compressed && supportedFormat && validFormatVersion &&
           modernEncodingIsValid && input.vlrs().empty() &&
           input.evlrs().empty();
}

std::vector<std::byte>
translateDefault(const FileView& input,
                 const DefaultTranslationMetadata& metadata,
                 std::size_t maximumWorkers)
{
    std::vector<std::byte> output(defaultTranslationSize(input));
    translateDefaultInto(input, metadata, output, maximumWorkers);
    return output;
}

bool supportsExtraDoubleTranslation(const FileView& input) noexcept
{
    const Header& header = input.header();
    if (header.versionMajor != 1U || header.versionMinor != 4U ||
        header.headerSize != OutputHeaderBytes ||
        header.pointDataOffset != ExtraDoubleInputPointOffset ||
        header.variableLengthRecordCount != 1U || header.pointFormat != 7U ||
        header.compressed ||
        header.pointRecordLength != ExtraDoubleInputPointBytes ||
        header.globalEncoding != (1U << 4U) || header.waveformDataOffset ||
        header.extendedVariableLengthRecordOffset ||
        header.extendedVariableLengthRecordCount || input.vlrs().size() != 1U ||
        !input.evlrs().empty() || !exactExtraDoubleInputExtent(input))
        return false;
    const VariableLengthRecord& vlr = input.vlrs().front();
    if (vlr.headerOffset != OutputHeaderBytes ||
        vlr.payloadOffset != OutputHeaderBytes + VlrHeaderBytes ||
        vlr.payloadLength != ExtraBytesDescriptorBytes ||
        vlr.userId != "LASF_Spec" || vlr.recordId != 4U ||
        vlr.description != "Extra Bytes Record")
        return false;
    const std::span<const std::byte> rawHeader =
        input.bytes().subspan(vlr.headerOffset, VlrHeaderBytes);
    const auto& expectedHeader = offsetTimeVlrHeader();
    if (!std::equal(rawHeader.begin(), rawHeader.end(), expectedHeader.begin()))
        return false;
    const std::span<const std::byte> payload =
        input.bytes().subspan(vlr.payloadOffset, ExtraBytesDescriptorBytes);
    const auto& descriptor = offsetTimeDescriptor();
    return std::equal(payload.begin(), payload.end(), descriptor.begin());
}

std::size_t extraDoubleTranslationSize(const FileView& input)
{
    if (!supportsExtraDoubleTranslation(input))
        throw Error("LAS input is outside the exact one-binary64 Extra Bytes "
                    "translation envelope");
    const std::uint64_t count = input.header().pointCount;
    if (count > (std::numeric_limits<std::size_t>::max() -
                 ExtraDoubleOutputPointOffset) /
                    ExtraDoubleOutputPointBytes)
        throw Error("native Extra Bytes translation output size overflows "
                    "size_t");
    return ExtraDoubleOutputPointOffset +
           static_cast<std::size_t>(count) * ExtraDoubleOutputPointBytes;
}

void translateExtraDoubleInto(const FileView& input,
                              const DefaultTranslationMetadata& metadata,
                              ExtraDoubleDimension dimension,
                              std::span<const std::uint64_t> valueBits,
                              std::span<std::byte> output,
                              std::size_t maximumWorkers)
{
    const std::size_t outputBytes = extraDoubleTranslationSize(input);
    if (valueBits.size() != input.header().pointCount)
        throw Error("native Extra Bytes column cardinality does not match "
                    "the LAS input");
    if (output.size() != outputBytes)
        throw Error("native Extra Bytes translation output span has the "
                    "wrong size");
    const auto descriptor =
        extraBytesDescriptor(10U, dimension.name, dimension.description);

    std::copy_n(input.bytes().begin(), ExtraDoubleInputPointOffset,
                output.begin());
    writePdalIdentity(output, metadata);
    write<std::uint32_t>(output, 96U, ExtraDoubleOutputPointOffset);
    write<std::uint16_t>(output, 105U, ExtraDoubleOutputPointBytes);
    write<std::uint16_t>(output, OutputHeaderBytes + 20U,
                         2U * ExtraBytesDescriptorBytes);
    std::copy(descriptor.begin(), descriptor.end(),
              output.begin() + ExtraDoubleInputPointOffset);

    const std::size_t pointCount = valueBits.size();
    constexpr std::size_t MinimumPointsPerWorker = 65536U;
    const std::size_t usefulWorkers =
        pointCount / MinimumPointsPerWorker +
        static_cast<std::size_t>(pointCount % MinimumPointsPerWorker != 0U);
    const std::size_t hardwareWorkers =
        std::max<std::size_t>(1U, std::thread::hardware_concurrency());
    const std::size_t availableWorkers =
        maximumWorkers ? maximumWorkers
                       : std::min(hardwareWorkers, CanonicalCopyDefaultWorkers);
    const std::size_t workerCount =
        std::max<std::size_t>(1U, std::min(usefulWorkers, availableWorkers));
    std::vector<Summary> summaries(workerCount);
    if (workerCount == 1U)
        copyExtraDoubleRange(input, valueBits, output, 0U, pointCount,
                             summaries.front());
    else
    {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        const std::size_t pointsPerWorker = pointCount / workerCount;
        const std::size_t remainder = pointCount % workerCount;
        for (std::size_t worker = 0; worker < workerCount; ++worker)
        {
            const std::size_t begin =
                pointsPerWorker * worker + std::min(worker, remainder);
            const std::size_t end =
                begin + pointsPerWorker +
                static_cast<std::size_t>(worker < remainder);
            workers.emplace_back(copyExtraDoubleRange, std::cref(input),
                                 valueBits, output, begin, end,
                                 std::ref(summaries[worker]));
        }
        for (std::thread& worker : workers)
            worker.join();
    }
    Summary summary;
    for (const Summary& workerSummary : summaries)
        summary.merge(workerSummary);
    writeExtraDoubleSummary(output, input.header(), input.header().pointCount,
                            summary);
}

std::vector<std::byte> translateExtraDouble(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    ExtraDoubleDimension dimension, std::span<const std::uint64_t> valueBits,
    std::size_t maximumWorkers)
{
    std::vector<std::byte> output(extraDoubleTranslationSize(input));
    translateExtraDoubleInto(input, metadata, dimension, valueBits, output,
                             maximumWorkers);
    return output;
}

std::size_t defaultTranslationSize(const FileView& input)
{
    if (!supportsDefaultTranslation(input))
        throw Error("LAS input is outside the exact native translation "
                    "compatibility envelope");
    const std::uint64_t count = input.header().pointCount;
    if (count > (std::numeric_limits<std::size_t>::max() - OutputHeaderBytes) /
                    OutputPointBytes)
        throw Error("native LAS translation output size overflows size_t");
    const std::size_t outputBytes =
        OutputHeaderBytes + static_cast<std::size_t>(count) * OutputPointBytes;
    return outputBytes;
}

void translateDefaultInto(const FileView& input,
                          const DefaultTranslationMetadata& metadata,
                          std::span<std::byte> output,
                          std::size_t maximumWorkers)
{
    const std::size_t outputBytes = defaultTranslationSize(input);
    if (output.size() != outputBytes)
        throw Error("native LAS translation output span has the wrong size");
    std::fill_n(output.begin(), OutputHeaderBytes, std::byte{});
    const std::uint64_t count = input.header().pointCount;
    const std::size_t pointCount = static_cast<std::size_t>(count);
    constexpr std::size_t MinimumPointsPerWorker = 65536;
    const std::size_t usefulWorkers =
        pointCount / MinimumPointsPerWorker +
        static_cast<std::size_t>(pointCount % MinimumPointsPerWorker != 0);
    const bool copyCanonicalModern =
        canCopyCanonicalModernRecords(input.header());
    const std::size_t hardwareWorkers =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t defaultWorkers =
        copyCanonicalModern
            ? std::min(hardwareWorkers, CanonicalCopyDefaultWorkers)
            : hardwareWorkers;
    const std::size_t availableWorkers =
        maximumWorkers ? maximumWorkers : defaultWorkers;
    const std::size_t workerCount =
        std::max<std::size_t>(1, std::min(usefulWorkers, availableWorkers));
    std::vector<Summary> workerSummaries(workerCount);
    std::vector<std::exception_ptr> workerErrors(workerCount);
    const auto translate =
        [&](std::size_t begin, std::size_t end, Summary& summary)
    {
        if (copyCanonicalModern)
            copyCanonicalModernRange(input, output, begin, end, summary);
        else
            translateRange(input, output, begin, end, summary);
    };

    if (workerCount == 1)
        translate(0, pointCount, workerSummaries[0]);
    else
    {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        const std::size_t pointsPerWorker = pointCount / workerCount;
        const std::size_t remainder = pointCount % workerCount;
        for (std::size_t worker = 0; worker < workerCount; ++worker)
        {
            const std::size_t begin =
                pointsPerWorker * worker + std::min(worker, remainder);
            const std::size_t end =
                begin + pointsPerWorker +
                static_cast<std::size_t>(worker < remainder);
            workers.emplace_back(
                [&, worker, begin, end]
                {
                    try
                    {
                        translate(begin, end, workerSummaries[worker]);
                    }
                    catch (...)
                    {
                        workerErrors[worker] = std::current_exception();
                    }
                });
        }
        for (std::thread& worker : workers)
            worker.join();
        for (const std::exception_ptr& error : workerErrors)
            if (error)
                std::rethrow_exception(error);
    }

    Summary summary;
    for (const Summary& workerSummary : workerSummaries)
        summary.merge(workerSummary);
    writeHeader(output, count, summary, metadata);
}

void overlayDefaultUserData(std::span<std::byte> canonicalOutput,
                            std::span<const std::uint8_t> values)
{
    const FileView output(canonicalOutput);
    const Header& header = output.header();
    if (header.versionMajor != 1U || header.versionMinor != 4U ||
        header.pointDataOffset != OutputHeaderBytes ||
        header.pointFormat != 7U || header.compressed ||
        header.pointRecordLength != OutputPointBytes ||
        !output.vlrs().empty() || !output.evlrs().empty() ||
        header.pointCount != static_cast<std::uint64_t>(values.size()) ||
        canonicalOutput.size() < OutputHeaderBytes ||
        (canonicalOutput.size() - OutputHeaderBytes) % OutputPointBytes != 0U ||
        (canonicalOutput.size() - OutputHeaderBytes) / OutputPointBytes !=
            values.size())
        throw Error("UserData overlay requires a matching canonical default "
                    "LAS image");
    constexpr std::size_t UserDataOffset = 17U;
    for (std::size_t point = 0; point < values.size(); ++point)
        canonicalOutput[OutputHeaderBytes + point * OutputPointBytes +
                        UserDataOffset] = static_cast<std::byte>(values[point]);
}

void overlayDefaultClassification(std::span<std::byte> canonicalOutput,
                                  std::span<const std::uint8_t> values)
{
    const FileView output(canonicalOutput);
    const Header& header = output.header();
    if (header.versionMajor != 1U || header.versionMinor != 4U ||
        header.pointDataOffset != OutputHeaderBytes ||
        header.pointFormat != 7U || header.compressed ||
        header.pointRecordLength != OutputPointBytes ||
        !output.vlrs().empty() || !output.evlrs().empty() ||
        header.pointCount != static_cast<std::uint64_t>(values.size()) ||
        canonicalOutput.size() < OutputHeaderBytes ||
        (canonicalOutput.size() - OutputHeaderBytes) % OutputPointBytes != 0U ||
        (canonicalOutput.size() - OutputHeaderBytes) / OutputPointBytes !=
            values.size())
        throw Error("Classification overlay requires a matching canonical "
                    "default LAS image");
    constexpr std::size_t ClassificationOffset = 16U;
    for (std::size_t point = 0; point < values.size(); ++point)
        canonicalOutput[OutputHeaderBytes + point * OutputPointBytes +
                        ClassificationOffset] =
            static_cast<std::byte>(values[point]);
}

void translateDefaultPermutedInto(const FileView& input,
                                  const DefaultTranslationMetadata& metadata,
                                  std::span<const std::uint64_t> sourceOrder,
                                  std::span<std::byte> output,
                                  std::size_t maximumWorkers)
{
    const std::size_t outputBytes = defaultTranslationSize(input);
    const std::size_t pointCount =
        static_cast<std::size_t>(input.header().pointCount);
    if (sourceOrder.size() != pointCount)
        throw Error("permuted LAS publication order does not match the "
                    "input cardinality");
    if (output.size() != outputBytes)
        throw Error("permuted LAS publication output span has the wrong "
                    "size");

    std::vector<std::uint8_t> seen(pointCount, 0U);
    for (const std::uint64_t source : sourceOrder)
    {
        if (source >= input.header().pointCount ||
            seen[static_cast<std::size_t>(source)] != 0U)
            throw Error("permuted LAS publication requires a complete "
                        "source permutation");
        seen[static_cast<std::size_t>(source)] = 1U;
    }

    std::vector<std::byte> canonical(outputBytes);
    translateDefaultInto(input, metadata, canonical, maximumWorkers);
    std::copy_n(canonical.begin(), OutputHeaderBytes, output.begin());
    for (std::size_t destination = 0U; destination < pointCount; ++destination)
    {
        const std::size_t source =
            static_cast<std::size_t>(sourceOrder[destination]);
        const std::size_t destinationOffset =
            OutputHeaderBytes + destination * OutputPointBytes;
        const std::size_t sourceOffset =
            OutputHeaderBytes + source * OutputPointBytes;
        std::copy_n(canonical.begin() + sourceOffset, OutputPointBytes,
                    output.begin() + destinationOffset);
    }
}

void translateDefaultPermutedClassificationInto(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    std::span<const std::uint64_t> sourceOrder,
    std::span<const std::uint8_t> classification, std::span<std::byte> output,
    std::size_t maximumWorkers)
{
    if (classification.size() != sourceOrder.size())
        throw Error("permuted LAS publication columns do not match the "
                    "input cardinality");
    translateDefaultPermutedInto(input, metadata, sourceOrder, output,
                                 maximumWorkers);
    overlayDefaultClassification(output, classification);
}

} // namespace pdg::las
