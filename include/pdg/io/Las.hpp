#pragma once

#include <pdg/Coordinate.hpp>
#include <pdg/PointBatch.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pdg::las
{

class Error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct Header
{
    std::uint16_t fileSourceId = 0;
    std::uint16_t globalEncoding = 0;
    std::uint8_t versionMajor = 0;
    std::uint8_t versionMinor = 0;
    std::string systemIdentifier;
    std::string generatingSoftware;
    std::uint16_t creationDayOfYear = 0;
    std::uint16_t creationYear = 0;
    std::uint16_t headerSize = 0;
    std::uint32_t pointDataOffset = 0;
    std::uint32_t variableLengthRecordCount = 0;
    std::uint8_t pointFormat = 0;
    bool compressed = false;
    std::uint16_t pointRecordLength = 0;
    std::uint64_t pointCount = 0;
    std::array<std::uint64_t, 15> pointsByReturn{};
    std::array<double, 3> scale{};
    std::array<double, 3> offset{};
    std::array<double, 3> minimum{};
    std::array<double, 3> maximum{};
    std::uint64_t waveformDataOffset = 0;
    std::uint64_t extendedVariableLengthRecordOffset = 0;
    std::uint32_t extendedVariableLengthRecordCount = 0;

    [[nodiscard]] CoordinateEncoding coordinateEncoding() const;
};

// B0190/D0218: does this LAS point data record format carry `id` as one of
// its own fields?
//
// This is pdg's existing per-format table, previously private to `LasFerry`.
// It is the fork's own equivalent of upstream's `las::pdrfDims`, which is
// compiled into `libpdalcpp` as a hidden symbol and therefore unlinkable
// (B0190). Reusing what pdg already knows keeps the answer in one place rather
// than duplicating upstream's table into a third.
[[nodiscard]] bool formatCarriesField(std::uint8_t format,
                                      DimensionId id) noexcept;

struct VariableLengthRecord
{
    std::size_t headerOffset = 0;
    std::size_t payloadOffset = 0;
    std::uint64_t payloadLength = 0;
    std::string userId;
    std::uint16_t recordId = 0;
    std::string description;
    bool extended = false;
};

class FileView
{
public:
    explicit FileView(std::span<const std::byte> bytes);

    [[nodiscard]] const Header& header() const noexcept;
    [[nodiscard]] const std::vector<VariableLengthRecord>&
    vlrs() const noexcept;
    [[nodiscard]] const std::vector<VariableLengthRecord>&
    evlrs() const noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::span<const std::byte>
    pointRecord(std::uint64_t index) const;
    [[nodiscard]] std::uint8_t returnNumber(std::uint64_t index) const;
    [[nodiscard]] std::int32_t rawCoordinate(std::uint64_t index,
                                             std::size_t axis) const;
    [[nodiscard]] std::string describeOffset(std::size_t offset) const;

private:
    Header m_header;
    std::span<const std::byte> m_bytes;
    std::vector<VariableLengthRecord> m_vlrs;
    std::vector<VariableLengthRecord> m_evlrs;
};

[[nodiscard]] std::uint16_t minimumPointRecordLength(std::uint8_t pointFormat);
[[nodiscard]] std::string_view pointFieldAt(std::uint8_t pointFormat,
                                            std::size_t recordOffset);

void decodeCoordinates(const FileView& file, std::uint64_t firstPoint,
                       std::size_t count, PointBatch& output);
void packCoordinates(const PointBatch& input,
                     std::span<std::byte> interleavedRecords,
                     std::size_t recordStride);

} // namespace pdg::las
