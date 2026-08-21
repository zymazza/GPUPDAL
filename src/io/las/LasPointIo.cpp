#include <pdg/io/Las.hpp>

#include <cstring>
#include <limits>

namespace pdg::las
{

namespace
{
std::size_t checkedRecordBytes(std::size_t count, std::size_t stride)
{
    if (stride < 12)
        throw Error("LAS record stride is too small for XYZ");
    if (count > std::numeric_limits<std::size_t>::max() / stride)
        throw Error("LAS record buffer size overflows the host address space");
    return count * stride;
}
} // unnamed namespace

void decodeCoordinates(const FileView& file, std::uint64_t firstPoint,
                       std::size_t count, PointBatch& output)
{
    if (file.header().compressed)
        throw Error("compressed LAZ coordinates require the chunk decoder");
    if (count > output.capacity())
        throw Error(
            "output PointBatch capacity is smaller than the LAS decode");
    if (firstPoint > file.header().pointCount ||
        count > file.header().pointCount - firstPoint)
        throw std::out_of_range("LAS decode range exceeds the point count");
    if (output.coordinateEncoding().scale() != file.header().scale ||
        output.coordinateEncoding().offset() != file.header().offset)
        throw Error(
            "PointBatch coordinate encoding does not match the LAS header");

    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    output.materialize(x);
    output.materialize(y);
    output.materialize(z);
    std::int32_t* xValues = output.data<std::int32_t>(x);
    std::int32_t* yValues = output.data<std::int32_t>(y);
    std::int32_t* zValues = output.data<std::int32_t>(z);

    for (std::size_t index = 0; index < count; ++index)
    {
        const auto record = file.pointRecord(firstPoint + index);
        std::memcpy(xValues + index, record.data(), sizeof(std::int32_t));
        std::memcpy(yValues + index, record.data() + 4, sizeof(std::int32_t));
        std::memcpy(zValues + index, record.data() + 8, sizeof(std::int32_t));
    }
    output.setSize(count);
}

void packCoordinates(const PointBatch& input,
                     std::span<std::byte> interleavedRecords,
                     std::size_t recordStride)
{
    const std::size_t required = checkedRecordBytes(input.size(), recordStride);
    if (interleavedRecords.size() < required)
        throw Error("interleaved LAS output buffer is too small");

    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    const std::int32_t* xValues = input.data<std::int32_t>(x);
    const std::int32_t* yValues = input.data<std::int32_t>(y);
    const std::int32_t* zValues = input.data<std::int32_t>(z);
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        std::byte* record = interleavedRecords.data() + index * recordStride;
        std::memcpy(record, xValues + index, sizeof(std::int32_t));
        std::memcpy(record + 4, yValues + index, sizeof(std::int32_t));
        std::memcpy(record + 8, zValues + index, sizeof(std::int32_t));
    }
}

} // namespace pdg::las
