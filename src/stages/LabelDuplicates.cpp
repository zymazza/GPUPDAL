#include <pdg/PointBatch.hpp>
#include <pdg/stages/LabelDuplicates.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pdg
{
namespace
{
int coordinateAxis(DimensionId id) noexcept
{
    if (id == DimensionId(StandardDimension::X))
        return 0;
    if (id == DimensionId(StandardDimension::Y))
        return 1;
    if (id == DimensionId(StandardDimension::Z))
        return 2;
    return -1;
}

template <typename T>
double valueAsDouble(const void* data, std::size_t point) noexcept
{
    return static_cast<double>(static_cast<const T*>(data)[point]);
}

double valueAsDouble(const PointBatch& batch, DimensionId dimension,
                     std::size_t point)
{
    const ColumnInfo& column = batch.columnInfo(dimension);
    const void* data = batch.rawData(dimension);
    const int axis = coordinateAxis(dimension);
    if (axis >= 0 && column.physicalType == DimensionType::Signed32)
        return batch.coordinateEncoding().decode(
            static_cast<std::size_t>(axis),
            static_cast<const std::int32_t*>(data)[point]);
    switch (column.physicalType)
    {
    case DimensionType::Signed8:
        return valueAsDouble<std::int8_t>(data, point);
    case DimensionType::Signed16:
        return valueAsDouble<std::int16_t>(data, point);
    case DimensionType::Signed32:
        return valueAsDouble<std::int32_t>(data, point);
    case DimensionType::Signed64:
        return valueAsDouble<std::int64_t>(data, point);
    case DimensionType::Unsigned8:
        return valueAsDouble<std::uint8_t>(data, point);
    case DimensionType::Unsigned16:
        return valueAsDouble<std::uint16_t>(data, point);
    case DimensionType::Unsigned32:
        return valueAsDouble<std::uint32_t>(data, point);
    case DimensionType::Unsigned64:
        return valueAsDouble<std::uint64_t>(data, point);
    case DimensionType::Float:
        return valueAsDouble<float>(data, point);
    case DimensionType::Double:
        return valueAsDouble<double>(data, point);
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument("label_duplicates input has no physical type");
}

bool supportedDimension(const PointBatch& batch, DimensionId dimension)
{
    return batch.has(dimension) &&
           batch.columnInfo(dimension).physicalType != DimensionType::None;
}
} // unnamed namespace

void labelDuplicatesDevice(PointBatch& batch,
                           const LabelDuplicatesProgram& program,
                           std::uint8_t* output);

bool labelDuplicatesMaySupportExactDevice(
    const PointBatch& hostBatch, const LabelDuplicatesProgram& program) noexcept
{
    if (hostBatch.memoryKind() != MemoryKind::Host &&
        hostBatch.memoryKind() != MemoryKind::PinnedHost)
        return false;
    try
    {
        for (DimensionId dimension : program.dimensions)
        {
            // Upstream writes Duplicate in place after every adjacent
            // comparison. Selecting that output as an input therefore makes
            // the next row depend on the preceding write and is not a
            // parallel adjacent-comparison operation.
            if (dimension == DimensionId(StandardDimension::Duplicate))
                return false;
            if (!supportedDimension(hostBatch, dimension))
                return false;
        }
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

void labelDuplicates(PointBatch& batch, const LabelDuplicatesProgram& program,
                     std::uint8_t* output)
{
    for (DimensionId dimension : program.dimensions)
        if (!supportedDimension(batch, dimension))
            throw std::invalid_argument(
                "label_duplicates input column is not materialized");
    if (batch.size() && !output)
        throw std::invalid_argument("label_duplicates output is null");
    if (batch.size() < 2U)
        return;
    if (batch.memoryKind() == MemoryKind::Device)
    {
        labelDuplicatesDevice(batch, program, output);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported label_duplicates memory kind");

    for (std::size_t point = 1U; point < batch.size(); ++point)
    {
        bool duplicate = true;
        for (DimensionId dimension : program.dimensions)
            if (valueAsDouble(batch, dimension, point) !=
                valueAsDouble(batch, dimension, point - 1U))
            {
                duplicate = false;
                break;
            }
        output[point] = static_cast<std::uint8_t>(duplicate);
    }
}

} // namespace pdg
