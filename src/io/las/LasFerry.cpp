#include <pdg/io/LasFerry.hpp>

#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace pdg::las
{

namespace
{
constexpr std::size_t CanonicalHeaderBytes = 375;
constexpr std::size_t CanonicalPointBytes = 36;
constexpr std::size_t BatchCapacity = 1U << 20U;
constexpr double CanonicalScale = 0.01;

template <typename T> T read(const std::byte* record, std::size_t offset)
{
    T value;
    std::memcpy(&value, record + offset, sizeof(T));
    return value;
}

template <typename T> void write(std::byte* record, std::size_t offset, T value)
{
    std::memcpy(record + offset, &value, sizeof(T));
}

bool isOneOf(DimensionId id,
             std::initializer_list<StandardDimension> dimensions) noexcept
{
    return std::any_of(dimensions.begin(), dimensions.end(),
                       [&](StandardDimension candidate)
                       { return id == DimensionId(candidate); });
}

bool isCanonicalField(DimensionId id) noexcept
{
    return isOneOf(id, {StandardDimension::X,
                        StandardDimension::Y,
                        StandardDimension::Z,
                        StandardDimension::Intensity,
                        StandardDimension::ReturnNumber,
                        StandardDimension::NumberOfReturns,
                        StandardDimension::ScanDirectionFlag,
                        StandardDimension::EdgeOfFlightLine,
                        StandardDimension::Classification,
                        StandardDimension::ScanAngleRank,
                        StandardDimension::UserData,
                        StandardDimension::PointSourceId,
                        StandardDimension::Red,
                        StandardDimension::Green,
                        StandardDimension::Blue,
                        StandardDimension::GpsTime,
                        StandardDimension::ScanChannel,
                        StandardDimension::Synthetic,
                        StandardDimension::KeyPoint,
                        StandardDimension::Withheld,
                        StandardDimension::Overlap});
}

} // unnamed namespace

bool formatCarriesField(std::uint8_t format, DimensionId id) noexcept
{
    if (isOneOf(id,
                {StandardDimension::X, StandardDimension::Y,
                 StandardDimension::Z, StandardDimension::Intensity,
                 StandardDimension::ReturnNumber,
                 StandardDimension::NumberOfReturns,
                 StandardDimension::ScanDirectionFlag,
                 StandardDimension::EdgeOfFlightLine,
                 StandardDimension::Classification,
                 StandardDimension::ScanAngleRank, StandardDimension::UserData,
                 StandardDimension::PointSourceId, StandardDimension::Synthetic,
                 StandardDimension::KeyPoint, StandardDimension::Withheld,
                 StandardDimension::Overlap}))
        return true;
    if (id == DimensionId(StandardDimension::GpsTime))
        return format == 1 || format == 3 || (format >= 6 && format <= 8);
    if (isOneOf(id, {StandardDimension::Red, StandardDimension::Green,
                     StandardDimension::Blue}))
        return format == 2 || format == 3 || format == 7 || format == 8;
    if (id == DimensionId(StandardDimension::ScanChannel))
        return format >= 6 && format <= 8;
    return false;
}

namespace
{

// Retained as the file-local spelling so the many call sites below read
// unchanged; `formatCarriesField` above is the same predicate, exported.
bool inputHasField(std::uint8_t format, DimensionId id) noexcept
{
    return formatCarriesField(format, id);
}

enum class TypeBase
{
    None,
    Signed,
    Unsigned,
    Floating
};

TypeBase typeBase(DimensionType type) noexcept
{
    const std::uint16_t encoded = static_cast<std::uint16_t>(type) & 0xf00U;
    if (encoded == 0x100U)
        return TypeBase::Signed;
    if (encoded == 0x200U)
        return TypeBase::Unsigned;
    if (encoded == 0x400U)
        return TypeBase::Floating;
    return TypeBase::None;
}

DimensionType resolveLayoutType(DimensionType first,
                                DimensionType second) noexcept
{
    if (first == DimensionType::None)
        return second;
    if (second == DimensionType::None || first == second)
        return first;
    const TypeBase firstBase = typeBase(first);
    const TypeBase secondBase = typeBase(second);
    if (firstBase == secondBase)
        return static_cast<std::uint16_t>(first) >
                       static_cast<std::uint16_t>(second)
                   ? first
                   : second;
    if (firstBase == TypeBase::Floating)
        return first;
    if (secondBase == TypeBase::Floating)
        return second;
    const std::size_t firstSize = dimensionTypeSize(first);
    const std::size_t secondSize = dimensionTypeSize(second);
    if (firstBase == TypeBase::Unsigned && firstSize < secondSize)
        return second;
    if (secondBase == TypeBase::Unsigned && secondSize < firstSize)
        return first;
    switch (std::max(firstSize, secondSize))
    {
    case 1:
        return DimensionType::Signed16;
    case 2:
        return DimensionType::Signed32;
    case 4:
        return DimensionType::Signed64;
    default:
        return DimensionType::Double;
    }
}

void appendUnique(std::vector<DimensionId>& values, DimensionId value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

void decodeColumn(std::span<const std::byte> records, PointBatch& batch,
                  DimensionId id)
{
    const std::size_t count = batch.size();
    if (id == DimensionId(StandardDimension::X) ||
        id == DimensionId(StandardDimension::Y) ||
        id == DimensionId(StandardDimension::Z))
    {
        const std::size_t axis =
            id.value() - DimensionId(StandardDimension::X).value();
        std::int32_t* output = batch.data<std::int32_t>(id);
        for (std::size_t index = 0; index < count; ++index)
            output[index] =
                read<std::int32_t>(records.data() + index * CanonicalPointBytes,
                                   axis * sizeof(std::int32_t));
        return;
    }

    if (id == DimensionId(StandardDimension::Intensity) ||
        id == DimensionId(StandardDimension::PointSourceId) ||
        id == DimensionId(StandardDimension::Red) ||
        id == DimensionId(StandardDimension::Green) ||
        id == DimensionId(StandardDimension::Blue))
    {
        std::size_t offset = 12;
        if (id == DimensionId(StandardDimension::PointSourceId))
            offset = 20;
        else if (id == DimensionId(StandardDimension::Red))
            offset = 30;
        else if (id == DimensionId(StandardDimension::Green))
            offset = 32;
        else if (id == DimensionId(StandardDimension::Blue))
            offset = 34;
        std::uint16_t* output = batch.data<std::uint16_t>(id);
        for (std::size_t index = 0; index < count; ++index)
            output[index] = read<std::uint16_t>(
                records.data() + index * CanonicalPointBytes, offset);
        return;
    }

    if (id == DimensionId(StandardDimension::GpsTime))
    {
        double* output = batch.data<double>(id);
        for (std::size_t index = 0; index < count; ++index)
            output[index] =
                read<double>(records.data() + index * CanonicalPointBytes, 22);
        return;
    }

    if (id == DimensionId(StandardDimension::ScanAngleRank))
    {
        float* output = batch.data<float>(id);
        for (std::size_t index = 0; index < count; ++index)
        {
            const std::int16_t raw = read<std::int16_t>(
                records.data() + index * CanonicalPointBytes, 18);
            output[index] =
                static_cast<float>(static_cast<double>(raw) * 0.006);
        }
        return;
    }

    std::uint8_t* output = batch.data<std::uint8_t>(id);
    for (std::size_t index = 0; index < count; ++index)
    {
        const std::byte* record = records.data() + index * CanonicalPointBytes;
        const std::uint8_t returns = std::to_integer<std::uint8_t>(record[14]);
        const std::uint8_t flags = std::to_integer<std::uint8_t>(record[15]);
        if (id == DimensionId(StandardDimension::ReturnNumber))
            output[index] = returns & 0x0fU;
        else if (id == DimensionId(StandardDimension::NumberOfReturns))
            output[index] = (returns >> 4U) & 0x0fU;
        else if (id == DimensionId(StandardDimension::Synthetic))
            output[index] = flags & 0x01U;
        else if (id == DimensionId(StandardDimension::KeyPoint))
            output[index] = (flags >> 1U) & 0x01U;
        else if (id == DimensionId(StandardDimension::Withheld))
            output[index] = (flags >> 2U) & 0x01U;
        else if (id == DimensionId(StandardDimension::Overlap))
            output[index] = (flags >> 3U) & 0x01U;
        else if (id == DimensionId(StandardDimension::ScanChannel))
            output[index] = (flags >> 4U) & 0x03U;
        else if (id == DimensionId(StandardDimension::ScanDirectionFlag))
            output[index] = (flags >> 6U) & 0x01U;
        else if (id == DimensionId(StandardDimension::EdgeOfFlightLine))
            output[index] = (flags >> 7U) & 0x01U;
        else if (id == DimensionId(StandardDimension::Classification))
            output[index] = std::to_integer<std::uint8_t>(record[16]);
        else if (id == DimensionId(StandardDimension::UserData))
            output[index] = std::to_integer<std::uint8_t>(record[17]);
        else
            throw Error("unsupported canonical LAS ferry dimension");
    }
}

bool written(const std::array<bool, 133>& writes,
             StandardDimension dimension) noexcept
{
    return writes[DimensionId(dimension).value()];
}

std::uint8_t byteValue(const PointBatch& batch, DimensionId id,
                       std::size_t index)
{
    return batch.data<std::uint8_t>(id)[index];
}

void packColumns(std::span<std::byte> records, const PointBatch& batch,
                 const std::array<bool, 133>& writes)
{
    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    const DimensionId returnNumber(StandardDimension::ReturnNumber);
    const DimensionId numberOfReturns(StandardDimension::NumberOfReturns);
    const DimensionId synthetic(StandardDimension::Synthetic);
    const DimensionId keyPoint(StandardDimension::KeyPoint);
    const DimensionId withheld(StandardDimension::Withheld);
    const DimensionId overlap(StandardDimension::Overlap);
    const DimensionId scanChannel(StandardDimension::ScanChannel);
    const DimensionId scanDirection(StandardDimension::ScanDirectionFlag);
    const DimensionId edge(StandardDimension::EdgeOfFlightLine);
    const bool writesReturns =
        writes[returnNumber.value()] || writes[numberOfReturns.value()];
    const bool writesFlags =
        writes[synthetic.value()] || writes[keyPoint.value()] ||
        writes[withheld.value()] || writes[overlap.value()] ||
        writes[scanChannel.value()] || writes[scanDirection.value()] ||
        writes[edge.value()];

    for (std::size_t index = 0; index < batch.size(); ++index)
    {
        std::byte* record = records.data() + index * CanonicalPointBytes;
        if (writes[x.value()])
            write(record, 0, batch.data<std::int32_t>(x)[index]);
        if (writes[y.value()])
            write(record, 4, batch.data<std::int32_t>(y)[index]);
        if (writes[z.value()])
            write(record, 8, batch.data<std::int32_t>(z)[index]);
        if (written(writes, StandardDimension::Intensity))
            write(record, 12,
                  batch.data<std::uint16_t>(
                      DimensionId(StandardDimension::Intensity))[index]);

        if (writesReturns)
        {
            const std::uint8_t old = std::to_integer<std::uint8_t>(record[14]);
            const std::uint8_t currentReturn =
                writes[returnNumber.value()]
                    ? byteValue(batch, returnNumber, index)
                    : static_cast<std::uint8_t>(old & 0x0fU);
            const std::uint8_t currentCount =
                writes[numberOfReturns.value()]
                    ? byteValue(batch, numberOfReturns, index)
                    : static_cast<std::uint8_t>((old >> 4U) & 0x0fU);
            record[14] = static_cast<std::byte>(static_cast<std::uint8_t>(
                currentReturn | (currentCount << 4U)));
        }

        if (writesFlags)
        {
            const std::uint8_t old = std::to_integer<std::uint8_t>(record[15]);
            const auto value =
                [&](DimensionId id, unsigned shift, std::uint8_t mask)
            {
                return writes[id.value()]
                           ? static_cast<std::uint8_t>(
                                 (byteValue(batch, id, index) & mask) << shift)
                           : static_cast<std::uint8_t>(old & (mask << shift));
            };
            record[15] = static_cast<std::byte>(
                value(synthetic, 0, 0x01U) | value(keyPoint, 1, 0x01U) |
                value(withheld, 2, 0x01U) | value(overlap, 3, 0x01U) |
                value(scanChannel, 4, 0x03U) | value(scanDirection, 6, 0x01U) |
                value(edge, 7, 0x01U));
        }

        if (written(writes, StandardDimension::Classification))
            record[16] = static_cast<std::byte>(byteValue(
                batch, DimensionId(StandardDimension::Classification), index));
        if (written(writes, StandardDimension::UserData))
            record[17] = static_cast<std::byte>(byteValue(
                batch, DimensionId(StandardDimension::UserData), index));
        if (written(writes, StandardDimension::PointSourceId))
            write(record, 20,
                  batch.data<std::uint16_t>(
                      DimensionId(StandardDimension::PointSourceId))[index]);
        if (written(writes, StandardDimension::GpsTime))
            write(record, 22,
                  batch.data<double>(
                      DimensionId(StandardDimension::GpsTime))[index]);
        if (written(writes, StandardDimension::Red))
            write(record, 30,
                  batch.data<std::uint16_t>(
                      DimensionId(StandardDimension::Red))[index]);
        if (written(writes, StandardDimension::Green))
            write(record, 32,
                  batch.data<std::uint16_t>(
                      DimensionId(StandardDimension::Green))[index]);
        if (written(writes, StandardDimension::Blue))
            write(record, 34,
                  batch.data<std::uint16_t>(
                      DimensionId(StandardDimension::Blue))[index]);
    }
}

void rewriteBounds(std::span<std::byte> output, std::size_t pointCount)
{
    std::array<std::int32_t, 3> minimum{};
    std::array<std::int32_t, 3> maximum{};
    bool populated = false;
    const std::byte* records = output.data() + CanonicalHeaderBytes;
    for (std::size_t index = 0; index < pointCount; ++index)
    {
        const std::byte* record = records + index * CanonicalPointBytes;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::int32_t value =
                read<std::int32_t>(record, axis * sizeof(std::int32_t));
            if (!populated || value < minimum[axis])
                minimum[axis] = value;
            if (!populated || value > maximum[axis])
                maximum[axis] = value;
        }
        populated = true;
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double minimumValue =
            populated ? static_cast<double>(minimum[axis]) * CanonicalScale
                      : 0.0;
        const double maximumValue =
            populated ? static_cast<double>(maximum[axis]) * CanonicalScale
                      : 0.0;
        write(output.data(), 179 + axis * 16U, maximumValue);
        write(output.data(), 187 + axis * 16U, minimumValue);
    }
}
} // unnamed namespace

bool supportsDefaultPointProgram(const FileView& input,
                                 const AssignProgram& program,
                                 const DimensionRegistry& dimensions) noexcept
{
    try
    {
        if (!supportsDefaultTranslation(input))
            return false;
        const std::uint8_t format = input.header().pointFormat;
        std::vector<DimensionId> createdCustomDimensions;
        for (const PointAssignment& assignment : program.assignments)
        {
            const DimensionDefinition* destination =
                dimensions.find(assignment.destination);
            if (!destination)
                return false;
            if (destination->standard)
            {
                if (!isCanonicalField(destination->id) ||
                    !inputHasField(format, destination->id) ||
                    destination->id ==
                        DimensionId(StandardDimension::ScanAngleRank))
                    return false;
            }
            else if (assignment.destinationCreated)
                appendUnique(createdCustomDimensions, destination->id);
        }

        const auto validExpression = [&](const CompiledExpression& expression)
        {
            for (const ExpressionInstruction& instruction :
                 expression.instructions)
            {
                if (instruction.op != ExpressionOp::LoadDimension)
                    continue;
                const DimensionDefinition* source =
                    dimensions.find(instruction.dimension);
                if (!source)
                    return false;
                if (!source->standard)
                {
                    if (std::find(createdCustomDimensions.begin(),
                                  createdCustomDimensions.end(),
                                  source->id) == createdCustomDimensions.end())
                        return false;
                    continue;
                }
                if (!isCanonicalField(source->id) ||
                    !inputHasField(format, source->id))
                    return false;
                if (source->id ==
                        DimensionId(StandardDimension::ScanAngleRank) &&
                    format <= 3)
                    return false;
                if (isOneOf(source->id,
                            {StandardDimension::X, StandardDimension::Y,
                             StandardDimension::Z}))
                {
                    const std::size_t axis =
                        source->id.value() -
                        DimensionId(StandardDimension::X).value();
                    if (input.header().scale[axis] != CanonicalScale ||
                        input.header().offset[axis] != 0.0)
                        return false;
                }
            }
            return true;
        };
        for (const PointAssignment& assignment : program.assignments)
            if (!validExpression(assignment.value) ||
                !validExpression(assignment.condition))
                return false;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool supportsDefaultPointProgram(const FileView& input,
                                 const OrderedPointProgram& program,
                                 const DimensionRegistry& dimensions) noexcept
{
    try
    {
        AssignProgram assignments;
        for (const PointOperation& operation : program.operations)
            if (const auto* assignment = std::get_if<AssignProgram>(&operation))
                appendAssignments(assignments, *assignment);
        if (!supportsDefaultPointProgram(input, assignments, dimensions))
            return false;

        const std::uint8_t format = input.header().pointFormat;
        std::vector<DimensionId> createdCustomDimensions;
        for (const PointOperation& operation : program.operations)
        {
            if (const auto* assignment = std::get_if<AssignProgram>(&operation))
            {
                for (const PointAssignment& statement : assignment->assignments)
                {
                    const DimensionDefinition& destination =
                        dimensions.require(statement.destination);
                    if (!destination.standard && statement.destinationCreated)
                        appendUnique(createdCustomDimensions, destination.id);
                }
                continue;
            }

            const auto* predicate = std::get_if<PredicateProgram>(&operation);
            if (!predicate)
                continue;
            for (DimensionId id : predicate->reads)
            {
                const DimensionDefinition& source = dimensions.require(id);
                if (!source.standard)
                {
                    if (std::find(createdCustomDimensions.begin(),
                                  createdCustomDimensions.end(),
                                  id) == createdCustomDimensions.end())
                        return false;
                    continue;
                }
                if (!isCanonicalField(id) || !inputHasField(format, id))
                    return false;
                if (id == DimensionId(StandardDimension::ScanAngleRank) &&
                    format <= 3)
                    return false;
                if (isOneOf(id, {StandardDimension::X, StandardDimension::Y,
                                 StandardDimension::Z}))
                {
                    const std::size_t axis =
                        id.value() - DimensionId(StandardDimension::X).value();
                    if (input.header().scale[axis] != CanonicalScale ||
                        input.header().offset[axis] != 0.0)
                        return false;
                }
            }
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool supportsDefaultFerry(const FileView& input, const FerryProgram& program,
                          const DimensionRegistry& dimensions) noexcept
{
    if (!supportsDefaultTranslation(input))
        return false;
    const std::uint8_t format = input.header().pointFormat;
    std::array<DimensionType, 133> layoutTypes{};
    std::array<bool, 133> writtenDimensions{};
    for (std::size_t value = 1; value < layoutTypes.size(); ++value)
    {
        const DimensionId id(static_cast<std::uint32_t>(value));
        if (inputHasField(format, id))
            if (const DimensionDefinition* definition = dimensions.find(id))
                layoutTypes[value] = definition->type;
    }
    for (const FerryCopy& copy : program.copies)
    {
        const DimensionDefinition* destination =
            dimensions.find(copy.destination);
        if (!destination || !destination->standard || copy.destinationCreated ||
            !isCanonicalField(copy.destination) ||
            !inputHasField(format, copy.destination) ||
            copy.destination == DimensionId(StandardDimension::ScanAngleRank))
            return false;
        if (copy.hasSource)
        {
            const DimensionDefinition* source = dimensions.find(copy.source);
            if (!source || !source->standard ||
                !isCanonicalField(copy.source) ||
                !inputHasField(format, copy.source))
                return false;
            if (copy.source == DimensionId(StandardDimension::ScanAngleRank) &&
                format <= 3)
                return false;
            if (isOneOf(copy.source,
                        {StandardDimension::X, StandardDimension::Y,
                         StandardDimension::Z}))
            {
                if (writtenDimensions[copy.source.value()])
                    return false;
                const std::size_t axis =
                    copy.source.value() -
                    DimensionId(StandardDimension::X).value();
                if (input.header().scale[axis] != CanonicalScale ||
                    input.header().offset[axis] != 0.0)
                    return false;
            }
        }
        const DimensionType sourceType = copy.hasSource
                                             ? layoutTypes[copy.source.value()]
                                             : DimensionType::Double;
        const DimensionType destinationType =
            layoutTypes[copy.destination.value()];
        const bool coordinateDestination = isOneOf(
            copy.destination,
            {StandardDimension::X, StandardDimension::Y, StandardDimension::Z});
        const DimensionType resolvedType =
            coordinateDestination
                ? DimensionType::Double
                : resolveLayoutType(sourceType, destinationType);
        if (sourceType == DimensionType::None ||
            resolvedType != destinationType)
            return false;
        layoutTypes[copy.destination.value()] = resolvedType;
        writtenDimensions[copy.destination.value()] = true;
    }
    return true;
}

void applyDefaultPointProgram(std::span<std::byte> output,
                              const FileView& input,
                              const AssignProgram& program,
                              DimensionRegistry& dimensions,
                              std::size_t maximumHostWorkers)
{
    if (!supportsDefaultPointProgram(input, program, dimensions))
        throw Error("point program is outside the exact native LAS envelope");
    const FileView canonical(
        std::span<const std::byte>(output.data(), output.size()));
    const Header& header = canonical.header();
    if (header.versionMajor != 1 || header.versionMinor != 4 ||
        header.pointFormat != 7 || header.pointRecordLength != 36 ||
        header.pointDataOffset != CanonicalHeaderBytes || header.compressed ||
        header.scale != std::array<double, 3>{CanonicalScale, CanonicalScale,
                                              CanonicalScale} ||
        header.offset != std::array<double, 3>{0.0, 0.0, 0.0})
        throw Error(
            "point program requires canonical default LAS translation output");
    if (header.pointCount > std::numeric_limits<std::size_t>::max())
        throw Error("canonical LAS point count exceeds size_t");
    const std::size_t pointCount = static_cast<std::size_t>(header.pointCount);
    if (program.assignments.empty() || pointCount == 0)
        return;

    std::vector<DimensionId> touched = program.reads;
    for (DimensionId id : program.writes)
        appendUnique(touched, id);
    std::array<bool, 133> writes{};
    bool writesCoordinates = false;
    bool writesReturnNumber = false;
    for (DimensionId destination : program.writes)
    {
        const DimensionDefinition& definition = dimensions.require(destination);
        if (!definition.standard)
            continue;
        writes.at(destination.value()) = true;
        writesCoordinates =
            writesCoordinates ||
            isOneOf(destination, {StandardDimension::X, StandardDimension::Y,
                                  StandardDimension::Z});
        writesReturnNumber =
            writesReturnNumber ||
            destination == DimensionId(StandardDimension::ReturnNumber);
    }

    const std::size_t capacity = std::min(pointCount, BatchCapacity);
    HostMemoryResource memory;
    PointBatch batch(capacity, header.coordinateEncoding(), dimensions, memory);
    for (DimensionId id : touched)
        batch.materialize(id);

    std::array<std::uint64_t, 15> returns{};
    for (std::size_t begin = 0; begin < pointCount; begin += capacity)
    {
        const std::size_t count = std::min(capacity, pointCount - begin);
        batch.setSize(count);
        std::span<std::byte> records =
            output.subspan(CanonicalHeaderBytes + begin * CanonicalPointBytes,
                           count * CanonicalPointBytes);
        const std::span<const std::byte> constRecords(records.data(),
                                                      records.size());
        for (DimensionId id : touched)
            if (dimensions.require(id).standard)
                decodeColumn(constRecords, batch, id);
        executeAssign(batch, program, maximumHostWorkers);
        if (writesReturnNumber)
        {
            const std::uint8_t* values = batch.data<std::uint8_t>(
                DimensionId(StandardDimension::ReturnNumber));
            for (std::size_t index = 0; index < count; ++index)
                if (values[index] >= 1 && values[index] <= returns.size())
                    ++returns[values[index] - 1U];
        }
        packColumns(records, batch, writes);
    }

    if (writesCoordinates)
        rewriteBounds(output, pointCount);
    if (writesReturnNumber)
        for (std::size_t index = 0; index < returns.size(); ++index)
            write(output.data(), 255 + index * sizeof(std::uint64_t),
                  returns[index]);
}

void applyDefaultFerry(std::span<std::byte> output, const FileView& input,
                       const FerryProgram& program,
                       DimensionRegistry& dimensions,
                       std::size_t maximumHostWorkers)
{
    if (!supportsDefaultFerry(input, program, dimensions))
        throw Error("ferry pipeline is outside the exact native LAS envelope");
    AssignProgram lowered;
    appendFerry(lowered, program);
    applyDefaultPointProgram(output, input, lowered, dimensions,
                             maximumHostWorkers);
}

} // namespace pdg::las
