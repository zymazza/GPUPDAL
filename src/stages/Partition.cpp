#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr DimensionId ReturnNumber(StandardDimension::ReturnNumber);
constexpr DimensionId NumberOfReturns(StandardDimension::NumberOfReturns);
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);

bool validColumns(const PointBatch& batch) noexcept
{
    try
    {
        return batch.has(ReturnNumber) && batch.has(NumberOfReturns) &&
               batch.columnInfo(ReturnNumber).physicalType ==
                   DimensionType::Unsigned8 &&
               batch.columnInfo(NumberOfReturns).physicalType ==
                   DimensionType::Unsigned8;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void validateProgram(const ReturnsProgram& program)
{
    if ((program.groups & static_cast<std::uint8_t>(~AllReturnGroups)) != 0U)
        throw std::invalid_argument("returns program contains invalid groups");
}

bool validDividerProgram(const DividerProgram& program) noexcept
{
    return (program.mode == DividerMode::Partition ||
            program.mode == DividerMode::RoundRobin) &&
           program.count >= 2U && program.count <= 1000U;
}

bool validSplitterColumns(const PointBatch& batch) noexcept
{
    try
    {
        return batch.has(X) && batch.has(Y) &&
               batch.columnInfo(X).physicalType == DimensionType::Double &&
               batch.columnInfo(Y).physicalType == DimensionType::Double;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool validSplitterProgram(const SplitterProgram& program) noexcept
{
    return std::isfinite(program.length) && program.length > 0.0 &&
           std::isfinite(program.originX) && std::isfinite(program.originY) &&
           std::isfinite(program.buffer) &&
           program.buffer < program.length / 2.0;
}

bool splitterCell(double coordinate, double origin, double length,
                  std::int32_t& cell) noexcept
{
    const double delta = coordinate - origin;
    const double quotient = delta / length;
    if (!std::isfinite(coordinate) || !std::isfinite(delta) ||
        !std::isfinite(quotient) ||
        quotient > static_cast<double>((std::numeric_limits<int>::max)()) ||
        quotient <= static_cast<double>((std::numeric_limits<int>::min)()))
        return false;
    int result = static_cast<int>(quotient);
    if (delta < 0.0)
        --result;
    cell = static_cast<std::int32_t>(result);
    return true;
}
} // unnamed namespace

ReturnsPartitionResult partitionReturnsDevice(PointBatch& batch,
                                              const ReturnsProgram& program,
                                              std::uint64_t* permutation);
DividerPartitionResult partitionDividerDevice(PointBatch& batch,
                                              const DividerProgram& program,
                                              std::uint64_t* permutation);
void computeSplitterCellsDevice(PointBatch& batch,
                                const SplitterProgram& program,
                                std::int32_t* xCells, std::int32_t* yCells);

std::uint64_t ReturnsPartitionResult::selectedCount() const noexcept
{
    std::uint64_t count = 0;
    for (std::uint64_t group : counts)
        count += group;
    return count;
}

std::uint64_t DividerPartitionResult::selectedCount() const noexcept
{
    return std::accumulate(counts.begin(), counts.end(), std::uint64_t{0});
}

std::uint8_t returnGroupIndex(std::uint8_t returnNumber,
                              std::uint8_t numberOfReturns,
                              std::uint8_t groups) noexcept
{
    if ((groups & ReturnFirst) && returnNumber == 1U && numberOfReturns > 1U)
        return 0U;
    if ((groups & ReturnIntermediate) && returnNumber > 1U &&
        returnNumber < numberOfReturns && numberOfReturns > 2U)
        return 1U;
    if ((groups & ReturnLast) && returnNumber == numberOfReturns &&
        numberOfReturns > 1U)
        return 2U;
    if ((groups & ReturnOnly) && numberOfReturns == 1U)
        return 3U;
    return UnselectedReturnGroup;
}

bool returnsMaySupportExactDevice(const PointBatch& hostBatch,
                                  const ReturnsProgram& program) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        (program.groups & static_cast<std::uint8_t>(~AllReturnGroups)) != 0U ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;
    return validColumns(hostBatch);
}

ReturnsPartitionResult partitionReturns(PointBatch& batch,
                                        const ReturnsProgram& program,
                                        std::uint64_t* permutation)
{
    validateProgram(program);
    if (!validColumns(batch))
        throw std::invalid_argument(
            "returns columns must be materialized as Unsigned8");
    if (batch.size() && !permutation)
        throw std::invalid_argument("returns permutation is null");
    if (!batch.size())
        return {};
    if (batch.memoryKind() == MemoryKind::Device)
        return partitionReturnsDevice(batch, program, permutation);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported returns memory kind");

    const auto* returnNumbers = batch.data<std::uint8_t>(ReturnNumber);
    const auto* numbersOfReturns = batch.data<std::uint8_t>(NumberOfReturns);
    ReturnsPartitionResult result;
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::uint8_t group = returnGroupIndex(
            returnNumbers[point], numbersOfReturns[point], program.groups);
        if (group < result.counts.size())
            ++result.counts[group];
    }

    std::array<std::uint64_t, 4> cursors{};
    for (std::size_t group = 1; group < cursors.size(); ++group)
        cursors[group] = cursors[group - 1U] + result.counts[group - 1U];
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::uint8_t group = returnGroupIndex(
            returnNumbers[point], numbersOfReturns[point], program.groups);
        if (group < cursors.size())
            permutation[cursors[group]++] = static_cast<std::uint64_t>(point);
    }
    return result;
}

bool dividerMaySupportExactDevice(const PointBatch& hostBatch,
                                  const DividerProgram& program) noexcept
{
    return (hostBatch.memoryKind() == MemoryKind::Host ||
            hostBatch.memoryKind() == MemoryKind::PinnedHost) &&
           validDividerProgram(program) &&
           hostBatch.size() <=
               static_cast<std::size_t>((std::numeric_limits<int>::max)());
}

DividerPartitionResult partitionDivider(PointBatch& batch,
                                        const DividerProgram& program,
                                        std::uint64_t* permutation)
{
    if (!validDividerProgram(program))
        throw std::invalid_argument("divider count must be between 2 and 1000");
    if (batch.size() && !permutation)
        throw std::invalid_argument("divider permutation is null");
    if (batch.memoryKind() == MemoryKind::Device)
        return partitionDividerDevice(batch, program, permutation);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported divider memory kind");

    DividerPartitionResult result;
    result.counts.assign(program.count, 0U);
    if (!batch.size())
        return result;

    if (program.mode == DividerMode::Partition)
    {
        const std::size_t count = static_cast<std::size_t>(program.count);
        const std::size_t limit = (batch.size() - 1U) / count + 1U;
        std::size_t remaining = batch.size();
        for (std::size_t view = 0; view < count; ++view)
        {
            const std::size_t points = (std::min)(limit, remaining);
            result.counts[view] = static_cast<std::uint64_t>(points);
            remaining -= points;
        }
        std::iota(permutation, permutation + batch.size(), std::uint64_t{0});
        return result;
    }

    for (std::size_t view = 0; view < result.counts.size(); ++view)
        result.counts[view] = static_cast<std::uint64_t>(
            batch.size() / result.counts.size() +
            (view < batch.size() % result.counts.size()));
    std::size_t output = 0;
    for (std::size_t view = 0; view < result.counts.size(); ++view)
        for (std::size_t point = view; point < batch.size();
             point += result.counts.size())
            permutation[output++] = static_cast<std::uint64_t>(point);
    return result;
}

bool splitterCellsMaySupportExactDevice(const PointBatch& hostBatch,
                                        const SplitterProgram& program) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !validSplitterColumns(hostBatch) || !validSplitterProgram(program) ||
        program.buffer > 0.0 ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;

    const auto* x = hostBatch.data<double>(X);
    const auto* y = hostBatch.data<double>(Y);
    for (std::size_t point = 0; point < hostBatch.size(); ++point)
    {
        std::int32_t xCell = 0;
        std::int32_t yCell = 0;
        if (!splitterCell(x[point], program.originX, program.length, xCell) ||
            !splitterCell(y[point], program.originY, program.length, yCell))
            return false;
    }
    return true;
}

void computeSplitterCells(PointBatch& batch, const SplitterProgram& program,
                          std::int32_t* xCells, std::int32_t* yCells)
{
    if (!validSplitterProgram(program))
        throw std::invalid_argument("invalid splitter program");
    if (!validSplitterColumns(batch))
        throw std::invalid_argument(
            "splitter columns must be materialized as Double");
    if (batch.size() && (!xCells || !yCells))
        throw std::invalid_argument("splitter cell output is null");
    if (!batch.size())
        return;
    if (batch.memoryKind() == MemoryKind::Device)
    {
        computeSplitterCellsDevice(batch, program, xCells, yCells);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported splitter memory kind");

    const auto* x = batch.data<double>(X);
    const auto* y = batch.data<double>(Y);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        if (!splitterCell(x[point], program.originX, program.length,
                          xCells[point]) ||
            !splitterCell(y[point], program.originY, program.length,
                          yCells[point]))
            throw std::invalid_argument(
                "splitter coordinate exceeds cell range");
    }
}

} // namespace pdg
