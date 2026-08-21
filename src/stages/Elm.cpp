/******************************************************************************
 * ELM compatibility implementation derived from PDAL's BSD-licensed
 * filters.elm implementation.
 * Copyright (c) 2017 Bradley J Chambers
 * Copyright (c) 2026 PDAL-GPU contributors
 ******************************************************************************/

#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>

namespace pdg
{
namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId Classification(StandardDimension::Classification);

struct GridFrame
{
    double minimumX = 0.0;
    double minimumY = 0.0;
    std::size_t rows = 0U;
    std::size_t columns = 0U;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return rows * columns;
    }
};

bool programValid(const ElmProgram& program) noexcept
{
    return std::isfinite(program.cell) && program.cell > 0.0 &&
           std::isfinite(program.threshold);
}

GridFrame frameFor(const PointBatch& batch, const ElmProgram& program)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    double minimumX = x[0];
    double maximumX = x[0];
    double minimumY = y[0];
    double maximumY = y[0];
    for (std::size_t point = 1U; point < batch.size(); ++point)
    {
        minimumX = (std::min)(minimumX, x[point]);
        maximumX = (std::max)(maximumX, x[point]);
        minimumY = (std::min)(minimumY, y[point]);
        maximumY = (std::max)(maximumY, y[point]);
    }
    const double columnsValue = (maximumX - minimumX) / program.cell + 1.0;
    const double rowsValue = (maximumY - minimumY) / program.cell + 1.0;
    if (!std::isfinite(columnsValue) || !std::isfinite(rowsValue) ||
        columnsValue < 1.0 || rowsValue < 1.0 ||
        columnsValue >
            static_cast<double>((std::numeric_limits<std::size_t>::max)()) ||
        rowsValue >
            static_cast<double>((std::numeric_limits<std::size_t>::max)()))
        throw std::invalid_argument("ELM grid dimensions are invalid");
    GridFrame frame{minimumX, minimumY, static_cast<std::size_t>(rowsValue),
                    static_cast<std::size_t>(columnsValue)};
    if (frame.columns &&
        frame.rows > (std::numeric_limits<std::size_t>::max)() / frame.columns)
        throw std::overflow_error("ELM grid cell count overflows");
    return frame;
}

std::size_t cellFor(double x, double y, const GridFrame& frame, double cell)
{
    // Preserve upstream ELM's parenthesis placement exactly.
    const double columnValue = std::floor(x - frame.minimumX) / cell;
    const double rowValue = std::floor(y - frame.minimumY) / cell;
    if (!std::isfinite(columnValue) || !std::isfinite(rowValue) ||
        columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        throw std::out_of_range("ELM point lies outside its grid frame");
    return static_cast<std::size_t>(columnValue) * frame.rows +
           static_cast<std::size_t>(rowValue);
}

ElmResult classifyHost(PointBatch& batch, const ElmProgram& program)
{
    const GridFrame frame = frameFor(batch, program);
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    auto* classification = batch.data<std::uint8_t>(Classification);

    std::map<std::uint32_t, std::multimap<double, std::size_t>> buckets;
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t cell =
            cellFor(x[point], y[point], frame, program.cell);
        if (cell > (std::numeric_limits<std::uint32_t>::max)())
            throw std::overflow_error("ELM grid key exceeds uint32");
        buckets[static_cast<std::uint32_t>(cell)].emplace(z[point], point);
    }

    ElmResult result{frame.rows, frame.columns, 0U};
    for (std::size_t column = 0U; column < frame.columns; ++column)
        for (std::size_t row = 0U; row < frame.rows; ++row)
        {
            const auto position = buckets.find(
                static_cast<std::uint32_t>(column * frame.rows + row));
            if (position == buckets.end() || position->second.size() <= 1U)
                continue;
            const auto& points = position->second;
            for (auto point = points.begin(); point != std::prev(points.end());
                 ++point)
            {
                if (std::fabs(point->first - std::next(point)->first) <
                    program.threshold)
                    break;
                classification[point->second] = program.classification;
                ++result.classifiedPoints;
            }
        }
    return result;
}
} // unnamed namespace

ElmResult classifyElmDevice(PointBatch& batch, const ElmProgram& program);

bool elmProgramWithinExactDeviceEnvelope(const ElmProgram& program) noexcept
{
    return programValid(program);
}

bool elmSupportsExactDevice(const PointBatch& hostBatch,
                            const ElmProgram& program) noexcept
{
    if (!programValid(program) ||
        (hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !hostBatch.size() || !hostBatch.has(X) || !hostBatch.has(Y) ||
        !hostBatch.has(Z) || !hostBatch.has(Classification) ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;
    try
    {
        if (hostBatch.columnInfo(X).physicalType != DimensionType::Double ||
            hostBatch.columnInfo(Y).physicalType != DimensionType::Double ||
            hostBatch.columnInfo(Z).physicalType != DimensionType::Double ||
            hostBatch.columnInfo(Classification).physicalType !=
                DimensionType::Unsigned8)
            return false;
        const double* x = hostBatch.data<double>(X);
        const double* y = hostBatch.data<double>(Y);
        const double* z = hostBatch.data<double>(Z);
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
            if (!std::isfinite(x[point]) || !std::isfinite(y[point]) ||
                !std::isfinite(z[point]))
                return false;
        const GridFrame frame = frameFor(hostBatch, program);
        if (frame.size() > ElmExactDeviceMaximumGridCells ||
            frame.size() > static_cast<std::size_t>(
                               (std::numeric_limits<std::uint32_t>::max)()))
            return false;
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
            static_cast<void>(cellFor(x[point], y[point], frame, program.cell));
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

ElmResult classifyElm(PointBatch& batch, const ElmProgram& program)
{
    if (!programValid(program))
        throw std::invalid_argument("invalid ELM program");
    if (!batch.size())
        return {};
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        !batch.has(Classification))
        throw std::invalid_argument("ELM input columns are not materialized");
    if (batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double ||
        batch.columnInfo(Classification).physicalType !=
            DimensionType::Unsigned8)
        throw std::invalid_argument("ELM input columns have invalid types");
    if (batch.memoryKind() == MemoryKind::Device)
        return classifyElmDevice(batch, program);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported ELM memory kind");
    return classifyHost(batch, program);
}

} // namespace pdg
