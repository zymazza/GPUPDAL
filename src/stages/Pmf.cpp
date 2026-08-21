/******************************************************************************
 * PMF compatibility implementation derived from PDAL's BSD-licensed
 * filters.pmf and MathUtils morphology code.
 * Copyright (c) 2015-2017, 2020 Bradley J Chambers
 * Copyright (c) 2026 PDAL-GPU contributors
 ******************************************************************************/

#include <pdg/PointBatch.hpp>
#include <pdg/index/RasterGrid.hpp>
#include <pdg/stages/Pmf.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace pdg
{
namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr DimensionId Classification(StandardDimension::Classification);

using RasterFrame = PmfRasterFrame;

constexpr std::size_t SizeBits = sizeof(std::size_t) * 8U;
// The literal scan remains faster through 255 populated cells on the measured
// sparse matrices.  The hierarchy starts at 256, where it first breaks even,
// while this fixed cap keeps the scan's work linear in raster cells.
constexpr std::size_t BoundedSourceScanMaximum = 255U;

struct OccupancyLevel
{
    std::size_t rows = 0U;
    std::size_t columns = 0U;
    std::size_t offset = 0U;
    std::size_t block = 1U;
};

struct OccupancyNode
{
    std::size_t row = 0U;
    std::size_t column = 0U;
    std::size_t level = 0U;
};

struct PmfPass
{
    int radius = 0;
    double threshold = 0.0;
};

bool programValid(const PmfProgram& program) noexcept
{
    return std::isfinite(program.cellSize) && program.cellSize > 0.0 &&
           std::isfinite(program.initialDistance) &&
           std::isfinite(program.maxDistance) &&
           std::isfinite(program.maxWindowSize) &&
           program.maxWindowSize >= 0.0 && std::isfinite(program.slope) &&
           (program.onlyGround || program.groundClass != program.otherClass);
}

bool makeSchedule(const PmfProgram& program, std::vector<PmfPass>& passes,
                  bool bounded) noexcept
{
    if (!programValid(program))
        return false;
    double previousWindow = 0.0;
    for (std::size_t iteration = 0U; previousWindow < program.maxWindowSize;
         ++iteration)
    {
        if (bounded && iteration >= PmfExactDeviceMaximumPasses)
            return false;
        if (iteration >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            return false;
        double window = 0.0;
        if (program.exponential)
            window = program.cellSize *
                     (2.0 * std::pow(2.0, static_cast<int>(iteration)) + 1.0);
        else
            window = program.cellSize *
                     (2.0 * static_cast<double>((iteration + 1U) * 2U) + 1.0);
        if (!std::isfinite(window) || window <= previousWindow)
            return false;

        double threshold = program.initialDistance;
        if (iteration != 0U)
            threshold =
                program.slope * (window - previousWindow) * program.cellSize +
                program.initialDistance;
        if (threshold > program.maxDistance)
            threshold = program.maxDistance;
        if (!std::isfinite(threshold))
            return false;

        const double radiusValue = 0.5 * (window - 1.0);
        if (!std::isfinite(radiusValue) ||
            radiusValue >
                static_cast<double>((std::numeric_limits<int>::max)()) ||
            radiusValue <
                static_cast<double>((std::numeric_limits<int>::min)()))
            return false;
        const int radius = static_cast<int>(radiusValue);
        if (bounded && radius > PmfExactDeviceMaximumMorphologyRadius)
            return false;
        try
        {
            passes.push_back({radius, threshold});
        }
        catch (const std::exception&)
        {
            return false;
        }
        previousWindow = window;
    }
    return true;
}

RasterFrame frameFor(const PointBatch& batch, const PmfProgram& program)
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
    const double columns = (maximumX - minimumX) / program.cellSize + 1.0;
    const double rows = (maximumY - minimumY) / program.cellSize + 1.0;
    if (!std::isfinite(columns) || !std::isfinite(rows) || columns < 1.0 ||
        rows < 1.0 ||
        columns > static_cast<double>((std::numeric_limits<int>::max)()) ||
        rows > static_cast<double>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("PMF raster dimensions are invalid");
    RasterFrame frame{minimumX, minimumY, static_cast<std::size_t>(rows),
                      static_cast<std::size_t>(columns)};
    if (frame.columns != 0U &&
        frame.rows > (std::numeric_limits<std::size_t>::max)() / frame.columns)
        throw std::overflow_error("PMF raster cell count overflows");
    return frame;
}

std::size_t initialRasterCell(double x, double y, const RasterFrame& frame,
                              double cellSize)
{
    // This deliberately preserves upstream PMF's parenthesis placement.
    const double columnValue = std::floor(x - frame.minimumX) / cellSize;
    const double rowValue = std::floor(y - frame.minimumY) / cellSize;
    if (!std::isfinite(columnValue) || !std::isfinite(rowValue) ||
        columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        throw std::out_of_range("PMF point lies outside its initial raster");
    return static_cast<std::size_t>(columnValue) * frame.rows +
           static_cast<std::size_t>(rowValue);
}

std::size_t lookupRasterCell(double x, double y, const RasterFrame& frame,
                             double cellSize)
{
    const double columnValue = std::floor((x - frame.minimumX) / cellSize);
    const double rowValue = std::floor((y - frame.minimumY) / cellSize);
    if (!std::isfinite(columnValue) || !std::isfinite(rowValue) ||
        columnValue < 0.0 || rowValue < 0.0 ||
        columnValue >= static_cast<double>(frame.columns) ||
        rowValue >= static_cast<double>(frame.rows))
        throw std::out_of_range("PMF point lies outside its lookup raster");
    return static_cast<std::size_t>(columnValue) * frame.rows +
           static_cast<std::size_t>(rowValue);
}

std::size_t checkedOccupancyAdd(std::size_t left, std::size_t right)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
        throw std::overflow_error("PMF occupancy hierarchy size overflows");
    return left + right;
}

std::size_t checkedOccupancyCells(std::size_t rows, std::size_t columns)
{
    if (columns != 0U &&
        rows > (std::numeric_limits<std::size_t>::max)() / columns)
        throw std::overflow_error("PMF occupancy hierarchy size overflows");
    return rows * columns;
}

double rasterCenter(double minimum, std::size_t cell, double cellSize)
{
    return minimum + (static_cast<double>(cell) + 0.5) * cellSize;
}

bool rasterCentersFinite(const RasterFrame& frame, double cellSize)
{
    return frame.rows != 0U && frame.columns != 0U &&
           std::isfinite(rasterCenter(frame.minimumX, 0U, cellSize)) &&
           std::isfinite(
               rasterCenter(frame.minimumX, frame.columns - 1U, cellSize)) &&
           std::isfinite(rasterCenter(frame.minimumY, 0U, cellSize)) &&
           std::isfinite(
               rasterCenter(frame.minimumY, frame.rows - 1U, cellSize));
}

std::size_t occupancyIndex(const OccupancyLevel& level, std::size_t row,
                           std::size_t column)
{
    return level.offset + column * level.rows + row;
}

std::size_t nodeFirst(std::size_t coordinate, std::size_t block,
                      std::size_t limit)
{
    if (coordinate > (limit - 1U) / block)
        return limit - 1U;
    return coordinate * block;
}

std::size_t nodeLast(std::size_t first, std::size_t block, std::size_t limit)
{
    const std::size_t remaining = limit - 1U - first;
    return first + (std::min)(remaining, block - 1U);
}

double nodeDistanceSquared(const OccupancyNode& node,
                           const OccupancyLevel& level,
                           const RasterFrame& frame, double cellSize,
                           std::size_t targetRow, std::size_t targetColumn,
                           double targetX, double targetY)
{
    const std::size_t firstRow = nodeFirst(node.row, level.block, frame.rows);
    const std::size_t lastRow = nodeLast(firstRow, level.block, frame.rows);
    const std::size_t firstColumn =
        nodeFirst(node.column, level.block, frame.columns);
    const std::size_t lastColumn =
        nodeLast(firstColumn, level.block, frame.columns);

    double deltaX = 0.0;
    if (targetColumn < firstColumn)
        deltaX = targetX - rasterCenter(frame.minimumX, firstColumn, cellSize);
    else if (targetColumn > lastColumn)
        deltaX = targetX - rasterCenter(frame.minimumX, lastColumn, cellSize);
    double deltaY = 0.0;
    if (targetRow < firstRow)
        deltaY = targetY - rasterCenter(frame.minimumY, firstRow, cellSize);
    else if (targetRow > lastRow)
        deltaY = targetY - rasterCenter(frame.minimumY, lastRow, cellSize);
    return deltaX * deltaX + deltaY * deltaY;
}

// Builds a byte occupancy pyramid in caller-owned scratch and fills voids in
// place.  Every hierarchy bound uses the same represented cell centers and
// binary64 subtract/multiply/add order as a leaf distance.  Center coordinates
// are monotone for a finite positive cell size, so the nearest endpoint on
// each axis is a conservative literal lower bound.  Pruning is strictly `>`:
// all leaves that can equal the winning distance are therefore visited and
// distinct-elevation ties remain observable before publication.
void fillNearestExact(std::span<double> values, const RasterFrame& frame,
                      double cellSize, std::span<std::uint8_t> scratch,
                      PmfRasterBuildFacts* facts)
{
    std::array<OccupancyLevel, SizeBits + 1U> levels{};
    std::size_t levelCount = 1U;
    levels[0] = {frame.rows, frame.columns, 0U, 1U};
    std::size_t required = frame.size();
    while (levels[levelCount - 1U].rows != 1U ||
           levels[levelCount - 1U].columns != 1U)
    {
        if (levelCount == levels.size())
            throw std::overflow_error("PMF occupancy hierarchy is too deep");
        const OccupancyLevel& previous = levels[levelCount - 1U];
        OccupancyLevel& level = levels[levelCount];
        level.rows = previous.rows / 2U + previous.rows % 2U;
        level.columns = previous.columns / 2U + previous.columns % 2U;
        level.offset = required;
        level.block =
            previous.block > (std::numeric_limits<std::size_t>::max)() / 2U
                ? (std::numeric_limits<std::size_t>::max)()
                : previous.block * 2U;
        required = checkedOccupancyAdd(
            required, checkedOccupancyCells(level.rows, level.columns));
        ++levelCount;
    }
    if (scratch.size() < required)
        throw std::invalid_argument(
            "PMF raster product lacks occupancy hierarchy scratch");

    std::size_t populatedCells = 0U;
    for (std::size_t cell = 0U; cell < values.size(); ++cell)
    {
        scratch[cell] = static_cast<std::uint8_t>(!std::isnan(values[cell]));
        populatedCells += static_cast<std::size_t>(scratch[cell] != 0U);
    }
    if (facts)
        facts->populatedCells = populatedCells;
    if (populatedCells == 0U || populatedCells == values.size())
        return;

    // A small fixed source set is already linear in the raster size and has a
    // much lower constant than hierarchy traversal. Keep that exact literal
    // scan bounded. Source counts above the cap take the hierarchy, which
    // improves the measured sparse layouts without asserting a worst-case
    // subquadratic bound for branch-and-bound traversal.
    if (populatedCells <= BoundedSourceScanMaximum)
    {
        if (facts)
            facts->usedBoundedSourceScan = true;
        std::array<std::size_t, BoundedSourceScanMaximum> sources{};
        std::size_t sourceCount = 0U;
        for (std::size_t cell = 0U; cell < values.size(); ++cell)
            if (scratch[cell] != 0U)
                sources[sourceCount++] = cell;
        for (std::size_t cell = 0U; cell < values.size(); ++cell)
        {
            if (scratch[cell] != 0U)
                continue;
            const std::size_t targetColumn = cell / frame.rows;
            const std::size_t targetRow = cell % frame.rows;
            const double targetX =
                rasterCenter(frame.minimumX, targetColumn, cellSize);
            const double targetY =
                rasterCenter(frame.minimumY, targetRow, cellSize);
            bool found = false;
            bool ambiguous = false;
            double bestDistance = (std::numeric_limits<double>::max)();
            std::size_t bestCell = 0U;
            for (std::size_t sourceIndex = 0U; sourceIndex < sourceCount;
                 ++sourceIndex)
            {
                if (facts && facts->sourceSlotsVisited !=
                                 (std::numeric_limits<std::size_t>::max)())
                    ++facts->sourceSlotsVisited;
                const std::size_t source = sources[sourceIndex];
                const std::size_t sourceColumn = source / frame.rows;
                const std::size_t sourceRow = source % frame.rows;
                const double deltaX =
                    targetX -
                    rasterCenter(frame.minimumX, sourceColumn, cellSize);
                const double deltaY =
                    targetY - rasterCenter(frame.minimumY, sourceRow, cellSize);
                const double distance = deltaX * deltaX + deltaY * deltaY;
                if (!found || distance < bestDistance)
                {
                    found = true;
                    ambiguous = false;
                    bestDistance = distance;
                    bestCell = source;
                }
                else if (distance == bestDistance && source != bestCell &&
                         std::bit_cast<std::uint64_t>(values[source]) !=
                             std::bit_cast<std::uint64_t>(values[bestCell]))
                    ambiguous = true;
            }
            if (!found)
                throw std::logic_error("PMF occupancy hierarchy is empty");
            if (ambiguous)
                throw std::invalid_argument(
                    "PMF nearest-fill tie has distinct source elevations");
            values[cell] = values[bestCell];
        }
        return;
    }
    if (facts)
        facts->usedOccupancyHierarchy = true;

    for (std::size_t levelIndex = 1U; levelIndex < levelCount; ++levelIndex)
    {
        const OccupancyLevel& previous = levels[levelIndex - 1U];
        const OccupancyLevel& level = levels[levelIndex];
        for (std::size_t column = 0U; column < level.columns; ++column)
            for (std::size_t row = 0U; row < level.rows; ++row)
            {
                bool occupied = false;
                for (std::size_t childColumn = column * 2U;
                     childColumn <
                     (std::min)(previous.columns, column * 2U + 2U);
                     ++childColumn)
                    for (std::size_t childRow = row * 2U;
                         childRow < (std::min)(previous.rows, row * 2U + 2U);
                         ++childRow)
                        occupied = occupied ||
                                   scratch[occupancyIndex(previous, childRow,
                                                          childColumn)] != 0U;
                scratch[occupancyIndex(level, row, column)] =
                    static_cast<std::uint8_t>(occupied);
            }
    }

    const auto occupied = [&](const OccupancyNode& node)
    {
        return scratch[occupancyIndex(levels[node.level], node.row,
                                      node.column)] != 0U;
    };
    const auto children =
        [&](const OccupancyNode& node, std::array<OccupancyNode, 4U>& result)
    {
        std::size_t count = 0U;
        const OccupancyLevel& childLevel = levels[node.level - 1U];
        for (std::size_t column = node.column * 2U;
             column < (std::min)(childLevel.columns, node.column * 2U + 2U);
             ++column)
            for (std::size_t row = node.row * 2U;
                 row < (std::min)(childLevel.rows, node.row * 2U + 2U); ++row)
            {
                const OccupancyNode child{row, column, node.level - 1U};
                if (occupied(child))
                    result[count++] = child;
            }
        return count;
    };

    constexpr std::size_t MaximumStack = 3U * SizeBits + 4U;
    std::array<OccupancyNode, MaximumStack> stack{};
    const OccupancyNode root{0U, 0U, levelCount - 1U};
    for (std::size_t cell = 0U; cell < values.size(); ++cell)
    {
        if (scratch[cell] != 0U)
            continue;
        const std::size_t targetColumn = cell / frame.rows;
        const std::size_t targetRow = cell % frame.rows;
        const double targetX =
            rasterCenter(frame.minimumX, targetColumn, cellSize);
        const double targetY =
            rasterCenter(frame.minimumY, targetRow, cellSize);

        // Greedy descent provides a nearby finite incumbent before the proof
        // traversal, improving pruning without changing the accepted value.
        OccupancyNode bestNode = root;
        while (bestNode.level != 0U)
        {
            std::array<OccupancyNode, 4U> candidates{};
            const std::size_t count = children(bestNode, candidates);
            if (count == 0U)
                throw std::logic_error("PMF occupancy hierarchy is empty");
            std::size_t bestChild = 0U;
            double bestBound = nodeDistanceSquared(
                candidates[0], levels[candidates[0].level], frame, cellSize,
                targetRow, targetColumn, targetX, targetY);
            for (std::size_t candidate = 1U; candidate < count; ++candidate)
            {
                const double bound = nodeDistanceSquared(
                    candidates[candidate], levels[candidates[candidate].level],
                    frame, cellSize, targetRow, targetColumn, targetX, targetY);
                if (bound < bestBound)
                {
                    bestBound = bound;
                    bestChild = candidate;
                }
            }
            bestNode = candidates[bestChild];
        }
        std::size_t bestCell = bestNode.column * frame.rows + bestNode.row;
        double bestDistance =
            nodeDistanceSquared(bestNode, levels[0], frame, cellSize, targetRow,
                                targetColumn, targetX, targetY);
        bool ambiguous = false;

        std::size_t stackSize = 1U;
        stack[0] = root;
        while (stackSize != 0U)
        {
            const OccupancyNode node = stack[--stackSize];
            if (facts && facts->hierarchyNodesVisited !=
                             (std::numeric_limits<std::size_t>::max)())
                ++facts->hierarchyNodesVisited;
            const double bound =
                nodeDistanceSquared(node, levels[node.level], frame, cellSize,
                                    targetRow, targetColumn, targetX, targetY);
            if (bound > bestDistance)
                continue;
            if (node.level != 0U)
            {
                std::array<OccupancyNode, 4U> candidates{};
                const std::size_t count = children(node, candidates);
                if (stackSize + count > stack.size())
                    throw std::logic_error(
                        "PMF occupancy hierarchy traversal overflow");
                for (std::size_t candidate = count; candidate != 0U;
                     --candidate)
                    stack[stackSize++] = candidates[candidate - 1U];
                continue;
            }
            if (facts && facts->sourceSlotsVisited !=
                             (std::numeric_limits<std::size_t>::max)())
                ++facts->sourceSlotsVisited;
            const std::size_t source = node.column * frame.rows + node.row;
            if (bound < bestDistance)
            {
                bestDistance = bound;
                bestCell = source;
                ambiguous = false;
            }
            else if (bound == bestDistance && source != bestCell &&
                     std::bit_cast<std::uint64_t>(values[source]) !=
                         std::bit_cast<std::uint64_t>(values[bestCell]))
                ambiguous = true;
        }
        if (ambiguous)
            throw std::invalid_argument(
                "PMF nearest-fill tie has distinct source elevations");
        values[cell] = values[bestCell];
    }
}

void fillNearest(std::vector<double>& values, const RasterFrame& frame,
                 double cellSize)
{
    if (values.size() >
        (std::numeric_limits<std::size_t>::max)() / std::size_t{2U})
        throw std::overflow_error("PMF occupancy hierarchy size overflows");
    std::vector<std::uint8_t> hierarchy(values.size() * 2U);
    fillNearestExact(std::span<double>(values), frame, cellSize,
                     std::span<std::uint8_t>(hierarchy), nullptr);
}

void dilateDiamond(std::vector<double>& data, std::size_t rows,
                   std::size_t columns, int iterations)
{
    std::vector<double> output(data.size(),
                               std::numeric_limits<double>::lowest());
    std::array<std::size_t, 5> neighbors{};
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        for (std::size_t column = 0U; column < columns; ++column)
        {
            const std::size_t offset = column * rows;
            for (std::size_t row = 0U; row < rows; ++row)
            {
                std::size_t count = 0U;
                neighbors[count++] = offset + row;
                if (row > 0U)
                    neighbors[count++] = offset + row - 1U;
                if (row + 1U < rows)
                    neighbors[count++] = offset + row + 1U;
                if (column > 0U)
                    neighbors[count++] = offset + row - rows;
                if (column + 1U < columns)
                    neighbors[count++] = offset + row + rows;
                for (std::size_t index = 0U; index < count; ++index)
                    if (data[neighbors[index]] > output[offset + row])
                        output[offset + row] = data[neighbors[index]];
            }
        }
        data.swap(output);
    }
}

void erodeDiamond(std::vector<double>& data, std::size_t rows,
                  std::size_t columns, int iterations)
{
    std::vector<double> output(data.size(),
                               (std::numeric_limits<double>::max)());
    std::array<std::size_t, 5> neighbors{};
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        for (std::size_t column = 0U; column < columns; ++column)
        {
            const std::size_t offset = column * rows;
            for (std::size_t row = 0U; row < rows; ++row)
            {
                std::size_t count = 0U;
                neighbors[count++] = offset + row;
                if (row > 0U)
                    neighbors[count++] = offset + row - 1U;
                if (row + 1U < rows)
                    neighbors[count++] = offset + row + 1U;
                if (column > 0U)
                    neighbors[count++] = offset + row - rows;
                if (column + 1U < columns)
                    neighbors[count++] = offset + row + rows;
                for (std::size_t index = 0U; index < count; ++index)
                    if (data[neighbors[index]] < output[offset + row])
                        output[offset + row] = data[neighbors[index]];
            }
        }
        data.swap(output);
    }
}

PmfResult classifyHost(PointBatch& batch, const PmfProgram& program)
{
    const RasterFrame frame = frameFor(batch, program);
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    auto* classification = batch.data<std::uint8_t>(Classification);

    std::vector<double> minimum(frame.size(),
                                std::numeric_limits<double>::quiet_NaN());
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t cell =
            initialRasterCell(x[point], y[point], frame, program.cellSize);
        if (std::isnan(minimum[cell]) || z[point] < minimum[cell])
            minimum[cell] = z[point];
    }
    fillNearest(minimum, frame, program.cellSize);

    std::vector<PmfPass> passes;
    if (!makeSchedule(program, passes, false))
        throw std::invalid_argument("invalid PMF window schedule");
    std::vector<std::uint8_t> ground(batch.size(), std::uint8_t{1U});
    if (!program.onlyGround)
        std::fill(classification, classification + batch.size(),
                  program.otherClass);
    for (const PmfPass& pass : passes)
    {
        erodeDiamond(minimum, frame.rows, frame.columns, pass.radius);
        dilateDiamond(minimum, frame.rows, frame.columns, pass.radius);
        for (std::size_t point = 0U; point < batch.size(); ++point)
        {
            if (!ground[point])
                continue;
            const std::size_t cell =
                lookupRasterCell(x[point], y[point], frame, program.cellSize);
            if (!((z[point] - minimum[cell]) < pass.threshold))
                ground[point] = 0U;
        }
    }

    PmfResult result{frame.rows, frame.columns, 0U, 0U};
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        if (ground[point])
        {
            classification[point] = program.groundClass;
            ++result.groundPoints;
        }
        else
            ++result.nongroundPoints;
    }
    return result;
}
} // unnamed namespace

PmfResult classifyPmfDevice(PointBatch& batch, const PmfProgram& program);

bool pmfProgramWithinExactDeviceEnvelope(const PmfProgram& program) noexcept
{
    std::vector<PmfPass> passes;
    return makeSchedule(program, passes, true);
}

bool pmfSupportsExactDevice(const PointBatch& hostBatch,
                            const PmfProgram& program) noexcept
{
    if (!pmfProgramWithinExactDeviceEnvelope(program) ||
        (hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !hostBatch.size() || !hostBatch.has(X) || !hostBatch.has(Y) ||
        !hostBatch.has(Z) || !hostBatch.has(Classification))
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
        const RasterFrame frame = frameFor(hostBatch, program);
        if (!rasterCentersFinite(frame, program.cellSize) ||
            frame.size() > PmfExactDeviceMaximumRasterCells ||
            hostBatch.size() >
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            return false;
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
        {
            static_cast<void>(
                initialRasterCell(x[point], y[point], frame, program.cellSize));
            static_cast<void>(
                lookupRasterCell(x[point], y[point], frame, program.cellSize));
        }
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool pmfSupportsExactTiledDevice(const PointBatch& hostBatch,
                                 const PmfProgram& program) noexcept
{
    if (!pmfProgramWithinExactDeviceEnvelope(program) ||
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
        const RasterFrame frame = frameFor(hostBatch, program);
        if (!rasterCentersFinite(frame, program.cellSize))
            return false;
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
        {
            static_cast<void>(
                initialRasterCell(x[point], y[point], frame, program.cellSize));
            static_cast<void>(
                lookupRasterCell(x[point], y[point], frame, program.cellSize));
        }
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

PmfRasterFrame pmfRasterFrame(const PointBatch& hostBatch,
                              const PmfProgram& program)
{
    if (!pmfSupportsExactTiledDevice(hostBatch, program))
        throw std::invalid_argument(
            "PMF input is outside the exact tiled device envelope");
    return frameFor(hostBatch, program);
}

void buildPmfTiledRaster(const PointBatch& hostBatch, const PmfProgram& program,
                         RasterGridProduct& product, PmfRasterBuildFacts* facts)
{
    if (facts)
        *facts = {};
    if (!pmfSupportsExactTiledDevice(hostBatch, program))
        throw std::invalid_argument(
            "PMF input is outside the exact tiled device envelope");
    if (&hostBatch.memoryResource() != &product.stagingMemory())
        throw std::invalid_argument(
            "PMF raster build requires its planner-owned staging resource");
    if (product.hasPendingRasterBuild())
        throw std::logic_error(
            "PMF raster product already has a pending build");

    const RasterFrame frame = frameFor(hostBatch, program);
    const RasterGridFrame& productFrame = product.frame();
    if (productFrame.policy != RasterGridFramePolicy::PmfV1 ||
        productFrame.minimumX != frame.minimumX ||
        productFrame.minimumY != frame.minimumY ||
        productFrame.rows != frame.rows ||
        productFrame.columns != frame.columns ||
        productFrame.cellSize != program.cellSize ||
        product.backingBytes() % sizeof(double) != 0U ||
        product.backingBytes() / sizeof(double) != frame.size())
        throw std::invalid_argument(
            "PMF raster product differs from its exact source frame");

    product.prepareHostRasterBuild();
    auto sparse = std::span<double>(
        static_cast<double*>(product.currentBacking()), frame.size());
    std::fill(sparse.begin(), sparse.end(),
              std::numeric_limits<double>::quiet_NaN());

    const double* x = hostBatch.data<double>(X);
    const double* y = hostBatch.data<double>(Y);
    const double* z = hostBatch.data<double>(Z);
    for (std::size_t point = 0U; point < hostBatch.size(); ++point)
    {
        const std::size_t cell =
            initialRasterCell(x[point], y[point], frame, program.cellSize);
        static_cast<void>(
            lookupRasterCell(x[point], y[point], frame, program.cellSize));
        if (std::isnan(sparse[cell]) || z[point] < sparse[cell])
            sparse[cell] = z[point];
    }

    fillNearestExact(sparse, frame, program.cellSize,
                     std::span<std::uint8_t>(
                         static_cast<std::uint8_t*>(product.nextBacking()),
                         product.backingBytes()),
                     facts);
    product.publishRasterBuild();
}

PmfResult classifyPmf(PointBatch& batch, const PmfProgram& program)
{
    if (!programValid(program))
        throw std::invalid_argument("invalid PMF program");
    if (!batch.size())
        throw std::invalid_argument("PMF requires a nonempty point batch");
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        !batch.has(Classification))
        throw std::invalid_argument("PMF input columns are not materialized");
    if (batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double ||
        batch.columnInfo(Classification).physicalType !=
            DimensionType::Unsigned8)
        throw std::invalid_argument("PMF input columns have invalid types");
    if (batch.memoryKind() == MemoryKind::Device)
        return classifyPmfDevice(batch, program);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported PMF memory kind");
    return classifyHost(batch, program);
}

} // namespace pdg
