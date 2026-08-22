#include <pdg/FastMode.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include "PinnedCovariance.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace pdg
{

namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr std::uint32_t MaximumCell = (1U << 21U) - 1U;
constexpr std::uint32_t MaximumKnnNeighbors = 64U;
constexpr std::uint32_t MaximumKnnShell = 4096U;
// The RTX 4090 G1 trials show that switching merely because a sampled cell is
// denser than k is counterproductive: the grid still wins until a broadly
// clustered profile reaches roughly two thousand candidates per hot cell.
// Keep this threshold conservative and benchmark any change (D0034/B0006).
constexpr std::size_t BvhEstimatedCellPopulation = 2048U;
using BvhBounds = detail::MortonBvhBounds;

std::uint64_t deterministicProbe(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

bool validDimensions(std::uint8_t dimensions) noexcept
{
    return dimensions == 2 || dimensions == 3;
}

bool validColumns(const PointBatch& batch, std::uint8_t dimensions) noexcept
{
    try
    {
        if (!batch.has(X) || !batch.has(Y) ||
            batch.columnInfo(X).physicalType != DimensionType::Double ||
            batch.columnInfo(Y).physicalType != DimensionType::Double)
            return false;
        return dimensions == 2 ||
               (batch.has(Z) &&
                batch.columnInfo(Z).physicalType == DimensionType::Double);
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::uint32_t cellCoordinate(double value, double origin, double cellSize)
{
    const double quotient = (value - origin) / cellSize;
    if (!std::isfinite(quotient) || quotient < 0.0 ||
        quotient > static_cast<double>(MaximumCell))
        throw std::out_of_range(
            "coordinate is outside the uniform-grid 21-bit cell frame");
    const double position = std::floor(quotient);
    if (position > static_cast<double>(MaximumCell))
        throw std::out_of_range(
            "coordinate is outside the uniform-grid 21-bit cell frame");
    return static_cast<std::uint32_t>(position);
}

std::uint64_t cellKey(std::uint32_t x, std::uint32_t y,
                      std::uint32_t z) noexcept
{
    std::uint64_t key = 0;
    for (unsigned int bit = 0; bit < 21U; ++bit)
    {
        key |= static_cast<std::uint64_t>((x >> bit) & 1U) << (3U * bit);
        key |= static_cast<std::uint64_t>((y >> bit) & 1U) << (3U * bit + 1U);
        key |= static_cast<std::uint64_t>((z >> bit) & 1U) << (3U * bit + 2U);
    }
    return key;
}

std::array<std::uint32_t, 3> pointCell(const PointBatch& batch,
                                       const UniformGridConfig& config,
                                       std::size_t point)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = config.dimensions == 3 ? batch.data<double>(Z) : nullptr;
    return {cellCoordinate(x[point], config.origin[0], config.cellSize),
            cellCoordinate(y[point], config.origin[1], config.cellSize),
            config.dimensions == 3
                ? cellCoordinate(z[point], config.origin[2], config.cellSize)
                : 0U};
}

void validateConfigShape(const UniformGridConfig& config)
{
    if (!validDimensions(config.dimensions))
        throw std::invalid_argument(
            "uniform spatial index dimensions must be 2 or 3");
    if (!std::isfinite(config.cellSize) || config.cellSize <= 0.0)
        throw std::invalid_argument(
            "uniform spatial index cell size must be finite and positive");
    if (config.backend != SpatialIndexBackend::UniformGrid &&
        config.backend != SpatialIndexBackend::MortonBvh)
        throw std::invalid_argument("invalid spatial index backend");
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        if (!std::isfinite(config.origin[axis]) ||
            config.maximumCell[axis] > MaximumCell)
            throw std::invalid_argument("invalid uniform spatial index frame");
    }
}

void validateRadius(const SpatialIndex& index, double radius)
{
    if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument("radius must be finite and positive");
    if (index.config().backend == SpatialIndexBackend::UniformGrid &&
        radius > index.config().cellSize)
        throw std::invalid_argument(
            "radius exceeds the spatial index construction cell edge");
}

BvhBounds invalidBvhBounds() noexcept
{
    BvhBounds bounds;
    bounds.minimum.fill((std::numeric_limits<float>::infinity)());
    bounds.maximum.fill(-(std::numeric_limits<float>::infinity)());
    return bounds;
}

bool validBvhBounds(const BvhBounds& bounds) noexcept
{
    return bounds.minimum[0] <= bounds.maximum[0];
}

BvhBounds mergeBvhBounds(const BvhBounds& left, const BvhBounds& right) noexcept
{
    if (!validBvhBounds(left))
        return right;
    if (!validBvhBounds(right))
        return left;
    BvhBounds result;
    for (std::size_t axis = 0; axis < 3U; ++axis)
    {
        result.minimum[axis] =
            (std::min)(left.minimum[axis], right.minimum[axis]);
        result.maximum[axis] =
            (std::max)(left.maximum[axis], right.maximum[axis]);
    }
    return result;
}

BvhBounds pointBvhBounds(const PointBatch& batch,
                         const UniformGridConfig& config,
                         std::uint32_t point) noexcept
{
    const std::array<const double*, 3> values{
        batch.data<double>(X), batch.data<double>(Y),
        config.dimensions == 3 ? batch.data<double>(Z) : nullptr};
    BvhBounds result;
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        const double local = values[axis][point] - config.origin[axis];
        const float rounded = static_cast<float>(local);
        result.minimum[axis] =
            std::nextafter(rounded, -(std::numeric_limits<float>::infinity)());
        result.maximum[axis] =
            std::nextafter(rounded, (std::numeric_limits<float>::infinity)());
    }
    for (std::uint8_t axis = config.dimensions; axis < 3U; ++axis)
        result.minimum[axis] = result.maximum[axis] = 0.0F;
    return result;
}

BvhBounds heapBvhBounds(std::size_t node, std::size_t leafBase,
                        std::size_t pointCount,
                        const std::uint32_t* sortedPointIds,
                        const BvhBounds* internalBounds,
                        const PointBatch& batch,
                        const UniformGridConfig& config) noexcept
{
    const std::size_t leafStart = leafBase - 1U;
    if (node < leafStart)
        return internalBounds[node];
    const std::size_t position = node - leafStart;
    if (position >= pointCount)
        return invalidBvhBounds();
    return pointBvhBounds(batch, config, sortedPointIds[position]);
}

double bvhDistanceSquared(const std::array<double, 3>& query,
                          const UniformGridConfig& config,
                          const BvhBounds& bounds) noexcept
{
    if (!validBvhBounds(bounds))
        return (std::numeric_limits<double>::infinity)();
    double distance = 0.0;
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        const double local = query[axis] - config.origin[axis];
        double delta = 0.0;
        if (local < static_cast<double>(bounds.minimum[axis]))
            delta = static_cast<double>(bounds.minimum[axis]) - local;
        else if (local > static_cast<double>(bounds.maximum[axis]))
            delta = local - static_cast<double>(bounds.maximum[axis]);
        const double scale = std::abs(query[axis]) +
                             std::abs(config.origin[axis]) + std::abs(local) +
                             std::abs(delta) + 1.0;
        const double error =
            scale * 16.0 * std::numeric_limits<double>::epsilon();
        delta = delta > error ? delta - error : 0.0;
        const double term = std::nextafter(delta * delta, 0.0);
        distance = std::nextafter(distance + term, 0.0);
    }
    return distance;
}

void validateKnnNeighbors(std::uint32_t neighbors)
{
    if (neighbors == 0U || neighbors > MaximumKnnNeighbors)
        throw std::invalid_argument(
            "k-nearest spatial queries require between 1 and 64 neighbors");
}

void validateKnnSummary(const KnnConfigSummary& summary)
{
    if (!validDimensions(summary.dimensions))
        throw std::invalid_argument(
            "k-nearest summary dimensions must be 2 or 3");
    if (summary.pointCount >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "k-nearest summary currently supports at most INT_MAX points");
    const std::size_t expectedProbes =
        (std::min)(summary.pointCount, KnnConfigMaximumProbePoints);
    if (summary.probes.size() != expectedProbes)
        throw std::invalid_argument(
            "k-nearest summary has an invalid adaptive probe count");
    if (summary.pointCount == 0U)
        return;
    for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
    {
        if (!std::isfinite(summary.minimum[axis]) ||
            !std::isfinite(summary.maximum[axis]) ||
            summary.maximum[axis] < summary.minimum[axis])
            throw std::invalid_argument(
                "k-nearest summary coordinates must form a finite envelope");
    }
    for (const std::array<double, 3>& probe : summary.probes)
        for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
            if (!std::isfinite(probe[axis]))
                throw std::invalid_argument(
                    "k-nearest summary probe coordinates must be finite");
}

UniformGridConfig summaryGridConfig(const KnnConfigSummary& summary,
                                    double cellSize)
{
    UniformGridConfig config;
    config.dimensions = summary.dimensions;
    config.cellSize = cellSize;
    validateConfigShape(config);
    if (summary.pointCount == 0U)
        return config;
    for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
    {
        config.origin[axis] = summary.minimum[axis];
        config.maximumCell[axis] =
            cellCoordinate(summary.maximum[axis], summary.minimum[axis],
                           cellSize);
    }
    return config;
}

std::array<std::uint32_t, 3>
summaryPointCell(const std::array<double, 3>& point,
                 const UniformGridConfig& config)
{
    return {cellCoordinate(point[0], config.origin[0], config.cellSize),
            cellCoordinate(point[1], config.origin[1], config.cellSize),
            config.dimensions == 3
                ? cellCoordinate(point[2], config.origin[2], config.cellSize)
                : 0U};
}

double conservativeFaceDistanceSquared(double query, double origin,
                                       double cellSize,
                                       std::uint32_t boundaryCell) noexcept
{
    const double offset = static_cast<double>(boundaryCell) * cellSize;
    const double boundary = origin + offset;
    const double raw = std::abs(query - boundary);
    const double scale = std::abs(query) + std::abs(origin) + std::abs(offset) +
                         std::abs(cellSize);
    const double error = scale * 64.0 * std::numeric_limits<double>::epsilon() +
                         std::numeric_limits<double>::denorm_min();
    if (raw <= error)
        return 0.0;
    const double lower = std::nextafter(raw - error, 0.0);
    return std::nextafter(lower * lower, 0.0);
}

double
outsideShellDistanceSquared(const std::array<double, 3>& queryCoordinates,
                            const std::array<std::uint32_t, 3>& queryCell,
                            const UniformGridConfig& config,
                            std::uint32_t shell) noexcept
{
    double lower = (std::numeric_limits<double>::infinity)();
    for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
    {
        if (queryCell[axis] > shell)
            lower = (std::min)(lower,
                               conservativeFaceDistanceSquared(
                                   queryCoordinates[axis], config.origin[axis],
                                   config.cellSize, queryCell[axis] - shell));
        if (queryCell[axis] + shell < config.maximumCell[axis])
            lower = (std::min)(lower, conservativeFaceDistanceSquared(
                                          queryCoordinates[axis],
                                          config.origin[axis], config.cellSize,
                                          queryCell[axis] + shell + 1U));
    }
    return lower;
}
} // unnamed namespace

std::size_t knnConfigProbePoint(std::size_t pointCount, std::size_t probe)
{
    const std::size_t probeCount =
        (std::min)(pointCount, KnnConfigMaximumProbePoints);
    if (probe >= probeCount)
        throw std::out_of_range(
            "k-nearest adaptive probe index is outside the summary");
    return probeCount == pointCount
               ? probe
               : static_cast<std::size_t>(
                     deterministicProbe(static_cast<std::uint64_t>(probe)) %
                     static_cast<std::uint64_t>(pointCount));
}

UniformGridConfig makeUniformGridConfig(const PointBatch& hostBatch,
                                        std::uint8_t dimensions,
                                        double cellSize)
{
    UniformGridConfig config;
    config.dimensions = dimensions;
    config.cellSize = cellSize;
    validateConfigShape(config);
    if (hostBatch.memoryKind() == MemoryKind::Device)
        throw std::invalid_argument(
            "uniform grid configuration requires host-visible coordinates");
    if (!validColumns(hostBatch, dimensions))
        throw std::invalid_argument(
            "uniform spatial index requires logical double coordinate columns");
    if (hostBatch.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "uniform spatial index currently supports at most INT_MAX points");
    if (hostBatch.size() == 0)
        return config;

    const std::array<const double*, 3> values{
        hostBatch.data<double>(X), hostBatch.data<double>(Y),
        dimensions == 3 ? hostBatch.data<double>(Z) : nullptr};
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
    {
        double minimum = values[axis][0];
        double maximum = values[axis][0];
        if (!std::isfinite(minimum))
            throw std::invalid_argument(
                "uniform spatial index coordinates must be finite");
        for (std::size_t point = 1; point < hostBatch.size(); ++point)
        {
            const double value = values[axis][point];
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "uniform spatial index coordinates must be finite");
            minimum = (std::min)(minimum, value);
            maximum = (std::max)(maximum, value);
        }
        config.origin[axis] = minimum;
        config.maximumCell[axis] = cellCoordinate(maximum, minimum, cellSize);
    }
    return config;
}

UniformGridConfig makeUniformGridConfig(const KnnConfigSummary& summary,
                                        double cellSize)
{
    UniformGridConfig config;
    config.dimensions = summary.dimensions;
    config.cellSize = cellSize;
    validateConfigShape(config);
    if (summary.pointCount >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "uniform spatial index currently supports at most INT_MAX points");
    if (summary.pointCount == 0U)
        return config;
    for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
    {
        const double minimum = summary.minimum[axis];
        const double maximum = summary.maximum[axis];
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            maximum < minimum)
            throw std::invalid_argument(
                "uniform spatial index coordinates must be finite");
        config.origin[axis] = minimum;
        config.maximumCell[axis] =
            cellCoordinate(maximum, minimum, cellSize);
    }
    return config;
}

UniformGridConfig makeKnnGridConfig(const PointBatch& hostBatch,
                                    std::uint8_t dimensions,
                                    std::uint32_t neighbors)
{
    validateKnnNeighbors(neighbors);
    if (!validDimensions(dimensions))
        throw std::invalid_argument(
            "uniform spatial index dimensions must be 2 or 3");
    if (hostBatch.memoryKind() == MemoryKind::Device)
        throw std::invalid_argument(
            "k-nearest grid configuration requires host-visible coordinates");
    if (!validColumns(hostBatch, dimensions))
        throw std::invalid_argument(
            "uniform spatial index requires logical double coordinate columns");
    if (hostBatch.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "uniform spatial index currently supports at most INT_MAX points");
    if (hostBatch.size() == 0)
    {
        UniformGridConfig config =
            makeUniformGridConfig(hostBatch, dimensions, 1.0);
        config.knnCandidateArrays = true;
        return config;
    }

    const std::array<const double*, 3> values{
        hostBatch.data<double>(X), hostBatch.data<double>(Y),
        dimensions == 3 ? hostBatch.data<double>(Z) : nullptr};
    double maximumSpan = 0.0;
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
    {
        double minimum = values[axis][0];
        double maximum = values[axis][0];
        if (!std::isfinite(minimum))
            throw std::invalid_argument(
                "uniform spatial index coordinates must be finite");
        for (std::size_t point = 1; point < hostBatch.size(); ++point)
        {
            const double value = values[axis][point];
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "uniform spatial index coordinates must be finite");
            minimum = (std::min)(minimum, value);
            maximum = (std::max)(maximum, value);
        }
        const double span = maximum - minimum;
        if (!std::isfinite(span))
            throw std::invalid_argument(
                "uniform spatial index coordinate span must be finite");
        maximumSpan = (std::max)(maximumSpan, span);
    }

    if (maximumSpan == 0.0)
    {
        UniformGridConfig config =
            makeUniformGridConfig(hostBatch, dimensions, 1.0);
        config.knnCandidateArrays = true;
        return config;
    }

    // Aim for roughly k candidates across the 3x3 (2D) or 3x3x3
    // (3D) neighborhood surrounding a query cell. This is only a search
    // heuristic; the query's conservative shell proof controls correctness.
    const double neighboringCells = dimensions == 3 ? 27.0 : 9.0;
    const double ratio = neighboringCells *
                         static_cast<double>(hostBatch.size()) /
                         static_cast<double>(neighbors);
    const double inverseDimensions = 1.0 / static_cast<double>(dimensions);
    const double cellsAcross =
        (std::max)(1.0, std::pow(ratio, inverseDimensions));
    double cellSize = maximumSpan / cellsAcross;
    if (!std::isfinite(cellSize) || cellSize <= 0.0)
        cellSize = maximumSpan;
    UniformGridConfig config =
        makeUniformGridConfig(hostBatch, dimensions, cellSize);
    config.knnCandidateArrays = true;
    return config;
}

UniformGridConfig makeMortonBvhConfig(const PointBatch& hostBatch,
                                      std::uint8_t dimensions)
{
    if (!validDimensions(dimensions))
        throw std::invalid_argument("Morton BVH dimensions must be 2 or 3");
    if (hostBatch.memoryKind() == MemoryKind::Device)
        throw std::invalid_argument(
            "Morton BVH configuration requires host-visible coordinates");
    if (!validColumns(hostBatch, dimensions))
        throw std::invalid_argument(
            "Morton BVH requires logical double coordinate columns");
    if (hostBatch.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "Morton BVH currently supports at most INT_MAX points");
    if (hostBatch.size() == 0)
    {
        UniformGridConfig config =
            makeUniformGridConfig(hostBatch, dimensions, 1.0);
        config.backend = SpatialIndexBackend::MortonBvh;
        return config;
    }

    const std::array<const double*, 3> values{
        hostBatch.data<double>(X), hostBatch.data<double>(Y),
        dimensions == 3 ? hostBatch.data<double>(Z) : nullptr};
    double maximumSpan = 0.0;
    for (std::uint8_t axis = 0; axis < dimensions; ++axis)
    {
        double minimum = values[axis][0];
        double maximum = values[axis][0];
        if (!std::isfinite(minimum))
            throw std::invalid_argument(
                "Morton BVH coordinates must be finite");
        for (std::size_t point = 1; point < hostBatch.size(); ++point)
        {
            const double value = values[axis][point];
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "Morton BVH coordinates must be finite");
            minimum = (std::min)(minimum, value);
            maximum = (std::max)(maximum, value);
        }
        const double span = maximum - minimum;
        if (!std::isfinite(span) ||
            span > static_cast<double>((std::numeric_limits<float>::max)()))
            throw std::invalid_argument(
                "Morton BVH local coordinate span exceeds binary32 bounds");
        maximumSpan = (std::max)(maximumSpan, span);
    }

    double cellSize = 1.0;
    if (maximumSpan > 0.0)
    {
        cellSize =
            std::nextafter(maximumSpan / static_cast<double>(MaximumCell),
                           (std::numeric_limits<double>::infinity)());
        if (!std::isfinite(cellSize) || cellSize <= 0.0)
            cellSize = maximumSpan;
    }
    UniformGridConfig config =
        makeUniformGridConfig(hostBatch, dimensions, cellSize);
    config.backend = SpatialIndexBackend::MortonBvh;
    return config;
}

UniformGridConfig makeAdaptiveKnnConfig(const PointBatch& hostBatch,
                                        std::uint8_t dimensions,
                                        std::uint32_t neighbors)
{
    UniformGridConfig grid =
        makeKnnGridConfig(hostBatch, dimensions, neighbors);
    const std::size_t probeCount =
        (std::min)(hostBatch.size(), KnnConfigMaximumProbePoints);
    if (probeCount < 64U)
        return grid;

    std::vector<std::uint64_t> keys(probeCount);
    for (std::size_t probe = 0; probe < probeCount; ++probe)
    {
        // A fixed-stride probe aliases common source orderings (scan lines,
        // interleaved returns, and synthetic cluster ids). A deterministic
        // hash is still reproducible but samples those periodic layouts
        // without making input order itself a density signal.
        const std::size_t point =
            knnConfigProbePoint(hostBatch.size(), probe);
        const auto cell = pointCell(hostBatch, grid, point);
        keys[probe] = cellKey(cell[0], cell[1], cell[2]);
    }
    std::sort(keys.begin(), keys.end());
    std::size_t maximumRun = 1U;
    std::size_t run = 1U;
    std::size_t unique = 1U;
    for (std::size_t point = 1U; point < keys.size(); ++point)
    {
        if (keys[point] == keys[point - 1U])
        {
            ++run;
            maximumRun = (std::max)(maximumRun, run);
        }
        else
        {
            ++unique;
            run = 1U;
        }
    }

    const std::uint64_t estimatedMaximumPopulation =
        (static_cast<std::uint64_t>(maximumRun) *
             static_cast<std::uint64_t>(hostBatch.size()) +
         static_cast<std::uint64_t>(probeCount - 1U)) /
        static_cast<std::uint64_t>(probeCount);
    const bool broadlyClustered = unique * 8U < probeCount;
    if (broadlyClustered &&
        estimatedMaximumPopulation >= BvhEstimatedCellPopulation)
        return makeMortonBvhConfig(hostBatch, dimensions);
    return grid;
}

UniformGridConfig makeKnnGridConfig(const KnnConfigSummary& summary,
                                    std::uint32_t neighbors)
{
    validateKnnNeighbors(neighbors);
    validateKnnSummary(summary);
    if (summary.pointCount == 0U)
    {
        UniformGridConfig config = summaryGridConfig(summary, 1.0);
        config.knnCandidateArrays = true;
        return config;
    }

    double maximumSpan = 0.0;
    for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
    {
        const double span = summary.maximum[axis] - summary.minimum[axis];
        if (!std::isfinite(span))
            throw std::invalid_argument(
                "uniform spatial index coordinate span must be finite");
        maximumSpan = (std::max)(maximumSpan, span);
    }

    double cellSize = 1.0;
    if (maximumSpan > 0.0)
    {
        const double neighboringCells = summary.dimensions == 3 ? 27.0 : 9.0;
        const double ratio = neighboringCells *
                             static_cast<double>(summary.pointCount) /
                             static_cast<double>(neighbors);
        const double inverseDimensions =
            1.0 / static_cast<double>(summary.dimensions);
        const double cellsAcross =
            (std::max)(1.0, std::pow(ratio, inverseDimensions));
        cellSize = maximumSpan / cellsAcross;
        if (!std::isfinite(cellSize) || cellSize <= 0.0)
            cellSize = maximumSpan;
    }
    UniformGridConfig config = summaryGridConfig(summary, cellSize);
    config.knnCandidateArrays = true;
    return config;
}

UniformGridConfig makeMortonBvhConfig(const KnnConfigSummary& summary)
{
    validateKnnSummary(summary);
    double maximumSpan = 0.0;
    for (std::uint8_t axis = 0; axis < summary.dimensions; ++axis)
    {
        const double span = summary.maximum[axis] - summary.minimum[axis];
        if (!std::isfinite(span) ||
            span > static_cast<double>((std::numeric_limits<float>::max)()))
            throw std::invalid_argument(
                "Morton BVH local coordinate span exceeds binary32 bounds");
        maximumSpan = (std::max)(maximumSpan, span);
    }

    double cellSize = 1.0;
    if (maximumSpan > 0.0)
    {
        cellSize =
            std::nextafter(maximumSpan / static_cast<double>(MaximumCell),
                           (std::numeric_limits<double>::infinity)());
        if (!std::isfinite(cellSize) || cellSize <= 0.0)
            cellSize = maximumSpan;
    }
    UniformGridConfig config = summaryGridConfig(summary, cellSize);
    config.backend = SpatialIndexBackend::MortonBvh;
    return config;
}

UniformGridConfig makeAdaptiveKnnConfig(const KnnConfigSummary& summary,
                                        std::uint32_t neighbors)
{
    UniformGridConfig grid = makeKnnGridConfig(summary, neighbors);
    const std::size_t probeCount = summary.probes.size();
    if (probeCount < 64U)
        return grid;

    std::vector<std::uint64_t> keys(probeCount);
    for (std::size_t probe = 0; probe < probeCount; ++probe)
    {
        const auto cell = summaryPointCell(summary.probes[probe], grid);
        keys[probe] = cellKey(cell[0], cell[1], cell[2]);
    }
    std::sort(keys.begin(), keys.end());
    std::size_t maximumRun = 1U;
    std::size_t run = 1U;
    std::size_t unique = 1U;
    for (std::size_t point = 1U; point < keys.size(); ++point)
    {
        if (keys[point] == keys[point - 1U])
        {
            ++run;
            maximumRun = (std::max)(maximumRun, run);
        }
        else
        {
            ++unique;
            run = 1U;
        }
    }

    const std::uint64_t estimatedMaximumPopulation =
        (static_cast<std::uint64_t>(maximumRun) *
             static_cast<std::uint64_t>(summary.pointCount) +
         static_cast<std::uint64_t>(probeCount - 1U)) /
        static_cast<std::uint64_t>(probeCount);
    const bool broadlyClustered = unique * 8U < probeCount;
    if (broadlyClustered &&
        estimatedMaximumPopulation >= BvhEstimatedCellPopulation)
        return makeMortonBvhConfig(summary);
    return grid;
}

bool uniformGridMaySupportExactDevice(const PointBatch& batch,
                                      const UniformGridConfig& config) noexcept
{
    try
    {
        validateConfigShape(config);
        if (!validColumns(batch, config.dimensions) ||
            batch.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            batch.capacity() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return false;
        if (batch.memoryKind() == MemoryKind::Device)
            return true;
        for (std::size_t point = 0; point < batch.size(); ++point)
        {
            const auto cell = pointCell(batch, config, point);
            for (std::uint8_t axis = 0; axis < config.dimensions; ++axis)
            {
                if (cell[axis] > config.maximumCell[axis])
                    return false;
                if (config.backend == SpatialIndexBackend::MortonBvh)
                {
                    const DimensionId dimension =
                        axis == 0 ? X : (axis == 1 ? Y : Z);
                    const double local = batch.data<double>(dimension)[point] -
                                         config.origin[axis];
                    if (!std::isfinite(static_cast<float>(local)))
                        return false;
                }
            }
        }
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

SpatialIndex::SpatialIndex(PointBatch& batch, UniformGridConfig config)
    : m_batch(&batch), m_config(std::move(config))
{
    validateConfigShape(m_config);
    if (!uniformGridMaySupportExactDevice(batch, m_config))
        throw std::invalid_argument(
            "point batch is outside the exact uniform-grid envelope");
}

SpatialIndex::~SpatialIndex() = default;
SpatialIndex::SpatialIndex(SpatialIndex&&) noexcept = default;
SpatialIndex& SpatialIndex::operator=(SpatialIndex&&) noexcept = default;

void SpatialIndex::allocateStorage()
{
    if (m_batch->capacity() == 0 || m_sortedKeys)
        return;
    MemoryResource& memory = m_batch->memoryResource();
    const std::size_t capacity = m_batch->capacity();
    m_sortedKeys = memory.allocate(capacity * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
    m_sortedPointIds = memory.allocate(capacity * sizeof(std::uint32_t),
                                       alignof(std::uint32_t));
    m_uniqueKeys = memory.allocate(capacity * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
    m_cellOffsets = memory.allocate(capacity * sizeof(std::uint32_t),
                                    alignof(std::uint32_t));
    m_cellCounts = memory.allocate(capacity * sizeof(std::uint32_t),
                                   alignof(std::uint32_t));
    // The contiguous-candidate copies exist only where the uniform-grid
    // device kNN kernels read them; host mirrors and radius-only or BVH
    // indexes keep gathering through sortedPointIds.
    if (m_batch->memoryKind() == MemoryKind::Device &&
        m_config.knnCandidateArrays &&
        m_config.backend == SpatialIndexBackend::UniformGrid)
    {
        m_sortedCoordinates =
            memory.allocate(capacity * 3U * sizeof(double), alignof(double));
        m_sortedFloatCoordinates =
            memory.allocate(capacity * 3U * sizeof(float), alignof(float));
    }
}

void SpatialIndex::allocateBvhStorage()
{
    if (m_config.backend != SpatialIndexBackend::MortonBvh ||
        m_batch->capacity() == 0 || m_bvhLeafBase != 0)
        return;
    m_bvhLeafBase = std::bit_ceil(m_batch->capacity());
    if (m_bvhLeafBase <= 1U)
        return;
    const std::size_t internalCount = m_bvhLeafBase - 1U;
    m_bvhBounds = m_batch->memoryResource().allocate(
        internalCount * sizeof(BvhBounds), alignof(BvhBounds));
}

void SpatialIndex::build()
{
    if (!uniformGridMaySupportExactDevice(*m_batch, m_config))
        throw std::invalid_argument(
            "point batch is outside the exact uniform-grid envelope");
    allocateStorage();
    allocateBvhStorage();
    m_cellCount = 0;
    if (m_batch->size() != 0)
    {
        if (m_batch->memoryKind() == MemoryKind::Device)
            detail::buildUniformGridDevice(*this);
        else
        {
            struct Entry
            {
                std::uint64_t key;
                std::uint32_t point;
            };
            std::vector<Entry> entries(m_batch->size());
            for (std::size_t point = 0; point < m_batch->size(); ++point)
            {
                const auto cell = pointCell(*m_batch, m_config, point);
                entries[point] = {cellKey(cell[0], cell[1], cell[2]),
                                  static_cast<std::uint32_t>(point)};
            }
            std::stable_sort(entries.begin(), entries.end(),
                             [](const Entry& left, const Entry& right)
                             { return left.key < right.key; });
            auto* sortedKeys =
                static_cast<std::uint64_t*>(m_sortedKeys->data());
            auto* sortedPointIds =
                static_cast<std::uint32_t*>(m_sortedPointIds->data());
            auto* uniqueKeys =
                static_cast<std::uint64_t*>(m_uniqueKeys->data());
            auto* cellOffsets =
                static_cast<std::uint32_t*>(m_cellOffsets->data());
            auto* cellCounts =
                static_cast<std::uint32_t*>(m_cellCounts->data());
            for (std::size_t position = 0; position < entries.size();
                 ++position)
            {
                sortedKeys[position] = entries[position].key;
                sortedPointIds[position] = entries[position].point;
                if (position == 0 ||
                    entries[position].key != entries[position - 1].key)
                {
                    uniqueKeys[m_cellCount] = entries[position].key;
                    cellOffsets[m_cellCount] =
                        static_cast<std::uint32_t>(position);
                    cellCounts[m_cellCount] = 0;
                    ++m_cellCount;
                }
                ++cellCounts[m_cellCount - 1];
            }

            if (m_config.backend == SpatialIndexBackend::MortonBvh &&
                m_bvhLeafBase > 1U)
            {
                auto* bounds = static_cast<BvhBounds*>(m_bvhBounds->data());
                const std::size_t leafStart = m_bvhLeafBase - 1U;
                for (std::size_t node = leafStart; node-- > 0U;)
                {
                    const std::size_t left = node * 2U + 1U;
                    const std::size_t right = left + 1U;
                    bounds[node] = mergeBvhBounds(
                        heapBvhBounds(left, m_bvhLeafBase, m_batch->size(),
                                      sortedPointIds, bounds, *m_batch,
                                      m_config),
                        heapBvhBounds(right, m_bvhLeafBase, m_batch->size(),
                                      sortedPointIds, bounds, *m_batch,
                                      m_config));
                }
            }
        }
    }
    m_valid = true;
    ++m_buildCount;
}

void SpatialIndex::invalidate() noexcept
{
    m_valid = false;
}

bool SpatialIndex::valid() const noexcept
{
    return m_valid;
}

std::size_t SpatialIndex::buildCount() const noexcept
{
    return m_buildCount;
}

std::size_t SpatialIndex::cellCount() const noexcept
{
    return m_cellCount;
}

std::size_t SpatialIndex::allocatedBytes() const noexcept
{
    const auto bytes = [](const std::unique_ptr<Allocation>& allocation)
    { return allocation ? allocation->size() : 0U; };
    return bytes(m_sortedKeys) + bytes(m_sortedPointIds) + bytes(m_uniqueKeys) +
           bytes(m_cellOffsets) + bytes(m_cellCounts) +
           bytes(m_sortedCoordinates) + bytes(m_sortedFloatCoordinates) +
           bytes(m_bvhBounds);
}

MemoryKind SpatialIndex::memoryKind() const noexcept
{
    return m_batch->memoryKind();
}

const UniformGridConfig& SpatialIndex::config() const noexcept
{
    return m_config;
}

const PointBatch& SpatialIndex::batch() const noexcept
{
    return *m_batch;
}

void radiusCounts(const SpatialIndex& index, double radius,
                  std::uint32_t* counts)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateRadius(index, radius);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (!counts)
        throw std::invalid_argument("radius count output cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::radiusCountsDevice(index, radius, counts);
        return;
    }

    const auto* uniqueKeys =
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data());
    const auto* cellOffsets =
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data());
    const auto* cellCounts =
        static_cast<const std::uint32_t*>(index.m_cellCounts->data());
    const auto* pointIds =
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data());
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z =
        index.m_config.dimensions == 3 ? batch.data<double>(Z) : nullptr;
    const double radiusSquared = radius * radius;
    if (index.m_config.backend == SpatialIndexBackend::MortonBvh)
    {
        const auto* bounds = static_cast<const BvhBounds*>(
            index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr);
        const std::size_t leafBase = index.m_bvhLeafBase;
        const std::size_t leafStart = leafBase - 1U;
        for (std::size_t query = 0; query < batch.size(); ++query)
        {
            const std::array<double, 3> queryCoordinates{
                x[query], y[query],
                index.m_config.dimensions == 3 ? z[query] : 0.0};
            std::array<std::size_t, 64> stack{};
            std::size_t stackSize = 1U;
            stack[0] = leafBase > 1U ? 0U : leafStart;
            std::uint32_t count = 0U;
            while (stackSize != 0U)
            {
                const std::size_t node = stack[--stackSize];
                const BvhBounds nodeBounds =
                    heapBvhBounds(node, leafBase, batch.size(), pointIds,
                                  bounds, batch, index.m_config);
                if (bvhDistanceSquared(queryCoordinates, index.m_config,
                                       nodeBounds) >= radiusSquared)
                    continue;
                if (node >= leafStart)
                {
                    const std::size_t position = node - leafStart;
                    if (position >= batch.size())
                        continue;
                    const std::uint32_t candidate = pointIds[position];
                    const double dx = x[query] - x[candidate];
                    const double dy = y[query] - y[candidate];
                    double distance = dx * dx + dy * dy;
                    if (index.m_config.dimensions == 3)
                    {
                        const double dz = z[query] - z[candidate];
                        distance += dz * dz;
                    }
                    if (distance < radiusSquared)
                        ++count;
                    continue;
                }
                if (stackSize + 2U > stack.size())
                    throw std::logic_error("Morton BVH traversal overflow");
                stack[stackSize++] = node * 2U + 2U;
                stack[stackSize++] = node * 2U + 1U;
            }
            counts[query] = count;
        }
        return;
    }
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        const auto queryCell = pointCell(batch, index.m_config, query);
        std::uint32_t count = 0;
        const int zMinimum = index.m_config.dimensions == 3 ? -1 : 0;
        const int zMaximum = index.m_config.dimensions == 3 ? 1 : 0;
        for (int dzCell = zMinimum; dzCell <= zMaximum; ++dzCell)
            for (int dyCell = -1; dyCell <= 1; ++dyCell)
                for (int dxCell = -1; dxCell <= 1; ++dxCell)
                {
                    const std::int64_t cellX =
                        static_cast<std::int64_t>(queryCell[0]) + dxCell;
                    const std::int64_t cellY =
                        static_cast<std::int64_t>(queryCell[1]) + dyCell;
                    const std::int64_t cellZ =
                        static_cast<std::int64_t>(queryCell[2]) + dzCell;
                    if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                        cellX > MaximumCell || cellY > MaximumCell ||
                        cellZ > MaximumCell)
                        continue;
                    const std::uint64_t key =
                        cellKey(static_cast<std::uint32_t>(cellX),
                                static_cast<std::uint32_t>(cellY),
                                static_cast<std::uint32_t>(cellZ));
                    const auto position = std::lower_bound(
                        uniqueKeys, uniqueKeys + index.m_cellCount, key);
                    if (position == uniqueKeys + index.m_cellCount ||
                        *position != key)
                        continue;
                    const std::size_t cell =
                        static_cast<std::size_t>(position - uniqueKeys);
                    const std::size_t end =
                        static_cast<std::size_t>(cellOffsets[cell]) +
                        cellCounts[cell];
                    for (std::size_t item = cellOffsets[cell]; item < end;
                         ++item)
                    {
                        const std::uint32_t candidate = pointIds[item];
                        const double dx = x[query] - x[candidate];
                        const double dy = y[query] - y[candidate];
                        double distance = dx * dx + dy * dy;
                        if (index.m_config.dimensions == 3)
                        {
                            const double dz = z[query] - z[candidate];
                            distance += dz * dz;
                        }
                        if (distance < radiusSquared)
                            ++count;
                    }
                }
        counts[query] = count;
    }
}

void radiusAny(const SpatialIndex& index, double radius,
               const std::uint8_t* sourceMask,
               const std::uint8_t* referenceMask, double maximumAbove,
               double maximumBelow, std::uint8_t* matches)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateRadius(index, radius);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0U)
        return;
    if (!matches)
        throw std::invalid_argument("radius-any output cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::radiusAnyDevice(index, radius, sourceMask, referenceMask,
                                maximumAbove, maximumBelow, matches);
        return;
    }

    const auto* uniqueKeys =
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data());
    const auto* cellOffsets =
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data());
    const auto* cellCounts =
        static_cast<const std::uint32_t*>(index.m_cellCounts->data());
    const auto* pointIds =
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data());
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    const double radiusSquared = radius * radius;
    const auto accepts = [&](std::size_t query, std::uint32_t candidate)
    {
        if (referenceMask && !referenceMask[candidate])
            return false;
        const double dx = x[query] - x[candidate];
        const double dy = y[query] - y[candidate];
        double distance = dx * dx + dy * dy;
        if (index.m_config.dimensions == 3)
        {
            const double dz = z[query] - z[candidate];
            distance += dz * dz;
        }
        if (!(distance < radiusSquared))
            return false;
        if (index.m_config.dimensions == 2)
        {
            const double difference = z[candidate] - z[query];
            if (maximumAbove >= 0.0 && difference > 0.0 &&
                difference > maximumAbove)
                return false;
            if (maximumBelow >= 0.0 && difference < 0.0 &&
                -difference > maximumBelow)
                return false;
        }
        return true;
    };

    if (index.m_config.backend == SpatialIndexBackend::MortonBvh)
    {
        const auto* bounds = static_cast<const BvhBounds*>(
            index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr);
        const std::size_t leafBase = index.m_bvhLeafBase;
        const std::size_t leafStart = leafBase - 1U;
        for (std::size_t query = 0; query < batch.size(); ++query)
        {
            matches[query] = 0U;
            if (sourceMask && !sourceMask[query])
                continue;
            const std::array<double, 3> queryCoordinates{
                x[query], y[query],
                index.m_config.dimensions == 3 ? z[query] : 0.0};
            std::array<std::size_t, 64> stack{};
            std::size_t stackSize = 1U;
            stack[0] = leafBase > 1U ? 0U : leafStart;
            while (stackSize != 0U && !matches[query])
            {
                const std::size_t node = stack[--stackSize];
                const BvhBounds nodeBounds =
                    heapBvhBounds(node, leafBase, batch.size(), pointIds,
                                  bounds, batch, index.m_config);
                if (bvhDistanceSquared(queryCoordinates, index.m_config,
                                       nodeBounds) >= radiusSquared)
                    continue;
                if (node >= leafStart)
                {
                    const std::size_t position = node - leafStart;
                    if (position < batch.size() &&
                        accepts(query, pointIds[position]))
                        matches[query] = 1U;
                    continue;
                }
                if (stackSize + 2U > stack.size())
                    throw std::logic_error("Morton BVH traversal overflow");
                stack[stackSize++] = node * 2U + 2U;
                stack[stackSize++] = node * 2U + 1U;
            }
        }
        return;
    }

    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        matches[query] = 0U;
        if (sourceMask && !sourceMask[query])
            continue;
        const auto queryCell = pointCell(batch, index.m_config, query);
        const int zMinimum = index.m_config.dimensions == 3 ? -1 : 0;
        const int zMaximum = index.m_config.dimensions == 3 ? 1 : 0;
        for (int dzCell = zMinimum; dzCell <= zMaximum && !matches[query];
             ++dzCell)
            for (int dyCell = -1; dyCell <= 1 && !matches[query]; ++dyCell)
                for (int dxCell = -1; dxCell <= 1 && !matches[query]; ++dxCell)
                {
                    const std::int64_t cellX =
                        static_cast<std::int64_t>(queryCell[0]) + dxCell;
                    const std::int64_t cellY =
                        static_cast<std::int64_t>(queryCell[1]) + dyCell;
                    const std::int64_t cellZ =
                        static_cast<std::int64_t>(queryCell[2]) + dzCell;
                    if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                        cellX > MaximumCell || cellY > MaximumCell ||
                        cellZ > MaximumCell)
                        continue;
                    const std::uint64_t key =
                        cellKey(static_cast<std::uint32_t>(cellX),
                                static_cast<std::uint32_t>(cellY),
                                static_cast<std::uint32_t>(cellZ));
                    const auto position = std::lower_bound(
                        uniqueKeys, uniqueKeys + index.m_cellCount, key);
                    if (position == uniqueKeys + index.m_cellCount ||
                        *position != key)
                        continue;
                    const std::size_t cell =
                        static_cast<std::size_t>(position - uniqueKeys);
                    const std::size_t end =
                        static_cast<std::size_t>(cellOffsets[cell]) +
                        cellCounts[cell];
                    for (std::size_t item = cellOffsets[cell]; item < end;
                         ++item)
                        if (accepts(query, pointIds[item]))
                        {
                            matches[query] = 1U;
                            break;
                        }
                }
    }
}

void radiusScaledValues(const SpatialIndex& index, double radius, double factor,
                        double* values)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateRadius(index, radius);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0U)
        return;
    if (!values)
        throw std::invalid_argument("scaled radius output cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::radiusScaledValuesDevice(index, radius, factor, values);
        return;
    }

    std::vector<std::uint32_t> counts(batch.size());
    radiusCounts(index, radius, counts.data());
    for (std::size_t point = 0; point < batch.size(); ++point)
        values[point] = static_cast<double>(counts[point]) * factor;
}

void knnGather(const SpatialIndex& index, std::uint32_t neighbors,
               std::uint32_t* pointIds, double* squaredDistances,
               std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (!pointIds || !squaredDistances || !status)
        throw std::invalid_argument("k-nearest query outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnGatherDevice(index, neighbors, pointIds, squaredDistances,
                                status);
        return;
    }
    // D0271: ties are reported only under the default contract.
    const std::uint8_t statusMask = knnStatusMask();

    struct Neighbor
    {
        std::uint32_t point = 0U;
        double distance = 0.0;
    };
    const std::size_t outputCount =
        (std::min)(batch.size(), static_cast<std::size_t>(neighbors));
    const std::size_t retainedCount =
        (std::min)(batch.size(), outputCount + (batch.size() > outputCount));
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z =
        index.m_config.dimensions == 3 ? batch.data<double>(Z) : nullptr;
    const auto* uniqueKeys =
        static_cast<const std::uint64_t*>(index.m_uniqueKeys->data());
    const auto* cellOffsets =
        static_cast<const std::uint32_t*>(index.m_cellOffsets->data());
    const auto* cellCounts =
        static_cast<const std::uint32_t*>(index.m_cellCounts->data());
    const auto* sortedPointIds =
        static_cast<const std::uint32_t*>(index.m_sortedPointIds->data());
    std::array<Neighbor, MaximumKnnNeighbors + 1U> best{};
    if (index.m_config.backend == SpatialIndexBackend::MortonBvh)
    {
        const auto* bounds = static_cast<const BvhBounds*>(
            index.m_bvhBounds ? index.m_bvhBounds->data() : nullptr);
        const std::size_t leafBase = index.m_bvhLeafBase;
        const std::size_t leafStart = leafBase - 1U;
        for (std::size_t query = 0; query < batch.size(); ++query)
        {
            std::size_t bestCount = 0U;
            const std::array<double, 3> queryCoordinates{
                x[query], y[query],
                index.m_config.dimensions == 3 ? z[query] : 0.0};
            std::array<std::size_t, 64> stack{};
            std::size_t stackSize = 1U;
            stack[0] = leafBase > 1U ? 0U : leafStart;
            while (stackSize != 0U)
            {
                const std::size_t node = stack[--stackSize];
                const BvhBounds nodeBounds =
                    heapBvhBounds(node, leafBase, batch.size(), sortedPointIds,
                                  bounds, batch, index.m_config);
                const double lower = bvhDistanceSquared(
                    queryCoordinates, index.m_config, nodeBounds);
                if (bestCount >= outputCount &&
                    lower > best[outputCount - 1U].distance)
                    continue;

                if (node >= leafStart)
                {
                    const std::size_t position = node - leafStart;
                    if (position >= batch.size())
                        continue;
                    const std::uint32_t candidate = sortedPointIds[position];
                    const double dx = queryCoordinates[0] - x[candidate];
                    const double dy = queryCoordinates[1] - y[candidate];
                    double distance = dx * dx + dy * dy;
                    if (index.m_config.dimensions == 3)
                    {
                        const double dz = queryCoordinates[2] - z[candidate];
                        distance += dz * dz;
                    }
                    const Neighbor value{candidate, distance};
                    std::size_t insertion = 0U;
                    while (insertion < bestCount &&
                           (best[insertion].distance < value.distance ||
                            (best[insertion].distance == value.distance &&
                             best[insertion].point < value.point)))
                        ++insertion;
                    if (insertion < retainedCount)
                    {
                        const std::size_t newCount =
                            (std::min)(retainedCount, bestCount + 1U);
                        for (std::size_t move = newCount; move > insertion + 1U;
                             --move)
                            best[move - 1U] = best[move - 2U];
                        best[insertion] = value;
                        bestCount = newCount;
                    }
                    continue;
                }

                if (stackSize + 2U > stack.size())
                    throw std::logic_error("Morton BVH traversal overflow");
                const std::size_t left = node * 2U + 1U;
                const std::size_t right = left + 1U;
                const double leftDistance = bvhDistanceSquared(
                    queryCoordinates, index.m_config,
                    heapBvhBounds(left, leafBase, batch.size(), sortedPointIds,
                                  bounds, batch, index.m_config));
                const double rightDistance = bvhDistanceSquared(
                    queryCoordinates, index.m_config,
                    heapBvhBounds(right, leafBase, batch.size(), sortedPointIds,
                                  bounds, batch, index.m_config));
                if (leftDistance <= rightDistance)
                {
                    stack[stackSize++] = right;
                    stack[stackSize++] = left;
                }
                else
                {
                    stack[stackSize++] = left;
                    stack[stackSize++] = right;
                }
            }

            std::uint8_t queryStatus = KnnExact;
            for (std::size_t item = 1U; item < bestCount; ++item)
                if (best[item - 1U].distance == best[item].distance)
                    queryStatus =
                        static_cast<std::uint8_t>(queryStatus | KnnDistanceTie);
            status[query] = static_cast<std::uint8_t>(queryStatus & statusMask);
            const std::size_t row = query * static_cast<std::size_t>(neighbors);
            const std::size_t available = (std::min)(outputCount, bestCount);
            for (std::size_t item = 0; item < available; ++item)
            {
                pointIds[row + item] = best[item].point;
                squaredDistances[row + item] = best[item].distance;
            }
            for (std::size_t item = available; item < neighbors; ++item)
            {
                pointIds[row + item] =
                    (std::numeric_limits<std::uint32_t>::max)();
                squaredDistances[row + item] =
                    (std::numeric_limits<double>::infinity)();
            }
        }
        return;
    }
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        std::size_t bestCount = 0;
        std::size_t visited = 0;
        const std::array<double, 3> queryCoordinates{
            x[query], y[query],
            index.m_config.dimensions == 3 ? z[query] : 0.0};
        const std::array<std::uint32_t, 3> queryCell =
            pointCell(batch, index.m_config, query);
        std::uint32_t maximumShell = 0U;
        for (std::uint8_t axis = 0; axis < index.m_config.dimensions; ++axis)
        {
            const std::uint32_t positive =
                index.m_config.maximumCell[axis] - queryCell[axis];
            maximumShell =
                (std::max)(maximumShell, (std::max)(queryCell[axis], positive));
        }
        const std::uint32_t shellLimit =
            (std::min)(maximumShell, MaximumKnnShell);
        bool provenComplete = false;
        for (std::uint32_t shell = 0U; shell <= shellLimit; ++shell)
        {
            const int signedShell = static_cast<int>(shell);
            const int xMinimum =
                -static_cast<int>((std::min)(shell, queryCell[0]));
            const int xMaximum = static_cast<int>((
                std::min)(shell, index.m_config.maximumCell[0] - queryCell[0]));
            const int yMinimum =
                -static_cast<int>((std::min)(shell, queryCell[1]));
            const int yMaximum = static_cast<int>((
                std::min)(shell, index.m_config.maximumCell[1] - queryCell[1]));
            const int zMinimum =
                index.m_config.dimensions == 3
                    ? -static_cast<int>((std::min)(shell, queryCell[2]))
                    : 0;
            const int zMaximum =
                index.m_config.dimensions == 3
                    ? static_cast<int>(
                          (std::min)(shell, index.m_config.maximumCell[2] -
                                                queryCell[2]))
                    : 0;
            for (int dzCell = zMinimum; dzCell <= zMaximum; ++dzCell)
                for (int dyCell = yMinimum; dyCell <= yMaximum; ++dyCell)
                    for (int dxCell = xMinimum; dxCell <= xMaximum; ++dxCell)
                    {
                        const int absoluteX = dxCell < 0 ? -dxCell : dxCell;
                        const int absoluteY = dyCell < 0 ? -dyCell : dyCell;
                        const int absoluteZ = dzCell < 0 ? -dzCell : dzCell;
                        if ((std::max)(absoluteX,
                                       (std::max)(absoluteY, absoluteZ)) !=
                            signedShell)
                            continue;
                        const std::int64_t cellX =
                            static_cast<std::int64_t>(queryCell[0]) + dxCell;
                        const std::int64_t cellY =
                            static_cast<std::int64_t>(queryCell[1]) + dyCell;
                        const std::int64_t cellZ =
                            static_cast<std::int64_t>(queryCell[2]) + dzCell;
                        if (cellX < 0 || cellY < 0 || cellZ < 0 ||
                            cellX > index.m_config.maximumCell[0] ||
                            cellY > index.m_config.maximumCell[1] ||
                            cellZ > index.m_config.maximumCell[2])
                            continue;
                        const std::uint64_t key =
                            cellKey(static_cast<std::uint32_t>(cellX),
                                    static_cast<std::uint32_t>(cellY),
                                    static_cast<std::uint32_t>(cellZ));
                        const auto position = std::lower_bound(
                            uniqueKeys, uniqueKeys + index.m_cellCount, key);
                        if (position == uniqueKeys + index.m_cellCount ||
                            *position != key)
                            continue;
                        const std::size_t cell =
                            static_cast<std::size_t>(position - uniqueKeys);
                        const std::size_t end =
                            static_cast<std::size_t>(cellOffsets[cell]) +
                            cellCounts[cell];
                        for (std::size_t item = cellOffsets[cell]; item < end;
                             ++item)
                        {
                            const std::uint32_t candidate =
                                sortedPointIds[item];
                            const double dx =
                                queryCoordinates[0] - x[candidate];
                            const double dy =
                                queryCoordinates[1] - y[candidate];
                            double distance = dx * dx + dy * dy;
                            if (index.m_config.dimensions == 3)
                            {
                                const double dz =
                                    queryCoordinates[2] - z[candidate];
                                distance += dz * dz;
                            }
                            const Neighbor value{candidate, distance};
                            std::size_t insertion = 0;
                            while (
                                insertion < bestCount &&
                                (best[insertion].distance < value.distance ||
                                 (best[insertion].distance == value.distance &&
                                  best[insertion].point < value.point)))
                                ++insertion;
                            if (insertion < retainedCount)
                            {
                                const std::size_t newCount =
                                    (std::min)(retainedCount, bestCount + 1U);
                                for (std::size_t move = newCount;
                                     move > insertion + 1U; --move)
                                    best[move - 1U] = best[move - 2U];
                                best[insertion] = value;
                                bestCount = newCount;
                            }
                            ++visited;
                        }
                    }
            if (visited == batch.size())
            {
                provenComplete = true;
                break;
            }
            if (bestCount >= outputCount &&
                best[outputCount - 1U].distance <
                    outsideShellDistanceSquared(queryCoordinates, queryCell,
                                                index.m_config, shell))
            {
                provenComplete = true;
                break;
            }
        }

        std::uint8_t queryStatus =
            provenComplete ? KnnExact : KnnSearchIncomplete;
        for (std::size_t item = 1; item < bestCount; ++item)
            if (best[item - 1U].distance == best[item].distance)
                queryStatus =
                    static_cast<std::uint8_t>(queryStatus | KnnDistanceTie);
        status[query] = static_cast<std::uint8_t>(queryStatus & statusMask);
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        const std::size_t available = (std::min)(outputCount, bestCount);
        for (std::size_t item = 0; item < available; ++item)
        {
            pointIds[row + item] = best[item].point;
            squaredDistances[row + item] = best[item].distance;
        }
        for (std::size_t item = available; item < neighbors; ++item)
        {
            pointIds[row + item] = (std::numeric_limits<std::uint32_t>::max)();
            squaredDistances[row + item] =
                (std::numeric_limits<double>::infinity)();
        }
    }
}

void knnGatherMasked(const SpatialIndex& index, std::uint32_t neighbors,
                     std::uint32_t referenceCount,
                     const std::uint8_t* sourceMask,
                     const std::uint8_t* referenceMask,
                     std::uint32_t* pointIds, double* squaredDistances,
                     std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0U)
        return;
    if (!referenceMask || !pointIds || !squaredDistances || !status)
        throw std::invalid_argument(
            "masked k-nearest query masks and outputs cannot be null");
    if (referenceCount == 0U || referenceCount > batch.size())
        throw std::invalid_argument(
            "masked k-nearest reference count is outside the batch");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnGatherMaskedDevice(
            index, neighbors, referenceCount, sourceMask, referenceMask,
            pointIds, squaredDistances, status);
        return;
    }
    // D0271: ties are reported only under the default contract.
    const std::uint8_t statusMask = knnStatusMask();

    std::size_t observedReferences = 0U;
    for (std::size_t point = 0U; point < batch.size(); ++point)
        observedReferences += static_cast<std::size_t>(referenceMask[point] != 0U);
    if (observedReferences != referenceCount)
        throw std::invalid_argument(
            "masked k-nearest reference count does not match the mask");

    struct Neighbor
    {
        std::uint32_t point = 0U;
        double distance = 0.0;
    };
    const std::size_t outputCount =
        (std::min)(static_cast<std::size_t>(referenceCount),
                   static_cast<std::size_t>(neighbors));
    const std::size_t retainedCount = (std::min)(
        static_cast<std::size_t>(referenceCount),
        outputCount + (referenceCount > outputCount ? 1U : 0U));
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z =
        index.m_config.dimensions == 3 ? batch.data<double>(Z) : nullptr;
    std::array<Neighbor, MaximumKnnNeighbors + 1U> best{};
    for (std::size_t query = 0U; query < batch.size(); ++query)
    {
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        if (sourceMask && sourceMask[query] == 0U)
        {
            status[query] = KnnExact;
            for (std::size_t item = 0U; item < neighbors; ++item)
            {
                pointIds[row + item] =
                    (std::numeric_limits<std::uint32_t>::max)();
                squaredDistances[row + item] =
                    (std::numeric_limits<double>::infinity)();
            }
            continue;
        }

        std::size_t bestCount = 0U;
        for (std::size_t candidate = 0U; candidate < batch.size(); ++candidate)
        {
            if (referenceMask[candidate] == 0U)
                continue;
            const double dx = x[query] - x[candidate];
            const double dy = y[query] - y[candidate];
            double distance = dx * dx + dy * dy;
            if (index.m_config.dimensions == 3)
            {
                const double dz = z[query] - z[candidate];
                distance += dz * dz;
            }
            const Neighbor value{static_cast<std::uint32_t>(candidate),
                                 distance};
            std::size_t insertion = 0U;
            while (insertion < bestCount &&
                   (best[insertion].distance < value.distance ||
                    (best[insertion].distance == value.distance &&
                     best[insertion].point < value.point)))
                ++insertion;
            if (insertion >= retainedCount)
                continue;
            const std::size_t newCount =
                (std::min)(retainedCount, bestCount + 1U);
            for (std::size_t move = newCount; move > insertion + 1U; --move)
                best[move - 1U] = best[move - 2U];
            best[insertion] = value;
            bestCount = newCount;
        }

        std::uint8_t queryStatus = KnnExact;
        for (std::size_t item = 1U; item < bestCount; ++item)
            if (best[item - 1U].distance == best[item].distance)
                queryStatus = static_cast<std::uint8_t>(queryStatus |
                                                        KnnDistanceTie);
        status[query] = static_cast<std::uint8_t>(queryStatus & statusMask);
        const std::size_t available = (std::min)(outputCount, bestCount);
        for (std::size_t item = 0U; item < available; ++item)
        {
            pointIds[row + item] = best[item].point;
            squaredDistances[row + item] = best[item].distance;
        }
        for (std::size_t item = available; item < neighbors; ++item)
        {
            pointIds[row + item] =
                (std::numeric_limits<std::uint32_t>::max)();
            squaredDistances[row + item] =
                (std::numeric_limits<double>::infinity)();
        }
    }
}

void knnMeanDistances(const SpatialIndex& index, std::uint32_t neighbors,
                      double* means, std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (!means || !status)
        throw std::invalid_argument(
            "k-nearest mean-distance outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnMeanDistancesDevice(index, neighbors, means, status);
        return;
    }

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    knnGather(index, neighbors, pointIds.data(), squaredDistances.data(),
              status);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        double mean = 0.0;
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        for (std::uint32_t neighbor = 1U; neighbor < neighbors; ++neighbor)
        {
            const double delta =
                std::sqrt(squaredDistances[row + neighbor]) - mean;
            mean += delta / static_cast<double>(neighbor);
        }
        means[point] = mean;
    }
}

void knnDistanceValues(const SpatialIndex& index, std::uint32_t neighbors,
                       KnnDistanceMode mode, double* values,
                       std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (neighbors < 2U)
        throw std::invalid_argument(
            "k-nearest distance requires at least one non-query neighbor");
    if (mode != KnnDistanceMode::Kth && mode != KnnDistanceMode::Average)
        throw std::invalid_argument("invalid k-nearest distance mode");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "k-nearest distance requires at least k + 1 input points");
    if (!values || !status)
        throw std::invalid_argument(
            "k-nearest distance outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnDistanceValuesDevice(index, neighbors, mode, values, status);
        return;
    }

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    knnGather(index, neighbors, pointIds.data(), squaredDistances.data(),
              status);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        if (mode == KnnDistanceMode::Kth)
        {
            values[point] = std::sqrt(squaredDistances[row + neighbors - 1U]);
            continue;
        }

        double value = 0.0;
        for (std::uint32_t neighbor = 1U; neighbor < neighbors; ++neighbor)
            value += std::sqrt(squaredDistances[row + neighbor]);
        values[point] = value / static_cast<double>(neighbors - 1U);
    }
}

void projectKnnMeanDistances(const SpatialIndex& index,
                             std::uint32_t rowNeighbors,
                             std::uint32_t meanNeighbors,
                             const double* squaredDistances, double* means)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before projecting");
    validateKnnNeighbors(rowNeighbors);
    validateKnnNeighbors(meanNeighbors);
    if (meanNeighbors > rowNeighbors)
        throw std::invalid_argument(
            "k-nearest mean projection exceeds the gathered row width");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0U)
        return;
    if (!squaredDistances || !means)
        throw std::invalid_argument(
            "k-nearest mean projection inputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::projectKnnMeanDistancesDevice(
            index, rowNeighbors, meanNeighbors, squaredDistances, means);
        return;
    }

    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t row =
            point * static_cast<std::size_t>(rowNeighbors);
        double mean = 0.0;
        for (std::uint32_t neighbor = 1U; neighbor < meanNeighbors; ++neighbor)
        {
            const double delta =
                std::sqrt(squaredDistances[row + neighbor]) - mean;
            mean += delta / static_cast<double>(neighbor);
        }
        means[point] = mean;
    }
}

void projectKnnDistanceValues(const SpatialIndex& index,
                              std::uint32_t rowNeighbors,
                              std::uint32_t distanceNeighbors,
                              KnnDistanceMode mode,
                              const double* squaredDistances, double* values)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before projecting");
    validateKnnNeighbors(rowNeighbors);
    validateKnnNeighbors(distanceNeighbors);
    if (distanceNeighbors < 2U || distanceNeighbors > rowNeighbors)
        throw std::invalid_argument(
            "k-nearest distance projection has an invalid requested width");
    if (mode != KnnDistanceMode::Kth && mode != KnnDistanceMode::Average)
        throw std::invalid_argument("invalid k-nearest distance mode");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0U)
        return;
    if (!squaredDistances || !values)
        throw std::invalid_argument(
            "k-nearest distance projection inputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::projectKnnDistanceValuesDevice(
            index, rowNeighbors, distanceNeighbors, mode, squaredDistances,
            values);
        return;
    }

    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t row =
            point * static_cast<std::size_t>(rowNeighbors);
        if (mode == KnnDistanceMode::Kth)
        {
            values[point] =
                std::sqrt(squaredDistances[row + distanceNeighbors - 1U]);
            continue;
        }
        double value = 0.0;
        for (std::uint32_t neighbor = 1U; neighbor < distanceNeighbors;
             ++neighbor)
            value += std::sqrt(squaredDistances[row + neighbor]);
        values[point] =
            value / static_cast<double>(distanceNeighbors - 1U);
    }
}

void knnCovariances(const SpatialIndex& index, std::uint32_t neighbors,
                    Covariance3d* covariances, std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (neighbors < 3U)
        throw std::invalid_argument(
            "k-nearest covariance requires at least three neighbors");
    if (index.m_config.dimensions != 3U)
        throw std::invalid_argument(
            "k-nearest covariance requires a three-dimensional index");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "k-nearest covariance requires at least k input points");
    if (!covariances || !status)
        throw std::invalid_argument(
            "k-nearest covariance outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnCovariancesDevice(index, neighbors, covariances, status);
        return;
    }

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    knnGather(index, neighbors, pointIds.data(), squaredDistances.data(),
              status);
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    const std::array<const double*, 3> coordinates{x, y, z};
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        if (pointIds[row + neighbors - 1U] ==
            (std::numeric_limits<std::uint32_t>::max)())
        {
            const double missing = (std::numeric_limits<double>::quiet_NaN)();
            covariances[query] = {missing, missing, missing,
                                  missing, missing, missing};
            continue;
        }
        const Eigen::Matrix3d matrix =
            detail::pinnedCovariance(coordinates, pointIds.data() + row,
                                     neighbors);
        covariances[query] = {matrix(0, 0), matrix(0, 1), matrix(0, 2),
                              matrix(1, 1), matrix(1, 2), matrix(2, 2)};
    }
}

void knnRankValues(const SpatialIndex& index, std::uint32_t neighbors,
                   double threshold, std::uint8_t* ranks, std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (neighbors < 3U)
        throw std::invalid_argument(
            "k-nearest rank requires at least three neighbors");
    if (index.m_config.dimensions != 3U)
        throw std::invalid_argument(
            "k-nearest rank requires a three-dimensional index");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "k-nearest rank requires at least k input points");
    if (!ranks || !status)
        throw std::invalid_argument("k-nearest rank outputs cannot be null");

    // The covariances carry PDAL's exact float-demeaned arithmetic on either
    // memory kind; the decomposition itself stays on the host because the
    // oracle's Eigen JacobiSVD is host-only, and bit-identical rank demands
    // the oracle's own solver. Rows are independent, so the host pass
    // parallelizes without changing a bit. `ranks` and `status` are
    // host-visible on both paths.
    std::vector<Covariance3d> covariances(batch.size());
    if (batch.memoryKind() == MemoryKind::Device)
        detail::knnRankCovariancesDevice(index, neighbors, covariances.data(),
                                         status);
    else
        knnCovariances(index, neighbors, covariances.data(), status);

    const auto rankRange = [&](std::size_t begin, std::size_t end)
    {
        for (std::size_t point = begin; point < end; ++point)
        {
            if ((status[point] & KnnSearchIncomplete) != 0U)
            {
                ranks[point] = 0U;
                continue;
            }
            const Covariance3d& covariance = covariances[point];
            Eigen::Matrix3d matrix;
            matrix(0, 0) = covariance.xx;
            matrix(0, 1) = covariance.xy;
            matrix(0, 2) = covariance.xz;
            matrix(1, 0) = covariance.xy;
            matrix(1, 1) = covariance.yy;
            matrix(1, 2) = covariance.yz;
            matrix(2, 0) = covariance.xz;
            matrix(2, 1) = covariance.yz;
            matrix(2, 2) = covariance.zz;
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(matrix);
            svd.setThreshold(static_cast<float>(threshold));
            ranks[point] = static_cast<std::uint8_t>(svd.rank());
        }
    };
    const std::size_t workers =
        (std::min<
            std::size_t>)((std::max<
                              std::size_t>)(std::thread::hardware_concurrency(),
                                            1U),
                          (std::max<std::size_t>)(batch.size() / 65536U, 1U));
    if (workers <= 1U)
        rankRange(0, batch.size());
    else
    {
        std::vector<std::thread> threads;
        const std::size_t chunk = (batch.size() + workers - 1U) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker)
        {
            const std::size_t begin = worker * chunk;
            const std::size_t end = (std::min)(begin + chunk, batch.size());
            if (begin < end)
                threads.emplace_back(rankRange, begin, end);
        }
        for (std::thread& thread : threads)
            thread.join();
    }
}

void knnNeighborVotes(const SpatialIndex& index, std::uint32_t neighbors,
                      const std::uint8_t* values, std::uint8_t* results,
                      std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (index.m_config.dimensions != 3U)
        throw std::invalid_argument(
            "neighbor vote requires a three-dimensional index");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "neighbor vote requires at least k input points");
    if (!values || !results || !status)
        throw std::invalid_argument("neighbor vote buffers cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnNeighborVotesDevice(index, neighbors, values, results,
                                       status);
        return;
    }

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    knnGather(index, neighbors, pointIds.data(), squaredDistances.data(),
              status);
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        const std::uint8_t oldValue = values[query];
        if ((status[query] & KnnSearchIncomplete) != 0U)
        {
            results[query] = oldValue;
            continue;
        }
        const std::size_t row = query * static_cast<std::size_t>(neighbors);
        std::uint32_t bestCount = 0U;
        std::uint32_t bestValue = 0U;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t candidate = values[pointIds[row + item]];
            std::uint32_t count = 0U;
            for (std::uint32_t other = 0; other < neighbors; ++other)
                count += static_cast<std::uint32_t>(
                    values[pointIds[row + other]] == candidate);
            if (count > bestCount ||
                (count == bestCount && candidate < bestValue))
            {
                bestCount = count;
                bestValue = candidate;
            }
        }
        const bool apply = 2U * bestCount > neighbors &&
                           bestValue != static_cast<std::uint32_t>(oldValue);
        results[query] =
            apply ? static_cast<std::uint8_t>(bestValue) : oldValue;
    }
}

void knnOptimalValues(const SpatialIndex& index, std::uint32_t minimumK,
                      std::uint32_t maximumK, std::uint64_t* optimalK,
                      double* optimalRadius, std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(maximumK);
    if (minimumK < 1U || minimumK > maximumK)
        throw std::invalid_argument(
            "optimal neighborhood requires 1 <= min_k <= max_k");
    if (index.m_config.dimensions != 3U)
        throw std::invalid_argument(
            "optimal neighborhood requires a three-dimensional index");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (batch.size() < static_cast<std::size_t>(maximumK))
        throw std::invalid_argument(
            "optimal neighborhood requires at least max_k input points");
    if (!optimalK || !optimalRadius || !status)
        throw std::invalid_argument(
            "optimal neighborhood outputs cannot be null");

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(maximumK);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    if (batch.memoryKind() == MemoryKind::Device)
        detail::knnAdjacencyHostDevice(index, maximumK, pointIds.data(),
                                       squaredDistances.data(), status);
    else
        knnGather(index, maximumK, pointIds.data(), squaredDistances.data(),
                  status);

    // Verbatim transcription of upstream's Welford covariance sweep and
    // host-transcendental eigenentropy selection; rows are independent, so
    // the worker split is bit-exact.
    const double* x = batch.memoryKind() == MemoryKind::Device
                          ? nullptr
                          : batch.data<double>(X);
    const double* y = x ? batch.data<double>(Y) : nullptr;
    const double* z = x ? batch.data<double>(Z) : nullptr;
    std::vector<double> hostX;
    std::vector<double> hostY;
    std::vector<double> hostZ;
    if (!x)
    {
        hostX.resize(batch.size());
        hostY.resize(batch.size());
        hostZ.resize(batch.size());
        detail::copyCoordinateColumnsToHost(index, hostX.data(), hostY.data(),
                                            hostZ.data());
        x = hostX.data();
        y = hostY.data();
        z = hostZ.data();
    }
    const auto sweepRange = [&](std::size_t begin, std::size_t end)
    {
        for (std::size_t point = begin; point < end; ++point)
        {
            if ((status[point] & KnnSearchIncomplete) != 0U)
            {
                optimalK[point] = 0U;
                optimalRadius[point] = 0.0;
                continue;
            }
            const std::size_t row = point * static_cast<std::size_t>(maximumK);
            double minimumEntropy = (std::numeric_limits<double>::max)();
            std::uint64_t bestK = 0U;
            double bestSquared = 0.0;
            double mx = 0.0;
            double my = 0.0;
            double mz = 0.0;
            Eigen::Matrix3d accumulated = Eigen::Matrix3d::Zero(3, 3);
            const auto update = [&](std::size_t item)
            {
                const std::uint32_t q = pointIds[row + item];
                const double dx = x[q] - mx;
                const double dy = y[q] - my;
                const double dz = z[q] - mz;
                const double n = double(item + 1);
                mx += dx / n;
                my += dy / n;
                mz += dz / n;
                const double scale = (n - 1) / n;
                accumulated(0, 0) = accumulated(0, 0) + scale * dx * dx;
                accumulated(1, 1) = accumulated(1, 1) + scale * dy * dy;
                accumulated(2, 2) = accumulated(2, 2) + scale * dz * dz;
                accumulated(1, 0) = accumulated(0, 1) =
                    accumulated(0, 1) + scale * dx * dy;
                accumulated(2, 0) = accumulated(0, 2) =
                    accumulated(0, 2) + scale * dx * dz;
                accumulated(1, 2) = accumulated(2, 1) =
                    accumulated(2, 1) + scale * dy * dz;
                return n;
            };
            for (std::size_t k = 0; k + 1U < minimumK; ++k)
                update(k);
            for (std::size_t k = minimumK - 1U; k < maximumK; ++k)
            {
                const double n = update(k);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(
                    accumulated / (n - 1));
                if (solver.info() != Eigen::Success)
                    throw std::runtime_error(
                        "Cannot perform eigen decomposition.");
                const Eigen::Vector3d values = solver.eigenvalues();
                double lambda2 = (std::max)(values[2], 0.0);
                double lambda1 = (std::max)(values[1], 0.0);
                double lambda0 = (std::max)(values[0], 0.0);
                const double sum = lambda2 + lambda1 + lambda0;
                lambda2 /= sum;
                lambda1 /= sum;
                lambda0 /= sum;
                const double entropy = -(lambda0 * std::log(lambda0) +
                                         lambda1 * std::log(lambda1) +
                                         lambda2 * std::log(lambda2));
                if (entropy < minimumEntropy)
                {
                    minimumEntropy = entropy;
                    bestK = static_cast<std::uint64_t>(k + 1U);
                    bestSquared = squaredDistances[row + k];
                }
            }
            optimalK[point] = bestK;
            optimalRadius[point] = std::sqrt(bestSquared);
        }
    };
    const std::size_t workers =
        (std::min<
            std::size_t>)((std::max<
                              std::size_t>)(std::thread::hardware_concurrency(),
                                            1U),
                          (std::max<std::size_t>)(batch.size() / 16384U, 1U));
    if (workers <= 1U)
        sweepRange(0, batch.size());
    else
    {
        std::vector<std::thread> threads;
        const std::size_t chunk = (batch.size() + workers - 1U) / workers;
        for (std::size_t worker = 0; worker < workers; ++worker)
        {
            const std::size_t begin = worker * chunk;
            const std::size_t end = (std::min)(begin + chunk, batch.size());
            if (begin < end)
                threads.emplace_back(sweepRange, begin, end);
        }
        for (std::thread& thread : threads)
            thread.join();
    }
}

void knnLofValues(const SpatialIndex& index, std::uint32_t neighbors,
                  double* kDistances, double* reachabilityDensities,
                  double* outlierFactors, std::uint8_t* status,
                  std::uint8_t* neighborStatus,
                  std::uint8_t* neighborNeighborStatus)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (neighbors < 2U)
        throw std::invalid_argument(
            "local outlier factor requires at least one non-query neighbor");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() == 0)
        return;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "local outlier factor requires at least k + 1 input points");
    if (!kDistances || !reachabilityDensities || !outlierFactors || !status ||
        !neighborStatus || !neighborNeighborStatus)
        throw std::invalid_argument(
            "local outlier factor outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnLofValuesDevice(
            index, neighbors, kDistances, reachabilityDensities, outlierFactors,
            status, neighborStatus, neighborNeighborStatus);
        return;
    }

    const std::size_t resultCount =
        batch.size() * static_cast<std::size_t>(neighbors);
    std::vector<std::uint32_t> pointIds(resultCount);
    std::vector<double> squaredDistances(resultCount);
    knnGather(index, neighbors, pointIds.data(), squaredDistances.data(),
              status);
    // An incomplete row carries sentinel neighbor entries and must not be
    // dereferenced; its consumer repairs it from its own status bit. The
    // values written here for such rows are placeholders.
    const auto complete = [&](std::size_t point)
    { return (status[point] & KnnSearchIncomplete) == 0U; };
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        kDistances[point] =
            complete(point) ? std::sqrt(squaredDistances[row + neighbors - 1U])
                            : (std::numeric_limits<double>::quiet_NaN)();
    }
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        if (!complete(point))
        {
            reachabilityDensities[point] =
                (std::numeric_limits<double>::quiet_NaN)();
            continue;
        }
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        double mean = 0.0;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const double distance = std::sqrt(squaredDistances[row + item]);
            const double kDistance = kDistances[pointIds[row + item]];
            // std::max: the second argument wins only when strictly greater.
            const double reachability =
                kDistance < distance ? distance : kDistance;
            mean += (reachability - mean) /
                    static_cast<double>(static_cast<std::size_t>(item) + 1U);
        }
        reachabilityDensities[point] = 1.0 / mean;
    }
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        if (!complete(point))
        {
            outlierFactors[point] = (std::numeric_limits<double>::quiet_NaN)();
            neighborStatus[point] = KnnSearchIncomplete;
            continue;
        }
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        const double density = reachabilityDensities[point];
        double mean = 0.0;
        std::uint8_t observed = KnnExact;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t neighbor = pointIds[row + item];
            observed = static_cast<std::uint8_t>(observed | status[neighbor]);
            const double ratio = reachabilityDensities[neighbor] / density;
            mean += (ratio - mean) /
                    static_cast<double>(static_cast<std::size_t>(item) + 1U);
        }
        outlierFactors[point] = mean;
        neighborStatus[point] = observed;
    }
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        if (!complete(point))
        {
            neighborNeighborStatus[point] = KnnSearchIncomplete;
            continue;
        }
        const std::size_t row = point * static_cast<std::size_t>(neighbors);
        std::uint8_t observed = KnnExact;
        for (std::uint32_t item = 0; item < neighbors; ++item)
        {
            const std::uint32_t neighbor = pointIds[row + item];
            observed = static_cast<std::uint8_t>(observed | status[neighbor] |
                                                 neighborStatus[neighbor]);
        }
        neighborNeighborStatus[point] = observed;
    }
}

void knnEigenSystems(const SpatialIndex& index, std::uint32_t neighbors,
                     EigenSystem3d* systems, std::uint8_t* status)
{
    if (!index.m_valid)
        throw std::logic_error("spatial index must be built before querying");
    validateKnnNeighbors(neighbors);
    if (neighbors < 3U)
        throw std::invalid_argument(
            "k-nearest eigensystem requires at least three neighbors");
    if (index.m_config.dimensions != 3U)
        throw std::invalid_argument(
            "k-nearest eigensystem requires a three-dimensional index");
    const PointBatch& batch = *index.m_batch;
    if (batch.size() < static_cast<std::size_t>(neighbors))
        throw std::invalid_argument(
            "k-nearest eigensystem requires at least k input points");
    if (!systems || !status)
        throw std::invalid_argument(
            "k-nearest eigensystem outputs cannot be null");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        detail::knnEigenSystemsDevice(index, neighbors, systems, status);
        return;
    }

    std::vector<Covariance3d> covariances(batch.size());
    knnCovariances(index, neighbors, covariances.data(), status);
    for (std::size_t query = 0; query < batch.size(); ++query)
    {
        systems[query] = {};
        if ((status[query] & KnnSearchIncomplete) != 0U)
            continue;
        const Covariance3d& covariance = covariances[query];
        Eigen::Matrix3d matrix;
        matrix(0, 0) = covariance.xx;
        matrix(0, 1) = covariance.xy;
        matrix(0, 2) = covariance.xz;
        matrix(1, 0) = covariance.xy;
        matrix(1, 1) = covariance.yy;
        matrix(1, 2) = covariance.yz;
        matrix(2, 0) = covariance.xz;
        matrix(2, 1) = covariance.yz;
        matrix(2, 2) = covariance.zz;
        if (matrix.isZero())
        {
            status[query] =
                static_cast<std::uint8_t>(status[query] | KnnCovarianceZero);
            continue;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(matrix);
        if (solver.info() != Eigen::Success)
        {
            status[query] =
                static_cast<std::uint8_t>(status[query] | KnnEigenFailure);
            continue;
        }
        for (std::size_t eigen = 0; eigen < 3; ++eigen)
        {
            systems[query].values[eigen] =
                solver.eigenvalues()[static_cast<Eigen::Index>(eigen)];
            for (std::size_t axis = 0; axis < 3; ++axis)
                systems[query].vectors[axis * 3U + eigen] =
                    solver.eigenvectors()(static_cast<Eigen::Index>(axis),
                                          static_cast<Eigen::Index>(eigen));
        }
    }
}

} // namespace pdg
