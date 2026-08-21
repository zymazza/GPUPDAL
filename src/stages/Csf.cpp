/******************************************************************************
 * CSF compatibility implementation derived from PDAL's BSD-licensed wrapper
 * and the Apache-2.0 CSF cloth implementation under filters/private/csf.
 * Copyright (c) 2017 State Key Laboratory of Remote Sensing Science,
 * Institute of Remote Sensing Science and Engineering,
 * Beijing Normal University
 * Copyright (c) 2019 Bradley J Chambers
 * Copyright (c) 2026 PDAL-GPU contributors
 ******************************************************************************/

#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
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
constexpr double MinimumHeight = -9999999999.0;
constexpr double MaximumDistance = 9999999999.0;
constexpr double Damping = 0.01;
constexpr double Gravity = 0.2;
constexpr int BufferCells = 2;

constexpr double SingleMove[15] = {0.0,     0.3,     0.51,    0.657,   0.7599,
                                   0.83193, 0.88235, 0.91765, 0.94235, 0.95965,
                                   0.97175, 0.98023, 0.98616, 0.99031, 0.99322};
constexpr double DoubleMove[15] = {0.0,    0.3,   0.42,   0.468,  0.4872,
                                   0.4949, 0.498, 0.4992, 0.4997, 0.4999,
                                   0.4999, 0.5,   0.5,    0.5,    0.5};

struct ClothFrame
{
    double originX = 0.0;
    double originY = 0.0;
    double originZ = 0.0;
    std::size_t width = 0U;
    std::size_t height = 0U;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return width * height;
    }
};

struct Particle
{
    double position = 0.0;
    double oldPosition = 0.0;
    double nearestHeight = MinimumHeight;
    double temporaryDistance = MaximumDistance;
    bool movable = true;
    bool visited = false;
};

bool programValid(const CsfProgram& program) noexcept
{
    return CsfPinnedOracleHasSerialExecution && !program.smooth &&
           std::isfinite(program.timeStep) &&
           std::isfinite(program.classThreshold) &&
           std::isfinite(program.heightThreshold) &&
           std::isfinite(program.resolution) && program.resolution > 0.0 &&
           program.rigidness >= 0 && program.iterations >= 0 &&
           program.iterations <= CsfExactDeviceMaximumIterations &&
           (program.onlyGround || program.groundClass != program.otherClass);
}

ClothFrame frameFor(const PointBatch& batch, const CsfProgram& program)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    double minimumX = x[0];
    double maximumX = x[0];
    double minimumY = y[0];
    double maximumY = y[0];
    double minimumInvertedZ = -z[0];
    double maximumInvertedZ = -z[0];
    for (std::size_t point = 1U; point < batch.size(); ++point)
    {
        if (x[point] < minimumX)
            minimumX = x[point];
        else if (x[point] > maximumX)
            maximumX = x[point];
        if (y[point] < minimumY)
            minimumY = y[point];
        else if (y[point] > maximumY)
            maximumY = y[point];
        const double invertedZ = -z[point];
        if (invertedZ < minimumInvertedZ)
            minimumInvertedZ = invertedZ;
        else if (invertedZ > maximumInvertedZ)
            maximumInvertedZ = invertedZ;
    }
    const double widthValue =
        std::floor((maximumX - minimumX) / program.resolution) +
        2.0 * BufferCells;
    const double heightValue =
        std::floor((maximumY - minimumY) / program.resolution) +
        2.0 * BufferCells;
    if (!std::isfinite(widthValue) || !std::isfinite(heightValue) ||
        widthValue < 1.0 || heightValue < 1.0 ||
        widthValue > static_cast<double>((std::numeric_limits<int>::max)()) ||
        heightValue > static_cast<double>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("CSF cloth dimensions are invalid");
    ClothFrame frame{minimumX - BufferCells * program.resolution,
                     maximumInvertedZ + 0.05,
                     minimumY - BufferCells * program.resolution,
                     static_cast<std::size_t>(widthValue),
                     static_cast<std::size_t>(heightValue)};
    if (!std::isfinite(frame.originX) || !std::isfinite(frame.originY) ||
        !std::isfinite(frame.originZ) ||
        frame.width > (std::numeric_limits<std::size_t>::max)() / frame.height)
        throw std::invalid_argument("CSF cloth frame is invalid");
    return frame;
}

std::size_t cell(std::size_t column, std::size_t row,
                 const ClothFrame& frame) noexcept
{
    return row * frame.width + column;
}

void addConstraint(std::vector<std::vector<std::size_t>>& neighbors,
                   std::size_t first, std::size_t second)
{
    neighbors[first].push_back(second);
    neighbors[second].push_back(first);
}

std::vector<std::vector<std::size_t>> makeNeighbors(const ClothFrame& frame)
{
    std::vector<std::vector<std::size_t>> neighbors(frame.size());
    for (std::size_t column = 0U; column < frame.width; ++column)
        for (std::size_t row = 0U; row < frame.height; ++row)
        {
            const std::size_t current = cell(column, row, frame);
            if (column + 1U < frame.width)
                addConstraint(neighbors, current,
                              cell(column + 1U, row, frame));
            if (row + 1U < frame.height)
                addConstraint(neighbors, current,
                              cell(column, row + 1U, frame));
            if (column + 1U < frame.width && row + 1U < frame.height)
                addConstraint(neighbors, current,
                              cell(column + 1U, row + 1U, frame));
            if (column + 1U < frame.width && row + 1U < frame.height)
                addConstraint(neighbors, cell(column + 1U, row, frame),
                              cell(column, row + 1U, frame));
        }
    for (std::size_t column = 0U; column < frame.width; ++column)
        for (std::size_t row = 0U; row < frame.height; ++row)
        {
            const std::size_t current = cell(column, row, frame);
            if (column + 2U < frame.width)
                addConstraint(neighbors, current,
                              cell(column + 2U, row, frame));
            if (row + 2U < frame.height)
                addConstraint(neighbors, current,
                              cell(column, row + 2U, frame));
            if (column + 2U < frame.width && row + 2U < frame.height)
                addConstraint(neighbors, current,
                              cell(column + 2U, row + 2U, frame));
            if (column + 2U < frame.width && row + 2U < frame.height)
                addConstraint(neighbors, cell(column + 2U, row, frame),
                              cell(column, row + 2U, frame));
        }
    return neighbors;
}

double
fillByBreadthFirst(std::size_t root, std::vector<Particle>& particles,
                   const std::vector<std::vector<std::size_t>>& neighbors)
{
    std::deque<std::size_t> queue;
    std::vector<std::size_t> backlist;
    for (const std::size_t neighbor : neighbors[root])
    {
        particles[root].visited = true;
        queue.push_back(neighbor);
    }
    while (!queue.empty())
    {
        const std::size_t current = queue.front();
        queue.pop_front();
        backlist.push_back(current);
        if (particles[current].nearestHeight > MinimumHeight)
        {
            for (const std::size_t visited : backlist)
                particles[visited].visited = false;
            while (!queue.empty())
            {
                particles[queue.front()].visited = false;
                queue.pop_front();
            }
            return particles[current].nearestHeight;
        }
        for (const std::size_t neighbor : neighbors[current])
            if (!particles[neighbor].visited)
            {
                particles[neighbor].visited = true;
                queue.push_back(neighbor);
            }
    }
    return MinimumHeight;
}

double fillHeight(std::size_t index, std::vector<Particle>& particles,
                  const ClothFrame& frame,
                  const std::vector<std::vector<std::size_t>>& neighbors)
{
    const std::size_t column = index % frame.width;
    const std::size_t row = index / frame.width;
    for (std::size_t candidate = column + 1U; candidate < frame.width;
         ++candidate)
        if (particles[cell(candidate, row, frame)].nearestHeight >
            MinimumHeight)
            return particles[cell(candidate, row, frame)].nearestHeight;
    for (std::size_t candidate = column; candidate-- > 0U;)
        if (particles[cell(candidate, row, frame)].nearestHeight >
            MinimumHeight)
            return particles[cell(candidate, row, frame)].nearestHeight;
    for (std::size_t candidate = row; candidate-- > 0U;)
        if (particles[cell(column, candidate, frame)].nearestHeight >
            MinimumHeight)
            return particles[cell(column, candidate, frame)].nearestHeight;
    for (std::size_t candidate = row + 1U; candidate < frame.height;
         ++candidate)
        if (particles[cell(column, candidate, frame)].nearestHeight >
            MinimumHeight)
            return particles[cell(column, candidate, frame)].nearestHeight;
    return fillByBreadthFirst(index, particles, neighbors);
}

void rasterize(const PointBatch& batch, const CsfProgram& program,
               const ClothFrame& frame, std::vector<Particle>& particles,
               const std::vector<std::vector<std::size_t>>& neighbors,
               std::vector<double>& heights)
{
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const double deltaX = x[point] - frame.originX;
        const double deltaZ = y[point] - frame.originZ;
        const int column = static_cast<int>(deltaX / program.resolution + 0.5);
        const int row = static_cast<int>(deltaZ / program.resolution + 0.5);
        if (column < 0 || row < 0 ||
            static_cast<std::size_t>(column) >= frame.width ||
            static_cast<std::size_t>(row) >= frame.height)
            throw std::out_of_range("CSF point lies outside its cloth raster");
        const std::size_t index = cell(static_cast<std::size_t>(column),
                                       static_cast<std::size_t>(row), frame);
        const double particleX =
            frame.originX + static_cast<double>(column) * program.resolution;
        const double particleZ =
            frame.originZ + static_cast<double>(row) * program.resolution;
        const double deltaParticleX = x[point] - particleX;
        const double deltaParticleZ = y[point] - particleZ;
        const double distance =
            deltaParticleX * deltaParticleX + deltaParticleZ * deltaParticleZ;
        if (distance < particles[index].temporaryDistance)
        {
            particles[index].temporaryDistance = distance;
            particles[index].nearestHeight = -z[point];
        }
    }
    heights.resize(frame.size());
    for (std::size_t index = 0U; index < frame.size(); ++index)
        heights[index] = particles[index].nearestHeight > MinimumHeight
                             ? particles[index].nearestHeight
                             : fillHeight(index, particles, frame, neighbors);
}

std::size_t simulate(const CsfProgram& program,
                     const std::vector<std::vector<std::size_t>>& neighbors,
                     const std::vector<double>& heights,
                     std::vector<Particle>& particles)
{
    const double stepSquared = program.timeStep * program.timeStep;
    const double acceleration = -Gravity * stepSquared;
    const double doubleCoefficient =
        program.rigidness > 14 ? 0.5 : DoubleMove[program.rigidness];
    const double singleCoefficient =
        program.rigidness > 14 ? 1.0 : SingleMove[program.rigidness];
    std::size_t executed = 0U;
    for (int iteration = 0; iteration < program.iterations; ++iteration)
    {
        ++executed;
        for (Particle& particle : particles)
            if (particle.movable)
            {
                const double previous = particle.position;
                const double velocity =
                    (particle.position - particle.oldPosition) *
                    (1.0 - Damping);
                const double force = acceleration * stepSquared;
                particle.position = particle.position + velocity + force;
                particle.oldPosition = previous;
            }
        for (std::size_t first = 0U; first < particles.size(); ++first)
            for (const std::size_t second : neighbors[first])
            {
                const double correction =
                    particles[second].position - particles[first].position;
                if (particles[first].movable && particles[second].movable)
                {
                    const double half = correction * doubleCoefficient;
                    particles[first].position += half;
                    particles[second].position += -half;
                }
                else if (particles[first].movable && !particles[second].movable)
                    particles[first].position += correction * singleCoefficient;
                else if (!particles[first].movable && particles[second].movable)
                    particles[second].position +=
                        -(correction * singleCoefficient);
            }
        double maximumDifference = 0.0;
        for (const Particle& particle : particles)
            if (particle.movable)
            {
                const double difference =
                    std::fabs(particle.oldPosition - particle.position);
                if (difference > maximumDifference)
                    maximumDifference = difference;
            }
        for (std::size_t index = 0U; index < particles.size(); ++index)
            if (particles[index].position < heights[index])
            {
                if (particles[index].movable)
                    particles[index].position +=
                        heights[index] - particles[index].position;
                particles[index].movable = false;
            }
        if (maximumDifference != 0.0 && maximumDifference < 0.005)
            break;
    }
    return executed;
}

CsfResult classifyHost(PointBatch& batch, const CsfProgram& program)
{
    const ClothFrame frame = frameFor(batch, program);
    std::vector<Particle> particles(frame.size());
    for (Particle& particle : particles)
    {
        particle.position = frame.originY;
        particle.oldPosition = frame.originY;
    }
    const std::vector<std::vector<std::size_t>> neighbors =
        makeNeighbors(frame);
    std::vector<double> heights;
    rasterize(batch, program, frame, particles, neighbors, heights);
    const std::size_t executed =
        simulate(program, neighbors, heights, particles);

    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    auto* classification = batch.data<std::uint8_t>(Classification);
    CsfResult result{frame.width, frame.height, executed, 0U, 0U};
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const double pointY = -z[point];
        const double deltaX = x[point] - frame.originX;
        const double deltaZ = y[point] - frame.originZ;
        const int column0 = static_cast<int>(deltaX / program.resolution);
        const int row0 = static_cast<int>(deltaZ / program.resolution);
        if (column0 < 0 || row0 < 0 ||
            static_cast<std::size_t>(column0 + 1) >= frame.width ||
            static_cast<std::size_t>(row0 + 1) >= frame.height)
            throw std::out_of_range(
                "CSF point lies outside its classification raster");
        const double subdeltaX =
            (deltaX - column0 * program.resolution) / program.resolution;
        const double subdeltaZ =
            (deltaZ - row0 * program.resolution) / program.resolution;
        const std::size_t c0 = static_cast<std::size_t>(column0);
        const std::size_t r0 = static_cast<std::size_t>(row0);
        const double interpolated =
            particles[cell(c0, r0, frame)].position * (1.0 - subdeltaX) *
                (1.0 - subdeltaZ) +
            particles[cell(c0, r0 + 1U, frame)].position * (1.0 - subdeltaX) *
                subdeltaZ +
            particles[cell(c0 + 1U, r0 + 1U, frame)].position * subdeltaX *
                subdeltaZ +
            particles[cell(c0 + 1U, r0, frame)].position * subdeltaX *
                (1.0 - subdeltaZ);
        const bool ground =
            std::fabs(interpolated - pointY) < program.classThreshold;
        if (ground)
        {
            classification[point] = program.groundClass;
            ++result.groundPoints;
        }
        else
        {
            if (!program.onlyGround)
                classification[point] = program.otherClass;
            ++result.nongroundPoints;
        }
    }
    return result;
}
} // unnamed namespace

CsfResult classifyCsfDevice(PointBatch& batch, const CsfProgram& program);

bool csfProgramWithinExactDeviceEnvelope(const CsfProgram& program) noexcept
{
    return programValid(program);
}

bool csfSupportsExactDevice(const PointBatch& hostBatch,
                            const CsfProgram& program) noexcept
{
    if (!programValid(program) ||
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
                DimensionType::Unsigned8 ||
            hostBatch.size() >
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            return false;
        const double* x = hostBatch.data<double>(X);
        const double* y = hostBatch.data<double>(Y);
        const double* z = hostBatch.data<double>(Z);
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
            if (!std::isfinite(x[point]) || !std::isfinite(y[point]) ||
                !std::isfinite(z[point]))
                return false;
        const ClothFrame frame = frameFor(hostBatch, program);
        return frame.size() <= CsfExactDeviceMaximumClothCells;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

CsfResult classifyCsf(PointBatch& batch, const CsfProgram& program)
{
    if (!programValid(program))
        throw std::invalid_argument("invalid CSF program");
    if (!batch.size())
        throw std::invalid_argument("CSF requires a nonempty point batch");
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        !batch.has(Classification))
        throw std::invalid_argument("CSF input columns are not materialized");
    if (batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double ||
        batch.columnInfo(Classification).physicalType !=
            DimensionType::Unsigned8)
        throw std::invalid_argument("CSF input columns have invalid types");
    if (batch.memoryKind() == MemoryKind::Device)
        return classifyCsfDevice(batch, program);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported CSF memory kind");
    return classifyHost(batch, program);
}

} // namespace pdg
