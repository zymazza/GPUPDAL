/******************************************************************************
 * SMRF compatibility implementation derived from PDAL's BSD-licensed
 * filters.smrf and MathUtils morphology code.
 * Copyright (c) 2016-2017, 2020 Bradley J Chambers
 * Copyright (c) 2026 PDAL-GPU contributors
 ******************************************************************************/

#include <pdg/PointBatch.hpp>
#include <pdg/stages/Smrf.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
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
constexpr DimensionId Classification(StandardDimension::Classification);

struct RasterFrame
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

struct RasterNeighbor
{
    std::uint64_t squaredDistance = 0U;
    std::size_t cell = 0U;
};

bool programValid(const SmrfProgram& program) noexcept
{
    return std::isfinite(program.cell) && program.cell > 0.0 &&
           std::isfinite(program.slope) && std::isfinite(program.window) &&
           program.window >= 0.0 && std::isfinite(program.scalar) &&
           std::isfinite(program.threshold) && std::isfinite(program.cut) &&
           program.cut >= 0.0 &&
           (program.onlyGround || program.groundClass != program.otherClass);
}

RasterFrame frameFor(const PointBatch& batch, const SmrfProgram& program)
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
    const double columns = (maximumX - minimumX) / program.cell + 1.0;
    const double rows = (maximumY - minimumY) / program.cell + 1.0;
    if (!std::isfinite(columns) || !std::isfinite(rows) || columns < 1.0 ||
        rows < 1.0 ||
        columns > static_cast<double>((std::numeric_limits<int>::max)()) ||
        rows > static_cast<double>((std::numeric_limits<int>::max)()))
        throw std::invalid_argument("SMRF raster dimensions are invalid");
    RasterFrame frame{minimumX, minimumY, static_cast<std::size_t>(rows),
                      static_cast<std::size_t>(columns)};
    if (frame.columns != 0U &&
        frame.rows > (std::numeric_limits<std::size_t>::max)() / frame.columns)
        throw std::overflow_error("SMRF raster cell count overflows");
    return frame;
}

std::size_t rasterCell(double x, double y, const RasterFrame& frame,
                       double cell)
{
    const auto column =
        static_cast<std::size_t>(std::floor((x - frame.minimumX) / cell));
    const auto row =
        static_cast<std::size_t>(std::floor((y - frame.minimumY) / cell));
    if (column >= frame.columns || row >= frame.rows)
        throw std::out_of_range("SMRF point lies outside its raster frame");
    return column * frame.rows + row;
}

// B0265/D0265: a minimal persistent pass pool for the diamond morphology.
// Each pass evaluates the exact serial fold over disjoint column ranges;
// every output cell depends only on the pass's read-only input and its own
// carried-over start value, so any worker count yields the serial bits. The
// engine core does not link the PDAL library, so this small pool replaces
// pdal::ThreadPool here. Measured cap of eight workers as in B0264.
class PassPool
{
public:
    explicit PassPool(std::size_t workers) : m_workers(workers)
    {
        m_threads.reserve(workers);
        for (std::size_t worker = 0U; worker < workers; ++worker)
            m_threads.emplace_back([this, worker] { loop(worker); });
    }

    ~PassPool()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
            ++m_generation;
        }
        m_wake.notify_all();
        for (std::thread& thread : m_threads)
            thread.join();
    }

    std::size_t size() const noexcept { return m_workers; }

    // Runs task(worker) on every worker and returns when all have finished.
    template <typename Task>
    void run(Task&& task)
    {
        std::function<void(std::size_t)> bound = std::forward<Task>(task);
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_task = std::move(bound);
            m_pending = m_workers;
            ++m_generation;
        }
        m_wake.notify_all();
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done.wait(lock, [this] { return m_pending == 0U; });
        m_task = nullptr;
    }

private:
    void loop(std::size_t worker)
    {
        std::uint64_t seen = 0U;
        for (;;)
        {
            std::function<void(std::size_t)> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, [&] { return m_generation != seen; });
                seen = m_generation;
                if (m_stop)
                    return;
                task = m_task;
            }
            if (task)
                task(worker);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (--m_pending == 0U)
                    m_done.notify_all();
            }
        }
    }

    std::size_t m_workers;
    std::vector<std::thread> m_threads;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    std::function<void(std::size_t)> m_task;
    std::uint64_t m_generation = 0U;
    std::size_t m_pending = 0U;
    bool m_stop = false;
};

constexpr std::size_t MorphologyPoolCap = 8U;
constexpr std::size_t MorphologyMinimumCellsPerTask = 16384U;

std::unique_ptr<PassPool> morphologyPool(std::size_t cells)
{
    if (cells < 2U * MorphologyMinimumCellsPerTask ||
        std::getenv("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS"))
        return nullptr;
    std::size_t workers = (std::min)(
        MorphologyPoolCap,
        (std::max<std::size_t>)(1U, std::thread::hardware_concurrency()));
    if (const char* configured = std::getenv("PDG_NATIVE_WORKERS");
        configured && *configured)
    {
        char* end = nullptr;
        const unsigned long cap = std::strtoul(configured, &end, 10);
        if (end && !*end && cap != 0UL)
            workers = (std::min)(workers, static_cast<std::size_t>(cap));
    }
    workers = (std::min)(workers, cells / MorphologyMinimumCellsPerTask);
    return workers > 1U ? std::make_unique<PassPool>(workers) : nullptr;
}

template <typename Better>
void diamondPassColumns(const std::vector<double>& data,
                        std::vector<double>& output, std::size_t rows,
                        std::size_t columns, std::size_t columnBegin,
                        std::size_t columnEnd, Better better)
{
    std::array<std::size_t, 5> neighbors{};
    for (std::size_t column = columnBegin; column < columnEnd; ++column)
    {
        const std::size_t offset = column * rows;
        for (std::size_t row = 0; row < rows; ++row)
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
                if (better(data[neighbors[index]], output[offset + row]))
                    output[offset + row] = data[neighbors[index]];
        }
    }
}

template <typename Better>
void diamondPasses(std::vector<double>& data, std::vector<double>& output,
                   std::size_t rows, std::size_t columns, int iterations,
                   PassPool* pool, Better better)
{
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        if (!pool)
            diamondPassColumns(data, output, rows, columns, 0U, columns,
                               better);
        else
        {
            const std::size_t workers = pool->size();
            const std::size_t columnsPerWorker = columns / workers;
            const std::size_t remainder = columns % workers;
            pool->run([&, workers, columnsPerWorker, remainder](std::size_t w)
            {
                const std::size_t begin =
                    w * columnsPerWorker + (std::min)(w, remainder);
                const std::size_t end = begin + columnsPerWorker +
                                        static_cast<std::size_t>(w < remainder);
                diamondPassColumns(data, output, rows, columns, begin, end,
                                   better);
            });
        }
        data.swap(output);
    }
}

void dilateDiamond(std::vector<double>& data, std::size_t rows,
                   std::size_t columns, int iterations,
                   PassPool* pool = nullptr)
{
    std::vector<double> output(data.size(),
                               std::numeric_limits<double>::lowest());
    diamondPasses(data, output, rows, columns, iterations, pool,
                  [](double candidate, double current)
                  { return candidate > current; });
}

void erodeDiamond(std::vector<double>& data, std::size_t rows,
                  std::size_t columns, int iterations,
                  PassPool* pool = nullptr)
{
    std::vector<double> output(data.size(),
                               (std::numeric_limits<double>::max)());
    diamondPasses(data, output, rows, columns, iterations, pool,
                  [](double candidate, double current)
                  { return candidate < current; });
}

void insertNeighbor(std::array<RasterNeighbor, 8>& neighbors,
                    std::size_t& count, RasterNeighbor candidate)
{
    std::size_t position = count;
    while (position > 0U)
    {
        const RasterNeighbor& previous = neighbors[position - 1U];
        if (previous.squaredDistance < candidate.squaredDistance ||
            (previous.squaredDistance == candidate.squaredDistance &&
             previous.cell <= candidate.cell))
            break;
        if (position < neighbors.size())
            neighbors[position] = previous;
        --position;
    }
    if (position < neighbors.size())
        neighbors[position] = candidate;
    if (count < neighbors.size())
        ++count;
}

// B0213: dump an intermediate raster so it can be compared against upstream's.
//
// Upstream SMRF already writes `zimin`, `zimin_fill`, `zipro`, `zipro_fill`,
// `gsurfs` and `gsurfs_fill` when its `dir` option is set
// (`filters/SMRFilter.cpp`), and B0210 established that pdg's port disagrees
// with it on 99 of 1,000,000 points. Nothing on this side could be compared
// against those, so every hypothesis about *where* the two diverge has been
// argued from output classifications and counts -- and B0211's was withdrawn
// by B0212 for exactly that reason.
//
// This writes the same stages as raw little-endian doubles in the port's own
// cell order, so the first raster that differs localises the divergence
// instead of leaving it to inference. Inert unless `PDG_DEBUG_SMRF_DUMP` names
// a path prefix; it changes no value.
void dumpRaster(const char* stage, const std::vector<double>& values)
{
    const char* prefix = std::getenv("PDG_DEBUG_SMRF_DUMP");
    if (!prefix)
        return;
    std::ofstream out(std::string(prefix) + stage + ".f64",
                      std::ios::binary | std::ios::trunc);
    if (!out)
        return;
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
}

// B0212: report how much work the void fill actually does.
//
// B0211 located pdg's divergence from upstream SMRF in this function: upstream
// picks the eight neighbours with a 2D KD-tree over cell centres in world
// coordinates, this picks them by an integer cell-unit distance, and the mean
// is a Welford recurrence, so a different neighbour set *or order* moves the
// filled value's low bits. A cell-size sweep supported that indirectly --
// disagreement fell from 99 points to 1 as the raster densified -- but nothing
// yet reports whether this function ran at all.
//
// Inert unless `PDG_DEBUG_SMRF_FILL` is set, and it changes no value. It exists
// so the correlation can be measured directly rather than inferred from cell
// size, which is a proxy for void count and not the thing itself.
void reportFill(std::size_t voids, std::size_t cells)
{
    if (!std::getenv("PDG_DEBUG_SMRF_FILL"))
        return;
    std::clog << "pdg-smrf-fill: filled " << voids << " void cells of " << cells
              << " (" << (cells ? 100.0 * static_cast<double>(voids) /
                                      static_cast<double>(cells)
                                : 0.0)
              << "%)\n";
}

void fillRaster(std::vector<double>& values, const RasterFrame& frame)
{
    std::vector<std::size_t> valid;
    valid.reserve(values.size());
    for (std::size_t cell = 0U; cell < values.size(); ++cell)
        if (!std::isnan(values[cell]))
            valid.push_back(cell);
    if (valid.empty())
    {
        reportFill(0U, values.size());
        return;
    }
    std::size_t filled = 0U;

    for (std::size_t column = 0U; column < frame.columns; ++column)
    {
        for (std::size_t row = 0U; row < frame.rows; ++row)
        {
            const std::size_t cell = column * frame.rows + row;
            if (!std::isnan(values[cell]))
                continue;
            std::array<RasterNeighbor, 8> nearest{};
            std::size_t count = 0U;
            for (std::size_t source : valid)
            {
                const std::size_t sourceColumn = source / frame.rows;
                const std::size_t sourceRow = source % frame.rows;
                const std::uint64_t deltaColumn = column > sourceColumn
                                                      ? column - sourceColumn
                                                      : sourceColumn - column;
                const std::uint64_t deltaRow =
                    row > sourceRow ? row - sourceRow : sourceRow - row;
                insertNeighbor(
                    nearest, count,
                    {deltaColumn * deltaColumn + deltaRow * deltaRow, source});
            }
            double mean = 0.0;
            for (std::size_t index = 0U; index < count; ++index)
            {
                const double delta = values[nearest[index].cell] - mean;
                mean += delta / static_cast<double>(index + 1U);
            }
            values[cell] = mean;
            ++filled;
        }
    }
    reportFill(filled, values.size());
}

// B0216: use the caller's neighbour selection when it supplied one.
void applyVoidFill(std::vector<double>& values, const RasterFrame& frame,
                   double cell, const RasterVoidFill& fill)
{
    if (fill)
        fill(values, frame.rows, frame.columns, frame.minimumX, frame.minimumY,
             cell);
    else
        fillRaster(values, frame);
}


std::vector<int> progressiveFilter(const std::vector<double>& input,
                                   const RasterFrame& frame, double cell,
                                   double slope, double maximumWindow)
{
    const int maximumRadius = static_cast<int>(std::ceil(maximumWindow / cell));
    std::vector<double> previous = input;
    std::vector<double> erosion = input;
    std::vector<int> objects(frame.size(), 0);
    const std::unique_ptr<PassPool> pool = morphologyPool(frame.size());
    for (int radius = 1; radius <= maximumRadius; ++radius)
    {
        erodeDiamond(erosion, frame.rows, frame.columns, 1, pool.get());
        std::vector<double> opening = erosion;
        dilateDiamond(opening, frame.rows, frame.columns, radius, pool.get());
        const double elevationThreshold = slope * cell * radius;
        for (std::size_t index = 0U; index < objects.size(); ++index)
        {
            const double difference =
                std::fabs(previous[index] - opening[index]);
            objects[index] =
                (std::max)(objects[index],
                           difference > elevationThreshold ? 1 : 0);
        }
        previous = std::move(opening);
    }
    return objects;
}

std::vector<double> gradientX(const std::vector<double>& values,
                              const RasterFrame& frame)
{
    std::vector<double> output(frame.size(), 0.0);
    for (std::size_t column = 0U; column < frame.columns; ++column)
        for (std::size_t row = 0U; row < frame.rows; ++row)
        {
            const std::size_t cell = column * frame.rows + row;
            if (column == 0U)
                output[cell] = values[cell + frame.rows] - values[cell];
            else if (column + 1U == frame.columns)
                output[cell] = values[cell] - values[cell - frame.rows];
            else
                output[cell] = 0.5 * (values[cell + frame.rows] -
                                      values[cell - frame.rows]);
        }
    return output;
}

std::vector<double> gradientY(const std::vector<double>& values,
                              const RasterFrame& frame)
{
    std::vector<double> output(frame.size(), 0.0);
    for (std::size_t column = 0U; column < frame.columns; ++column)
        for (std::size_t row = 0U; row < frame.rows; ++row)
        {
            const std::size_t cell = column * frame.rows + row;
            if (row == 0U)
                output[cell] = values[cell + 1U] - values[cell];
            else if (row + 1U == frame.rows)
                output[cell] = values[cell] - values[cell - 1U];
            else
                output[cell] = 0.5 * (values[cell + 1U] - values[cell - 1U]);
        }
    return output;
}

SmrfResult classifyHost(PointBatch& batch, const SmrfProgram& program,
                        const RasterVoidFill& fill)
{
    const RasterFrame frame = frameFor(batch, program);
    if (frame.rows < 2U || frame.columns < 2U)
        throw std::invalid_argument(
            "SMRF exact path requires at least two raster rows and columns");
    const double* x = batch.data<double>(X);
    const double* y = batch.data<double>(Y);
    const double* z = batch.data<double>(Z);
    auto* classification = batch.data<std::uint8_t>(Classification);

    std::vector<double> minimum(frame.size(),
                                std::numeric_limits<double>::quiet_NaN());
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t cell =
            rasterCell(x[point], y[point], frame, program.cell);
        if (std::isnan(minimum[cell]) || z[point] < minimum[cell])
            minimum[cell] = z[point];
    }
    dumpRaster("minimum", minimum);
    applyVoidFill(minimum, frame, program.cell, fill);
    dumpRaster("minimum_fill", minimum);

    std::vector<double> negativeMinimum(minimum.size());
    std::transform(minimum.begin(), minimum.end(), negativeMinimum.begin(),
                   [](double value) { return -value; });
    const std::vector<int> low = progressiveFilter(
        negativeMinimum, frame, program.cell, 5.0, program.cell);

    std::vector<int> net(frame.size(), 0);
    std::vector<double> netMinimum = minimum;
    if (program.cut > 0.0)
    {
        const int spacing =
            static_cast<int>(std::ceil(program.cut / program.cell));
        if (spacing <= 0)
            throw std::invalid_argument("SMRF cut spacing is invalid");
        for (std::size_t column = 0U; column < frame.columns;
             column += static_cast<std::size_t>(spacing))
            for (std::size_t row = 0U; row < frame.rows; ++row)
                net[column * frame.rows + row] = 1;
        for (std::size_t column = 0U; column < frame.columns; ++column)
            for (std::size_t row = 0U; row < frame.rows;
                 row += static_cast<std::size_t>(spacing))
                net[column * frame.rows + row] = 1;
        std::vector<double> opening = minimum;
        const std::unique_ptr<PassPool> pool = morphologyPool(frame.size());
        erodeDiamond(opening, frame.rows, frame.columns, 2 * spacing,
                     pool.get());
        dilateDiamond(opening, frame.rows, frame.columns, 2 * spacing,
                      pool.get());
        for (std::size_t cell = 0U; cell < net.size(); ++cell)
            if (net[cell] == 1)
                netMinimum[cell] = opening[cell];
    }

    const std::vector<int> objects = progressiveFilter(
        netMinimum, frame, program.cell, program.slope, program.window);
    std::vector<double> provisional = minimum;
    for (std::size_t cell = 0U; cell < provisional.size(); ++cell)
        if (objects[cell] == 1 || low[cell] == 1 || net[cell] == 1)
            provisional[cell] = std::numeric_limits<double>::quiet_NaN();
    dumpRaster("provisional", provisional);
    applyVoidFill(provisional, frame, program.cell, fill);
    dumpRaster("provisional_fill", provisional);

    std::vector<double> scaled(provisional.size());
    std::transform(provisional.begin(), provisional.end(), scaled.begin(),
                   [cell = program.cell](double value)
                   { return value / cell; });
    const std::vector<double> gx = gradientX(scaled, frame);
    const std::vector<double> gy = gradientY(scaled, frame);
    std::vector<double> surfaceGradient(frame.size());
    for (std::size_t cell = 0U; cell < frame.size(); ++cell)
        surfaceGradient[cell] =
            std::sqrt(gx[cell] * gx[cell] + gy[cell] * gy[cell]);
    dumpRaster("gradient", surfaceGradient);
    applyVoidFill(surfaceGradient, frame, program.cell, fill);
    dumpRaster("gradient_fill", surfaceGradient);

    SmrfResult result{frame.rows, frame.columns, 0U, 0U};
    if (!program.onlyGround)
        std::fill(classification, classification + batch.size(),
                  program.otherClass);
    for (std::size_t point = 0U; point < batch.size(); ++point)
    {
        const std::size_t cell =
            rasterCell(x[point], y[point], frame, program.cell);
        if (std::isnan(provisional[cell]) || std::isnan(surfaceGradient[cell]))
            continue;
        const double elevationThreshold =
            program.threshold + program.scalar * surfaceGradient[cell];
        if (std::fabs(provisional[cell] - z[point]) > elevationThreshold)
        {
            ++result.nongroundPoints;
            if (!program.onlyGround)
                classification[point] = program.otherClass;
        }
        else
        {
            ++result.groundPoints;
            classification[point] = program.groundClass;
        }
    }
    return result;
}
} // unnamed namespace

SmrfResult classifySmrfDevice(PointBatch& batch, const SmrfProgram& program);

bool smrfSupportsExactDevice(const PointBatch& hostBatch,
                             const SmrfProgram& program) noexcept
{
    if (!SmrfExactDeviceQualified)
        return false;
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
        for (std::size_t point = 0U; point < hostBatch.size(); ++point)
            static_cast<void>(
                rasterCell(x[point], y[point], frame, program.cell));
        const double objectRadius = std::ceil(program.window / program.cell);
        const double netRadius =
            program.cut > 0.0 ? 2.0 * std::ceil(program.cut / program.cell)
                              : 0.0;
        return hostBatch.size() <= static_cast<std::size_t>(
                                       (std::numeric_limits<int>::max)()) &&
               frame.rows >= 2U && frame.columns >= 2U &&
               frame.size() <= SmrfExactDeviceMaximumRasterCells &&
               objectRadius <= SmrfExactDeviceMaximumMorphologyRadius &&
               netRadius <= SmrfExactDeviceMaximumMorphologyRadius;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

SmrfResult classifySmrf(PointBatch& batch, const SmrfProgram& program,
                        const RasterVoidFill& fill)
{
    if (!programValid(program))
        throw std::invalid_argument("invalid SMRF program");
    if (!batch.size())
        throw std::invalid_argument("SMRF requires a nonempty point batch");
    if (!batch.has(X) || !batch.has(Y) || !batch.has(Z) ||
        !batch.has(Classification))
        throw std::invalid_argument("SMRF input columns are not materialized");
    if (batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double ||
        batch.columnInfo(Z).physicalType != DimensionType::Double ||
        batch.columnInfo(Classification).physicalType !=
            DimensionType::Unsigned8)
        throw std::invalid_argument("SMRF input columns have invalid types");
    if (batch.memoryKind() == MemoryKind::Device)
        return classifySmrfDevice(batch, program);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported SMRF memory kind");
    return classifyHost(batch, program, fill);
}

} // namespace pdg
