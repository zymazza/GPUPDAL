#include <pdg/PointBatch.hpp>
#include <pdg/stages/Summary.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pdg
{

void updateSummariesDevice(PointBatch& batch,
                           std::span<const DimensionId> dimensions,
                           SummaryState* states, bool advanced);

namespace
{
bool validColumns(const PointBatch& batch,
                  std::span<const DimensionId> dimensions) noexcept
{
    try
    {
        return std::all_of(
            dimensions.begin(), dimensions.end(),
            [&](DimensionId dimension)
            {
                return batch.has(dimension) &&
                       batch.columnInfo(dimension).physicalType ==
                           DimensionType::Double;
            });
    }
    catch (const std::exception&)
    {
        return false;
    }
}
} // unnamed namespace

void insertSummary(SummaryState& state, double value, bool advanced)
{
    ++state.count;
    state.minimum = (std::min)(state.minimum, value);
    state.maximum = (std::max)(state.maximum, value);

    const std::uint64_t n = state.count;
    const double delta = value - state.m1;
    const double deltaN = delta / static_cast<double>(n);
    const double term1 = delta * deltaN * static_cast<double>(n - 1U);
    state.m1 += deltaN;

    if (advanced)
    {
        const double deltaN2 = std::pow(deltaN, 2.0);
        state.m4 += term1 * deltaN2 * static_cast<double>(n * n - 3U * n + 3U) +
                    (6.0 * deltaN2 * state.m2) - (4.0 * deltaN * state.m3);
        state.m3 += term1 * deltaN * static_cast<double>(n - 2U) -
                    3.0 * deltaN * state.m2;
    }
    state.m2 += term1;
}

double summaryAverage(const SummaryState& state) noexcept
{
    return state.m1;
}

double summaryPopulationVariance(const SummaryState& state) noexcept
{
    return state.m2 / static_cast<double>(state.count);
}

double summarySampleVariance(const SummaryState& state) noexcept
{
    return state.m2 / (static_cast<double>(state.count) - 1.0);
}

double summaryPopulationStddev(const SummaryState& state) noexcept
{
    return std::sqrt(summaryPopulationVariance(state));
}

double summarySampleStddev(const SummaryState& state) noexcept
{
    return std::sqrt(summarySampleVariance(state));
}

double summaryPopulationSkewness(const SummaryState& state,
                                 bool advanced) noexcept
{
    if (!state.m2 || !advanced)
        return 0.0;
    return std::sqrt(static_cast<double>(state.count)) * state.m3 /
           std::pow(state.m2, 1.5);
}

double summarySampleSkewness(const SummaryState& state, bool advanced) noexcept
{
    if (state.m2 == 0.0 || state.count <= 2U || !advanced)
        return 0.0;
    const double count = static_cast<double>(state.count);
    return summaryPopulationSkewness(state, advanced) * std::sqrt(count) *
           std::sqrt(count - 1.0) / (count - 2.0);
}

double summaryPopulationKurtosis(const SummaryState& state,
                                 bool advanced) noexcept
{
    if (state.m2 == 0.0 || !advanced)
        return 0.0;
    return static_cast<double>(state.count) * state.m4 / (state.m2 * state.m2);
}

double summarySampleKurtosis(const SummaryState& state, bool advanced) noexcept
{
    if (state.m2 == 0.0 || state.count <= 3U || !advanced)
        return 0.0;
    const double count = static_cast<double>(state.count);
    return summaryPopulationKurtosis(state, advanced) * (count + 1.0) *
           (count - 1.0) / ((count - 2.0) * (count - 3.0));
}

double summarySampleExcessKurtosis(const SummaryState& state,
                                   bool advanced) noexcept
{
    if (state.m2 == 0.0 || state.count <= 3U || !advanced)
        return 0.0;
    const double count = static_cast<double>(state.count);
    return summarySampleKurtosis(state, advanced) -
           3.0 * (count - 1.0) * (count - 1.0) /
               ((count - 2.0) * (count - 3.0));
}

bool summariesMaySupportExactDevice(const PointBatch& hostBatch,
                                    std::span<const DimensionId> dimensions,
                                    bool advanced) noexcept
{
    if (advanced || dimensions.empty() ||
        (hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !validColumns(hostBatch, dimensions))
        return false;

    try
    {
        for (DimensionId dimension : dimensions)
        {
            const double* values = hostBatch.data<double>(dimension);
            for (std::size_t point = 0; point < hostBatch.size(); ++point)
                if (!std::isfinite(values[point]))
                    return false;
        }
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

void updateSummaries(PointBatch& batch, std::span<const DimensionId> dimensions,
                     SummaryState* states, bool advanced)
{
    if (!states && !dimensions.empty())
        throw std::invalid_argument("summary states pointer is null");
    if (!validColumns(batch, dimensions))
        throw std::invalid_argument(
            "summary dimensions must be materialized Double columns");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        updateSummariesDevice(batch, dimensions, states, advanced);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported summary memory kind");

    for (std::size_t index = 0; index < dimensions.size(); ++index)
    {
        const double* values = batch.data<double>(dimensions[index]);
        for (std::size_t point = 0; point < batch.size(); ++point)
            insertSummary(states[index], values[point], advanced);
    }
}

} // namespace pdg
