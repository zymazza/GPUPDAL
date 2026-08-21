#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace pdg
{

std::vector<HistogramBin>
selectedHistogramDevice(PointBatch& batch, DimensionId target,
                        const PredicateProgram& predicate,
                        std::uint64_t indexOffset);

namespace
{
bool validTarget(const PointBatch& batch, DimensionId target) noexcept
{
    try
    {
        return batch.has(target) &&
               batch.columnInfo(target).physicalType == DimensionType::Double;
    }
    catch (const std::exception&)
    {
        return false;
    }
}
} // unnamed namespace

bool preferDefaultCudaExpressionStats(std::size_t points,
                                      std::size_t expressions) noexcept
{
    if (expressions >= 3U)
        return points >= 1'000'000U;
    if (expressions == 2U)
        return points >= 2'000'000U;
    if (expressions == 1U)
        return points >= 4'000'000U;
    return false;
}

bool histogramMaySupportExactDevice(const PointBatch& hostBatch,
                                    DimensionId target,
                                    const PredicateProgram& predicate) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !validTarget(hostBatch, target) ||
        !predicateSupportsExactDevice(hostBatch, predicate) ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return false;
    try
    {
        const double* values = hostBatch.data<double>(target);
        for (std::size_t point = 0; point < hostBatch.size(); ++point)
            if (!std::isfinite(values[point]))
                return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

std::vector<HistogramBin> selectedHistogram(PointBatch& batch,
                                            DimensionId target,
                                            const PredicateProgram& predicate,
                                            std::uint64_t indexOffset,
                                            std::size_t maximumHostWorkers)
{
    if (!validTarget(batch, target))
        throw std::invalid_argument(
            "histogram target must be a materialized Double column");
    if (batch.size() >
        (std::numeric_limits<std::uint64_t>::max)() - indexOffset)
        throw std::overflow_error("histogram point index overflow");
    if (batch.memoryKind() == MemoryKind::Device)
        return selectedHistogramDevice(batch, target, predicate, indexOffset);
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported histogram memory kind");
    if (!batch.size())
        return {};

    std::vector<std::uint8_t> keep(batch.size());
    evaluatePredicate(batch, predicate, keep.data(), maximumHostWorkers);
    std::map<double, HistogramBin> histogram;
    const double* values = batch.data<double>(target);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        if (!keep[point])
            continue;
        const double value = values[point];
        const std::uint64_t index =
            indexOffset + static_cast<std::uint64_t>(point);
        auto [position, inserted] =
            histogram.emplace(value, HistogramBin{value, 0, index});
        (void)inserted;
        ++position->second.count;
    }

    std::vector<HistogramBin> result;
    result.reserve(histogram.size());
    for (const auto& [value, bin] : histogram)
    {
        (void)value;
        result.push_back(bin);
    }
    return result;
}

} // namespace pdg
