#include <pdg/Compaction.hpp>
#include <pdg/PointBatch.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace pdg
{

std::size_t compactPointBatchDevice(const PointBatch& source,
                                    PointBatch& destination,
                                    std::span<const DimensionId> dimensions,
                                    const std::uint8_t* keep);

namespace
{
std::size_t compactHost(const PointBatch& source, PointBatch& destination,
                        std::span<const DimensionId> dimensions,
                        const std::uint8_t* keep)
{
    std::size_t selected = 0;
    for (std::size_t point = 0; point < source.size(); ++point)
    {
        if (!keep[point])
            continue;
        for (DimensionId id : dimensions)
        {
            const ColumnInfo& input = source.columnInfo(id);
            const ColumnInfo& output = destination.columnInfo(id);
            if (input.physicalType != output.physicalType)
                throw std::invalid_argument(
                    "compaction column physical types do not match");
            const std::size_t bytes = dimensionTypeSize(input.physicalType);
            const std::byte* sourceBytes =
                static_cast<const std::byte*>(source.rawData(id));
            std::byte* destinationBytes =
                static_cast<std::byte*>(destination.rawData(id));
            std::memcpy(destinationBytes + selected * bytes,
                        sourceBytes + point * bytes, bytes);
        }
        ++selected;
    }
    destination.setSize(selected);
    return selected;
}
} // unnamed namespace

std::size_t compactPointBatch(const PointBatch& source, PointBatch& destination,
                              std::span<const DimensionId> dimensions,
                              const std::uint8_t* keep)
{
    if (&source == &destination)
        throw std::invalid_argument(
            "compaction source and destination must be distinct");
    if (!keep && source.size())
        throw std::invalid_argument("compaction predicate is null");
    if (destination.capacity() < source.size())
        throw std::invalid_argument(
            "compaction destination capacity is too small");
    if (source.memoryKind() != destination.memoryKind())
        throw std::invalid_argument(
            "compaction source and destination memory kinds differ");
    if (source.memoryKind() == MemoryKind::Device)
        return compactPointBatchDevice(source, destination, dimensions, keep);
    if (source.memoryKind() == MemoryKind::Host ||
        source.memoryKind() == MemoryKind::PinnedHost)
        return compactHost(source, destination, dimensions, keep);
    throw std::invalid_argument("unsupported compaction memory kind");
}

} // namespace pdg
