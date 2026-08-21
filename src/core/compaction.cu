#include <pdg/Compaction.hpp>
#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>

#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace pdg
{

namespace
{
template <typename T>
void selectColumn(void* temporary, std::size_t& temporaryBytes,
                  const PointBatch& source, PointBatch& destination,
                  DimensionId id, const std::uint8_t* keep, int* selectedCount,
                  int count, cudaStream_t stream)
{
    PDG_CUDA_CHECK(cub::DeviceSelect::Flagged(
        temporary, temporaryBytes, static_cast<const T*>(source.rawData(id)),
        keep, static_cast<T*>(destination.rawData(id)), selectedCount, count,
        stream));
}

void dispatchSelect(void* temporary, std::size_t& temporaryBytes,
                    const PointBatch& source, PointBatch& destination,
                    DimensionId id, const std::uint8_t* keep,
                    int* selectedCount, int count, cudaStream_t stream)
{
    const ColumnInfo& input = source.columnInfo(id);
    if (input.physicalType != destination.columnInfo(id).physicalType)
        throw std::invalid_argument(
            "compaction column physical types do not match");
    switch (input.physicalType)
    {
    case DimensionType::Signed8:
        return selectColumn<std::int8_t>(temporary, temporaryBytes, source,
                                         destination, id, keep, selectedCount,
                                         count, stream);
    case DimensionType::Signed16:
        return selectColumn<std::int16_t>(temporary, temporaryBytes, source,
                                          destination, id, keep, selectedCount,
                                          count, stream);
    case DimensionType::Signed32:
        return selectColumn<std::int32_t>(temporary, temporaryBytes, source,
                                          destination, id, keep, selectedCount,
                                          count, stream);
    case DimensionType::Signed64:
        return selectColumn<std::int64_t>(temporary, temporaryBytes, source,
                                          destination, id, keep, selectedCount,
                                          count, stream);
    case DimensionType::Unsigned8:
        return selectColumn<std::uint8_t>(temporary, temporaryBytes, source,
                                          destination, id, keep, selectedCount,
                                          count, stream);
    case DimensionType::Unsigned16:
        return selectColumn<std::uint16_t>(temporary, temporaryBytes, source,
                                           destination, id, keep, selectedCount,
                                           count, stream);
    case DimensionType::Unsigned32:
        return selectColumn<std::uint32_t>(temporary, temporaryBytes, source,
                                           destination, id, keep, selectedCount,
                                           count, stream);
    case DimensionType::Unsigned64:
        return selectColumn<std::uint64_t>(temporary, temporaryBytes, source,
                                           destination, id, keep, selectedCount,
                                           count, stream);
    case DimensionType::Float:
        return selectColumn<float>(temporary, temporaryBytes, source,
                                   destination, id, keep, selectedCount, count,
                                   stream);
    case DimensionType::Double:
        return selectColumn<double>(temporary, temporaryBytes, source,
                                    destination, id, keep, selectedCount, count,
                                    stream);
    case DimensionType::None:
        break;
    }
    throw std::invalid_argument(
        "compaction requires a concrete physical column type");
}
} // unnamed namespace

std::size_t compactPointBatchDevice(const PointBatch& source,
                                    PointBatch& destination,
                                    std::span<const DimensionId> dimensions,
                                    const std::uint8_t* keep)
{
    if (source.memoryKind() != MemoryKind::Device ||
        destination.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA compaction requires device batches");
    if (source.nativeStreamHandle() != destination.nativeStreamHandle())
        throw std::invalid_argument(
            "CUDA compaction batches must share one stream");
    if (source.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(
            "CUDA compaction chunk exceeds the CUB item-count domain");
    if (source.size() == 0)
    {
        destination.setSize(0);
        return 0;
    }

    const int count = static_cast<int>(source.size());
    const cudaStream_t stream =
        static_cast<cudaStream_t>(source.nativeStreamHandle());
    std::unique_ptr<Allocation> selectedAllocation =
        source.memoryResource().allocate(sizeof(int), alignof(int));
    auto* selectedCount = static_cast<int*>(selectedAllocation->data());

    std::size_t maximumTemporaryBytes = 0;
    for (DimensionId id : dimensions)
    {
        std::size_t temporaryBytes = 0;
        dispatchSelect(nullptr, temporaryBytes, source, destination, id, keep,
                       selectedCount, count, stream);
        maximumTemporaryBytes = std::max(maximumTemporaryBytes, temporaryBytes);
    }
    std::unique_ptr<Allocation> temporary = source.memoryResource().allocate(
        maximumTemporaryBytes, alignof(std::max_align_t));
    for (DimensionId id : dimensions)
    {
        std::size_t temporaryBytes = maximumTemporaryBytes;
        dispatchSelect(temporary->data(), temporaryBytes, source, destination,
                       id, keep, selectedCount, count, stream);
    }

    int selected = 0;
    PDG_CUDA_CHECK(cudaMemcpyAsync(&selected, selectedCount, sizeof(selected),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (selected < 0 || selected > count)
        throw std::runtime_error("CUDA compaction returned an invalid count");
    destination.setSize(static_cast<std::size_t>(selected));
    return static_cast<std::size_t>(selected);
}

} // namespace pdg
