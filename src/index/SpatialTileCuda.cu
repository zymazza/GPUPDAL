#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/index/SpatialIndex.hpp>
#include <pdg/index/SpatialTile.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace pdg::detail
{
namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);
constexpr DimensionId Z(StandardDimension::Z);
constexpr std::size_t ConservativeBytesPerPoint = 128U;

class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};

std::size_t releaseThreshold(std::size_t capacity)
{
    if (capacity >
        (std::numeric_limits<std::size_t>::max)() / ConservativeBytesPerPoint)
        return (std::numeric_limits<std::size_t>::max)();
    return capacity * ConservativeBytesPerPoint;
}

template <typename Output> class RadiusTileLane
{
public:
    RadiusTileLane(std::size_t capacity,
                   const CoordinateEncoding& coordinateEncoding,
                   DimensionRegistry& registry,
                   std::span<const DimensionId> selected,
                   MemoryResource& stagingMemory, MemoryResource& deviceMemory)
        : m_stagingMemory(&stagingMemory), m_deviceMemory(&deviceMemory)
    {
        initialize(capacity, coordinateEncoding, registry, selected);
    }

    RadiusTileLane(std::size_t capacity,
                   const CoordinateEncoding& coordinateEncoding,
                   DimensionRegistry& registry,
                   std::span<const DimensionId> selected)
        : m_ownedStagingMemory(makeCudaPinnedMemoryResource()),
          m_ownedDeviceMemory(
              makeCudaMemoryResource(releaseThreshold(capacity))),
          m_stagingMemory(m_ownedStagingMemory.get()),
          m_deviceMemory(m_ownedDeviceMemory.get())
    {
        initialize(capacity, coordinateEncoding, registry, selected);
    }

    ~RadiusTileLane()
    {
        // A submission may throw after queuing a transfer but before recording
        // its completion event. Keep pinned buffers alive until every queued
        // operation is complete even on that exceptional path.
        if (m_workQueued && m_stream)
            PDG_CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(m_stream));
        if (m_completion)
            PDG_CUDA_CHECK_NOEXCEPT(cudaEventDestroy(m_completion));
    }

    RadiusTileLane(const RadiusTileLane&) = delete;
    RadiusTileLane& operator=(const RadiusTileLane&) = delete;

    template <typename Query>
    void submit(const PointBatch& source, const SpatialTile& tile,
                std::span<const DimensionId> selected,
                std::uint8_t queryDimensions, double radius, Query&& query)
    {
        if (m_pending)
            throw std::logic_error(
                "spatial tile lane reused before completion");

        NvtxRange range("pdg.spatial_tiles.submit");
        gatherSpatialTileInto(source, tile, selected, *m_hostTile);
        m_deviceTile->setSize(m_hostTile->size());
        for (DimensionId dimension : selected)
        {
            PDG_CUDA_CHECK(cudaMemcpyAsync(m_deviceTile->rawData(dimension),
                                           m_hostTile->rawData(dimension),
                                           m_hostTile->size() * sizeof(double),
                                           cudaMemcpyHostToDevice, m_stream));
            m_workQueued = true;
        }
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            m_deviceTile->ghostData(), m_hostTile->ghostData(),
            m_hostTile->size(), cudaMemcpyHostToDevice, m_stream));
        m_workQueued = true;

        const UniformGridConfig config =
            makeUniformGridConfig(*m_hostTile, queryDimensions, radius);
        if (!uniformGridMaySupportExactDevice(*m_deviceTile, config))
            throw std::runtime_error(
                "spatial tile is outside the exact device index envelope");
        m_index = std::make_unique<SpatialIndex>(*m_deviceTile, config);
        m_index->build();
        query(*m_index, static_cast<Output*>(m_deviceOutput->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(m_hostOutput->data(),
                                       m_deviceOutput->data(),
                                       m_hostTile->size() * sizeof(Output),
                                       cudaMemcpyDeviceToHost, m_stream));
        PDG_CUDA_CHECK(cudaEventRecord(m_completion, m_stream));
        m_tile = &tile;
        m_pending = true;
    }

    void drain(std::span<Output> output)
    {
        if (!m_pending)
            return;

        NvtxRange range("pdg.spatial_tiles.drain");
        PDG_CUDA_CHECK(cudaEventSynchronize(m_completion));
        scatterSpatialTileOwned(
            *m_tile,
            std::span<const Output>(
                static_cast<const Output*>(m_hostOutput->data()),
                m_hostTile->size()),
            output);
        m_index.reset();
        m_tile = nullptr;
        m_pending = false;
    }

private:
    void initialize(std::size_t capacity,
                    const CoordinateEncoding& coordinateEncoding,
                    DimensionRegistry& registry,
                    std::span<const DimensionId> selected)
    {
        if (!capacity)
            throw std::invalid_argument(
                "spatial tile lane capacity must be positive");
        if (capacity >
            (std::numeric_limits<std::size_t>::max)() / sizeof(Output))
            throw std::overflow_error(
                "spatial tile output allocation size overflow");

        m_hostTile = std::make_unique<PointBatch>(capacity, coordinateEncoding,
                                                  registry, *m_stagingMemory);
        m_deviceTile = std::make_unique<PointBatch>(
            capacity, coordinateEncoding, registry, *m_deviceMemory);
        for (DimensionId dimension : selected)
        {
            m_hostTile->materialize(dimension, DimensionType::Double);
            m_deviceTile->materialize(dimension, DimensionType::Double);
        }
        m_hostTile->materializeGhostMask();
        m_deviceTile->materializeGhostMask();
        const std::size_t outputBytes = capacity * sizeof(Output);
        m_hostOutput = m_stagingMemory->allocate(outputBytes, alignof(Output));
        m_deviceOutput = m_deviceMemory->allocate(outputBytes, alignof(Output));
        m_stream =
            static_cast<cudaStream_t>(m_deviceMemory->nativeStreamHandle());
        if (!m_stream)
            throw std::invalid_argument(
                "device spatial tile lane has no CUDA stream");
        PDG_CUDA_CHECK(
            cudaEventCreateWithFlags(&m_completion, cudaEventDisableTiming));
    }

    // Resources precede every object that allocates from them so reverse
    // member destruction releases batches and allocations first.
    std::unique_ptr<MemoryResource> m_ownedStagingMemory;
    std::unique_ptr<MemoryResource> m_ownedDeviceMemory;
    MemoryResource* m_stagingMemory = nullptr;
    MemoryResource* m_deviceMemory = nullptr;
    std::unique_ptr<PointBatch> m_hostTile;
    std::unique_ptr<PointBatch> m_deviceTile;
    std::unique_ptr<Allocation> m_hostOutput;
    std::unique_ptr<Allocation> m_deviceOutput;
    std::unique_ptr<SpatialIndex> m_index;
    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_completion = nullptr;
    const SpatialTile* m_tile = nullptr;
    bool m_pending = false;
    bool m_workQueued = false;
};

template <typename Output, typename Query>
void tiledRadiusDevice(const PointBatch& source, const SpatialTileSet& tiles,
                       std::uint8_t queryDimensions, double radius,
                       DimensionRegistry& registry,
                       MemoryResource& stagingMemory,
                       MemoryResource& deviceMemory, std::span<Output> output,
                       Query&& query, const char* rangeName)
{
    NvtxRange range(rangeName);
    if ((stagingMemory.kind() != MemoryKind::Host &&
         stagingMemory.kind() != MemoryKind::PinnedHost) ||
        deviceMemory.kind() != MemoryKind::Device)
        throw std::invalid_argument(
            "device tiled radius requires host staging and device execution "
            "memory");
    if (tiles.tiles().empty())
        return;

    const std::array<DimensionId, 3> coordinateIds{X, Y, Z};
    const std::span<const DimensionId> selected(coordinateIds.data(),
                                                queryDimensions);
    const std::size_t laneCount =
        stagingMemory.kind() == MemoryKind::PinnedHost &&
                tiles.tiles().size() > 1U
            ? makeTiledSchedule(
                  {.pipelineClass = PipelineClass::RadiusNeighborhood,
                   .itemCount = tiles.tiles().size(),
                   .tileItems = 1U,
                   .bytesPerLane = releaseThreshold(tiles.peakPointCount()),
                   .memoryBudgetBytes = tiles.config().deviceMemoryBudgetBytes,
                   .requestedLanes = tiles.config().schedulerLanes})
                  .activeLaneCount
            : 1U;
    std::vector<std::unique_ptr<RadiusTileLane<Output>>> lanes;
    lanes.reserve(laneCount);
    for (std::size_t lane = 0; lane < laneCount; ++lane)
    {
        if (lane == 0U)
            lanes.push_back(std::make_unique<RadiusTileLane<Output>>(
                tiles.peakPointCount(), source.coordinateEncoding(), registry,
                selected, stagingMemory, deviceMemory));
        else
            lanes.push_back(std::make_unique<RadiusTileLane<Output>>(
                tiles.peakPointCount(), source.coordinateEncoding(), registry,
                selected));
    }

    std::size_t sequence = 0;
    for (const SpatialTile& tile : tiles.tiles())
    {
        RadiusTileLane<Output>& lane = *lanes[sequence % laneCount];
        lane.drain(output);
        lane.submit(source, tile, selected, queryDimensions, radius, query);
        ++sequence;
    }
    for (const auto& lane : lanes)
        lane->drain(output);
}
} // unnamed namespace

void tiledRadiusCountsDevice(const PointBatch& source,
                             const SpatialTileSet& tiles,
                             std::uint8_t queryDimensions, double radius,
                             DimensionRegistry& registry,
                             MemoryResource& stagingMemory,
                             MemoryResource& deviceMemory,
                             std::span<std::uint32_t> counts)
{
    tiledRadiusDevice(
        source, tiles, queryDimensions, radius, registry, stagingMemory,
        deviceMemory, counts,
        [radius](const SpatialIndex& index, std::uint32_t* output)
        { radiusCounts(index, radius, output); }, "pdg.spatial_tiles.radius");
}

void tiledRadiusScaledValuesDevice(const PointBatch& source,
                                   const SpatialTileSet& tiles,
                                   std::uint8_t queryDimensions, double radius,
                                   double factor, DimensionRegistry& registry,
                                   MemoryResource& stagingMemory,
                                   MemoryResource& deviceMemory,
                                   std::span<double> values)
{
    tiledRadiusDevice(
        source, tiles, queryDimensions, radius, registry, stagingMemory,
        deviceMemory, values,
        [radius, factor](const SpatialIndex& index, double* output)
        { radiusScaledValues(index, radius, factor, output); },
        "pdg.spatial_tiles.radius_scaled_values");
}

} // namespace pdg::detail
