#include <pdg/Cuda.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/index/SpatialIndex.hpp>
#include <pdg/index/SpatialTile.hpp>
#include <pdg/stages/Neighborhood.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

bool spatialDeviceAvailable()
{
    try
    {
        return !pdg::cudaDevices().empty();
    }
    catch (const pdg::CudaError&)
    {
        return false;
    }
}

std::vector<double> copyDoubleColumn(const pdg::PointBatch& batch,
                                     pdg::DimensionId dimension,
                                     cudaStream_t stream)
{
    std::vector<double> result(batch.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(result.data(), batch.rawData(dimension),
                                   batch.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

std::vector<std::uint8_t> copyUnsigned8Column(const pdg::PointBatch& batch,
                                              pdg::DimensionId dimension,
                                              cudaStream_t stream)
{
    std::vector<std::uint8_t> result(batch.size());
    PDG_CUDA_CHECK(cudaMemcpyAsync(result.data(), batch.rawData(dimension),
                                   batch.size(), cudaMemcpyDeviceToHost,
                                   stream));
    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

TEST(CudaSpatialIndex, CellTableAndRadiusCountsMatchHost)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 131103;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.01, 0.01, 0.01},
                                        {500000.0, 4800000.0, 100.0}};
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(256U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, pdg::DimensionType::Double);
        batch->materialize(Y, pdg::DimensionType::Double);
        batch->materialize(Z, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    for (std::size_t point = 0; point < Count; ++point)
    {
        host.hostSpan<double>(X)[point] =
            500000.0 + static_cast<double>((point * 7919U) % 100003U) * 0.01;
        host.hostSpan<double>(Y)[point] =
            4800000.0 + static_cast<double>((point * 1543U) % 70001U) * 0.01;
        host.hostSpan<double>(Z)[point] =
            100.0 + static_cast<double>((point * 313U) % 1009U) * 0.01;
    }
    host.hostSpan<double>(X)[1] = host.hostSpan<double>(X)[0];
    host.hostSpan<double>(Y)[1] = host.hostSpan<double>(Y)[0];
    host.hostSpan<double>(Z)[1] = host.hostSpan<double>(Z)[0];
    host.hostSpan<double>(X)[2] = host.hostSpan<double>(X)[0] + 2.0;
    host.hostSpan<double>(Y)[2] = host.hostSpan<double>(Y)[0];
    host.hostSpan<double>(Z)[2] = host.hostSpan<double>(Z)[0];

    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeUniformGridConfig(host, 3, 2.0),
        pdg::makeMortonBvhConfig(host, 3)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex hostIndex(host, config);
        pdg::SpatialIndex deviceIndex(device, config);
        hostIndex.build();
        deviceIndex.build();
        EXPECT_EQ(deviceIndex.cellCount(), hostIndex.cellCount());

        std::vector<std::uint32_t> expected(Count);
        std::vector<std::uint32_t> actual(Count);
        pdg::radiusCounts(hostIndex, 2.0, expected.data());
        std::unique_ptr<pdg::Allocation> deviceCounts = deviceMemory->allocate(
            Count * sizeof(std::uint32_t), alignof(std::uint32_t));
        pdg::radiusCounts(deviceIndex, 2.0,
                          static_cast<std::uint32_t*>(deviceCounts->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceCounts->data(),
                                       Count * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, expected);

        const double factor = 1.0 / ((4.0 / 3.0) * 3.14159 * (2.0 * 2.0 * 2.0));
        std::vector<double> expectedScaled(Count);
        pdg::radiusScaledValues(hostIndex, 2.0, factor, expectedScaled.data());
        std::unique_ptr<pdg::Allocation> deviceScaled =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        pdg::radiusScaledValues(deviceIndex, 2.0, factor,
                                static_cast<double*>(deviceScaled->data()));
        std::vector<double> actualScaled(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualScaled.data(), deviceScaled->data(), Count * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualScaled, expectedScaled);

        deviceIndex.invalidate();
        deviceIndex.build();
        EXPECT_EQ(deviceIndex.buildCount(), 2U);
    }
}

TEST(CudaSpatialIndex, MaskedNearestGroundMatchesHostForBothBackends)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 7U;
    constexpr std::uint32_t Neighbors = 1U;
    constexpr std::uint32_t References = 3U;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                        {0.0, 0.0, 0.0});
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(8U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, pdg::DimensionType::Double);
        batch->materialize(Y, pdg::DimensionType::Double);
        batch->materialize(Z, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    constexpr std::array<double, Count> XValues{0.0, 2.0, 5.0, 8.0,
                                                 10.0, 13.0, 15.0};
    for (std::size_t point = 0U; point < Count; ++point)
    {
        host.hostSpan<double>(X)[point] = XValues[point];
        host.hostSpan<double>(Y)[point] = 0.0;
        host.hostSpan<double>(Z)[point] =
            1000.0 - static_cast<double>(point) * 113.0;
    }
    const std::array<std::uint8_t, Count> sources{1U, 0U, 1U, 0U,
                                                  1U, 0U, 1U};
    const std::array<std::uint8_t, Count> references{0U, 1U, 0U, 1U,
                                                     0U, 1U, 0U};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> deviceSources =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    std::unique_ptr<pdg::Allocation> deviceReferences =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSources->data(), sources.data(), Count,
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceReferences->data(), references.data(),
                                   Count, cudaMemcpyHostToDevice, stream));

    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(host, 2U, Neighbors),
        pdg::makeMortonBvhConfig(host, 2U)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex hostIndex(host, config);
        pdg::SpatialIndex deviceIndex(device, config);
        hostIndex.build();
        deviceIndex.build();
        std::array<std::uint32_t, Count> expectedIds{};
        std::array<double, Count> expectedDistances{};
        std::array<std::uint8_t, Count> expectedStatus{};
        pdg::knnGatherMasked(
            hostIndex, Neighbors, References, sources.data(),
            references.data(), expectedIds.data(), expectedDistances.data(),
            expectedStatus.data());

        std::unique_ptr<pdg::Allocation> deviceIds = deviceMemory->allocate(
            Count * sizeof(std::uint32_t), alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> deviceDistances =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> deviceStatus =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        pdg::knnGatherMasked(
            deviceIndex, Neighbors, References,
            static_cast<const std::uint8_t*>(deviceSources->data()),
            static_cast<const std::uint8_t*>(deviceReferences->data()),
            static_cast<std::uint32_t*>(deviceIds->data()),
            static_cast<double*>(deviceDistances->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()));
        std::array<std::uint32_t, Count> actualIds{};
        std::array<double, Count> actualDistances{};
        std::array<std::uint8_t, Count> actualStatus{};
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualIds.data(), deviceIds->data(),
                                       Count * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualDistances.data(),
                                       deviceDistances->data(),
                                       Count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualIds, expectedIds);
        EXPECT_EQ(actualDistances, expectedDistances);
        EXPECT_EQ(actualStatus, expectedStatus);
    }
}

TEST(CudaSpatialIndex, RadiusAnyMatchesHostDomainsAndVerticalCaps)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 6U;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}};
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(16U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        for (pdg::DimensionId dimension : {X, Y, Z})
            batch->materialize(dimension, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    const std::array<std::array<double, 3>, Count> points{{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 2.0},
        {10.0, 0.0, 0.0},
        {10.5, 0.0, 2.01},
        {20.0, 0.0, 0.0},
        {21.0, 0.0, 0.0},
    }};
    for (std::size_t point = 0; point < Count; ++point)
    {
        host.hostSpan<double>(X)[point] = points[point][0];
        host.hostSpan<double>(Y)[point] = points[point][1];
        host.hostSpan<double>(Z)[point] = points[point][2];
    }
    const std::vector<std::uint8_t> source{1, 0, 1, 0, 1, 0};
    const std::vector<std::uint8_t> reference{0, 1, 0, 1, 0, 1};
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));
    std::unique_ptr<pdg::Allocation> deviceSource =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    std::unique_ptr<pdg::Allocation> deviceReference =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    std::unique_ptr<pdg::Allocation> deviceMatches =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSource->data(), source.data(), Count,
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceReference->data(), reference.data(),
                                   Count, cudaMemcpyHostToDevice, stream));

    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeUniformGridConfig(host, 2, 1.0),
        pdg::makeMortonBvhConfig(host, 2)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex hostIndex(host, config);
        pdg::SpatialIndex deviceIndex(device, config);
        hostIndex.build();
        deviceIndex.build();
        std::vector<std::uint8_t> expected(Count);
        std::vector<std::uint8_t> actual(Count);
        pdg::radiusAny(hostIndex, 1.0, source.data(), reference.data(), 2.0,
                       -1.0, expected.data());
        pdg::radiusAny(
            deviceIndex, 1.0,
            static_cast<const std::uint8_t*>(deviceSource->data()),
            static_cast<const std::uint8_t*>(deviceReference->data()), 2.0,
            -1.0, static_cast<std::uint8_t*>(deviceMatches->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actual.data(), deviceMatches->data(),
                                       Count, cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actual, expected);
    }
}

TEST(CudaSpatialIndex, TiledRadiusMosaicMatchesWholeViewAcrossGhostSeams)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Columns = 128;
    constexpr std::size_t Rows = 65;
    constexpr std::size_t Count = Columns * Rows + 3U;
    constexpr double Radius = 0.51;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.01, 0.01, 0.01},
                                        {500000.0, 4800000.0, 100.0}};
    pdg::HostMemoryResource hostMemory;
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    for (pdg::DimensionId dimension : {X, Y, Z})
        host.materialize(dimension, pdg::DimensionType::Double);
    host.setSize(Count);
    for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t column = 0; column < Columns; ++column)
        {
            const std::size_t point = row * Columns + column;
            host.data<double>(X)[point] =
                500000.0 + static_cast<double>(column) * 0.25;
            host.data<double>(Y)[point] =
                4800000.0 + static_cast<double>(row) * 0.25;
            host.data<double>(Z)[point] =
                100.0 +
                static_cast<double>((row * 17U + column * 13U) % 9U) * 0.01;
        }
    for (std::size_t point = Columns * Rows; point < Count; ++point)
    {
        host.data<double>(X)[point] = host.data<double>(X)[17];
        host.data<double>(Y)[point] = host.data<double>(Y)[17];
        host.data<double>(Z)[point] = host.data<double>(Z)[17];
    }

    const pdg::UniformGridConfig wholeConfig =
        pdg::makeUniformGridConfig(host, 3, Radius);
    pdg::SpatialIndex wholeIndex(host, wholeConfig);
    wholeIndex.build();
    std::vector<std::uint32_t> expected(Count);
    pdg::radiusCounts(wholeIndex, Radius, expected.data());

    const pdg::SpatialTileSet tiles = pdg::makeSpatialTiles(
        host, {2, 4.0, Radius, {500000.0, 4800000.0, 0.0}, 2048U});
    ASSERT_GT(tiles.tiles().size(), 1U);
    ASSERT_GT(tiles.ghostPointCount(), 0U);
    std::unique_ptr<pdg::MemoryResource> pinnedMemory =
        pdg::makeCudaPinnedMemoryResource();
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(256U * 1024U * 1024U);
    std::vector<std::uint32_t> actual(Count);
    const pdg::SpatialTileExecutionStats stats =
        pdg::tiledRadiusCounts(host, tiles, 3, Radius, dimensions,
                               *pinnedMemory, *deviceMemory, actual);

    EXPECT_EQ(actual, expected);
    EXPECT_EQ(stats.tileCount, tiles.tiles().size());
    EXPECT_EQ(stats.indexBuilds, tiles.tiles().size());
    EXPECT_EQ(stats.ghostPointCount, tiles.ghostPointCount());
    EXPECT_EQ(stats.peakPointCount, tiles.peakPointCount());
    EXPECT_EQ(stats.executionLaneCount, 2U);
    EXPECT_EQ(stats.laneReuseCount, tiles.tiles().size() - 2U);

    // Exercise every permitted scheduler width against the same whole-view
    // oracle while forcing each lane to submit and later be reused.
    for (std::size_t lanes = pdg::MinimumSweptLaneCount;
         lanes <= pdg::MaximumSweptLaneCount; ++lanes)
    {
        SCOPED_TRACE(lanes);
        const pdg::SpatialTileSet sweptTiles = pdg::makeSpatialTiles(
            host, {2, 4.0, Radius, {500000.0, 4800000.0, 0.0}, 2048U, lanes});
        ASSERT_GT(sweptTiles.tiles().size(), lanes);
        std::vector<std::uint32_t> sweptActual(Count);
        const pdg::SpatialTileExecutionStats sweptStats =
            pdg::tiledRadiusCounts(host, sweptTiles, 3, Radius, dimensions,
                                   *pinnedMemory, *deviceMemory, sweptActual);
        EXPECT_EQ(sweptActual, expected);
        EXPECT_EQ(sweptStats.executionLaneCount, lanes);
        EXPECT_EQ(sweptStats.laneReuseCount, sweptTiles.tiles().size() - lanes);
    }

    const std::size_t threeLaneBudget = tiles.peakPointCount() * 128U * 3U;
    const pdg::SpatialTileSet budgetedTiles =
        pdg::makeSpatialTiles(host, {2,
                                     4.0,
                                     Radius,
                                     {500000.0, 4800000.0, 0.0},
                                     2048U,
                                     6U,
                                     threeLaneBudget});
    std::vector<std::uint32_t> budgetedActual(Count);
    const pdg::SpatialTileExecutionStats budgetedStats =
        pdg::tiledRadiusCounts(host, budgetedTiles, 3, Radius, dimensions,
                               *pinnedMemory, *deviceMemory, budgetedActual);
    EXPECT_EQ(budgetedActual, expected);
    EXPECT_EQ(budgetedStats.executionLaneCount, 3U);
    EXPECT_EQ(budgetedStats.laneReuseCount, budgetedTiles.tiles().size() - 3U);

    const double factor =
        1.0 / ((4.0 / 3.0) * 3.14159 * (Radius * Radius * Radius));
    std::vector<double> expectedScaled(Count);
    pdg::radiusScaledValues(wholeIndex, Radius, factor, expectedScaled.data());
    std::vector<double> actualScaled(Count);
    const pdg::SpatialTileExecutionStats scaledStats =
        pdg::tiledRadiusScaledValues(host, tiles, 3, Radius, factor, dimensions,
                                     *pinnedMemory, *deviceMemory,
                                     actualScaled);
    EXPECT_EQ(actualScaled, expectedScaled);
    EXPECT_EQ(scaledStats.tileCount, tiles.tiles().size());
    EXPECT_EQ(scaledStats.executionLaneCount, 2U);
    EXPECT_EQ(scaledStats.laneReuseCount, tiles.tiles().size() - 2U);

    // Pageable staging remains a supported exact API path, but it cannot
    // promise transfer/compute overlap and therefore reuses one lane.
    pdg::HostMemoryResource pageableMemory;
    std::vector<std::uint32_t> pageableActual(Count);
    const pdg::SpatialTileExecutionStats pageableStats =
        pdg::tiledRadiusCounts(host, tiles, 3, Radius, dimensions,
                               pageableMemory, *deviceMemory, pageableActual);
    EXPECT_EQ(pageableActual, expected);
    EXPECT_EQ(pageableStats.executionLaneCount, 1U);
    EXPECT_EQ(pageableStats.laneReuseCount, tiles.tiles().size() - 1U);

    // Fail a later submission after both streams have queued work. Lane
    // destruction must drain outstanding transfers/kernels before pinned
    // storage is released, and the caller-owned pool must remain reusable.
    std::vector<std::size_t> firstTile(
        Count, (std::numeric_limits<std::size_t>::max)());
    for (std::size_t tileIndex = 0; tileIndex < tiles.tiles().size();
         ++tileIndex)
        for (std::size_t sourcePoint : tiles.tiles()[tileIndex].sourcePointIds)
            firstTile[sourcePoint] =
                (std::min)(firstTile[sourcePoint], tileIndex);
    const auto poisonPosition =
        std::max_element(firstTile.begin(), firstTile.end());
    ASSERT_NE(poisonPosition, firstTile.end());
    ASSERT_GE(*poisonPosition, 2U);
    const std::size_t poisonPoint =
        static_cast<std::size_t>(poisonPosition - firstTile.begin());
    const double originalX = host.data<double>(X)[poisonPoint];
    host.data<double>(X)[poisonPoint] =
        (std::numeric_limits<double>::quiet_NaN)();
    std::vector<std::uint32_t> rejected(Count);
    EXPECT_THROW(static_cast<void>(pdg::tiledRadiusCounts(
                     host, tiles, 3, Radius, dimensions, *pinnedMemory,
                     *deviceMemory, rejected)),
                 std::invalid_argument);
    host.data<double>(X)[poisonPoint] = originalX;

    std::vector<std::uint32_t> recovered(Count);
    const pdg::SpatialTileExecutionStats recoveredStats =
        pdg::tiledRadiusCounts(host, tiles, 3, Radius, dimensions,
                               *pinnedMemory, *deviceMemory, recovered);
    EXPECT_EQ(recovered, expected);
    EXPECT_EQ(recoveredStats.executionLaneCount, 2U);
    EXPECT_EQ(recoveredStats.laneReuseCount, tiles.tiles().size() - 2U);
}

class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
    }

    ~ScopedEnvironment()
    {
        if (m_prior)
            ::setenv(m_name.c_str(), m_prior->c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
    }

private:
    std::string m_name;
    std::optional<std::string> m_prior;
};

TEST(CudaSpatialIndex, KnnDistancePrefilterNeverChangesABit)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    // A quantized cloud dense with duplicate points and coincident
    // distances keeps candidates at and around the retained boundary,
    // where an unsound prefilter margin would first change a bit.
    constexpr std::size_t Count = 4099;
    constexpr std::uint32_t Neighbors = 11;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.001, 0.001, 0.001},
                                        {500000.0, 4800000.0, 100.0}};
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(256U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, pdg::DimensionType::Double);
        batch->materialize(Y, pdg::DimensionType::Double);
        batch->materialize(Z, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    for (std::size_t point = 0; point < Count; ++point)
    {
        // Coarse 0.5-unit quantization with every fourth point duplicating
        // its predecessor: many exactly-equal squared distances.
        const std::size_t base = point - (point % 4U == 3U ? 1U : 0U);
        host.hostSpan<double>(X)[point] =
            500000.0 + static_cast<double>((base * 7919U) % 97U) * 0.5;
        host.hostSpan<double>(Y)[point] =
            4800000.0 + static_cast<double>((base * 1543U) % 89U) * 0.5;
        host.hostSpan<double>(Z)[point] =
            100.0 + static_cast<double>((base * 313U) % 83U) * 0.5;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const pdg::UniformGridConfig config =
        pdg::makeKnnGridConfig(host, 3, Neighbors);
    const std::size_t resultCount = Count * Neighbors;
    const auto run = [&](const char* disable)
    {
        ScopedEnvironment scope("PDG_DISABLE_KNN_DISTANCE_PREFILTER", disable);
        pdg::SpatialIndex index(device, config);
        index.build();
        std::unique_ptr<pdg::Allocation> ids = deviceMemory->allocate(
            resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> distances = deviceMemory->allocate(
            resultCount * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> status =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        pdg::knnGather(index, Neighbors,
                       static_cast<std::uint32_t*>(ids->data()),
                       static_cast<double*>(distances->data()),
                       static_cast<std::uint8_t*>(status->data()));
        std::vector<std::uint32_t> hostIds(resultCount);
        std::vector<double> hostDistances(resultCount);
        std::vector<std::uint8_t> hostStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostIds.data(), ids->data(),
                                       resultCount * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostDistances.data(), distances->data(),
                                       resultCount * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus.data(), status->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return std::tuple{hostIds, hostDistances, hostStatus};
    };

    const auto [filteredIds, filteredDistances, filteredStatus] = run(nullptr);
    const auto [exactIds, exactDistances, exactStatus] = run("1");
    EXPECT_EQ(filteredStatus, exactStatus);
    EXPECT_EQ(filteredIds, exactIds);
    // Bitwise, not value, comparison: NaN placeholders must agree too.
    ASSERT_EQ(filteredDistances.size(), exactDistances.size());
    for (std::size_t item = 0; item < filteredDistances.size(); ++item)
    {
        std::uint64_t filteredBits = 0;
        std::uint64_t exactBits = 0;
        std::memcpy(&filteredBits, &filteredDistances[item], sizeof(double));
        std::memcpy(&exactBits, &exactDistances[item], sizeof(double));
        ASSERT_EQ(filteredBits, exactBits) << "row " << item;
    }
}

// D0271: under the fast marker the device gather publishes no tie bit for
// either backend; ids, distances, and every other status bit are unchanged.
TEST(CudaSpatialIndex, RelaxedTieOrderMasksOnlyTheTieBitOnBothBackends)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 4099;
    constexpr std::uint32_t Neighbors = 11;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.001, 0.001, 0.001},
                                        {500000.0, 4800000.0, 100.0}};
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(256U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, pdg::DimensionType::Double);
        batch->materialize(Y, pdg::DimensionType::Double);
        batch->materialize(Z, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    for (std::size_t point = 0; point < Count; ++point)
    {
        const std::size_t base = point - (point % 4U == 3U ? 1U : 0U);
        host.hostSpan<double>(X)[point] =
            500000.0 + static_cast<double>((base * 7919U) % 97U) * 0.5;
        host.hostSpan<double>(Y)[point] =
            4800000.0 + static_cast<double>((base * 1543U) % 89U) * 0.5;
        host.hostSpan<double>(Z)[point] =
            100.0 + static_cast<double>((base * 313U) % 83U) * 0.5;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(host, 3, Neighbors),
        pdg::makeMortonBvhConfig(host, 3)};
    const std::size_t resultCount = Count * Neighbors;
    const auto run = [&](const pdg::UniformGridConfig& config,
                         const char* marker)
    {
        ScopedEnvironment scope("PDG_INTERNAL_FAST_MODE", marker);
        pdg::SpatialIndex index(device, config);
        index.build();
        std::unique_ptr<pdg::Allocation> ids = deviceMemory->allocate(
            resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> distances = deviceMemory->allocate(
            resultCount * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> status =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        pdg::knnGather(index, Neighbors,
                       static_cast<std::uint32_t*>(ids->data()),
                       static_cast<double*>(distances->data()),
                       static_cast<std::uint8_t*>(status->data()));
        std::vector<std::uint32_t> hostIds(resultCount);
        std::vector<double> hostDistances(resultCount);
        std::vector<std::uint8_t> hostStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostIds.data(), ids->data(),
                                       resultCount * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostDistances.data(), distances->data(),
                                       resultCount * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(hostStatus.data(), status->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return std::tuple{hostIds, hostDistances, hostStatus};
    };

    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        const auto [exactIds, exactDistances, exactStatus] =
            run(config, nullptr);
        const auto [relaxedIds, relaxedDistances, relaxedStatus] =
            run(config, "1");
        std::size_t ties = 0;
        for (const std::uint8_t status : exactStatus)
            ties += (status & pdg::KnnDistanceTie) != 0U;
        EXPECT_GT(ties, 0U) << "fixture must expose distance ties";
        for (std::size_t point = 0; point < Count; ++point)
        {
            EXPECT_EQ(relaxedStatus[point] & pdg::KnnDistanceTie, 0U);
            EXPECT_EQ(relaxedStatus[point],
                      exactStatus[point] & ~pdg::KnnDistanceTie);
        }
        EXPECT_EQ(relaxedIds, exactIds);
        ASSERT_EQ(relaxedDistances.size(), exactDistances.size());
        for (std::size_t item = 0; item < exactDistances.size(); ++item)
        {
            std::uint64_t relaxedBits = 0;
            std::uint64_t exactBits = 0;
            std::memcpy(&relaxedBits, &relaxedDistances[item], sizeof(double));
            std::memcpy(&exactBits, &exactDistances[item], sizeof(double));
            ASSERT_EQ(relaxedBits, exactBits) << "row " << item;
        }
    }
    // Leave the device mask in its default state for later tests.
    {
        pdg::SpatialIndex index(device, configs[0]);
        index.build();
    }
}

TEST(CudaSpatialIndex, KnnGatherMatchesUniqueHostOrder)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 4099;
    constexpr std::uint32_t Neighbors = 8;
    pdg::DimensionRegistry dimensions;
    pdg::CoordinateEncoding coordinates{{0.001, 0.001, 0.001},
                                        {500000.0, 4800000.0, 100.0}};
    pdg::HostMemoryResource hostMemory;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource(256U * 1024U * 1024U);
    pdg::PointBatch host(Count, coordinates, dimensions, hostMemory);
    pdg::PointBatch device(Count, coordinates, dimensions, *deviceMemory);
    for (pdg::PointBatch* batch : {&host, &device})
    {
        batch->materialize(X, pdg::DimensionType::Double);
        batch->materialize(Y, pdg::DimensionType::Double);
        batch->materialize(Z, pdg::DimensionType::Double);
        batch->setSize(Count);
    }
    for (std::size_t point = 0; point < Count; ++point)
    {
        host.hostSpan<double>(X)[point] =
            500000.0 + static_cast<double>((point * 7919U) % 100003U) * 0.013;
        host.hostSpan<double>(Y)[point] =
            4800000.0 + static_cast<double>((point * 1543U) % 70001U) * 0.017;
        host.hostSpan<double>(Z)[point] =
            100.0 + static_cast<double>((point * 313U) % 1009U) * 0.019;
    }
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    for (pdg::DimensionId dimension : {X, Y, Z})
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            device.rawData(dimension), host.rawData(dimension),
            Count * sizeof(double), cudaMemcpyHostToDevice, stream));

    const std::array<pdg::UniformGridConfig, 2> configs{
        pdg::makeKnnGridConfig(host, 3, Neighbors),
        pdg::makeMortonBvhConfig(host, 3)};
    for (const pdg::UniformGridConfig& config : configs)
    {
        SCOPED_TRACE(config.backend == pdg::SpatialIndexBackend::MortonBvh
                         ? "MortonBvh"
                         : "UniformGrid");
        pdg::SpatialIndex hostIndex(host, config);
        pdg::SpatialIndex deviceIndex(device, config);
        hostIndex.build();
        deviceIndex.build();

        const std::size_t resultCount = Count * Neighbors;
        std::vector<std::uint32_t> expectedIds(resultCount);
        std::vector<double> expectedDistances(resultCount);
        std::vector<std::uint8_t> expectedStatus(Count);
        pdg::knnGather(hostIndex, Neighbors, expectedIds.data(),
                       expectedDistances.data(), expectedStatus.data());

        std::unique_ptr<pdg::Allocation> deviceIds = deviceMemory->allocate(
            resultCount * sizeof(std::uint32_t), alignof(std::uint32_t));
        std::unique_ptr<pdg::Allocation> deviceDistances =
            deviceMemory->allocate(resultCount * sizeof(double),
                                   alignof(double));
        std::unique_ptr<pdg::Allocation> deviceStatus =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        pdg::knnGather(deviceIndex, Neighbors,
                       static_cast<std::uint32_t*>(deviceIds->data()),
                       static_cast<double*>(deviceDistances->data()),
                       static_cast<std::uint8_t*>(deviceStatus->data()));

        std::vector<std::uint32_t> actualIds(resultCount);
        std::vector<double> actualDistances(resultCount);
        std::vector<std::uint8_t> actualStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualIds.data(), deviceIds->data(),
                                       resultCount * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualDistances.data(), deviceDistances->data(),
            resultCount * sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualStatus, expectedStatus);
        EXPECT_EQ(actualIds, expectedIds);
        EXPECT_EQ(actualDistances, expectedDistances);

        constexpr std::uint32_t ProjectedNeighbors = 6U;
        std::vector<double> expectedProjectedMeans(Count);
        pdg::projectKnnMeanDistances(
            hostIndex, Neighbors, ProjectedNeighbors,
            expectedDistances.data(), expectedProjectedMeans.data());
        std::unique_ptr<pdg::Allocation> deviceProjected =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        pdg::projectKnnMeanDistances(
            deviceIndex, Neighbors, ProjectedNeighbors,
            static_cast<const double*>(deviceDistances->data()),
            static_cast<double*>(deviceProjected->data()));
        std::vector<double> actualProjected(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualProjected.data(), deviceProjected->data(),
            Count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualProjected, expectedProjectedMeans);

        for (const pdg::KnnDistanceMode mode :
             {pdg::KnnDistanceMode::Kth, pdg::KnnDistanceMode::Average})
        {
            std::vector<double> expectedProjectedValues(Count);
            pdg::projectKnnDistanceValues(
                hostIndex, Neighbors, ProjectedNeighbors, mode,
                expectedDistances.data(), expectedProjectedValues.data());
            pdg::projectKnnDistanceValues(
                deviceIndex, Neighbors, ProjectedNeighbors, mode,
                static_cast<const double*>(deviceDistances->data()),
                static_cast<double*>(deviceProjected->data()));
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                actualProjected.data(), deviceProjected->data(),
                Count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            EXPECT_EQ(actualProjected, expectedProjectedValues);
        }

        std::vector<double> expectedMeans(Count);
        std::vector<std::uint8_t> expectedMeanStatus(Count);
        pdg::knnMeanDistances(hostIndex, Neighbors, expectedMeans.data(),
                              expectedMeanStatus.data());
        std::unique_ptr<pdg::Allocation> deviceMeans =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        pdg::knnMeanDistances(deviceIndex, Neighbors,
                              static_cast<double*>(deviceMeans->data()),
                              static_cast<std::uint8_t*>(deviceStatus->data()));
        std::vector<double> actualMeans(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualMeans.data(), deviceMeans->data(),
                                       Count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualStatus, expectedMeanStatus);
        EXPECT_EQ(actualMeans, expectedMeans);

        for (const pdg::KnnDistanceMode mode :
             {pdg::KnnDistanceMode::Kth, pdg::KnnDistanceMode::Average})
        {
            std::vector<double> expectedValues(Count);
            std::vector<std::uint8_t> expectedValueStatus(Count);
            pdg::knnDistanceValues(hostIndex, Neighbors, mode,
                                   expectedValues.data(),
                                   expectedValueStatus.data());
            pdg::knnDistanceValues(
                deviceIndex, Neighbors, mode,
                static_cast<double*>(deviceMeans->data()),
                static_cast<std::uint8_t*>(deviceStatus->data()));
            std::vector<double> actualValues(Count);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                actualValues.data(), deviceMeans->data(),
                Count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                           deviceStatus->data(), Count,
                                           cudaMemcpyDeviceToHost, stream));
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            EXPECT_EQ(actualStatus, expectedValueStatus);
            EXPECT_EQ(actualValues, expectedValues);
        }

        std::vector<double> expectedKDistances(Count);
        std::vector<double> expectedDensities(Count);
        std::vector<double> expectedFactors(Count);
        std::vector<std::uint8_t> expectedLofStatus(Count);
        std::vector<std::uint8_t> expectedNeighborStatus(Count);
        std::vector<std::uint8_t> expectedClosureStatus(Count);
        pdg::knnLofValues(hostIndex, Neighbors, expectedKDistances.data(),
                          expectedDensities.data(), expectedFactors.data(),
                          expectedLofStatus.data(),
                          expectedNeighborStatus.data(),
                          expectedClosureStatus.data());
        std::unique_ptr<pdg::Allocation> deviceKDistances =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> deviceDensities =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> deviceFactors =
            deviceMemory->allocate(Count * sizeof(double), alignof(double));
        std::unique_ptr<pdg::Allocation> deviceNeighborStatus =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceClosureStatus =
            deviceMemory->allocate(Count, alignof(std::uint8_t));
        pdg::knnLofValues(
            deviceIndex, Neighbors,
            static_cast<double*>(deviceKDistances->data()),
            static_cast<double*>(deviceDensities->data()),
            static_cast<double*>(deviceFactors->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()),
            static_cast<std::uint8_t*>(deviceNeighborStatus->data()),
            static_cast<std::uint8_t*>(deviceClosureStatus->data()));
        std::vector<double> actualKDistances(Count);
        std::vector<double> actualDensities(Count);
        std::vector<double> actualFactors(Count);
        std::vector<std::uint8_t> actualNeighborStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualKDistances.data(), deviceKDistances->data(),
            Count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualDensities.data(), deviceDensities->data(),
            Count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualFactors.data(), deviceFactors->data(), Count * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualNeighborStatus.data(),
                                       deviceNeighborStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        std::vector<std::uint8_t> actualClosureStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualClosureStatus.data(),
                                       deviceClosureStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualStatus, expectedLofStatus);
        EXPECT_EQ(actualNeighborStatus, expectedNeighborStatus);
        EXPECT_EQ(actualClosureStatus, expectedClosureStatus);
        EXPECT_EQ(actualKDistances, expectedKDistances);
        EXPECT_EQ(actualDensities, expectedDensities);
        EXPECT_EQ(actualFactors, expectedFactors);

        std::vector<pdg::Covariance3d> expectedCovariances(Count);
        pdg::knnCovariances(hostIndex, Neighbors, expectedCovariances.data(),
                            expectedMeanStatus.data());
        std::unique_ptr<pdg::Allocation> deviceCovariances =
            deviceMemory->allocate(Count * sizeof(pdg::Covariance3d),
                                   alignof(pdg::Covariance3d));
        pdg::knnCovariances(
            deviceIndex, Neighbors,
            static_cast<pdg::Covariance3d*>(deviceCovariances->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()));
        std::vector<pdg::Covariance3d> actualCovariances(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(
            actualCovariances.data(), deviceCovariances->data(),
            Count * sizeof(pdg::Covariance3d), cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualStatus, expectedMeanStatus);
        for (std::size_t point = 0; point < Count; ++point)
        {
            EXPECT_EQ(actualCovariances[point].xx,
                      expectedCovariances[point].xx)
                << point << ":xx";
            EXPECT_EQ(actualCovariances[point].xy,
                      expectedCovariances[point].xy)
                << point << ":xy";
            EXPECT_EQ(actualCovariances[point].xz,
                      expectedCovariances[point].xz)
                << point << ":xz";
            EXPECT_EQ(actualCovariances[point].yy,
                      expectedCovariances[point].yy)
                << point << ":yy";
            EXPECT_EQ(actualCovariances[point].yz,
                      expectedCovariances[point].yz)
                << point << ":yz";
            EXPECT_EQ(actualCovariances[point].zz,
                      expectedCovariances[point].zz)
                << point << ":zz";
        }

        std::vector<pdg::EigenSystem3d> expectedSystems(Count);
        pdg::knnEigenSystems(hostIndex, Neighbors, expectedSystems.data(),
                             expectedMeanStatus.data());
        std::unique_ptr<pdg::Allocation> deviceSystems = deviceMemory->allocate(
            Count * sizeof(pdg::EigenSystem3d), alignof(pdg::EigenSystem3d));
        pdg::knnEigenSystems(
            deviceIndex, Neighbors,
            static_cast<pdg::EigenSystem3d*>(deviceSystems->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()));
        std::vector<pdg::EigenSystem3d> actualSystems(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualSystems.data(),
                                       deviceSystems->data(),
                                       Count * sizeof(pdg::EigenSystem3d),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(actualStatus, expectedMeanStatus);
        for (std::size_t point = 0; point < Count; ++point)
        {
            EXPECT_EQ(actualSystems[point].values,
                      expectedSystems[point].values)
                << point << ":eigenvalues";
            EXPECT_EQ(actualSystems[point].vectors,
                      expectedSystems[point].vectors)
                << point << ":eigenvectors";
        }
    }
}

TEST(CudaNeighborhoodColumns, NormalAndEigenvaluesMatchHostBits)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 513;
    constexpr double Sentinel = -987654.125;
    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    device.setSize(Count);

    std::vector<pdg::EigenSystem3d> systems(Count);
    std::vector<std::uint8_t> status(Count, pdg::KnnExact);
    for (std::size_t point = 0; point < Count; ++point)
    {
        systems[point].values = {
            0.125 + static_cast<double>(point % 7U) * 0.03125,
            1.0 + static_cast<double>(point % 11U) * 0.0625,
            4.0 + static_cast<double>(point % 13U) * 0.125};
        systems[point].vectors = {0.25,
                                  -0.5,
                                  0.75,
                                  -0.125,
                                  0.375,
                                  -0.625,
                                  point % 2U ? -0.875 : 0.875,
                                  0.5,
                                  -0.25};
    }
    status[1] = pdg::KnnCovarianceZero;
    status[2] = pdg::KnnEigenFailure;
    status[3] = pdg::KnnDistanceTie;
    status[4] = pdg::KnnSearchIncomplete;

    std::unique_ptr<pdg::Allocation> deviceSystems = deviceMemory->allocate(
        Count * sizeof(pdg::EigenSystem3d), alignof(pdg::EigenSystem3d));
    std::unique_ptr<pdg::Allocation> deviceStatus =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSystems->data(), systems.data(),
                                   Count * sizeof(pdg::EigenSystem3d),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceStatus->data(), status.data(), Count,
                                   cudaMemcpyHostToDevice, stream));

    const std::array normalDimensions{
        pdg::DimensionId(pdg::StandardDimension::NormalX),
        pdg::DimensionId(pdg::StandardDimension::NormalY),
        pdg::DimensionId(pdg::StandardDimension::NormalZ),
        pdg::DimensionId(pdg::StandardDimension::Curvature)};
    const std::array eigenDimensions{
        pdg::DimensionId(pdg::StandardDimension::Eigenvalue0),
        pdg::DimensionId(pdg::StandardDimension::Eigenvalue1),
        pdg::DimensionId(pdg::StandardDimension::Eigenvalue2)};
    std::vector<double> initial(Count, Sentinel);
    for (pdg::DimensionId dimension : normalDimensions)
    {
        device.materialize(dimension, pdg::DimensionType::Double);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                       initial.data(), Count * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    }
    for (pdg::DimensionId dimension : eigenDimensions)
    {
        device.materialize(dimension, pdg::DimensionType::Double);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                       initial.data(), Count * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    }

    pdg::projectNormalColumns(
        device, static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
        static_cast<const std::uint8_t*>(deviceStatus->data()),
        pdg::NormalProgram{8, true});
    pdg::projectEigenvalueColumns(
        device, static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
        static_cast<const std::uint8_t*>(deviceStatus->data()),
        pdg::EigenvaluesProgram{8, true});

    const std::vector<double> normalX =
        copyDoubleColumn(device, normalDimensions[0], stream);
    const std::vector<double> normalY =
        copyDoubleColumn(device, normalDimensions[1], stream);
    const std::vector<double> normalZ =
        copyDoubleColumn(device, normalDimensions[2], stream);
    const std::vector<double> curvature =
        copyDoubleColumn(device, normalDimensions[3], stream);
    const std::vector<double> value0 =
        copyDoubleColumn(device, eigenDimensions[0], stream);
    const std::vector<double> value1 =
        copyDoubleColumn(device, eigenDimensions[1], stream);
    const std::vector<double> value2 =
        copyDoubleColumn(device, eigenDimensions[2], stream);

    for (std::size_t point = 0; point < Count; ++point)
    {
        SCOPED_TRACE(point);
        if (status[point] != pdg::KnnExact)
        {
            EXPECT_EQ(normalX[point], Sentinel);
            EXPECT_EQ(normalY[point], Sentinel);
            EXPECT_EQ(normalZ[point], Sentinel);
            EXPECT_EQ(curvature[point], Sentinel);
            EXPECT_EQ(value0[point], Sentinel);
            EXPECT_EQ(value1[point], Sentinel);
            EXPECT_EQ(value2[point], Sentinel);
            continue;
        }
        double nx = systems[point].vectors[0];
        double ny = systems[point].vectors[3];
        double nz = systems[point].vectors[6];
        if (nz < 0.0)
        {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        const double sum = systems[point].values[0] + systems[point].values[1] +
                           systems[point].values[2];
        EXPECT_EQ(normalX[point], nx);
        EXPECT_EQ(normalY[point], ny);
        EXPECT_EQ(normalZ[point], nz);
        EXPECT_EQ(curvature[point], std::fabs(systems[point].values[0] / sum));
        EXPECT_EQ(value0[point], systems[point].values[0] / sum);
        EXPECT_EQ(value1[point], systems[point].values[1] / sum);
        EXPECT_EQ(value2[point], systems[point].values[2] / sum);
    }
}

TEST(CudaNeighborhoodColumns,
     ApproximateCoplanarUsesStrictThresholdsAndPreservesSkippedBytes)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 513;
    constexpr std::uint8_t Sentinel = 0xa5U;
    constexpr pdg::DimensionId Coplanar(pdg::StandardDimension::Coplanar);
    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    device.setSize(Count);
    device.materialize(Coplanar, pdg::DimensionType::Unsigned8);
    EXPECT_EQ(device.columnInfo(Coplanar).physicalType,
              pdg::DimensionType::Unsigned8);

    std::vector<pdg::EigenSystem3d> systems(Count);
    std::vector<std::uint8_t> status(Count, pdg::KnnExact);
    for (pdg::EigenSystem3d& system : systems)
        system.values = {1.0, 26.0, 155.0};
    systems[1].values = {1.0, 25.0, 149.0};
    systems[2].values = {1.0, 26.0, 156.0};
    systems[3].values = {1.0, 24.0, 100.0};
    status[4] = pdg::KnnCovarianceZero;
    status[5] = pdg::KnnEigenFailure;
    status[6] = pdg::KnnDistanceTie;
    status[7] = pdg::KnnSearchIncomplete;

    std::unique_ptr<pdg::Allocation> deviceSystems = deviceMemory->allocate(
        Count * sizeof(pdg::EigenSystem3d), alignof(pdg::EigenSystem3d));
    std::unique_ptr<pdg::Allocation> deviceStatus =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    const std::vector<std::uint8_t> initial(Count, Sentinel);
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSystems->data(), systems.data(),
                                   Count * sizeof(pdg::EigenSystem3d),
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceStatus->data(), status.data(), Count,
                                   cudaMemcpyHostToDevice, stream));
    PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(Coplanar), initial.data(),
                                   Count, cudaMemcpyHostToDevice, stream));

    pdg::projectApproximateCoplanarColumn(
        device, static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
        static_cast<const std::uint8_t*>(deviceStatus->data()),
        pdg::ApproximateCoplanarProgram{8, 25.0, 6.0});
    const std::vector<std::uint8_t> actual =
        copyUnsigned8Column(device, Coplanar, stream);

    EXPECT_EQ(actual[0], 1U);
    EXPECT_EQ(actual[1], 0U) << "first strict comparison equality must fail";
    EXPECT_EQ(actual[2], 0U) << "second strict comparison equality must fail";
    EXPECT_EQ(actual[3], 0U);
    for (std::size_t point = 4; point < 8; ++point)
        EXPECT_EQ(actual[point], Sentinel) << point;
    for (std::size_t point = 8; point < Count; ++point)
        EXPECT_EQ(actual[point], 1U) << point;

    systems[8].values = {std::numeric_limits<double>::denorm_min(),
                         std::numeric_limits<double>::denorm_min(), 1.0};
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSystems->data(), systems.data(),
                                   Count * sizeof(pdg::EigenSystem3d),
                                   cudaMemcpyHostToDevice, stream));
    constexpr std::array NumericEdgePrograms{
        pdg::ApproximateCoplanarProgram{8, 0.0, 0.0},
        pdg::ApproximateCoplanarProgram{8, -25.0, 6.0},
        pdg::ApproximateCoplanarProgram{8, (std::numeric_limits<double>::max)(),
                                        (std::numeric_limits<double>::max)()},
        pdg::ApproximateCoplanarProgram{8, 1.0, -0.0}};
    for (const pdg::ApproximateCoplanarProgram& program : NumericEdgePrograms)
    {
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(Coplanar), initial.data(),
                                       Count, cudaMemcpyHostToDevice, stream));
        pdg::projectApproximateCoplanarColumn(
            device,
            static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
            static_cast<const std::uint8_t*>(deviceStatus->data()), program);
        const std::vector<std::uint8_t> edgeActual =
            copyUnsigned8Column(device, Coplanar, stream);
        for (std::size_t point = 0; point < Count; ++point)
        {
            if (status[point] != pdg::KnnExact)
            {
                EXPECT_EQ(edgeActual[point], Sentinel) << point;
                continue;
            }
            const std::array<double, 3>& values = systems[point].values;
            const bool expected = values[1] > program.threshold1 * values[0] &&
                                  program.threshold2 * values[1] > values[2];
            EXPECT_EQ(edgeActual[point], expected ? 1U : 0U) << point;
        }
    }
}

TEST(CudaNeighborhoodColumns, CovarianceFeaturesMatchHostBitsAcrossModes)
{
    if (!spatialDeviceAvailable())
        GTEST_SKIP() << "no CUDA device is available";

    constexpr std::size_t Count = 257;
    constexpr double Sentinel = -123456.75;
    constexpr std::uint32_t Features =
        pdg::CovarianceLinearity | pdg::CovariancePlanarity |
        pdg::CovarianceScattering | pdg::CovarianceVerticality |
        pdg::CovarianceAnisotropy | pdg::CovarianceEigenvalueSum |
        pdg::CovarianceSurfaceVariation | pdg::CovarianceDemantkeVerticality;
    constexpr std::array dimensionsToCheck{
        pdg::DimensionId(pdg::StandardDimension::Linearity),
        pdg::DimensionId(pdg::StandardDimension::Planarity),
        pdg::DimensionId(pdg::StandardDimension::Scattering),
        pdg::DimensionId(pdg::StandardDimension::Verticality),
        pdg::DimensionId(pdg::StandardDimension::Anisotropy),
        pdg::DimensionId(pdg::StandardDimension::EigenvalueSum),
        pdg::DimensionId(pdg::StandardDimension::SurfaceVariation),
        pdg::DimensionId(pdg::StandardDimension::DemantkeVerticality)};

    pdg::DimensionRegistry dimensions;
    std::unique_ptr<pdg::MemoryResource> deviceMemory =
        pdg::makeCudaMemoryResource();
    pdg::PointBatch device(
        Count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
        dimensions, *deviceMemory);
    device.setSize(Count);
    for (pdg::DimensionId dimension : dimensionsToCheck)
        device.materialize(dimension, pdg::DimensionType::Double);

    std::vector<pdg::EigenSystem3d> systems(Count);
    std::vector<std::uint8_t> originalStatus(Count, pdg::KnnExact);
    for (std::size_t point = 0; point < Count; ++point)
    {
        systems[point].values = {
            0.125 + static_cast<double>(point % 7U) * 0.03125,
            1.0 + static_cast<double>(point % 11U) * 0.0625,
            4.0 + static_cast<double>(point % 13U) * 0.125};
        systems[point].vectors = {0.25,   -0.5,  0.75, -0.125, 0.375,
                                  -0.625, 0.875, 0.5,  -0.25};
    }
    systems[0].values = {0.0, 0.0, 0.0};
    originalStatus[1] = pdg::KnnCovarianceZero;

    std::unique_ptr<pdg::Allocation> deviceSystems = deviceMemory->allocate(
        Count * sizeof(pdg::EigenSystem3d), alignof(pdg::EigenSystem3d));
    std::unique_ptr<pdg::Allocation> deviceStatus =
        deviceMemory->allocate(Count, alignof(std::uint8_t));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(device.nativeStreamHandle());
    PDG_CUDA_CHECK(cudaMemcpyAsync(deviceSystems->data(), systems.data(),
                                   Count * sizeof(pdg::EigenSystem3d),
                                   cudaMemcpyHostToDevice, stream));
    EXPECT_THROW(
        pdg::projectCovarianceFeatureColumns(
            device,
            static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()),
            pdg::CovarianceFeaturesProgram{10, pdg::EigenvalueMode::Raw,
                                           pdg::CovarianceOmnivariance |
                                               pdg::CovarianceEigenentropy}),
        std::invalid_argument);

    for (pdg::EigenvalueMode mode :
         {pdg::EigenvalueMode::Raw, pdg::EigenvalueMode::Sqrt,
          pdg::EigenvalueMode::Normalized})
    {
        SCOPED_TRACE(static_cast<unsigned int>(mode));
        std::vector<double> initial(Count, Sentinel);
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceStatus->data(),
                                       originalStatus.data(), Count,
                                       cudaMemcpyHostToDevice, stream));
        for (pdg::DimensionId dimension : dimensionsToCheck)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                device.rawData(dimension), initial.data(),
                Count * sizeof(double), cudaMemcpyHostToDevice, stream));

        pdg::projectCovarianceFeatureColumns(
            device,
            static_cast<const pdg::EigenSystem3d*>(deviceSystems->data()),
            static_cast<std::uint8_t*>(deviceStatus->data()),
            pdg::CovarianceFeaturesProgram{10, mode, Features});

        std::vector<std::uint8_t> actualStatus(Count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(actualStatus.data(),
                                       deviceStatus->data(), Count,
                                       cudaMemcpyDeviceToHost, stream));
        std::array<std::vector<double>, dimensionsToCheck.size()> actual;
        for (std::size_t dimension = 0; dimension < dimensionsToCheck.size();
             ++dimension)
            actual[dimension] =
                copyDoubleColumn(device, dimensionsToCheck[dimension], stream);
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

        EXPECT_EQ(actualStatus[0], pdg::KnnFeatureInvalid);
        EXPECT_EQ(actualStatus[1], pdg::KnnCovarianceZero);
        for (std::size_t point = 0; point < Count; ++point)
        {
            SCOPED_TRACE(point);
            if (point < 2U)
            {
                for (const std::vector<double>& column : actual)
                    EXPECT_EQ(column[point], Sentinel);
                continue;
            }
            std::array<double, 3> lambda{
                (std::max)(systems[point].values[2], 0.0),
                (std::max)(systems[point].values[1], 0.0),
                (std::max)(systems[point].values[0], 0.0)};
            const double sum = lambda[0] + lambda[1] + lambda[2];
            if (mode == pdg::EigenvalueMode::Sqrt)
                for (double& value : lambda)
                    value = std::sqrt(value);
            else if (mode == pdg::EigenvalueMode::Normalized)
                for (double& value : lambda)
                    value /= sum;
            double unary[3];
            double norm = 0.0;
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                unary[axis] =
                    lambda[0] *
                        std::fabs(systems[point].vectors[axis * 3U + 2U]) +
                    lambda[1] *
                        std::fabs(systems[point].vectors[axis * 3U + 1U]) +
                    lambda[2] * std::fabs(systems[point].vectors[axis * 3U]);
                norm += unary[axis] * unary[axis];
            }
            norm = std::sqrt(norm);
            const std::array<double, dimensionsToCheck.size()> expected{
                (lambda[0] - lambda[1]) / lambda[0],
                (lambda[1] - lambda[2]) / lambda[0],
                lambda[2] / lambda[0],
                unary[2] / norm,
                (lambda[0] - lambda[2]) / lambda[0],
                sum,
                lambda[2] / sum,
                1.0 - std::fabs(systems[point].vectors[6])};
            for (std::size_t dimension = 0;
                 dimension < dimensionsToCheck.size(); ++dimension)
                EXPECT_EQ(actual[dimension][point], expected[dimension])
                    << "dimension " << dimension << " actual bits 0x"
                    << std::hex
                    << std::bit_cast<std::uint64_t>(actual[dimension][point])
                    << " expected bits 0x"
                    << std::bit_cast<std::uint64_t>(expected[dimension]);
        }
    }
}
} // unnamed namespace
