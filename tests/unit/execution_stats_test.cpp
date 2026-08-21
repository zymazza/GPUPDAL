#include <pdg/ExecutionStats.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <type_traits>

namespace
{

constexpr std::size_t index(pdg::ExecutionEventKind kind)
{
    return static_cast<std::size_t>(kind);
}

static_assert(noexcept(pdg::ExecutionObservationScope{}));
static_assert(noexcept(pdg::ExecutionObservationScope::record(
    pdg::ExecutionEventKind::HostToDevice, 0U, 0U)));
static_assert(std::is_nothrow_destructible_v<pdg::ExecutionObservationScope>);

TEST(ExecutionStats, IsInactiveAndBehaviorFreeWithoutAnObserver)
{
    EXPECT_FALSE(pdg::ExecutionObservationScope::active());
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::HostToDevice, 4U, 4096U);
    EXPECT_FALSE(pdg::ExecutionObservationScope::active());
}

TEST(ExecutionStats, PreservesEventsAndAggregatesBytesDeterministically)
{
    pdg::ExecutionObservationScope scope;
    ASSERT_TRUE(pdg::ExecutionObservationScope::active());

    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::DeviceRegionBegin, 3U);
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::HostToDevice, 3U, 1024U, 768U);
    pdg::ExecutionObservationScope::record(pdg::ExecutionEventKind::IndexBuild,
                                           3U, 4096U);
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::HostToDevice, 3U, 64U);
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::DeviceToHost, 3U, 512U);
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::FallbackBoundary, 3U);
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::DeviceRegionEnd, 3U);

    const pdg::ExecutionStatsSnapshot snapshot = scope.snapshot();
    ASSERT_EQ(snapshot.events.size(), 7U);
    EXPECT_EQ(snapshot.events[0].kind,
              pdg::ExecutionEventKind::DeviceRegionBegin);
    EXPECT_EQ(snapshot.events[1].regionId, 3U);
    EXPECT_EQ(snapshot.events[1].bytes, 1024U);
    EXPECT_EQ(snapshot.events[1].packingBytes, 768U);
    EXPECT_EQ(snapshot.events[4].kind, pdg::ExecutionEventKind::DeviceToHost);
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::HostToDevice)].count,
        2U);
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::HostToDevice)].bytes,
        1088U);
    EXPECT_EQ(snapshot.totals[index(pdg::ExecutionEventKind::HostToDevice)]
                  .packingBytes,
              768U);
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::DeviceToHost)].count,
        1U);
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::DeviceToHost)].bytes,
        512U);
    EXPECT_EQ(snapshot.totals[index(pdg::ExecutionEventKind::IndexBuild)].bytes,
              4096U);
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::FallbackBoundary)].count,
        1U);
}

TEST(ExecutionStats, NestedScopesRestoreTheEnclosingObserver)
{
    pdg::ExecutionStatsSnapshot innerSnapshot;
    pdg::ExecutionObservationScope outer;
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::HostToDevice, 1U, 128U);
    {
        pdg::ExecutionObservationScope inner;
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::IndexRebuild, 2U, 2048U);
        innerSnapshot = inner.snapshot();
    }
    pdg::ExecutionObservationScope::record(
        pdg::ExecutionEventKind::DeviceToHost, 1U, 64U);

    const pdg::ExecutionStatsSnapshot outerSnapshot = outer.snapshot();
    ASSERT_EQ(innerSnapshot.events.size(), 1U);
    EXPECT_EQ(innerSnapshot.events[0].kind,
              pdg::ExecutionEventKind::IndexRebuild);
    ASSERT_EQ(outerSnapshot.events.size(), 2U);
    EXPECT_EQ(outerSnapshot.events[0].kind,
              pdg::ExecutionEventKind::HostToDevice);
    EXPECT_EQ(outerSnapshot.events[1].kind,
              pdg::ExecutionEventKind::DeviceToHost);
}

TEST(ExecutionStats, IsThreadLocal)
{
    pdg::ExecutionObservationScope scope;
    std::atomic<bool> workerSawActive = false;
    std::thread worker(
        [&workerSawActive]
        {
            workerSawActive.store(pdg::ExecutionObservationScope::active(),
                                  std::memory_order_relaxed);
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::HostToDevice, 9U, 99U);
        });
    worker.join();

    const pdg::ExecutionStatsSnapshot snapshot = scope.snapshot();
    EXPECT_FALSE(workerSawActive.load(std::memory_order_relaxed));
    EXPECT_TRUE(snapshot.events.empty());
    EXPECT_EQ(
        snapshot.totals[index(pdg::ExecutionEventKind::HostToDevice)].count,
        0U);
}

} // namespace
