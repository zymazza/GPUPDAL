#include <pdg/Scheduler.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

TEST(TiledScheduler, UsesFixedBoundedDefaultsWithoutAutotuning)
{
    for (const pdg::PipelineClass pipelineClass : {
             pdg::PipelineClass::LasTranslation,
             pdg::PipelineClass::FusedPointProgram,
             pdg::PipelineClass::OrderedPointProgram,
             pdg::PipelineClass::RadiusNeighborhood,
         })
    {
        const std::size_t lanes = pdg::fixedLaneCount(pipelineClass);
        EXPECT_GE(lanes, pdg::MinimumSweptLaneCount);
        EXPECT_LE(lanes, pdg::MaximumSweptLaneCount);
        const pdg::TiledSchedule schedule =
            pdg::makeTiledSchedule({.pipelineClass = pipelineClass,
                                    .itemCount = 1000U,
                                    .tileItems = 100U});
        EXPECT_EQ(schedule.configuredLaneCount, lanes);
        EXPECT_EQ(schedule.activeLaneCount, lanes);
        EXPECT_EQ(schedule.itemCount, 1000U);
        EXPECT_EQ(schedule.tileItemCapacity, 100U);
        EXPECT_EQ(schedule.tileCount, 10U);
        EXPECT_EQ(schedule.laneReuseCount, 10U - lanes);
    }
}

TEST(TiledScheduler, ParsesOnlySweptLaneOverrides)
{
    for (std::size_t lanes = pdg::MinimumSweptLaneCount;
         lanes <= pdg::MaximumSweptLaneCount; ++lanes)
        EXPECT_EQ(pdg::parseSchedulerLaneCount(std::to_string(lanes)), lanes);
    for (const std::string_view invalid : {"", "0", "1", "7", "2x", "-2"})
        EXPECT_THROW(static_cast<void>(pdg::parseSchedulerLaneCount(invalid)),
                     std::invalid_argument);
}

TEST(TiledScheduler, BoundsExplicitSweepAndSmallInputs)
{
    for (std::size_t lanes = pdg::MinimumSweptLaneCount;
         lanes <= pdg::MaximumSweptLaneCount; ++lanes)
    {
        const pdg::TiledSchedule schedule = pdg::makeTiledSchedule(
            {.pipelineClass = pdg::PipelineClass::FusedPointProgram,
             .itemCount = 1000U,
             .tileItems = 100U,
             .requestedLanes = lanes});
        EXPECT_EQ(schedule.configuredLaneCount, lanes);
        EXPECT_EQ(schedule.activeLaneCount, lanes);
    }
    EXPECT_THROW(static_cast<void>(pdg::makeTiledSchedule(
                     {.itemCount = 1U, .tileItems = 1U, .requestedLanes = 1U})),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::makeTiledSchedule(
                     {.itemCount = 1U, .tileItems = 1U, .requestedLanes = 7U})),
                 std::invalid_argument);

    const pdg::TiledSchedule empty = pdg::makeTiledSchedule(
        {.itemCount = 0U, .tileItems = 100U, .requestedLanes = 6U});
    EXPECT_EQ(empty.tileCount, 0U);
    EXPECT_EQ(empty.itemCount, 0U);
    EXPECT_EQ(empty.tileItemCapacity, 100U);
    EXPECT_EQ(empty.activeLaneCount, 0U);
    EXPECT_EQ(empty.configuredLaneCount, 6U);
    const pdg::TiledSchedule one = pdg::makeTiledSchedule(
        {.itemCount = 99U, .tileItems = 100U, .requestedLanes = 6U});
    EXPECT_EQ(one.tileCount, 1U);
    EXPECT_EQ(one.itemCount, 99U);
    EXPECT_EQ(one.tileItemCapacity, 100U);
    EXPECT_EQ(one.activeLaneCount, 1U);
}

TEST(TiledScheduler, AppliesPlannerMemoryBudgetAndSerialDependencies)
{
    const pdg::TiledSchedule memoryLimited = pdg::makeTiledSchedule(
        {.pipelineClass = pdg::PipelineClass::RadiusNeighborhood,
         .itemCount = 1000U,
         .tileItems = 100U,
         .bytesPerLane = 400U,
         .memoryBudgetBytes = 900U,
         .requestedLanes = 6U});
    EXPECT_EQ(memoryLimited.activeLaneCount, 2U);
    EXPECT_EQ(memoryLimited.peakLaneBytes, 800U);
    EXPECT_TRUE(memoryLimited.memoryLimited);
    EXPECT_THROW(
        static_cast<void>(pdg::makeTiledSchedule({.itemCount = 1000U,
                                                  .tileItems = 100U,
                                                  .bytesPerLane = 400U,
                                                  .memoryBudgetBytes = 399U})),
        std::runtime_error);

    const pdg::TiledSchedule serial = pdg::makeTiledSchedule(
        {.pipelineClass = pdg::PipelineClass::OrderedPointProgram,
         .itemCount = 1000U,
         .tileItems = 100U,
         .bytesPerLane = 400U,
         .memoryBudgetBytes = 2400U,
         .requestedLanes = 6U,
         .serialDependency = true});
    EXPECT_EQ(serial.activeLaneCount, 1U);
    EXPECT_EQ(serial.laneReuseCount, 9U);

    EXPECT_THROW(static_cast<void>(pdg::makeTiledSchedule(
                     {.itemCount = (std::numeric_limits<std::size_t>::max)(),
                      .tileItems = 1U,
                      .bytesPerLane = (std::numeric_limits<std::size_t>::max)(),
                      .requestedLanes = 2U})),
                 std::overflow_error);
}
