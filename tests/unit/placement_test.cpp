#include <pdg/LocalProfile.hpp>
#include <pdg/Placement.hpp>
#include <pdg/Plan.hpp>
#include <pdg/Scheduler.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
pdg::PlacementRequest uniformRequest(const pdg::Plan& plan, std::size_t points)
{
    pdg::PlacementRequest request;
    request.stageInputPointCounts.assign(plan.stages().size(), points);
    request.stageOutputPointCounts.assign(plan.stages().size(), points);
    request.stagePointCapacities.assign(plan.stages().size(), points);
    request.stageScratchBytes.assign(plan.stages().size(), 0U);
    request.stageCosts.resize(plan.stages().size());
    return request;
}
} // unnamed namespace

TEST(PlacementModel, AccountsForBoundariesPackingIndexReuseAndMemory)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
          {"type":"filters.outlier","method":"radius","radius":2,
           "min_k":4},
          {"type":"filters.outlier","method":"radius","radius":1,
           "min_k":2},
          "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);

    constexpr std::size_t Points = 1000U;
    pdg::PlacementRequest request = uniformRequest(plan, Points);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    for (std::size_t stage : {1U, 2U})
    {
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .deviceNanosecondsPerPoint = 10.0,
                                     .calibrated = true};
    }
    const pdg::PlacementModelCoefficients coefficients{
        .cudaStartupNanoseconds = 1000.0,
        .hostToDeviceNanosecondsPerByte = 2.0,
        .deviceToHostNanosecondsPerByte = 3.0,
        .packingNanosecondsPerByte = 4.0,
        .indexBuildNanosecondsPerByte = 5.0,
        .synchronizationNanoseconds = 6.0};

    const pdg::PlacementEstimate estimate =
        pdg::evaluatePlacement(plan, request, coefficients);
    EXPECT_TRUE(estimate.calibrationComplete);
    EXPECT_TRUE(estimate.layoutComplete);
    EXPECT_TRUE(estimate.deviceMemoryFeasible);
    EXPECT_EQ(estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(estimate.reason, pdg::PlacementReason::DeviceFaster);
    EXPECT_EQ(estimate.synchronizationCount, 2U);
    EXPECT_EQ(estimate.packingBytes, 66U * Points);
    EXPECT_EQ(estimate.indexBuildBytes, 28U * Points);
    const std::size_t terminalStageResultBytes =
        plan.stages()[2].descriptor.deviceToHostBytesPerInputPoint * Points;
    EXPECT_EQ(estimate.hostToDeviceBytes + estimate.deviceToHostBytes,
              plan.summary().hostDeviceTransferBytesPerPoint * Points +
                  terminalStageResultBytes);
    EXPECT_EQ(estimate.stageResultBytes, 2U * terminalStageResultBytes);
    EXPECT_DOUBLE_EQ(estimate.host.stageNanoseconds, 2'000'000.0);
    EXPECT_DOUBLE_EQ(estimate.device.stageNanoseconds, 20'000.0);
    EXPECT_DOUBLE_EQ(estimate.device.startupNanoseconds, 1000.0);
    EXPECT_DOUBLE_EQ(estimate.device.transferNanoseconds,
                     static_cast<double>(estimate.hostToDeviceBytes) * 2.0 +
                         static_cast<double>(estimate.deviceToHostBytes) * 3.0);
    EXPECT_DOUBLE_EQ(estimate.device.packingNanoseconds, 264'000.0);
    EXPECT_DOUBLE_EQ(estimate.device.indexBuildNanoseconds, 140'000.0);
    EXPECT_DOUBLE_EQ(estimate.device.synchronizationNanoseconds, 12.0);

    pdg::PlacementRequest constrained = request;
    ASSERT_GT(estimate.peakDeviceBytes, 0U);
    constrained.deviceMemoryBudgetBytes = estimate.peakDeviceBytes - 1U;
    const pdg::PlacementEstimate rejected =
        pdg::evaluatePlacement(plan, constrained, coefficients);
    EXPECT_FALSE(rejected.deviceMemoryFeasible);
    EXPECT_EQ(rejected.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(rejected.reason,
              pdg::PlacementReason::DeviceMemoryBudgetExceeded);
}

TEST(PlacementModel, UsesInputAndOutputCardinalityAtEachBoundary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.expression",
             "expression":"Classification <= 5"},"out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.stageInputPointCounts = {0U, 1000U, 250U};
    request.stageOutputPointCounts = {1000U, 250U, 250U};
    request.stagePointCapacities = {1000U, 1000U, 250U};
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.stageCosts[1] = {.hostNanosecondsPerPoint = 10.0,
                             .deviceNanosecondsPerPoint = 1.0,
                             .calibrated = true};

    const pdg::PlacementEstimate estimate =
        pdg::evaluatePlacement(plan, request,
                               {.hostToDeviceNanosecondsPerByte = 1.0,
                                .deviceToHostNanosecondsPerByte = 1.0,
                                .packingNanosecondsPerByte = 1.0});
    const auto upload = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(), [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Upload; });
    const auto spill = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(), [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Spill; });
    ASSERT_NE(upload, plan.summary().residencyBoundaries.end());
    ASSERT_NE(spill, plan.summary().residencyBoundaries.end());
    EXPECT_EQ(estimate.hostToDeviceBytes, upload->bytesPerPoint * 1000U);
    EXPECT_EQ(estimate.deviceToHostBytes, spill->bytesPerPoint * 250U);
    EXPECT_EQ(estimate.packingBytes, 30U * 1000U + 36U * 250U);
    EXPECT_DOUBLE_EQ(estimate.host.stageNanoseconds, 10'000.0);
    EXPECT_DOUBLE_EQ(estimate.device.stageNanoseconds, 1000.0);
}

TEST(PlacementModel, AccountsForDescriptorDeclaredTopologyResults)
{
    pdg::DimensionRegistry orderingDimensions;
    const pdg::Plan ordering = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.sort","dimension":"Z"},
             "out.las"])",
        orderingDimensions);
    pdg::PlacementRequest orderingRequest = uniformRequest(ordering, 1000U);
    orderingRequest.inputRecordBytes = 30U;
    orderingRequest.outputRecordBytes = 36U;
    orderingRequest.stageCosts[1].calibrated = true;
    const pdg::PlacementEstimate orderingEstimate =
        pdg::evaluatePlacement(ordering, orderingRequest, {});
    EXPECT_EQ(orderingEstimate.stageResultBytes, 1000U * sizeof(std::uint64_t));
    EXPECT_GE(orderingEstimate.deviceToHostBytes,
              orderingEstimate.stageResultBytes);

    pdg::DimensionRegistry robustDimensions;
    const pdg::Plan robust = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.iqr","dimension":"Z"},
             "out.las"])",
        robustDimensions);
    pdg::PlacementRequest robustRequest = uniformRequest(robust, 1000U);
    robustRequest.stageOutputPointCounts = {1000U, 250U, 250U};
    robustRequest.inputRecordBytes = 30U;
    robustRequest.outputRecordBytes = 36U;
    robustRequest.stageCosts[1].calibrated = true;
    const pdg::PlacementEstimate robustEstimate =
        pdg::evaluatePlacement(robust, robustRequest, {});
    EXPECT_EQ(robustEstimate.stageResultBytes, 1000U);

    pdg::DimensionRegistry statsDimensions;
    const pdg::Plan stats = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.stats",
             "dimensions":["X","Y","Z"]},"out.las"])",
        statsDimensions);
    pdg::PlacementRequest statsRequest = uniformRequest(stats, 1000U);
    statsRequest.inputRecordBytes = 30U;
    statsRequest.outputRecordBytes = 36U;
    statsRequest.stageCosts[1].calibrated = true;
    const pdg::PlacementEstimate statsEstimate =
        pdg::evaluatePlacement(stats, statsRequest, {});
    EXPECT_EQ(statsEstimate.stageResultBytes, 3U * sizeof(pdg::SummaryState));
}

TEST(PlacementModel, SmallColdWorkloadSelectsHostAndLargeSelectsDevice)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId"},"out.las"])",
        dimensions);
    const pdg::PlacementModelCoefficients coefficients{.cudaStartupNanoseconds =
                                                           10'000.0};

    pdg::PlacementRequest small = uniformRequest(plan, 100U);
    small.inputRecordBytes = 30U;
    small.outputRecordBytes = 36U;
    small.stageCosts[1] = {.hostNanosecondsPerPoint = 20.0,
                           .deviceNanosecondsPerPoint = 1.0,
                           .calibrated = true};
    const pdg::PlacementEstimate host =
        pdg::evaluatePlacement(plan, small, coefficients);
    EXPECT_EQ(host.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(host.reason, pdg::PlacementReason::HostFasterOrEqual);

    pdg::PlacementRequest large = uniformRequest(plan, 10'000U);
    large.inputRecordBytes = 30U;
    large.outputRecordBytes = 36U;
    large.stageCosts[1] = small.stageCosts[1];
    const pdg::PlacementEstimate device =
        pdg::evaluatePlacement(plan, large, coefficients);
    EXPECT_EQ(device.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(device.reason, pdg::PlacementReason::DeviceFaster);

    small.cudaContextWarm = true;
    const pdg::PlacementEstimate warm =
        pdg::evaluatePlacement(plan, small, coefficients);
    EXPECT_EQ(warm.choice, pdg::PlacementChoice::Device);
    EXPECT_DOUBLE_EQ(warm.device.startupNanoseconds, 0.0);

    small.cudaContextWarm = false;
    small.stageCosts[1].minimumDevicePointCount = 1000U;
    const pdg::PlacementEstimate outsideEnvelope =
        pdg::evaluatePlacement(plan, small, coefficients);
    EXPECT_FALSE(outsideEnvelope.calibrationEnvelopeSatisfied);
    EXPECT_EQ(outsideEnvelope.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(outsideEnvelope.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);
}

TEST(PlacementModel, FailsClosedForMissingCalibrationLayoutAndBadInputs)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId"},"out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 100U);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    pdg::PlacementEstimate estimate = pdg::evaluatePlacement(plan, request, {});
    EXPECT_FALSE(estimate.calibrationComplete);
    EXPECT_EQ(estimate.reason, pdg::PlacementReason::UncalibratedStage);

    request.stageCosts[1].calibrated = true;
    request.inputRecordBytes = 0U;
    estimate = pdg::evaluatePlacement(plan, request, {});
    EXPECT_FALSE(estimate.layoutComplete);
    EXPECT_EQ(estimate.reason, pdg::PlacementReason::MissingRecordLayout);

    request.stageInputPointCounts.pop_back();
    EXPECT_THROW(static_cast<void>(pdg::evaluatePlacement(plan, request, {})),
                 std::invalid_argument);

    request = uniformRequest(plan, (std::numeric_limits<std::size_t>::max)());
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.stageCosts[1].calibrated = true;
    EXPECT_THROW(static_cast<void>(pdg::evaluatePlacement(plan, request, {})),
                 std::overflow_error);
}

TEST(PlacementProfile, MatchesOnlyThePhysicallyCalibratedEnvironment)
{
    const pdg::PlacementDeviceKey sm89{.name = "NVIDIA GeForce RTX 4090",
                                       .computeCapability = "8.9",
                                       .driverVersion = "610.43.03",
                                       .cudaToolkitVersion = "13.3"};
    const pdg::PlacementCalibrationProfile* profile =
        pdg::placementCalibrationFor(sm89);
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->id, "sm89-2026-08-17-r6-large-layouts");
    EXPECT_EQ(profile->stageModels.size(), 42U);
    const pdg::StagePlacementCost* approximate =
        pdg::placementStageCalibration(*profile, "approximatecoplanar");
    ASSERT_NE(approximate, nullptr);
    EXPECT_TRUE(approximate->calibrated);
    EXPECT_EQ(approximate->minimumDevicePointCount, 131072U);
    const pdg::StagePlacementCost* directApproximate =
        pdg::placementStageCalibration(*profile,
                                       "approximatecoplanar-direct-compose");
    ASSERT_NE(directApproximate, nullptr);
    EXPECT_TRUE(directApproximate->calibrated);
    EXPECT_DOUBLE_EQ(directApproximate->hostFixedNanoseconds,
                     87387779.74090123);
    EXPECT_DOUBLE_EQ(directApproximate->hostNanosecondsPerPoint,
                     3385.876023673574);
    EXPECT_EQ(directApproximate->minimumDevicePointCount, 250000U);
    EXPECT_EQ(directApproximate->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* directSkewness =
        pdg::placementStageCalibration(*profile, "skewness-direct-compose");
    ASSERT_NE(directSkewness, nullptr);
    EXPECT_TRUE(directSkewness->calibrated);
    EXPECT_DOUBLE_EQ(directSkewness->hostNanosecondsPerPoint, 953.993317777778);
    EXPECT_DOUBLE_EQ(directSkewness->deviceNanosecondsPerPoint,
                     290.879214148791);
    EXPECT_EQ(directSkewness->minimumDevicePointCount, 450000U);
    EXPECT_EQ(directSkewness->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* directHagNn = pdg::placementStageCalibration(
        *profile, "hag-nn-count1-direct-compose");
    ASSERT_NE(directHagNn, nullptr);
    EXPECT_TRUE(directHagNn->calibrated);
    EXPECT_DOUBLE_EQ(directHagNn->hostFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directHagNn->hostNanosecondsPerPoint, 923.5390448076193);
    EXPECT_DOUBLE_EQ(directHagNn->deviceFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directHagNn->deviceNanosecondsPerPoint,
                     229.06845240043793);
    EXPECT_EQ(directHagNn->minimumDevicePointCount, 450000U);
    EXPECT_EQ(directHagNn->maximumDevicePointCount, 16000002U);
    const pdg::StagePlacementCost* directHagDelaunay =
        pdg::placementStageCalibration(*profile,
                                       "hag-delaunay-count3-direct-compose");
    ASSERT_NE(directHagDelaunay, nullptr);
    EXPECT_TRUE(directHagDelaunay->calibrated);
    EXPECT_DOUBLE_EQ(directHagDelaunay->hostFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directHagDelaunay->hostNanosecondsPerPoint,
                     823.250610498779);
    EXPECT_DOUBLE_EQ(directHagDelaunay->deviceFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directHagDelaunay->deviceNanosecondsPerPoint,
                     185.91405462266016);
    EXPECT_EQ(directHagDelaunay->minimumDevicePointCount, 500001U);
    EXPECT_EQ(directHagDelaunay->maximumDevicePointCount, 16000002U);
    const pdg::StagePlacementCost* eigenFamily =
        pdg::placementStageCalibration(*profile, "eigen-family-compose");
    ASSERT_NE(eigenFamily, nullptr);
    EXPECT_TRUE(eigenFamily->calibrated);
    EXPECT_DOUBLE_EQ(eigenFamily->hostFixedNanoseconds, 120004600.90127563);
    EXPECT_DOUBLE_EQ(eigenFamily->hostNanosecondsPerPoint, 14659.538986578957);
    EXPECT_DOUBLE_EQ(eigenFamily->deviceFixedNanoseconds, 155357777.57434177);
    EXPECT_DOUBLE_EQ(eigenFamily->deviceNanosecondsPerPoint,
                     1234.7451449457265);
    EXPECT_EQ(eigenFamily->minimumDevicePointCount, 250000U);
    EXPECT_EQ(eigenFamily->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* rankOptimal =
        pdg::placementStageCalibration(*profile, "rank-optimal-compose");
    ASSERT_NE(rankOptimal, nullptr);
    EXPECT_TRUE(rankOptimal->calibrated);
    EXPECT_DOUBLE_EQ(rankOptimal->hostFixedNanoseconds, 133650155.04406099);
    EXPECT_DOUBLE_EQ(rankOptimal->hostNanosecondsPerPoint, 11448.442351895543);
    EXPECT_DOUBLE_EQ(rankOptimal->deviceFixedNanoseconds, 240029160.187007);
    EXPECT_DOUBLE_EQ(rankOptimal->deviceNanosecondsPerPoint,
                     1186.5275421400943);
    EXPECT_EQ(rankOptimal->minimumDevicePointCount, 250000U);
    EXPECT_EQ(rankOptimal->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* directNeighborClassifier =
        pdg::placementStageCalibration(*profile,
                                       "neighborclassifier-direct-compose");
    ASSERT_NE(directNeighborClassifier, nullptr);
    EXPECT_TRUE(directNeighborClassifier->calibrated);
    EXPECT_DOUBLE_EQ(directNeighborClassifier->hostFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directNeighborClassifier->hostNanosecondsPerPoint,
                     3689.146693032069);
    EXPECT_DOUBLE_EQ(directNeighborClassifier->deviceFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directNeighborClassifier->deviceNanosecondsPerPoint,
                     790.7876247929496);
    EXPECT_EQ(directNeighborClassifier->minimumDevicePointCount, 250000U);
    EXPECT_EQ(directNeighborClassifier->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* directSort =
        pdg::placementStageCalibration(*profile, "sort-direct-compose");
    ASSERT_NE(directSort, nullptr);
    EXPECT_TRUE(directSort->calibrated);
    EXPECT_DOUBLE_EQ(directSort->hostFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directSort->hostNanosecondsPerPoint, 793.782365);
    EXPECT_DOUBLE_EQ(directSort->deviceFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directSort->deviceNanosecondsPerPoint, 58.74206627340533);
    EXPECT_EQ(directSort->minimumDevicePointCount, 600000U);
    EXPECT_EQ(directSort->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* directRadius =
        pdg::placementStageCalibration(*profile, "radiusassign-direct");
    ASSERT_NE(directRadius, nullptr);
    EXPECT_TRUE(directRadius->calibrated);
    EXPECT_DOUBLE_EQ(directRadius->hostFixedNanoseconds, 0.0);
    EXPECT_DOUBLE_EQ(directRadius->hostNanosecondsPerPoint, 1985.1180564323188);
    EXPECT_DOUBLE_EQ(directRadius->deviceFixedNanoseconds, 66189962.889582455);
    EXPECT_DOUBLE_EQ(directRadius->deviceNanosecondsPerPoint, 67.0868094224532);
    EXPECT_EQ(directRadius->minimumDevicePointCount, 250000U);
    EXPECT_EQ(directRadius->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* outlierNnDistance =
        pdg::placementStageCalibration(*profile,
                                       "outlier-nndistance-direct-compose");
    ASSERT_NE(outlierNnDistance, nullptr);
    EXPECT_TRUE(outlierNnDistance->calibrated);
    EXPECT_EQ(outlierNnDistance->minimumDevicePointCount, 50000U);
    EXPECT_EQ(outlierNnDistance->maximumDevicePointCount, 16000000U);
    const pdg::StagePlacementCost* radiusComposition =
        pdg::placementStageCalibration(
            *profile, "radius-outlier-radialdensity-direct-compose");
    ASSERT_NE(radiusComposition, nullptr);
    EXPECT_TRUE(radiusComposition->calibrated);
    EXPECT_DOUBLE_EQ(radiusComposition->deviceFixedNanoseconds,
                     266587419.5420545);
    EXPECT_DOUBLE_EQ(radiusComposition->hostNanosecondsPerPoint,
                     6228.969755310317);
    EXPECT_EQ(radiusComposition->minimumDevicePointCount, 250000U);
    EXPECT_EQ(radiusComposition->maximumDevicePointCount, 4000000U);
    const pdg::StagePlacementCost* normalCovariance =
        pdg::placementStageCalibration(
            *profile, "normal-covariancefeatures-compose");
    ASSERT_NE(normalCovariance, nullptr);
    EXPECT_EQ(normalCovariance->minimumDevicePointCount, 50000U);
    EXPECT_EQ(normalCovariance->maximumDevicePointCount, 47478228U);
    const pdg::StagePlacementCost* normalCovarianceExtraDimensions =
        pdg::placementStageCalibration(
            *profile, "normal-covariancefeatures-compose-extradims");
    ASSERT_NE(normalCovarianceExtraDimensions, nullptr);
    EXPECT_EQ(normalCovarianceExtraDimensions->minimumDevicePointCount,
              250000U);
    EXPECT_EQ(normalCovarianceExtraDimensions->maximumDevicePointCount,
              47478228U);

    // A higher-priority exact-machine profile must never hide a model from the
    // shipped profile for the same GPU class. B0280 caught this on the
    // reference machine. Envelopes remain independently bounded: the embedded
    // ceiling is the real 47,478,228-point local row, not the shipped profile's
    // rounded 48M class ceiling.
    const pdg::PlacementCalibrationProfile* shipped =
        pdg::shippedPlacementProfileFor(sm89);
    ASSERT_NE(shipped, nullptr);
    for (const pdg::PlacementStageCalibration& shippedModel :
         shipped->stageModels)
    {
        const pdg::StagePlacementCost* embeddedModel =
            pdg::placementStageCalibration(*profile, shippedModel.name);
        ASSERT_NE(embeddedModel, nullptr) << shippedModel.name;
    }
    EXPECT_EQ(pdg::placementStageCalibration(*profile, "unknown"), nullptr);

    // The embedded reference profile itself is keyed to the exact
    // model/SM/driver/toolkit. Since D0279 the shipped GPU-class and generic
    // tiers may answer for other keys; with those tiers disabled, every
    // mismatch must fail closed exactly as before.
    ::setenv(std::string(pdg::DisableShippedProfilesEnvironment).c_str(), "1",
             1);
    for (const pdg::PlacementDeviceKey mismatch : {
             pdg::PlacementDeviceKey{.name = "NVIDIA GeForce RTX 4080",
                                     .computeCapability = "8.9",
                                     .driverVersion = "610.43.03",
                                     .cudaToolkitVersion = "13.3"},
             pdg::PlacementDeviceKey{.name = "NVIDIA GeForce RTX 4090",
                                     .computeCapability = "9.0",
                                     .driverVersion = "610.43.03",
                                     .cudaToolkitVersion = "13.3"},
             pdg::PlacementDeviceKey{.name = "NVIDIA GeForce RTX 4090",
                                     .computeCapability = "8.9",
                                     .driverVersion = "future",
                                     .cudaToolkitVersion = "13.3"},
             pdg::PlacementDeviceKey{.name = "NVIDIA GeForce RTX 4090",
                                     .computeCapability = "8.9",
                                     .driverVersion = "610.43.03",
                                     .cudaToolkitVersion = "13.4"},
         })
        EXPECT_EQ(pdg::placementCalibrationFor(mismatch), nullptr);
    ::unsetenv(std::string(pdg::DisableShippedProfilesEnvironment).c_str());
    // With the tiers enabled: a shipped RTX 4090 profile answers for a 4090
    // on another driver (driver is not part of the shipped key), a shipped
    // 4080 profile answers for a 4080, and a toolkit mismatch still fails
    // closed (every embedded profile is keyed to the compiled toolkit).
    if (const pdg::PlacementCalibrationProfile* other = pdg::placementCalibrationFor(
            {.name = "NVIDIA GeForce RTX 4090",
             .computeCapability = "8.9",
             .driverVersion = "future",
             .cudaToolkitVersion = "13.3"}))
        EXPECT_EQ(pdg::placementProfileTier(other), "shipped");
    EXPECT_EQ(pdg::placementCalibrationFor(
                  {.name = "NVIDIA GeForce RTX 4090",
                   .computeCapability = "8.9",
                   .driverVersion = "610.43.03",
                   .cudaToolkitVersion = "13.4"}),
              nullptr);
}

TEST(PlacementProfile, AppliesOneMeasuredResidualPerResidentRegion)
{
    const pdg::PlacementCalibrationProfile* profile =
        pdg::placementCalibrationFor({.name = "NVIDIA GeForce RTX 4090",
                                      .computeCapability = "8.9",
                                      .driverVersion = "610.43.03",
                                      .cudaToolkitVersion = "13.3"});
    ASSERT_NE(profile, nullptr);
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.unimplemented"},
             {"type":"filters.transformation",
              "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    const std::vector<pdg::PlacementRegionCalibration> calibrations{
        {.residentRegion = 0U, .model = "simple-ferry"},
        {.residentRegion = 1U, .model = "transformation"}};
    ASSERT_TRUE(pdg::applyPlacementRegionCalibrations(plan, *profile,
                                                      calibrations, request));
    EXPECT_TRUE(request.stageCosts[1].calibrated);
    EXPECT_DOUBLE_EQ(request.stageCosts[1].deviceNanosecondsPerPoint,
                     1.015529713932);
    EXPECT_TRUE(request.stageCosts[3].calibrated);
    EXPECT_DOUBLE_EQ(request.stageCosts[3].deviceFixedNanoseconds,
                     9690019.501705825);

    const std::vector<pdg::PlacementRegionCalibration> unknown{
        {.residentRegion = 0U, .model = "simple-ferry"},
        {.residentRegion = 1U, .model = "not-calibrated"}};
    EXPECT_FALSE(pdg::applyPlacementRegionCalibrations(plan, *profile, unknown,
                                                       request));
    for (const pdg::PlannedStage& stage : plan.stages())
        if (stage.residentRegion != pdg::NoResidentRegion)
            EXPECT_FALSE(request.stageCosts[stage.id].calibrated);
}

TEST(PlanPlacementModel, AmortizesOneStartupAcrossProfitableResidentRegions)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.randomize","seed":1},
             {"type":"filters.transformation",
              "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 2U);

    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 36U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 36U;
    for (std::size_t stage : {1U, 3U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 100.0,
                                     .calibrated = true};

    pdg::PlacementModelCoefficients coefficients;
    coefficients.cudaStartupNanoseconds = 150'000.0;
    const pdg::PlanPlacementEstimate combined =
        pdg::evaluatePlanPlacement(plan, request, coefficients);
    ASSERT_EQ(combined.regions.size(), 2U);
    EXPECT_EQ(combined.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(combined.selectedRegionCount, 2U);
    EXPECT_TRUE(combined.regions[0].selected);
    EXPECT_TRUE(combined.regions[1].selected);
    EXPECT_EQ(combined.reason, pdg::PlacementReason::DeviceFaster);
    EXPECT_DOUBLE_EQ(combined.allHostPlacement.totalNanoseconds, 200'000.0);
    EXPECT_DOUBLE_EQ(combined.selectedPlacement.startupNanoseconds, 150'000.0);
    EXPECT_DOUBLE_EQ(combined.selectedPlacement.totalNanoseconds, 150'000.0);

    coefficients.cudaStartupNanoseconds = 250'000.0;
    const pdg::PlanPlacementEstimate coldHost =
        pdg::evaluatePlanPlacement(plan, request, coefficients);
    EXPECT_EQ(coldHost.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(coldHost.selectedRegionCount, 0U);
    EXPECT_EQ(coldHost.reason,
              pdg::PlacementReason::SharedDeviceTollNotAmortized);
    EXPECT_DOUBLE_EQ(coldHost.selectedPlacement.totalNanoseconds,
                     coldHost.allHostPlacement.totalNanoseconds);
    for (const pdg::PlacementRegionEstimate& region : coldHost.regions)
        EXPECT_EQ(region.estimate.reason,
                  pdg::PlacementReason::SharedDeviceTollNotAmortized);
}

TEST(PlanPlacementModel, SelectsOnlyWarmProfitableRegionsAcrossHostBoundary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.randomize","seed":1},
             {"type":"filters.transformation",
              "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 36U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 36U;
    request.stageCosts[1] = {.hostNanosecondsPerPoint = 100.0,
                             .calibrated = true};
    request.stageCosts[3] = {.hostNanosecondsPerPoint = 100.0,
                             .deviceNanosecondsPerPoint = 200.0,
                             .calibrated = true};

    const pdg::PlanPlacementEstimate placement = pdg::evaluatePlanPlacement(
        plan, request, {.cudaStartupNanoseconds = 50'000.0});
    ASSERT_EQ(placement.regions.size(), 2U);
    EXPECT_EQ(placement.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(placement.selectedRegionCount, 1U);
    EXPECT_TRUE(placement.regions[0].selected);
    EXPECT_FALSE(placement.regions[1].selected);
    EXPECT_EQ(placement.regions[1].estimate.reason,
              pdg::PlacementReason::HostFasterOrEqual);
    EXPECT_DOUBLE_EQ(placement.allHostPlacement.totalNanoseconds, 200'000.0);
    EXPECT_DOUBLE_EQ(placement.selectedPlacement.totalNanoseconds, 150'000.0);
}

TEST(PlanPlacementModel, ValidatesAndReturnsHostWhenNoDeviceRegionExists)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.randomize","seed":1},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 0U);
    const pdg::PlacementRequest request = uniformRequest(plan, 100U);
    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request, {});
    EXPECT_EQ(placement.choice, pdg::PlacementChoice::Host);
    EXPECT_TRUE(placement.regions.empty());
    EXPECT_EQ(placement.selectedRegionCount, 0U);

    EXPECT_THROW(static_cast<void>(pdg::evaluatePlanPlacement(
                     plan, request, {.cudaStartupNanoseconds = -1.0})),
                 std::invalid_argument);
}

TEST(PlanPlacementModel, ReportsSharedSynchronizationTollOnWarmContext)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId"},"out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 36U;
    request.outputRecordBytes = 36U;
    request.cudaContextWarm = true;
    request.additionalSynchronizations = 60U;
    request.stageCosts[1] = {.hostNanosecondsPerPoint = 500.0,
                             .deviceNanosecondsPerPoint = 1.0,
                             .calibrated = true};

    const pdg::PlanPlacementEstimate placement = pdg::evaluatePlanPlacement(
        plan, request, {.synchronizationNanoseconds = 10'000.0});
    ASSERT_EQ(placement.regions.size(), 1U);
    EXPECT_EQ(placement.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(placement.reason,
              pdg::PlacementReason::SharedDeviceTollNotAmortized);
    EXPECT_EQ(placement.regions[0].estimate.reason,
              pdg::PlacementReason::SharedDeviceTollNotAmortized);
    EXPECT_DOUBLE_EQ(placement.selectedPlacement.totalNanoseconds,
                     placement.allHostPlacement.totalNanoseconds);
    EXPECT_DOUBLE_EQ(placement.selectedPlacement.startupNanoseconds, 0.0);
    EXPECT_EQ(placement.synchronizationCount, 0U);
}

TEST(PlanPlacementModel, AggregatesOnlySelectedRegionBoundaryCosts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.unimplemented"},
             {"type":"filters.transformation",
              "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    ASSERT_EQ(plan.summary().fallbackBoundaries, 2U);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 34U;
    request.cudaContextWarm = true;
    request.additionalSynchronizations = 2U;
    for (std::size_t stage : {1U, 3U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .deviceNanosecondsPerPoint = 1.0,
                                     .calibrated = true};
    const pdg::PlacementModelCoefficients coefficients{
        .hostToDeviceNanosecondsPerByte = 1.0,
        .deviceToHostNanosecondsPerByte = 2.0,
        .packingNanosecondsPerByte = 1.0,
        .synchronizationNanoseconds = 3.0};

    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request, coefficients);
    ASSERT_EQ(placement.selectedRegionCount, 2U);
    std::size_t hostToDeviceBytes = 0U;
    std::size_t deviceToHostBytes = 0U;
    std::size_t packingBytes = 0U;
    std::size_t regionSynchronizations = 0U;
    std::size_t peakDeviceBytes = 0U;
    double selectedNanoseconds = 0.0;
    for (const pdg::PlacementRegionEstimate& region : placement.regions)
    {
        ASSERT_TRUE(region.selected);
        hostToDeviceBytes += region.estimate.hostToDeviceBytes;
        deviceToHostBytes += region.estimate.deviceToHostBytes;
        packingBytes += region.estimate.packingBytes;
        regionSynchronizations += region.estimate.synchronizationCount;
        peakDeviceBytes =
            (std::max)(peakDeviceBytes, region.estimate.peakDeviceBytes);
        selectedNanoseconds += region.estimate.device.totalNanoseconds;
    }
    selectedNanoseconds += 2U * coefficients.synchronizationNanoseconds;
    EXPECT_EQ(placement.hostToDeviceBytes, hostToDeviceBytes);
    EXPECT_EQ(placement.deviceToHostBytes, deviceToHostBytes);
    EXPECT_EQ(placement.packingBytes, packingBytes);
    EXPECT_GT(placement.packingBytes, (30U + 36U) * 1000U);
    EXPECT_EQ(placement.synchronizationCount,
              regionSynchronizations + request.additionalSynchronizations);
    EXPECT_EQ(placement.peakDeviceBytes, peakDeviceBytes);
    EXPECT_DOUBLE_EQ(placement.selectedPlacement.totalNanoseconds,
                     selectedNanoseconds);

    // The selected mixed plan reports its crossings in immutable plan-boundary
    // order.  Derive transfer direction from that stable boundary ID rather
    // than relying on the order resident regions happened to be evaluated.
    ASSERT_EQ(placement.boundaries.size(),
              plan.summary().residencyBoundaries.size());
    std::size_t boundaryHostToDeviceBytes = 0U;
    std::size_t boundaryDeviceToHostBytes = 0U;
    std::size_t boundaryPackingBytes = 0U;
    for (std::size_t index = 0; index < placement.boundaries.size(); ++index)
    {
        const pdg::PlacementBoundaryEstimate& accounting =
            placement.boundaries.at(index);
        ASSERT_EQ(accounting.boundaryId, index);
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries.at(accounting.boundaryId);
        EXPECT_EQ(accounting.fullRecordBytes,
                  request.fallbackRecordBytes * accounting.pointCount);
        boundaryPackingBytes += accounting.predictedPackingBytes;
        if (boundary.kind == pdg::ResidencyBoundaryKind::Upload)
            boundaryHostToDeviceBytes += accounting.predictedTransferBytes;
        else
            boundaryDeviceToHostBytes += accounting.predictedTransferBytes;
    }
    EXPECT_EQ(boundaryHostToDeviceBytes, placement.hostToDeviceBytes);
    EXPECT_EQ(boundaryDeviceToHostBytes, placement.deviceToHostBytes);
    EXPECT_EQ(boundaryPackingBytes, placement.packingBytes);
}

TEST(PlanPlacementModel,
     UsesExecutorDeclaredPhysicalFactsWithoutLogicalDoubleCounting)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.unimplemented"},
             {"type":"filters.ferry",
              "dimensions":"Classification=>UserData"},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 4U);

    constexpr std::size_t Points = 1000U;
    constexpr std::size_t TilePoints = 128U;
    constexpr std::size_t PhysicalRecordBytes = 41U;
    pdg::PlacementRequest request = uniformRequest(plan, Points);
    request.stagePointCapacities.assign(plan.stages().size(), TilePoints);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 34U;
    request.cudaContextWarm = true;
    request.executorLaneCount = 2U;
    for (std::size_t stage : {1U, 3U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .deviceNanosecondsPerPoint = 1.0,
                                     .calibrated = true};

    std::size_t spill = 0U;
    std::size_t expectedPackingBytes = 0U;
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const std::size_t packingBytesPerPoint =
            boundary.kind == pdg::ResidencyBoundaryKind::Upload
                ? PhysicalRecordBytes
                : 3U + 2U * spill++;
        expectedPackingBytes += packingBytesPerPoint * Points;
        request.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = PhysicalRecordBytes,
             .packingBytesPerPoint = packingBytesPerPoint,
             .deviceStagingBytesPerPoint = PhysicalRecordBytes});
    }

    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request,
                                   {.hostToDeviceNanosecondsPerByte = 1.0,
                                    .deviceToHostNanosecondsPerByte = 1.0,
                                    .packingNanosecondsPerByte = 1.0});
    ASSERT_EQ(placement.selectedRegionCount, 2U);
    EXPECT_EQ(placement.hostToDeviceBytes, 2U * PhysicalRecordBytes * Points);
    EXPECT_EQ(placement.deviceToHostBytes, 2U * PhysicalRecordBytes * Points);
    EXPECT_EQ(placement.packingBytes, expectedPackingBytes);
    const std::size_t bytesPerLane = PhysicalRecordBytes * TilePoints +
                                     plan.estimatedDeviceBytes(TilePoints);
    const pdg::TiledSchedule schedule = pdg::makeTiledSchedule(
        {.pipelineClass = pdg::PipelineClass::FusedPointProgram,
         .itemCount = Points,
         .tileItems = TilePoints,
         .bytesPerLane = bytesPerLane,
         .requestedLanes = request.executorLaneCount});
    ASSERT_EQ(schedule.activeLaneCount, 2U);
    EXPECT_EQ(placement.configuredDeviceLaneCount, 2U);
    EXPECT_EQ(placement.activeDeviceLaneCount, 2U);
    EXPECT_EQ(placement.peakDeviceBytes, schedule.peakLaneBytes);
    const std::size_t untiledDeviceBytes =
        Points * (PhysicalRecordBytes + plan.summary().peakDeviceBytesPerPoint);
    EXPECT_EQ(placement.untiledDeviceBytes, untiledDeviceBytes);
    EXPECT_GT(placement.untiledDeviceBytes, placement.peakDeviceBytes);
    ASSERT_EQ(placement.boundaries.size(), 4U);
    for (const pdg::PlacementBoundaryEstimate& boundary : placement.boundaries)
    {
        EXPECT_EQ(boundary.predictedTransferBytes,
                  PhysicalRecordBytes * Points);
        EXPECT_NE(boundary.predictedTransferBytes, boundary.logicalColumnBytes);
        EXPECT_EQ(boundary.predictedPackingBytes,
                  request.boundaryExecutionFacts[boundary.boundaryId]
                          .packingBytesPerPoint *
                      Points);
    }

    pdg::PlacementRequest constrained = request;
    constrained.deviceMemoryBudgetBytes = bytesPerLane + bytesPerLane / 2U;
    const pdg::PlanPlacementEstimate rejected =
        pdg::evaluatePlanPlacement(plan, constrained,
                                   {.hostToDeviceNanosecondsPerByte = 1.0,
                                    .deviceToHostNanosecondsPerByte = 1.0,
                                    .packingNanosecondsPerByte = 1.0});
    EXPECT_EQ(rejected.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(rejected.reason,
              pdg::PlacementReason::DeviceMemoryBudgetExceeded);
    EXPECT_EQ(rejected.selectedRegionCount, 0U);
    ASSERT_EQ(rejected.regions.size(), 2U);
    for (const pdg::PlacementRegionEstimate& region : rejected.regions)
    {
        EXPECT_EQ(region.estimate.configuredDeviceLaneCount, 2U);
        EXPECT_EQ(region.estimate.activeDeviceLaneCount, 2U);
        EXPECT_GT(region.estimate.peakDeviceBytes,
                  constrained.deviceMemoryBudgetBytes);
    }

    pdg::PlacementRequest naturallyTiled = request;
    naturallyTiled.deviceMemoryBudgetBytes = schedule.peakLaneBytes;
    const pdg::PlanPlacementEstimate tiled =
        pdg::evaluatePlanPlacement(plan, naturallyTiled,
                                   {.hostToDeviceNanosecondsPerByte = 1.0,
                                    .deviceToHostNanosecondsPerByte = 1.0,
                                    .packingNanosecondsPerByte = 1.0});
    EXPECT_EQ(tiled.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(tiled.selectedRegionCount, 2U);
    EXPECT_EQ(tiled.peakDeviceBytes, schedule.peakLaneBytes);
    EXPECT_EQ(tiled.untiledDeviceBytes, untiledDeviceBytes);
    EXPECT_GT(tiled.untiledDeviceBytes, naturallyTiled.deviceMemoryBudgetBytes);

    constexpr std::size_t SmallPoints = 64U;
    pdg::PlacementRequest oneTile = request;
    oneTile.stageInputPointCounts.assign(plan.stages().size(), SmallPoints);
    oneTile.stageOutputPointCounts.assign(plan.stages().size(), SmallPoints);
    oneTile.stagePointCapacities.assign(plan.stages().size(), SmallPoints);
    const pdg::PlanPlacementEstimate oneTilePlacement =
        pdg::evaluatePlanPlacement(plan, oneTile,
                                   {.hostToDeviceNanosecondsPerByte = 1.0,
                                    .deviceToHostNanosecondsPerByte = 1.0,
                                    .packingNanosecondsPerByte = 1.0});
    ASSERT_EQ(oneTilePlacement.selectedRegionCount, 2U);
    EXPECT_EQ(oneTilePlacement.configuredDeviceLaneCount, 2U);
    EXPECT_EQ(oneTilePlacement.activeDeviceLaneCount, 1U);
    EXPECT_EQ(oneTilePlacement.peakDeviceBytes,
              PhysicalRecordBytes * SmallPoints +
                  plan.estimatedDeviceBytes(SmallPoints));
    EXPECT_EQ(oneTilePlacement.untiledDeviceBytes,
              oneTilePlacement.peakDeviceBytes);
}

TEST(PlanPlacementModel, ChargesTheDeclaredKeepMaskThroughExecutorFacts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry","dimensions":"Classification=>UserData"},
             {"type":"filters.expression","expression":"Intensity <= 30000"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);

    constexpr std::size_t Points = 1000U;
    constexpr std::size_t TilePoints = 128U;
    constexpr std::size_t PhysicalRecordBytes = 41U;
    constexpr std::size_t KeepMaskBytes = 1U;
    pdg::PlacementRequest request = uniformRequest(plan, Points);
    request.stagePointCapacities.assign(plan.stages().size(), TilePoints);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 34U;
    request.cudaContextWarm = true;
    request.executorLaneCount = 2U;
    for (std::size_t stage : {1U, 2U, 3U, 4U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .deviceNanosecondsPerPoint = 1.0,
                                     .calibrated = true};

    std::size_t spillRepackBytesPerPoint = 0U;
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const bool spill = boundary.kind == pdg::ResidencyBoundaryKind::Spill;
        if (spill)
            spillRepackBytesPerPoint = boundary.repackBytesPerPoint;
        request.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint =
                 PhysicalRecordBytes + (spill ? KeepMaskBytes : 0U),
             .packingBytesPerPoint =
                 spill ? boundary.repackBytesPerPoint : PhysicalRecordBytes,
             .deviceStagingBytesPerPoint =
                 PhysicalRecordBytes + (spill ? KeepMaskBytes : 0U)});
    }
    ASSERT_GT(spillRepackBytesPerPoint, 0U);

    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request,
                                   {.hostToDeviceNanosecondsPerByte = 1.0,
                                    .deviceToHostNanosecondsPerByte = 1.0,
                                    .packingNanosecondsPerByte = 1.0});
    ASSERT_EQ(placement.selectedRegionCount, 1U);
    EXPECT_EQ(placement.hostToDeviceBytes, PhysicalRecordBytes * Points);
    EXPECT_EQ(placement.deviceToHostBytes,
              (PhysicalRecordBytes + KeepMaskBytes) * Points);
    EXPECT_EQ(placement.packingBytes,
              PhysicalRecordBytes * Points + spillRepackBytesPerPoint * Points);

    // The spill staging fact carries the declared one-byte keep mask, so the
    // per-lane peak and the untiled diagnostic both include it.
    const std::size_t bytesPerLane =
        (PhysicalRecordBytes + KeepMaskBytes) * TilePoints +
        plan.estimatedDeviceBytes(TilePoints);
    const pdg::TiledSchedule schedule = pdg::makeTiledSchedule(
        {.pipelineClass = pdg::PipelineClass::OrderedPointProgram,
         .itemCount = Points,
         .tileItems = TilePoints,
         .bytesPerLane = bytesPerLane,
         .requestedLanes = request.executorLaneCount});
    ASSERT_EQ(schedule.activeLaneCount, 2U);
    EXPECT_EQ(placement.peakDeviceBytes, schedule.peakLaneBytes);
    EXPECT_EQ(placement.untiledDeviceBytes,
              Points * (PhysicalRecordBytes + KeepMaskBytes +
                        plan.summary().peakDeviceBytesPerPoint));
    ASSERT_EQ(placement.boundaries.size(), 2U);
    for (const pdg::PlacementBoundaryEstimate& boundary : placement.boundaries)
    {
        const pdg::ResidencyBoundary& planned =
            plan.summary().residencyBoundaries[boundary.boundaryId];
        const bool spill = planned.kind == pdg::ResidencyBoundaryKind::Spill;
        EXPECT_EQ(boundary.pointCount, Points);
        EXPECT_EQ(boundary.predictedTransferBytes,
                  (PhysicalRecordBytes + (spill ? KeepMaskBytes : 0U)) *
                      Points);
        EXPECT_EQ(boundary.predictedPackingBytes,
                  (spill ? spillRepackBytesPerPoint : PhysicalRecordBytes) *
                      Points);
    }
}

TEST(PlanPlacementModel, KeepsMemoryInfeasibleRegionOnHost)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.ferry",
              "dimensions":"Intensity=>PointSourceId"},
             {"type":"filters.unimplemented"},
             {"type":"filters.transformation",
              "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    request.fallbackRecordBytes = 34U;
    request.deviceMemoryBudgetBytes = 100'000U;
    request.stageScratchBytes[3] = 200'000U;
    for (std::size_t stage : {1U, 3U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .calibrated = true};

    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request, {});
    ASSERT_EQ(placement.regions.size(), 2U);
    EXPECT_EQ(placement.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(placement.selectedRegionCount, 1U);
    EXPECT_TRUE(placement.regions[0].selected);
    EXPECT_FALSE(placement.regions[1].selected);
    EXPECT_EQ(placement.regions[1].estimate.reason,
              pdg::PlacementReason::DeviceMemoryBudgetExceeded);
    EXPECT_LE(placement.peakDeviceBytes, request.deviceMemoryBudgetBytes);
}

TEST(PlanPlacementModel, FailsClosedForBranchedExecutionTopology)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([
          {"type":"readers.las","filename":"in.las","tag":"source"},
          {"type":"filters.outlier","method":"radius","radius":2,
           "inputs":"source","tag":"left"},
          {"type":"filters.outlier","method":"radius","radius":3,
           "inputs":"source","tag":"right"},
          {"type":"writers.las","filename":"out.las","inputs":"right"}
        ])",
        dimensions);
    pdg::PlacementRequest request = uniformRequest(plan, 1000U);
    request.inputRecordBytes = 30U;
    request.outputRecordBytes = 36U;
    for (std::size_t stage : {1U, 2U})
        request.stageCosts[stage] = {.hostNanosecondsPerPoint = 1000.0,
                                     .calibrated = true};

    const pdg::PlanPlacementEstimate placement =
        pdg::evaluatePlanPlacement(plan, request, {});
    EXPECT_EQ(placement.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(placement.reason, pdg::PlacementReason::UnsupportedPlanTopology);
    EXPECT_EQ(placement.selectedRegionCount, 0U);
    for (const pdg::PlacementRegionEstimate& region : placement.regions)
        EXPECT_EQ(region.estimate.reason,
                  pdg::PlacementReason::UnsupportedPlanTopology);
}
