#include <pdg/Plan.hpp>
#include <pdg/RuntimePlacement.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace
{
constexpr std::string_view FusedPointProgramPipeline = R"(["in.las",
  {"type":"filters.assign","value":[
    "Scratch = Intensity * 2 - 1",
    "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
  {"type":"filters.ferry","dimensions":"Classification=>UserData"},
  {"type":"filters.assign","value":[
    "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
    "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
  "out.las"])";

constexpr std::string_view EigenFamilyComposePipeline = R"(["in.las",
  {"type":"filters.normal","knn":12,"always_up":false},
  {"type":"filters.eigenvalues","knn":12,"normalize":true},
  {"type":"filters.covariancefeatures","knn":12,"mode":"raw",
   "feature_set":"dimensionality"},
  {"type":"filters.assign","value":[
    "Classification = Linearity * 10",
    "Intensity = Curvature * 1000",
    "UserData = Eigenvalue0 * 100"]},
  "out.las"])";

constexpr std::string_view RankOptimalComposePipeline = R"(["in.las",
  {"type":"filters.estimaterank","knn":14,"thresh":0.01},
  {"type":"filters.optimalneighborhood","min_k":10,"max_k":14},
  {"type":"filters.assign","value":[
    "Classification = Rank",
    "Intensity = OptimalKNN",
    "PointSourceId = OptimalRadius"]},
  "out.las"])";

constexpr std::string_view DirectRadiusAssignPipeline = R"(["in.las",
  {"type":"filters.radiusassign","radius":2.0,"is3d":true,
   "src_domain":"ReturnNumber[1:1]",
   "reference_domain":"ReturnNumber[2:15]",
   "update_expression":"UserData = 9"},
  "out.las"])";

constexpr std::string_view DirectNeighborClassifierPipeline = R"(["in.las",
  {"type":"filters.neighborclassifier","k":7},
  "out.las"])";

constexpr std::string_view DirectSortPipeline = R"({"pipeline":[
  {"type":"readers.las","filename":"in.las"},
  {"type":"filters.sort","dimension":"Z","order":"ASC",
   "algorithm":"NORMAL"},
  {"type":"writers.las","filename":"out.las","extra_dims":"all"}
]})";

constexpr std::string_view DirectSkewnessPipeline = R"({"pipeline":[
  {"type":"readers.las","filename":"in.las"},
  {"type":"filters.skewnessbalancing"},
  {"type":"writers.las","filename":"out.las","extra_dims":"all"}
]})";

constexpr std::string_view DirectHagNnCount1Pipeline = R"({"pipeline":[
  {"type":"readers.las","filename":"in.las"},
  {"type":"filters.hag_nn","count":1},
  {"type":"writers.las","filename":"out.las","extra_dims":"all"}
]})";

constexpr std::string_view DirectHagDelaunayCount3Pipeline =
    R"({"pipeline":[
  {"type":"readers.las","filename":"in.las"},
  {"type":"filters.hag_delaunay","count":3},
  {"type":"writers.las","filename":"out.las","extra_dims":"all"}
]})";

constexpr std::string_view DirectOutlierNnDistancePipeline = R"(["in.las",
  {"type":"filters.outlier","method":"statistical","mean_k":8,
   "multiplier":2.0,"class":7},
  {"type":"filters.nndistance","mode":"kth","k":10},
  "out.las"])";

constexpr std::string_view DirectRadiusOutlierRadialDensityPipeline =
    R"(["in.las",
  {"type":"filters.outlier","method":"radius","radius":1.01,
   "min_k":2,"class":7},
  {"type":"filters.radialdensity","radius":1.01},
  {"type":"filters.assign",
   "value":"UserData = 1 WHERE RadialDensity >= 0.2"},
  "out.las"])";

constexpr std::string_view DirectApproximateCoplanarPipeline = R"(["in.las",
  {"type":"filters.approximatecoplanar","knn":8},
  {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
  "out.las"])";

const pdg::PlacementCalibrationProfile& sm89Profile()
{
    const pdg::PlacementCalibrationProfile* profile =
        pdg::placementCalibrationFor({.name = "NVIDIA GeForce RTX 4090",
                                      .computeCapability = "8.9",
                                      .driverVersion = "610.43.03",
                                      .cudaToolkitVersion = "13.3"});
    EXPECT_NE(profile, nullptr);
    return *profile;
}

pdg::RuntimePlacementFacts factsFor(const pdg::Plan& plan,
                                    std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts;
    facts.inputPointCount = pointCount;
    facts.inputRecordBytes = 30U;
    facts.outputRecordBytes = 36U;
    facts.fallbackRecordBytes = 36U;
    facts.tilePointCapacity = 131072U;
    facts.stageScratchBytes.assign(plan.stages().size(), 0U);
    facts.stageAdditionalSynchronizations.assign(plan.stages().size(), 0U);
    facts.deviceMemoryBudgetBytes = 2U * 1024U * 1024U * 1024U;
    return facts;
}

pdg::RuntimePlacementFacts directRadiusFactsFor(const pdg::Plan& plan,
                                                std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputRecordBytes = 36U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directRadiusAssignBoundaryExecutor = true;
    facts.exactDirectRadiusAssignExecutor = true;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = boundary.bytesPerPoint,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts
directNeighborClassifierFactsFor(const pdg::Plan& plan, std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 36U;
    facts.outputRecordBytes = 36U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directNeighborClassifierBoundaryExecutor = true;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = boundary.bytesPerPoint,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts directSortFactsFor(const pdg::Plan& plan,
                                              std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 36U;
    facts.outputRecordBytes = 36U;
    // PDAL's prepared PointLayout may contain internal dimensions that the
    // mapped-source/permutation executor never stages. Its size is therefore
    // deliberately not part of the calibrated physical LAS envelope.
    facts.fallbackRecordBytes = 57U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directSortBoundaryExecutor = true;
    facts.stageScratchBytes[1] =
        pointCount * pdg::OrderingExactDeviceScratchBytesPerPoint;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint =
                 boundary.kind == pdg::ResidencyBoundaryKind::Spill
                     ? sizeof(std::uint64_t)
                     : boundary.bytesPerPoint,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts directSkewnessFactsFor(const pdg::Plan& plan,
                                                  std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 36U;
    facts.outputRecordBytes = 36U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directSkewnessBoundaryExecutor = true;
    facts.stageScratchBytes[1] =
        pointCount * pdg::SkewnessExactDeviceScratchBytesPerPoint;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = 8U,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts directHagNnFactsFor(const pdg::Plan& plan,
                                               std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 40U;
    facts.outputRecordBytes = 48U;
    facts.outputCompressed = false;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directHagNnBoundaryExecutor = true;
    facts.stageScratchBytes[1] =
        pointCount * pdg::HagNnCountOneExactDeviceScratchBytesPerPoint;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint =
                 boundary.kind == pdg::ResidencyBoundaryKind::Upload ? 25U : 8U,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 40U});
    }
    return facts;
}

pdg::RuntimePlacementFacts directHagDelaunayFactsFor(const pdg::Plan& plan,
                                                     std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 40U;
    facts.outputRecordBytes = 48U;
    facts.outputCompressed = false;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directHagDelaunayBoundaryExecutor = true;
    facts.stageScratchBytes[1] =
        pointCount * pdg::HagDelaunayCountThreeExactDeviceScratchBytesPerPoint;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint =
                 boundary.kind == pdg::ResidencyBoundaryKind::Upload ? 25U : 8U,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 40U});
    }
    return facts;
}

pdg::RuntimePlacementFacts
directOutlierNnDistanceFactsFor(const pdg::Plan& plan, std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputRecordBytes = 36U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directOutlierNnDistanceBoundaryExecutor = true;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = boundary.bytesPerPoint,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts
directRadiusOutlierRadialDensityFactsFor(const pdg::Plan& plan,
                                         std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 36U;
    facts.tilePointCapacity = pointCount;
    facts.executorLaneCount = 1U;
    facts.directRadiusOutlierRadialDensityBoundaryExecutor = true;
    for (std::size_t boundaryId = 0U;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = boundary.bytesPerPoint,
             .packingBytesPerPoint = 0U,
             .deviceStagingBytesPerPoint = 36U});
    }
    return facts;
}

pdg::RuntimePlacementFacts
directApproximateCoplanarFactsFor(const pdg::Plan& plan, std::size_t pointCount)
{
    pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
    facts.inputPointFormat = 7U;
    facts.inputRecordBytes = 36U;
    facts.directApproximateCoplanarOutputExecutor = true;
    return facts;
}
} // unnamed namespace

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectNeighborClassifierComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectNeighborClassifierPipeline, dimensions);

    const pdg::RuntimePlacementResult selected = pdg::buildRuntimePlacement(
        plan, directNeighborClassifierFactsFor(plan, 250'000U), sm89Profile());
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "neighborclassifier-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    EXPECT_NEAR(selected.estimate.allHostPlacement.totalNanoseconds,
                922286673.2580172, 0.01);
    EXPECT_NEAR(selected.estimate.selectedPlacement.totalNanoseconds,
                437401412.38582367, 0.01);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              25U * 250'000U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              1U * 250'000U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 112U * 250'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directNeighborClassifierFactsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model,
              "neighborclassifier-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directNeighborClassifierFactsFor(plan, 16'000'001U),
        sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    ASSERT_TRUE(ordinary.available());
    ASSERT_EQ(ordinary.regionCalibrations.size(), 1U);
    EXPECT_EQ(ordinary.regionCalibrations[0].model, "neighborclassifier");
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectSortComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(DirectSortPipeline, dimensions);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].bytesPerPoint, 8U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].bytesPerPoint, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].producer, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].consumer, 1U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].producer, 1U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].consumer, 2U);
    EXPECT_FALSE(plan.summary().residencyBoundaries[0].requiresFullPointRecord);
    EXPECT_FALSE(plan.summary().residencyBoundaries[1].requiresFullPointRecord);
    ASSERT_EQ(plan.stages().size(), 3U);
    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[0].payload);
    const auto* ordering =
        std::get_if<pdg::OrderingProgram>(&plan.stages()[1].payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[2].payload);
    ASSERT_NE(reader, nullptr);
    ASSERT_NE(ordering, nullptr);
    ASSERT_NE(writer, nullptr);
    EXPECT_TRUE(reader->optionFreeLasFamily);
    EXPECT_TRUE(writer->extraDimensionsAll);
    EXPECT_EQ(plan.stages()[1].descriptor.deviceToHostBytesPerInputPoint, 8U);
    EXPECT_EQ(plan.stages()[1].residentRegion, 0U);

    const pdg::RuntimePlacementFacts selectedFacts =
        directSortFactsFor(plan, 600'000U);
    ASSERT_EQ(selectedFacts.boundaryExecutionFacts.size(), 2U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[0].transferBytesPerPoint,
              8U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[1].transferBytesPerPoint,
              8U);
    const pdg::RuntimePlacementResult selected =
        pdg::buildRuntimePlacement(plan, selectedFacts, sm89Profile());
    EXPECT_EQ(selected.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::None);
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model, "sort-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    EXPECT_NEAR(selected.estimate.allHostPlacement.totalNanoseconds,
                476269419.0, 0.01);
    EXPECT_NEAR(selected.estimate.selectedPlacement.totalNanoseconds,
                274868465.4498456, 0.01);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              8U * 600'000U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              8U * 600'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 0U);
    EXPECT_EQ(selected.estimate.stageResultBytes, 8U * 600'000U);
    EXPECT_EQ(selected.estimate.untiledDeviceBytes,
              pdg::OrderingExactDevicePeakBytesPerPoint * 600'000U);
    EXPECT_EQ(selected.estimate.peakDeviceBytes,
              pdg::OrderingExactDevicePeakBytesPerPoint * 600'000U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directSortFactsFor(plan, 550'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model, "sort-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directSortFactsFor(plan, 16'000'001U), sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    EXPECT_FALSE(ordinary.available());
    EXPECT_EQ(ordinary.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MissingCalibrationModel);
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectSkewnessComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectSkewnessPipeline, dimensions);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].bytesPerPoint, 9U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].bytesPerPoint, 1U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].repackBytesPerPoint, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].repackBytesPerPoint, 1U);
    EXPECT_FALSE(plan.summary().residencyBoundaries[0].requiresFullPointRecord);
    EXPECT_FALSE(plan.summary().residencyBoundaries[1].requiresFullPointRecord);
    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[0].payload);
    const auto* skewness =
        std::get_if<pdg::SkewnessProgram>(&plan.stages()[1].payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[2].payload);
    ASSERT_EQ(plan.stages().size(), 3U);
    ASSERT_NE(reader, nullptr);
    ASSERT_NE(skewness, nullptr);
    ASSERT_NE(writer, nullptr);
    EXPECT_TRUE(reader->optionFreeLasFamily);
    EXPECT_TRUE(writer->extraDimensionsAll);
    EXPECT_EQ(skewness->groundClass, 2U);
    EXPECT_EQ(skewness->otherClass, 1U);
    EXPECT_FALSE(skewness->onlyGround);
    EXPECT_TRUE(plan.stages()[1].descriptor.placementModel.empty());
    EXPECT_EQ(plan.stages()[1].descriptor.deviceToHostBytesPerInputPoint, 8U);
    EXPECT_EQ(plan.stages()[1].residentRegion, 0U);

    const pdg::RuntimePlacementFacts selectedFacts =
        directSkewnessFactsFor(plan, 450'000U);
    ASSERT_EQ(selectedFacts.boundaryExecutionFacts.size(), 2U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[0].transferBytesPerPoint,
              8U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[1].transferBytesPerPoint,
              8U);
    EXPECT_EQ(selectedFacts.stageScratchBytes[1],
              pdg::SkewnessExactDeviceScratchBytesPerPoint * 450'000U);

    const pdg::RuntimePlacementResult selected =
        pdg::buildRuntimePlacement(plan, selectedFacts, sm89Profile());
    EXPECT_EQ(selected.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::None);
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model, "skewness-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              8U * 450'000U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              8U * 450'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 0U);
    EXPECT_EQ(selected.estimate.untiledDeviceBytes,
              pdg::SkewnessExactDevicePeakBytesPerPoint * 450'000U);
    EXPECT_EQ(selected.estimate.peakDeviceBytes,
              pdg::SkewnessExactDevicePeakBytesPerPoint * 450'000U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directSkewnessFactsFor(plan, 400'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model, "skewness-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directSkewnessFactsFor(plan, 16'000'001U), sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectHagNnCount1Composition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectHagNnCount1Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].bytesPerPoint, 25U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].bytesPerPoint, 8U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].repackBytesPerPoint, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].repackBytesPerPoint, 8U);
    EXPECT_FALSE(plan.summary().residencyBoundaries[0].requiresFullPointRecord);
    EXPECT_FALSE(plan.summary().residencyBoundaries[1].requiresFullPointRecord);
    ASSERT_EQ(plan.stages().size(), 3U);
    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[0].payload);
    const auto* hagNn =
        std::get_if<pdg::HagNnProgram>(&plan.stages()[1].payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[2].payload);
    ASSERT_NE(reader, nullptr);
    ASSERT_NE(hagNn, nullptr);
    ASSERT_NE(writer, nullptr);
    EXPECT_TRUE(reader->optionFreeLasFamily);
    EXPECT_TRUE(writer->extraDimensionsAll);
    EXPECT_EQ(hagNn->count, 1U);
    EXPECT_EQ(plan.stages()[1].residentRegion, 0U);

    const pdg::RuntimePlacementFacts selectedFacts =
        directHagNnFactsFor(plan, 450'000U);
    ASSERT_EQ(selectedFacts.boundaryExecutionFacts.size(), 2U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[0].transferBytesPerPoint,
              25U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[1].transferBytesPerPoint,
              8U);
    EXPECT_EQ(selectedFacts.stageScratchBytes[1],
              pdg::HagNnCountOneExactDeviceScratchBytesPerPoint * 450'000U);

    const pdg::RuntimePlacementResult selected =
        pdg::buildRuntimePlacement(plan, selectedFacts, sm89Profile());
    EXPECT_EQ(selected.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::None);
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "hag-nn-count1-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              25U * 450'000U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              8U * 450'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 112U * 450'000U);
    EXPECT_EQ(selected.estimate.untiledDeviceBytes,
              pdg::HagNnCountOneExactDevicePeakBytesPerPoint * 450'000U);
    EXPECT_EQ(selected.estimate.peakDeviceBytes,
              pdg::HagNnCountOneExactDevicePeakBytesPerPoint * 450'000U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directHagNnFactsFor(plan, 400'002U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model,
              "hag-nn-count1-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directHagNnFactsFor(plan, 16'000'003U), sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    pdg::RuntimePlacementFacts capFacts =
        directHagNnFactsFor(plan, 16'000'002U);
    capFacts.deviceMemoryBudgetBytes =
        pdg::HagNnCountOneExactDevicePeakBytesPerPoint * 16'000'002U;
    const pdg::RuntimePlacementResult cap =
        pdg::buildRuntimePlacement(plan, capFacts, sm89Profile());
    ASSERT_TRUE(cap.available());
    EXPECT_EQ(cap.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(cap.estimate.peakDeviceBytes,
              pdg::HagNnCountOneExactDevicePeakBytesPerPoint * 16'000'002U);

    pdg::RuntimePlacementFacts exactBudget =
        directHagNnFactsFor(plan, 450'000U);
    exactBudget.deviceMemoryBudgetBytes =
        pdg::HagNnCountOneExactDevicePeakBytesPerPoint * 450'000U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, exactBudget, sm89Profile())
                  .estimate.choice,
              pdg::PlacementChoice::Device);
    --exactBudget.deviceMemoryBudgetBytes;
    const pdg::RuntimePlacementResult belowBudget =
        pdg::buildRuntimePlacement(plan, exactBudget, sm89Profile());
    EXPECT_EQ(belowBudget.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(belowBudget.estimate.reason,
              pdg::PlacementReason::DeviceMemoryBudgetExceeded);
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectHagDelaunayCount3Composition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectHagDelaunayCount3Pipeline, dimensions);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].bytesPerPoint, 25U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].bytesPerPoint, 8U);
    EXPECT_EQ(plan.summary().residencyBoundaries[0].repackBytesPerPoint, 0U);
    EXPECT_EQ(plan.summary().residencyBoundaries[1].repackBytesPerPoint, 8U);
    EXPECT_FALSE(plan.summary().residencyBoundaries[0].requiresFullPointRecord);
    EXPECT_FALSE(plan.summary().residencyBoundaries[1].requiresFullPointRecord);
    ASSERT_EQ(plan.stages().size(), 3U);
    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[0].payload);
    const auto* hagDelaunay =
        std::get_if<pdg::HagDelaunayProgram>(&plan.stages()[1].payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[2].payload);
    ASSERT_NE(reader, nullptr);
    ASSERT_NE(hagDelaunay, nullptr);
    ASSERT_NE(writer, nullptr);
    EXPECT_TRUE(reader->optionFreeLasFamily);
    EXPECT_TRUE(writer->extraDimensionsAll);
    EXPECT_EQ(hagDelaunay->count, 3U);
    EXPECT_EQ(plan.stages()[1].residentRegion, 0U);

    const pdg::RuntimePlacementFacts selectedFacts =
        directHagDelaunayFactsFor(plan, 500'001U);
    ASSERT_EQ(selectedFacts.boundaryExecutionFacts.size(), 2U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[0].transferBytesPerPoint,
              25U);
    EXPECT_EQ(selectedFacts.boundaryExecutionFacts[1].transferBytesPerPoint,
              8U);
    EXPECT_EQ(selectedFacts.stageScratchBytes[1],
              pdg::HagDelaunayCountThreeExactDeviceScratchBytesPerPoint *
                  500'001U);

    const pdg::RuntimePlacementResult selected =
        pdg::buildRuntimePlacement(plan, selectedFacts, sm89Profile());
    EXPECT_EQ(selected.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::None);
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "hag-delaunay-count3-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              25U * 500'001U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              8U * 500'001U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 112U * 500'001U);
    EXPECT_EQ(selected.estimate.untiledDeviceBytes,
              pdg::HagDelaunayCountThreeExactDevicePeakBytesPerPoint *
                  500'001U);
    EXPECT_EQ(selected.estimate.peakDeviceBytes,
              pdg::HagDelaunayCountThreeExactDevicePeakBytesPerPoint *
                  500'001U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directHagDelaunayFactsFor(plan, 500'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model,
              "hag-delaunay-count3-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directHagDelaunayFactsFor(plan, 16'000'003U), sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    pdg::RuntimePlacementFacts capFacts =
        directHagDelaunayFactsFor(plan, 16'000'002U);
    capFacts.deviceMemoryBudgetBytes =
        pdg::HagDelaunayCountThreeExactDevicePeakBytesPerPoint * 16'000'002U;
    const pdg::RuntimePlacementResult cap =
        pdg::buildRuntimePlacement(plan, capFacts, sm89Profile());
    ASSERT_TRUE(cap.available());
    EXPECT_EQ(cap.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(cap.estimate.peakDeviceBytes,
              pdg::HagDelaunayCountThreeExactDevicePeakBytesPerPoint *
                  16'000'002U);

    pdg::RuntimePlacementFacts exactBudget =
        directHagDelaunayFactsFor(plan, 500'001U);
    exactBudget.deviceMemoryBudgetBytes =
        pdg::HagDelaunayCountThreeExactDevicePeakBytesPerPoint * 500'001U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, exactBudget, sm89Profile())
                  .estimate.choice,
              pdg::PlacementChoice::Device);
    --exactBudget.deviceMemoryBudgetBytes;
    const pdg::RuntimePlacementResult belowBudget =
        pdg::buildRuntimePlacement(plan, exactBudget, sm89Profile());
    EXPECT_EQ(belowBudget.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(belowBudget.estimate.reason,
              pdg::PlacementReason::DeviceMemoryBudgetExceeded);
}

TEST(RuntimePlacement, RejectsDirectSkewnessShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.skewnessbalancing","ground_class":3,"other_class":1,"only_ground":false},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.skewnessbalancing","ground_class":2,"other_class":9,"only_ground":false},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.skewnessbalancing","ground_class":2,"other_class":1,"only_ground":true},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.skewnessbalancing"},{"type":"writers.las","filename":"out.las"}])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_EQ(
            pdg::buildRuntimePlacement(
                plan, directSkewnessFactsFor(plan, 1'000'000U), sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectSkewnessPipeline, dimensions);
    for (const pdg::RuntimePlacementFacts& outside :
         {
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.inputPointFormat = 6U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.inputRecordBytes = 34U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.outputRecordBytes = 34U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[0].transferBytesPerPoint = 7U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[1].packingBytesPerPoint = 1U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[1].deviceStagingBytesPerPoint =
                     35U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSkewnessFactsFor(plan, 1'000'000U);
                 --value.stageScratchBytes[1];
                 return value;
             }(),
         })
        EXPECT_EQ(
            pdg::buildRuntimePlacement(plan, outside, sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, RejectsDirectHagNnCount1ShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.hag_nn","count":2},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_nn","count":1,"class":9},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_nn","count":1,"max_distance":17.5},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_nn","count":1,"allow_extrapolation":false},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_nn","count":1},{"type":"writers.las","filename":"out.las"}])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_EQ(
            pdg::buildRuntimePlacement(
                plan, directHagNnFactsFor(plan, 1'000'000U), sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectHagNnCount1Pipeline, dimensions);
    for (const pdg::RuntimePlacementFacts& outside :
         {
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.inputPointFormat = 6U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.inputRecordBytes = 39U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.outputRecordBytes = 47U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[0].transferBytesPerPoint = 24U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[1].packingBytesPerPoint = 1U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagNnFactsFor(plan, 1'000'000U);
                 --value.stageScratchBytes[1];
                 return value;
             }(),
         })
        EXPECT_EQ(
            pdg::buildRuntimePlacement(plan, outside, sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, RejectsDirectHagDelaunayCount3ShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.hag_delaunay","count":3,"class":9},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_delaunay","count":3,"allow_extrapolation":false},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.hag_delaunay","count":3},{"type":"writers.las","filename":"out.las"}])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_EQ(
            pdg::buildRuntimePlacement(
                plan, directHagDelaunayFactsFor(plan, 1'000'000U),
                sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectHagDelaunayCount3Pipeline, dimensions);
    for (const pdg::RuntimePlacementFacts& outside :
         {
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.inputPointFormat = 6U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.inputRecordBytes = 39U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.outputRecordBytes = 47U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[0].transferBytesPerPoint = 24U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[1].packingBytesPerPoint = 1U;
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 --value.stageScratchBytes[1];
                 return value;
             }(),
             [&]
             {
                 auto value = directHagDelaunayFactsFor(plan, 1'000'000U);
                 value.tilePointCapacity = 999'999U;
                 return value;
             }(),
         })
        EXPECT_EQ(
            pdg::buildRuntimePlacement(plan, outside, sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::RuntimePlacementFacts invalidLanes =
        directHagDelaunayFactsFor(plan, 1'000'000U);
    invalidLanes.executorLaneCount = 2U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, invalidLanes, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);
}

TEST(RuntimePlacement, RejectsDirectSortShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.sort","dimension":"X","order":"ASC","algorithm":"NORMAL"},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.sort","dimension":"Z","order":"DESC","algorithm":"NORMAL"},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.sort","dimension":"Z","order":"ASC","algorithm":"STABLE"},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
            R"(["in.las",{"type":"filters.sort","dimension":"Z","order":"ASC","algorithm":"NORMAL"},"out.las"])",
            R"(["in.las",{"type":"filters.sort","dimension":"Z","order":"ASC","algorithm":"NORMAL"},{"type":"filters.ferry","dimensions":"Intensity=>UserData"},{"type":"writers.las","filename":"out.las","extra_dims":"all"}])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_EQ(
            pdg::buildRuntimePlacement(
                plan, directSortFactsFor(plan, 1'000'000U), sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(DirectSortPipeline, dimensions);
    for (const pdg::RuntimePlacementFacts& outside :
         {
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.inputPointFormat = 6U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.inputRecordBytes = 38U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.outputRecordBytes = 38U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[0].transferBytesPerPoint = 7U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[1].packingBytesPerPoint = 1U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 value.boundaryExecutionFacts[0].deviceStagingBytesPerPoint =
                     38U;
                 return value;
             }(),
             [&]
             {
                 auto value = directSortFactsFor(plan, 1'000'000U);
                 --value.stageScratchBytes[1];
                 return value;
             }(),
         })
        EXPECT_EQ(
            pdg::buildRuntimePlacement(plan, outside, sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, RejectsDirectNeighborClassifierShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.neighborclassifier","k":8},"out.las"])",
            R"(["in.las",{"type":"filters.neighborclassifier","k":7},{"type":"filters.ferry","dimensions":"Classification=>UserData"},"out.las"])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        const pdg::RuntimePlacementResult rejected = pdg::buildRuntimePlacement(
            plan, directNeighborClassifierFactsFor(plan, 1'000'000U),
            sm89Profile());
        EXPECT_FALSE(rejected.available()) << pipeline;
        EXPECT_EQ(
            rejected.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectNeighborClassifierPipeline, dimensions);
    for (const pdg::RuntimePlacementFacts& outside :
         {
             [&]
             {
                 pdg::RuntimePlacementFacts value =
                     directNeighborClassifierFactsFor(plan, 1'000'000U);
                 value.inputPointFormat = 6U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value =
                     directNeighborClassifierFactsFor(plan, 1'000'000U);
                 value.inputRecordBytes = 34U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value =
                     directNeighborClassifierFactsFor(plan, 1'000'000U);
                 value.outputRecordBytes = 34U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value =
                     directNeighborClassifierFactsFor(plan, 1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value =
                     directNeighborClassifierFactsFor(plan, 1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
         })
    {
        EXPECT_EQ(
            pdg::buildRuntimePlacement(plan, outside, sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    }

    std::vector<pdg::PlannedStage> wrongIndexStages = plan.stages();
    wrongIndexStages[1].deviceIndexBuildBytesPerPoint = 111U;
    const pdg::Plan wrongIndexPlan(std::move(wrongIndexStages), plan.summary());
    EXPECT_EQ(
        pdg::buildRuntimePlacement(
            wrongIndexPlan,
            directNeighborClassifierFactsFor(wrongIndexPlan, 1'000'000U),
            sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectApproximateCoplanarComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectApproximateCoplanarPipeline, dimensions);

    const pdg::RuntimePlacementResult selected = pdg::buildRuntimePlacement(
        plan, directApproximateCoplanarFactsFor(plan, 250'000U), sm89Profile());
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "approximatecoplanar-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_FALSE(selected.request.intrinsicSingleLaneExecutor);
    EXPECT_NEAR(selected.estimate.allHostPlacement.totalNanoseconds,
                933856785.6592948, 0.01);
    EXPECT_NEAR(selected.estimate.selectedPlacement.totalNanoseconds,
                239765579.61658925, 0.01);
    EXPECT_EQ(selected.estimate.hostToDeviceBytes, 25U * 250'000U);
    EXPECT_EQ(selected.estimate.deviceToHostBytes, 2U * 250'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 72U * 250'000U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 112U * 250'000U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directApproximateCoplanarFactsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directApproximateCoplanarFactsFor(plan, 20'000'000U),
        sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    ASSERT_TRUE(ordinary.available());
    ASSERT_EQ(ordinary.regionCalibrations.size(), 1U);
    EXPECT_EQ(ordinary.regionCalibrations[0].model, "approximatecoplanar");
}

TEST(RuntimePlacement, RejectsDirectApproximateCoplanarShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.approximatecoplanar","knn":9},{"type":"filters.ferry","dimensions":"Coplanar=>UserData"},"out.las"])",
            R"(["in.las",{"type":"filters.approximatecoplanar","knn":8,"thresh1":24.0},{"type":"filters.ferry","dimensions":"Coplanar=>UserData"},"out.las"])",
            R"(["in.las",{"type":"filters.approximatecoplanar","knn":8,"thresh2":5.0},{"type":"filters.ferry","dimensions":"Coplanar=>UserData"},"out.las"])",
            R"(["in.las",{"type":"filters.approximatecoplanar","knn":8},{"type":"filters.ferry","dimensions":"Coplanar=>Classification"},"out.las"])",
            R"(["in.las",{"type":"filters.approximatecoplanar","knn":8},"out.las"])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        const pdg::RuntimePlacementResult rejected = pdg::buildRuntimePlacement(
            plan, directApproximateCoplanarFactsFor(plan, 1'000'000U),
            sm89Profile());
        EXPECT_FALSE(rejected.available()) << pipeline;
        EXPECT_EQ(
            rejected.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectApproximateCoplanarPipeline, dimensions);
    pdg::RuntimePlacementFacts wrongLayout =
        directApproximateCoplanarFactsFor(plan, 1'000'000U);
    wrongLayout.inputRecordBytes = 34U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongLayout, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::RuntimePlacementFacts wrongFormat =
        directApproximateCoplanarFactsFor(plan, 1'000'000U);
    wrongFormat.inputPointFormat = 6U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongFormat, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    std::vector<pdg::PlannedStage> wrongIndexStages = plan.stages();
    wrongIndexStages[1].deviceIndexBuildBytesPerPoint = 111U;
    const pdg::Plan wrongIndexPlan(std::move(wrongIndexStages), plan.summary());
    EXPECT_EQ(
        pdg::buildRuntimePlacement(
            wrongIndexPlan,
            directApproximateCoplanarFactsFor(wrongIndexPlan, 1'000'000U),
            sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, SelectsOnlyMeasuredDirectOutlierNnDistanceComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectOutlierNnDistancePipeline, dimensions);

    const pdg::RuntimePlacementResult selected = pdg::buildRuntimePlacement(
        plan, directOutlierNnDistanceFactsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "outlier-nndistance-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directOutlierNnDistanceFactsFor(plan, 25'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model,
              "outlier-nndistance-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directOutlierNnDistanceFactsFor(plan, 20'000'000U),
        sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    EXPECT_FALSE(ordinary.available());
    EXPECT_EQ(ordinary.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MissingCalibrationModel);
}

TEST(RuntimePlacement, RejectsDirectOutlierNnDistanceShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":9,"multiplier":2.0,"class":7},{"type":"filters.nndistance","mode":"kth","k":10},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8,"multiplier":2.5,"class":7},{"type":"filters.nndistance","mode":"kth","k":10},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8,"multiplier":2.0,"class":8},{"type":"filters.nndistance","mode":"kth","k":10},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8,"multiplier":2.0,"class":7},{"type":"filters.nndistance","mode":"average","k":10},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8,"multiplier":2.0,"class":7},{"type":"filters.nndistance","mode":"kth","k":9},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"statistical","mean_k":8,"multiplier":2.0,"class":7},{"type":"filters.ferry","dimensions":"Classification=>UserData"},{"type":"filters.nndistance","mode":"kth","k":10},"out.las"])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        const pdg::RuntimePlacementResult rejected = pdg::buildRuntimePlacement(
            plan, directOutlierNnDistanceFactsFor(plan, 1'000'000U),
            sm89Profile());
        EXPECT_FALSE(rejected.available()) << pipeline;
        for (const pdg::PlacementRegionCalibration& calibration :
             rejected.regionCalibrations)
            EXPECT_NE(calibration.model, "outlier-nndistance-direct-compose")
                << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectOutlierNnDistancePipeline, dimensions);
    pdg::RuntimePlacementFacts wrongLayout =
        directOutlierNnDistanceFactsFor(plan, 1'000'000U);
    wrongLayout.inputRecordBytes = 34U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongLayout, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    std::vector<pdg::PlannedStage> wrongQueryStages = plan.stages();
    wrongQueryStages[1].deviceQueryBytesPerPoint = 144U;
    wrongQueryStages[2].deviceQueryBytesPerPoint = 144U;
    const pdg::Plan wrongQueryPlan(std::move(wrongQueryStages), plan.summary());
    const pdg::RuntimePlacementResult wrongQuery = pdg::buildRuntimePlacement(
        wrongQueryPlan,
        directOutlierNnDistanceFactsFor(wrongQueryPlan, 1'000'000U),
        sm89Profile());
    EXPECT_FALSE(wrongQuery.available());
    EXPECT_EQ(
        wrongQuery.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::PlanSummary wrongIndexSummary = plan.summary();
    wrongIndexSummary.indexBuilds = 2U;
    const pdg::Plan wrongIndexPlan(plan.stages(), std::move(wrongIndexSummary));
    const pdg::RuntimePlacementResult wrongIndex = pdg::buildRuntimePlacement(
        wrongIndexPlan,
        directOutlierNnDistanceFactsFor(wrongIndexPlan, 1'000'000U),
        sm89Profile());
    EXPECT_FALSE(wrongIndex.available());
    EXPECT_EQ(
        wrongIndex.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement,
     SelectsOnlyMeasuredDirectRadiusOutlierRadialDensityComposition)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        DirectRadiusOutlierRadialDensityPipeline, dimensions);

    const pdg::RuntimePlacementResult selected = pdg::buildRuntimePlacement(
        plan, directRadiusOutlierRadialDensityFactsFor(plan, 250'000U),
        sm89Profile());
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "radius-outlier-radialdensity-direct-compose");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);
    EXPECT_EQ(selected.estimate.hostToDeviceBytes, 24U * 250'000U);
    EXPECT_EQ(selected.estimate.deviceToHostBytes, 10U * 250'000U);
    EXPECT_EQ(selected.estimate.indexBuildBytes, 28U * 250'000U);
    EXPECT_EQ(selected.estimate.stageResultBytes, 4U * 250'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directRadiusOutlierRadialDensityFactsFor(plan, 50'000U),
        sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model,
              "radius-outlier-radialdensity-direct-compose");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult beyond = pdg::buildRuntimePlacement(
        plan, directRadiusOutlierRadialDensityFactsFor(plan, 4'000'001U),
        sm89Profile());
    ASSERT_TRUE(beyond.available());
    EXPECT_EQ(beyond.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(beyond.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    EXPECT_FALSE(ordinary.available());
    EXPECT_EQ(ordinary.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MissingCalibrationModel);
}

TEST(RuntimePlacement,
     RejectsDirectRadiusOutlierRadialDensityShapeAndLayoutDrift)
{
    for (
        const std::string_view pipeline : {
            R"(["in.las",{"type":"filters.outlier","method":"radius","radius":1.0,"min_k":2,"class":7},{"type":"filters.radialdensity","radius":1.01},{"type":"filters.assign","value":"UserData = 1 WHERE RadialDensity >= 0.2"},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"radius","radius":1.01,"min_k":3,"class":7},{"type":"filters.radialdensity","radius":1.01},{"type":"filters.assign","value":"UserData = 1 WHERE RadialDensity >= 0.2"},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"radius","radius":1.01,"min_k":2,"class":8},{"type":"filters.radialdensity","radius":1.01},{"type":"filters.assign","value":"UserData = 1 WHERE RadialDensity >= 0.2"},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"radius","radius":1.01,"min_k":2,"class":7},{"type":"filters.radialdensity","radius":1.0},{"type":"filters.assign","value":"UserData = 1 WHERE RadialDensity >= 0.2"},"out.las"])",
            R"(["in.las",{"type":"filters.outlier","method":"radius","radius":1.01,"min_k":2,"class":7},{"type":"filters.radialdensity","radius":1.01},{"type":"filters.assign","value":"UserData = 2 WHERE RadialDensity >= 0.2"},"out.las"])",
        })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        const pdg::RuntimePlacementResult rejected = pdg::buildRuntimePlacement(
            plan, directRadiusOutlierRadialDensityFactsFor(plan, 1'000'000U),
            sm89Profile());
        EXPECT_FALSE(rejected.available()) << pipeline;
        EXPECT_EQ(
            rejected.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope)
            << pipeline;
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        DirectRadiusOutlierRadialDensityPipeline, dimensions);
    pdg::RuntimePlacementFacts wrongFormat =
        directRadiusOutlierRadialDensityFactsFor(plan, 1'000'000U);
    wrongFormat.inputPointFormat = 6U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongFormat, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::RuntimePlacementFacts wrongLayout =
        directRadiusOutlierRadialDensityFactsFor(plan, 1'000'000U);
    wrongLayout.inputRecordBytes = 34U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongLayout, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    std::vector<pdg::PlannedStage> wrongQueryStages = plan.stages();
    wrongQueryStages[1].deviceQueryBytesPerPoint = 8U;
    const pdg::Plan wrongQueryPlan(std::move(wrongQueryStages), plan.summary());
    EXPECT_EQ(
        pdg::buildRuntimePlacement(wrongQueryPlan,
                                   directRadiusOutlierRadialDensityFactsFor(
                                       wrongQueryPlan, 1'000'000U),
                                   sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::PlanSummary wrongIndexSummary = plan.summary();
    wrongIndexSummary.indexBuilds = 2U;
    const pdg::Plan wrongIndexPlan(plan.stages(), std::move(wrongIndexSummary));
    EXPECT_EQ(
        pdg::buildRuntimePlacement(wrongIndexPlan,
                                   directRadiusOutlierRadialDensityFactsFor(
                                       wrongIndexPlan, 1'000'000U),
                                   sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, SelectsOnlyTheMeasuredDirectRadiusAssignExecutor)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectRadiusAssignPipeline, dimensions);
    const pdg::RuntimePlacementResult selected = pdg::buildRuntimePlacement(
        plan, directRadiusFactsFor(plan, 250'000U), sm89Profile());
    ASSERT_TRUE(selected.available());
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model, "radiusassign-direct");
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(selected.estimate.selectedRegionCount, 1U);
    EXPECT_NEAR(selected.estimate.allHostPlacement.totalNanoseconds,
                1985.1180564323188 * 250'000.0, 0.01);
    EXPECT_NEAR(selected.estimate.selectedPlacement.totalNanoseconds,
                305407938.01419646 + 68.33597116527429 * 250'000.0, 0.01);
    ASSERT_EQ(selected.estimate.boundaries.size(), 2U);
    EXPECT_EQ(selected.estimate.boundaries[0].predictedTransferBytes,
              25U * 250'000U);
    EXPECT_EQ(selected.estimate.boundaries[1].predictedTransferBytes,
              1U * 250'000U);
    EXPECT_EQ(selected.estimate.packingBytes, 0U);
    EXPECT_EQ(selected.estimate.configuredDeviceLaneCount, 1U);
    EXPECT_EQ(selected.estimate.activeDeviceLaneCount, 1U);
    EXPECT_TRUE(selected.request.intrinsicSingleLaneExecutor);

    const pdg::RuntimePlacementResult below = pdg::buildRuntimePlacement(
        plan, directRadiusFactsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(below.available());
    EXPECT_EQ(below.regionCalibrations[0].model, "radiusassign-direct");
    EXPECT_EQ(below.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(below.estimate.reason,
              pdg::PlacementReason::OutsideCalibrationEnvelope);

    const pdg::RuntimePlacementResult ordinary = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    ASSERT_TRUE(ordinary.available());
    EXPECT_EQ(ordinary.regionCalibrations[0].model, "radiusassign");
}

TEST(RuntimePlacement, RejectsDirectRadiusAssignShapeAndLayoutDrift)
{
    for (const std::string_view pipeline : {
             R"(["in.las", {"type":"filters.radiusassign","radius":3.0,
                  "is3d":true,"src_domain":"ReturnNumber[1:1]",
                  "reference_domain":"ReturnNumber[2:15]",
                  "update_expression":"UserData = 9"}, "out.las"])",
             R"(["in.las", {"type":"filters.radiusassign","radius":2.0,
                  "is3d":true,"src_domain":"ReturnNumber[1:1]",
                  "reference_domain":"ReturnNumber[2:15]",
                  "update_expression":"UserData = 8"}, "out.las"])",
             R"(["in.las", {"type":"filters.radiusassign","radius":2.0,
                  "is3d":true,"src_domain":"ReturnNumber[1:1]",
                  "reference_domain":"ReturnNumber[2:15]",
                  "update_expression":"UserData = 9"},
                  {"type":"filters.ferry",
                   "dimensions":"UserData=>Classification"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_EQ(
            pdg::buildRuntimePlacement(
                plan, directRadiusFactsFor(plan, 1'000'000U), sm89Profile())
                .unavailableReason,
            pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    }

    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(DirectRadiusAssignPipeline, dimensions);
    pdg::RuntimePlacementFacts wrongLayout =
        directRadiusFactsFor(plan, 1'000'000U);
    wrongLayout.inputRecordBytes = 34U;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, wrongLayout, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, BuildsFusedPointProgramRequestFromExactFacts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(FusedPointProgramPipeline, dimensions);
    pdg::RuntimePlacementFacts facts = factsFor(plan, 20'000'000U);
    facts.tilePointCapacity = 131'072U;

    const pdg::RuntimePlacementResult result =
        pdg::buildRuntimePlacement(plan, facts, sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model, "fused-point-program");
    ASSERT_EQ(result.request.stageInputPointCounts.size(),
              plan.stages().size());
    EXPECT_EQ(result.request.stageInputPointCounts[0], 0U);
    EXPECT_EQ(result.request.stageOutputPointCounts[0], facts.inputPointCount);
    for (std::size_t index = 1U; index < plan.stages().size(); ++index)
    {
        EXPECT_EQ(result.request.stageInputPointCounts[index],
                  facts.inputPointCount);
        EXPECT_EQ(result.request.stageOutputPointCounts[index],
                  facts.inputPointCount);
        EXPECT_EQ(result.request.stagePointCapacities[index],
                  facts.tilePointCapacity);
    }
    EXPECT_TRUE(result.request.stageCosts[1].calibrated);
    EXPECT_TRUE(result.request.stageCosts[3].calibrated);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[3].hostNanosecondsPerPoint, 0.0);
}

TEST(RuntimePlacement, UsesDescriptorDeclaredApproximateCoplanarModel)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.approximatecoplanar","knn":8},
             "out.las"])",
        dimensions);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 262'144U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model, "approximatecoplanar");
    EXPECT_EQ(result.request.stageCosts[1].minimumDevicePointCount, 131'072U);
}

TEST(RuntimePlacement, SmallPointProgramIsTheHostNegativeControl)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(FusedPointProgramPipeline, dimensions);
    const pdg::RuntimePlacementResult result =
        pdg::buildRuntimePlacement(plan, factsFor(plan, 1000U), sm89Profile());
    ASSERT_TRUE(result.available());
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Host);
    EXPECT_EQ(result.estimate.selectedRegionCount, 0U);
    EXPECT_TRUE(result.estimate.reason ==
                    pdg::PlacementReason::HostFasterOrEqual ||
                result.estimate.reason ==
                    pdg::PlacementReason::SharedDeviceTollNotAmortized);
}

TEST(RuntimePlacement, SelectsRegionsAcrossKnownRandomizeHostBoundary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry",
               "dimensions":"Classification=>UserData"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             {"type":"filters.randomize","seed":17},
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry",
               "dimensions":"Classification=>UserData"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             "out.las"])",
        dimensions);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 20'000'000U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 2U);
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(result.estimate.selectedRegionCount, 2U);
    EXPECT_TRUE(result.estimate.regions[0].selected);
    EXPECT_TRUE(result.estimate.regions[1].selected);

    pdg::RuntimePlacementFacts residentFacts = factsFor(plan, 20'000'000U);
    residentFacts.executorLaneCount = 2U;
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        residentFacts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = residentFacts.fallbackRecordBytes,
             .packingBytesPerPoint =
                 boundary.kind == pdg::ResidencyBoundaryKind::Upload
                     ? residentFacts.fallbackRecordBytes
                     : boundary.repackBytesPerPoint,
             .deviceStagingBytesPerPoint = residentFacts.fallbackRecordBytes});
    }
    const pdg::RuntimePlacementResult resident =
        pdg::buildRuntimePlacement(plan, residentFacts, sm89Profile());
    ASSERT_TRUE(resident.available());
    EXPECT_EQ(resident.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(resident.estimate.selectedRegionCount, 2U);
    ASSERT_EQ(resident.estimate.boundaries.size(),
              plan.summary().residencyBoundaries.size());
    for (const pdg::PlacementBoundaryEstimate& boundary :
         resident.estimate.boundaries)
    {
        EXPECT_EQ(boundary.predictedTransferBytes,
                  36U * residentFacts.inputPointCount);
        const pdg::ResidencyBoundary& planned =
            plan.summary().residencyBoundaries[boundary.boundaryId];
        const std::size_t expectedPacking =
            planned.kind == pdg::ResidencyBoundaryKind::Upload
                ? 36U
                : planned.repackBytesPerPoint;
        EXPECT_EQ(boundary.predictedPackingBytes,
                  expectedPacking * residentFacts.inputPointCount);
    }
}

TEST(RuntimePlacement, ValidatesExecutorBoundaryFactsAsACompleteIdTable)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry",
               "dimensions":"Classification=>UserData"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             "out.las"])",
        dimensions);
    pdg::RuntimePlacementFacts facts = factsFor(plan, 20'000'000U);
    facts.executorLaneCount = 2U;
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
        facts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint = 36U,
             .packingBytesPerPoint = boundaryId == 0U ? 36U : 5U,
             .deviceStagingBytesPerPoint = 36U});

    const pdg::RuntimePlacementResult accepted =
        pdg::buildRuntimePlacement(plan, facts, sm89Profile());
    ASSERT_TRUE(accepted.available());
    EXPECT_EQ(accepted.request.executorLaneCount, 2U);
    ASSERT_EQ(accepted.request.boundaryExecutionFacts.size(),
              facts.boundaryExecutionFacts.size());
    EXPECT_EQ(accepted.request.boundaryExecutionFacts.front().boundaryId, 0U);
    EXPECT_EQ(
        accepted.request.boundaryExecutionFacts.back().packingBytesPerPoint,
        5U);

    pdg::RuntimePlacementFacts missing = facts;
    missing.boundaryExecutionFacts.pop_back();
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, missing, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts zeroLanes = facts;
    zeroLanes.executorLaneCount = 0U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, zeroLanes, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts unsweptSingleLane = facts;
    unsweptSingleLane.executorLaneCount = 1U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, unsweptSingleLane, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts falseDirectBoundaryLane = facts;
    falseDirectBoundaryLane.executorLaneCount = 1U;
    falseDirectBoundaryLane.directRadiusAssignBoundaryExecutor = false;
    EXPECT_EQ(
        pdg::buildRuntimePlacement(plan, falseDirectBoundaryLane, sm89Profile())
            .unavailableReason,
        pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts tooManyLanes = facts;
    tooManyLanes.executorLaneCount = 7U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, tooManyLanes, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts duplicate = facts;
    duplicate.boundaryExecutionFacts.back().boundaryId = 0U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, duplicate, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts outOfRange = facts;
    outOfRange.boundaryExecutionFacts.back().boundaryId =
        plan.summary().residencyBoundaries.size();
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, outOfRange, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts zeroWidth = facts;
    zeroWidth.boundaryExecutionFacts.front().transferBytesPerPoint = 0U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, zeroWidth, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);

    pdg::RuntimePlacementFacts zeroStaging = facts;
    zeroStaging.boundaryExecutionFacts.front().deviceStagingBytesPerPoint = 0U;
    EXPECT_EQ(pdg::buildRuntimePlacement(plan, zeroStaging, sm89Profile())
                  .unavailableReason,
              pdg::RuntimePlacementUnavailableReason::InvalidRuntimeFacts);
}

TEST(RuntimePlacement, UsesSimpleFerryModelAndRejectsUnmeasuredPointProgram)
{
    pdg::DimensionRegistry ferryDimensions;
    const pdg::Plan ferry = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId"}, "out.las"])",
        ferryDimensions);
    const pdg::RuntimePlacementResult ferryResult = pdg::buildRuntimePlacement(
        ferry, factsFor(ferry, 20'000'000U), sm89Profile());
    ASSERT_TRUE(ferryResult.available());
    ASSERT_EQ(ferryResult.regionCalibrations.size(), 1U);
    EXPECT_EQ(ferryResult.regionCalibrations[0].model, "simple-ferry");
    EXPECT_EQ(ferryResult.estimate.choice, pdg::PlacementChoice::Host);

    pdg::DimensionRegistry assignDimensions;
    const pdg::Plan assign = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.assign",
             "value":"Classification = 7"}, "out.las"])",
        assignDimensions);
    const pdg::RuntimePlacementResult assignResult = pdg::buildRuntimePlacement(
        assign, factsFor(assign, 20'000'000U), sm89Profile());
    EXPECT_FALSE(assignResult.available());
    EXPECT_EQ(
        assignResult.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, RejectsPointProgramBeyondSingleLaunchEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.assign","value":[
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3",
             "Classification = Classification + 1 + 2 + 3"]},
             "out.las"])",
        dimensions);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 20'000'000U), sm89Profile());
    EXPECT_FALSE(result.available());
    EXPECT_EQ(
        result.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
}

TEST(RuntimePlacement, AdmitsOneDeclaredExpressionInsideAFusedRegion)
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
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 20'000'000U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    // A declared-predicate region resolves to the measured
    // ordered-point-program residual (D0064).
    EXPECT_EQ(result.regionCalibrations[0].model, "ordered-point-program");
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);

    pdg::RuntimePlacementFacts residentFacts = factsFor(plan, 20'000'000U);
    residentFacts.executorLaneCount = 2U;
    for (std::size_t boundaryId = 0;
         boundaryId < plan.summary().residencyBoundaries.size(); ++boundaryId)
    {
        const pdg::ResidencyBoundary& boundary =
            plan.summary().residencyBoundaries[boundaryId];
        const bool spill = boundary.kind == pdg::ResidencyBoundaryKind::Spill;
        residentFacts.boundaryExecutionFacts.push_back(
            {.boundaryId = boundaryId,
             .transferBytesPerPoint =
                 residentFacts.fallbackRecordBytes + (spill ? 1U : 0U),
             .packingBytesPerPoint = spill ? boundary.repackBytesPerPoint
                                           : residentFacts.fallbackRecordBytes,
             .deviceStagingBytesPerPoint =
                 residentFacts.fallbackRecordBytes + (spill ? 1U : 0U)});
    }
    const pdg::RuntimePlacementResult resident =
        pdg::buildRuntimePlacement(plan, residentFacts, sm89Profile());
    ASSERT_TRUE(resident.available());
    EXPECT_EQ(resident.estimate.choice, pdg::PlacementChoice::Device);
    ASSERT_EQ(resident.estimate.boundaries.size(),
              plan.summary().residencyBoundaries.size());
    for (const pdg::PlacementBoundaryEstimate& boundary :
         resident.estimate.boundaries)
    {
        const pdg::ResidencyBoundary& planned =
            plan.summary().residencyBoundaries[boundary.boundaryId];
        const bool spill = planned.kind == pdg::ResidencyBoundaryKind::Spill;
        EXPECT_EQ(boundary.pointCount, residentFacts.inputPointCount);
        EXPECT_EQ(boundary.predictedTransferBytes,
                  (36U + (spill ? 1U : 0U)) * residentFacts.inputPointCount);
    }

    // The V1 process chain adds one post-predicate assignment; its region
    // must stay inside the fused envelope bounds.
    pdg::DimensionRegistry processDimensions;
    const pdg::Plan processChain = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry","dimensions":"Classification=>UserData"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             {"type":"filters.expression","expression":"Intensity <= 10000"},
             {"type":"filters.assign","value":[
               "UserData = 3 WHERE Classification == 7"]},
             "out.las"])",
        processDimensions);
    ASSERT_EQ(processChain.summary().residentRegions, 1U);
    const pdg::RuntimePlacementResult processResult =
        pdg::buildRuntimePlacement(
            processChain, factsFor(processChain, 21'970'934U), sm89Profile());
    ASSERT_TRUE(processResult.available());
    ASSERT_EQ(processResult.regionCalibrations.size(), 1U);
    EXPECT_EQ(processResult.regionCalibrations[0].model,
              "ordered-point-program");
    EXPECT_EQ(processResult.estimate.choice, pdg::PlacementChoice::Device);
}

TEST(RuntimePlacement, KeepsUncomposedOrRepeatedCardinalityChangesUnavailable)
{
    // A coordinate predicate stays host-preferred, so its cardinality change
    // is not declared for device execution.
    pdg::DimensionRegistry hostDimensions;
    const pdg::Plan hostPredicate = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.expression","expression":"Z > 0"},
             "out.las"])",
        hostDimensions);
    const pdg::RuntimePlacementResult hostPredicateResult =
        pdg::buildRuntimePlacement(
            hostPredicate, factsFor(hostPredicate, 1000U), sm89Profile());
    EXPECT_FALSE(hostPredicateResult.available());
    EXPECT_EQ(
        hostPredicateResult.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::NonCardinalityPreservingStage);

    // A device-capable predicate without a measured fused region around it
    // has no calibration envelope of its own.
    pdg::DimensionRegistry standaloneDimensions;
    const pdg::Plan standalone = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.expression","expression":"Intensity <= 30000"},
             "out.las"])",
        standaloneDimensions);
    const pdg::RuntimePlacementResult standaloneResult =
        pdg::buildRuntimePlacement(standalone, factsFor(standalone, 1000U),
                                   sm89Profile());
    EXPECT_FALSE(standaloneResult.available());
    EXPECT_EQ(
        standaloneResult.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    pdg::DimensionRegistry repeatedDimensions;
    const pdg::Plan repeated = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry","dimensions":"Classification=>UserData"},
             {"type":"filters.expression","expression":"Intensity <= 30000"},
             {"type":"filters.expression","expression":"Intensity >= 100"},
             {"type":"filters.assign","value":[
               "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
               "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
             "out.las"])",
        repeatedDimensions);
    const pdg::RuntimePlacementResult repeatedResult =
        pdg::buildRuntimePlacement(repeated, factsFor(repeated, 20'000'000U),
                                   sm89Profile());
    EXPECT_FALSE(repeatedResult.available());
    EXPECT_EQ(
        repeatedResult.unavailableReason,
        pdg::RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);

    // The host-fallback `where` form stays refused. B0163 makes the reason
    // more precise: `hasOnlyOptions` excludes `where`, so this compiles to a
    // fallback stage with `native == false`, and its cardinality behaviour is
    // therefore *unknown* rather than known-bad. It now reports
    // `UnsupportedStage`. The refusal itself is unchanged — only the label,
    // which is the point of the split.
    pdg::DimensionRegistry whereDimensions;
    const pdg::Plan whereForm = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.expression","expression":"Z > 0",
              "where":"Intensity > 5"},
             "out.las"])",
        whereDimensions);
    const pdg::RuntimePlacementResult whereResult = pdg::buildRuntimePlacement(
        whereForm, factsFor(whereForm, 1000U), sm89Profile());
    EXPECT_FALSE(whereResult.available());
    EXPECT_EQ(whereResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::UnsupportedStage);
}

TEST(RuntimePlacement, AnchorsAComposedNeighborhoodRegionOnItsMeasuredModel)
{
    // V2: a shared-index neighborhood stage with trailing point-program
    // consumers forms one resident region. Per D0055, the region charges the
    // one measured non-point-program residual at its first device stage;
    // the point-program remainder is zero-incremental, not a mixed-model
    // rejection.
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.approximatecoplanar","knn":8},
             {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 21'970'934U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model, "approximatecoplanar");
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);

    // Below the measured envelope floor the region fails closed to host.
    const pdg::RuntimePlacementResult small = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 37'566U), sm89Profile());
    ASSERT_TRUE(small.available());
    EXPECT_EQ(small.estimate.choice, pdg::PlacementChoice::Host);

    // Two distinct measured models in one region remain a mixed-model
    // rejection.
    pdg::DimensionRegistry mixedDimensions;
    const pdg::Plan mixedBase = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.approximatecoplanar","knn":8},
             {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
             "out.las"])",
        mixedDimensions);
    std::vector<pdg::PlannedStage> stages = mixedBase.stages();
    stages[2].descriptor.placementModel = "transformation";
    const pdg::Plan mixed(std::move(stages), mixedBase.summary());
    const pdg::RuntimePlacementResult mixedResult = pdg::buildRuntimePlacement(
        mixed, factsFor(mixed, 21'970'934U), sm89Profile());
    EXPECT_FALSE(mixedResult.available());
    EXPECT_EQ(mixedResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels);
}

// B0187: the adjacent normal -> covariancefeatures pair a real features
// pipeline writes. B0186 traced r6-features' 1.003x to this pair being
// planner-assigned to device and then refused by D0077's mixed-models rule
// with no composition model covering it.
constexpr std::string_view NormalCovarianceComposePipeline = R"(["in.las",
  {"type":"filters.normal","knn":8},
  {"type":"filters.covariancefeatures","knn":8,"feature_set":"Dimensionality"},
  "out.las"])";

constexpr std::string_view NormalCovarianceExtraDimensionsAllPipeline = R"([
  {"type":"readers.las","filename":"in.las"},
  {"type":"filters.normal","knn":8},
  {"type":"filters.covariancefeatures","knn":8,
   "feature_set":"Dimensionality"},
  {"type":"writers.las","filename":"out.las","extra_dims":"all"}
])";

constexpr std::string_view NormalCovarianceLazExtraDimensionsAllPipeline = R"([
  {"type":"readers.las","filename":"in.laz"},
  {"type":"filters.normal","knn":8},
  {"type":"filters.covariancefeatures","knn":8,
   "feature_set":"Dimensionality"},
  {"type":"writers.las","filename":"out.laz","compression":true,
   "extra_dims":"all"}
])";

TEST(RuntimePlacement, UsesOnlyTheMeasuredNormalCovarianceCompositionModel)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(NormalCovarianceComposePipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const auto originalPlainFacts = [&plan](std::size_t pointCount)
    {
        pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
        facts.inputPointFormat = 7U;
        facts.inputCompressed = false;
        facts.outputCompressed = false;
        facts.inputRecordBytes = 36U;
        facts.outputRecordBytes = 36U;
        return facts;
    };
    const auto ahn4PlainFacts = [&plan](std::size_t pointCount)
    {
        pdg::RuntimePlacementFacts facts = factsFor(plan, pointCount);
        facts.inputPointFormat = 8U;
        facts.inputCompressed = true;
        facts.outputCompressed = false;
        facts.inputRecordBytes = 44U;
        facts.outputRecordBytes = 36U;
        return facts;
    };

    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, originalPlainFacts(1'000'000U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model,
              "normal-covariancefeatures-compose");
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].hostNanosecondsPerPoint,
                     8428.236852266);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].deviceNanosecondsPerPoint,
                     1087.699620198);
    EXPECT_TRUE(result.request.stageCosts[2].calibrated);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[2].hostNanosecondsPerPoint, 0.0);

    // The reference profile retains B0187's measured 50K floor and ends at
    // B0280's independent 47,478,228-point exact-machine row. The built-in
    // profile must not shadow a separately calibrated composition merely
    // because it has higher tier precedence.
    const pdg::RuntimePlacementResult floorRow = pdg::buildRuntimePlacement(
        plan, originalPlainFacts(50'000U), sm89Profile());
    ASSERT_TRUE(floorRow.available());
    EXPECT_EQ(floorRow.estimate.choice, pdg::PlacementChoice::Device);
    const pdg::RuntimePlacementResult largeMeasuredRow = pdg::buildRuntimePlacement(
        plan, ahn4PlainFacts(47'478'228U), sm89Profile());
    ASSERT_TRUE(largeMeasuredRow.available());
    EXPECT_EQ(largeMeasuredRow.estimate.choice,
              pdg::PlacementChoice::Device);
    const pdg::RuntimePlacementResult beyondMeasuredMaximum =
        pdg::buildRuntimePlacement(plan, ahn4PlainFacts(47'478'229U),
                                   sm89Profile());
    ASSERT_TRUE(beyondMeasuredMaximum.available());
    EXPECT_EQ(beyondMeasuredMaximum.estimate.choice,
              pdg::PlacementChoice::Host);

    // Neither model presence nor the point-count envelope admits unrelated
    // physical layouts. These are exact functional fallbacks but lack matching
    // complete-process performance evidence for automatic placement.
    for (const pdg::RuntimePlacementFacts& unsupported :
         {
             [&]
             {
                 pdg::RuntimePlacementFacts value = originalPlainFacts(1'000'000U);
                 value.inputPointFormat = 6U;
                 value.inputRecordBytes = 30U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = originalPlainFacts(1'000'000U);
                 value.inputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = ahn4PlainFacts(1'000'000U);
                 value.inputRecordBytes = 45U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = ahn4PlainFacts(1'000'000U);
                 value.inputPointFormat = 9U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = originalPlainFacts(1'000'000U);
                 value.outputCompressed = true;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = originalPlainFacts(1'000'000U);
                 value.outputRecordBytes = 37U;
                 return value;
             }(),
         })
    {
        const pdg::RuntimePlacementResult refused =
            pdg::buildRuntimePlacement(plan, unsupported, sm89Profile());
        EXPECT_FALSE(refused.available());
        EXPECT_EQ(
            refused.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels);
    }

    // Every neighbouring shape stays a mixed-model rejection. knn is pinned
    // because the model carries no neighbour-count term; `mode` and
    // `feature_set` change the eigensystem work the residual was fitted to.
    const std::vector<std::string_view> rejected{
        R"(["in.las",{"type":"filters.normal","knn":9},{"type":"filters.covariancefeatures","knn":9,"feature_set":"Dimensionality"},"out.las"])",
        R"(["in.las",{"type":"filters.normal","knn":8},{"type":"filters.covariancefeatures","knn":12,"feature_set":"Dimensionality"},"out.las"])",
        R"(["in.las",{"type":"filters.normal","knn":8},{"type":"filters.covariancefeatures","knn":8,"mode":"raw","feature_set":"Dimensionality"},"out.las"])"};
    for (std::string_view text : rejected)
    {
        pdg::DimensionRegistry other;
        const pdg::Plan rejectedPlan = pdg::compilePipeline(text, other);
        const pdg::RuntimePlacementResult refused = pdg::buildRuntimePlacement(
            rejectedPlan, originalPlainFacts(1'000'000U), sm89Profile());
        EXPECT_FALSE(refused.available()) << text;
        EXPECT_EQ(
            refused.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels)
            << text;
    }

    // A different feature set also refuses, but earlier and for its own
    // reason, so this pins only that it is refused rather than which gate
    // caught it first.
    {
        pdg::DimensionRegistry other;
        const pdg::Plan densityPlan = pdg::compilePipeline(
            R"(["in.las",{"type":"filters.normal","knn":8},{"type":"filters.covariancefeatures","knn":8,"feature_set":"Density"},"out.las"])",
            other);
        const pdg::RuntimePlacementResult refused = pdg::buildRuntimePlacement(
            densityPlan, originalPlainFacts(1'000'000U), sm89Profile());
        EXPECT_FALSE(refused.available());
    }
}

TEST(RuntimePlacement,
     UsesTheSeparateExtraDimensionsNormalCovarianceModelAcrossQualifiedLayouts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        NormalCovarianceExtraDimensionsAllPipeline, dimensions);
    ASSERT_TRUE(plan.summary().allStagesNative);

    pdg::RuntimePlacementFacts measured = factsFor(plan, 1'000'000U);
    measured.inputPointFormat = 7U;
    measured.inputCompressed = true;
    measured.outputCompressed = false;
    measured.inputRecordBytes = 36U;
    measured.outputRecordBytes = 100U;
    const pdg::RuntimePlacementResult selected =
        pdg::buildRuntimePlacement(plan, measured, sm89Profile());
    ASSERT_TRUE(selected.available());
    EXPECT_EQ(selected.estimate.choice, pdg::PlacementChoice::Device);
    ASSERT_EQ(selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(selected.regionCalibrations[0].model,
              "normal-covariancefeatures-compose-extradims");

    pdg::DimensionRegistry lazDimensions;
    const pdg::Plan lazPlan = pdg::compilePipeline(
        NormalCovarianceLazExtraDimensionsAllPipeline, lazDimensions);
    ASSERT_TRUE(lazPlan.summary().allStagesNative);
    pdg::RuntimePlacementFacts lazMeasured = factsFor(lazPlan, 1'000'000U);
    lazMeasured.inputPointFormat = 7U;
    lazMeasured.inputCompressed = true;
    lazMeasured.outputCompressed = true;
    lazMeasured.inputRecordBytes = 36U;
    lazMeasured.outputRecordBytes = 100U;
    const pdg::RuntimePlacementResult lazSelected =
        pdg::buildRuntimePlacement(lazPlan, lazMeasured, sm89Profile());
    ASSERT_TRUE(lazSelected.available());
    EXPECT_EQ(lazSelected.estimate.choice, pdg::PlacementChoice::Device);
    ASSERT_EQ(lazSelected.regionCalibrations.size(), 1U);
    EXPECT_EQ(lazSelected.regionCalibrations[0].model,
              "normal-covariancefeatures-compose-extradims");

    pdg::RuntimePlacementFacts veil = lazMeasured;
    veil.inputPointCount = 35'976'465U;
    veil.inputPointFormat = 6U;
    veil.inputCompressed = true;
    veil.outputCompressed = true;
    veil.inputRecordBytes = 30U;
    veil.outputRecordBytes = 100U;
    const pdg::RuntimePlacementResult veilSelected =
        pdg::buildRuntimePlacement(lazPlan, veil, sm89Profile());
    ASSERT_TRUE(veilSelected.available());
    EXPECT_EQ(veilSelected.estimate.choice, pdg::PlacementChoice::Device);
    ASSERT_EQ(veilSelected.regionCalibrations.size(), 1U);
    EXPECT_EQ(veilSelected.regionCalibrations[0].model,
              "normal-covariancefeatures-compose-extradims");

    pdg::RuntimePlacementFacts ahn4 = lazMeasured;
    ahn4.inputPointCount = 47'478'228U;
    ahn4.inputPointFormat = 8U;
    ahn4.inputCompressed = true;
    ahn4.outputCompressed = true;
    ahn4.inputRecordBytes = 44U;
    ahn4.outputRecordBytes = 120U;
    const pdg::RuntimePlacementResult ahn4Selected =
        pdg::buildRuntimePlacement(lazPlan, ahn4, sm89Profile());
    ASSERT_TRUE(ahn4Selected.available());
    EXPECT_EQ(ahn4Selected.estimate.choice, pdg::PlacementChoice::Device);
    ASSERT_EQ(ahn4Selected.regionCalibrations.size(), 1U);
    EXPECT_EQ(ahn4Selected.regionCalibrations[0].model,
              "normal-covariancefeatures-compose-extradims");

    // A qualified physical tuple uses the profile's calibrated cardinality
    // curve; it is not restricted to the identity/count of the retained
    // full-size corpus tile. Keep interior counts selected so the stale-profile
    // failure cannot recur merely because another ordinary tile is smaller.
    for (const std::size_t points : {274'625U, 4'000'000U})
    {
        for (pdg::RuntimePlacementFacts interior : {veil, ahn4})
        {
            interior.inputPointCount = points;
            const pdg::RuntimePlacementResult result =
                pdg::buildRuntimePlacement(lazPlan, interior, sm89Profile());
            ASSERT_TRUE(result.available());
            EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);
            ASSERT_EQ(result.regionCalibrations.size(), 1U);
            EXPECT_EQ(result.regionCalibrations[0].model,
                      "normal-covariancefeatures-compose-extradims");
        }
    }

    for (const std::size_t points : {249'999U, 47'478'229U})
    {
        pdg::RuntimePlacementFacts outside = veil;
        outside.inputPointCount = points;
        const pdg::RuntimePlacementResult host =
            pdg::buildRuntimePlacement(lazPlan, outside, sm89Profile());
        ASSERT_TRUE(host.available());
        EXPECT_EQ(host.estimate.choice, pdg::PlacementChoice::Host);
        EXPECT_EQ(host.estimate.reason,
                  pdg::PlacementReason::OutsideCalibrationEnvelope);
    }

    // Model presence is not a wildcard for width/compression cross-products.
    // Even an interior value next to a measured tuple remains fail-closed.
    for (const pdg::RuntimePlacementFacts& unsupported :
         {
             [&]
             {
                 pdg::RuntimePlacementFacts value = lazMeasured;
                 value.inputPointFormat = 5U;
                 value.inputRecordBytes = 63U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = veil;
                 value.inputRecordBytes = 31U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = veil;
                 value.outputCompressed = false;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = ahn4;
                 value.inputRecordBytes = 38U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = ahn4;
                 value.inputRecordBytes = 63U;
                 value.outputRecordBytes = 127U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = lazMeasured;
                 value.inputPointCount = 999'999U;
                 return value;
             }(),
             [&]
             {
                 pdg::RuntimePlacementFacts value = lazMeasured;
                 value.inputCompressed = false;
                 return value;
             }(),
         })
    {
        const pdg::RuntimePlacementResult refused = pdg::buildRuntimePlacement(
            lazPlan, unsupported, sm89Profile());
        EXPECT_FALSE(refused.available());
        EXPECT_EQ(
            refused.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels);
    }

    // A compressed extra_dims=all sink changes publication cost.  No other
    // composition model may inherit B0224's one measured normal/covariance
    // row merely because the writer is functionally native.
    for (
        std::string_view text : {
            R"([{"type":"readers.las","filename":"in.laz"},{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},{"type":"writers.las","filename":"out.laz","compression":true,"extra_dims":"all"}])",
            R"([{"type":"readers.las","filename":"in.laz"},{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},{"type":"writers.las","filename":"out.laz","compression":true,"extra_dims":"all"}])",
        })
    {
        pdg::DimensionRegistry otherDimensions;
        const pdg::Plan other = pdg::compilePipeline(text, otherDimensions);
        ASSERT_TRUE(other.summary().allStagesNative) << text;
        pdg::RuntimePlacementFacts otherFacts = factsFor(other, 1'000'000U);
        otherFacts.inputPointFormat = 7U;
        otherFacts.inputCompressed = true;
        otherFacts.outputCompressed = true;
        otherFacts.inputRecordBytes = 36U;
        otherFacts.outputRecordBytes = 100U;
        const pdg::RuntimePlacementResult refused =
            pdg::buildRuntimePlacement(other, otherFacts, sm89Profile());
        EXPECT_FALSE(refused.available()) << text;
        EXPECT_EQ(
            refused.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels)
            << text;
    }
}

// `always_up` is deliberately not pinned: it normalizes a computed normal's
// sign per point after the neighbourhood work and cannot move the cost the
// model predicts. B0183 recorded what pinning such an option costs.
TEST(RuntimePlacement, NormalCovarianceCompositionIgnoresAlwaysUp)
{
    for (
        std::string_view text :
        {R"(["in.las",{"type":"filters.normal","knn":8,"always_up":true},{"type":"filters.covariancefeatures","knn":8,"feature_set":"Dimensionality"},"out.las"])",
         R"(["in.las",{"type":"filters.normal","knn":8,"always_up":false},{"type":"filters.covariancefeatures","knn":8,"feature_set":"Dimensionality"},"out.las"])"})
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(text, dimensions);
        pdg::RuntimePlacementFacts facts = factsFor(plan, 1'000'000U);
        facts.inputPointFormat = 7U;
        facts.inputCompressed = false;
        facts.outputCompressed = false;
        facts.inputRecordBytes = 36U;
        facts.outputRecordBytes = 36U;
        const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
            plan, facts, sm89Profile());
        ASSERT_TRUE(result.available()) << text;
        ASSERT_EQ(result.regionCalibrations.size(), 1U) << text;
        EXPECT_EQ(result.regionCalibrations[0].model,
                  "normal-covariancefeatures-compose")
            << text;
    }
}

TEST(RuntimePlacement, UsesOnlyTheMeasuredEigenFamilyCompositionModel)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(EigenFamilyComposePipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model, "eigen-family-compose");
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(result.estimate.selectedRegionCount, 1U);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].hostFixedNanoseconds,
                     120004600.90127563);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].hostNanosecondsPerPoint,
                     14659.538986578957);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].deviceFixedNanoseconds,
                     155357777.57434177);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].deviceNanosecondsPerPoint,
                     1234.7451449457265);
    for (std::size_t stage : {2U, 3U, 4U})
    {
        EXPECT_TRUE(result.request.stageCosts[stage].calibrated);
        EXPECT_DOUBLE_EQ(
            result.request.stageCosts[stage].hostNanosecondsPerPoint, 0.0);
        EXPECT_DOUBLE_EQ(
            result.request.stageCosts[stage].deviceNanosecondsPerPoint, 0.0);
    }

    const pdg::RuntimePlacementResult small = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(small.available());
    EXPECT_EQ(small.estimate.choice, pdg::PlacementChoice::Host);
    const pdg::RuntimePlacementResult beyondMeasuredMaximum =
        pdg::buildRuntimePlacement(plan, factsFor(plan, 20'000'000U),
                                   sm89Profile());
    ASSERT_TRUE(beyondMeasuredMaximum.available());
    EXPECT_EQ(beyondMeasuredMaximum.estimate.choice,
              pdg::PlacementChoice::Host);

    const std::vector<std::string_view>
        outsideEnvelope{
            R"(["in.las",{"type":"filters.normal","knn":11,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":11,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":11,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":true},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":false},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"sqrt","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"linearity"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 11","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Intensity = Curvature * 1000","Classification = Linearity * 10","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100","PointSourceId = Planarity * 10"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.assign","value":["Classification = NormalX * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
            R"(["in.las",{"type":"filters.normal","knn":12,"always_up":false},{"type":"filters.eigenvalues","knn":12,"normalize":true},{"type":"filters.covariancefeatures","knn":12,"mode":"raw","feature_set":"dimensionality"},{"type":"filters.approximatecoplanar","knn":12},{"type":"filters.assign","value":["Classification = Linearity * 10","Intensity = Curvature * 1000","UserData = Eigenvalue0 * 100"]},"out.las"])",
        };
    for (std::string_view pipeline : outsideEnvelope)
    {
        pdg::DimensionRegistry rejectedDimensions;
        const pdg::Plan rejected =
            pdg::compilePipeline(pipeline, rejectedDimensions);
        const pdg::RuntimePlacementResult rejectedResult =
            pdg::buildRuntimePlacement(rejected, factsFor(rejected, 1'000'000U),
                                       sm89Profile());
        EXPECT_FALSE(rejectedResult.available()) << pipeline;
        EXPECT_EQ(
            rejectedResult.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels)
            << pipeline;
    }
}

TEST(RuntimePlacement, UsesOnlyTheMeasuredRankOptimalCompositionModel)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan =
        pdg::compilePipeline(RankOptimalComposePipeline, dimensions);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::RuntimePlacementResult result = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 1'000'000U), sm89Profile());
    ASSERT_TRUE(result.available());
    ASSERT_EQ(result.regionCalibrations.size(), 1U);
    EXPECT_EQ(result.regionCalibrations[0].model, "rank-optimal-compose");
    EXPECT_EQ(result.estimate.choice, pdg::PlacementChoice::Device);
    EXPECT_EQ(result.estimate.selectedRegionCount, 1U);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].hostFixedNanoseconds,
                     133650155.04406099);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].hostNanosecondsPerPoint,
                     11448.442351895543);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].deviceFixedNanoseconds,
                     240029160.187007);
    EXPECT_DOUBLE_EQ(result.request.stageCosts[1].deviceNanosecondsPerPoint,
                     1186.5275421400943);
    for (std::size_t stage : {2U, 3U})
    {
        EXPECT_TRUE(result.request.stageCosts[stage].calibrated);
        EXPECT_DOUBLE_EQ(
            result.request.stageCosts[stage].hostNanosecondsPerPoint, 0.0);
        EXPECT_DOUBLE_EQ(
            result.request.stageCosts[stage].deviceNanosecondsPerPoint, 0.0);
    }

    const pdg::RuntimePlacementResult small = pdg::buildRuntimePlacement(
        plan, factsFor(plan, 50'000U), sm89Profile());
    ASSERT_TRUE(small.available());
    EXPECT_EQ(small.estimate.choice, pdg::PlacementChoice::Host);
    const pdg::RuntimePlacementResult beyondMeasuredMaximum =
        pdg::buildRuntimePlacement(plan, factsFor(plan, 20'000'000U),
                                   sm89Profile());
    ASSERT_TRUE(beyondMeasuredMaximum.available());
    EXPECT_EQ(beyondMeasuredMaximum.estimate.choice,
              pdg::PlacementChoice::Host);

    const std::vector<std::string_view> outsideEnvelope{
        R"(["in.las",{"type":"filters.estimaterank","knn":13,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.02},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":9,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":13},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank + 1","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Intensity = OptimalKNN","Classification = Rank","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN"]},"out.las"])",
        R"(["in.las",{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.approximatecoplanar","knn":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
    };
    for (std::string_view pipeline : outsideEnvelope)
    {
        pdg::DimensionRegistry rejectedDimensions;
        const pdg::Plan rejected =
            pdg::compilePipeline(pipeline, rejectedDimensions);
        const pdg::RuntimePlacementResult rejectedResult =
            pdg::buildRuntimePlacement(rejected, factsFor(rejected, 1'000'000U),
                                       sm89Profile());
        EXPECT_FALSE(rejectedResult.available()) << pipeline;
        EXPECT_EQ(
            rejectedResult.unavailableReason,
            pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels)
            << pipeline;
    }

    // Even an identity transform invalidates XYZ and splits the measured
    // adjacency; it must never inherit the one-region composition residual.
    pdg::DimensionRegistry invalidatedDimensions;
    const pdg::Plan invalidated = pdg::compilePipeline(
        R"(["in.las",{"type":"filters.estimaterank","knn":14,"thresh":0.01},{"type":"filters.transformation","matrix":"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"},{"type":"filters.optimalneighborhood","min_k":10,"max_k":14},{"type":"filters.assign","value":["Classification = Rank","Intensity = OptimalKNN","PointSourceId = OptimalRadius"]},"out.las"])",
        invalidatedDimensions);
    const pdg::RuntimePlacementResult invalidatedResult =
        pdg::buildRuntimePlacement(
            invalidated, factsFor(invalidated, 1'000'000U), sm89Profile());
    for (const pdg::PlacementRegionCalibration& calibration :
         invalidatedResult.regionCalibrations)
        EXPECT_NE(calibration.model, "rank-optimal-compose");
}

TEST(RuntimePlacement, RejectsMissingAndMixedDescriptorModels)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan missing = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.colorinterp","dimension":"Z"},
             "out.las"])",
        dimensions);
    const pdg::RuntimePlacementResult missingResult =
        pdg::buildRuntimePlacement(missing, factsFor(missing, 1000U),
                                   sm89Profile());
    EXPECT_FALSE(missingResult.available());
    EXPECT_EQ(missingResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MissingCalibrationModel);

    pdg::DimensionRegistry mixedDimensions;
    const pdg::Plan base = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.assign","value":"Tmp = Intensity"},
             {"type":"filters.ferry","dimensions":"Tmp=>Classification"},
             "out.las"])",
        mixedDimensions);
    // Two distinct measured (non-point-program) models in one region reject;
    // a point-program stage joining one measured model anchors instead
    // (D0055, covered separately).
    std::vector<pdg::PlannedStage> stages = base.stages();
    stages[1].descriptor.placementModel = "transformation";
    stages[2].descriptor.placementModel = "approximatecoplanar";
    const pdg::Plan mixed(std::move(stages), base.summary());
    const pdg::RuntimePlacementResult mixedResult = pdg::buildRuntimePlacement(
        mixed, factsFor(mixed, 1000U), sm89Profile());
    EXPECT_FALSE(mixedResult.available());
    EXPECT_EQ(mixedResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::MixedCalibrationModels);

    std::vector<pdg::PlannedStage> unknownStages = base.stages();
    unknownStages[1].descriptor.placementModel = "not-calibrated";
    unknownStages[2].descriptor.placementModel = "not-calibrated";
    const pdg::Plan unknown(std::move(unknownStages), base.summary());
    const pdg::RuntimePlacementResult unknownResult =
        pdg::buildRuntimePlacement(unknown, factsFor(unknown, 1000U),
                                   sm89Profile());
    EXPECT_FALSE(unknownResult.available());
    EXPECT_EQ(unknownResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::UnknownCalibrationModel);
}

TEST(RuntimePlacement, RejectsCopiedProfileAndNonLinearTopology)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId"}, "out.las"])",
        dimensions);
    const pdg::PlacementCalibrationProfile copied = sm89Profile();
    const pdg::RuntimePlacementResult copiedResult =
        pdg::buildRuntimePlacement(plan, factsFor(plan, 1000U), copied);
    EXPECT_FALSE(copiedResult.available());
    EXPECT_EQ(copiedResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::ProfileNotExact);

    pdg::DimensionRegistry branchDimensions;
    const pdg::Plan branched = pdg::compilePipeline(
        R"([
            {"type":"readers.las","filename":"in.las","tag":"source"},
            {"type":"filters.ferry","dimensions":"Intensity=>PointSourceId",
             "inputs":"source","tag":"left"},
            {"type":"filters.ferry","dimensions":"Intensity=>Classification",
             "inputs":"source","tag":"right"},
            {"type":"writers.las","filename":"out.las","inputs":"right"}
        ])",
        branchDimensions);
    const pdg::RuntimePlacementResult branchedResult =
        pdg::buildRuntimePlacement(branched, factsFor(branched, 1000U),
                                   sm89Profile());
    EXPECT_FALSE(branchedResult.available());
    EXPECT_EQ(branchedResult.unavailableReason,
              pdg::RuntimePlacementUnavailableReason::UnsupportedTopology);
}
