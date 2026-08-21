#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace
{
bool contains(const std::vector<pdg::DimensionId>& values,
              pdg::DimensionId value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}
} // unnamed namespace

TEST(PipelinePlan, CompilesNativeFerryDagAndResourceSummary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"({
            "pipeline": [
                {"type":"readers.las", "filename":"input.las",
                 "tag":"source"},
                {"type":"filters.ferry",
                 "dimensions":["Intensity=>PointSourceId", "=>Scratch"],
                 "inputs":"source", "tag":"copy"},
                {"type":"writers.las", "filename":"output.las",
                 "inputs":["copy"]}
            ]
        })",
        dimensions);

    const auto& stages = plan.stages();
    ASSERT_EQ(stages.size(), 3U);
    EXPECT_EQ(stages[0].role, pdg::StageRole::Reader);
    EXPECT_EQ(stages[1].role, pdg::StageRole::Filter);
    EXPECT_EQ(stages[2].role, pdg::StageRole::Writer);
    EXPECT_TRUE(stages[0].inputs.empty());
    EXPECT_EQ(stages[1].inputs, std::vector<std::size_t>({0}));
    EXPECT_EQ(stages[2].inputs, std::vector<std::size_t>({1}));
    EXPECT_EQ(stages[1].descriptor.kind, pdg::StageKind::Pointwise);
    EXPECT_EQ(stages[1].preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stages[1].descriptor.placementModel.empty());

    const auto& ferry = std::get<pdg::FerryProgram>(stages[1].payload);
    ASSERT_EQ(ferry.copies.size(), 2U);
    EXPECT_TRUE(ferry.copies[0].hasSource);
    EXPECT_EQ(ferry.copies[0].source,
              pdg::DimensionId(pdg::StandardDimension::Intensity));
    EXPECT_EQ(ferry.copies[0].destination,
              pdg::DimensionId(pdg::StandardDimension::PointSourceId));
    EXPECT_FALSE(ferry.copies[0].destinationCreated);
    EXPECT_FALSE(ferry.copies[1].hasSource);
    EXPECT_TRUE(ferry.copies[1].destinationCreated);

    const auto& scratch = dimensions.require("Scratch");
    EXPECT_EQ(scratch.type, pdg::DimensionType::Double);
    EXPECT_EQ(ferry.copies[1].destination, scratch.id);

    const auto intensity = pdg::DimensionId(pdg::StandardDimension::Intensity);
    const auto pointSource =
        pdg::DimensionId(pdg::StandardDimension::PointSourceId);
    EXPECT_TRUE(contains(stages[1].liveBefore, intensity));
    EXPECT_TRUE(contains(stages[1].liveBefore, pointSource));
    EXPECT_FALSE(contains(stages[1].liveBefore, scratch.id));
    EXPECT_FALSE(contains(stages[1].liveAfter, intensity));
    EXPECT_TRUE(contains(stages[1].liveAfter, pointSource));
    EXPECT_FALSE(contains(stages[1].liveAfter, scratch.id));
    EXPECT_EQ(stages[1].residentRegion, 0U);
    EXPECT_TRUE(contains(stages[1].deviceRelease, intensity));
    EXPECT_TRUE(contains(stages[1].deviceRelease, scratch.id));
    EXPECT_FALSE(contains(stages[1].deviceRelease, pointSource));

    const pdg::PlanSummary& summary = plan.summary();
    EXPECT_TRUE(summary.allStagesNative);
    EXPECT_TRUE(summary.fallbackReasons.empty());
    EXPECT_EQ(summary.touchedDimensions.size(), 3U);
    EXPECT_EQ(summary.bytesPerPoint, 12U);
    EXPECT_EQ(summary.peakDeviceColumnBytesPerPoint, 12U);
    EXPECT_EQ(summary.peakDeviceBytesPerPoint, 12U);
    EXPECT_EQ(summary.hostDeviceTransferBytesPerPoint, 6U);
    EXPECT_EQ(summary.hostDeviceTransfers, 2U);
    EXPECT_EQ(summary.spillBoundaries, 1U);
    EXPECT_EQ(summary.residentRegions, 1U);
    ASSERT_EQ(summary.residencyBoundaries.size(), 2U);
    const auto spill = std::find_if(
        summary.residencyBoundaries.begin(), summary.residencyBoundaries.end(),
        [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Spill; });
    ASSERT_NE(spill, summary.residencyBoundaries.end());
    EXPECT_EQ(spill->dimensions, std::vector<pdg::DimensionId>({pointSource}));
    EXPECT_EQ(spill->releaseDimensions,
              std::vector<pdg::DimensionId>({pointSource}));
    EXPECT_EQ(plan.estimatedDeviceBytes(100), 1200U);
    const std::size_t overflowingCapacity =
        std::numeric_limits<std::size_t>::max() /
            summary.peakDeviceBytesPerPoint +
        1U;
    EXPECT_THROW(
        static_cast<void>(plan.estimatedDeviceBytes(overflowingCapacity)),
        std::overflow_error);
}

TEST(PipelinePlan, DeclaresAndAssemblesDescriptorOwnedFusion)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([{"type":"readers.las","filename":"input.las"},
             {"type":"filters.assign","value":"TmpA = Intensity * 2"},
             {"type":"filters.ferry","dimensions":"TmpA=>Classification"},
             {"type":"writers.las","filename":"output.las"}])",
        dimensions, {.strict = false, .deterministic = true});

    ASSERT_EQ(plan.stages().size(), 4U);
    const pdg::FusionSemantics& reader = plan.stages()[0].descriptor.fusion;
    const pdg::FusionSemantics& assign = plan.stages()[1].descriptor.fusion;
    const pdg::FusionSemantics& ferry = plan.stages()[2].descriptor.fusion;
    const pdg::FusionSemantics& writer = plan.stages()[3].descriptor.fusion;
    EXPECT_EQ(plan.stages()[1].descriptor.placementModel, "point-program");
    EXPECT_TRUE(plan.stages()[2].descriptor.placementModel.empty());
    EXPECT_TRUE(reader.acceptsFusedEpilogue);
    EXPECT_FALSE(reader.acceptsFusedPrologue);
    EXPECT_TRUE(writer.acceptsFusedPrologue);
    EXPECT_TRUE(writer.prologueConsumesPointWrites);
    for (const pdg::FusionSemantics* point : {&assign, &ferry})
    {
        EXPECT_TRUE(point->pure);
        EXPECT_TRUE(point->cardinalityPreserving);
        EXPECT_TRUE(point->fusableAsPrologue);
        EXPECT_TRUE(point->fusableAsEpilogue);
        EXPECT_TRUE(point->deterministicSafe);
        EXPECT_FALSE(point->hasWhere);
        EXPECT_EQ(point->whereMerge, pdg::WhereMergeMode::NotApplicable);
    }
    EXPECT_EQ(assign.dimsRead, plan.stages()[1].descriptor.reads);
    EXPECT_EQ(assign.dimsWritten, plan.stages()[1].descriptor.writes);
    EXPECT_EQ(ferry.dimsRead, plan.stages()[2].descriptor.reads);
    EXPECT_EQ(ferry.dimsWritten, plan.stages()[2].descriptor.writes);

    const pdg::PlanSummary& summary = plan.summary();
    EXPECT_TRUE(summary.deterministic);
    ASSERT_EQ(summary.fusionCandidates.size(), 2U);
    EXPECT_EQ(summary.fusionCandidates[0].anchorStage, 0U);
    EXPECT_EQ(summary.fusionCandidates[0].placement,
              pdg::FusionPlacement::ProducerEpilogue);
    EXPECT_EQ(summary.fusionCandidates[0].pointStages,
              (std::vector<std::size_t>{1U, 2U}));
    EXPECT_EQ(summary.fusionCandidates[1].anchorStage, 3U);
    EXPECT_EQ(summary.fusionCandidates[1].placement,
              pdg::FusionPlacement::ConsumerPrologue);
    EXPECT_EQ(summary.fusionCandidates[1].pointStages,
              (std::vector<std::size_t>{1U, 2U}));
}

TEST(PipelinePlan, DeclaresConditionalSemanticsAndBlocksFusion)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",{"type":"filters.ferry",
             "dimensions":"Intensity=>PointSourceId","where":"Z > 0",
             "where_merge":false},"output.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::FusionSemantics& fusion = plan.stages()[1].descriptor.fusion;
    EXPECT_TRUE(fusion.hasWhere);
    EXPECT_EQ(fusion.whereMerge, pdg::WhereMergeMode::SeparateSkipped);
    EXPECT_FALSE(plan.stages()[1].native);
    EXPECT_TRUE(plan.summary().fusionCandidates.empty());
}

TEST(PipelinePlan, ExpressionDeclaresSplitCardinalityAndWhereSemantics)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.expression","expression":"Intensity <= 30000"},
             "output.las"])",
        dimensions, {.strict = false, .deterministic = true});

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& expression = plan.stages()[1];
    EXPECT_TRUE(expression.native);
    EXPECT_EQ(expression.descriptor.kind, pdg::StageKind::Split);
    EXPECT_EQ(expression.descriptor.placementModel, "point-program");
    EXPECT_EQ(expression.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(expression.descriptor.preservesOrder);
    EXPECT_FALSE(expression.descriptor.mutatesCoordinates);
    const pdg::FusionSemantics& fusion = expression.descriptor.fusion;
    EXPECT_TRUE(fusion.pure);
    EXPECT_FALSE(fusion.cardinalityPreserving);
    EXPECT_TRUE(fusion.deterministicSafe);
    EXPECT_TRUE(fusion.fusableAsPrologue);
    EXPECT_FALSE(fusion.fusableAsEpilogue);
    EXPECT_FALSE(fusion.hasWhere);
    EXPECT_EQ(fusion.whereMerge, pdg::WhereMergeMode::NotApplicable);
    EXPECT_TRUE(
        std::holds_alternative<pdg::PredicateProgram>(expression.payload));
    // A declared cardinality change may fuse only as a consumer prologue of
    // an anchor that declares compaction support; it never fuses as a
    // producer epilogue.
    ASSERT_EQ(plan.summary().fusionCandidates.size(), 1U);
    EXPECT_EQ(plan.summary().fusionCandidates[0].anchorStage, 2U);
    EXPECT_EQ(plan.summary().fusionCandidates[0].placement,
              pdg::FusionPlacement::ConsumerPrologue);
    EXPECT_EQ(plan.summary().fusionCandidates[0].pointStages,
              (std::vector<std::size_t>{1U}));
}

TEST(PipelinePlan, DeclaresCompactingWriterPrologueForTheExpressionChain)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([{"type":"readers.las","filename":"input.las"},
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry","dimensions":"Classification=>UserData"},
             {"type":"filters.expression","expression":"Intensity <= 10000"},
             {"type":"filters.assign","value":[
               "UserData = 3 WHERE Classification == 7"]},
             {"type":"writers.las","filename":"output.las"}])",
        dimensions, {.strict = false, .deterministic = true});

    ASSERT_EQ(plan.stages().size(), 6U);
    EXPECT_TRUE(plan.stages()[5].descriptor.fusion.acceptsCompactingPrologue);
    EXPECT_FALSE(plan.stages()[0].descriptor.fusion.acceptsCompactingPrologue);
    // The complete compacting middle chain fuses as the writer's prologue;
    // the reader epilogue must not host a cardinality change.
    const std::vector<std::size_t> middle{1U, 2U, 3U, 4U};
    bool writerPrologue = false;
    for (const pdg::FusionCandidate& candidate :
         plan.summary().fusionCandidates)
    {
        EXPECT_FALSE(candidate.anchorStage == 0U &&
                     candidate.pointStages == middle);
        if (candidate.anchorStage == 5U &&
            candidate.placement == pdg::FusionPlacement::ConsumerPrologue &&
            candidate.pointStages == middle)
        {
            writerPrologue = true;
            EXPECT_TRUE(candidate.deterministicSafe);
        }
    }
    EXPECT_TRUE(writerPrologue);

    // The where-bearing form is a host fallback and declares no candidate
    // covering the chain.
    pdg::DimensionRegistry whereDimensions;
    const pdg::Plan whereForm = pdg::compilePipeline(
        R"([{"type":"readers.las","filename":"input.las"},
             {"type":"filters.assign","value":[
               "Scratch = Intensity * 2 - 1",
               "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
             {"type":"filters.ferry","dimensions":"Classification=>UserData"},
             {"type":"filters.expression","expression":"Intensity <= 10000",
              "where":"ReturnNumber >= 1"},
             {"type":"filters.assign","value":[
               "UserData = 3 WHERE Classification == 7"]},
             {"type":"writers.las","filename":"output.las"}])",
        whereDimensions);
    for (const pdg::FusionCandidate& candidate :
         whereForm.summary().fusionCandidates)
        EXPECT_NE(candidate.pointStages, middle);
}

TEST(PipelinePlan, ExpressionDeclaresWhereSemanticsOnTheHostFallback)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.expression","expression":"Intensity <= 30000",
              "where":"Z > 0","where_merge":true},
             "output.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& expression = plan.stages()[1];
    EXPECT_FALSE(expression.native);
    EXPECT_EQ(expression.preferredResidency, pdg::MemoryKind::Host);
    EXPECT_TRUE(expression.descriptor.fusion.hasWhere);
    EXPECT_EQ(expression.descriptor.fusion.whereMerge,
              pdg::WhereMergeMode::MergeSkipped);
    EXPECT_EQ(plan.summary().residentRegions, 0U);
}

TEST(PipelinePlan, FusionLegalityHonorsDimensionsAndDeterminism)
{
    const pdg::DimensionId classification(
        pdg::StandardDimension::Classification);
    pdg::FusionSemantics point;
    point.pure = true;
    point.cardinalityPreserving = true;
    point.fusableAsPrologue = true;
    point.fusableAsEpilogue = true;
    point.deterministicSafe = true;
    point.dimsWritten = {classification};

    pdg::FusionSemantics anchor;
    anchor.acceptsFusedPrologue = true;
    anchor.acceptsFusedEpilogue = true;
    anchor.deterministicSafe = true;
    anchor.dimsRead = {classification};
    EXPECT_FALSE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ConsumerPrologue, false));
    anchor.prologueConsumesPointWrites = true;
    EXPECT_TRUE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ConsumerPrologue, true));
    EXPECT_TRUE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ProducerEpilogue, true));

    point.deterministicSafe = false;
    EXPECT_FALSE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ConsumerPrologue, true));
    EXPECT_TRUE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ConsumerPrologue, false));
    point.hasWhere = true;
    point.whereMerge = pdg::WhereMergeMode::Auto;
    EXPECT_FALSE(pdg::pointFusionLegal(
        point, anchor, pdg::FusionPlacement::ConsumerPrologue, false));
}

TEST(PipelinePlan, ResidentLivenessReusesTemporaryColumnStorage)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign","value":"TmpA = Intensity * 2"},
             {"type":"filters.assign","value":"TmpB = TmpA + 1"},
             {"type":"filters.assign","value":"Classification = TmpB"},
             "output.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 5U);
    const auto intensity = pdg::DimensionId(pdg::StandardDimension::Intensity);
    const auto classification =
        pdg::DimensionId(pdg::StandardDimension::Classification);
    const pdg::DimensionId tmpA = dimensions.require("TmpA").id;
    const pdg::DimensionId tmpB = dimensions.require("TmpB").id;
    const pdg::PlannedStage& first = plan.stages()[1];
    const pdg::PlannedStage& second = plan.stages()[2];
    const pdg::PlannedStage& third = plan.stages()[3];

    EXPECT_EQ(first.residentRegion, 0U);
    EXPECT_EQ(second.residentRegion, 0U);
    EXPECT_EQ(third.residentRegion, 0U);
    EXPECT_TRUE(contains(first.deviceMaterialize, intensity));
    EXPECT_TRUE(contains(first.deviceMaterialize, tmpA));
    EXPECT_TRUE(contains(first.deviceRelease, intensity));
    EXPECT_TRUE(contains(second.deviceMaterialize, tmpB));
    EXPECT_TRUE(contains(second.deviceRelease, tmpA));
    EXPECT_TRUE(contains(third.deviceMaterialize, classification));
    EXPECT_TRUE(contains(third.deviceRelease, tmpB));
    EXPECT_FALSE(contains(third.deviceRelease, classification));
    EXPECT_EQ(first.deviceColumnBytesPerPoint, 10U);
    EXPECT_EQ(second.deviceColumnBytesPerPoint, 16U);
    EXPECT_EQ(third.deviceColumnBytesPerPoint, 9U);
    EXPECT_EQ(plan.summary().bytesPerPoint, 19U);
    EXPECT_EQ(plan.summary().peakDeviceColumnBytesPerPoint, 16U);
    EXPECT_EQ(plan.summary().peakDeviceBytesPerPoint, 16U);
    EXPECT_EQ(plan.estimatedDeviceBytes(100U), 1600U);

    const auto spill = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(), [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Spill; });
    ASSERT_NE(spill, plan.summary().residencyBoundaries.end());
    EXPECT_EQ(spill->releaseDimensions,
              std::vector<pdg::DimensionId>({classification}));
    EXPECT_EQ(spill->repackDimensions,
              std::vector<pdg::DimensionId>({classification}));
    EXPECT_EQ(spill->repackBytesPerPoint, 1U);

    pdg::HostMemoryResource memory;
    pdg::PointBatch batch(10U, {{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}},
                          dimensions, memory);
    pdg::preparePlannedDeviceColumns(batch, first);
    EXPECT_EQ(batch.allocatedBytes(), 100U);
    pdg::releasePlannedDeviceColumns(batch, first);
    EXPECT_EQ(batch.allocatedBytes(), 80U);
    pdg::preparePlannedDeviceColumns(batch, second);
    EXPECT_EQ(batch.allocatedBytes(), 160U);
    pdg::releasePlannedDeviceColumns(batch, second);
    EXPECT_EQ(batch.allocatedBytes(), 80U);
    pdg::preparePlannedDeviceColumns(batch, third);
    EXPECT_EQ(batch.allocatedBytes(), 90U);
    pdg::releasePlannedDeviceColumns(batch, third);
    EXPECT_EQ(batch.allocatedBytes(), 10U);
    pdg::releaseSpilledDeviceColumns(batch, *spill);
    EXPECT_EQ(batch.allocatedBytes(), 0U);
    EXPECT_THROW(pdg::releaseSpilledDeviceColumns(
                     batch, plan.summary().residencyBoundaries.front()),
                 std::invalid_argument);
}

TEST(PipelinePlan, ReportsExplicitFallbackSpillAndUploadBoundaries)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign","value":"Scratch = Intensity"},
             {"type":"filters.unimplemented"},
             {"type":"filters.assign","value":"Classification = Intensity"},
             "output.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 5U);
    EXPECT_FALSE(plan.stages()[2].native);
    EXPECT_EQ(plan.stages()[1].residentRegion, 0U);
    EXPECT_EQ(plan.stages()[3].residentRegion, 1U);
    const pdg::PlanSummary& summary = plan.summary();
    EXPECT_EQ(summary.residentRegions, 2U);
    EXPECT_EQ(summary.hostDeviceTransfers, 4U);
    EXPECT_EQ(summary.spillBoundaries, 2U);
    EXPECT_EQ(summary.fallbackBoundaries, 2U);
    ASSERT_EQ(summary.residencyBoundaries.size(), 4U);
    EXPECT_EQ(std::count_if(summary.residencyBoundaries.begin(),
                            summary.residencyBoundaries.end(),
                            [](const auto& boundary)
                            { return boundary.requiresFullPointRecord; }),
              2);
    EXPECT_TRUE(std::all_of(
        summary.residencyBoundaries.begin(), summary.residencyBoundaries.end(),
        [](const auto& boundary)
        { return boundary.fallback == boundary.requiresFullPointRecord; }));

    const auto fallbackSpill = std::find_if(
        summary.residencyBoundaries.begin(), summary.residencyBoundaries.end(),
        [](const auto& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Spill &&
                   boundary.fallback;
        });
    ASSERT_NE(fallbackSpill, summary.residencyBoundaries.end());
    const pdg::DimensionId intensity(pdg::StandardDimension::Intensity);
    const pdg::DimensionId scratch = dimensions.require("Scratch").id;
    EXPECT_TRUE(contains(fallbackSpill->releaseDimensions, intensity));
    EXPECT_TRUE(contains(fallbackSpill->releaseDimensions, scratch));
    EXPECT_EQ(fallbackSpill->repackDimensions,
              std::vector<pdg::DimensionId>({scratch}));
    EXPECT_EQ(fallbackSpill->repackBytesPerPoint, 8U);
}

TEST(PipelinePlan, SupportsStringShorthandAndConservativeFallback)
{
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(
            R"(["input.LAS",
                 {"type":"filters.ferry",
                  "dimensions":"Intensity=>PointSourceId"},
                 "output.las"])",
            dimensions);
        ASSERT_EQ(plan.stages().size(), 3U);
        EXPECT_EQ(plan.stages()[0].descriptor.type, "readers.las");
        EXPECT_EQ(plan.stages()[2].descriptor.type, "writers.las");
        EXPECT_TRUE(plan.summary().allStagesNative);
    }

    // B0188/D0217: a `.laz` *reader* is native. Decode is not a new cost
    // (pinned `laz -> las` translate measured faster than `las -> las`), and
    // the direct source that really memory-maps records stays independently
    // gated on `!compressedReader`. A `.laz` *writer* must still encode, which
    // is a real unmodelled cost, so it stays non-native.
    for (const std::string& pipeline : {
             std::string(R"(["input.laz", "output.las"])"),
             std::string(R"([{"type":"readers.las", "filename":"input.laz"},
             "output.las"])")})
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_TRUE(plan.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(plan.summary().fallbackReasons.empty()) << pipeline;
        EXPECT_TRUE(plan.stages().front().native) << pipeline;
    }
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan =
            pdg::compilePipeline(R"(["input.las", "output.laz"])", dimensions);
        EXPECT_FALSE(plan.stages().back().native);
        EXPECT_FALSE(plan.summary().allStagesNative);
    }

    const std::vector<std::string> pipelines = {
        R"([{"type":"readers.las", "filename":"input.las",
              "count":10}, "output.las"])",
        R"(["input.las",
             {"type":"filters.ferry", "dimensions":"X=>CopyX",
              "extra":"unsupported"},
             "output.las"])",
        R"(["input.las", {"type":"filters.range"}, "output.las"])",
    };
    for (const std::string& pipeline : pipelines)
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
        EXPECT_FALSE(plan.summary().allStagesNative) << pipeline;
        EXPECT_FALSE(plan.summary().fallbackReasons.empty()) << pipeline;

        pdg::DimensionRegistry strictDimensions;
        EXPECT_THROW(
            static_cast<void>(pdg::compilePipeline(pipeline, strictDimensions,
                                                   pdg::PlannerOptions{true})),
            pdg::PlanError)
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesNativeOrderedAssignProgram)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([
            "input.las",
            {"type":"filters.assign", "tag":"calculate",
             "value":["Scratch = Intensity * 2",
                      "Classification = 7 WHERE Scratch >= 10"]},
            {"type":"writers.las", "filename":"output.las",
             "inputs":"calculate"}
        ])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Pointwise);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_FALSE(stage.descriptor.mutatesCoordinates);

    const auto& program = std::get<pdg::AssignProgram>(stage.payload);
    ASSERT_EQ(program.assignments.size(), 2U);
    const pdg::DimensionId scratch = dimensions.require("Scratch").id;
    EXPECT_TRUE(program.assignments[0].destinationCreated);
    EXPECT_EQ(program.assignments[0].destination, scratch);
    EXPECT_EQ(program.assignments[1].destination,
              pdg::DimensionId(pdg::StandardDimension::Classification));
    EXPECT_TRUE(contains(stage.descriptor.reads,
                         pdg::DimensionId(pdg::StandardDimension::Intensity)));
    EXPECT_TRUE(contains(stage.descriptor.reads, scratch));
    EXPECT_TRUE(contains(stage.descriptor.writes, scratch));
    EXPECT_TRUE(plan.summary().allStagesNative);
    EXPECT_TRUE(plan.summary().fallbackReasons.empty());

    pdg::DimensionRegistry coordinateDimensions;
    const pdg::Plan coordinatePlan = pdg::compilePipeline(
        R"(["input.las",
             {"type":"filters.assign", "value":"X = X + 0.01"},
             "output.las"])",
        coordinateDimensions);
    EXPECT_TRUE(coordinatePlan.summary().allStagesNative);
    EXPECT_EQ(coordinatePlan.stages()[1].preferredResidency,
              pdg::MemoryKind::Host);

    pdg::DimensionRegistry mathDimensions;
    const pdg::Plan mathPlan = pdg::compilePipeline(
        R"json(["input.las",
                 {"type":"filters.assign", "value":"Z = sqrt(X)"},
                 "output.las"])json",
        mathDimensions);
    EXPECT_TRUE(mathPlan.summary().allStagesNative);
    EXPECT_EQ(mathPlan.stages()[1].preferredResidency, pdg::MemoryKind::Host);
}

TEST(PipelinePlan, CompilesStableExpressionPredicate)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.expression",
             "expression":"Intensity >= 10 && X < 20"}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& filter = plan.stages()[1];
    EXPECT_TRUE(filter.native);
    EXPECT_EQ(filter.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(filter.descriptor.preservesOrder);
    const auto& predicate = std::get<pdg::PredicateProgram>(filter.payload);
    EXPECT_EQ(predicate.reads.size(), 2U);
    EXPECT_TRUE(predicate.expression.boolean);
}

TEST(PipelinePlan, CompilesRangeIntoTheStablePredicatePrimitive)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.range",
             "limits":["X[0:10], X![20:30]", "Classification[1:4]"]},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& filter = plan.stages()[1];
    EXPECT_TRUE(filter.native);
    EXPECT_EQ(filter.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(filter.descriptor.preservesOrder);
    const auto& predicate = std::get<pdg::PredicateProgram>(filter.payload);
    EXPECT_EQ(predicate.reads.size(), 2U);
    EXPECT_TRUE(predicate.expression.boolean);
}

TEST(PipelinePlan, CompilesSingleBoundsCropIntoTheStablePredicatePrimitive)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"json(["in.las", {"type":"filters.crop",
             "bounds":"([0,10],[20,30],[-5,5])", "outside":true},
             "out.las"])json",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& filter = plan.stages()[1];
    EXPECT_TRUE(filter.native);
    EXPECT_EQ(filter.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(filter.descriptor.preservesOrder);
    const auto& predicate = std::get<pdg::PredicateProgram>(filter.payload);
    EXPECT_EQ(predicate.reads.size(), 3U);
    EXPECT_TRUE(predicate.expression.boolean);

    const std::vector<std::string> fallbackPipelines = {
        R"json(["in.las", {"type":"filters.crop",
             "bounds":["([0,1],[0,1])", "([2,3],[2,3])"]},
             "out.las"])json",
        R"json(["in.las", {"type":"filters.crop",
             "bounds":"([0,1],[0,1])", "a_srs":"EPSG:4326"},
             "out.las"])json",
        R"json(["in.las", {"type":"filters.crop",
             "bounds":"([0,1],[0,1])", "outside":"true"},
             "out.las"])json",
    };
    for (const std::string& pipeline : fallbackPipelines)
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesExactOrdinalSelectionPrograms)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.decimation", "step":2.6,
              "offset":10, "limit":90},
             {"type":"filters.head", "count":7, "invert":true},
             {"type":"filters.tail", "count":3},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 5U);
    EXPECT_TRUE(plan.summary().allStagesNative);
    const auto& decimation =
        std::get<pdg::OrdinalProgram>(plan.stages()[1].payload);
    EXPECT_EQ(decimation.kind, pdg::OrdinalKind::Decimation);
    EXPECT_DOUBLE_EQ(decimation.step, 2.6);
    EXPECT_EQ(decimation.offset, 10U);
    EXPECT_EQ(decimation.limit, 90U);
    const auto& head = std::get<pdg::OrdinalProgram>(plan.stages()[2].payload);
    EXPECT_EQ(head.kind, pdg::OrdinalKind::Head);
    EXPECT_EQ(head.count, 7U);
    EXPECT_TRUE(head.invert);
    EXPECT_EQ(std::get<pdg::OrdinalProgram>(plan.stages()[3].payload).kind,
              pdg::OrdinalKind::Tail);

    for (
        const std::string& pipeline : {
            R"(["in.las", {"type":"filters.head", "count":-1}, "out.las"])",
            R"(["in.las", {"type":"filters.tail", "invert":"yes"}, "out.las"])",
            R"(["in.las", {"type":"filters.decimation", "step":"2"}, "out.las"])",
            R"(["in.las", {"type":"filters.decimation", "step":0.5}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
    }
}

TEST(PipelinePlan, CompilesExactLocateReduction)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.locate", "dimension":"Z",
             "minmax":"MIN"}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& locate = plan.stages()[1];
    EXPECT_TRUE(locate.native);
    EXPECT_EQ(locate.descriptor.kind, pdg::StageKind::Global);
    EXPECT_EQ(locate.preferredResidency, pdg::MemoryKind::Device);
    ASSERT_EQ(locate.descriptor.reads.size(), 1U);
    EXPECT_EQ(locate.descriptor.reads.front(),
              pdg::DimensionId(pdg::StandardDimension::Z));
    const auto& program = std::get<pdg::LocateProgram>(locate.payload);
    EXPECT_EQ(program.kind, pdg::LocateKind::Minimum);

    const pdg::Plan invalidKind = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.locate", "dimension":"Z",
             "minmax":"sideways"}, "out.las"])",
        dimensions);
    EXPECT_EQ(
        std::get<pdg::LocateProgram>(invalidKind.stages()[1].payload).kind,
        pdg::LocateKind::None);

    for (
        const std::string& pipeline : {
            R"(["in.las", {"type":"filters.locate"}, "out.las"])",
            R"(["in.las", {"type":"filters.locate", "dimension":7}, "out.las"])",
            R"(["in.las", {"type":"filters.locate", "dimension":"Z", "minmax":1}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload));
    }
}

TEST(PipelinePlan, CompilesExactTransformationAndDelegatesOtherModes)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.transformation",
             "matrix":"2 3 5 7 11 13 17 19 23 29 31 37 0.5 0.25 0 1"},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& transformation = plan.stages()[1];
    EXPECT_TRUE(transformation.native);
    EXPECT_EQ(transformation.descriptor.kind, pdg::StageKind::Pointwise);
    EXPECT_TRUE(transformation.descriptor.mutatesCoordinates);
    EXPECT_TRUE(transformation.descriptor.preservesOrder);
    EXPECT_EQ(transformation.preferredResidency, pdg::MemoryKind::Host);
    EXPECT_EQ(transformation.descriptor.reads.size(), 3U);
    EXPECT_EQ(transformation.descriptor.writes,
              transformation.descriptor.reads);
    const auto& program =
        std::get<pdg::TransformationProgram>(transformation.payload);
    EXPECT_DOUBLE_EQ(program.matrix[0], 2.0);
    EXPECT_DOUBLE_EQ(program.matrix[15], 1.0);

    pdg::DimensionRegistry affineDimensions;
    const pdg::Plan affine = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.transformation",
             "matrix":"1 0 0 4 0 1 0 5 0 0 1 6 0 0 0 1"},
             "out.las"])",
        affineDimensions);
    EXPECT_EQ(affine.stages()[1].preferredResidency, pdg::MemoryKind::Device);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.transformation",
                  "matrix":"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                  "invert":true}, "out.las"])",
             R"(["in.las", {"type":"filters.transformation",
                  "matrix":"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                  "override_srs":"EPSG:4326"}, "out.las"])",
             R"(["in.las", {"type":"filters.transformation"},
                  "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }

    pdg::DimensionRegistry malformedDimensions;
    EXPECT_THROW(static_cast<void>(pdg::compilePipeline(
                     R"(["in.las", {"type":"filters.transformation",
                          "matrix":"1 0 0"}, "out.las"])",
                     malformedDimensions)),
                 pdg::PlanError);
}

TEST(PipelinePlan, CompilesExactRobustStatisticsPrograms)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.iqr","dimension":"Z","k":2.25},
             {"type":"filters.mad","dimension":"Intensity","k":3,
              "mad_multiplier":1.25},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 4U);
    EXPECT_TRUE(plan.summary().allStagesNative);
    for (std::size_t index : {1U, 2U})
    {
        EXPECT_TRUE(plan.stages()[index].native);
        EXPECT_EQ(plan.stages()[index].descriptor.kind, pdg::StageKind::Global);
        EXPECT_EQ(plan.stages()[index].preferredResidency,
                  pdg::MemoryKind::Device);
        EXPECT_TRUE(plan.stages()[index].descriptor.preservesOrder);
    }
    const auto& iqr = std::get<pdg::RobustProgram>(plan.stages()[1].payload);
    EXPECT_EQ(iqr.kind, pdg::RobustKind::Iqr);
    EXPECT_DOUBLE_EQ(iqr.multiplier, 2.25);
    const auto& mad = std::get<pdg::RobustProgram>(plan.stages()[2].payload);
    EXPECT_EQ(mad.kind, pdg::RobustKind::Mad);
    EXPECT_DOUBLE_EQ(mad.multiplier, 3.0);
    EXPECT_DOUBLE_EQ(mad.madMultiplier, 1.25);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.iqr","dimension":"Z",
                  "k":"1.5"}, "out.las"])",
             R"(["in.las", {"type":"filters.mad","dimension":"Z",
                  "where":"Z > 0"}, "out.las"])",
             R"(["in.las", {"type":"filters.mad","dimension":"Z",
                  "mad_multiplier":"1.4862"}, "out.las"])",
             R"(["in.las", {"type":"filters.iqr"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesExactOrderingAndDelegatesOptionRichForms)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.sort",
             "dimensions":["Classification","Intensity"],
             "order":"desc","algorithm":"stable"}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& ordering = plan.stages()[1];
    EXPECT_TRUE(ordering.native);
    EXPECT_EQ(ordering.descriptor.kind, pdg::StageKind::Global);
    EXPECT_FALSE(ordering.descriptor.preservesOrder);
    EXPECT_EQ(ordering.preferredResidency, pdg::MemoryKind::Device);
    ASSERT_EQ(ordering.descriptor.reads.size(), 2U);
    const auto& program = std::get<pdg::OrderingProgram>(ordering.payload);
    EXPECT_EQ(program.dimensions.size(), 2U);
    EXPECT_EQ(program.direction, pdg::OrderingDirection::Descending);
    EXPECT_EQ(program.algorithm, pdg::OrderingAlgorithm::Stable);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.sort","dimension":"Z",
                  "where":"Classification == 2"}, "out.las"])",
             R"(["in.las", {"type":"filters.sort","dimension":7},
                  "out.las"])",
             R"(["in.las", {"type":"filters.sort"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.sort","dimension":"Z",
                  "order":"sideways"}, "out.las"])",
             R"(["in.las", {"type":"filters.sort","dimension":"Z",
                  "algorithm":"radix"}, "out.las"])",
             R"(["in.las", {"type":"filters.sort",
                  "dimension":"NotHere"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry invalidDimensions;
        EXPECT_THROW(static_cast<void>(
                         pdg::compilePipeline(pipeline, invalidDimensions)),
                     pdg::PlanError)
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesExactAdjacentDuplicateLabeling)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.label_duplicates",
             "dimensions":"Classification, ReturnNumber,Classification"},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Global);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stage.descriptor.placementModel.empty());
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_FALSE(stage.descriptor.fusion.fusableAsPrologue);
    EXPECT_FALSE(stage.descriptor.fusion.fusableAsEpilogue);

    const pdg::DimensionId classification(
        pdg::StandardDimension::Classification);
    const pdg::DimensionId returnNumber(pdg::StandardDimension::ReturnNumber);
    const pdg::DimensionId duplicate(pdg::StandardDimension::Duplicate);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{classification, returnNumber,
                                             duplicate}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{duplicate}));
    const auto& program = std::get<pdg::LabelDuplicatesProgram>(stage.payload);
    EXPECT_EQ(program.dimensions,
              (std::vector<pdg::DimensionId>{classification, returnNumber,
                                             classification}));

    pdg::DimensionRegistry emptyDimensions;
    const pdg::Plan empty = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.label_duplicates",
             "dimensions":[]}, "out.las"])",
        emptyDimensions);
    ASSERT_TRUE(empty.stages()[1].native);
    const auto& emptyProgram =
        std::get<pdg::LabelDuplicatesProgram>(empty.stages()[1].payload);
    EXPECT_TRUE(emptyProgram.dimensions.empty());
    EXPECT_EQ(empty.stages()[1].descriptor.reads,
              (std::vector<pdg::DimensionId>{duplicate}));

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":7}, "out.las"])",
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":"Missing"}, "out.las"])",
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":"Duplicate"}, "out.las"])",
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":"X","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesBoundedExactSmrfGridContract)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.smrf","cell":2.0,
             "slope":0.2,"scalar":1.1,"threshold":0.4,"window":8.0,
             "cut":3.0,"returns":[],"ground_class":9,"other_class":9,
             "only_ground":true}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(std::holds_alternative<pdg::SmrfProgram>(stage.payload));
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Grid);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stage.descriptor.placementModel.empty());
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    const pdg::SmrfProgram& program = std::get<pdg::SmrfProgram>(stage.payload);
    EXPECT_DOUBLE_EQ(program.cell, 2.0);
    EXPECT_DOUBLE_EQ(program.slope, 0.2);
    EXPECT_DOUBLE_EQ(program.scalar, 1.1);
    EXPECT_DOUBLE_EQ(program.threshold, 0.4);
    EXPECT_DOUBLE_EQ(program.window, 8.0);
    EXPECT_DOUBLE_EQ(program.cut, 3.0);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_EQ(program.otherClass, 9U);
    EXPECT_TRUE(program.onlyGround);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.smrf","returns":""}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","cell":0}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","window":65}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","ground_class":4,"other_class":4}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesBoundedExactPmfGridContract)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.pmf","cell_size":2.0,
             "exponential":false,"initial_distance":0.2,
             "max_distance":1.1,"max_window_size":9.0,"slope":0.4,
             "returns":[],"ground_class":9,"other_class":9,
             "only_ground":true}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(std::holds_alternative<pdg::PmfProgram>(stage.payload));
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Grid);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stage.descriptor.placementModel.empty());
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_EQ(stage.descriptor.grid.framePolicy,
              pdg::GridFramePolicy::PmfInitialLookupV1);
    EXPECT_DOUBLE_EQ(stage.descriptor.grid.cellSize, 2.0);
    EXPECT_EQ(stage.descriptor.grid.deviceBytesPerCell,
              pdg::PmfTiledDeviceBytesPerCell);
    EXPECT_EQ(stage.descriptor.grid.deviceBackingCount, 2U);
    EXPECT_EQ(stage.descriptor.grid.deviceProofBytesPerCell,
              pdg::PmfTiledDeviceProofBytesPerCell);
    EXPECT_EQ(stage.descriptor.grid.deviceFixedBytes,
              pdg::PmfTiledDeviceFixedScratchBytes);
    EXPECT_EQ(stage.descriptor.grid.hostBytesPerPoint,
              pdg::PmfTiledHostStagingBytesPerPoint);
    EXPECT_EQ(stage.descriptor.grid.hostBytesPerCell,
              pdg::PmfTiledHostBytesPerCell);
    EXPECT_EQ(stage.descriptor.grid.hostTileBytesPerExpandedCell,
              sizeof(double));
    EXPECT_EQ(stage.descriptor.grid.maximumHaloCells, 1U);
    EXPECT_TRUE(stage.descriptor.grid.phaseSynchronized);
    EXPECT_EQ(stage.deviceGridBuildBytesPerCell,
              pdg::PmfTiledDeviceBytesPerCell);
    EXPECT_EQ(stage.deviceGridProofBytesPerCell,
              pdg::PmfTiledDeviceProofBytesPerCell);
    EXPECT_EQ(stage.deviceGridFixedBytes, pdg::PmfTiledDeviceFixedScratchBytes);
    EXPECT_EQ(plan.summary().gridBuilds, 1U);
    EXPECT_EQ(plan.summary().peakDeviceGridBytesPerCell,
              pdg::PmfTiledDeviceBytesPerCell);
    EXPECT_EQ(plan.summary().peakDeviceGridProofBytesPerCell,
              pdg::PmfTiledDeviceProofBytesPerCell);
    EXPECT_EQ(plan.summary().peakDeviceGridFixedBytes,
              pdg::PmfTiledDeviceFixedScratchBytes);
    EXPECT_EQ(plan.estimatedDeviceBytes(0U),
              pdg::PmfTiledDeviceFixedScratchBytes);
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    const pdg::PmfProgram& program = std::get<pdg::PmfProgram>(stage.payload);
    EXPECT_DOUBLE_EQ(program.cellSize, 2.0);
    EXPECT_FALSE(program.exponential);
    EXPECT_DOUBLE_EQ(program.initialDistance, 0.2);
    EXPECT_DOUBLE_EQ(program.maxDistance, 1.1);
    EXPECT_DOUBLE_EQ(program.maxWindowSize, 9.0);
    EXPECT_DOUBLE_EQ(program.slope, 0.4);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_EQ(program.otherClass, 9U);
    EXPECT_TRUE(program.onlyGround);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.pmf","returns":""}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","cell_size":0}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","max_window_size":1000}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","ground_class":4,"other_class":4}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesBoundedExactCsfGridContract)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.csf","smooth":false,
             "step":1.0,"threshold":0.4,"hdiff":0.2,
             "resolution":2.0,"rigidness":4,"iterations":3,
             "returns":[],"ground_class":9,"other_class":9,
             "only_ground":true}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(std::holds_alternative<pdg::CsfProgram>(stage.payload));
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Grid);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stage.descriptor.placementModel.empty());
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    const pdg::CsfProgram& program = std::get<pdg::CsfProgram>(stage.payload);
    EXPECT_FALSE(program.smooth);
    EXPECT_DOUBLE_EQ(program.timeStep, 1.0);
    EXPECT_DOUBLE_EQ(program.classThreshold, 0.4);
    EXPECT_DOUBLE_EQ(program.heightThreshold, 0.2);
    EXPECT_DOUBLE_EQ(program.resolution, 2.0);
    EXPECT_EQ(program.rigidness, 4);
    EXPECT_EQ(program.iterations, 3);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_EQ(program.otherClass, 9U);
    EXPECT_TRUE(program.onlyGround);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.csf"}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"resolution":0}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"iterations":65}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"ground_class":4,"other_class":4}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"debug":true}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesBoundedExactElmGridContract)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.elm","cell":1.25,
             "class":18,"threshold":-1.0}, "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(std::holds_alternative<pdg::ElmProgram>(stage.payload));
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Grid);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(stage.descriptor.placementModel.empty());
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    const pdg::ElmProgram& program = std::get<pdg::ElmProgram>(stage.payload);
    EXPECT_DOUBLE_EQ(program.cell, 1.25);
    EXPECT_EQ(program.classification, 18U);
    EXPECT_DOUBLE_EQ(program.threshold, -1.0);

    for (std::string_view unsupported : {
             R"(["in.las", {"type":"filters.elm","cell":0}, "out.las"])",
             R"(["in.las", {"type":"filters.elm","class":256}, "out.las"])",
             R"(["in.las", {"type":"filters.elm","where":"Z > 0"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << unsupported;
    }
}

TEST(PipelinePlan, DeclaresSkewnessBalancingAsADeviceGlobalOrderingBoundary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.skewnessbalancing",
             "ground_class":3,"other_class":9,"only_ground":false},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    EXPECT_TRUE(stage.native);
    ASSERT_TRUE(std::holds_alternative<pdg::SkewnessProgram>(stage.payload));
    const pdg::SkewnessProgram& program =
        std::get<pdg::SkewnessProgram>(stage.payload);
    EXPECT_EQ(program.groundClass, 3U);
    EXPECT_EQ(program.otherClass, 9U);
    EXPECT_FALSE(program.onlyGround);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Global);
    EXPECT_FALSE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::None);
    EXPECT_EQ(stage.descriptor.deviceToHostBytesPerInputPoint,
              sizeof(std::uint64_t));
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.cardinalityPreserving);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_TRUE(contains(stage.descriptor.reads,
                         pdg::DimensionId(pdg::StandardDimension::Z)));
    EXPECT_TRUE(
        contains(stage.descriptor.reads,
                 pdg::DimensionId(pdg::StandardDimension::Classification)));
    EXPECT_FALSE(contains(stage.descriptor.writes,
                          pdg::DimensionId(pdg::StandardDimension::Z)));
    EXPECT_TRUE(contains(
        stage.descriptor.writes,
        pdg::DimensionId(pdg::StandardDimension::Classification)));

    pdg::DimensionRegistry fallbackDimensions;
    const pdg::Plan fallback = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.skewnessbalancing",
             "where":"Classification != 7"}, "out.las"])",
        fallbackDimensions);
    EXPECT_FALSE(fallback.stages()[1U].native);
    EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
        fallback.stages()[1U].payload));
}

TEST(PipelinePlan, CompilesMortonOrderingAndDelegatesOptionRichForms)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.mortonorder","reverse":true},
             "out.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& ordering = plan.stages()[1];
    EXPECT_TRUE(ordering.native);
    EXPECT_EQ(ordering.descriptor.kind, pdg::StageKind::Global);
    EXPECT_FALSE(ordering.descriptor.preservesOrder);
    EXPECT_EQ(ordering.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(ordering.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y)}));
    EXPECT_TRUE(std::get<pdg::MortonProgram>(ordering.payload).reverse);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.mortonorder",
                  "reverse":"true"}, "out.las"])",
             R"(["in.las", {"type":"filters.mortonorder",
                  "where":"X > 0"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesCategoricalGroupingAndDelegatesOptionRichForms)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.groupby",
             "dimension":"Classification"}, "out#.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& grouping = plan.stages()[1];
    EXPECT_TRUE(grouping.native);
    EXPECT_EQ(grouping.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(grouping.descriptor.preservesOrder);
    EXPECT_EQ(grouping.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::GroupByProgram>(grouping.payload);
    EXPECT_EQ(program.dimension,
              pdg::DimensionId(pdg::StandardDimension::Classification));
    EXPECT_EQ(grouping.descriptor.reads,
              (std::vector<pdg::DimensionId>{program.dimension}));

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.groupby"}, "out#.las"])",
             R"(["in.las", {"type":"filters.groupby","dimension":7},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.groupby",
                  "dimension":"Classification","where":"X > 0"},
                  "out#.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }

    pdg::DimensionRegistry missingDimensions;
    EXPECT_THROW(static_cast<void>(pdg::compilePipeline(
                     R"(["in.las", {"type":"filters.groupby",
                          "dimension":"NotHere"}, "out#.las"])",
                     missingDimensions)),
                 pdg::PlanError);
}

TEST(PipelinePlan, CompilesReturnPartitionAndViewMergeContracts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.returns",
             "groups":" last, first, only "},
             {"type":"filters.merge"}, "out.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 4U);
    const pdg::PlannedStage& returns = plan.stages()[1];
    EXPECT_TRUE(returns.native);
    EXPECT_EQ(returns.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(returns.descriptor.preservesOrder);
    EXPECT_EQ(returns.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(std::get<pdg::ReturnsProgram>(returns.payload).groups,
              pdg::ReturnLast | pdg::ReturnFirst | pdg::ReturnOnly);
    EXPECT_EQ(returns.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::ReturnNumber),
                  pdg::DimensionId(pdg::StandardDimension::NumberOfReturns)}));

    const pdg::PlannedStage& merge = plan.stages()[2];
    EXPECT_TRUE(merge.native);
    EXPECT_EQ(merge.descriptor.kind, pdg::StageKind::Global);
    EXPECT_EQ(merge.preferredResidency, pdg::MemoryKind::Host);
    EXPECT_TRUE(std::holds_alternative<pdg::MergeProgram>(merge.payload));

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.returns","groups":7},
                  "out.las"])",
             R"(["in.las", {"type":"filters.returns","where":"X > 0"},
                  "out.las"])",
             R"(["in.las", {"type":"filters.merge",
                  "spatialreference":"EPSG:4326"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }

    pdg::DimensionRegistry invalidDimensions;
    EXPECT_THROW(static_cast<void>(pdg::compilePipeline(
                     R"(["in.las", {"type":"filters.returns",
                          "groups":"last,sideways"}, "out.las"])",
                     invalidDimensions)),
                 pdg::PlanError);
}

TEST(PipelinePlan, CompilesDividerAndSplitterExactEnvelopes)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.divider","mode":"ROUND_ROBIN",
             "count":7},
             {"type":"filters.splitter","length":250,
              "origin_x":636000,"origin_y":848000,"buffer":-1},
             "out#.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 4U);

    const pdg::PlannedStage& divider = plan.stages()[1];
    EXPECT_TRUE(divider.native);
    EXPECT_EQ(divider.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(divider.descriptor.preservesOrder);
    EXPECT_EQ(divider.preferredResidency, pdg::MemoryKind::Device);
    const auto& dividerProgram = std::get<pdg::DividerProgram>(divider.payload);
    EXPECT_EQ(dividerProgram.mode, pdg::DividerMode::RoundRobin);
    EXPECT_EQ(dividerProgram.count, 7U);

    const pdg::PlannedStage& splitter = plan.stages()[2];
    EXPECT_TRUE(splitter.native);
    EXPECT_EQ(splitter.descriptor.kind, pdg::StageKind::Split);
    EXPECT_TRUE(splitter.descriptor.preservesOrder);
    EXPECT_EQ(splitter.preferredResidency, pdg::MemoryKind::Device);
    const auto& splitterProgram =
        std::get<pdg::SplitterProgram>(splitter.payload);
    EXPECT_DOUBLE_EQ(splitterProgram.length, 250.0);
    EXPECT_DOUBLE_EQ(splitterProgram.originX, 636000.0);
    EXPECT_DOUBLE_EQ(splitterProgram.originY, 848000.0);
    EXPECT_DOUBLE_EQ(splitterProgram.buffer, -1.0);
    EXPECT_EQ(splitter.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y)}));

    pdg::DimensionRegistry bufferedDimensions;
    const pdg::Plan buffered = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.splitter","length":1000,
             "buffer":20}, "out#.las"])",
        bufferedDimensions);
    EXPECT_TRUE(buffered.stages()[1].native);
    EXPECT_EQ(buffered.stages()[1].preferredResidency, pdg::MemoryKind::Host);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.divider","capacity":25},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.divider","count":1},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.divider","count":3,
                  "mode":"sideways"}, "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":"10"},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":10,
                  "buffer":5}, "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":10,
                  "where":"X > 0"}, "out#.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesColorinterpExactEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.colorinterp",
             "dimension":"Intensity","minimum":10,"maximum":500,
             "clamp":true,"ramp":"heat_map","invert":true,
             "mad":true,"mad_multiplier":2.5,"k":1.25}, "out.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Pointwise);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::ColorinterpProgram>(stage.payload);
    EXPECT_EQ(program.dimension,
              pdg::DimensionId(pdg::StandardDimension::Intensity));
    EXPECT_DOUBLE_EQ(program.minimum, 10.0);
    EXPECT_DOUBLE_EQ(program.maximum, 500.0);
    EXPECT_TRUE(program.clamp);
    EXPECT_EQ(program.ramp, "heat_map");
    EXPECT_TRUE(program.invert);
    EXPECT_TRUE(program.mad);
    EXPECT_DOUBLE_EQ(program.madMultiplier, 2.5);
    EXPECT_DOUBLE_EQ(program.k, 1.25);
    EXPECT_TRUE(contains(stage.descriptor.reads, program.dimension));
    for (const pdg::StandardDimension dimension :
         {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
          pdg::StandardDimension::Blue})
    {
        const pdg::DimensionId id(dimension);
        EXPECT_TRUE(contains(stage.descriptor.reads, id));
        EXPECT_TRUE(contains(stage.descriptor.writes, id));
    }

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.colorinterp",
                  "minimum":"0","maximum":1}, "out.las"])",
             R"(["in.las", {"type":"filters.colorinterp",
                  "minimum":1,"maximum":1}, "out.las"])",
             R"(["in.las", {"type":"filters.colorinterp",
                  "clamp":"true"}, "out.las"])",
             R"(["in.las", {"type":"filters.colorinterp",
                  "where":"Z > 0"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesStatsMetadataAndExactDeviceEnvelope)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan devicePlan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.stats",
             "dimensions":["Z","Classification"],
             "commonsrs":"EPSG:3857"}, "out.las"])",
        dimensions);
    ASSERT_EQ(devicePlan.stages().size(), 3U);
    const pdg::PlannedStage& device = devicePlan.stages()[1];
    EXPECT_TRUE(device.native);
    EXPECT_EQ(device.descriptor.kind, pdg::StageKind::Global);
    EXPECT_TRUE(device.descriptor.preservesOrder);
    EXPECT_EQ(device.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::StatsProgram>(device.payload);
    EXPECT_EQ(program.dimensions.size(), 2U);
    EXPECT_EQ(program.modes.size(), program.dimensions.size());
    EXPECT_FALSE(program.advanced);
    EXPECT_EQ(program.commonSrs, "EPSG:3857");
    EXPECT_TRUE(contains(program.dimensions,
                         pdg::DimensionId(pdg::StandardDimension::Z)));
    EXPECT_TRUE(
        contains(program.dimensions,
                 pdg::DimensionId(pdg::StandardDimension::Classification)));

    pdg::DimensionRegistry hostDimensions;
    const pdg::Plan hostPlan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.stats",
             "dimensions":"Z,Classification",
             "count":"Classification","advanced":true}, "out.las"])",
        hostDimensions);
    const pdg::PlannedStage& host = hostPlan.stages()[1];
    EXPECT_TRUE(host.native);
    EXPECT_EQ(host.preferredResidency, pdg::MemoryKind::Host);
    const auto& hostProgram = std::get<pdg::StatsProgram>(host.payload);
    EXPECT_TRUE(hostProgram.advanced);
    ASSERT_EQ(hostProgram.modes.size(), 2U);
    EXPECT_TRUE(std::find(hostProgram.modes.begin(), hostProgram.modes.end(),
                          pdg::SummaryMode::Count) != hostProgram.modes.end());

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.stats",
                  "advanced":"true"}, "out.las"])",
             R"(["in.las", {"type":"filters.stats",
                  "where":"Z > 0"}, "out.las"])",
         })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload))
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesInfoAndExpressionStatsMetadataStages)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
          {"type":"filters.info","point":"0,2-3"},
          {"type":"filters.expressionstats","dimension":"Classification",
           "expressions":["Classification == 2","Intensity < 100"]},
          "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 4U);
    const auto& infoStage = plan.stages()[1];
    ASSERT_TRUE(std::holds_alternative<pdg::InfoProgram>(infoStage.payload));
    EXPECT_TRUE(infoStage.native);
    EXPECT_EQ(infoStage.preferredResidency, pdg::MemoryKind::Host);
    EXPECT_EQ(std::get<pdg::InfoProgram>(infoStage.payload).pointSpec, "0,2-3");
    EXPECT_EQ(infoStage.descriptor.reads.size(), 3U);

    const auto& expressionStage = plan.stages()[2];
    ASSERT_TRUE(std::holds_alternative<pdg::ExpressionStatsProgram>(
        expressionStage.payload));
    EXPECT_TRUE(expressionStage.native);
    EXPECT_EQ(expressionStage.preferredResidency, pdg::MemoryKind::Device);
    const auto& program =
        std::get<pdg::ExpressionStatsProgram>(expressionStage.payload);
    EXPECT_EQ(program.dimension,
              pdg::DimensionId(pdg::StandardDimension::Classification));
    EXPECT_EQ(program.expressions.size(), 2U);

    pdg::DimensionRegistry basicInfoDimensions;
    const pdg::Plan basicInfo = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.info"}, "out.las"])",
        basicInfoDimensions);
    EXPECT_EQ(basicInfo.stages()[1].preferredResidency,
              pdg::MemoryKind::Device);

    const pdg::Plan unsupported = pdg::compilePipeline(
        R"(["in.las",
          {"type":"filters.info","p":"0"},
          {"type":"filters.expressionstats","dimension":"Classification",
           "expressions":true}, "out.las"])",
        dimensions);
    EXPECT_FALSE(unsupported.stages()[1].native);
    EXPECT_FALSE(unsupported.stages()[2].native);
}

TEST(PipelinePlan, CompilesOutlierRequestsAndSharedIndexLifecycle)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
          {"type":"filters.outlier","method":"radius","radius":2.5,
           "min_k":4,"class":18},
          {"type":"filters.outlier","method":"radius","radius":1.25,
           "min_k":2},
          {"type":"filters.transformation",
           "matrix":"1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1"},
          {"type":"filters.outlier","method":"radius","radius":3},
          "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 6U);
    const pdg::PlannedStage& first = plan.stages()[1];
    EXPECT_TRUE(first.native);
    EXPECT_EQ(first.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(first.descriptor.index.kind, pdg::IndexKind::Radius);
    EXPECT_DOUBLE_EQ(first.descriptor.index.radius, 2.5);
    EXPECT_DOUBLE_EQ(first.descriptor.maximumRadius, 2.5);
    EXPECT_EQ(first.deviceQueryBytesPerPoint, sizeof(std::uint32_t));
    EXPECT_EQ(first.descriptor.deviceToHostBytesPerInputPoint,
              sizeof(std::uint32_t));
    EXPECT_EQ(first.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(first.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(first.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    const auto& program = std::get<pdg::OutlierProgram>(first.payload);
    EXPECT_EQ(program.method, pdg::OutlierMethod::Radius);
    EXPECT_EQ(program.minimumNeighbors, 4);
    EXPECT_DOUBLE_EQ(program.radius, 2.5);
    EXPECT_EQ(program.classification, 18U);

    EXPECT_EQ(plan.stages()[2].descriptor.index.kind, pdg::IndexKind::Radius);
    EXPECT_TRUE(plan.stages()[3].descriptor.mutatesCoordinates);
    EXPECT_EQ(plan.stages()[4].descriptor.index.kind, pdg::IndexKind::Radius);
    EXPECT_EQ(plan.summary().indexBuilds, 2U);
    EXPECT_DOUBLE_EQ(plan.summary().maximumRadius, 3.0);
    EXPECT_EQ(plan.summary().indexBytesPerPoint, 28U);
    EXPECT_EQ(plan.summary().peakDeviceQueryBytesPerPoint,
              sizeof(std::uint32_t));
    EXPECT_EQ(plan.stages()[1].deviceIndexBuildBytesPerPoint, 28U);
    EXPECT_EQ(plan.stages()[1].deviceIndexBytesPerPoint, 28U);
    EXPECT_EQ(plan.stages()[1].deviceIndexReleaseBytesPerPoint, 0U);
    EXPECT_EQ(plan.stages()[2].deviceIndexBytesPerPoint, 28U);
    EXPECT_EQ(plan.stages()[2].deviceIndexReleaseBytesPerPoint, 28U);
    EXPECT_EQ(plan.stages()[3].deviceIndexBytesPerPoint, 0U);
    EXPECT_EQ(plan.stages()[4].deviceIndexBuildBytesPerPoint, 28U);
    EXPECT_EQ(plan.stages()[4].deviceIndexReleaseBytesPerPoint, 28U);
    EXPECT_EQ(plan.estimatedDeviceBytes(100), 5700U);
    pdg::HostMemoryResource residentMemory;
    pdg::PointBatch residentBatch(1U, {{0.01, 0.01, 0.01}, {0.0, 0.0, 0.0}},
                                  dimensions, residentMemory);
    pdg::preparePlannedDeviceColumns(residentBatch, plan.stages()[1]);
    EXPECT_EQ(
        residentBatch.columnInfo(pdg::DimensionId(pdg::StandardDimension::X))
            .physicalType,
        pdg::DimensionType::Double);

    pdg::DimensionRegistry branchDimensions;
    const pdg::Plan branches = pdg::compilePipeline(
        R"([
          {"type":"readers.las","filename":"in.las","tag":"source"},
          {"type":"filters.outlier","method":"radius","radius":2,
           "inputs":"source","tag":"left"},
          {"type":"filters.outlier","method":"radius","radius":3,
           "inputs":"source","tag":"right"},
          {"type":"writers.las","filename":"out.las","inputs":"right"}
        ])",
        branchDimensions);
    EXPECT_EQ(branches.summary().indexBuilds, 1U);
    const pdg::DimensionId branchX(pdg::StandardDimension::X);
    EXPECT_EQ(branches.stages()[1].residentRegion,
              branches.stages()[2].residentRegion);
    EXPECT_FALSE(contains(branches.stages()[1].deviceRelease, branchX));
    EXPECT_TRUE(contains(branches.stages()[2].deviceRelease, branchX));
    EXPECT_EQ(branches.summary().hostDeviceTransfers, 2U);
    const auto branchUpload = std::find_if(
        branches.summary().residencyBoundaries.begin(),
        branches.summary().residencyBoundaries.end(), [](const auto& boundary)
        { return boundary.kind == pdg::ResidencyBoundaryKind::Upload; });
    ASSERT_NE(branchUpload, branches.summary().residencyBoundaries.end());
    EXPECT_EQ(branchUpload->consumers, std::vector<std::size_t>({1U, 2U}));

    for (std::string_view invalidator : {
             R"({"type":"filters.expression","expression":"Z > 0"})",
             R"({"type":"filters.sort","dimension":"Z"})",
             R"({"type":"filters.returns"})",
         })
    {
        pdg::DimensionRegistry invalidationDimensions;
        const std::string pipeline =
            std::string("[\"in.las\",{") +
            R"("type":"filters.outlier","method":"radius","radius":2},)" +
            std::string(invalidator) +
            R"json(,{"type":"filters.outlier","method":"radius","radius":2},"out.las"])json";
        const pdg::Plan invalidated =
            pdg::compilePipeline(pipeline, invalidationDimensions);
        EXPECT_EQ(invalidated.summary().indexBuilds, 2U) << invalidator;
    }

    pdg::DimensionRegistry statisticalDimensions;
    const pdg::Plan statistical = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.outlier","method":"statistical",
             "mean_k":12,"multiplier":1.5}, "out.las"])",
        statisticalDimensions);
    const pdg::PlannedStage& statisticalStage = statistical.stages()[1];
    EXPECT_TRUE(statisticalStage.native);
    EXPECT_EQ(statisticalStage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(statisticalStage.descriptor.index.neighbors, 13U);
    EXPECT_EQ(statisticalStage.preferredResidency, pdg::MemoryKind::Device);
    const auto& statisticalProgram =
        std::get<pdg::OutlierProgram>(statisticalStage.payload);
    EXPECT_EQ(statisticalProgram.method, pdg::OutlierMethod::Statistical);
    EXPECT_EQ(statisticalProgram.meanNeighbors, 12);
    EXPECT_DOUBLE_EQ(statisticalProgram.multiplier, 1.5);

    pdg::DimensionRegistry largeKDimensions;
    const pdg::Plan largeK = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.outlier","mean_k":64}, "out.las"])",
        largeKDimensions);
    EXPECT_EQ(largeK.stages()[1].descriptor.index.neighbors, 65U);
    EXPECT_EQ(largeK.stages()[1].preferredResidency, pdg::MemoryKind::Host);

    for (
        const std::string& pipeline : {
            R"(["in.las", {"type":"filters.outlier","radius":"1"}, "out.las"])",
            R"(["in.las", {"type":"filters.outlier","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.summary().allStagesNative) << pipeline;
        EXPECT_TRUE(std::holds_alternative<pdg::FallbackStagePlan>(
            fallback.stages()[1].payload));
    }
}

TEST(PipelinePlan, CompilesRadialDensityOnTheSharedRadiusIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.radialdensity","radius":2.5},
          "out.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 3U);
    const pdg::PlannedStage& stage = plan.stages()[1];
    EXPECT_TRUE(stage.native);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Radius);
    EXPECT_DOUBLE_EQ(stage.descriptor.index.radius, 2.5);
    EXPECT_DOUBLE_EQ(stage.descriptor.maximumRadius, 2.5);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::RadialDensity)}));
    EXPECT_DOUBLE_EQ(std::get<pdg::RadialDensityProgram>(stage.payload).radius,
                     2.5);
    EXPECT_EQ(plan.summary().indexBuilds, 1U);
    EXPECT_DOUBLE_EQ(plan.summary().maximumRadius, 2.5);

    for (
        std::string_view pipeline : {
            R"(["in.las", {"type":"filters.radialdensity","radius":"2"}, "out.las"])",
            R"(["in.las", {"type":"filters.radialdensity","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(pipeline, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << pipeline;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << pipeline;
    }
}

TEST(PipelinePlan, CompilesRadiusAssignOnTheSharedRadiusIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.radiusassign","radius":1.5,
              "src_domain":"Classification[1:1]",
              "reference_domain":"Classification[2:2]",
              "is3d":false,"max2d_above":2.0,
              "update_expression":"UserData = Intensity + 1 WHERE Classification == 1"},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Radius);
    EXPECT_DOUBLE_EQ(stage.descriptor.index.radius, 1.5);
    EXPECT_DOUBLE_EQ(stage.descriptor.maximumRadius, 1.5);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_TRUE(
        std::find(stage.descriptor.reads.begin(), stage.descriptor.reads.end(),
                  pdg::DimensionId(pdg::StandardDimension::Classification)) !=
        stage.descriptor.reads.end());
    EXPECT_TRUE(
        std::find(stage.descriptor.reads.begin(), stage.descriptor.reads.end(),
                  pdg::DimensionId(pdg::StandardDimension::Intensity)) !=
        stage.descriptor.reads.end());
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::UserData)}));
    const auto& program = std::get<pdg::RadiusAssignProgram>(stage.payload);
    EXPECT_DOUBLE_EQ(program.radius, 1.5);
    EXPECT_FALSE(program.search3d);
    EXPECT_DOUBLE_EQ(program.maximumAbove, 2.0);
    EXPECT_DOUBLE_EQ(program.maximumBelow, -1.0);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.radiusassign","radius":0,"update_expression":"UserData = 1"}, "out.las"])",
            R"(["in.las", {"type":"filters.radiusassign","radius":1,"update_expression":"X = 1"}, "out.las"])",
            R"(["in.las", {"type":"filters.radiusassign","radius":1,"src_domain":"Missing[0:1]","update_expression":"UserData = 1"}, "out.las"])",
            R"(["in.las", {"type":"filters.radiusassign","radius":1,"where":"Z > 0","update_expression":"UserData = 1"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesDefaultNormalOnTheSharedKnnIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.normal","knn":12,
                         "always_up":false}, "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 13U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::NormalX),
                  pdg::DimensionId(pdg::StandardDimension::NormalY),
                  pdg::DimensionId(pdg::StandardDimension::NormalZ),
                  pdg::DimensionId(pdg::StandardDimension::Curvature)}));
    const auto& program = std::get<pdg::NormalProgram>(stage.payload);
    EXPECT_EQ(program.neighbors, 12);
    EXPECT_FALSE(program.alwaysUp);
    EXPECT_EQ(plan.summary().indexBytesPerPoint, 112U);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.normal","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.normal","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.normal","refine":true}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesNnDistanceOnTheSharedKnnIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.nndistance","mode":"avg","k":12},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 13U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::NNDistance)}));
    const auto& program = std::get<pdg::NnDistanceProgram>(stage.payload);
    EXPECT_EQ(program.k, 12U);
    EXPECT_EQ(program.mode, pdg::KnnDistanceMode::Average);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.nndistance","k":0}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","k":64}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","mode":"mean"}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

// B0151/D0213: header-derived placement facts accept a LAZ reader because a
// LAZ public header carries the same point count, format, and record length.
// They must not accept a *configured* reader of either extension: a reader
// option like `count` makes the header's point count wrong, and an earlier
// revision of this rule dropped the option check along with the extension
// check, which the resident process gate caught.
TEST(PipelinePlan, LasFamilyPlacementFactsRequireAnOptionFreeReader)
{
    struct Case
    {
        const char* name;
        const char* json;
        bool optionFree;
        bool native;
    };
    const std::array cases{
        Case{"plain las",
             R"(["in.las", {"type":"filters.normal","knn":8}, "out.las"])",
             true, true},
        Case{"plain laz",
             R"(["in.laz", {"type":"filters.normal","knn":8}, "out.las"])",
             true, true},
        Case{"configured las",
             R"([{"type":"readers.las","filename":"in.las","count":1},
                 {"type":"filters.normal","knn":8}, "out.las"])",
             false, false},
        Case{"configured laz",
             R"([{"type":"readers.las","filename":"in.laz","count":1},
                 {"type":"filters.normal","knn":8}, "out.las"])",
             false, false},
    };
    for (const Case& item : cases)
    {
        SCOPED_TRACE(item.name);
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(item.json, dimensions);
        const auto* reader =
            std::get_if<pdg::FileStagePlan>(&plan.stages().front().payload);
        ASSERT_NE(reader, nullptr);
        EXPECT_EQ(reader->optionFreeLasFamily, item.optionFree);
        // B0188/D0217: an option-free `.laz` reader is native. An earlier
        // revision of this test asserted otherwise on the grounds that
        // `native` also authorizes memory-mapped record access; it does not,
        // and three independent guards keep compressed records off that path.
        // The only mapping consumer is `DirectResidentPointTable`, which is
        // constructed only under `directResidentLasSource`; that in turn
        // requires `!compressedReader` (B0151/D0213); and `FileView::
        // pointRecord` throws outright on compressed data, so the backstop
        // fails closed rather than reading garbage. A *configured* `.laz`
        // reader stays non-native for the original reason, which is unchanged:
        // an option like `count` makes the header's point count wrong.
        EXPECT_EQ(plan.stages().front().native, item.native);
    }
}

TEST(PipelinePlan, CompilesCountOneHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":1,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 1U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z),
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{pdg::DimensionId(
                  pdg::StandardDimension::HeightAboveGround)}));
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 1U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesCountTwoHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":2,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 2U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z),
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{pdg::DimensionId(
                  pdg::StandardDimension::HeightAboveGround)}));
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 2U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesCountThreeHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":3,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 3U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z),
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{pdg::DimensionId(
                  pdg::StandardDimension::HeightAboveGround)}));
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 3U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesCountFourHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":4,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 4U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z),
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{pdg::DimensionId(
                  pdg::StandardDimension::HeightAboveGround)}));
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 4U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesCountFiveHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":5,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 5U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 5U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    pdg::DimensionRegistry fallbackDimensions;
    const pdg::Plan fallback = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
        fallbackDimensions);
    EXPECT_FALSE(fallback.stages()[1].native);
    EXPECT_EQ(fallback.stages()[1].preferredResidency,
              pdg::MemoryKind::Host);
}

TEST(PipelinePlan, CompilesCountSixHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":6,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 6U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 6U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    pdg::DimensionRegistry fallbackDimensions;
    const pdg::Plan fallback = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
        fallbackDimensions);
    EXPECT_FALSE(fallback.stages()[1].native);
    EXPECT_EQ(fallback.stages()[1].preferredResidency,
              pdg::MemoryKind::Host);
}

TEST(PipelinePlan, CompilesCountSevenHagNnOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":7,
             "class":9,"max_distance":17.5,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 7U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    const auto& program = std::get<pdg::HagNnProgram>(stage.payload);
    EXPECT_EQ(program.count, 7U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_DOUBLE_EQ(program.maximumDistance, 17.5);
    EXPECT_FALSE(program.allowExtrapolation);

    pdg::DimensionRegistry fallbackDimensions;
    const pdg::Plan fallback = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
        fallbackDimensions);
    EXPECT_FALSE(fallback.stages()[1].native);
    EXPECT_EQ(fallback.stages()[1].preferredResidency,
              pdg::MemoryKind::Host);
}

TEST(PipelinePlan,
     CompilesOnlyCountThreeHagDelaunayOnTheSharedTwoDimensionalIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_delaunay","count":3,
             "class":9,"allow_extrapolation":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 3U);
    EXPECT_EQ(stage.descriptor.index.dimensions, 2U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z),
                  pdg::DimensionId(pdg::StandardDimension::Classification)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{pdg::DimensionId(
                  pdg::StandardDimension::HeightAboveGround)}));
    const auto& program = std::get<pdg::HagDelaunayProgram>(stage.payload);
    EXPECT_EQ(program.count, 3U);
    EXPECT_EQ(program.groundClass, 9U);
    EXPECT_FALSE(program.allowExtrapolation);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_delaunay"}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":10}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":4}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":2}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, SplitsGridAndCountTwoHagNnResidentProducts)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"([{"type":"readers.las","filename":"in.las"},
             {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
             {"type":"filters.hag_nn","count":2},
             {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
             {"type":"writers.las","filename":"out.las"}])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 5U);
    const std::size_t firstGrid = plan.stages()[1U].residentRegion;
    const std::size_t neighborhood = plan.stages()[2U].residentRegion;
    const std::size_t secondGrid = plan.stages()[3U].residentRegion;
    ASSERT_NE(firstGrid, pdg::NoResidentRegion);
    ASSERT_NE(neighborhood, pdg::NoResidentRegion);
    ASSERT_NE(secondGrid, pdg::NoResidentRegion);
    EXPECT_NE(firstGrid, neighborhood);
    EXPECT_NE(neighborhood, secondGrid);
    EXPECT_NE(firstGrid, secondGrid);

    const auto findBoundary = [&](pdg::ResidencyBoundaryKind kind,
                                  std::size_t producer, std::size_t consumer)
    {
        return std::find_if(plan.summary().residencyBoundaries.begin(),
                            plan.summary().residencyBoundaries.end(),
                            [&](const pdg::ResidencyBoundary& boundary)
                            {
                                return boundary.kind == kind &&
                                       boundary.producer == producer &&
                                       boundary.consumer == consumer;
                            });
    };
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 6U);
    EXPECT_EQ(plan.summary().hostDeviceTransfers, 6U);
    EXPECT_EQ(plan.summary().spillBoundaries, 3U);
    EXPECT_EQ(plan.summary().fallbackBoundaries, 0U);
    const auto initialUpload =
        findBoundary(pdg::ResidencyBoundaryKind::Upload, 0U, 1U);
    const auto firstSpill =
        findBoundary(pdg::ResidencyBoundaryKind::Spill, 1U, 2U);
    const auto hagUpload =
        findBoundary(pdg::ResidencyBoundaryKind::Upload, 1U, 2U);
    const auto hagSpill =
        findBoundary(pdg::ResidencyBoundaryKind::Spill, 2U, 3U);
    const auto secondGridUpload =
        findBoundary(pdg::ResidencyBoundaryKind::Upload, 2U, 3U);
    const auto finalSpill =
        findBoundary(pdg::ResidencyBoundaryKind::Spill, 3U, 4U);
    for (const auto boundary : {initialUpload, firstSpill, hagUpload, hagSpill,
                                secondGridUpload, finalSpill})
    {
        ASSERT_NE(boundary, plan.summary().residencyBoundaries.end());
        EXPECT_FALSE(boundary->dimensions.empty());
        EXPECT_GT(boundary->bytesPerPoint, 0U);
        EXPECT_FALSE(boundary->fallback);
        EXPECT_FALSE(boundary->requiresFullPointRecord);
    }
    const pdg::DimensionId classification(
        pdg::StandardDimension::Classification);
    const pdg::DimensionId hag(pdg::StandardDimension::HeightAboveGround);
    EXPECT_TRUE(contains(firstSpill->releaseDimensions, classification));
    EXPECT_TRUE(contains(firstSpill->repackDimensions, classification));
    EXPECT_TRUE(contains(hagUpload->dimensions, classification));
    EXPECT_TRUE(contains(hagSpill->releaseDimensions, hag));
    EXPECT_TRUE(contains(hagSpill->repackDimensions, hag));
    EXPECT_TRUE(contains(secondGridUpload->dimensions, hag));
}

TEST(PipelinePlan, RebuildsSharedKnnIndexWhenSpatialDimensionsChange)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":1},
             {"type":"filters.nndistance","k":3}, "out.las"])",
        dimensions);
    ASSERT_EQ(plan.stages().size(), 4U);
    ASSERT_TRUE(plan.stages()[1].native);
    ASSERT_TRUE(plan.stages()[2].native);
    EXPECT_EQ(plan.stages()[1].descriptor.index.dimensions, 2U);
    EXPECT_EQ(plan.stages()[2].descriptor.index.dimensions, 3U);
    EXPECT_EQ(plan.stages()[1].residentRegion, plan.stages()[2].residentRegion);
    EXPECT_EQ(plan.summary().indexBuilds, 2U);
    EXPECT_GT(plan.stages()[1].deviceIndexBuildBytesPerPoint, 0U);
    EXPECT_GT(plan.stages()[2].deviceIndexBuildBytesPerPoint, 0U);
}

TEST(PipelinePlan, CompilesApproximateCoplanarWithDirectNeighborCount)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.approximatecoplanar","knn":12,
              "thresh1":30.5,"thresh2":4.25},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 12U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Coplanar)}));
    const auto& program =
        std::get<pdg::ApproximateCoplanarProgram>(stage.payload);
    EXPECT_EQ(program.neighbors, 12);
    EXPECT_DOUBLE_EQ(program.threshold1, 30.5);
    EXPECT_DOUBLE_EQ(program.threshold2, 4.25);
    EXPECT_EQ(plan.summary().indexBytesPerPoint, 112U);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.approximatecoplanar","knn":2}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","knn":65}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","thresh2":"6"}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","where":"Z > 0"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesEigenvaluesOnTheSharedKnnIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.eigenvalues","knn":12,
                         "normalize":true,"stride":1,"min_k":5},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 13U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::Eigenvalue0),
                  pdg::DimensionId(pdg::StandardDimension::Eigenvalue1),
                  pdg::DimensionId(pdg::StandardDimension::Eigenvalue2)}));
    const auto& program = std::get<pdg::EigenvaluesProgram>(stage.payload);
    EXPECT_EQ(program.neighbors, 12);
    EXPECT_TRUE(program.normalize);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.eigenvalues","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.eigenvalues","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.eigenvalues","stride":2}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, CompilesCovarianceFeaturesOnTheSharedKnnIndex)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las", {"type":"filters.covariancefeatures","knn":12,
                         "threads":1,
                         "feature_set":["dimensionality","omnivariance",
                                        "anisotropy","eigenentropy",
                                        "eigenvaluesum","surfacevariation",
                                        "demantkeverticality"],
                         "stride":1,"min_k":5,"mode":"normalized",
                         "optimized":false},
             "out.las"])",
        dimensions);
    const pdg::PlannedStage& stage = plan.stages()[1];
    ASSERT_TRUE(stage.native);
    EXPECT_EQ(stage.descriptor.kind, pdg::StageKind::Knn);
    EXPECT_EQ(stage.descriptor.index.kind, pdg::IndexKind::Knn);
    EXPECT_EQ(stage.descriptor.index.neighbors, 13U);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    EXPECT_EQ(stage.descriptor.reads,
              (std::vector<pdg::DimensionId>{
                  pdg::DimensionId(pdg::StandardDimension::X),
                  pdg::DimensionId(pdg::StandardDimension::Y),
                  pdg::DimensionId(pdg::StandardDimension::Z)}));
    EXPECT_EQ(stage.descriptor.writes.size(), 10U);
    EXPECT_EQ(stage.descriptor.writes.front(),
              pdg::DimensionId(pdg::StandardDimension::Linearity));
    EXPECT_EQ(stage.descriptor.writes.back(),
              pdg::DimensionId(pdg::StandardDimension::DemantkeVerticality));
    const auto& program =
        std::get<pdg::CovarianceFeaturesProgram>(stage.payload);
    EXPECT_EQ(program.neighbors, 12);
    EXPECT_EQ(program.mode, pdg::EigenvalueMode::Normalized);
    EXPECT_EQ(program.features,
              pdg::CovarianceDimensionality | pdg::CovarianceOmnivariance |
                  pdg::CovarianceAnisotropy | pdg::CovarianceEigenentropy |
                  pdg::CovarianceEigenvalueSum |
                  pdg::CovarianceSurfaceVariation |
                  pdg::CovarianceDemantkeVerticality);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.covariancefeatures","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","stride":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","threads":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","feature_set":"density"}, "out.las"])",
        })
    {
        pdg::DimensionRegistry fallbackDimensions;
        const pdg::Plan fallback =
            pdg::compilePipeline(unsupported, fallbackDimensions);
        EXPECT_FALSE(fallback.stages()[1].native) << unsupported;
        EXPECT_EQ(fallback.stages()[1].preferredResidency,
                  pdg::MemoryKind::Host)
            << unsupported;
    }
}

TEST(PipelinePlan, RandomizeDeclaresCardinalityAcrossHostBoundary)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"(["in.las",
             {"type":"filters.assign","value":"Classification = 7"},
             {"type":"filters.randomize","seed":17},
             {"type":"filters.assign","value":"UserData = 3"},
             "out.las"])",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 5U);
    EXPECT_FALSE(plan.stages()[2].native);
    EXPECT_EQ(plan.stages()[2].preferredResidency, pdg::MemoryKind::Host);
    EXPECT_TRUE(plan.stages()[2].descriptor.fusion.cardinalityPreserving);
    EXPECT_FALSE(plan.stages()[2].descriptor.preservesOrder);
    EXPECT_EQ(plan.summary().residentRegions, 2U);
    EXPECT_EQ(plan.summary().fallbackBoundaries, 2U);
}

TEST(PipelinePlan, AdmitsExactExtraDimensionsAllWriterMetadata)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.las",
             "extra_dims":"all"}
        ]})",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    EXPECT_TRUE(plan.stages()[0U].native);
    EXPECT_TRUE(plan.stages()[1U].native);
    EXPECT_TRUE(plan.stages()[2U].native);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&plan.stages()[2U].payload);
    ASSERT_NE(writer, nullptr);
    EXPECT_TRUE(writer->extraDimensionsAll);
    EXPECT_EQ(plan.stages()[1U].residentRegion, 0U);
    EXPECT_EQ(plan.summary().residentRegions, 1U);
    EXPECT_TRUE(plan.summary().allStagesNative);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    const pdg::ResidencyBoundary& spill =
        plan.summary().residencyBoundaries.back();
    EXPECT_EQ(spill.kind, pdg::ResidencyBoundaryKind::Spill);
    EXPECT_FALSE(spill.fallback);
    EXPECT_EQ(spill.dimensions,
              plan.stages()[1U].descriptor.writes);

    const pdg::Plan changedWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.las",
             "extra_dims":"all","minor_version":4}
        ]})",
        dimensions);
    const auto* changed =
        std::get_if<pdg::FileStagePlan>(&changedWriter.stages()[2U].payload);
    ASSERT_NE(changed, nullptr);
    EXPECT_FALSE(changed->extraDimensionsAll);
    EXPECT_FALSE(changedWriter.stages()[2U].native);

    const pdg::Plan compressedWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.laz",
             "compression":true,"extra_dims":"all"}
        ]})",
        dimensions);
    const auto* compressed =
        std::get_if<pdg::FileStagePlan>(&compressedWriter.stages()[2U].payload);
    ASSERT_NE(compressed, nullptr);
    EXPECT_TRUE(compressed->extraDimensionsAll);
    EXPECT_TRUE(compressedWriter.stages()[2U].native);
    EXPECT_TRUE(compressedWriter.summary().allStagesNative);

    const pdg::Plan mixedCaseCompressedWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.laz",
             "compression":"TrUe","extra_dims":"all"}
        ]})",
        dimensions);
    const auto* mixedCase = std::get_if<pdg::FileStagePlan>(
        &mixedCaseCompressedWriter.stages()[2U].payload);
    ASSERT_NE(mixedCase, nullptr);
    EXPECT_TRUE(mixedCase->extraDimensionsAll);
    EXPECT_TRUE(mixedCaseCompressedWriter.stages()[2U].native);

    const pdg::Plan implicitCompressedWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.laz",
             "extra_dims":"all"}
        ]})",
        dimensions);
    const auto* implicit = std::get_if<pdg::FileStagePlan>(
        &implicitCompressedWriter.stages()[2U].payload);
    ASSERT_NE(implicit, nullptr);
    EXPECT_TRUE(implicit->extraDimensionsAll);
    EXPECT_TRUE(implicitCompressedWriter.stages()[2U].native);

    const pdg::Plan compressedWriterWithAdditionalOption =
        pdg::compilePipeline(
            R"({"pipeline":[
                {"type":"readers.las","filename":"input.las"},
                {"type":"filters.hag_nn","count":4},
                {"type":"writers.las","filename":"output.laz",
                 "compression":true,"extra_dims":"all","minor_version":4}
            ]})",
            dimensions);
    const auto* additional = std::get_if<pdg::FileStagePlan>(
        &compressedWriterWithAdditionalOption.stages()[2U].payload);
    ASSERT_NE(additional, nullptr);
    EXPECT_FALSE(additional->extraDimensionsAll);
    EXPECT_FALSE(compressedWriterWithAdditionalOption.stages()[2U].native);

    const pdg::Plan disabledCompressionWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.laz",
             "compression":false,"extra_dims":"all"}
        ]})",
        dimensions);
    const auto* disabled = std::get_if<pdg::FileStagePlan>(
        &disabledCompressionWriter.stages()[2U].payload);
    ASSERT_NE(disabled, nullptr);
    EXPECT_FALSE(disabled->extraDimensionsAll);
    EXPECT_FALSE(disabledCompressionWriter.stages()[2U].native);

    const pdg::Plan namedCompressedWriter = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.hag_nn","count":4},
            {"type":"writers.las","filename":"output.laz",
             "compression":true,
             "extra_dims":"HeightAboveGround=float32"}
        ]})",
        dimensions);
    const auto* named = std::get_if<pdg::FileStagePlan>(
        &namedCompressedWriter.stages()[2U].payload);
    ASSERT_NE(named, nullptr);
    EXPECT_FALSE(named->extraDimensionsAll);
    EXPECT_FALSE(namedCompressedWriter.stages()[2U].native);
}

TEST(PipelinePlan, PublishesAReorderOnlyRegionAsANonColumnResult)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(
        R"({"pipeline":[
            {"type":"readers.las","filename":"input.las"},
            {"type":"filters.sort","dimension":"Z","order":"ASC",
             "algorithm":"NORMAL"},
            {"type":"writers.las","filename":"output.las",
             "extra_dims":"all"}
        ]})",
        dimensions);

    ASSERT_EQ(plan.stages().size(), 3U);
    EXPECT_TRUE(plan.stages()[2U].native);
    ASSERT_EQ(plan.summary().residencyBoundaries.size(), 2U);
    const pdg::ResidencyBoundary& spill =
        plan.summary().residencyBoundaries.back();
    EXPECT_EQ(spill.kind, pdg::ResidencyBoundaryKind::Spill);
    EXPECT_FALSE(spill.fallback);
    EXPECT_TRUE(spill.dimensions.empty());
    EXPECT_EQ(plan.stages()[1U].descriptor.deviceToHostBytesPerInputPoint,
              sizeof(std::uint64_t));
    EXPECT_EQ(plan.stages()[1U].descriptor.deviceToHostFixedBytes, 0U);
}

TEST(PipelinePlan, ResolvesExplicitInputsAndRejectsInvalidPipelines)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan branched = pdg::compilePipeline(
        R"([
            {"type":"readers.las", "filename":"a.las", "tag":"A"},
            {"type":"readers.las", "filename":"b.las", "tag":"B"},
            {"type":"filters.ferry", "dimensions":"X=>CopyX",
             "inputs":["A", "B"], "tag":"Merge"},
            {"type":"writers.las", "filename":"out.las",
             "inputs":"Merge"}
        ])",
        dimensions);
    ASSERT_EQ(branched.stages().size(), 4U);
    EXPECT_EQ(branched.stages()[2].inputs, std::vector<std::size_t>({0, 1}));
    EXPECT_EQ(branched.stages()[3].inputs, std::vector<std::size_t>({2}));

    const std::vector<std::string> invalid = {
        "",
        "{}",
        "[]",
        "[12]",
        R"([{"type":"writers.las", "filename":"out.las"}])",
        R"([{"type":"readers.las", "filename":"in.las", "tag":"9bad"}])",
        R"([{"type":"readers.las", "filename":"a.las", "tag":"same"},
             {"type":"readers.las", "filename":"b.las", "tag":"same"}])",
        R"(["in.las", {"type":"filters.ferry",
                         "dimensions":"X=>CopyX", "inputs":"missing"}])",
        R"(["in.las", {"type":"filters.ferry", "dimensions":"X"}])",
        R"(["in.las", {"type":"filters.ferry", "dimensions":"X=>X"}])",
        R"(["in.las", {"type":"filters.ferry",
                         "dimensions":["X=>Copy", "Y=>Copy"]}])",
        R"(["in.las", {"type":"filters.ferry",
                         "dimensions":"Missing=>Copy"}])",
        R"(["in.las", {"type":"filters.ferry",
                         "dimensions":"X=>A=>B"}])",
        R"(["in.las", {"type":"filters.ferry", "dimensions":4}])",
        R"(["in.las", {"type":"filters.assign", "value":4}])",
        R"(["in.las", {"type":"filters.assign", "value":"X ="}])",
        R"(["in.las", {"type":"filters.assign",
                          "value":"X = 1, Y = 2"}])",
    };

    for (const std::string& pipeline : invalid)
    {
        pdg::DimensionRegistry caseDimensions;
        EXPECT_THROW(
            static_cast<void>(pdg::compilePipeline(pipeline, caseDimensions)),
            pdg::PlanError)
            << pipeline;
    }
}
