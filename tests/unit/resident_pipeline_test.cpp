#include <pdg/Hybrid.hpp>
#include <pdg/Placement.hpp>
#include <pdg/Plan.hpp>
#include <pdg/ResidentPipeline.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <iterator>
#include <string>

namespace
{
using Json = nlohmann::json;

pdg::Plan compile(std::string_view pipeline)
{
    pdg::DimensionRegistry dimensions;
    return pdg::compilePipeline(pipeline, dimensions);
}

pdg::PlanPlacementEstimate selectAll(const pdg::Plan& plan)
{
    pdg::PlanPlacementEstimate placement;
    placement.choice = pdg::PlacementChoice::Device;
    for (std::size_t region = 0; region < plan.summary().residentRegions;
         ++region)
    {
        pdg::PlacementRegionEstimate selected;
        selected.residentRegion = region;
        selected.selected = true;
        for (const pdg::PlannedStage& stage : plan.stages())
            if (stage.residentRegion == region)
                selected.stageIds.push_back(stage.id);
        placement.regions.push_back(std::move(selected));
    }
    placement.selectedRegionCount = placement.regions.size();
    return placement;
}
} // unnamed namespace

TEST(ResidentPipeline, MaterializesSelectedPointProgramRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Classification = 7"},
      {"type":"filters.ferry","dimensions":"Classification=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 5U);
    const Json& upload = document.at("pipeline").at(1U);
    EXPECT_EQ(upload.at("type"), pdg::HybridResidentBoundaryStage);
    EXPECT_EQ(upload.at("pdg_boundary_kind"), "upload");
    EXPECT_FALSE(upload.at("pdg_requires_full_point_record").get<bool>());
    const Json& replacement = document.at("pipeline").at(2U);
    EXPECT_EQ(replacement.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(replacement.at("pdg_plan_cuda").get<bool>());
    EXPECT_TRUE(replacement.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(replacement.at("pdg_execution_region"), 0U);
    EXPECT_EQ(Json::parse(replacement.at("program").get<std::string>()).size(),
              2U);
    const Json& spill = document.at("pipeline").at(3U);
    EXPECT_EQ(spill.at("type"), pdg::HybridResidentBoundaryStage);
    EXPECT_EQ(spill.at("pdg_boundary_kind"), "spill");
    EXPECT_FALSE(spill.at("pdg_requires_full_point_record").get<bool>());
}

TEST(ResidentPipeline, MaterializesFullRecordRandomizeBoundaryBetweenRegions)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Classification = 7"},
      {"type":"filters.randomize","seed":17},
      {"type":"filters.assign","value":"UserData = 3"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    const auto initialUpload = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(),
        [](const pdg::ResidencyBoundary& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Upload &&
                   boundary.producer == 0U && boundary.consumer == 1U &&
                   !boundary.fallback && !boundary.requiresFullPointRecord;
        });
    const auto fallbackSpill = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(),
        [](const pdg::ResidencyBoundary& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Spill &&
                   boundary.producer == 1U && boundary.consumer == 2U &&
                   boundary.fallback && boundary.requiresFullPointRecord;
        });
    const auto fallbackUpload = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(),
        [](const pdg::ResidencyBoundary& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Upload &&
                   boundary.producer == 2U && boundary.consumer == 3U &&
                   boundary.fallback && boundary.requiresFullPointRecord;
        });
    const auto finalSpill = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(),
        [](const pdg::ResidencyBoundary& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Spill &&
                   boundary.producer == 3U && boundary.consumer == 4U &&
                   !boundary.fallback && !boundary.requiresFullPointRecord;
        });
    ASSERT_NE(initialUpload, plan.summary().residencyBoundaries.end());
    ASSERT_NE(fallbackSpill, plan.summary().residencyBoundaries.end());
    ASSERT_NE(fallbackUpload, plan.summary().residencyBoundaries.end());
    ASSERT_NE(finalSpill, plan.summary().residencyBoundaries.end());
    const auto boundaryId = [&](const auto position)
    {
        return static_cast<std::size_t>(std::distance(
            plan.summary().residencyBoundaries.begin(), position));
    };

    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 9U);
    const auto expectBoundary = [&](std::size_t stageIndex,
                                    std::string_view kind, std::size_t id,
                                    bool fullRecord)
    {
        EXPECT_EQ(stages.at(stageIndex).at("type"),
                  pdg::HybridResidentBoundaryStage);
        EXPECT_EQ(stages.at(stageIndex).at("pdg_boundary_kind"), kind);
        EXPECT_EQ(stages.at(stageIndex).at("pdg_boundary_id"), id);
        EXPECT_EQ(stages.at(stageIndex)
                      .at("pdg_requires_full_point_record")
                      .get<bool>(),
                  fullRecord);
    };
    expectBoundary(1U, "upload", boundaryId(initialUpload), false);
    EXPECT_EQ(stages.at(2U).at("type"), pdg::HybridPointProgramStage);
    expectBoundary(3U, "spill", boundaryId(fallbackSpill), true);
    EXPECT_EQ(stages.at(4U).at("type"), "filters.randomize");
    EXPECT_EQ(stages.at(4U).at("seed"), 17U);
    expectBoundary(5U, "upload", boundaryId(fallbackUpload), true);
    EXPECT_EQ(stages.at(6U).at("type"), pdg::HybridPointProgramStage);
    expectBoundary(7U, "spill", boundaryId(finalSpill), false);
    EXPECT_EQ(stages.at(2U).at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(6U).at("pdg_execution_region"), 1U);
}

TEST(ResidentPipeline, MaterializesDeclaredCardinalityChangingExpressionRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":[
        "Scratch = Intensity * 2 - 1",
        "Classification = 7 WHERE Scratch >= 1000 && ReturnNumber >= 1"]},
      {"type":"filters.ferry","dimensions":"Classification=>UserData"},
      {"type":"filters.expression","expression":"Intensity <= 30000"},
      {"type":"filters.assign","value":[
        "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
        "ReturnNumber = UserData WHERE UserData >= 1 && UserData <= 15"]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.selectedStageIds,
              (std::vector<std::size_t>{1U, 2U, 3U, 4U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 5U);
    EXPECT_EQ(document.at("pipeline").at(1U).at("pdg_boundary_kind"), "upload");
    const Json& replacement = document.at("pipeline").at(2U);
    EXPECT_EQ(replacement.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(replacement.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(Json::parse(replacement.at("program").get<std::string>()).size(),
              4U);
    EXPECT_EQ(document.at("pipeline").at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, RejectsASecondDeclaredCardinalityChangeInOneRegion)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.expression","expression":"Intensity <= 30000"},
      {"type":"filters.expression","expression":"Intensity >= 100"},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "selected resident region declares more than one cardinality "
              "change");
}

TEST(ResidentPipeline, RejectsDeclaredWhereSemanticsByDescriptorFlags)
{
    // The rewrite must reject declared conditional semantics from the
    // descriptor contract itself, not from option names. Build the plan by
    // hand so a device-preferred predicate stage carries declared
    // where/where_merge state that pipeline compilation would have routed to
    // the host fallback.
    pdg::DimensionRegistry dimensions;
    const std::string_view pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.expression","expression":"Intensity <= 30000"},
      {"type":"writers.las","filename":"out.las"}
    ])";
    pdg::Plan compiled = compile(pipeline);
    std::vector<pdg::PlannedStage> stages = compiled.stages();
    pdg::PlanSummary summary = compiled.summary();
    ASSERT_EQ(stages.size(), 3U);
    ASSERT_EQ(stages[1].residentRegion, 0U);
    stages[1].descriptor.fusion.hasWhere = true;
    stages[1].descriptor.fusion.whereMerge = pdg::WhereMergeMode::Auto;
    const pdg::Plan declaredWhere(stages, summary);
    pdg::ResidentPipelineRewrite rewritten = pdg::rewriteResidentPlacement(
        pipeline, declaredWhere, selectAll(declaredWhere));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "selected resident stage declares conditional where semantics");

    stages = compiled.stages();
    stages[1].descriptor.preservesOrder = false;
    const pdg::Plan unordered(stages, summary);
    rewritten = pdg::rewriteResidentPlacement(pipeline, unordered,
                                              selectAll(unordered));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "selected resident cardinality change does not declare stable "
              "order");
}

TEST(ResidentPipeline, MaterializesASharedIndexNeighborhoodRegion)
{
    // V2: a declared kNN neighborhood stage with a trailing point-program
    // consumer forms one resident region between one upload/spill boundary
    // pair. The neighborhood stage rewrites to its shared-index wrapper and
    // the consumer to a point-program node bound to the same neighborhood
    // region so resident columns are consumed without a host round trip.
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.approximatecoplanar","knn":8},
      {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 6U);
    EXPECT_EQ(document.at("pipeline").at(1U).at("pdg_boundary_kind"), "upload");
    const Json& neighborhood = document.at("pipeline").at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridApproximateCoplanarStage);
    EXPECT_EQ(neighborhood.at("knn"), 8);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    const Json& consumer = document.at("pipeline").at(3U);
    EXPECT_EQ(consumer.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(consumer.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(consumer.at("pdg_neighborhood_region_id"),
              neighborhood.at("pdg_region_id"));
    EXPECT_TRUE(consumer.at("pdg_neighborhood_region_last").get<bool>());
    EXPECT_EQ(document.at("pipeline").at(4U).at("pdg_boundary_kind"), "spill");
    // D0262: the region is followed by the terminal writer alone, so no later
    // stage can consume a PointView KD3 product; the wrapper's exact tie
    // repair may keep its compatibility tree private.
    EXPECT_TRUE(neighborhood.at("pdg_region_terminal_sink").get<bool>());
}

TEST(ResidentPipeline, TerminalSinkMarkerIsFalseBeforeAHostConsumer)
{
    // The r11-like tail: a host neighbor classifier follows the resident
    // region, so a published KD3 product could be observed and the marker
    // must stay false (the repair keeps publishing the uncached tree).
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.normal","knn":8},
      {"type":"filters.assign","value":"X = X * 3"},
      {"type":"filters.neighborclassifier","k":7,"domain":"Classification[1:1]"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_GE(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    if (!rewritten.executable)
        GTEST_SKIP() << "graph not resident-executable: " << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    bool sawNormal = false;
    for (const Json& stage : document.at("pipeline"))
    {
        if (stage.contains("type") &&
            stage.at("type") == pdg::HybridNormalStage)
        {
            sawNormal = true;
            EXPECT_FALSE(stage.at("pdg_region_terminal_sink").get<bool>());
        }
    }
    EXPECT_TRUE(sawNormal);
}

TEST(ResidentPipeline, MaterializesAdjacentLabelingWithoutAPrivateIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.label_duplicates","dimensions":"Classification"},
      {"type":"filters.assign","value":"UserData = Duplicate"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& labeling = stages.at(2U);
    EXPECT_EQ(labeling.at("type"), pdg::HybridLabelDuplicatesStage);
    EXPECT_TRUE(labeling.at("pdg_resident_context").get<bool>());
    EXPECT_FALSE(labeling.at("pdg_region_index_required").get<bool>());
    EXPECT_FALSE(labeling.at("pdg_region_last").get<bool>());
    const Json& consumer = stages.at(3U);
    EXPECT_EQ(consumer.at("type"), pdg::HybridPointProgramStage);
    EXPECT_EQ(consumer.at("pdg_neighborhood_region_id"),
              labeling.at("pdg_region_id"));
    EXPECT_TRUE(consumer.at("pdg_neighborhood_region_last").get<bool>());
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesAStandaloneGlobalSmrfGridRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.smrf","cell":1.0,"window":7.0,"cut":5.0,
       "returns":[]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& smrf = stages.at(2U);
    EXPECT_EQ(smrf.at("type"), pdg::HybridSmrfStage);
    EXPECT_TRUE(smrf.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(smrf.at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesASmrfGridBridge)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.smrf","returns":[]},
      {"type":"filters.assign","value":"UserData = Classification"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 8U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& smrf = stages.at(2U);
    EXPECT_EQ(smrf.at("type"), pdg::HybridSmrfStage);
    EXPECT_TRUE(smrf.at("pdg_resident_context").get<bool>());
    const Json& smrfUpload = stages.at(1U);
    const Json& smrfSpill = stages.at(3U);
    EXPECT_EQ(smrfUpload.at("pdg_execution_region"),
              smrf.at("pdg_execution_region"));
    EXPECT_EQ(smrfSpill.at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(smrfSpill.at("pdg_execution_region"),
              smrf.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "upload");
    const Json& assign = stages.at(5U);
    EXPECT_EQ(assign.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(assign.at("pdg_resident_context").get<bool>());
    EXPECT_NE(smrf.at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(6U).at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(stages.at(6U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
}

TEST(ResidentPipeline, MaterializesAStandaloneGlobalPmfGridRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":9.0,
       "returns":[]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& pmf = stages.at(2U);
    EXPECT_EQ(pmf.at("type"), pdg::HybridPmfStage);
    EXPECT_TRUE(pmf.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(pmf.at("pdg_execution_region"), 0U);
    EXPECT_FALSE(pmf.at("pdg_grid_reuse").get<bool>());
    EXPECT_TRUE(pmf.at("pdg_grid_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesAdjacentPmfAllocationReuse)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":3.0,
       "returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"max_window_size":5.0,
       "returns":"only","only_ground":true,"ground_class":9,
       "other_class":9},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().gridBuilds, 2U);
    EXPECT_EQ(pdg::selectedGridBuildCount(plan, selectAll(plan)), 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& first = stages.at(2U);
    EXPECT_EQ(first.at("type"), pdg::HybridPmfStage);
    EXPECT_TRUE(first.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(first.at("pdg_execution_region"), 0U);
    EXPECT_FALSE(first.at("pdg_grid_reuse").get<bool>());
    EXPECT_FALSE(first.at("pdg_grid_region_last").get<bool>());
    const Json& second = stages.at(3U);
    EXPECT_EQ(second.at("type"), pdg::HybridPmfStage);
    EXPECT_TRUE(second.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(second.at("pdg_execution_region"), 0U);
    EXPECT_TRUE(second.at("pdg_grid_reuse").get<bool>());
    EXPECT_TRUE(second.at("pdg_grid_region_last").get<bool>());
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, RejectsAdjacentPmfWithDifferentSourceSelection)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"returns":"only"},
      {"type":"filters.pmf","cell_size":1.0,"returns":"last"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "adjacent resident pmf stages have different raster sources");
}

TEST(ResidentPipeline, RejectsAdjacentPmfWithImplicitAndExplicitReturnSource)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0},
      {"type":"filters.pmf","cell_size":1.0,
       "returns":["first","intermediate","last","only"]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "adjacent resident pmf stages have different raster sources");
}

TEST(ResidentPipeline, RejectsAdjacentPmfWithDifferentCellFrames)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","cell_size":1.0,"returns":"only"},
      {"type":"filters.pmf","cell_size":0.5,"returns":"only"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "adjacent resident pmf stages have different raster sources");
}

TEST(ResidentPipeline, MaterializesAPmfGridBridge)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","returns":[]},
      {"type":"filters.assign","value":"UserData = Classification"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 8U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& pmf = stages.at(2U);
    EXPECT_EQ(pmf.at("type"), pdg::HybridPmfStage);
    EXPECT_TRUE(pmf.at("pdg_resident_context").get<bool>());
    const Json& pmfUpload = stages.at(1U);
    const Json& pmfSpill = stages.at(3U);
    EXPECT_EQ(pmfUpload.at("pdg_execution_region"),
              pmf.at("pdg_execution_region"));
    EXPECT_EQ(pmfSpill.at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(pmfSpill.at("pdg_execution_region"),
              pmf.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "upload");
    const Json& assign = stages.at(5U);
    EXPECT_EQ(assign.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(assign.at("pdg_resident_context").get<bool>());
    EXPECT_NE(pmf.at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(6U).at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(stages.at(6U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
}

TEST(ResidentPipeline, RejectsPmfToDifferentGridKind)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","returns":[]},
      {"type":"filters.smrf","returns":[]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "selected resident pmf region has no composable grid bridge");
}

TEST(ResidentPipeline, MaterializesAStandaloneGlobalCsfGridRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.csf","smooth":false,"iterations":3,"returns":[]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& csf = stages.at(2U);
    EXPECT_EQ(csf.at("type"), pdg::HybridCsfStage);
    EXPECT_TRUE(csf.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(csf.at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesACsfGridBridge)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.csf","smooth":false,"iterations":3,"returns":[]},
      {"type":"filters.assign","value":"UserData = Classification"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 8U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& csf = stages.at(2U);
    EXPECT_EQ(csf.at("type"), pdg::HybridCsfStage);
    EXPECT_TRUE(csf.at("pdg_resident_context").get<bool>());
    const Json& csfUpload = stages.at(1U);
    const Json& csfSpill = stages.at(3U);
    EXPECT_EQ(csfUpload.at("pdg_execution_region"),
              csf.at("pdg_execution_region"));
    EXPECT_EQ(csfSpill.at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(csfSpill.at("pdg_execution_region"),
              csf.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "upload");
    const Json& assign = stages.at(5U);
    EXPECT_EQ(assign.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(assign.at("pdg_resident_context").get<bool>());
    EXPECT_NE(csf.at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(6U).at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(stages.at(6U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
}

TEST(ResidentPipeline, MaterializesAStandaloneGlobalElmGridRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.elm","cell":1.25,"threshold":-1.0},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& elm = stages.at(2U);
    EXPECT_EQ(elm.at("type"), pdg::HybridElmStage);
    EXPECT_TRUE(elm.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(elm.at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesAnElmGridBridge)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.elm"},
      {"type":"filters.assign","value":"UserData = Classification"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 8U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& elm = stages.at(2U);
    EXPECT_EQ(elm.at("type"), pdg::HybridElmStage);
    EXPECT_TRUE(elm.at("pdg_resident_context").get<bool>());
    const Json& elmUpload = stages.at(1U);
    const Json& elmSpill = stages.at(3U);
    EXPECT_EQ(elmUpload.at("pdg_execution_region"),
              elm.at("pdg_execution_region"));
    EXPECT_EQ(elmSpill.at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(elmSpill.at("pdg_execution_region"),
              elm.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_boundary_kind"), "upload");
    const Json& assign = stages.at(5U);
    EXPECT_EQ(assign.at("type"), pdg::HybridPointProgramStage);
    EXPECT_TRUE(assign.at("pdg_resident_context").get<bool>());
    EXPECT_NE(elm.at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(4U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
    EXPECT_EQ(stages.at(6U).at("pdg_boundary_kind"), "spill");
    EXPECT_EQ(stages.at(6U).at("pdg_execution_region"),
              assign.at("pdg_execution_region"));
}

TEST(ResidentPipeline, MaterializesAStandaloneSkewnessGlobalRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.skewnessbalancing"},
      {"type":"writers.las","filename":"out.las","extra_dims":"all"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& skewness = stages.at(2U);
    EXPECT_EQ(skewness.at("type"), pdg::HybridSkewnessStage);
    EXPECT_TRUE(skewness.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(skewness.at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, MaterializesAStandaloneSortGlobalRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.sort","dimension":"Z","order":"ASC",
       "algorithm":"NORMAL"},
      {"type":"writers.las","filename":"out.las","extra_dims":"all"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 5U);
    EXPECT_EQ(stages.at(1U).at("pdg_boundary_kind"), "upload");
    const Json& sort = stages.at(2U);
    EXPECT_EQ(sort.at("type"), pdg::HybridOrderStage);
    EXPECT_TRUE(sort.at("pdg_resident_context").get<bool>());
    EXPECT_EQ(sort.at("pdg_execution_region"), 0U);
    EXPECT_EQ(stages.at(3U).at("pdg_boundary_kind"), "spill");
}

TEST(ResidentPipeline, RejectsResidentSortOutsideTheDirectExactEnvelope)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.sort","dimension":"Z","order":"DESC",
       "algorithm":"NORMAL"},
      {"type":"writers.las","filename":"out.las","extra_dims":"all"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_NE(rewritten.reason.find("ordering region"), std::string::npos)
        << rewritten.reason;
}

TEST(ResidentPipeline, LabelingCanPrecedeAPlannerOwnedSharedIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.label_duplicates","dimensions":"Classification"},
      {"type":"filters.nndistance","k":3},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    // The stages share one index, but label_duplicates does not produce an
    // ordered kNN rowset. Gather reuse remains reserved for the adjacent
    // statistical-outlier projection pair.
    EXPECT_EQ(plan.stages().at(1U).deviceKnnGatherNeighbors, 0U);
    EXPECT_EQ(plan.stages().at(2U).deviceKnnGatherNeighbors, 0U);
    EXPECT_EQ(plan.stages().at(1U).deviceQueryBytesPerPoint, 0U);
    EXPECT_EQ(plan.stages().at(2U).deviceQueryBytesPerPoint, 0U);
    EXPECT_EQ(plan.summary().peakDeviceQueryBytesPerPoint, 0U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& labeling = stages.at(2U);
    const Json& neighborhood = stages.at(3U);
    EXPECT_EQ(labeling.at("type"), pdg::HybridLabelDuplicatesStage);
    EXPECT_TRUE(labeling.at("pdg_region_index_required").get<bool>());
    EXPECT_EQ(labeling.at("pdg_region_neighbors"), 4U);
    EXPECT_FALSE(labeling.at("pdg_region_last").get<bool>());
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridNnDistanceStage);
    EXPECT_EQ(neighborhood.at("pdg_region_id"), labeling.at("pdg_region_id"));
    EXPECT_TRUE(neighborhood.at("pdg_region_last").get<bool>());
}

TEST(ResidentPipeline, MaterializesCountOneHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":1,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 1U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountTwoHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":2,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 2U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountThreeHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":3,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 3U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountFourHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":4,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 4U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountFiveHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":5,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 5U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountSixHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":6,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 6U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline, MaterializesCountSevenHagNnWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":7,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 7U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    EXPECT_EQ(stages.at(3U).at("type"), pdg::HybridPointProgramStage);
}

TEST(ResidentPipeline,
     MaterializesCountThreeHagDelaunayWithATwoDimensionalIndex)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_delaunay","count":3,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 6U);
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridHagDelaunayStage);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 3U);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    const Json& bridge = stages.at(3U);
    EXPECT_EQ(bridge.at("type"), pdg::HybridPointProgramStage);
    EXPECT_EQ(bridge.at("pdg_neighborhood_region_id"),
              neighborhood.at("pdg_region_id"));
    EXPECT_TRUE(bridge.at("pdg_neighborhood_region_last").get<bool>());
}

TEST(ResidentPipeline, MaterializesPmfCountTwoHagNnPmfBoundarySequence)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"filters.hag_nn","count":2},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[]},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 3U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds,
              (std::vector<std::size_t>{1U, 2U, 3U}));
    const Json stages = Json::parse(rewritten.json).at("pipeline");
    ASSERT_EQ(stages.size(), 11U);
    const auto expectBoundary =
        [&](std::size_t index, std::string_view kind, std::uint64_t region)
    {
        EXPECT_EQ(stages.at(index).at("type"),
                  pdg::HybridResidentBoundaryStage);
        EXPECT_EQ(stages.at(index).at("pdg_boundary_kind"), kind);
        EXPECT_EQ(stages.at(index).at("pdg_execution_region"), region);
        EXPECT_FALSE(
            stages.at(index).at("pdg_requires_full_point_record").get<bool>());
    };
    expectBoundary(1U, "upload", 0U);
    EXPECT_EQ(stages.at(2U).at("type"), pdg::HybridPmfStage);
    EXPECT_EQ(stages.at(2U).at("pdg_execution_region"), 0U);
    expectBoundary(3U, "spill", 0U);
    expectBoundary(4U, "upload", 1U);
    EXPECT_EQ(stages.at(5U).at("type"), pdg::HybridHagNnStage);
    EXPECT_EQ(stages.at(5U).at("pdg_execution_region"), 1U);
    EXPECT_EQ(stages.at(5U).at("pdg_region_neighbors"), 2U);
    expectBoundary(6U, "spill", 1U);
    expectBoundary(7U, "upload", 2U);
    EXPECT_EQ(stages.at(8U).at("type"), pdg::HybridPmfStage);
    EXPECT_EQ(stages.at(8U).at("pdg_execution_region"), 2U);
    expectBoundary(9U, "spill", 2U);
}

TEST(ResidentPipeline, RejectsBranchedGridAndNeighborhoodSelection)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las","tag":"source"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[],
       "inputs":"source","tag":"grid_left"},
      {"type":"filters.pmf","max_window_size":3.0,"returns":[],
       "inputs":"source","tag":"grid_right"},
      {"type":"filters.hag_nn","count":2,"inputs":"source",
       "tag":"neighborhood"},
      {"type":"writers.las","filename":"out.las",
       "inputs":"neighborhood"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.stages().size(), 5U);
    ASSERT_NE(plan.stages()[1U].residentRegion, pdg::NoResidentRegion);
    EXPECT_EQ(plan.stages()[1U].residentRegion,
              plan.stages()[2U].residentRegion);
    ASSERT_NE(plan.stages()[3U].residentRegion, pdg::NoResidentRegion);
    EXPECT_NE(plan.stages()[1U].residentRegion,
              plan.stages()[3U].residentRegion);
    const auto branchingUpload = std::find_if(
        plan.summary().residencyBoundaries.begin(),
        plan.summary().residencyBoundaries.end(),
        [](const pdg::ResidencyBoundary& boundary)
        {
            return boundary.kind == pdg::ResidencyBoundaryKind::Upload &&
                   boundary.consumers == std::vector<std::size_t>({1U, 2U});
        });
    ASSERT_NE(branchingUpload, plan.summary().residencyBoundaries.end());

    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    EXPECT_FALSE(rewritten.executable);
    EXPECT_NE(rewritten.reason.find("unsupported branching topology"),
              std::string::npos)
        << rewritten.reason;
}

TEST(ResidentPipeline, LeavesUnsupportedCountSixtyFiveHagNnOnHost)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.hag_nn","count":65,"class":9},
      {"type":"filters.ferry","dimensions":"HeightAboveGround=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_FALSE(plan.stages()[1].native);
    EXPECT_EQ(plan.stages()[1].preferredResidency, pdg::MemoryKind::Host);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.json.find(pdg::HybridHagNnStage), std::string::npos);
}

TEST(ResidentPipeline, MaterializesASharedKnnLofRegion)
{
    // V3: filters.lof declares the same shared-kNN region shape as the V2
    // neighborhood family and rewrites to its own wrapper stage.
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.lof","minpts":10},
      {"type":"filters.ferry","dimensions":"LocalOutlierFactor=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 6U);
    const Json& neighborhood = document.at("pipeline").at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridLofStage);
    EXPECT_EQ(neighborhood.at("minpts"), 10);
    // The wrapper's kNN envelope carries upstream's self-inclusive increment.
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 11U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    const Json& consumer = document.at("pipeline").at(3U);
    EXPECT_EQ(consumer.at("type"), pdg::HybridPointProgramStage);
    EXPECT_EQ(consumer.at("pdg_neighborhood_region_id"),
              neighborhood.at("pdg_region_id"));
    EXPECT_TRUE(consumer.at("pdg_neighborhood_region_last").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedRadiusDensityRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.radialdensity","radius":1.01},
      {"type":"filters.assign",
       "value":"UserData = 1 WHERE RadialDensity >= 0.2"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 6U);
    const Json& neighborhood = document.at("pipeline").at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridRadialDensityStage);
    EXPECT_DOUBLE_EQ(neighborhood.at("pdg_region_radius").get<double>(), 1.01);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 3U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
    EXPECT_FALSE(neighborhood.at("pdg_region_last").get<bool>());
    const Json& consumer = document.at("pipeline").at(3U);
    EXPECT_EQ(consumer.at("type"), pdg::HybridPointProgramStage);
    EXPECT_EQ(consumer.at("pdg_neighborhood_region_id"),
              neighborhood.at("pdg_region_id"));
    EXPECT_TRUE(consumer.at("pdg_neighborhood_region_last").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedStatisticalOutlierKnnRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.outlier","method":"statistical","mean_k":8,
       "multiplier":1.5,"class":18},
      {"type":"filters.nndistance","mode":"kth","k":12},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    ASSERT_EQ(plan.summary().indexBuilds, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U, 2U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 6U);
    const Json& outlier = document.at("pipeline").at(2U);
    const Json& nndistance = document.at("pipeline").at(3U);
    EXPECT_EQ(outlier.at("type"), pdg::HybridOutlierStage);
    EXPECT_EQ(outlier.at("pdg_region_neighbors"), 13U);
    EXPECT_EQ(outlier.at("pdg_region_gather_neighbors"), 13U);
    EXPECT_TRUE(outlier.at("pdg_resident_context").get<bool>());
    EXPECT_FALSE(outlier.at("pdg_region_last").get<bool>());
    EXPECT_EQ(nndistance.at("type"), pdg::HybridNnDistanceStage);
    EXPECT_EQ(nndistance.at("pdg_region_id"), outlier.at("pdg_region_id"));
    EXPECT_EQ(nndistance.at("pdg_region_neighbors"), 13U);
    EXPECT_EQ(nndistance.at("pdg_region_gather_neighbors"), 13U);
    EXPECT_TRUE(nndistance.at("pdg_region_last").get<bool>());
}

TEST(ResidentPipeline, DoesNotRetainKnnGatherAcrossAnInterveningBridge)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.outlier","method":"statistical","mean_k":8},
      {"type":"filters.assign","value":"UserData = Classification"},
      {"type":"filters.nndistance","mode":"avg","k":10},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    EXPECT_EQ(plan.summary().peakDeviceQueryBytesPerPoint, 0U);
    EXPECT_EQ(plan.stages().at(1U).deviceKnnGatherNeighbors, 0U);
    EXPECT_EQ(plan.stages().at(3U).deviceKnnGatherNeighbors, 0U);

    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    const Json& outlier = document.at("pipeline").at(2U);
    const Json& nndistance = document.at("pipeline").at(4U);
    EXPECT_FALSE(outlier.contains("pdg_region_gather_neighbors"));
    EXPECT_FALSE(nndistance.contains("pdg_region_gather_neighbors"));
}

TEST(ResidentPipeline, MaterializesASharedRadiusOutlierRegion)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.outlier","method":"radius","radius":2.0},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 1U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json document = Json::parse(rewritten.json);
    ASSERT_EQ(document.at("pipeline").size(), 5U);
    const Json& outlier = document.at("pipeline").at(2U);
    EXPECT_EQ(outlier.at("type"), pdg::HybridOutlierStage);
    EXPECT_DOUBLE_EQ(outlier.at("pdg_region_radius").get<double>(), 2.0);
    EXPECT_EQ(outlier.at("pdg_region_dimensions"), 3U);
    EXPECT_TRUE(outlier.at("pdg_resident_context").get<bool>());
    EXPECT_TRUE(outlier.at("pdg_region_last").get<bool>());
}

TEST(ResidentPipeline, RewritesTwoNeighborhoodRegionsAroundACoordinateMutator)
{
    // V6: an XYZ-mutating host stage between two neighborhood stages splits
    // the plan into two shared-index regions whose indices cannot be shared;
    // the planner predicts one physical build per region.
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.approximatecoplanar","knn":8},
      {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
      {"type":"filters.reprojection","in_srs":"EPSG:32615",
       "out_srs":"EPSG:32616","error_on_failure":true},
      {"type":"filters.lof","minpts":10},
      {"type":"filters.assign",
       "value":"UserData = 1 WHERE LocalOutlierFactor >= 1.2"},
      {"type":"writers.las","filename":"out.las"}
    ]})";
    const pdg::Plan plan = compile(Pipeline);
    ASSERT_EQ(plan.summary().residentRegions, 2U);
    EXPECT_EQ(plan.summary().indexBuilds, 2U);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    std::vector<std::string> wrappers;
    for (const Json& node : document.at("pipeline"))
        if (node.contains("pdg_region_id"))
            wrappers.push_back(node.at("type").get<std::string>());
    EXPECT_EQ(wrappers, (std::vector<std::string>{
                            std::string(pdg::HybridApproximateCoplanarStage),
                            std::string(pdg::HybridLofStage)}));
    // The coordinate mutator stays a host stage between the two regions.
    bool sawReprojection = false;
    for (const Json& node : document.at("pipeline"))
        sawReprojection =
            sawReprojection ||
            node.at("type").get<std::string>() == "filters.reprojection";
    EXPECT_TRUE(sawReprojection);
}

TEST(ResidentPipeline, LeavesHostDecisionUntouched)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Classification = 7"},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    pdg::PlanPlacementEstimate placement;
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, placement);
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(Json::parse(rewritten.json), Json::parse(Pipeline));
    EXPECT_TRUE(rewritten.selectedStageIds.empty());
}

TEST(ResidentPipeline, MaterializesASharedKnnNeighborClassifierRegion)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.neighborclassifier","k":7},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    const Json& stages =
        document.is_object() ? document.at("pipeline") : document;
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridNeighborClassifierStage);
    EXPECT_EQ(neighborhood.at("k"), 7);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 7U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedRadiusAssignRegion)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.radiusassign","radius":1.5,
       "src_domain":"Classification[1:1]",
       "reference_domain":"Classification[2:2]",
       "is3d":false,"max2d_above":2.0,
       "update_expression":"UserData = 9"},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    const Json& stages =
        document.is_object() ? document.at("pipeline") : document;
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridRadiusAssignStage);
    EXPECT_DOUBLE_EQ(neighborhood.at("pdg_region_radius").get<double>(), 1.5);
    EXPECT_EQ(neighborhood.at("pdg_region_dimensions"), 2U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedKnnOptimalNeighborhoodRegion)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.optimalneighborhood","min_k":10,"max_k":14},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    const Json document = Json::parse(rewritten.json);
    const Json& stages =
        document.is_object() ? document.at("pipeline") : document;
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridOptimalNeighborhoodStage);
    EXPECT_EQ(neighborhood.at("max_k"), 14);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 14U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedKnnEstimateRankRegion)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.estimaterank","knn":8,"thresh":0.01},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json document = Json::parse(rewritten.json);
    const Json& stages =
        document.is_object() ? document.at("pipeline") : document;
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridEstimateRankStage);
    EXPECT_EQ(neighborhood.at("knn"), 8);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 8U);
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
    EXPECT_TRUE(neighborhood.at("pdg_region_last").get<bool>());
}

TEST(ResidentPipeline, MaterializesASharedKnnNormalRegion)
{
    // The catalog port admits filters.normal as a whole-view shared-index
    // producer with the same region shape as the V2 family.
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.normal","knn":8},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    const pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, selectAll(plan));
    ASSERT_TRUE(rewritten.executable) << rewritten.reason;
    EXPECT_EQ(rewritten.selectedStageIds, (std::vector<std::size_t>{1U}));
    const Json document = Json::parse(rewritten.json);
    const Json& stages =
        document.is_object() ? document.at("pipeline") : document;
    const Json& neighborhood = stages.at(2U);
    EXPECT_EQ(neighborhood.at("type"), pdg::HybridNormalStage);
    EXPECT_EQ(neighborhood.at("knn"), 8);
    EXPECT_EQ(neighborhood.at("pdg_region_neighbors"), 9U);
    EXPECT_FALSE(neighborhood.contains("pdg_region_dimensions"));
    EXPECT_TRUE(neighborhood.at("pdg_resident_context").get<bool>());
    EXPECT_TRUE(neighborhood.at("pdg_region_last").get<bool>());
}

TEST(ResidentPipeline, RejectsPartialOrInconsistentSelection)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.assign","value":"Classification = 7"},
      {"type":"filters.ferry","dimensions":"Classification=>UserData"},
      {"type":"writers.las","filename":"out.las"}
    ])";
    const pdg::Plan plan = compile(Pipeline);
    pdg::PlanPlacementEstimate placement = selectAll(plan);
    placement.regions.front().stageIds.pop_back();
    pdg::ResidentPipelineRewrite rewritten =
        pdg::rewriteResidentPlacement(Pipeline, plan, placement);
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "placement selected only part of a resident region");

    placement = selectAll(plan);
    placement.selectedRegionCount = 0U;
    rewritten = pdg::rewriteResidentPlacement(Pipeline, plan, placement);
    EXPECT_FALSE(rewritten.executable);
    EXPECT_EQ(rewritten.reason,
              "placement selected-region count is inconsistent");
}
