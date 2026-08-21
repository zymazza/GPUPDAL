#include <pdg/Hybrid.hpp>

#include <gtest/gtest.h>

#include <cstdlib>

#include <cstdint>
#include <limits>
#include <string>

namespace
{

std::size_t occurrences(const std::string& text, std::string_view needle)
{
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos)
    {
        ++count;
        position += needle.size();
    }
    return count;
}

TEST(HybridPipeline, FusesLinearRegionsAroundHostStages)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"({"pipeline":[
          {"type":"readers.las","filename":"in.las"},
          {"type":"filters.assign","value":"Classification = 2"},
          {"type":"filters.ferry","dimensions":"Classification=>UserData"},
          {"type":"filters.stats","dimensions":"Z"},
          {"type":"filters.assign","value":"PointSourceId = 7"},
          {"type":"writers.las","filename":"out.las"}]})");

    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 2U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridStatsStage), 0U);
    EXPECT_EQ(occurrences(rewritten.json, "\"type\":\"filters.stats\""), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.assign"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.ferry"), 1U);
}

TEST(HybridPipeline, LeavesTaggedAndOptionRichStagesWithUpstream)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","tag":"named","value":"Z = 1"},
          {"type":"filters.assign","value":"Z = 2","where":"X > 0"},
          {"type":"filters.ferry","dimensions":"Z=>Height"},
          "out.las"])");

    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.fusedStages, 1U);
    EXPECT_FALSE(rewritten.linearPipeline);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"tag\":\"named\""), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"where\":\"X > 0\""), 1U);
}

TEST(HybridPipeline, RejectsImplicitMultiReaderGraphsAsLinear)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.las","filename":"first.las"},
          {"type":"readers.las","filename":"second.las"},
          {"type":"filters.assign","value":"Classification = 9"},
          {"type":"writers.las","filename":"out.las"}])");

    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_FALSE(rewritten.linearPipeline);
}

TEST(HybridPipeline, KeepsOrderedPredicateInsideOneFusedRegion)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity * 2"},
          {"type":"filters.expression","expression":"Scratch >= 100"},
          {"type":"filters.assign","value":"Classification = 5"},
          "out.las"])");

    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.expression"), 1U);
}

TEST(HybridPipeline, FusesRangeWithAdjacentPointPrograms)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity * 2"},
          {"type":"filters.range","limits":"Scratch(100:300]"},
          {"type":"filters.ferry","dimensions":"Classification=>UserData"},
          "out.las"])");

    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.range"), 1U);
}

TEST(HybridPipeline, FusesOnlySingleBoundsCropEnvelope)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"json(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity * 2"},
          {"type":"filters.crop","bounds":"([0,10],[20,30])",
           "outside":true},
          {"type":"filters.ferry","dimensions":"Classification=>UserData"},
          "out.las"])json");

    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.crop"), 1U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"json(["in.las",
              {"type":"filters.crop",
               "bounds":["([0,1],[0,1])","([2,3],[2,3])"]},
              {"type":"filters.crop","bounds":"([0,1],[0,1])",
               "a_srs":"EPSG:4326"},
              "out.las"])json");
    EXPECT_EQ(unsupported.pointProgramRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridPointProgramStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.crop"), 2U);
}

TEST(HybridPipeline, FusesOrdinalStagesAndSplitsAfterValuePredicates)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Classification = 7"},
          {"type":"filters.head","count":100},
          {"type":"filters.expression","expression":"X > 0"},
          {"type":"filters.decimation","step":2.5},
          {"type":"filters.tail","count":10,"invert":true},
          "out.las"])");

    EXPECT_EQ(rewritten.fusedStages, 5U);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 2U);
    EXPECT_FALSE(rewritten.standardModeRequiresPointCountValidation);

    const pdg::HybridPipelineRewrite guarded = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.decimation","offset":4},
             "out.las"])");
    EXPECT_TRUE(guarded.standardModeRequiresPointCountValidation);
}

TEST(HybridPipeline, RewritesExactLocateBetweenPointProgramRegions)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity * 2"},
          {"type":"filters.locate","dimension":"Scratch","minmax":"MIN"},
          {"type":"filters.assign","value":"Classification = 8"},
          "out.las"])");

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridLocateStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.locate"), 0U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.locate","dimension":"Z",
                 "where":"Classification == 2"}, "out.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridLocateStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.locate"), 1U);
}

TEST(HybridPipeline, FusesTransformationWithAdjacentPointOperations)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"X = X + 0.25"},
          {"type":"filters.transformation",
           "matrix":"1 0.5 0 4 0 1 0.25 5 0.125 0 1 6 0 0 0 1"},
          {"type":"filters.ferry","dimensions":"Z=>HeightAboveGround"},
          "out.las"])");

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.replacementRegions, 1U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.transformation"), 1U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
              {"type":"filters.transformation",
               "matrix":"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
               "invert":true},
              {"type":"filters.transformation",
               "matrix":"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
               "override_srs":"EPSG:3857"},
              "out.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridPointProgramStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.transformation"), 2U);
}

TEST(HybridPipeline, RewritesRobustSelectionsBetweenPointRegions)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity * 2"},
          {"type":"filters.mad","dimension":"Scratch","k":2.5,
           "mad_multiplier":1.4862},
          {"type":"filters.assign","value":"Classification = 12"},
          {"type":"filters.iqr","dimension":"Z"},
          "out.las"])");

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 4U);
    EXPECT_EQ(rewritten.fusedStages, 4U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridRobustStage), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"method\":\"mad\""), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"method\":\"iqr\""), 1U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
              {"type":"filters.iqr","dimension":"Z","k":"1.5"},
              {"type":"filters.mad","dimension":"Z",
               "where":"Z > 0"}, "out.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridRobustStage), 0U);
}

TEST(HybridPipeline, RewritesExactOrderingBetweenPointRegions)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity"},
          {"type":"filters.sort","dimensions":["Classification","Scratch"],
           "order":"DESC","algorithm":"STABLE"},
          {"type":"filters.assign","value":"UserData = Classification"},
          "out.las"])");

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridOrderStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.sort"), 0U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
              {"type":"filters.sort","dimension":"Z",
               "where":"Classification == 2"},
              {"type":"filters.sort","dimension":"Z","order":7},
              "out.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridOrderStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.sort"), 2U);
}

TEST(HybridPipeline, KeepsNativeRegionsOpenAcrossAuditedRandomizeBridge)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.randomize","seed":7},
          {"type":"filters.assign","value":"UserData = Classification"},
          "out.las"])");

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_FALSE(rewritten.hasUnstableInputOrderRegion);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.replacementRegions, 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.randomize"), 1U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridPointProgramStage), 1U);

    const pdg::HybridPipelineRewrite where = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.randomize","seed":7,"where":"Z > 0"},
          {"type":"filters.assign","value":"UserData = Classification"},
          "out.las"])");
    EXPECT_TRUE(where.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite multiView = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.divider","count":2},
          {"type":"filters.randomize","seed":7},
          {"type":"filters.assign","value":"UserData = Classification"},
          "out#.las"])");
    EXPECT_TRUE(multiView.hasUnstableInputOrderRegion);
}

TEST(HybridPipeline, GatesExactAdjacentDuplicateLabeling)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.label_duplicates","dimensions":["X","GpsTime"]},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridLabelDuplicatesStage),
              0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridLabelDuplicatesStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite empty = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.label_duplicates"}, "out.las"])", false,
        true);
    EXPECT_EQ(occurrences(empty.json, pdg::HybridLabelDuplicatesStage), 1U);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":7}, "out.las"])",
            R"(["in.las", {"type":"filters.label_duplicates","dimensions":"X","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridLabelDuplicatesStage),
                  0U);
    }
}

TEST(HybridPipeline, AutomaticallySelectsMeasuredLabelNnDistanceComposition)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"input.las"},
      {"type":"filters.label_duplicates","dimensions":"Classification"},
      {"type":"filters.nndistance","k":10},
      {"type":"filters.assign","value":"UserData = Duplicate"},
      {"type":"writers.las","filename":"output.las"}]})";
    EXPECT_TRUE(pdg::automaticLabelNnDistanceHybridCandidate(Pipeline));

    const pdg::HybridPipelineRewrite unknown =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_TRUE(unknown.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(unknown.automaticLabelNnDistanceCuda);
    EXPECT_EQ(unknown.automaticPointCountFilename, "input.las");
    EXPECT_EQ(unknown.replacementRegions, 0U);

    const pdg::HybridPipelineRewrite selected = pdg::rewriteHybridPipeline(
        Pipeline, false, false, 250'000U, false, true);
    EXPECT_TRUE(selected.hasPointCountDependentCudaCandidate);
    EXPECT_TRUE(selected.automaticLabelNnDistanceCuda);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridLabelDuplicatesStage), 1U);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridNnDistanceStage), 1U);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(selected.json, "\"pdg_auto_cuda\":true"), 3U);

    EXPECT_FALSE(pdg::rewriteHybridPipeline(Pipeline, false, false, 249'999U,
                                            false, true)
                     .automaticLabelNnDistanceCuda);
    EXPECT_FALSE(pdg::rewriteHybridPipeline(Pipeline, false, false, 16'000'001U,
                                            false, true)
                     .automaticLabelNnDistanceCuda);
    EXPECT_FALSE(pdg::rewriteHybridPipeline(Pipeline, false, false, 250'000U,
                                            false, false)
                     .automaticLabelNnDistanceCuda);

    const pdg::HybridPipelineRewrite experimental = pdg::rewriteHybridPipeline(
        Pipeline, false, true, 250'000U, false, true);
    EXPECT_FALSE(experimental.automaticLabelNnDistanceCuda);
    EXPECT_EQ(occurrences(experimental.json, "\"pdg_auto_cuda\":true"), 0U);

    for (
        std::
            string_view
                rejected : {
                    R"([{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}])",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}],"metadata":"out.json"})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las","count":250000},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":["Classification"]},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","mode":"avg","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"filters.nndistance","k":10},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData=Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las","extra_dims":"all"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.laz"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.las"}]})",
                    R"({"pipeline":[{"type":"readers.las","filename":"input.las"},{"type":"filters.label_duplicates","dimensions":"Classification"},{"type":"filters.nndistance","k":10},{"type":"filters.assign","value":"UserData = Duplicate"},{"type":"writers.las","filename":"output.laz"}]})",
                })
    {
        const pdg::HybridPipelineRewrite fallback = pdg::rewriteHybridPipeline(
            rejected, false, false, 250'000U, false, true);
        EXPECT_FALSE(pdg::automaticLabelNnDistanceHybridCandidate(rejected));
        EXPECT_FALSE(fallback.hasPointCountDependentCudaCandidate) << rejected;
        EXPECT_FALSE(fallback.automaticLabelNnDistanceCuda) << rejected;
    }
}

TEST(HybridPipeline, GatesBoundedExactSmrfGridExecution)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.smrf","cell":1.0,"window":7.0,"cut":5.0,
       "returns":[],"ground_class":9,"other_class":9,
       "only_ground":true},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridSmrfStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridSmrfStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.smrf","returns":""}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","window":65}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","where":"Z > 0"}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","ignore":"Classification[7:7]"}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","classbits":"synthetic"}, "out.las"])",
            R"(["in.las", {"type":"filters.smrf","dir":"."}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridSmrfStage), 0U)
            << unsupported;
    }
}

TEST(HybridPipeline, GatesBoundedExactPmfGridExecution)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.pmf","cell_size":1.0,"exponential":false,
       "initial_distance":0.2,"max_distance":1.0,
       "max_window_size":9.0,"slope":0.4,"returns":[],
       "ground_class":9,"other_class":9,"only_ground":true},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridPmfStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridPmfStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.pmf","returns":""}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","cell_size":0}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","max_window_size":1000}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","where":"Z > 0"}, "out.las"])",
            R"(["in.las", {"type":"filters.pmf","ignore":"Classification[7:7]"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridPmfStage), 0U)
            << unsupported;
    }
}

TEST(HybridPipeline, GatesBoundedExactCsfGridExecution)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.csf","smooth":false,"step":1.0,
       "threshold":0.5,"hdiff":0.3,"resolution":1.0,
       "rigidness":3,"iterations":3,"returns":[],
       "ground_class":9,"other_class":9,"only_ground":true},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridCsfStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridCsfStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.csf"}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"returns":"bogus"}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"resolution":0}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"iterations":65}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"debug":true}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"where":"Z > 0"}, "out.las"])",
            R"(["in.las", {"type":"filters.csf","smooth":false,"ignore":"Classification[7:7]"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridCsfStage), 0U)
            << unsupported;
    }
}

TEST(HybridPipeline, GatesBoundedExactElmGridExecution)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.elm","cell":1.25,"class":18,
       "threshold":-1.0},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridElmStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridElmStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (std::string_view unsupported : {
             R"(["in.las", {"type":"filters.elm","cell":0}, "out.las"])",
             R"(["in.las", {"type":"filters.elm","class":256}, "out.las"])",
             R"(["in.las", {"type":"filters.elm","where":"Z > 0"}, "out.las"])",
             R"(["in.las", {"type":"filters.elm","extra":1}, "out.las"])",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridElmStage), 0U)
            << unsupported;
    }
}

TEST(HybridPipeline, GatesStandaloneSkewnessBalancingExecution)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.skewnessbalancing","ground_class":3,
       "other_class":9,"only_ground":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridSkewnessStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridSkewnessStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite equalOnlyGround =
        pdg::rewriteHybridPipeline(
            R"(["in.las", {"type":"filters.skewnessbalancing",
                  "ground_class":4,"other_class":4,"only_ground":true},
                  "out.las"])",
            false, true);
    EXPECT_EQ(occurrences(equalOnlyGround.json, pdg::HybridSkewnessStage), 1U);

    const pdg::HybridPipelineRewrite multipleViews = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.divider","count":2},
                  {"type":"filters.skewnessbalancing"}, "out#.las"])",
        false, true);
    EXPECT_EQ(occurrences(multipleViews.json, pdg::HybridSkewnessStage), 0U);
    EXPECT_EQ(occurrences(multipleViews.json, "filters.skewnessbalancing"), 1U);

    const pdg::HybridPipelineRewrite unprovenReader =
        pdg::rewriteHybridPipeline(
            R"([{"type":"readers.faux","count":5},
                  {"type":"filters.skewnessbalancing"}, "out.las"])",
            false, true);
    EXPECT_EQ(occurrences(unprovenReader.json, pdg::HybridSkewnessStage), 0U);
    EXPECT_EQ(occurrences(unprovenReader.json, "filters.skewnessbalancing"),
              1U);

    for (std::string_view unsupported : {
             R"(["in.las", {"type":"filters.skewnessbalancing",
                  "ground_class":256}, "out.las"])",
             R"(["in.las", {"type":"filters.skewnessbalancing",
                  "ground_class":4,"other_class":4}, "out.las"])",
             R"(["in.las", {"type":"filters.skewnessbalancing",
                  "where":"Z > 0"}, "out.las"])",
             R"(["in.las", {"type":"filters.skewnessbalancing",
                  "extra":1}, "out.las"])",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridSkewnessStage), 0U)
            << unsupported;
    }
}

TEST(HybridPipeline, RewritesExactStatsWithoutChangingViewTopology)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity"},
          {"type":"filters.stats","dimensions":["Z","Classification"],
           "count":"Classification","advanced":true},
          {"type":"filters.assign","value":"UserData = Classification"},
          "out.las"])",
        false, true);

    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_FALSE(rewritten.hasUnstableInputOrderRegion);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.pdg_stats"), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"type\":\"filters.stats\""), 0U);

    const pdg::HybridPipelineRewrite multiView = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.divider","count":2},
          {"type":"filters.stats","dimensions":"Z"},
          {"type":"filters.merge"}, "out.las"])",
        false, true);
    EXPECT_FALSE(multiView.hasUnstableInputOrderRegion);
    EXPECT_EQ(occurrences(multiView.json, "filters.pdg_stats"), 1U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.stats","where":"Z > 0"},
          {"type":"filters.stats","advanced":"true"}, "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(unsupported.json, "filters.pdg_stats"), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "\"type\":\"filters.stats\""), 2U);
}

TEST(HybridPipeline, LeavesUnqualifiedStatsOnPinnedUpstreamByDefault)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.stats","dimensions":"X,Y,Z"},
          "out.las"])");

    EXPECT_EQ(rewritten.replacementRegions, 0U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.pdg_stats"), 0U);
    EXPECT_EQ(occurrences(rewritten.json, "\"type\":\"filters.stats\""), 1U);
}

TEST(HybridPipeline, GatesSharedIndexOutlierReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.outlier","method":"radius","radius":2.5,
       "min_k":4,"class":18}, "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridOutlierStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridOutlierStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.outlier","radius":"2"}, "out.las"])",
            R"(["in.las", {"type":"filters.outlier","where":"Z > 0"}, "out.las"])",
            R"(["in.las", {"type":"filters.outlier","class":300}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridOutlierStage), 0U);
    }
}

TEST(HybridPipeline, GatesSharedIndexRadialDensityReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.radialdensity","radius":2.5}, "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridRadialDensityStage),
              0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridRadialDensityStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.radialdensity","radius":"2"}, "out.las"])",
            R"(["in.las", {"type":"filters.radialdensity","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridRadialDensityStage),
                  0U);
    }
}

TEST(HybridPipeline, GatesSharedIndexNormalReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.normal","knn":12,"always_up":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridNormalStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridNormalStage), 1U);
    EXPECT_EQ(enabled.json.find("pdg_region_dimensions"), std::string::npos);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.normal","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.normal","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.normal","refine":true}, "out.las"])",
            R"json(["in.las", {"type":"filters.normal","viewpoint":"POINT Z (0 0 1)"}, "out.las"])json",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridNormalStage), 0U);
    }
}

TEST(HybridPipeline, GatesSharedIndexEigenvaluesReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.eigenvalues","knn":12,"normalize":true,
       "stride":1,"min_k":5},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridEigenvaluesStage),
              0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridEigenvaluesStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.eigenvalues","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.eigenvalues","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.eigenvalues","stride":2}, "out.las"])",
            R"(["in.las", {"type":"filters.eigenvalues","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridEigenvaluesStage), 0U);
    }
}

TEST(HybridPipeline, GatesSharedIndexApproximateCoplanarReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.approximatecoplanar","knn":12,
       "thresh1":30.5,"thresh2":4.25},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(
        occurrences(defaultRewrite.json, pdg::HybridApproximateCoplanarStage),
        0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridApproximateCoplanarStage),
              1U);
    EXPECT_EQ(occurrences(enabled.json, "\"pdg_region_neighbors\":12"), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.approximatecoplanar","knn":2}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","knn":65}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","thresh1":"25"}, "out.las"])",
            R"(["in.las", {"type":"filters.approximatecoplanar","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(
            occurrences(fallback.json, pdg::HybridApproximateCoplanarStage),
            0U);
    }
}

TEST(HybridPipeline, SelectsApproximateCoplanarOnlyForMeasuredExactCardinality)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.laz"},
      {"type":"filters.approximatecoplanar",
       "thresh1":30.5,"thresh2":4.25},
      {"type":"writers.las","filename":"out.las",
       "extra_dims":"all"}])";

#if PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR
    constexpr bool QualificationBuild = true;
#else
    constexpr bool QualificationBuild = false;
#endif

    const pdg::HybridPipelineRewrite unknown =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(unknown.hasPointCountDependentCudaCandidate, QualificationBuild);
    EXPECT_FALSE(unknown.automaticApproximateCoplanarCuda);
    EXPECT_EQ(unknown.automaticPointCountFilename, "in.laz");
    EXPECT_EQ(unknown.automaticPointCountLimit,
              (std::numeric_limits<std::uint64_t>::max)());
    EXPECT_EQ(occurrences(unknown.json, pdg::HybridApproximateCoplanarStage),
              0U);

    const pdg::HybridPipelineRewrite below =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 262'143);
    EXPECT_EQ(below.hasPointCountDependentCudaCandidate, QualificationBuild);
    EXPECT_FALSE(below.automaticApproximateCoplanarCuda);
    EXPECT_EQ(occurrences(below.json, pdg::HybridApproximateCoplanarStage), 0U);

    const pdg::HybridPipelineRewrite selected =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 262'144);
    EXPECT_EQ(selected.automaticApproximateCoplanarCuda, QualificationBuild);
    EXPECT_EQ(selected.replacementRegions, QualificationBuild ? 1U : 0U);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridApproximateCoplanarStage),
              QualificationBuild ? 1U : 0U);
    EXPECT_EQ(occurrences(selected.json, "\"pdg_auto_cuda\":true"),
              QualificationBuild ? 1U : 0U);
    if (QualificationBuild)
    {
        EXPECT_EQ(occurrences(selected.json, "\"pdg_region_id\":2"), 1U);
        EXPECT_EQ(occurrences(selected.json, "\"pdg_region_neighbors\":8"), 1U);
        EXPECT_EQ(occurrences(selected.json, "\"pdg_region_reuse\":false"), 1U);
        EXPECT_EQ(occurrences(selected.json, "\"pdg_region_last\":true"), 1U);
        EXPECT_EQ(occurrences(selected.json, "\"thresh1\":30.5"), 1U);
        EXPECT_EQ(occurrences(selected.json, "\"thresh2\":4.25"), 1U);
    }

    const pdg::HybridPipelineRewrite experimental =
        pdg::rewriteHybridPipeline(Pipeline, false, true, 262'144);
    EXPECT_FALSE(experimental.automaticApproximateCoplanarCuda);
    EXPECT_EQ(occurrences(experimental.json, "\"pdg_auto_cuda\":true"), 0U);

    const pdg::HybridPipelineRewrite explicitKnn = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.las","filename":"explicit.las"},
              {"type":"filters.approximatecoplanar","knn":8},
              {"type":"writers.las","filename":"explicit-out.las",
               "extra_dims":"all"}])",
        false, false, 262'144);
    EXPECT_EQ(explicitKnn.automaticApproximateCoplanarCuda, QualificationBuild);
    EXPECT_EQ(occurrences(explicitKnn.json, "\"pdg_region_neighbors\":8"),
              QualificationBuild ? 1U : 0U);

    const pdg::HybridPipelineRewrite numericThresholds =
        pdg::rewriteHybridPipeline(
            R"([{"type":"readers.las","filename":"thresholds.las"},
              {"type":"filters.approximatecoplanar","knn":8,
               "thresh1":-0.5,"thresh2":0.0},
              {"type":"writers.las","filename":"thresholds-out.las",
               "extra_dims":"all"}])",
            false, false, 262'144);
    EXPECT_EQ(numericThresholds.automaticApproximateCoplanarCuda,
              QualificationBuild);
    EXPECT_EQ(occurrences(numericThresholds.json, "\"thresh1\":-0.5"), 1U);
    EXPECT_EQ(occurrences(numericThresholds.json, "\"thresh2\":0.0"), 1U);

    const pdg::HybridPipelineRewrite cardinalityPreserving =
        pdg::rewriteHybridPipeline(
            R"([{"type":"readers.las","filename":"bridge.las"},
              {"type":"filters.ferry","dimensions":"Intensity=>Scratch"},
              {"type":"filters.approximatecoplanar"},
              {"type":"writers.las","filename":"bridge-out.las",
               "extra_dims":"all"}])",
            false, false, 262'144);
    EXPECT_FALSE(cardinalityPreserving.automaticApproximateCoplanarCuda);
    EXPECT_EQ(occurrences(cardinalityPreserving.json,
                          pdg::HybridApproximateCoplanarStage),
              0U);

    for (std::string_view rejected : {
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":7},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":9},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.text","filename":"in.csv"},
               {"type":"filters.approximatecoplanar","knn":8},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.expression","expression":"Z > 0"},
               {"type":"filters.approximatecoplanar","knn":8},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"first.las"},
               {"type":"readers.las","filename":"second.las"},
               {"type":"filters.approximatecoplanar","knn":8},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8,
                "thresh1":"25"},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8,
                "thresh2":null},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8,
                "where":"Classification == 2"},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8,
                "tag":"approx"},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8,
                "inputs":"reader"},
               {"type":"writers.las","filename":"out.las",
                "extra_dims":"all"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8},
               {"type":"writers.las","filename":"out.las"}])",
             R"([{"type":"readers.las","filename":"in.las"},
               {"type":"filters.approximatecoplanar","knn":8},
               {"type":"writers.las","filename":"out.laz",
                "extra_dims":"all"}])",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(rejected, false, false, 262'144);
        EXPECT_FALSE(fallback.automaticApproximateCoplanarCuda) << rejected;
        EXPECT_EQ(
            occurrences(fallback.json, pdg::HybridApproximateCoplanarStage), 0U)
            << rejected;
    }
}

TEST(HybridPipeline, ProductionBuildDisablesUnqualifiedAutomaticProfile)
{
#if PDG_QUALIFY_AUTOMATIC_APPROXIMATECOPLANAR
    SUCCEED() << "qualification artifacts exercise the provisional profile";
#else
    EXPECT_FALSE(pdg::automaticApproximateCoplanarCudaDeviceQualified());
    const pdg::HybridPipelineRewrite rewrite = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.las","filename":"in.las"},
              {"type":"filters.approximatecoplanar"},
              {"type":"writers.las","filename":"out.las",
               "extra_dims":"all"}])",
        false, false, 262'144);
    EXPECT_FALSE(rewrite.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(rewrite.automaticApproximateCoplanarCuda);
    EXPECT_EQ(occurrences(rewrite.json, pdg::HybridApproximateCoplanarStage),
              0U);
#endif
}

TEST(HybridPipeline, SelectsR4OutlierOnlyForMeasuredReferenceFacts)
{
    constexpr std::string_view Pipeline = R"({"pipeline":[
      {"type":"readers.las","filename":"in.laz"},
      {"type":"filters.outlier","method":"statistical","mean_k":8,
       "multiplier":2.0},
      {"type":"filters.range","limits":"Classification![7:7]"},
      {"type":"filters.sample","radius":1.0},
      {"type":"writers.las","filename":"out.laz",
       "compression":"true"}]})";

    const pdg::HybridPipelineRewrite unknown =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_TRUE(unknown.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(unknown.automaticR4OutlierCuda);
    EXPECT_EQ(unknown.automaticPointCountFilename, "in.laz");
    EXPECT_EQ(unknown.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unknown.json, pdg::HybridOutlierStage), 0U);
    EXPECT_EQ(occurrences(unknown.json, pdg::HybridPointProgramStage), 0U);

    const pdg::HybridPipelineRewrite wrongCount =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 999'999, true);
    EXPECT_TRUE(wrongCount.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(wrongCount.automaticR4OutlierCuda);
    EXPECT_EQ(wrongCount.replacementRegions, 0U);
    EXPECT_EQ(occurrences(wrongCount.json, pdg::HybridOutlierStage), 0U);
    EXPECT_EQ(occurrences(wrongCount.json, pdg::HybridPointProgramStage), 0U);

    const pdg::HybridPipelineRewrite wrongLayout =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 1'000'000, false);
    EXPECT_TRUE(wrongLayout.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(wrongLayout.automaticR4OutlierCuda);
    EXPECT_EQ(wrongLayout.replacementRegions, 0U);
    EXPECT_EQ(occurrences(wrongLayout.json, pdg::HybridOutlierStage), 0U);
    EXPECT_EQ(occurrences(wrongLayout.json, pdg::HybridPointProgramStage), 0U);

    // D0272: the literal 1M layout no longer selects the route by default;
    // the exact host path is faster. The explicit experimental opt-in keeps
    // the route reachable for its differential lanes.
    const pdg::HybridPipelineRewrite retired =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 1'000'000, true);
    EXPECT_TRUE(retired.hasPointCountDependentCudaCandidate);
    EXPECT_FALSE(retired.automaticR4OutlierCuda);
    EXPECT_EQ(retired.replacementRegions, 0U);
    EXPECT_EQ(occurrences(retired.json, pdg::HybridOutlierStage), 0U);
    EXPECT_EQ(occurrences(retired.json, pdg::HybridPointProgramStage), 0U);

    ::setenv("PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA", "1", 1);
    const pdg::HybridPipelineRewrite selected =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 1'000'000, true);
    ::unsetenv("PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA");
    EXPECT_TRUE(selected.automaticR4OutlierCuda);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridOutlierStage), 1U);
    EXPECT_EQ(occurrences(selected.json, "\"pdg_auto_cuda\":true"), 1U);
    EXPECT_EQ(occurrences(selected.json, pdg::HybridPointProgramStage), 1U);
    EXPECT_EQ(occurrences(selected.json, "filters.sample"), 1U);

    const pdg::HybridPipelineRewrite experimental =
        pdg::rewriteHybridPipeline(Pipeline, false, true, 1'000'000, true);
    EXPECT_FALSE(experimental.automaticR4OutlierCuda);
    EXPECT_EQ(occurrences(experimental.json, pdg::HybridOutlierStage), 1U);
    EXPECT_EQ(occurrences(experimental.json, "\"pdg_auto_cuda\":true"), 0U);

    for (std::string_view rejected : {
             R"([{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}])",
             R"({"pipeline":[{"type":"readers.las","filename":"in.las"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.LAZ"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.LAZ",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":9,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![8:8]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":2.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":"true"}]})",
             R"({"pipeline":[{"type":"readers.las","filename":"in.laz"},
               {"type":"filters.outlier","method":"statistical",
                "mean_k":8,"multiplier":2.0},
               {"type":"filters.range","limits":"Classification![7:7]"},
               {"type":"filters.sample","radius":1.0},
               {"type":"writers.las","filename":"out.laz",
                "compression":true}]})",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(rejected, false, false, 1'000'000, true);
        EXPECT_FALSE(fallback.hasPointCountDependentCudaCandidate) << rejected;
        EXPECT_FALSE(fallback.automaticR4OutlierCuda) << rejected;
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridOutlierStage), 0U)
            << rejected;
    }
}

TEST(HybridPipeline, GatesSharedIndexNnDistanceReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.nndistance","mode":"avg","k":12},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridNnDistanceStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridNnDistanceStage), 1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.nndistance","k":0}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","k":64}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","mode":"mean"}, "out.las"])",
            R"(["in.las", {"type":"filters.nndistance","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridNnDistanceStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountOneHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":1,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountTwoHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":2,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":2"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountThreeHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":3,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":3"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountFourHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":4,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":4"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountFiveHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":5,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":5"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])", false,
        true);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridHagNnStage), 0U);
}

TEST(HybridPipeline, GatesCountSixHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":6,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":6"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountSevenHagNnReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_nn","count":7,"class":9,
       "max_distance":17.5,"allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagNnStage), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagNnStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":7"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_nn","count":0}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","count":65}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_nn","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagNnStage), 0U);
    }
}

TEST(HybridPipeline, GatesCountThreeHagDelaunayReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.hag_delaunay","count":3,"class":9,
       "allow_extrapolation":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, pdg::HybridHagDelaunayStage),
              0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridHagDelaunayStage), 1U);
    EXPECT_NE(enabled.json.find("\"pdg_region_neighbors\":3"),
              std::string::npos);
    EXPECT_NE(enabled.json.find("\"pdg_region_dimensions\":2"),
              std::string::npos);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.hag_delaunay","class":9}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":10,"class":9}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":4}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","count":2}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","class":256}, "out.las"])",
            R"(["in.las", {"type":"filters.hag_delaunay","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridHagDelaunayStage), 0U);
    }
}

TEST(HybridPipeline, GatesSharedIndexCovarianceFeaturesReplacement)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.covariancefeatures","knn":12,"threads":1,
       "feature_set":"dimensionality,omnivariance,anisotropy",
       "stride":1,"min_k":5,"mode":"normalized","optimized":false},
      "out.las"] )";
    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_EQ(defaultRewrite.replacementRegions, 0U);
    EXPECT_EQ(
        occurrences(defaultRewrite.json, pdg::HybridCovarianceFeaturesStage),
        0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_EQ(enabled.replacementRegions, 1U);
    EXPECT_EQ(occurrences(enabled.json, pdg::HybridCovarianceFeaturesStage),
              1U);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);

    for (
        std::string_view unsupported : {
            R"(["in.las", {"type":"filters.covariancefeatures","knn":64}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","radius":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","stride":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","threads":2}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","optimized":true}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","feature_set":"all"}, "out.las"])",
            R"(["in.las", {"type":"filters.covariancefeatures","where":"Z > 0"}, "out.las"])",
        })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(unsupported, false, true);
        EXPECT_EQ(
            occurrences(fallback.json, pdg::HybridCovarianceFeaturesStage), 0U);
    }
}

TEST(HybridPipeline, MarksOneResidentRegionAcrossNeighborhoodAndPointBridges)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.eigenvalues","knn":12},
          {"type":"filters.covariancefeatures","knn":8,
           "feature_set":"dimensionality"},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(rewritten.fusedStages, 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_neighbors\":13"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_last\":true"), 1U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridNormalStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridEigenvaluesStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridCovarianceFeaturesStage),
              1U);

    const pdg::HybridPipelineRewrite bridged = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.assign","value":"Classification = X"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(bridged.json, "\"pdg_region_id\":2"), 2U);
    EXPECT_EQ(occurrences(bridged.json, "\"pdg_region_reuse\":true"), 1U);
    EXPECT_EQ(occurrences(bridged.json, "\"pdg_region_last\":true"), 1U);
    EXPECT_EQ(occurrences(bridged.json, "\"pdg_neighborhood_region_id\":2"),
              1U);
    EXPECT_EQ(
        occurrences(bridged.json, "\"pdg_neighborhood_region_last\":false"),
        1U);
    EXPECT_EQ(occurrences(bridged.json, "\"pdg_region_id\":4"), 0U);
}

TEST(HybridPipeline, ExtendsNeighborhoodColumnsIntoAdjacentPointProgram)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.covariancefeatures","knn":12,
           "feature_set":"dimensionality"},
          {"type":"filters.assign",
           "value":"Classification = Linearity * 10"},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 2U);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_last\":false"), 1U);

    const pdg::HybridPipelineRewrite predicate = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.covariancefeatures","knn":12,
           "feature_set":"dimensionality"},
          {"type":"filters.range","limits":"Linearity[0.5:]"},
          "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(predicate.json, "pdg_neighborhood_region_id"), 0U);
    EXPECT_EQ(occurrences(predicate.json, "\"pdg_region_last\":true"), 1U);
}

TEST(HybridPipeline, SharesNnDistanceAndRetainsItsColumnForPointPrograms)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.nndistance","mode":"avg","k":12},
          {"type":"filters.ferry","dimensions":"NNDistance=>GpsTime"},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_neighbors\":13"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_last\":false"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              1U);
    EXPECT_EQ(
        occurrences(rewritten.json, "\"pdg_neighborhood_region_last\":true"),
        1U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridNnDistanceStage), 1U);
}

TEST(HybridPipeline, ExtendsResidentNeighborhoodAcrossMultipleBridges)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.ferry","dimensions":"X=>GpsTime"},
          {"type":"filters.nndistance","mode":"kth","k":12},
          {"type":"filters.assign","value":"Classification = NNDistance * 10"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 5U);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_neighbors\":13"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              2U);
    EXPECT_EQ(
        occurrences(rewritten.json, "\"pdg_neighborhood_region_last\":false"),
        2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_last\":true"), 1U);
}

TEST(HybridPipeline, ReusesDirectCoplanarNeighborCountAndUnsignedByteColumn)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":7},
          {"type":"filters.approximatecoplanar","knn":8},
          {"type":"filters.ferry","dimensions":"Coplanar=>UserData"},
          {"type":"filters.eigenvalues","knn":7},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 4U);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_neighbors\":8"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridApproximateCoplanarStage),
              1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              1U);
}

TEST(HybridPipeline, MarksOnlyLasExtraDimsAllAsLayoutOrderObservable)
{
    const pdg::HybridPipelineRewrite ordinary = pdg::rewriteHybridPipeline(
        R"(["in.ply",
          {"type":"filters.assign","value":"Scratch = X"},
          {"type":"writers.las","filename":"out.las"}])",
        false, true);
    EXPECT_FALSE(ordinary.preparedLayoutOrderObservable);

    const pdg::HybridPipelineRewrite explicitDimension =
        pdg::rewriteHybridPipeline(
            R"(["in.ply",
          {"type":"filters.assign","value":"Scratch = X"},
          {"type":"writers.las","filename":"out.las",
           "extra_dims":"Scratch=double"}])",
            false, true);
    EXPECT_FALSE(explicitDimension.preparedLayoutOrderObservable);

    const pdg::HybridPipelineRewrite all = pdg::rewriteHybridPipeline(
        R"(["in.ply",
          {"type":"filters.assign","value":"Scratch = X"},
          {"type":"writers.las","filename":"out.las",
           "extra_dims":[" all "]}])",
        false, true);
    EXPECT_TRUE(all.preparedLayoutOrderObservable);
}

TEST(HybridPipeline, PlansVaryingNeighborCachesAcrossResidentBridges)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":5},
          {"type":"filters.ferry","dimensions":"NormalZ=>GpsTime"},
          {"type":"filters.eigenvalues","knn":11},
          {"type":"filters.ferry","dimensions":"Eigenvalue2=>GpsTime"},
          {"type":"filters.covariancefeatures","knn":5,
           "feature_set":"dimensionality","mode":"raw"},
          "out.las"])",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 5U);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_neighbors\":12"), 3U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              2U);
    EXPECT_EQ(
        occurrences(rewritten.json, "\"pdg_neighborhood_region_last\":false"),
        2U);
}

TEST(HybridPipeline, RetainsRuntimeCheckedUnsupportedBridgeInResidentPlan)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"json(["in.las",
          {"type":"filters.normal","knn":5},
          {"type":"filters.assign","value":"GpsTime = sqrt(Intensity)"},
          {"type":"filters.eigenvalues","knn":5},
          "out.las"])json",
        false, true);

    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_id\":2"), 2U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_region_reuse\":true"), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "\"pdg_neighborhood_region_id\":2"),
              1U);
    EXPECT_EQ(
        occurrences(rewritten.json, "\"pdg_neighborhood_region_last\":false"),
        1U);
}

TEST(HybridPipeline, ClosesResidentRegionAtCoordinateOrPredicateBridge)
{
    const pdg::HybridPipelineRewrite coordinate = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.assign","value":"X = X + 1"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(coordinate.json, "pdg_neighborhood_region_id"), 0U);
    EXPECT_EQ(occurrences(coordinate.json, "\"pdg_region_reuse\":true"), 0U);

    const pdg::HybridPipelineRewrite ferryCoordinate =
        pdg::rewriteHybridPipeline(
            R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.ferry","dimensions":"Intensity=>Z"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
            false, true);
    EXPECT_EQ(occurrences(ferryCoordinate.json, "pdg_neighborhood_region_id"),
              0U);
    EXPECT_EQ(occurrences(ferryCoordinate.json, "\"pdg_region_reuse\":true"),
              0U);

    const pdg::HybridPipelineRewrite ambiguous = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.assign","value":"Classification == X"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(ambiguous.json, "pdg_neighborhood_region_id"), 0U);
    EXPECT_EQ(occurrences(ambiguous.json, "\"pdg_region_reuse\":true"), 0U);

    const pdg::HybridPipelineRewrite predicate = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.normal","knn":8},
          {"type":"filters.range","limits":"Classification[2:2]"},
          {"type":"filters.eigenvalues","knn":8},
          "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(predicate.json, "pdg_neighborhood_region_id"), 0U);
    EXPECT_EQ(occurrences(predicate.json, "\"pdg_region_reuse\":true"), 0U);
}

TEST(HybridPipeline, GatesMetadataReplacementsAndPreservesHostBridges)
{
    constexpr std::string_view Pipeline = R"(["in.las",
      {"type":"filters.assign","value":"Scratch = Intensity"},
      {"type":"filters.info","point":"0,3"},
      {"type":"filters.expressionstats","dimension":"Classification",
       "expressions":["Classification == 2","Scratch < 100"]},
      {"type":"filters.stats","dimensions":"Z"},
      {"type":"filters.assign","value":"UserData = Classification"},
      "out.las"] )";

    const pdg::HybridPipelineRewrite defaultRewrite =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_FALSE(defaultRewrite.hasUnstableInputOrderRegion);
    EXPECT_EQ(defaultRewrite.pointProgramRegions, 2U);
    EXPECT_EQ(defaultRewrite.replacementRegions, 2U);
    EXPECT_EQ(occurrences(defaultRewrite.json, "filters.pdg_info"), 0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, "filters.pdg_expressionstats"),
              0U);
    EXPECT_EQ(occurrences(defaultRewrite.json, "filters.pdg_stats"), 0U);

    const pdg::HybridPipelineRewrite enabled =
        pdg::rewriteHybridPipeline(Pipeline, false, true);
    EXPECT_FALSE(enabled.hasUnstableInputOrderRegion);
    EXPECT_EQ(enabled.pointProgramRegions, 2U);
    EXPECT_EQ(enabled.replacementRegions, 5U);
    EXPECT_EQ(occurrences(enabled.json, "filters.pdg_info"), 1U);
    EXPECT_EQ(occurrences(enabled.json, "filters.pdg_expressionstats"), 1U);
    EXPECT_EQ(occurrences(enabled.json, "filters.pdg_stats"), 1U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
              {"type":"filters.info","point":7},
              {"type":"filters.info","p":"0,1"},
              {"type":"filters.expressionstats","dimension":"Z",
               "expressions":true}, "out.las"])",
        false, true);
    EXPECT_EQ(occurrences(unsupported.json, "filters.pdg_info"), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.pdg_expressionstats"), 0U);
}

TEST(HybridPipeline, SelectsExpressionStatsOnlyForMeasuredExactCardinality)
{
    constexpr std::string_view Pipeline = R"([
      {"type":"readers.las","filename":"in.las"},
      {"type":"filters.expressionstats","dimension":"Classification",
       "expressions":["Classification == 2","Intensity < 100",
                      "ReturnNumber >= 1"]},
      {"type":"writers.las","filename":"out.las"}])";

    const pdg::HybridPipelineRewrite unknown =
        pdg::rewriteHybridPipeline(Pipeline);
    EXPECT_TRUE(unknown.hasPointCountDependentCudaCandidate);
    EXPECT_EQ(unknown.automaticPointCountFilename, "in.las");
    EXPECT_EQ(unknown.automaticPointCountLimit,
              (std::numeric_limits<std::uint64_t>::max)());
    EXPECT_EQ(occurrences(unknown.json, "filters.pdg_expressionstats"), 0U);

    const pdg::HybridPipelineRewrite below =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 999'999);
    EXPECT_EQ(occurrences(below.json, "filters.pdg_expressionstats"), 0U);

    const pdg::HybridPipelineRewrite selected =
        pdg::rewriteHybridPipeline(Pipeline, false, false, 1'000'000);
    EXPECT_EQ(occurrences(selected.json, "filters.pdg_expressionstats"), 1U);
    EXPECT_EQ(occurrences(selected.json, "\"pdg_auto_cuda\":true"), 1U);

    constexpr std::string_view TwoExpressions = R"(["in.las",
      {"type":"filters.expressionstats","dimension":"Z",
       "expressions":["Z > 0","Z < 10"]}, "out.las"])";
    const pdg::HybridPipelineRewrite tooLittleExpressionWork =
        pdg::rewriteHybridPipeline(TwoExpressions, false, false, 1'999'999);
    EXPECT_TRUE(tooLittleExpressionWork.hasPointCountDependentCudaCandidate);
    EXPECT_EQ(occurrences(tooLittleExpressionWork.json,
                          "filters.pdg_expressionstats"),
              0U);
    const pdg::HybridPipelineRewrite twoSelected =
        pdg::rewriteHybridPipeline(TwoExpressions, false, false, 2'000'000);
    EXPECT_EQ(occurrences(twoSelected.json, "filters.pdg_expressionstats"), 1U);

    const pdg::HybridPipelineRewrite filtered = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.expression","expression":"Z > 0"},
          {"type":"filters.expressionstats","dimension":"Z",
           "expressions":["Z > 0","Z < 10","X != Y"]}, "out.las"])",
        false, false, 22'000'000);
    EXPECT_FALSE(filtered.hasPointCountDependentCudaCandidate);
    EXPECT_EQ(occurrences(filtered.json, "filters.pdg_expressionstats"), 0U);

    const pdg::HybridPipelineRewrite limited = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.las","filename":"limited.laz","count":500000},
          {"type":"filters.expressionstats","dimension":"Z",
           "expressions":["Z > 0","Z < 10","X != Y"]}, "out.las"])");
    EXPECT_TRUE(limited.hasPointCountDependentCudaCandidate);
    EXPECT_EQ(limited.automaticPointCountFilename, "limited.laz");
    EXPECT_EQ(limited.automaticPointCountLimit, 500'000U);
}

TEST(HybridPipeline, RewritesMortonOrderingAsGlobalBarrier)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Intensity"},
          {"type":"filters.mortonorder","reverse":true},
          {"type":"filters.expression","expression":"Classification == 2"},
          "out.las"])");
    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_EQ(rewritten.pointProgramRegions, 2U);
    EXPECT_EQ(rewritten.replacementRegions, 3U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridMortonOrderStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.mortonorder"), 0U);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
              {"type":"filters.mortonorder","reverse":"true"},
              {"type":"filters.mortonorder","where":"X > 0"},
              "out.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridMortonOrderStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.mortonorder"), 2U);
}

TEST(HybridPipeline, RewritesTerminalCategoricalGrouping)
{
    const pdg::HybridPipelineRewrite rewritten = pdg::rewriteHybridPipeline(
        R"(["in.las",
          {"type":"filters.assign","value":"Scratch = Classification"},
          {"type":"filters.groupby","dimension":"Scratch"},
          "out#.las"])");
    EXPECT_TRUE(rewritten.linearPipeline);
    EXPECT_FALSE(rewritten.hasUnstableInputOrderRegion);
    EXPECT_EQ(rewritten.pointProgramRegions, 1U);
    EXPECT_EQ(rewritten.replacementRegions, 2U);
    EXPECT_EQ(rewritten.fusedStages, 2U);
    EXPECT_EQ(occurrences(rewritten.json, pdg::HybridGroupByStage), 1U);
    EXPECT_EQ(occurrences(rewritten.json, "filters.groupby"), 0U);

    const pdg::HybridPipelineRewrite splitInput = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.splitter","length":1000},
             {"type":"filters.groupby","dimension":"Classification"},
             "out#.las"])");
    EXPECT_FALSE(splitInput.hasUnstableInputOrderRegion);
    EXPECT_EQ(occurrences(splitInput.json, pdg::HybridGroupByStage), 1U);

    const pdg::HybridPipelineRewrite unsafeSplitInput =
        pdg::rewriteHybridPipeline(
            R"(["in.las", {"type":"filters.splitter","length":1000},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(unsafeSplitInput.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite downstream = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.groupby","dimension":"Classification"},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(downstream.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite orderedDownstream =
        pdg::rewriteHybridPipeline(
            R"(["in.las", {"type":"filters.groupby","dimension":"Classification"},
                 {"type":"filters.sort","dimension":"X"},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(orderedDownstream.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite unsupported = pdg::rewriteHybridPipeline(
        R"(["in.las",
                 {"type":"filters.groupby","dimension":"Classification",
                  "where":"Classification > 1"}, "out#.las"])");
    EXPECT_EQ(unsupported.replacementRegions, 0U);
    EXPECT_EQ(occurrences(unsupported.json, pdg::HybridGroupByStage), 0U);
    EXPECT_EQ(occurrences(unsupported.json, "filters.groupby"), 1U);

    const pdg::HybridPipelineRewrite unsupportedThenNative =
        pdg::rewriteHybridPipeline(
            R"(["in.las",
                 {"type":"filters.groupby","dimension":"Classification",
                  "where":"Classification > 1"},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(unsupportedThenNative.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite unprovenFilterThenNative =
        pdg::rewriteHybridPipeline(
            R"json(["in.las",
                    {"type":"filters.crop",
                     "bounds":["([0, 1], [0, 1])",
                               "([2, 3], [2, 3])"]},
                    {"type":"filters.head","count":1},
                    "out#.las"])json");
    EXPECT_TRUE(unprovenFilterThenNative.hasUnstableInputOrderRegion);
}

TEST(HybridPipeline, RewritesReturnsAndUsesMergeAsMultiViewBarrier)
{
    const pdg::HybridPipelineRewrite exact = pdg::rewriteHybridPipeline(
        R"(["in.las",
             {"type":"filters.returns","groups":["first","last","only"]},
             {"type":"filters.merge"},
             {"type":"filters.head","count":7}, "out.las"])");
    EXPECT_TRUE(exact.linearPipeline);
    EXPECT_FALSE(exact.hasUnstableInputOrderRegion);
    EXPECT_TRUE(exact.standardModeRewriteIsExact);
    EXPECT_EQ(exact.replacementRegions, 3U);
    EXPECT_EQ(exact.fusedStages, 3U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridReturnsStage), 1U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridMergeStage), 1U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridPointProgramStage), 1U);

    const pdg::HybridPipelineRewrite withoutMerge = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.returns","groups":"last"},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(withoutMerge.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite splitterMerge = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.splitter","length":1000},
                 {"type":"filters.merge"},
                 {"type":"filters.head","count":1}, "out.las"])");
    EXPECT_FALSE(splitterMerge.hasUnstableInputOrderRegion);
    EXPECT_EQ(occurrences(splitterMerge.json, pdg::HybridMergeStage), 1U);
    EXPECT_EQ(occurrences(splitterMerge.json, pdg::HybridPointProgramStage),
              1U);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.returns","groups":7},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.returns","groups":"last",
                  "where":"X > 0"}, "out#.las"])",
             R"(["in.las", {"type":"filters.merge",
                  "spatialreference":"EPSG:4326"}, "out.las"])",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(pipeline);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridReturnsStage), 0U)
            << pipeline;
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridMergeStage), 0U)
            << pipeline;
    }
}

TEST(HybridPipeline, RewritesAuditedDividerAndSplitterForms)
{
    const pdg::HybridPipelineRewrite exact = pdg::rewriteHybridPipeline(
        R"(["in.las",
             {"type":"filters.divider","mode":"ROUND_ROBIN","count":7},
             {"type":"filters.splitter","length":250,"buffer":-1},
             {"type":"filters.merge"},
             {"type":"filters.head","count":3}, "out.las"])");
    EXPECT_TRUE(exact.linearPipeline);
    EXPECT_FALSE(exact.hasUnstableInputOrderRegion);
    EXPECT_TRUE(exact.standardModeRewriteIsExact);
    EXPECT_EQ(exact.replacementRegions, 4U);
    EXPECT_EQ(exact.fusedStages, 4U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridDividerStage), 1U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridSplitterStage), 1U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridMergeStage), 1U);

    const pdg::HybridPipelineRewrite buffered = pdg::rewriteHybridPipeline(
        R"(["in.las", {"type":"filters.splitter","length":1000,
             "buffer":20}, "out#.las"])");
    EXPECT_FALSE(buffered.hasUnstableInputOrderRegion);
    EXPECT_EQ(occurrences(buffered.json, pdg::HybridSplitterStage), 1U);

    const pdg::HybridPipelineRewrite unsafeDownstream =
        pdg::rewriteHybridPipeline(
            R"(["in.las", {"type":"filters.divider","count":2},
                 {"type":"filters.head","count":1}, "out#.las"])");
    EXPECT_TRUE(unsafeDownstream.hasUnstableInputOrderRegion);

    for (const std::string& pipeline : {
             R"(["in.las", {"type":"filters.divider","capacity":25},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.divider","count":1},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.divider","count":2,
                  "mode":"sideways"}, "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":"10"},
                  "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":10,
                  "buffer":5}, "out#.las"])",
             R"(["in.las", {"type":"filters.splitter","length":10,
                  "where":"X > 0"}, "out#.las"])",
         })
    {
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(pipeline);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridDividerStage), 0U)
            << pipeline;
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridSplitterStage), 0U)
            << pipeline;
    }
}

TEST(HybridPipeline, RewritesAuditedColorinterpFormsAcrossViews)
{
    const pdg::HybridPipelineRewrite exact = pdg::rewriteHybridPipeline(
        R"(["in.las",
             {"type":"filters.colorinterp","dimension":"Intensity",
              "minimum":0,"maximum":65535,"clamp":true,
              "ramp":"heat_map","invert":true},
             {"type":"filters.divider","count":3},
             {"type":"filters.colorinterp","k":1.5,"mad":true,
              "mad_multiplier":2.5}, "out#.las"])");
    EXPECT_TRUE(exact.linearPipeline);
    EXPECT_FALSE(exact.hasUnstableInputOrderRegion);
    EXPECT_TRUE(exact.standardModeRewriteIsExact);
    EXPECT_EQ(exact.replacementRegions, 3U);
    EXPECT_EQ(exact.fusedStages, 3U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridColorinterpStage), 2U);
    EXPECT_EQ(occurrences(exact.json, pdg::HybridDividerStage), 1U);

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
        const pdg::HybridPipelineRewrite fallback =
            pdg::rewriteHybridPipeline(pipeline);
        EXPECT_EQ(occurrences(fallback.json, pdg::HybridColorinterpStage), 0U)
            << pipeline;
    }
}

TEST(HybridPipeline, FlagsUnsortedParallelReaderOrder)
{
    const pdg::HybridPipelineRewrite unsafe = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.copc","filename":"in.copc.laz"},
          {"type":"filters.expression","expression":"X < Y"},
          "out.las"])");
    EXPECT_TRUE(unsafe.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite unsafeLocate = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.copc","filename":"in.copc.laz"},
              {"type":"filters.locate","dimension":"Z"}, "out.las"])");
    EXPECT_TRUE(unsafeLocate.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite unsafeRobust = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.copc","filename":"in.copc.laz"},
              {"type":"filters.iqr","dimension":"Z"}, "out.las"])");
    EXPECT_TRUE(unsafeRobust.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite unsafeGroup = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.copc","filename":"in.copc.laz"},
                 {"type":"filters.groupby","dimension":"Classification"},
                 "out#.las"])");
    EXPECT_TRUE(unsafeGroup.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite normalized = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.copc","filename":"in.copc.laz"},
          {"type":"filters.sort","dimensions":"X"},
          {"type":"filters.expression","expression":"X < Y"},
          "out.las"])");
    EXPECT_FALSE(normalized.hasUnstableInputOrderRegion);
}

TEST(HybridPipeline, ProvesStreamToStandardModeOnlyForCoveredLinearIo)
{
    const pdg::HybridPipelineRewrite covered = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.las","filename":"in.laz"},
          {"type":"filters.expression","expression":"X < Y"},
          {"type":"writers.las","filename":"out.las"}])");
    EXPECT_TRUE(covered.standardModeRewriteIsExact);

    const pdg::HybridPipelineRewrite shorthand = pdg::rewriteHybridPipeline(
        R"(["in.bpf",
          {"type":"filters.expression","expression":"X < Y"},
          "out.las"])");
    EXPECT_TRUE(shorthand.standardModeRewriteIsExact);

    const pdg::HybridPipelineRewrite unknown = pdg::rewriteHybridPipeline(
        R"([{"type":"readers.e57","filename":"in.e57"},
          {"type":"filters.expression","expression":"X < Y"},
          {"type":"writers.las","filename":"out.las"}])");
    EXPECT_FALSE(unknown.standardModeRewriteIsExact);
    EXPECT_TRUE(unknown.hasUnstableInputOrderRegion);

    const pdg::HybridPipelineRewrite compound = pdg::rewriteHybridPipeline(
        R"(["in.copc.laz",
          {"type":"filters.expression","expression":"X < Y"},
          "out.las"])");
    EXPECT_FALSE(compound.standardModeRewriteIsExact);
    EXPECT_TRUE(compound.hasUnstableInputOrderRegion);
}

TEST(HybridPipeline, RejectsNonPipelineRoots)
{
    EXPECT_THROW(static_cast<void>(pdg::rewriteHybridPipeline(R"({"x":1})")),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(pdg::rewriteHybridPipeline("{")),
                 std::invalid_argument);
}

} // unnamed namespace
