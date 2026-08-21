#include "src/cli/Dispatch.hpp"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <string_view>

namespace
{

using pdg::cli::DispatchRoute;

TEST(Dispatcher, DirectRoutesOnlyParsedOpaquePipelines)
{
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"(["input.e57", {"type":"filters.opaque_plugin"},
                      "output.las"])"),
              DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"({"pipeline":[{"type":"readers.las","filename":"in.las"},
                      {"type":"writers.copc","filename":"out.copc.laz"}]})"),
              DispatchRoute::Oracle);
}

TEST(Dispatcher, KeepsPlausibleNativeLasPipelinesInEngine)
{
    // A LAS reader and writer with a filter between them keeps its engine
    // route: a filter is what every accelerated shape has.
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"(["input.LAS", {"type":"filters.normal","knn":8},
                      "output.las"])"),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"({"pipeline":[{"type":"readers.las","filename":"in.las"},
                      {"type":"filters.sort","dimension":"X"},
                      {"type":"writers.las","filename":"out.las"}]})"),
              DispatchRoute::Engine);
}

// B0197/D0219: a bare LAS reader-to-writer pipeline goes straight to the
// oracle. No `pipeline` route accelerates it -- the automatic resident attempt
// declines at plan compilation in 0.17 ms, `tryNativePipeline` requires three
// stages, and the engine ends at `runOracle` anyway -- so routing it through
// the engine loads an 11 MB image and 76 shared libraries to hand the work
// straight back, a measured ~15-17 ms.
TEST(Dispatcher, SendsBareLasTranslateStraightToTheOracle)
{
    EXPECT_EQ(
        pdg::cli::classifyPipelineForDispatch(R"(["input.LAS", "output.las"])"),
        DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"({"pipeline":[{"type":"readers.las","filename":"in.las"},
                      {"type":"writers.las","filename":"out.las"}]})"),
              DispatchRoute::Oracle);
    // Three stages are not a bare translate even when the middle stage is
    // unknown to the classifier.
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"({"pipeline":[{"type":"readers.las","filename":"in.las"},
                      {"type":"filters.ferry","dimensions":"X=>CopyX"},
                      {"type":"writers.las","filename":"out.las"}]})"),
              DispatchRoute::Engine);
}

// B0226: the checked-in r1 grammar has exactly one engine candidate, crop,
// followed by a reprojection stage which no engine route implements. Loading
// the engine to substitute crop and then exec the oracle is slower on the
// exact measured 1M reference layout and bounds.
TEST(Dispatcher, SendsTheMeasuredR1TranslateShapeStraightToTheOracle)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.crop",
       "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
      {"type":"filters.reprojection","in_srs":"EPSG:28992",
       "out_srs":"EPSG:3857"},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, true}),
              DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::dispatchPointCountProbeFilename(Reference),
              "input.laz");
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{999'999U, true}),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'001U, true}),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, false}),
              DispatchRoute::Engine);

    // Refuse every neighboring grammar that is outside the measured route.
    for (const std::string_view pipeline : {
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz","nosrs":true},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "outside":true},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
               {"type":"filters.reprojection","in_srs":"EPSG:4326",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":true}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop","bounds":"([1,2],[3,4])"},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])"},
               {"type":"filters.reprojection","in_srs":"EPSG:28992",
                "out_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}],"metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                      pipeline, pdg::cli::DispatchInputFacts{1'000'000U, true}),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(pipeline))
            << pipeline;
    }
}

// B0243: direct delegation of r8 stayed exact but did not establish a stable
// same-final-binary win over the existing engine route, so the measured graph
// remains fail-closed in the engine even when its input facts match.
TEST(Dispatcher, KeepsUnprovedR8ColorizationInEngine)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.reprojection","in_srs":"EPSG:28992",
       "out_srs":"EPSG:3857"},
      {"type":"filters.colorization","raster":"orthophoto.tif",
       "dimensions":"Red:1:1.0, Green:2:1.0, Blue:3:1.0"},
      {"type":"filters.reprojection","in_srs":"EPSG:3857",
       "out_srs":"EPSG:28992"},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";

    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, true}),
              DispatchRoute::Engine);
    EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(Reference));
    EXPECT_FALSE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
}

// B0245: the exact one-reader-worker r13 prototype did not clear its
// corrected-final cold gate.  Freeze the literal graph in the engine rather
// than leaving the rejected scheduling hypothesis implicit.
TEST(Dispatcher, KeepsUnprovedR13MergeInEngine)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"merge-a.laz","tag":"left"},
      {"type":"readers.las","filename":"merge-b.laz","tag":"right"},
      {"type":"filters.merge","inputs":["left","right"]},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";

    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Engine);
    EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(Reference));
    EXPECT_FALSE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
}

// B0246: a literal r12 direct-delegation prototype remained slower than
// pinned PDAL and its same-final direct-versus-engine A/B was unresolved in
// both cache states.  Freeze the measured graph in the engine.
TEST(Dispatcher, KeepsUnprovedR12TilingInEngine)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.splitter","length":256.0,
       "origin_x":"184320.0","origin_y":"494848.0","buffer":0.0},
      {"type":"writers.las","filename":"output-tile-#.laz",
       "compression":"true"}]})json";

    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, true}),
              DispatchRoute::Engine);
    EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(Reference));
    EXPECT_FALSE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
}

// B0247: corrected r9 is a substantive multipolygon/hole crop. Both direct
// exec and in-process fallback prototypes were exact but resolved slower than
// pinned PDAL, so the literal graph and neighboring shapes remain engine-owned.
TEST(Dispatcher, KeepsCorrectedR9PolygonClipInEngine)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz",
       "override_srs":"EPSG:28992"},
      {"type":"filters.crop",
       "polygon":"MULTIPOLYGON(((5.823268532898 52.441354258557,5.834298426905 52.441313140400,5.834304006406 52.441865170849,5.823273974811 52.441906289668,5.823268532898 52.441354258557),(5.826578874353 52.441480038643,5.829887853234 52.441467718545,5.829889924905 52.441674730066,5.826580930545 52.441687050237,5.826578874353 52.441480038643)),((5.836505103731 52.441373797364,5.840917059642 52.441356980476,5.840921306182 52.441771003009,5.836509308994 52.441787820100,5.836505103731 52.441373797364)))",
       "a_srs":"EPSG:4326"},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";

    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Engine);
    EXPECT_FALSE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
    EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(Reference));

    for (const std::string_view pipeline : {
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.crop",
                "polygon":"MULTIPOLYGON(((5.823268532898 52.441354258557,5.834298426905 52.441313140400,5.834304006406 52.441865170849,5.823273974811 52.441906289668,5.823268532898 52.441354258557),(5.826578874353 52.441480038643,5.829887853234 52.441467718545,5.829889924905 52.441674730066,5.826580930545 52.441687050237,5.826578874353 52.441480038643)),((5.836505103731 52.441373797364,5.840917059642 52.441356980476,5.840921306182 52.441771003009,5.836509308994 52.441787820100,5.836505103731 52.441373797364)))",
                "a_srs":"EPSG:4326"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.crop",
                "polygon":"POLYGON((5.82 52.44,5.84 52.44,5.84 52.45,5.82 52.44))",
                "a_srs":"EPSG:4326"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.crop",
                "polygon":"MULTIPOLYGON(((5.823268532898 52.441354258557,5.834298426905 52.441313140400,5.834304006406 52.441865170849,5.823273974811 52.441906289668,5.823268532898 52.441354258557),(5.826578874353 52.441480038643,5.829887853234 52.441467718545,5.829889924905 52.441674730066,5.826580930545 52.441687050237,5.826578874353 52.441480038643)),((5.836505103731 52.441373797364,5.840917059642 52.441356980476,5.840921306182 52.441771003009,5.836509308994 52.441787820100,5.836505103731 52.441373797364)))",
                "a_srs":"EPSG:3857"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.crop",
                "polygon":"MULTIPOLYGON(((5.823268532898 52.441354258557,5.834298426905 52.441313140400,5.834304006406 52.441865170849,5.823273974811 52.441906289668,5.823268532898 52.441354258557),(5.826578874353 52.441480038643,5.829887853234 52.441467718545,5.829889924905 52.441674730066,5.826580930545 52.441687050237,5.826578874353 52.441480038643)),((5.836505103731 52.441373797364,5.840917059642 52.441356980476,5.840921306182 52.441771003009,5.836509308994 52.441787820100,5.836505103731 52.441373797364)))",
                "a_srs":"EPSG:4326","outside":true},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }
}

// B0228: the measured r5 COPC/stats route loads the engine only to delegate
// the original pipeline. Fifteen paired attribution runs and a current-binary
// reproduction locate its complete deficit in that unused image load.
TEST(Dispatcher, SendsTheMeasuredR5CopcQueryStraightToTheOracle)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.copc","filename":"input.copc.laz",
       "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
       "resolution":1.0,"requests":1},
      {"type":"filters.stats"},
      {"type":"writers.las","filename":"output.las"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Oracle);
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));

    for (const std::string_view pipeline : {
             R"json([{"type":"readers.copc","filename":"input.copc.laz",
               "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
               "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}])json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.COPC.LAZ",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1,"threads":2},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([1,2],[3,4])","resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":2.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":2},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1.0},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats","dimensions":"X,Y,Z"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"writers.las","filename":"output.las"},
               {"type":"filters.stats"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.laz"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.LAS"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las",
                "forward":"all"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "bounds":"([184874.9975,185624.9925],[494942.405,494980.795])",
                "resolution":1.0,"requests":1},
               {"type":"filters.stats"},
               {"type":"writers.las","filename":"output.las"}],
               "metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }
}

// B0239: only the exact 1M reference layout has measured evidence for the
// SMRF/HAG hybrid. Neighboring cardinalities and layouts retain B0230's direct
// pinned-oracle route.
TEST(Dispatcher, SelectsTheMeasuredR2GroundNormalizationFactsOnly)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.smrf"},
      {"type":"filters.hag_nn"},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true",
       "extra_dims":"HeightAboveGround=float32"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, true}),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{999'999U, true}),
              DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  Reference, pdg::cli::DispatchInputFacts{1'000'000U, false}),
              DispatchRoute::Oracle);
    EXPECT_EQ(pdg::cli::dispatchPointCountProbeFilename(Reference),
              "input.laz");
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));

    for (const std::string_view pipeline : {
             R"json([{"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}])json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.LAZ"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.COPC.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz","count":1},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf","cell":1.0},
               {"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.hag_nn","count":2},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.hag_nn"},{"type":"filters.smrf"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.LAZ",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.COPC.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":true,
                "extra_dims":"HeightAboveGround=float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float64"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32","forward":"all"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},{"type":"filters.hag_nn"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true",
                "extra_dims":"HeightAboveGround=float32"}],
               "metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }
}

// B0229: the measured r3 DTM route reaches structural refusal after unchanged
// SMRF makes the adjacent range rewrite unstable, then delegates the original
// pipeline. Direct delegation removes only that discarded engine work.
TEST(Dispatcher, SendsTheMeasuredR3DtmStraightToTheOracle)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.smrf"},
      {"type":"filters.range","limits":"Classification[2:2]"},
      {"type":"writers.gdal","filename":"output.tif",
       "resolution":1.0,"output_type":"idw"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Oracle);
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));

    for (const std::string_view pipeline : {
             R"json([{"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}])json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.LAZ"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.copc.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.COPC.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz","count":1},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf","cell":1.0},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[1:1]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]",
                "tag":"ground"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"filters.smrf"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.TIF",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tiff",
                "resolution":1.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":2.0,"output_type":"idw"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"min"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw",
                "data_type":"float32"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.smrf"},
               {"type":"filters.range","limits":"Classification[2:2]"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"idw"}],
               "metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }
}

// B0248, extended by B0254: the measured r7 DSM engine path performs no
// productive work before delegating the unchanged pipeline.  Direct routing
// is restricted to the complete checked-in grammar and raster policy.
TEST(Dispatcher, SendsOnlyTheMeasuredR7DsmStraightToTheOracle)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz",
       "override_srs":"EPSG:28992"},
      {"type":"filters.returns","groups":"first,only"},
      {"type":"writers.gdal","filename":"output.tif",
       "resolution":1.0,"output_type":"max","dimension":"Z",
       "binmode":true,"data_type":"float64","nodata":-9999.0,
       "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Oracle);
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
    EXPECT_EQ(pdg::cli::dispatchPointCountProbeFilename(Reference),
              "input.laz");
    EXPECT_TRUE(
        pdg::cli::dispatchUsesAutomaticR7ReaderThreads(
            Reference,
            pdg::cli::DispatchInputFacts{1'000'000U, true, false}));
    EXPECT_FALSE(
        pdg::cli::dispatchUsesAutomaticR7ReaderThreads(
            Reference,
            pdg::cli::DispatchInputFacts{999'999U, true, false}));
    EXPECT_FALSE(
        pdg::cli::dispatchUsesAutomaticR7ReaderThreads(
            Reference,
            pdg::cli::DispatchInputFacts{1'000'000U, false, false}));

    for (const std::string_view pipeline : {
             R"json([{"type":"readers.las","filename":"input.laz",
               "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}])json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.LAZ",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.copc.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:3857"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"last"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":["first","only"]},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.TIF",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"min","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"X",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":false,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float32","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,186000],[494923.21,494999.99])"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz",
                "override_srs":"EPSG:28992"},
               {"type":"filters.returns","groups":"first,only"},
               {"type":"writers.gdal","filename":"output.tif",
                "resolution":1.0,"output_type":"max","dimension":"Z",
                "binmode":true,"data_type":"float64","nodata":-9999.0,
                "bounds":"([184500,185999.99],[494923.21,494999.99])"}],
                "metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }
}

// B0249/B0254: the r10 headline's voxel filter is unchanged, and B0254's
// four-worker reader schedule did not clear its final cold gate.
// PDAL work behind an engine that can only delegate the graph. Keep direct
// host dispatch restricted to the complete measured grammar.
TEST(Dispatcher, SendsOnlyTheMeasuredR10DecimationStraightToTheOracle)
{
    constexpr std::string_view Reference = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.laz"},
      {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Reference),
              DispatchRoute::Oracle);
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Reference));
    EXPECT_FALSE(pdg::cli::dispatchPointCountProbeFilename(Reference));
    EXPECT_FALSE(
        pdg::cli::dispatchUsesAutomaticR7ReaderThreads(
            Reference,
            pdg::cli::DispatchInputFacts{1'000'000U, true, false}));

    for (const std::string_view pipeline : {
             R"json([{"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}])json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.LAZ"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.copc.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":1.0},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5,
                "where":"Classification == 2"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.LAZ",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":true}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true","forward":"all"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"filters.voxelcentroidnearestneighbor","cell":2.5},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}],"metadata":"extra.json"})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchUsesAutomaticR7ReaderThreads(
                pipeline,
                pdg::cli::DispatchInputFacts{1'000'000U, true, false}))
            << pipeline;
    }
}

// B0251/D0250, corrected by B0252/D0251: only the literal r14 LAS -> LAZ
// headline and measured uncompressed format-7 facts arm exact two-worker
// lazperf compression. Other conversion
// directions retain generic unchanged-PDAL delegation.
TEST(Dispatcher, SelectsOnlyTheMeasuredR14ParallelCompressionFacts)
{
    constexpr std::string_view Headline = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.las"},
      {"type":"writers.las","filename":"output.laz",
       "compression":"true"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Headline),
              DispatchRoute::Oracle);
    EXPECT_TRUE(pdg::cli::dispatchRequiresPlainPipelineInvocation(Headline));
    EXPECT_EQ(pdg::cli::dispatchPointCountProbeFilename(Headline),
              "input.las");
    EXPECT_TRUE(pdg::cli::dispatchUsesAutomaticR14ParallelCompression(
        Headline,
        pdg::cli::DispatchInputFacts{1'000'000U, false, true}));
    EXPECT_FALSE(pdg::cli::dispatchUsesAutomaticR14ParallelCompression(
        Headline,
        pdg::cli::DispatchInputFacts{999'999U, false, true}));
    EXPECT_FALSE(pdg::cli::dispatchUsesAutomaticR14ParallelCompression(
        Headline,
        pdg::cli::DispatchInputFacts{1'000'000U, false, false}));

    for (const std::string_view pipeline : {
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.las"},
               {"type":"writers.copc","filename":"output.copc.laz",
                "threads":1,"fixed_seed":true}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.laz"},
               {"type":"writers.copc","filename":"output.copc.laz",
                "threads":1,"fixed_seed":true}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "requests":1},
               {"type":"writers.las","filename":"output.las"}]})json",
             R"json({"pipeline":[
               {"type":"readers.copc","filename":"input.copc.laz",
                "requests":1},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
         })
    {
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Oracle)
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
        EXPECT_FALSE(pdg::cli::dispatchUsesAutomaticR14ParallelCompression(
            pipeline,
            pdg::cli::DispatchInputFacts{1'000'000U, false, true}))
            << pipeline;
    }

    for (const std::string_view pipeline : {
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.LAS"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.las",
                "nosrs":true},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.las"},
               {"type":"writers.las","filename":"output.LAZ",
                "compression":"true"}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.las"},
               {"type":"writers.las","filename":"output.laz",
                "compression":true}]})json",
             R"json({"pipeline":[
               {"type":"readers.las","filename":"input.las"},
               {"type":"writers.las","filename":"output.laz",
                "compression":"true","forward":"all"}]})json",
         })
    {
        EXPECT_FALSE(pdg::cli::dispatchUsesAutomaticR14ParallelCompression(
            pipeline,
            pdg::cli::DispatchInputFacts{1'000'000U, false, true}))
            << pipeline;
        EXPECT_FALSE(
            pdg::cli::dispatchRequiresPlainPipelineInvocation(pipeline))
            << pipeline;
    }

    // Two-stage LAS translation has its own older generic direct rule. Adding
    // a candidate filter remains engine-owned and is not evidence for a
    // generalized conversion bypass.
    constexpr std::string_view FilteredLasToLas = R"json({"pipeline":[
      {"type":"readers.las","filename":"input.las"},
      {"type":"filters.assign","value":"Classification = 2"},
      {"type":"writers.las","filename":"output.las"}]})json";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(FilteredLasToLas),
              DispatchRoute::Engine);
}

TEST(Dispatcher, RoutesMalformedAndAmbiguousPipelinesToEngine)
{
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch("{"),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(R"({"pipeline":{}})"),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(
                  R"([{"filename":"input.las"}, "output.las"])"),
              DispatchRoute::Engine);
}

TEST(Dispatcher, RoutesEveryCurrentCandidateStageToEngine)
{
    constexpr std::array<std::string_view, 31> Stages{
        "filters.approximatecoplanar",
        "filters.assign",
        "filters.colorinterp",
        "filters.covariancefeatures",
        "filters.crop",
        "filters.decimation",
        "filters.divider",
        "filters.eigenvalues",
        "filters.expression",
        "filters.expressionstats",
        "filters.ferry",
        "filters.groupby",
        "filters.head",
        "filters.info",
        "filters.iqr",
        "filters.locate",
        "filters.mad",
        "filters.merge",
        "filters.mortonorder",
        "filters.nndistance",
        "filters.normal",
        "filters.outlier",
        "filters.radialdensity",
        "filters.randomize",
        "filters.range",
        "filters.returns",
        "filters.sort",
        "filters.splitter",
        "filters.stats",
        "filters.tail",
        "filters.transformation",
    };
    for (const std::string_view stage : Stages)
    {
        const std::string pipeline =
            std::string("{\"pipeline\":[\"input.las\",{\"type\":\"") +
            std::string(stage) + "\"},\"output.las\"]}";
        EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(pipeline),
                  DispatchRoute::Engine)
            << stage;
    }
}

TEST(Dispatcher, KeepsExpressionStatisticsInEngineBelowAndAtThreshold)
{
    constexpr std::string_view Below = R"({"pipeline":[
      {"type":"readers.las","filename":"below.las"},
      {"type":"filters.expressionstats","dimension":"Classification",
       "expressions":["Classification == 1","Classification == 2"]},
      {"type":"writers.las","filename":"below-out.las"}]})";
    constexpr std::string_view At = R"({"pipeline":[
      {"type":"readers.las","filename":"at.las"},
      {"type":"filters.expressionstats","dimension":"Classification",
       "expressions":["Classification == 1","Classification == 2"]},
      {"type":"writers.las","filename":"at-out.las"}]})";
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(Below),
              DispatchRoute::Engine);
    EXPECT_EQ(pdg::cli::classifyPipelineForDispatch(At), DispatchRoute::Engine);
}

TEST(Dispatcher, OverridesRemainInEngine)
{
    const std::span<const std::string_view> variables =
        pdg::cli::dispatchEngineEnvironmentVariables();
    ASSERT_FALSE(variables.empty());
    for (const std::string_view variable : variables)
        EXPECT_TRUE(
            pdg::cli::dispatchEnvironmentRequiresEngine({&variable, 1U}))
            << variable;

    for (const std::string_view variable :
         {"PDG_DEBUG_HAG_NN_PHASES",
          "PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
          "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA",
          "PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA",
          "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE",
          "PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT",
          "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT",
          "PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT",
          "PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID",
          "PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE",
          "PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK",
          "PDG_TEST_APPROXIMATECOPLANAR_RECOVERABLE_CUDA_FAILURE",
          "PDG_TEST_R4_OUTLIER_RECOVERABLE_CUDA_FAILURE",
          "PDG_TEST_AUTOMATIC_HAG_NN_PROOF_FAILURE",
          "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE",
          "PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE",
          "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_DEVICE_DECLINE"})
    {
        const std::array<std::string_view, 1> environment{variable};
        EXPECT_TRUE(pdg::cli::dispatchEnvironmentRequiresEngine(environment))
            << variable;
    }

    constexpr std::array<std::string_view, 1> OracleOnly{"PDG_ORACLE_PDAL"};
    EXPECT_FALSE(pdg::cli::dispatchEnvironmentRequiresEngine(OracleOnly));

    constexpr std::array<std::string_view, 1> FutureControl{
        "PDG_FUTURE_UNLISTED_PROOF_CONTROL"};
    EXPECT_TRUE(pdg::cli::dispatchEnvironmentRequiresEngine(FutureControl));

    constexpr std::array<std::string_view, 1> Unrelated{"PROJ_DATA"};
    EXPECT_FALSE(pdg::cli::dispatchEnvironmentRequiresEngine(Unrelated));

    constexpr std::array<std::string_view, 1> FrozenClock{
        "PDAL_TEST_FROZEN_EPOCH"};
    EXPECT_FALSE(pdg::cli::dispatchEnvironmentRequiresEngine(FrozenClock));
}

} // unnamed namespace
