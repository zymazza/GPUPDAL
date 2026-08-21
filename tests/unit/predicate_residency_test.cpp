// B0159/B0162: plan-time residency for predicates.
//
// `buildRuntimePlacement` refuses a plan containing a non-cardinality-preserving
// stage unless that stage qualifies as a declared predicate, and one clause of
// that test is `preferredResidency == MemoryKind::Device`. Plan-time residency
// comes from `predicateSupportsExactDevice(program)`, which calls
// `exactDeviceExpression(expression, /*allowCoordinateLoads=*/false)` — so any
// predicate reading X, Y, or Z is forced to Host and the whole pipeline is
// refused, reported as `non_cardinality_preserving_stage`.
//
// The conservatism has a real basis: a coordinate load is only exact on device
// when the column is binary64, or signed-32 with a zero scale offset, which is
// a property of the runtime batch rather than the plan. The two-argument
// overload checks exactly that. But the consequence is that `filters.crop`,
// which is inherently a coordinate predicate, can never be placed, and neither
// can `filters.range` on Z — two of the most-used filters in PDAL.
//
// B0162 changed that. Plan-time residency now uses
// `predicateMaySupportExactDevice`, which permits coordinate loads, and defers
// the column question to the resident preflight and `evaluatePredicateDevice`
// — both of which check a real batch and fail closed. B0161/B0162 verified
// that every device predicate path in the engine builds its batch with a zero
// coordinate offset, so those two checks agree with each other rather than
// only appearing to.
//
// These cases pin the behaviour on both sides of that boundary so a future
// change is deliberate and visible rather than silent. This file previously
// asserted the opposite for coordinate predicates and failed when B0162
// landed, which is exactly what it exists to do.

#include <pdg/Plan.hpp>

#include <gtest/gtest.h>

#include <variant>

namespace
{

pdg::PlannedStage compiledFilter(const char* pipeline)
{
    pdg::DimensionRegistry dimensions;
    const pdg::Plan plan = pdg::compilePipeline(pipeline, dimensions);
    return plan.stages().at(1U);
}

void expectQualifiesDeclaredPredicate(const pdg::PlannedStage& stage)
{
    EXPECT_TRUE(stage.native);
    EXPECT_TRUE(std::holds_alternative<pdg::PredicateProgram>(stage.payload));
    EXPECT_TRUE(stage.descriptor.fusion.pure);
    EXPECT_TRUE(stage.descriptor.fusion.deterministicSafe);
    EXPECT_TRUE(stage.descriptor.preservesOrder);
    EXPECT_FALSE(stage.descriptor.mutatesCoordinates);
    EXPECT_FALSE(stage.descriptor.fusion.hasWhere);
    EXPECT_EQ(stage.descriptor.fusion.whereMerge,
              pdg::WhereMergeMode::NotApplicable);
    EXPECT_FALSE(stage.descriptor.fusion.cardinalityPreserving);
}

} // unnamed namespace

TEST(PredicateResidency, CoordinateCropAndRangeArePlannedToDevice)
{
    for (const char* pipeline : {
             R"JSON(["in.las", {"type":"filters.range","limits":"Z[0:200]"}, "out.las"])JSON",
             R"JSON(["in.las", {"type":"filters.crop","bounds":"([0,10],[0,10])"}, "out.las"])JSON"})
    {
        SCOPED_TRACE(pipeline);
        const pdg::PlannedStage stage = compiledFilter(pipeline);
        expectQualifiesDeclaredPredicate(stage);
        // Before B0162 this was Host, which disqualified these filters from
        // the declared-predicate exemption and refused the whole pipeline.
        EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    }
}

TEST(PredicateResidency, NonCoordinatePredicatesArePlannedToDevice)
{
    for (const char* pipeline : {
             R"JSON(["in.las", {"type":"filters.expression","expression":"Intensity > 100"}, "out.las"])JSON",
             R"JSON(["in.las", {"type":"filters.range","limits":"Intensity[100:200]"}, "out.las"])JSON"})
    {
        SCOPED_TRACE(pipeline);
        const pdg::PlannedStage stage = compiledFilter(pipeline);
        expectQualifiesDeclaredPredicate(stage);
        EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Device);
    }
}

// B0162 deliberately did not widen `filters.expression`. It carries
// `placementModel = "point-program"` and feeds the fused point-program
// selection paths, so relaxing its residency has a materially larger blast
// radius than `crop` and `range` and deserves its own slice and measurement.
// The asymmetry is pinned here so it reads as a decision rather than an
// oversight: a coordinate expression is still planned to Host and therefore
// still refuses its pipeline.
TEST(PredicateResidency, CoordinateExpressionIsStillPlannedToHost)
{
    const pdg::PlannedStage stage = compiledFilter(
        R"JSON(["in.las", {"type":"filters.expression","expression":"Z > 0"}, "out.las"])JSON");
    expectQualifiesDeclaredPredicate(stage);
    EXPECT_EQ(stage.preferredResidency, pdg::MemoryKind::Host);
}
