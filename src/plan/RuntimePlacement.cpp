#include <pdg/RuntimePlacement.hpp>

#include <pdg/Plan.hpp>
#include <pdg/Scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace pdg
{
namespace
{
RuntimePlacementResult unavailable(RuntimePlacementUnavailableReason reason)
{
    RuntimePlacementResult result;
    result.unavailableReason = reason;
    return result;
}

bool isExactProfile(const PlacementCalibrationProfile& profile) noexcept
{
    return placementCalibrationFor(profile.device) == &profile;
}

bool isLinearSingleReaderSingleWriter(const Plan& plan) noexcept
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() < 2U || stages.front().role != StageRole::Reader ||
        stages.back().role != StageRole::Writer)
        return false;
    for (std::size_t index = 0U; index < stages.size(); ++index)
    {
        const PlannedStage& stage = stages[index];
        if (index == 0U)
        {
            if (!stage.inputs.empty())
                return false;
            continue;
        }
        if (stage.inputs.size() != 1U || stage.inputs.front() != index - 1U)
            return false;
        if (index + 1U < stages.size() && stage.role != StageRole::Filter)
            return false;
    }
    return true;
}

bool validFacts(const Plan& plan, const RuntimePlacementFacts& facts) noexcept
{
    const std::size_t stageCount = plan.stages().size();
    if (facts.inputRecordBytes == 0U || facts.outputRecordBytes == 0U ||
        facts.fallbackRecordBytes == 0U || facts.tilePointCapacity == 0U ||
        facts.deviceMemoryBudgetBytes == 0U ||
        facts.stageScratchBytes.size() != stageCount ||
        facts.stageAdditionalSynchronizations.size() != stageCount)
        return false;
    const bool intrinsicSingleLane =
        facts.directRadiusAssignBoundaryExecutor ||
        facts.directNeighborClassifierBoundaryExecutor ||
        facts.directHagNnBoundaryExecutor ||
        facts.directHagDelaunayBoundaryExecutor ||
        facts.directSkewnessBoundaryExecutor ||
        facts.directSortBoundaryExecutor ||
        facts.directOutlierNnDistanceBoundaryExecutor ||
        facts.directRadiusOutlierRadialDensityBoundaryExecutor;
    const std::size_t specializedExecutors =
        (facts.directRadiusAssignBoundaryExecutor ? 1U : 0U) +
        (facts.directNeighborClassifierBoundaryExecutor ? 1U : 0U) +
        (facts.directHagNnBoundaryExecutor ? 1U : 0U) +
        (facts.directHagDelaunayBoundaryExecutor ? 1U : 0U) +
        (facts.directSkewnessBoundaryExecutor ? 1U : 0U) +
        (facts.directSortBoundaryExecutor ? 1U : 0U) +
        (facts.directOutlierNnDistanceBoundaryExecutor ? 1U : 0U) +
        (facts.directApproximateCoplanarOutputExecutor ? 1U : 0U) +
        (facts.directRadiusOutlierRadialDensityBoundaryExecutor ? 1U : 0U);
    if (specializedExecutors > 1U)
        return false;
    if (facts.boundaryExecutionFacts.empty())
        return !facts.directRadiusAssignBoundaryExecutor &&
               !facts.exactDirectRadiusAssignExecutor &&
               !facts.directNeighborClassifierBoundaryExecutor &&
               !facts.directHagNnBoundaryExecutor &&
               !facts.directHagDelaunayBoundaryExecutor &&
               !facts.directSkewnessBoundaryExecutor &&
               !facts.directSortBoundaryExecutor &&
               !facts.directOutlierNnDistanceBoundaryExecutor &&
               !facts.directRadiusOutlierRadialDensityBoundaryExecutor;
    if (facts.exactDirectRadiusAssignExecutor &&
        !facts.directRadiusAssignBoundaryExecutor)
        return false;
    if (intrinsicSingleLane)
    {
        if (facts.executorLaneCount != 1U)
            return false;
    }
    else if (facts.executorLaneCount < MinimumSweptLaneCount ||
             facts.executorLaneCount > MaximumSweptLaneCount)
        return false;
    if (facts.boundaryExecutionFacts.size() !=
        plan.summary().residencyBoundaries.size())
        return false;
    for (std::size_t index = 0; index < facts.boundaryExecutionFacts.size();
         ++index)
    {
        const PlacementBoundaryExecutionFact& fact =
            facts.boundaryExecutionFacts[index];
        if (fact.boundaryId >= facts.boundaryExecutionFacts.size() ||
            !fact.transferBytesPerPoint || !fact.deviceStagingBytesPerPoint)
            return false;
        for (std::size_t previous = 0; previous < index; ++previous)
            if (facts.boundaryExecutionFacts[previous].boundaryId ==
                fact.boundaryId)
                return false;
    }
    return true;
}

std::size_t instructionCount(const AssignProgram& program) noexcept
{
    std::size_t count = 0U;
    for (const PointAssignment& assignment : program.assignments)
        count += assignment.value.instructions.size() +
                 assignment.condition.instructions.size();
    return count;
}

std::size_t touchedDimensionCount(const AssignProgram& program)
{
    std::vector<DimensionId> touched = program.reads;
    for (DimensionId dimension : program.writes)
        if (std::find(touched.begin(), touched.end(), dimension) ==
            touched.end())
            touched.push_back(dimension);
    return touched.size();
}

bool fusedPointProgramCalibrationEnvelope(const AssignProgram& program)
{
    const std::size_t assignments = program.assignments.size();
    const std::size_t instructions = instructionCount(program);
    // The measured class starts at the B0005 selector boundary and is bounded
    // by the one-launch fused LAS kernel envelope from D0052. Larger programs
    // may require another launch and therefore cannot inherit this residual.
    return assignments >= 5U && assignments <= 8U && instructions >= 28U &&
           instructions <= 96U && program.writes.size() <= 5U &&
           touchedDimensionCount(program) <= 6U &&
           assignSupportsExactDevice(program);
}

bool sameInstruction(const ExpressionInstruction& left,
                     const ExpressionInstruction& right) noexcept
{
    return left.op == right.op && left.dimension == right.dimension &&
           left.immediate == right.immediate;
}

bool sameExpression(const CompiledExpression& left,
                    const CompiledExpression& right) noexcept
{
    return left.reads == right.reads &&
           left.maximumStackDepth == right.maximumStackDepth &&
           left.boolean == right.boolean &&
           left.instructions.size() == right.instructions.size() &&
           std::equal(left.instructions.begin(), left.instructions.end(),
                      right.instructions.begin(), sameInstruction);
}

bool sameAssignProgram(const AssignProgram& left, const AssignProgram& right)
{
    if (left.reads != right.reads || left.writes != right.writes ||
        left.assignments.size() != right.assignments.size())
        return false;
    for (std::size_t index = 0U; index < left.assignments.size(); ++index)
    {
        const PointAssignment& actual = left.assignments[index];
        const PointAssignment& expected = right.assignments[index];
        if (actual.destination != expected.destination ||
            actual.destinationCreated != expected.destinationCreated ||
            !sameExpression(actual.value, expected.value) ||
            !sameExpression(actual.condition, expected.condition))
            return false;
    }
    return true;
}

const AssignProgram& measuredEigenFamilyAssignments()
{
    static const AssignProgram program = []
    {
        DimensionRegistry dimensions;
        const std::vector<std::string> specifications{
            "Classification = Linearity * 10", "Intensity = Curvature * 1000",
            "UserData = Eigenvalue0 * 100"};
        return compileAssignments(specifications, dimensions);
    }();
    return program;
}

std::optional<std::size_t> measuredEigenFamilyRegion(const Plan& plan)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 6U || plan.summary().residentRegions != 1U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* normal = std::get_if<NormalProgram>(&stages[1].payload);
    const auto* eigenvalues =
        std::get_if<EigenvaluesProgram>(&stages[2].payload);
    const auto* covariance =
        std::get_if<CovarianceFeaturesProgram>(&stages[3].payload);
    const auto* assignments = std::get_if<AssignProgram>(&stages[4].payload);
    if (!normal || !eigenvalues || !covariance || !assignments ||
        stages[1].descriptor.type != "filters.normal" ||
        stages[2].descriptor.type != "filters.eigenvalues" ||
        stages[3].descriptor.type != "filters.covariancefeatures" ||
        stages[4].descriptor.type != "filters.assign" ||
        normal->neighbors != 12 || normal->alwaysUp ||
        eigenvalues->neighbors != 12 || !eigenvalues->normalize ||
        covariance->neighbors != 12 ||
        covariance->mode != EigenvalueMode::Raw ||
        covariance->features != CovarianceDimensionality ||
        !sameAssignProgram(*assignments, measuredEigenFamilyAssignments()))
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 4U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

// B0187: the adjacent `filters.normal` + `filters.covariancefeatures` pair as
// a real features pipeline writes it.
//
// B0186 measured r6-features at 1.003x and traced the refusal: both stages are
// already planner-assigned to device, but they carry separate per-stage models
// and D0077's mixed-models rule declines the pair. The eigen family would
// cover them and reaches 10.021x, but additionally demands an `eigenvalues`
// stage, an `assign` with three exact expressions, `knn=12`,
// `always_up=false` and `normalize=true` — none of which a features pipeline
// has. This envelope is deliberately shaped around what users write rather
// than around a calibration fixture.
//
// `knn` is pinned to 8 because the composition model has no neighbor-count
// term and kNN cost scales with it, so admitting other widths would predict
// with a model that was never measured for them. Eight is both PDAL's default
// for these stages and r6's value. A knn-parameterized model is the follow-up,
// not this slice.
//
// `alwaysUp` is deliberately *not* pinned. It normalizes a computed normal's
// sign per point after the neighborhood work is complete; it changes no
// neighbor search, no covariance, and no eigensystem, so it cannot move the
// cost this model predicts. Pinning it would repeat exactly the defect B0183
// found, where an option whose default is the opposite of the envelope's made
// a qualified route unreachable.
std::optional<std::size_t>
measuredNormalCovarianceRegion(const Plan& plan,
                               const RuntimePlacementFacts& facts,
                               bool extraDimsAllProfileModel = false)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 4U || plan.summary().residentRegions != 1U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* normal = std::get_if<NormalProgram>(&stages[1].payload);
    const auto* covariance =
        std::get_if<CovarianceFeaturesProgram>(&stages[2].payload);
    if (!normal || !covariance ||
        stages[1].descriptor.type != "filters.normal" ||
        stages[2].descriptor.type != "filters.covariancefeatures" ||
        normal->neighbors != 8 || covariance->neighbors != 8 ||
        covariance->mode != EigenvalueMode::Sqrt ||
        covariance->features != CovarianceDimensionality)
        return std::nullopt;

    const auto* writer = std::get_if<FileStagePlan>(&stages.back().payload);
    if (!writer)
        return std::nullopt;
    // B0223/B0224 add two performance-qualified wider sinks to B0187's
    // plain-writer ladder: the exact uncompressed and compressed publication
    // rows. Do not project either across the full 50K--4M model or across
    // carried source Extra Bytes: both change publication work that the
    // composition residual never measured. Functional execution remains exact
    // through resident/host fallback; automatic device selection is bounded
    // to the measured format-7 36 -> 100-byte, 1M-point rows — unless the
    // active profile carries the separately measured extra_dims=all
    // composition model (D0279), which the caller substitutes below.
    if (writer && writer->extraDimensionsAll && !extraDimsAllProfileModel)
    {
        const bool commonMeasuredLayout =
            facts.inputPointCount == 1'000'000U &&
            facts.inputPointFormat == 7U && facts.inputCompressed &&
            facts.inputRecordBytes == 36U && facts.outputRecordBytes == 100U;
        const bool measuredUncompressedSink =
            commonMeasuredLayout && !facts.outputCompressed;
        const bool measuredCompressedSink =
            commonMeasuredLayout && facts.outputCompressed;
        if (!measuredUncompressedSink && !measuredCompressedSink)
            return std::nullopt;
    }
    // Model presence is never permission to interpolate across LAS layouts.
    // Keep the original B0223/B0224 1M format-7 rows and B0285's two large
    // complete-process tuples explicit. Compression, source PDRF/stride, and
    // emitted stride are independent performance facts; every cross-product
    // not listed here remains on host until it has retained evidence.
    if (writer && writer->extraDimensionsAll && extraDimsAllProfileModel)
    {
        const bool originalOneMillionRow =
            facts.inputPointCount == 1'000'000U &&
            facts.inputPointFormat == 7U && facts.inputCompressed &&
            facts.inputRecordBytes == 36U &&
            facts.outputRecordBytes == 100U;
        const bool veilRow = facts.inputPointFormat == 6U &&
                             facts.inputCompressed &&
                             facts.inputRecordBytes == 30U &&
                             facts.outputCompressed &&
                             facts.outputRecordBytes == 100U;
        const bool ahn4Row = facts.inputPointFormat == 8U &&
                             facts.inputCompressed &&
                             facts.inputRecordBytes == 44U &&
                             facts.outputCompressed &&
                             facts.outputRecordBytes == 120U;
        if (!originalOneMillionRow && !veilRow && !ahn4Row)
            return std::nullopt;
    }
    // The enlarged plain-publication model is likewise not a layout wildcard.
    // B0187 measured an uncompressed format-7/36 source and B0280 measured the
    // compressed AHN4 format-8/44 source; both publish uncompressed format-7/36
    // output. Other physical pairs keep the host path until independently
    // qualified.
    if (!writer->extraDimensionsAll)
    {
        const bool originalPlainRow =
            facts.inputPointFormat == 7U && !facts.inputCompressed &&
            facts.inputRecordBytes == 36U;
        const bool ahn4PlainRow = facts.inputPointFormat == 8U &&
                                  facts.inputCompressed &&
                                  facts.inputRecordBytes == 44U;
        if ((!originalPlainRow && !ahn4PlainRow) || facts.outputCompressed ||
            facts.outputRecordBytes != 36U)
            return std::nullopt;
    }

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 2U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

const AssignProgram& measuredRankOptimalAssignments()
{
    static const AssignProgram program = []
    {
        DimensionRegistry dimensions;
        const std::vector<std::string> specifications{
            "Classification = Rank", "Intensity = OptimalKNN",
            "PointSourceId = OptimalRadius"};
        return compileAssignments(specifications, dimensions);
    }();
    return program;
}

const AssignProgram& measuredDirectRadiusAssignUpdates()
{
    static const AssignProgram program = []
    {
        DimensionRegistry dimensions;
        const std::vector<std::string> specifications{"UserData = 9"};
        return compileAssignments(specifications, dimensions);
    }();
    return program;
}

const AssignProgram& measuredRadiusOutlierRadialDensityAssignments()
{
    static const AssignProgram program = []
    {
        DimensionRegistry dimensions;
        const std::vector<std::string> specifications{
            "UserData = 1 WHERE RadialDensity >= 0.2"};
        return compileAssignments(specifications, dimensions);
    }();
    return program;
}

std::optional<std::size_t> measuredDirectRadiusAssignRegion(const Plan& plan)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* radius = std::get_if<RadiusAssignProgram>(&stages[1].payload);
    if (!radius || stages[1].descriptor.type != "filters.radiusassign" ||
        stages[1].descriptor.placementModel != "radiusassign" ||
        radius->radius != 2.0 || !radius->search3d ||
        radius->maximumAbove != -1.0 || radius->maximumBelow != -1.0 ||
        !sameAssignProgram(radius->updates,
                           measuredDirectRadiusAssignUpdates()))
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectNeighborClassifierRegion(const Plan& plan)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* classifier =
        std::get_if<NeighborClassifierProgram>(&stages[1].payload);
    if (!classifier ||
        stages[1].descriptor.type != "filters.neighborclassifier" ||
        stages[1].descriptor.placementModel != "neighborclassifier" ||
        classifier->neighbors != 7 ||
        stages[1].deviceIndexBuildBytesPerPoint != 112U ||
        plan.summary().residencyBoundaries[0].kind !=
            ResidencyBoundaryKind::Upload ||
        plan.summary().residencyBoundaries[0].bytesPerPoint != 25U ||
        plan.summary().residencyBoundaries[1].kind !=
            ResidencyBoundaryKind::Spill ||
        plan.summary().residencyBoundaries[1].bytesPerPoint != 1U)
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectHagNnCountOneRegion(const Plan& plan,
                                  const RuntimePlacementFacts& facts)
{
    constexpr std::size_t InputRecordBytesPerPoint = 40U;
    constexpr std::size_t OutputRecordBytesPerPoint = 48U;
    constexpr std::size_t UploadBytesPerPoint = 25U;
    constexpr std::size_t SpillBytesPerPoint = sizeof(double);
    constexpr std::size_t IndexBytesPerPoint = 112U;

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* reader = std::get_if<FileStagePlan>(&stages.front().payload);
    const auto* hagNn = std::get_if<HagNnProgram>(&stages[1].payload);
    const auto* writer = std::get_if<FileStagePlan>(&stages.back().payload);
    if (!reader || !reader->optionFreeLasFamily || !hagNn || !writer ||
        !writer->extraDimensionsAll ||
        stages[1].descriptor.type != "filters.hag_nn" ||
        !stages[1].descriptor.placementModel.empty() || hagNn->count != 1U ||
        hagNn->maximumDistance != 0.0 || !hagNn->allowExtrapolation ||
        hagNn->groundClass != 2U ||
        stages[1].descriptor.index.kind != IndexKind::Knn ||
        stages[1].descriptor.index.neighbors != 1U ||
        stages[1].descriptor.index.dimensions != 2U ||
        stages[1].deviceIndexBuildBytesPerPoint != IndexBytesPerPoint)
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U || upload.bytesPerPoint != UploadBytesPerPoint ||
        upload.repackBytesPerPoint != 0U || upload.requiresFullPointRecord ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 1U ||
        spill.consumer != 2U || spill.bytesPerPoint != SpillBytesPerPoint ||
        spill.repackBytesPerPoint != SpillBytesPerPoint ||
        spill.requiresFullPointRecord)
        return std::nullopt;

    if (facts.inputPointFormat != 7U || facts.inputCompressed ||
        facts.outputCompressed ||
        facts.inputRecordBytes != InputRecordBytesPerPoint ||
        facts.outputRecordBytes != OutputRecordBytesPerPoint ||
        facts.tilePointCapacity != facts.inputPointCount ||
        facts.executorLaneCount != 1U ||
        facts.boundaryExecutionFacts.size() != 2U ||
        facts.stageScratchBytes.size() != stages.size() ||
        facts.stageScratchBytes[0] != 0U || facts.stageScratchBytes[2] != 0U ||
        facts.inputPointCount >
            (std::numeric_limits<std::size_t>::max)() /
                HagNnCountOneExactDeviceScratchBytesPerPoint ||
        facts.inputPointCount > (std::numeric_limits<std::size_t>::max)() /
                                    HagNnCountOneExactDevicePeakBytesPerPoint ||
        facts.stageScratchBytes[1] !=
            facts.inputPointCount *
                HagNnCountOneExactDeviceScratchBytesPerPoint)
        return std::nullopt;

    for (std::size_t boundaryId = 0U; boundaryId < 2U; ++boundaryId)
    {
        const PlacementBoundaryExecutionFact& fact =
            facts.boundaryExecutionFacts[boundaryId];
        const std::size_t transferBytesPerPoint =
            boundaryId == 0U ? UploadBytesPerPoint : SpillBytesPerPoint;
        if (fact.boundaryId != boundaryId ||
            fact.transferBytesPerPoint != transferBytesPerPoint ||
            fact.packingBytesPerPoint != 0U ||
            fact.deviceStagingBytesPerPoint != InputRecordBytesPerPoint)
            return std::nullopt;
    }

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectHagDelaunayCountThreeRegion(const Plan& plan,
                                          const RuntimePlacementFacts& facts)
{
    constexpr std::size_t InputRecordBytesPerPoint = 40U;
    constexpr std::size_t OutputRecordBytesPerPoint = 48U;
    constexpr std::size_t UploadBytesPerPoint = 25U;
    constexpr std::size_t SpillBytesPerPoint = sizeof(double);
    constexpr std::size_t IndexBytesPerPoint = 112U;

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* reader = std::get_if<FileStagePlan>(&stages.front().payload);
    const auto* hagDelaunay =
        std::get_if<HagDelaunayProgram>(&stages[1].payload);
    const auto* writer = std::get_if<FileStagePlan>(&stages.back().payload);
    if (!reader || !reader->optionFreeLasFamily || !hagDelaunay || !writer ||
        !writer->extraDimensionsAll ||
        stages[1].descriptor.type != "filters.hag_delaunay" ||
        !stages[1].descriptor.placementModel.empty() ||
        hagDelaunay->count != 3U || !hagDelaunay->allowExtrapolation ||
        hagDelaunay->groundClass != 2U ||
        stages[1].descriptor.index.kind != IndexKind::Knn ||
        stages[1].descriptor.index.neighbors != 3U ||
        stages[1].descriptor.index.dimensions != 2U ||
        stages[1].deviceIndexBuildBytesPerPoint != IndexBytesPerPoint)
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U || upload.bytesPerPoint != UploadBytesPerPoint ||
        upload.repackBytesPerPoint != 0U || upload.requiresFullPointRecord ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 1U ||
        spill.consumer != 2U || spill.bytesPerPoint != SpillBytesPerPoint ||
        spill.repackBytesPerPoint != SpillBytesPerPoint ||
        spill.requiresFullPointRecord)
        return std::nullopt;

    if (facts.inputPointFormat != 7U || facts.inputCompressed ||
        facts.outputCompressed ||
        facts.inputRecordBytes != InputRecordBytesPerPoint ||
        facts.outputRecordBytes != OutputRecordBytesPerPoint ||
        facts.tilePointCapacity != facts.inputPointCount ||
        facts.executorLaneCount != 1U ||
        facts.boundaryExecutionFacts.size() != 2U ||
        facts.stageScratchBytes.size() != stages.size() ||
        facts.stageScratchBytes[0] != 0U || facts.stageScratchBytes[2] != 0U ||
        facts.inputPointCount >
            (std::numeric_limits<std::size_t>::max)() /
                HagDelaunayCountThreeExactDeviceScratchBytesPerPoint ||
        facts.inputPointCount >
            (std::numeric_limits<std::size_t>::max)() /
                HagDelaunayCountThreeExactDevicePeakBytesPerPoint ||
        facts.stageScratchBytes[1] !=
            facts.inputPointCount *
                HagDelaunayCountThreeExactDeviceScratchBytesPerPoint)
        return std::nullopt;

    for (std::size_t boundaryId = 0U; boundaryId < 2U; ++boundaryId)
    {
        const PlacementBoundaryExecutionFact& fact =
            facts.boundaryExecutionFacts[boundaryId];
        const std::size_t transferBytesPerPoint =
            boundaryId == 0U ? UploadBytesPerPoint : SpillBytesPerPoint;
        if (fact.boundaryId != boundaryId ||
            fact.transferBytesPerPoint != transferBytesPerPoint ||
            fact.packingBytesPerPoint != 0U ||
            fact.deviceStagingBytesPerPoint != InputRecordBytesPerPoint)
            return std::nullopt;
    }

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectSkewnessRegion(const Plan& plan,
                             const RuntimePlacementFacts& facts)
{
    constexpr std::size_t KeyBytesPerPoint = sizeof(double);
    constexpr std::size_t RecordBytesPerPoint = 36U;

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 0U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* reader = std::get_if<FileStagePlan>(&stages.front().payload);
    const auto* skewness = std::get_if<SkewnessProgram>(&stages[1].payload);
    const auto* writer = std::get_if<FileStagePlan>(&stages.back().payload);
    if (!reader || !reader->optionFreeLasFamily || !skewness || !writer ||
        !writer->extraDimensionsAll ||
        stages[1].descriptor.type != "filters.skewnessbalancing" ||
        !stages[1].descriptor.placementModel.empty() ||
        skewness->groundClass != 2U || skewness->otherClass != 1U ||
        skewness->onlyGround ||
        stages[1].descriptor.deviceToHostBytesPerInputPoint != KeyBytesPerPoint)
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U ||
        upload.bytesPerPoint != KeyBytesPerPoint + 1U ||
        upload.repackBytesPerPoint != 0U || upload.requiresFullPointRecord ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 1U ||
        spill.consumer != 2U || spill.bytesPerPoint != 1U ||
        spill.repackBytesPerPoint != 1U || spill.requiresFullPointRecord)
        return std::nullopt;

    if (facts.inputPointFormat != 7U || facts.inputCompressed ||
        facts.outputCompressed ||
        facts.inputRecordBytes != RecordBytesPerPoint ||
        facts.outputRecordBytes != RecordBytesPerPoint ||
        facts.tilePointCapacity != facts.inputPointCount ||
        facts.executorLaneCount != 1U ||
        facts.boundaryExecutionFacts.size() != 2U ||
        facts.stageScratchBytes.size() != stages.size() ||
        facts.stageScratchBytes[0] != 0U || facts.stageScratchBytes[2] != 0U ||
        facts.inputPointCount > (std::numeric_limits<std::size_t>::max)() /
                                    SkewnessExactDeviceScratchBytesPerPoint ||
        facts.inputPointCount > (std::numeric_limits<std::size_t>::max)() /
                                    SkewnessExactDevicePeakBytesPerPoint ||
        facts.stageScratchBytes[1] !=
            facts.inputPointCount * SkewnessExactDeviceScratchBytesPerPoint)
        return std::nullopt;

    for (std::size_t boundaryId = 0U; boundaryId < 2U; ++boundaryId)
    {
        const PlacementBoundaryExecutionFact& fact =
            facts.boundaryExecutionFacts[boundaryId];
        if (fact.boundaryId != boundaryId ||
            fact.transferBytesPerPoint != KeyBytesPerPoint ||
            fact.packingBytesPerPoint != 0U ||
            fact.deviceStagingBytesPerPoint != RecordBytesPerPoint)
            return std::nullopt;
    }

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectSortRegion(const Plan& plan, const RuntimePlacementFacts& facts)
{
    constexpr std::size_t KeyBytesPerPoint = sizeof(double);
    constexpr std::size_t RecordBytesPerPoint = 36U;

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 3U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 0U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* reader = std::get_if<FileStagePlan>(&stages.front().payload);
    const auto* ordering = std::get_if<OrderingProgram>(&stages[1].payload);
    const auto* writer = std::get_if<FileStagePlan>(&stages.back().payload);
    if (!reader || !reader->optionFreeLasFamily || !ordering || !writer ||
        !writer->extraDimensionsAll ||
        stages[1].descriptor.type != "filters.sort" ||
        !stages[1].descriptor.placementModel.empty() ||
        ordering->dimensions.size() != 1U ||
        ordering->dimensions.front() != DimensionId(StandardDimension::Z) ||
        ordering->direction != OrderingDirection::Ascending ||
        ordering->algorithm != OrderingAlgorithm::Normal ||
        stages[1].descriptor.deviceToHostBytesPerInputPoint != KeyBytesPerPoint)
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U || upload.bytesPerPoint != KeyBytesPerPoint ||
        upload.repackBytesPerPoint != 0U || upload.requiresFullPointRecord ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 1U ||
        spill.consumer != 2U || spill.bytesPerPoint != 0U ||
        spill.repackBytesPerPoint != 0U || spill.requiresFullPointRecord)
        return std::nullopt;

    if (facts.inputPointFormat != 7U || facts.inputCompressed ||
        facts.outputCompressed ||
        facts.inputRecordBytes != RecordBytesPerPoint ||
        facts.outputRecordBytes != RecordBytesPerPoint ||
        facts.tilePointCapacity != facts.inputPointCount ||
        facts.executorLaneCount != 1U ||
        facts.boundaryExecutionFacts.size() != 2U ||
        facts.stageScratchBytes.size() != stages.size() ||
        facts.stageScratchBytes[0] != 0U || facts.stageScratchBytes[2] != 0U ||
        facts.inputPointCount > (std::numeric_limits<std::size_t>::max)() /
                                    OrderingExactDeviceScratchBytesPerPoint ||
        facts.inputPointCount > (std::numeric_limits<std::size_t>::max)() /
                                    OrderingExactDevicePeakBytesPerPoint ||
        facts.stageScratchBytes[1] !=
            facts.inputPointCount * OrderingExactDeviceScratchBytesPerPoint)
        return std::nullopt;

    for (std::size_t boundaryId = 0U; boundaryId < 2U; ++boundaryId)
    {
        const PlacementBoundaryExecutionFact& fact =
            facts.boundaryExecutionFacts[boundaryId];
        if (fact.boundaryId != boundaryId ||
            fact.transferBytesPerPoint != KeyBytesPerPoint ||
            fact.packingBytesPerPoint != 0U ||
            fact.deviceStagingBytesPerPoint != RecordBytesPerPoint)
            return std::nullopt;
    }

    const std::size_t region = stages[1].residentRegion;
    if (!stages[1].native ||
        stages[1].preferredResidency != MemoryKind::Device ||
        region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    return region;
}

std::optional<std::size_t> measuredRankOptimalRegion(const Plan& plan)
{
    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 5U || plan.summary().residentRegions != 1U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* estimate = std::get_if<EstimateRankProgram>(&stages[1].payload);
    const auto* optimal =
        std::get_if<OptimalNeighborhoodProgram>(&stages[2].payload);
    const auto* assignments = std::get_if<AssignProgram>(&stages[3].payload);
    if (!estimate || !optimal || !assignments ||
        stages[1].descriptor.type != "filters.estimaterank" ||
        stages[2].descriptor.type != "filters.optimalneighborhood" ||
        stages[3].descriptor.type != "filters.assign" ||
        estimate->neighbors != 14 || estimate->threshold != 0.01 ||
        optimal->minimumK != 10U || optimal->maximumK != 14U ||
        !sameAssignProgram(*assignments, measuredRankOptimalAssignments()))
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 3U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectOutlierNnDistanceRegion(const Plan& plan)
{
    constexpr std::size_t GatherNeighbors = 11U;
    constexpr std::size_t GatherBytesPerPoint =
        GatherNeighbors * (sizeof(std::uint32_t) + sizeof(double)) +
        sizeof(std::uint8_t);

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 4U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* outlier = std::get_if<OutlierProgram>(&stages[1].payload);
    const auto* nnDistance = std::get_if<NnDistanceProgram>(&stages[2].payload);
    if (!outlier || !nnDistance ||
        stages[1].descriptor.type != "filters.outlier" ||
        stages[2].descriptor.type != "filters.nndistance" ||
        outlier->method != OutlierMethod::Statistical ||
        outlier->meanNeighbors != 8 || outlier->multiplier != 2.0 ||
        outlier->classification != 7U || nnDistance->k != 10U ||
        nnDistance->mode != KnnDistanceMode::Kth ||
        stages[1].deviceKnnGatherNeighbors != GatherNeighbors ||
        stages[2].deviceKnnGatherNeighbors != GatherNeighbors ||
        stages[1].deviceQueryBytesPerPoint != GatherBytesPerPoint ||
        stages[2].deviceQueryBytesPerPoint != GatherBytesPerPoint)
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 2U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectRadiusOutlierRadialDensityRegion(const Plan& plan)
{
    constexpr std::size_t RadiusIndexBytesPerPoint = 28U;
    constexpr std::size_t CountBytesPerPoint = sizeof(std::uint32_t);
    constexpr std::size_t UploadBytesPerPoint = 24U;
    constexpr std::size_t SpillBytesPerPoint =
        sizeof(double) + 2U * sizeof(std::uint8_t);

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 5U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* outlier = std::get_if<OutlierProgram>(&stages[1].payload);
    const auto* density = std::get_if<RadialDensityProgram>(&stages[2].payload);
    const auto* assignments = std::get_if<AssignProgram>(&stages[3].payload);
    if (!outlier || !density || !assignments ||
        stages[1].descriptor.type != "filters.outlier" ||
        stages[2].descriptor.type != "filters.radialdensity" ||
        stages[3].descriptor.type != "filters.assign" ||
        outlier->method != OutlierMethod::Radius || outlier->radius != 1.01 ||
        outlier->minimumNeighbors != 2 || outlier->classification != 7U ||
        density->radius != 1.01 ||
        stages[1].deviceIndexBuildBytesPerPoint != RadiusIndexBytesPerPoint ||
        stages[2].deviceIndexBuildBytesPerPoint != 0U ||
        stages[1].deviceQueryBytesPerPoint != CountBytesPerPoint ||
        stages[1].descriptor.deviceToHostBytesPerInputPoint !=
            CountBytesPerPoint ||
        stages[2].descriptor.deviceToHostBytesPerInputPoint != 0U ||
        !sameAssignProgram(*assignments,
                           measuredRadiusOutlierRadialDensityAssignments()))
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U || upload.bytesPerPoint != UploadBytesPerPoint ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 3U ||
        spill.consumer != 4U || spill.bytesPerPoint != SpillBytesPerPoint ||
        spill.repackBytesPerPoint != SpillBytesPerPoint)
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 3U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

std::optional<std::size_t>
measuredDirectApproximateCoplanarRegion(const Plan& plan)
{
    constexpr std::size_t IndexBuildBytesPerPoint = 112U;
    constexpr std::size_t UploadBytesPerPoint = 25U;
    constexpr std::size_t SpillBytesPerPoint = 2U;

    const std::vector<PlannedStage>& stages = plan.stages();
    if (stages.size() != 4U || plan.summary().residentRegions != 1U ||
        plan.summary().indexBuilds != 1U ||
        plan.summary().residencyBoundaries.size() != 2U ||
        stages.front().role != StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || stages.back().role != StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" || !stages.back().native)
        return std::nullopt;

    const auto* approximate =
        std::get_if<ApproximateCoplanarProgram>(&stages[1].payload);
    const auto* ferry = std::get_if<FerryProgram>(&stages[2].payload);
    if (!approximate || !ferry ||
        stages[1].descriptor.type != "filters.approximatecoplanar" ||
        stages[1].descriptor.placementModel != "approximatecoplanar" ||
        approximate->neighbors != 8 || approximate->threshold1 != 25.0 ||
        approximate->threshold2 != 6.0 ||
        stages[1].deviceIndexBuildBytesPerPoint != IndexBuildBytesPerPoint ||
        stages[2].descriptor.type != "filters.ferry" ||
        stages[2].descriptor.placementModel != "point-program" ||
        ferry->copies.size() != 1U)
        return std::nullopt;

    const FerryCopy& copy = ferry->copies.front();
    if (!copy.hasSource || copy.destinationCreated ||
        copy.source != DimensionId(StandardDimension::Coplanar) ||
        copy.destination != DimensionId(StandardDimension::UserData))
        return std::nullopt;

    const ResidencyBoundary& upload = plan.summary().residencyBoundaries[0];
    const ResidencyBoundary& spill = plan.summary().residencyBoundaries[1];
    if (upload.kind != ResidencyBoundaryKind::Upload || upload.producer != 0U ||
        upload.consumer != 1U || upload.bytesPerPoint != UploadBytesPerPoint ||
        spill.kind != ResidencyBoundaryKind::Spill || spill.producer != 2U ||
        spill.consumer != 3U || spill.bytesPerPoint != SpillBytesPerPoint ||
        spill.repackBytesPerPoint != SpillBytesPerPoint)
        return std::nullopt;

    const std::size_t region = stages[1].residentRegion;
    if (region == NoResidentRegion || region >= plan.summary().residentRegions)
        return std::nullopt;
    for (std::size_t index = 1U; index <= 2U; ++index)
        if (!stages[index].native ||
            stages[index].preferredResidency != MemoryKind::Device ||
            stages[index].residentRegion != region)
            return std::nullopt;
    return region;
}

} // unnamed namespace

RuntimePlacementUnavailableReason
planStructureRefusal(const Plan& plan) noexcept
{
    if (!isLinearSingleReaderSingleWriter(plan))
        return RuntimePlacementUnavailableReason::UnsupportedTopology;

    const std::vector<PlannedStage>& stages = plan.stages();
    for (std::size_t index = 1U; index + 1U < stages.size(); ++index)
    {
        const PlannedStage& stage = stages[index];
        if (stage.descriptor.fusion.cardinalityPreserving)
            continue;
        // A cardinality change is admissible only when the descriptor
        // declares a pure, deterministic-safe, order-preserving predicate
        // without conditional where semantics; anything else fails closed.
        const bool declaredPredicate =
            stage.native && stage.preferredResidency == MemoryKind::Device &&
            std::holds_alternative<PredicateProgram>(stage.payload) &&
            stage.descriptor.fusion.pure &&
            stage.descriptor.fusion.deterministicSafe &&
            stage.descriptor.preservesOrder &&
            !stage.descriptor.mutatesCoordinates &&
            !stage.descriptor.fusion.hasWhere &&
            stage.descriptor.fusion.whereMerge == WhereMergeMode::NotApplicable;
        if (!declaredPredicate)
            // B0163: distinguish "not implemented natively, so properties are
            // unknown" from "declares a cardinality change and does not
            // qualify". Both refuse, but only the second is a statement about
            // cardinality, and conflating them sent three separate diagnoses
            // in one session after the wrong target.
            return stage.native
                       ? RuntimePlacementUnavailableReason::
                             NonCardinalityPreservingStage
                       : RuntimePlacementUnavailableReason::UnsupportedStage;
    }
    // B0205: a plan with no device-capable stage can never be placed, and
    // deciding that needs only the compiled plan.
    //
    // This function already exists to decline for free rather than pay runtime
    // initialisation first (B0156/D0214), but it only judged topology and
    // cardinality. A linear, cardinality-preserving pipeline whose filters
    // simply have no device implementation -- `filters.smrf` is the measured
    // case -- passed every check here and then paid full CUDA device and
    // profile discovery before `buildRuntimePlacement` reached its own
    // `hasDeviceCandidate` test and returned this same reason. B0205 measured
    // that at ~172 ms on a 1M-point SMRF pipeline.
    //
    // `buildRuntimePlacement` returns `NoDeviceCandidate` for exactly this
    // plan, so the contract above -- that a non-`None` reason is what it would
    // have returned -- is preserved.
    if (std::none_of(plan.stages().begin(), plan.stages().end(),
                     [](const PlannedStage& stage)
                     {
                         return stage.native &&
                                stage.preferredResidency == MemoryKind::Device;
                     }))
        return RuntimePlacementUnavailableReason::NoDeviceCandidate;

    // B0206: a device stage with no calibration model can never be placed
    // either, and that too is decidable from the compiled plan.
    //
    // `filters.smrf` is the measured case: native, device-preferred, so it
    // passes every check above, but with no placement model. The refusal
    // therefore happened inside `buildRuntimePlacement` after full CUDA device
    // and profile discovery, which B0205 measured at ~170 ms spent to reach a
    // conclusion the plan already determined.
    //
    // `buildRuntimePlacement` excuses an empty model only for the two direct
    // composition regions, and those exceptions are gated on executor facts
    // this function does not have. B0205 tried approximating them with their
    // plan-only region lookups and broke two process gates, because a drifted
    // shape that misses the plan-only matcher then gets a different -- and
    // less accurate -- reason than the facts-gated path would give it.
    //
    // So this guard is coarse on purpose: if the plan contains any stage type
    // those regions are built from, defer entirely and let
    // `buildRuntimePlacement` decide with the facts in hand. Every other plan,
    // including every SMRF pipeline, refuses for free.
    const auto participatesInDirectComposition = [](const PlannedStage& stage)
    {
        return stage.descriptor.type == "filters.outlier" ||
               stage.descriptor.type == "filters.nndistance" ||
               stage.descriptor.type == "filters.hag_nn" ||
               stage.descriptor.type == "filters.hag_delaunay" ||
               stage.descriptor.type == "filters.radialdensity" ||
               stage.descriptor.type == "filters.sort" ||
               stage.descriptor.type == "filters.skewnessbalancing";
    };
    if (std::none_of(plan.stages().begin(), plan.stages().end(),
                     participatesInDirectComposition))
        for (const PlannedStage& stage : plan.stages())
            if (stage.native &&
                stage.preferredResidency == MemoryKind::Device &&
                stage.descriptor.placementModel.empty())
                return RuntimePlacementUnavailableReason::
                    MissingCalibrationModel;
    return RuntimePlacementUnavailableReason::None;
}

RuntimePlacementResult
buildRuntimePlacement(const Plan& plan, const RuntimePlacementFacts& facts,
                      const PlacementCalibrationProfile& profile)
{
    if (!isExactProfile(profile))
        return unavailable(RuntimePlacementUnavailableReason::ProfileNotExact);
    if (!validFacts(plan, facts))
        return unavailable(
            RuntimePlacementUnavailableReason::InvalidRuntimeFacts);
    if (const RuntimePlacementUnavailableReason structural =
            planStructureRefusal(plan);
        structural != RuntimePlacementUnavailableReason::None)
        return unavailable(structural);

    const std::vector<PlannedStage>& stages = plan.stages();

    const std::optional<std::size_t> eigenFamilyRegion =
        measuredEigenFamilyRegion(plan);
    const std::optional<std::size_t> rankOptimalRegion =
        measuredRankOptimalRegion(plan);
    const auto* terminalWriter =
        stages.empty() ? nullptr
                       : std::get_if<FileStagePlan>(&stages.back().payload);
    // D0279: a profile that measured the extra_dims=all publication of the
    // normal/covariance composition (`normal-covariancefeatures-compose-
    // extradims`, calibrated by `gpupal calibrate` on plain and compressed
    // sinks) lifts B0224's one-layout bound. B0280 did that for shipped GPU
    // classes; B0280/B0281 then qualified and refreshed the embedded
    // reference profile after its higher precedence shadowed that evidence.
    const bool extraDimsAllComposeModel =
        terminalWriter && terminalWriter->extraDimensionsAll &&
        placementStageCalibration(
            profile, "normal-covariancefeatures-compose-extradims") != nullptr;
    const std::optional<std::size_t> normalCovarianceRegion =
        measuredNormalCovarianceRegion(plan, facts, extraDimsAllComposeModel);
    // B0224 qualifies compressed extra_dims=all publication only for the
    // measured normal/covariance row above.  Functional writer admission must
    // not let the older eigen-family, rank/optimal, or per-stage models inherit
    // an unmeasured LAZ encode/layout residual.  Existing uncompressed direct
    // publishers retain their separately proved envelopes.
    if (terminalWriter && terminalWriter->extraDimensionsAll &&
        facts.outputCompressed && !normalCovarianceRegion &&
        !facts.directHagNnBoundaryExecutor &&
        !facts.directHagDelaunayBoundaryExecutor &&
        !facts.directSkewnessBoundaryExecutor &&
        !facts.directSortBoundaryExecutor)
        return unavailable(
            RuntimePlacementUnavailableReason::MixedCalibrationModels);
    if (facts.exactDirectRadiusAssignExecutor && facts.inputRecordBytes != 36U)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directRadiusAssignBoundaryRegion =
        facts.directRadiusAssignBoundaryExecutor
            ? measuredDirectRadiusAssignRegion(plan)
            : std::nullopt;
    if (facts.directRadiusAssignBoundaryExecutor &&
        !directRadiusAssignBoundaryRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    if (facts.directNeighborClassifierBoundaryExecutor &&
        (facts.inputPointFormat != 7U || facts.inputCompressed ||
         facts.outputCompressed || facts.inputRecordBytes != 36U ||
         facts.outputRecordBytes != 36U))
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directNeighborClassifierRegion =
        facts.directNeighborClassifierBoundaryExecutor
            ? measuredDirectNeighborClassifierRegion(plan)
            : std::nullopt;
    if (facts.directNeighborClassifierBoundaryExecutor &&
        !directNeighborClassifierRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directHagNnCountOneRegion =
        facts.directHagNnBoundaryExecutor
            ? measuredDirectHagNnCountOneRegion(plan, facts)
            : std::nullopt;
    if (facts.directHagNnBoundaryExecutor && !directHagNnCountOneRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directHagDelaunayCountThreeRegion =
        facts.directHagDelaunayBoundaryExecutor
            ? measuredDirectHagDelaunayCountThreeRegion(plan, facts)
            : std::nullopt;
    if (facts.directHagDelaunayBoundaryExecutor &&
        !directHagDelaunayCountThreeRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directSkewnessRegion =
        facts.directSkewnessBoundaryExecutor
            ? measuredDirectSkewnessRegion(plan, facts)
            : std::nullopt;
    if (facts.directSkewnessBoundaryExecutor && !directSkewnessRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directSortRegion =
        facts.directSortBoundaryExecutor ? measuredDirectSortRegion(plan, facts)
                                         : std::nullopt;
    if (facts.directSortBoundaryExecutor && !directSortRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    if (facts.directOutlierNnDistanceBoundaryExecutor &&
        facts.inputRecordBytes != 36U)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directOutlierNnDistanceRegion =
        facts.directOutlierNnDistanceBoundaryExecutor
            ? measuredDirectOutlierNnDistanceRegion(plan)
            : std::nullopt;
    if (facts.directOutlierNnDistanceBoundaryExecutor &&
        !directOutlierNnDistanceRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    if (facts.directRadiusOutlierRadialDensityBoundaryExecutor &&
        (facts.inputPointFormat != 7U || facts.inputRecordBytes != 36U))
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directRadiusOutlierRadialDensityRegion =
        facts.directRadiusOutlierRadialDensityBoundaryExecutor
            ? measuredDirectRadiusOutlierRadialDensityRegion(plan)
            : std::nullopt;
    if (facts.directRadiusOutlierRadialDensityBoundaryExecutor &&
        !directRadiusOutlierRadialDensityRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    if (facts.directApproximateCoplanarOutputExecutor &&
        (facts.inputPointFormat != 7U || facts.inputRecordBytes != 36U))
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    const std::optional<std::size_t> directApproximateCoplanarRegion =
        facts.directApproximateCoplanarOutputExecutor
            ? measuredDirectApproximateCoplanarRegion(plan)
            : std::nullopt;
    if (facts.directApproximateCoplanarOutputExecutor &&
        !directApproximateCoplanarRegion)
        return unavailable(
            RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    std::vector<std::string_view> regionModels(plan.summary().residentRegions,
                                               std::string_view{});
    std::vector<AssignProgram> pointPrograms(plan.summary().residentRegions);
    std::vector<std::size_t> pointProgramStages(plan.summary().residentRegions,
                                                0U);
    std::vector<std::size_t> ferryCopies(plan.summary().residentRegions, 0U);
    std::vector<std::size_t> ferryStages(plan.summary().residentRegions, 0U);
    std::vector<std::size_t> predicateStages(plan.summary().residentRegions,
                                             0U);
    std::vector<bool> predicatesExact(plan.summary().residentRegions, true);
    bool hasDeviceCandidate = false;
    for (const PlannedStage& stage : stages)
    {
        if (!stage.native || stage.preferredResidency != MemoryKind::Device)
            continue;
        hasDeviceCandidate = true;
        if (stage.residentRegion == NoResidentRegion ||
            stage.residentRegion >= regionModels.size() ||
            (stage.descriptor.placementModel.empty() &&
             (!directOutlierNnDistanceRegion ||
              stage.residentRegion != *directOutlierNnDistanceRegion) &&
             (!directRadiusOutlierRadialDensityRegion ||
              stage.residentRegion !=
                  *directRadiusOutlierRadialDensityRegion) &&
             (!directHagNnCountOneRegion ||
              stage.residentRegion != *directHagNnCountOneRegion) &&
             (!directHagDelaunayCountThreeRegion ||
              stage.residentRegion != *directHagDelaunayCountThreeRegion) &&
             (!directSkewnessRegion ||
              stage.residentRegion != *directSkewnessRegion) &&
             (!directSortRegion || stage.residentRegion != *directSortRegion)))
            return unavailable(
                RuntimePlacementUnavailableReason::MissingCalibrationModel);
        std::string_view& model = regionModels[stage.residentRegion];
        // Point-program stages may join a region anchored by one measured
        // non-point-program model: D0055 charges that residual once at the
        // region's first device stage and the point-program remainder is
        // zero-incremental. Two distinct measured models stay a mixed-model
        // rejection.
        if (!model.empty() && model != stage.descriptor.placementModel &&
            model != "point-program" &&
            stage.descriptor.placementModel != "point-program" &&
            (!eigenFamilyRegion ||
             stage.residentRegion != *eigenFamilyRegion) &&
            (!rankOptimalRegion ||
             stage.residentRegion != *rankOptimalRegion) &&
            (!normalCovarianceRegion ||
             stage.residentRegion != *normalCovarianceRegion))
            return unavailable(
                RuntimePlacementUnavailableReason::MixedCalibrationModels);
        if (model.empty() || model == "point-program")
            model = stage.descriptor.placementModel;
        if (stage.descriptor.placementModel == "point-program")
        {
            ++pointProgramStages[stage.residentRegion];
            if (const auto* assignments =
                    std::get_if<AssignProgram>(&stage.payload))
                appendAssignments(pointPrograms[stage.residentRegion],
                                  *assignments);
            else if (const auto* ferry =
                         std::get_if<FerryProgram>(&stage.payload))
            {
                ++ferryStages[stage.residentRegion];
                ferryCopies[stage.residentRegion] += ferry->copies.size();
                appendFerry(pointPrograms[stage.residentRegion], *ferry);
            }
            else if (const auto* predicate =
                         std::get_if<PredicateProgram>(&stage.payload))
            {
                ++predicateStages[stage.residentRegion];
                predicatesExact[stage.residentRegion] =
                    predicatesExact[stage.residentRegion] &&
                    predicateSupportsExactDevice(*predicate);
            }
            else
                return unavailable(RuntimePlacementUnavailableReason::
                                       OutsideCalibrationEnvelope);
        }
    }
    if (!hasDeviceCandidate)
        return unavailable(
            RuntimePlacementUnavailableReason::NoDeviceCandidate);

    if (eigenFamilyRegion)
        regionModels[*eigenFamilyRegion] = "eigen-family-compose";
    if (rankOptimalRegion)
        regionModels[*rankOptimalRegion] = "rank-optimal-compose";
    if (normalCovarianceRegion)
        regionModels[*normalCovarianceRegion] =
            extraDimsAllComposeModel
                ? "normal-covariancefeatures-compose-extradims"
                : "normal-covariancefeatures-compose";
    if (facts.exactDirectRadiusAssignExecutor)
        regionModels[*directRadiusAssignBoundaryRegion] = "radiusassign-direct";
    if (directNeighborClassifierRegion)
        regionModels[*directNeighborClassifierRegion] =
            "neighborclassifier-direct-compose";
    if (directHagNnCountOneRegion)
        regionModels[*directHagNnCountOneRegion] =
            "hag-nn-count1-direct-compose";
    if (directHagDelaunayCountThreeRegion)
        regionModels[*directHagDelaunayCountThreeRegion] =
            "hag-delaunay-count3-direct-compose";
    if (directSkewnessRegion)
        regionModels[*directSkewnessRegion] = "skewness-direct-compose";
    if (directSortRegion)
        regionModels[*directSortRegion] = "sort-direct-compose";
    if (directOutlierNnDistanceRegion)
        regionModels[*directOutlierNnDistanceRegion] =
            "outlier-nndistance-direct-compose";
    if (directRadiusOutlierRadialDensityRegion)
        regionModels[*directRadiusOutlierRadialDensityRegion] =
            "radius-outlier-radialdensity-direct-compose";
    if (directApproximateCoplanarRegion)
        regionModels[*directApproximateCoplanarRegion] =
            "approximatecoplanar-direct-compose";

    for (std::size_t region = 0U; region < regionModels.size(); ++region)
    {
        if (regionModels[region] != "point-program")
            continue;
        // At most one exact declared predicate may join a fused point-program
        // region. A region with a predicate resolves to the measured
        // ordered-point-program residual (D0064); the assignment content must
        // stay inside the bounded program envelope that the measured class
        // used.
        if (predicateStages[region] > 1U || !predicatesExact[region])
            return unavailable(
                RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
        if (predicateStages[region] == 1U)
        {
            if (!fusedPointProgramCalibrationEnvelope(pointPrograms[region]))
                return unavailable(RuntimePlacementUnavailableReason::
                                       OutsideCalibrationEnvelope);
            regionModels[region] = "ordered-point-program";
        }
        else if (pointProgramStages[region] == 1U &&
                 ferryStages[region] == 1U && ferryCopies[region] == 1U)
            regionModels[region] = "simple-ferry";
        else if (fusedPointProgramCalibrationEnvelope(pointPrograms[region]))
            regionModels[region] = "fused-point-program";
        else
            return unavailable(
                RuntimePlacementUnavailableReason::OutsideCalibrationEnvelope);
    }

    RuntimePlacementResult result;
    result.unavailableReason = RuntimePlacementUnavailableReason::None;
    result.request.stageInputPointCounts.assign(stages.size(),
                                                facts.inputPointCount);
    result.request.stageOutputPointCounts.assign(stages.size(),
                                                 facts.inputPointCount);
    result.request.stageInputPointCounts.front() = 0U;
    const std::size_t capacity =
        (std::min)(facts.inputPointCount, facts.tilePointCapacity);
    result.request.stagePointCapacities.assign(stages.size(), capacity);
    result.request.stageScratchBytes = facts.stageScratchBytes;
    result.request.stageAdditionalSynchronizations =
        facts.stageAdditionalSynchronizations;
    result.request.boundaryExecutionFacts = facts.boundaryExecutionFacts;
    result.request.executorLaneCount = facts.executorLaneCount;
    result.request.intrinsicSingleLaneExecutor =
        facts.directRadiusAssignBoundaryExecutor ||
        facts.directNeighborClassifierBoundaryExecutor ||
        facts.directHagNnBoundaryExecutor ||
        facts.directHagDelaunayBoundaryExecutor ||
        facts.directSkewnessBoundaryExecutor ||
        facts.directSortBoundaryExecutor ||
        facts.directOutlierNnDistanceBoundaryExecutor ||
        facts.directRadiusOutlierRadialDensityBoundaryExecutor;
    if (directHagNnCountOneRegion || directHagDelaunayCountThreeRegion ||
        directSkewnessRegion || directSortRegion)
    {
        const std::size_t bytesPerPoint =
            directHagNnCountOneRegion
                ? HagNnCountOneExactDevicePeakBytesPerPoint
            : directHagDelaunayCountThreeRegion
                ? HagDelaunayCountThreeExactDevicePeakBytesPerPoint
            : directSkewnessRegion ? SkewnessExactDevicePeakBytesPerPoint
                                   : OrderingExactDevicePeakBytesPerPoint;
        result.request.executorUntiledDeviceBytes =
            facts.inputPointCount * bytesPerPoint;
        result.request.executorPeakDeviceBytes =
            facts.inputPointCount * bytesPerPoint;
    }
    result.request.inputRecordBytes = facts.inputRecordBytes;
    result.request.outputRecordBytes = facts.outputRecordBytes;
    result.request.fallbackRecordBytes = facts.fallbackRecordBytes;
    result.request.deviceMemoryBudgetBytes = facts.deviceMemoryBudgetBytes;
    result.request.additionalSynchronizations =
        facts.additionalSynchronizations;
    result.request.cudaContextWarm = facts.cudaContextWarm;

    for (std::size_t region = 0U; region < regionModels.size(); ++region)
    {
        if (regionModels[region].empty())
            return unavailable(
                RuntimePlacementUnavailableReason::MissingCalibrationModel);
        if (!placementStageCalibration(profile, regionModels[region]))
            return unavailable(
                RuntimePlacementUnavailableReason::UnknownCalibrationModel);
        result.regionCalibrations.push_back(
            {.residentRegion = region, .model = regionModels[region]});
    }
    if (!applyPlacementRegionCalibrations(
            plan, profile, result.regionCalibrations, result.request))
        return unavailable(
            RuntimePlacementUnavailableReason::UnknownCalibrationModel);

    try
    {
        result.estimate =
            evaluatePlanPlacement(plan, result.request, profile.coefficients);
    }
    catch (const std::exception&)
    {
        return unavailable(RuntimePlacementUnavailableReason::EvaluationFailed);
    }
    return result;
}

} // namespace pdg
