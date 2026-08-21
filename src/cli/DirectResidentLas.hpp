#pragma once

#include <pdg/Scheduler.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace pdg
{
class DimensionRegistry;
class Plan;
struct PlanPlacementEstimate;
} // namespace pdg

namespace pdal
{
class PointView;
}

namespace pdg::cli
{

struct DirectResidentLasResult
{
    std::size_t residentRegion = 0;
    std::vector<std::size_t> stageIds;
    TiledSchedule schedule;
    std::size_t hostToDeviceBytes = 0;
    std::size_t deviceToHostBytes = 0;
    // True when the declared compacting chain ran through the ordered
    // decode/predicate/pack sink. B0005 calibrated only the fused executor,
    // so an ordered result must keep reporting its calibration mismatch.
    bool orderedExecutor = false;
    std::size_t outputPointCount = 0;
};

// Executes only one terminal, whole-program-fused LAS region through the
// calibrated direct CUDA path. Ineligibility has no output side effects and
// returns nullopt. Once a temporary output is owned, execution/publication
// failures are propagated after removing that temporary file.
[[nodiscard]] std::optional<DirectResidentLasResult>
tryDirectResidentLas(const Plan& plan, DimensionRegistry& dimensions,
                     const PlanPlacementEstimate& placement,
                     std::size_t deviceMemoryBudgetBytes);

// Experimental second endpoint envelope: execute the planner-owned resident
// region through PDAL, then publish its final UserData or Classification
// column through the canonical default LAS image. Static support checks have
// no output side effects; publication is atomic and throws after cleaning its
// temporary.
[[nodiscard]] bool supportsDirectResidentLasOutput(
    const Plan& plan, bool allowSingleStageCanonicalFilter = false,
    bool allowPermutedClassification = false, bool allowPermutedSort = false);
[[nodiscard]] bool supportsDirectResidentExtraDoubleOutput(const Plan& plan);
void publishDirectResidentLasOutput(
    const Plan& plan, const pdal::PointView& view,
    bool allowSingleStageCanonicalFilter = false,
    bool allowPermutedClassification = false, bool allowPermutedSort = false);

} // namespace pdg::cli
