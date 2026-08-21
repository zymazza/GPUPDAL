#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace pdg
{

class Plan;
struct PlanPlacementEstimate;

struct ResidentPipelineRewrite
{
    std::string json;
    std::vector<std::size_t> selectedRegions;
    std::vector<std::size_t> selectedStageIds;
    std::size_t pointProgramRegions = 0;
    bool executable = false;
    std::string reason;
    bool preflightAttempted = false;
    bool preflightAccepted = false;
    std::string preflightReason;
};

// Counts planner-owned Grid allocations for the selected resident regions.
// Multiple compatible grid stages in one region share one allocation even
// though PlanSummary::gridBuilds retains the stage-local logical count.
[[nodiscard]] std::size_t
selectedGridBuildCount(const Plan& plan,
                       const PlanPlacementEstimate& placement);

// Materializes a placement decision into the diagnostic in-process point-
// program wrapper. This is observability plumbing, not the planner-owned
// resident executor calibrated by the direct fused-LAS measurements. The
// bounded lane supports only selected point-program regions; every other
// selected region fails closed.
[[nodiscard]] ResidentPipelineRewrite
rewriteResidentPlacement(std::string_view pipelineJson, const Plan& plan,
                         const PlanPlacementEstimate& placement);

} // namespace pdg
