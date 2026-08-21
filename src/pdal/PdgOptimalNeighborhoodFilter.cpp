#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace pdal
{

// Exact compatibility wrapper for the pinned OptimalNeighborhood filter:
// the same distance-sorted max_k adjacency, Welford covariance sweep, and
// host-transcendental eigenentropy minimization, on the host or through
// the shared-index CUDA client.
class PdgOptimalNeighborhoodFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.optimalneighborhood";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("min_k", "Minimum k-Nearest Neighbors", m_minimumK,
                 (point_count_t)10);
        args.add("max_k", "Maximum k-Nearest Neighbors", m_maximumK,
                 (point_count_t)14);
        args.add("pdg_region_id", "Internal resident-region identifier",
                 m_region.id, std::uint64_t(0))
            .setHidden();
        args.add("pdg_region_neighbors",
                 "Internal resident-region neighbor envelope",
                 m_region.maximumNeighbors, std::uint32_t(0))
            .setHidden();
        args.add("pdg_region_reuse", "Internal resident-region reuse marker",
                 m_region.reuseExpected, false)
            .setHidden();
        args.add("pdg_region_last", "Internal resident-region final marker",
                 m_region.last, true)
            .setHidden();
        args.add("pdg_region_terminal_sink",
                 "Internal resident-region terminal-sink marker",
                 m_region.terminalSink, false)
            .setHidden();
        args.add("pdg_resident_context",
                 "Internal planner-owned resident execution marker",
                 m_residentContext, false)
            .setHidden();
        args.add("pdg_execution_region",
                 "Internal planner-owned execution region identifier",
                 m_executionRegion, std::uint64_t(0))
            .setHidden();
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDim(Dimension::Id::OptimalKNN);
        layout->registerDim(Dimension::Id::OptimalRadius);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        for (PointRef point : view)
        {
            std::uint64_t optimalK = 0U;
            double optimalRadius = 0.0;
            pdg_detail::computeOptimalExact(
                view, index, point.pointId(),
                static_cast<std::uint32_t>(m_minimumK),
                static_cast<std::uint32_t>(m_maximumK), optimalK,
                optimalRadius);
            point.setField(Dimension::Id::OptimalKNN, optimalK);
            point.setField(Dimension::Id::OptimalRadius, optimalRadius);
        }
    }

    void filter(PointView& view) override
    {
        const bool eligible =
            m_minimumK >= 1U && m_minimumK <= m_maximumK && m_maximumK <= 64U;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                eligible && pdg_detail::tryCudaOptimalNeighborhoodColumns(
                                view, static_cast<std::uint32_t>(m_minimumK),
                                static_cast<std::uint32_t>(m_maximumK),
                                m_region, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "optimalneighborhood path was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        const bool usedCuda =
            requestCuda && eligible &&
            pdg_detail::tryCudaOptimalNeighborhoodColumns(
                view, static_cast<std::uint32_t>(m_minimumK),
                static_cast<std::uint32_t>(m_maximumK), m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid optimalneighborhood path "
                       "was not used");
        if (!usedCuda)
            computeKd(view);
    }

    point_count_t m_minimumK = 10U;
    point_count_t m_maximumK = 14U;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridOptimalNeighborhoodStage),
    "Internal exact PDG shared-index optimalneighborhood filter", ""};

CREATE_STATIC_STAGE(PdgOptimalNeighborhoodFilter, s_info)

} // namespace pdal
