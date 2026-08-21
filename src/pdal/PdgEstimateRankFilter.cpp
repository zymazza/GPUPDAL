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

// Exact compatibility wrapper for the pinned EstimateRankFilter: the same
// self-inclusive KD3 neighborhoods and the same float-demeaned covariance
// decomposed by Eigen's JacobiSVD with the float-cast threshold, on the
// host or through the shared-index CUDA client.
class PdgEstimateRankFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.estimaterank";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("knn", "k-Nearest Neighbors", m_knn, 8);
        args.add("thresh", "Threshold", m_thresh, 0.01);
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
        layout->registerDim(Dimension::Id::Rank);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        for (PointRef point : view)
        {
            const PointIdList ids =
                index.neighbors(point, static_cast<point_count_t>(m_knn));
            point.setField(Dimension::Id::Rank,
                           pdg_detail::computeRankExact(view, ids, m_thresh));
        }
    }

    void filter(PointView& view) override
    {
        const bool eligible = m_knn >= 3 && m_knn <= 64;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                eligible && pdg_detail::tryCudaEstimateRankColumn(
                                view, static_cast<std::uint32_t>(m_knn),
                                m_region, m_thresh, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "estimaterank path was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        const bool usedCuda = requestCuda && eligible &&
                              pdg_detail::tryCudaEstimateRankColumn(
                                  view, static_cast<std::uint32_t>(m_knn),
                                  m_region, m_thresh, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid estimaterank path was not used");
        if (!usedCuda)
            computeKd(view);
    }

    int m_knn = 8;
    double m_thresh = 0.01;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridEstimateRankStage),
    "Internal exact PDG shared-index estimaterank filter", ""};

CREATE_STATIC_STAGE(PdgEstimateRankFilter, s_info)

} // namespace pdal
