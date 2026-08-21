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
#include <vector>

namespace pdal
{

// Exact compatibility wrapper for the pinned NeighborClassifierFilter: the same
// self-inclusive KD3 neighborhoods and the same float-demeaned covariance
// decomposed by Eigen's JacobiSVD with the float-cast threshold, on the
// host or through the shared-index CUDA client.
class PdgNeighborClassifierFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.neighborclassifier";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("domain",
                 "Selects which points will be subject to KNN-based "
                 "assignment",
                 m_domainSpec);
        args.add("k", "Number of nearest neighbors to consult", m_k)
            .setPositional();
        args.add("candidate", "candidate file name", m_candidateFile);
        args.add("dimension",
                 "Dimension on which votes are calculated (treated as an "
                 "integer).",
                 m_dimensionName, "Classification");
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
        layout->registerDim(Dimension::Id::Classification);
    }

    void computeKd(PointView& view) const
    {
        // Two-phase like upstream: every vote reads the original values and
        // the changed results apply afterwards.
        const KD3Index& index = view.build3dIndex();
        std::vector<std::uint8_t> results(
            static_cast<std::size_t>(view.size()));
        for (PointId point = 0; point < view.size(); ++point)
            results[static_cast<std::size_t>(point)] =
                pdg_detail::computeNeighborVoteExact(
                    view, index, point, static_cast<std::uint32_t>(m_k));
        for (PointId point = 0; point < view.size(); ++point)
            view.setField(Dimension::Id::Classification, point,
                          results[static_cast<std::size_t>(point)]);
    }

    void filter(PointView& view) override
    {
        const bool eligible = m_domainSpec.empty() && m_candidateFile.empty() &&
                              m_dimensionName == "Classification" && m_k >= 1 &&
                              m_k <= 64;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                eligible && pdg_detail::tryCudaNeighborClassifierColumn(
                                view, static_cast<std::uint32_t>(m_k), m_region,
                                /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "neighborclassifier path was not used");
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
            pdg_detail::tryCudaNeighborClassifierColumn(
                view, static_cast<std::uint32_t>(m_k), m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid neighborclassifier path was "
                       "not used");
        if (!usedCuda)
            computeKd(view);
    }

    int m_k = 0;
    StringList m_domainSpec;
    std::string m_candidateFile;
    std::string m_dimensionName;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridNeighborClassifierStage),
    "Internal exact PDG shared-index neighborclassifier filter", ""};

CREATE_STATIC_STAGE(PdgNeighborClassifierFilter, s_info)

} // namespace pdal
