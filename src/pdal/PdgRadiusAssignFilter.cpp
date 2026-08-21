#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <filters/private/DimRange.hpp>
#include <filters/private/expr/AssignStatement.hpp>
#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/PluginHelper.hpp>
#include <pdal/PointView.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{

// Exact host compatibility wrapper for the pinned RadiusAssignFilter.  This
// deliberately retains the upstream two-pass selection/update order, OR domain
// semantics, strict nanoflann radius predicate, and diagnostic spelling.  The
// source is retained under PDAL's BSD-3-Clause license in this upstream-backed
// fork; the CUDA radius-query client is layered onto this wrapper separately.
class PdgRadiusAssignFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.radiusassign";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("src_domain",
                 "Selects which points will be subject to radius-based "
                 "neighbors search",
                 m_sourceDomain);
        args.add("reference_domain",
                 "Selects which points will be considered as potential "
                 "neighbors",
                 m_referenceDomain);
        args.add("radius", "Distance of neighbors to consult", m_radius);
        args.add("update_expression",
                 "Value to assign to dimension of points of src_domain that "
                 "have at least one neighbor in reference domain based on "
                 "expression.",
                 m_updates);
        args.add("is3d", "Search in 3d", m_search3d, false);
        args.add("max2d_above",
                 "if search in 2d : upward maximum distance in Z for "
                 "potential neighbors (corresponds to a search in a cylinder "
                 "with a height = max2d_above above the source point). Values "
                 "< 0 mean infinite height",
                 m_max2dAbove, -1.0);
        args.add("max2d_below",
                 "if search in 2d : downward maximum distance in Z for "
                 "potential neighbors (corresponds to a search in a cylinder "
                 "with a height = max2d_below below the source point). Values "
                 "< 0 mean infinite height",
                 m_max2dBelow, -1.0);
        args.add("pdg_region_id", "Internal resident-region identifier",
                 m_region.id, std::uint64_t(0))
            .setHidden();
        args.add("pdg_region_radius",
                 "Internal resident-region radius envelope",
                 m_region.maximumRadius, 0.0)
            .setHidden();
        args.add("pdg_region_dimensions",
                 "Internal resident-region spatial dimensions",
                 m_region.dimensions, std::uint32_t(3))
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

    void initialize() override
    {
        if (m_radius <= 0)
            throwError("Invalid 'radius' option: " + std::to_string(m_radius) +
                       ", must be > 0");
        if (m_updates.empty())
            throwError("Empty 'update_epxression' option, must be set to apply "
                       "any change on the data");
    }

    void prepared(PointTableRef table) override
    {
        Utils::StatusWithReason status;
        PointLayoutPtr layout(table.layout());
        status = m_sourceDomain.prepare(layout);
        if (!status)
            throwError("Invalid dimension name in 'src_domain': " +
                       status.what());
        status = m_referenceDomain.prepare(layout);
        if (!status)
            throwError("Invalid dimension name in 'reference_domain': " +
                       status.what());
        for (expr::AssignStatement& update : m_updates)
        {
            status = update.prepare(layout);
            if (!status)
                throwError("Invalid assignment expression in "
                           "'update_expression' option: " +
                           status.what());
        }
    }

    void ready(PointTableRef) override
    {
        m_pointsToUpdate.clear();
    }

    void markWithoutSourceDomain(PointRef& source)
    {
        PointIdList neighbors;
        if (m_search3d)
            neighbors =
                m_referenceView->build3dIndex().radius(source, m_radius);
        else
            neighbors =
                m_referenceView->build2dIndex().radius(source, m_radius);
        if (neighbors.empty())
            return;

        if (!m_search3d && (m_max2dBelow >= 0 || m_max2dAbove >= 0))
        {
            const double sourceZ = source.getFieldAs<double>(Dimension::Id::Z);
            bool take = false;
            for (PointId neighbor : neighbors)
            {
                const double referenceZ =
                    m_referenceView->point(neighbor).getFieldAs<double>(
                        Dimension::Id::Z);
                if (m_max2dAbove >= 0 && referenceZ > sourceZ &&
                    referenceZ - sourceZ > m_max2dAbove)
                    continue;
                if (m_max2dBelow >= 0 && sourceZ > referenceZ &&
                    sourceZ - referenceZ > m_max2dBelow)
                    continue;
                take = true;
                break;
            }
            if (!take)
                return;
        }
        m_pointsToUpdate.push_back(source.pointId());
    }

    void mark(PointRef& source)
    {
        if (m_sourceDomain.empty())
            markWithoutSourceDomain(source);
        for (const DimRange& range : m_sourceDomain.ranges())
        {
            if (range.valuePasses(source.getFieldAs<double>(range.m_id)))
            {
                markWithoutSourceDomain(source);
                break;
            }
        }
    }

    void filter(PointView& view) override
    {
        m_region.radiusIndex = true;
        m_region.dimensions = m_search3d ? 3U : 2U;
        if (m_region.maximumRadius <= 0.0)
            m_region.maximumRadius = m_radius;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda = pdg_detail::tryCudaRadiusAssign(
                view, m_radius, m_search3d, m_max2dAbove, m_max2dBelow,
                m_sourceDomain, m_referenceDomain, m_updates, m_region,
                /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "radiusassign path was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        const bool usedCuda =
            requestCuda && pdg_detail::tryCudaRadiusAssign(
                               view, m_radius, m_search3d, m_max2dAbove,
                               m_max2dBelow, m_sourceDomain, m_referenceDomain,
                               m_updates, m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid radiusassign path was not "
                       "used");
        if (usedCuda)
            return;

        PointRef point(view, 0);
        m_referenceView = view.makeNew();
        if (m_referenceDomain.empty())
        {
            for (PointId id = 0; id < view.size(); ++id)
                m_referenceView->appendPoint(view, id);
        }
        else
        {
            for (PointId id = 0; id < view.size(); ++id)
            {
                for (const DimRange& range : m_referenceDomain.ranges())
                {
                    point.setPointId(id);
                    if (range.valuePasses(point.getFieldAs<double>(range.m_id)))
                    {
                        m_referenceView->appendPoint(view, id);
                        break;
                    }
                }
            }
        }

        for (PointId id = 0; id < view.size(); ++id)
        {
            point.setPointId(id);
            mark(point);
        }

        for (PointId id : m_pointsToUpdate)
        {
            point.setPointId(id);
            for (expr::AssignStatement& update : m_updates)
            {
                if (update.conditionalExpr().eval(point))
                    point.setField(update.identExpr().eval(),
                                   update.valueExpr().eval(point));
            }
        }
    }

    DimRangeList m_referenceDomain;
    DimRangeList m_sourceDomain;
    double m_radius = 0.0;
    std::vector<expr::AssignStatement> m_updates;
    bool m_search3d = false;
    double m_max2dAbove = -1.0;
    double m_max2dBelow = -1.0;
    PointViewPtr m_referenceView;
    PointIdList m_pointsToUpdate;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridRadiusAssignStage),
                                     "Internal exact PDG radiusassign filter",
                                     ""};

CREATE_STATIC_STAGE(PdgRadiusAssignFilter, s_info)

} // namespace pdal
