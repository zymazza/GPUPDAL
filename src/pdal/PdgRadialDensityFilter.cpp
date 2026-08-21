/******************************************************************************
 * Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the conditions in the PDAL BSD
 * license are met. See LICENSE and NOTICE in this source tree.
 ****************************************************************************/

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdg/Hybrid.hpp>

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace pdal
{

// Exact compatibility wrapper for PDAL's radial-density stage. Radius search
// and the final count scaling both use the shared whole-view/tiled CUDA query;
// every unsupported or unqualified invocation retains the original KD3 path.
class PdgRadialDensityFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.radialdensity";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("radius", "Radius", m_radius, 1.0);
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

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDim(Dimension::Id::RadialDensity);
    }

    void computeKd(PointView& view)
    {
        const KD3Index& index = view.build3dIndex();
        log()->get(LogLevel::Debug) << "Computing densities...\n";
        const double factor =
            1.0 / ((4.0 / 3.0) * 3.14159 * (m_radius * m_radius * m_radius));
        for (PointRef point : view)
        {
            const PointIdList neighbors = index.radius(point, m_radius);
            point.setField(Dimension::Id::RadialDensity,
                           neighbors.size() * factor);
        }
    }

    void filter(PointView& view) override
    {
        m_region.radiusIndex = true;
        m_region.dimensions = 3U;
        if (m_region.maximumRadius <= 0.0)
            m_region.maximumRadius = m_radius;
        const double factor =
            1.0 / ((4.0 / 3.0) * 3.14159 * (m_radius * m_radius * m_radius));
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                pdg_detail::tryCudaResidentRadiusScaledValues(
                    view, m_radius, factor, m_region, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "radialdensity path was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        std::vector<double> values(static_cast<std::size_t>(view.size()));
        const bool usedCuda =
            requestCuda && pdg_detail::tryCudaRadiusScaledValues(
                               view, m_radius, factor, values, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid radialdensity path was not used");
        if (!usedCuda)
        {
            computeKd(view);
            return;
        }

        log()->get(LogLevel::Debug) << "Computing densities...\n";
        for (PointId point = 0; point < view.size(); ++point)
            view.setField(Dimension::Id::RadialDensity, point,
                          values[static_cast<std::size_t>(point)]);
    }

    double m_radius = 1.0;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridRadialDensityStage),
    "Internal exact PDG shared-index radial-density filter", ""};

CREATE_STATIC_STAGE(PdgRadialDensityFilter, s_info)

} // namespace pdal
