/******************************************************************************
 * Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the conditions in the PDAL BSD
 * license are met. See LICENSE and NOTICE in this source tree.
 ****************************************************************************/

#include <pdg/Hybrid.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace pdal
{

// Exact compatibility wrapper for filters.lof. The shared-index CUDA path
// computes the three passes on device and repairs the distance-tie closure on
// host; the upstream KD3Index implementation remains the compatibility
// fallback and preserves upstream's per-call self-inclusive minpts increment.
class PdgLofFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.lof";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("minpts", "Minimum number of points", m_minpts,
                 std::size_t(10));
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
        layout->registerDim(Dimension::Id::NNDistance);
        layout->registerDim(Dimension::Id::LocalReachabilityDistance);
        layout->registerDim(Dimension::Id::LocalOutlierFactor);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        const std::size_t neighbors = m_minpts;

        log()->get(LogLevel::Debug) << "Computing k-distances...\n";
        std::vector<PointIdList> ids(view.size());
        std::vector<std::vector<double>> distances(view.size());
        for (PointId point = 0; point < view.size(); ++point)
        {
            PointIdList indices(neighbors);
            std::vector<double> squaredDistances(neighbors);
            index.knnSearch(point, neighbors, &indices, &squaredDistances);
            std::vector<double>& row =
                distances[static_cast<std::size_t>(point)];
            row.resize(neighbors);
            for (std::size_t item = 0; item < neighbors; ++item)
                row[item] = std::sqrt(squaredDistances[item]);
            ids[static_cast<std::size_t>(point)] = std::move(indices);
            view.setField(Dimension::Id::NNDistance, point,
                          row[neighbors - 1U]);
        }

        log()->get(LogLevel::Debug) << "Computing lrd...\n";
        for (PointId point = 0; point < view.size(); ++point)
        {
            double mean = 0.0;
            point_count_t count = 0;
            for (std::size_t item = 0; item < neighbors; ++item)
            {
                const PointId neighbor =
                    ids[static_cast<std::size_t>(point)][item];
                const double kDistance = view.getFieldAs<double>(
                    Dimension::Id::NNDistance, neighbor);
                const double reachability =
                    (std::max)(kDistance, distances[static_cast<std::size_t>(
                                              point)][item]);
                mean += (reachability - mean) / ++count;
            }
            view.setField(Dimension::Id::LocalReachabilityDistance, point,
                          1.0 / mean);
        }

        log()->get(LogLevel::Debug) << "Computing LOF...\n";
        for (PointId point = 0; point < view.size(); ++point)
        {
            const double density = view.getFieldAs<double>(
                Dimension::Id::LocalReachabilityDistance, point);
            double mean = 0.0;
            point_count_t count = 0;
            for (std::size_t item = 0; item < neighbors; ++item)
            {
                const PointId neighbor =
                    ids[static_cast<std::size_t>(point)][item];
                const double ratio =
                    view.getFieldAs<double>(
                        Dimension::Id::LocalReachabilityDistance, neighbor) /
                    density;
                mean += (ratio - mean) / ++count;
            }
            view.setField(Dimension::Id::LocalOutlierFactor, point, mean);
        }
    }

    bool eligible(const PointView& view) const
    {
        return m_minpts >= 2U && m_minpts <= 64U &&
               view.hasDim(Dimension::Id::X) && view.hasDim(Dimension::Id::Y) &&
               view.hasDim(Dimension::Id::Z) &&
               view.layout()->dimType(Dimension::Id::NNDistance) ==
                   Dimension::Type::Double &&
               view.layout()->dimType(
                   Dimension::Id::LocalReachabilityDistance) ==
                   Dimension::Type::Double &&
               view.layout()->dimType(Dimension::Id::LocalOutlierFactor) ==
                   Dimension::Type::Double;
    }

    void filter(PointView& view) override
    {
        // Upstream increments its minpts member on every filter() call so the
        // k-nearest query includes the query point. The increment is
        // deliberately stateful across views for exact compatibility.
        m_minpts++;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                eligible(view) &&
                pdg_detail::tryCudaLofColumns(
                    view, static_cast<std::uint32_t>(m_minpts), m_region,
                    /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index lof path "
                           "was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        const bool usedCuda = requestCuda && eligible(view) &&
                              pdg_detail::tryCudaLofColumns(
                                  view, static_cast<std::uint32_t>(m_minpts),
                                  m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid lof path was not used");
        if (!usedCuda)
            computeKd(view);
    }

    std::size_t m_minpts = 10U;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridLofStage),
    "Internal exact PDG shared-index local outlier factor filter", ""};

CREATE_STATIC_STAGE(PdgLofFilter, s_info)

} // namespace pdal
