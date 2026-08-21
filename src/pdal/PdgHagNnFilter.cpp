/******************************************************************************
 * Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the conditions in the PDAL BSD
 * license are met. See LICENSE and NOTICE in this source tree.
 ****************************************************************************/

#include <pdg/Hybrid.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace pdal
{
namespace
{
#if PDG_HAS_CUDA
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#endif

double interpolateGround(PointViewPtr ground, const PointIdList& ids,
                         const std::vector<double>& squaredDistances,
                         double maximumDistanceSquared, double defaultZ)
{
    double weights = 0.0;
    double accumulatedZ = 0.0;
    for (std::size_t item = 0U; item < ids.size(); ++item)
    {
        const double z =
            ground->getFieldAs<double>(Dimension::Id::Z, ids[item]);
        const double squaredDistance = squaredDistances[item];
        if (maximumDistanceSquared > 0.0 &&
            squaredDistance > maximumDistanceSquared)
            break;
        const double weight = 1.0 / squaredDistance;
        weights += weight;
        accumulatedZ += weight * z;
    }
    return weights ? accumulatedZ / weights : defaultZ;
}
} // unnamed namespace

// Exact compatibility wrapper for the pinned filters.hag_nn implementation.
// The native count-one-through-64 lane consumes one planner-owned 2D index
// and a masked ground reference domain. Every other option shape, and any
// data-dependent nearest-distance ambiguity, executes the original ordered
// host algorithm.
class PdgHagNnFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.hag_nn";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("count",
                 "The number of points to fetch to determine the ground "
                 "point [Default: 1].",
                 m_count, point_count_t{1U});
        args.add("max_distance",
                 "Ground points beyond this distance will not influence "
                 "nearest neighbor interpolation of height above ground.",
                 m_maximumDistance);
        args.add("allow_extrapolation",
                 "If true and count > 1, allow extrapolation [Default: true].",
                 m_allowExtrapolation, true);
        args.add("class", "Class to use for ground points. [Default: 2]",
                 m_groundClass, ClassLabel::Ground);
        args.add("pdg_region_id", "Internal resident-region identifier",
                 m_region.id, std::uint64_t{0U})
            .setHidden();
        args.add("pdg_region_neighbors",
                 "Internal resident-region neighbor envelope",
                 m_region.maximumNeighbors, std::uint32_t{0U})
            .setHidden();
        args.add("pdg_region_dimensions",
                 "Internal resident-region spatial dimensions",
                 m_region.dimensions, std::uint32_t{2U})
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
                 m_executionRegion, std::uint64_t{0U})
            .setHidden();
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDim(Dimension::Id::HeightAboveGround);
    }

    void prepared(PointTableRef table) override
    {
        if (m_count == 0U)
            throwError("Option 'count' must be a positive integer.");
        if (!table.layout()->hasDim(Dimension::Id::Classification))
            throwError("Missing Classification dimension in input PointView.");
    }

    void prerun(const PointViewSet& views) override
    {
        if (!m_residentContext || views.size() != 1U ||
            !(*views.begin())->empty())
            return;
        pdg_detail::ResidentExecutionContext& context =
            pdg_detail::requireResidentExecutionContext();
        PointView& view = **views.begin();
        const std::size_t region = static_cast<std::size_t>(m_executionRegion);
        context.beginDelegatedRegion(view, region);
        if (m_region.last)
            context.endDelegatedRegion(view, region);
    }

    void computeExact(PointView& view)
    {
        PointViewPtr ground = view.makeNew();
        PointViewPtr nonGround = view.makeNew();
        for (PointId point = 0U; point < view.size(); ++point)
        {
            if (view.getFieldAs<std::uint8_t>(Dimension::Id::Classification,
                                              point) == m_groundClass)
            {
                view.setField(Dimension::Id::HeightAboveGround, point, 0);
                ground->appendPoint(view, point);
            }
            else
                nonGround->appendPoint(view, point);
        }
        BOX2D groundBounds;
        ground->calculateBounds(groundBounds);
        if (ground->empty())
        {
            log()->get(LogLevel::Error)
                << "Input PointView does not have any points classified as "
                   "ground.\n";
            return;
        }

        const KD2Index& index = ground->build2dIndex();
        const double maximumDistanceSquared = std::pow(m_maximumDistance, 2.0);
        for (PointId point = 0U; point < nonGround->size(); ++point)
        {
            PointRef query = nonGround->point(point);
            const double x0 = query.getFieldAs<double>(Dimension::Id::X);
            const double y0 = query.getFieldAs<double>(Dimension::Id::Y);
            const double z0 = query.getFieldAs<double>(Dimension::Id::Z);
            PointIdList ids(m_count);
            std::vector<double> squaredDistances(m_count);
            index.knnSearch(x0, y0, m_count, &ids, &squaredDistances);
            const double x =
                ground->getFieldAs<double>(Dimension::Id::X, ids[0]);
            const double y =
                ground->getFieldAs<double>(Dimension::Id::Y, ids[0]);
            const double z =
                ground->getFieldAs<double>(Dimension::Id::Z, ids[0]);
            double groundZ = z0;
            if ((x0 == x && y0 == y) || ids.size() == 1U)
                groundZ = z;
            else if (!groundBounds.contains(x0, y0) && !m_allowExtrapolation)
                groundZ = z0;
            else
                groundZ = interpolateGround(ground, ids, squaredDistances,
                                            maximumDistanceSquared, z0);
            nonGround->setField(Dimension::Id::HeightAboveGround, point,
                                z0 - groundZ);
        }
    }

    void filter(PointView& view) override
    {
#if PDG_HAS_CUDA
        NvtxRange range("pdg::filters.hag_nn");
#endif
        const bool eligible = m_count >= 1U && m_count <= 64U;
        const pdg::HagNnProgram program{static_cast<std::uint32_t>(m_count),
                                        m_maximumDistance, m_allowExtrapolation,
                                        m_groundClass};
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto executionRegion =
                static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, executionRegion);
            try
            {
                const bool usedCuda =
                    eligible && !view.empty() &&
                    pdg_detail::tryCudaHagNnColumn(view, program, m_region,
                                                   /*requireCuda=*/true);
                if (!usedCuda)
                {
                    computeExact(view);
                    if (!m_region.last && !view.empty() &&
                        !pdg_detail::retainCudaHagHostColumn(
                            view, program.count, m_region,
                            /*requireCuda=*/true))
                        throwError("planner-selected HAG host repair could not "
                                   "retain its resident output column");
                }
                if (m_region.last)
                    context.endDelegatedRegion(view, executionRegion);
            }
            catch (...)
            {
                context.endDelegatedRegion(view, executionRegion);
                throw;
            }
            return;
        }

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             std::getenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID"));
        const bool injectAutomaticR2Decline =
            std::getenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID") &&
            std::getenv("PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE");
        const bool usedCuda = requestCuda && eligible &&
                              !injectAutomaticR2Decline &&
                              pdg_detail::tryCudaHagNnColumn(
                                  view, program, m_region, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid hag_nn path was not used");
        if (std::getenv("PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR") && !usedCuda)
            throwError(
                "required HAG selective exact repair did not execute");
        if (!usedCuda)
            computeExact(view);
    }

    point_count_t m_count = 1U;
    double m_maximumDistance = 0.0;
    bool m_allowExtrapolation = true;
    std::uint8_t m_groundClass = ClassLabel::Ground;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridHagNnStage),
    "Internal exact PDG shared-index height-above-ground filter", ""};

CREATE_STATIC_STAGE(PdgHagNnFilter, s_info)

} // namespace pdal
