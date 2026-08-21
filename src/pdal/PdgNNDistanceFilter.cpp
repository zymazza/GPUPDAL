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
#include <pdal/util/Utils.hpp>

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
enum class DistanceMode
{
    Kth,
    Average
};

std::istream& operator>>(std::istream& input, DistanceMode& mode)
{
    std::string value;
    input >> value;
    value = Utils::tolower(value);
    if (value == "kth")
        mode = DistanceMode::Kth;
    else if (value == "avg")
        mode = DistanceMode::Average;
    else
        input.setstate(std::ios_base::failbit);
    return input;
}

std::ostream& operator<<(std::ostream& output, DistanceMode mode)
{
    if (mode == DistanceMode::Kth)
        output << "kth";
    else
        output << "avg";
    return output;
}
} // unnamed namespace

// Exact compatibility wrapper for PDAL's k-th/average nearest-neighbor
// distance stage. Both modes consume the planner-owned resident spatial index;
// the original KD3Index implementation remains the compatibility fallback.
class PdgNNDistanceFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.nndistance";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("mode", "Distance computation mode (kth, avg)", m_mode,
                 DistanceMode::Kth);
        args.add("k", "k neighbors", m_k, std::size_t(10));
        args.add("pdg_auto_cuda", "Internal automatic CUDA selection marker",
                 m_autoCuda, false)
            .setHidden();
        args.add("pdg_region_id", "Internal resident-region identifier",
                 m_region.id, std::uint64_t(0))
            .setHidden();
        args.add("pdg_region_neighbors",
                 "Internal resident-region neighbor envelope",
                 m_region.maximumNeighbors, std::uint32_t(0))
            .setHidden();
        args.add("pdg_region_gather_neighbors",
                 "Internal resident max-k projection width",
                 m_region.gatherNeighbors, std::uint32_t(0))
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
    }

    void computeKd(PointView& view) const
    {
        KD3Index& index = view.build3dIndex();
        const std::size_t neighbors = m_k + 1U;
        log()->get(LogLevel::Debug) << "Computing k-distances...\n";
        for (PointId point = 0; point < view.size(); ++point)
        {
            PointIdList indices(neighbors);
            std::vector<double> squaredDistances(neighbors);
            index.knnSearch(point, neighbors, &indices, &squaredDistances);
            double value = 0.0;
            if (m_mode == DistanceMode::Kth)
                value = std::sqrt(squaredDistances[neighbors - 1U]);
            else
            {
                for (std::size_t neighbor = 1U; neighbor < neighbors;
                     ++neighbor)
                    value += std::sqrt(squaredDistances[neighbor]);
                value /= static_cast<double>(neighbors - 1U);
            }
            view.setField(Dimension::Id::NNDistance, point, value);
        }
    }

    void filter(PointView& view) override
    {
        const bool eligible = m_k >= 1U && m_k < 64U;
        const pdg::KnnDistanceMode mode = m_mode == DistanceMode::Kth
                                              ? pdg::KnnDistanceMode::Kth
                                              : pdg::KnnDistanceMode::Average;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda =
                eligible && pdg_detail::tryCudaNnDistanceColumns(
                                view, static_cast<std::uint32_t>(m_k) + 1U,
                                m_region, mode, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "nndistance path was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requireAutomatic =
            std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID");
        const bool automaticCuda =
            m_autoCuda && pdg::automaticLabelNnDistanceCudaDeviceQualified();
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             automaticCuda);
        const bool usedCuda = requestCuda && eligible &&
                              pdg_detail::tryCudaNnDistanceColumns(
                                  view, static_cast<std::uint32_t>(m_k) + 1U,
                                  m_region, mode, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid nndistance path was not used");
        if (requireAutomatic && (!m_autoCuda || !usedCuda))
            throwError(
                "required automatic exact CUDA label/NNDistance hybrid path "
                "was not used");
        if (!usedCuda)
            computeKd(view);
    }

    std::size_t m_k = 10U;
    DistanceMode m_mode = DistanceMode::Kth;
    bool m_autoCuda = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridNnDistanceStage),
    "Internal exact PDG shared-index nearest-neighbor distance filter", ""};

CREATE_STATIC_STAGE(PdgNNDistanceFilter, s_info)

} // namespace pdal
