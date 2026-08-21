/******************************************************************************
 * Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the names
 *       of its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ****************************************************************************/

#include <pdg/Hybrid.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace pdal
{

// Exact compatibility wrapper for filters.approximatecoplanar. The shared
// index/eigensystem path remains force/experimental-only until its complete
// process differential, sanitizer, and break-even gates pass.
class PdgApproximateCoplanarFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.approximatecoplanar";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("knn", "k-Nearest Neighbors", m_knn, 8);
        args.add("thresh1", "Threshold 1", m_threshold1, 25.0);
        args.add("thresh2", "Threshold 2", m_threshold2, 6.0);
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
        layout->registerDim(Dimension::Id::Coplanar);
    }

    void logZeroCovariance(PointId point) const
    {
        log()->get(LogLevel::Info)
            << "Skipping point " << point
            << ". Covariance matrix is all zeros. This suggests a large "
               "number of redundant points. Consider using filters.sample "
               "with a small radius to remove redundant points.\n";
    }

    void writeResult(PointRef& point,
                     const pdg_detail::EigenResult& result) const
    {
        if (result.status == pdg_detail::EigenStatus::CovarianceZero)
        {
            logZeroCovariance(point.pointId());
            return;
        }
        if (result.status == pdg_detail::EigenStatus::SolverFailure)
            throwError("Cannot perform eigen decomposition.");
        const std::array<double, 3>& values = result.system.values;
        const bool coplanar = values[1] > m_threshold1 * values[0] &&
                              m_threshold2 * values[1] > values[2];
        point.setField(Dimension::Id::Coplanar, coplanar ? 1U : 0U);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        for (PointRef point : view)
        {
            const PointIdList ids =
                index.neighbors(point, static_cast<point_count_t>(m_knn));
            writeResult(point, pdg_detail::computeEigenSystem(view, ids));
        }
    }

    void handleCudaStatus(PointView& view,
                          const pdg_detail::CudaNeighborhoodResults& results,
                          const std::vector<std::uint8_t>& priorCoplanar) const
    {
        for (PointId point = 0; point < view.size(); ++point)
        {
            const std::uint8_t status =
                results.status[static_cast<std::size_t>(point)];
            if ((status & pdg::KnnCovarianceZero) != 0U)
                logZeroCovariance(point);
            else if ((status & pdg::KnnEigenFailure) != 0U)
            {
                // The device projection has already published every row.
                // Upstream mutates only the successful prefix before throwing,
                // so restore the failing row and untouched suffix exactly.
                for (PointId remaining = point; remaining < view.size();
                     ++remaining)
                    view.setField(
                        Dimension::Id::Coplanar, remaining,
                        priorCoplanar[static_cast<std::size_t>(remaining)]);
                view.invalidateProducts();
                throwError("Cannot perform eigen decomposition.");
            }
        }
    }

    void filter(PointView& view) override
    {
        if (m_residentContext)
        {
            // Planner-selected shared-index execution: the resident context
            // owns the region boundaries and budget, and the exact CUDA path
            // is mandatory rather than best-effort.
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            std::shared_ptr<const pdg_detail::CudaNeighborhoodResults> results;
            std::vector<std::uint8_t> priorCoplanar(
                static_cast<std::size_t>(view.size()));
            for (PointId point = 0; point < view.size(); ++point)
                priorCoplanar[static_cast<std::size_t>(point)] =
                    view.getFieldAs<std::uint8_t>(Dimension::Id::Coplanar,
                                                  point);
            const bool eligible =
                m_knn >= 3 && m_knn <= 64 && view.hasDim(Dimension::Id::X) &&
                view.hasDim(Dimension::Id::Y) &&
                view.hasDim(Dimension::Id::Z) &&
                view.layout()->dimType(Dimension::Id::Coplanar) ==
                    Dimension::Type::Unsigned8;
            const bool usedCuda =
                eligible && pdg_detail::tryCudaApproximateCoplanarColumn(
                                view, static_cast<std::uint32_t>(m_knn),
                                m_region, m_threshold1, m_threshold2, results,
                                /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "approximatecoplanar path was not used");
            handleCudaStatus(view, *results, priorCoplanar);
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requireAutomatic =
            std::getenv("PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA");
        const bool automaticCuda =
            m_autoCuda && m_knn == 8 &&
            view.size() >= pdg::AutomaticApproximateCoplanarCudaMinimumPoints &&
            pdg::automaticApproximateCoplanarCudaDeviceQualified();
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             automaticCuda);
        std::shared_ptr<const pdg_detail::CudaNeighborhoodResults> results;
        const bool eligible =
            m_knn >= 3 && m_knn <= 64 && view.hasDim(Dimension::Id::X) &&
            view.hasDim(Dimension::Id::Y) && view.hasDim(Dimension::Id::Z) &&
            view.layout()->dimType(Dimension::Id::Coplanar) ==
                Dimension::Type::Unsigned8;
        std::vector<std::uint8_t> priorCoplanar;
        if (requestCuda && eligible)
        {
            priorCoplanar.resize(static_cast<std::size_t>(view.size()));
            for (PointId point = 0; point < view.size(); ++point)
                priorCoplanar[static_cast<std::size_t>(point)] =
                    view.getFieldAs<std::uint8_t>(Dimension::Id::Coplanar,
                                                  point);
        }
        const bool usedCuda =
            requestCuda && eligible &&
            !std::getenv(
                "PDG_TEST_APPROXIMATECOPLANAR_RECOVERABLE_CUDA_FAILURE") &&
            pdg_detail::tryCudaApproximateCoplanarColumn(
                view, static_cast<std::uint32_t>(m_knn), m_region, m_threshold1,
                m_threshold2, results, requireCuda);
        if (std::getenv("PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK") &&
            (!requestCuda || usedCuda))
            throwError("required exact approximatecoplanar host fallback did "
                       "not occur");
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid approximatecoplanar path "
                       "was not used");
        if (requireAutomatic && (!m_autoCuda || !usedCuda))
            throwError(
                "required automatic exact CUDA hybrid approximatecoplanar "
                "path was not used");
        if (usedCuda)
            handleCudaStatus(view, *results, priorCoplanar);
        else
            computeKd(view);
    }

    int m_knn = 8;
    double m_threshold1 = 25.0;
    double m_threshold2 = 6.0;
    bool m_autoCuda = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridApproximateCoplanarStage),
    "Internal exact PDG shared-index approximate coplanar filter", ""};

CREATE_STATIC_STAGE(PdgApproximateCoplanarFilter, s_info)

} // namespace pdal
