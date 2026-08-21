#include <pdg/Hybrid.hpp>
#include <pdg/index/SpatialIndex.hpp>

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

// Exact compatibility wrapper for the bounded kNN eigenvalue family. The
// shared index/eigensolver is force/experimental-only until the RTX 4090
// differential, sanitizer, residency, and break-even gates pass.
class PdgEigenvaluesFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.eigenvalues";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("knn", "k-Nearest neighbors", m_knn, 8);
        args.add("normalize", "Normalize eigenvalues?", m_normalize, false);
        args.add("stride", "Compute features on strided neighbors", m_stride,
                 std::size_t(1));
        m_radiusArg =
            &args.add("radius", "Radius for nearest neighbor search", m_radius);
        args.add("min_k", "Minimum number of neighbors in radius", m_minK, 3);
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
        layout->registerDims({Dimension::Id::Eigenvalue0,
                              Dimension::Id::Eigenvalue1,
                              Dimension::Id::Eigenvalue2});
    }

    void prepared(PointTableRef) override
    {
        if (m_radiusArg->set())
        {
            log()->get(LogLevel::Warning)
                << "Radius has been set. Ignoring knn and stride values."
                << std::endl;
            if (m_radius <= 0.0)
                log()->get(LogLevel::Error)
                    << "Radius must be greater than 0." << std::endl;
        }
        else
        {
            log()->get(LogLevel::Warning)
                << "No radius specified. Proceeding with knn and stride, but "
                   "ignoring min_k."
                << std::endl;
        }
    }

    void writeValues(PointRef& point, const pdg::EigenSystem3d& system) const
    {
        double values[3] = {system.values[0], system.values[1],
                            system.values[2]};
        if (m_normalize)
        {
            const double sum = values[0] + values[1] + values[2];
            values[0] /= sum;
            values[1] /= sum;
            values[2] /= sum;
        }
        point.setField(Dimension::Id::Eigenvalue0, values[0]);
        point.setField(Dimension::Id::Eigenvalue1, values[1]);
        point.setField(Dimension::Id::Eigenvalue2, values[2]);
    }

    void handleResult(PointRef& point,
                      const pdg_detail::EigenResult& result) const
    {
        if (result.status == pdg_detail::EigenStatus::CovarianceZero)
        {
            log()->get(LogLevel::Info)
                << "Skipping point " << point.pointId()
                << ". Covariance matrix is all zeros. This suggests a large "
                   "number of redundant points. Consider using filters.sample "
                   "with a small radius to remove redundant points.\n";
            return;
        }
        if (result.status == pdg_detail::EigenStatus::SolverFailure)
            throwError("Cannot perform eigen decomposition.");
        writeValues(point, result.system);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        for (PointRef point : view)
        {
            PointIdList ids;
            if (m_radiusArg->set())
            {
                ids = index.radius(point, m_radius);
                if (ids.size() < static_cast<std::size_t>(m_minK))
                {
                    log()->get(LogLevel::Info)
                        << "Skipping point " << point.pointId() << ". Found "
                        << ids.size() << " neighbors but required " << m_minK
                        << ".\n";
                    continue;
                }
            }
            else
            {
                ids = index.neighbors(
                    point, static_cast<point_count_t>(m_knn) + 1U, m_stride);
            }
            handleResult(point, pdg_detail::computeEigenSystem(view, ids));
        }
    }

    void
    handleCudaStatus(PointView& view,
                     const pdg_detail::CudaNeighborhoodResults& results) const
    {
        for (PointRef point : view)
        {
            const std::size_t index = static_cast<std::size_t>(point.pointId());
            if ((results.status[index] & pdg::KnnCovarianceZero) != 0U)
            {
                log()->get(LogLevel::Info)
                    << "Skipping point " << point.pointId()
                    << ". Covariance matrix is all zeros. This suggests a "
                       "large number of redundant points. Consider using "
                       "filters.sample with a small radius to remove "
                       "redundant points.\n";
            }
            else if ((results.status[index] & pdg::KnnEigenFailure) != 0U)
                throwError("Cannot perform eigen decomposition.");
        }
    }

    void filter(PointView& view) override
    {
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            std::shared_ptr<const pdg_detail::CudaNeighborhoodResults>
                residentResults;
            const bool eligibleResident = !m_radiusArg->set() &&
                                          m_stride == 1U && m_knn >= 2 &&
                                          m_knn < 64;
            const bool usedCuda =
                eligibleResident &&
                pdg_detail::tryCudaEigenvalueColumns(
                    view, static_cast<std::uint32_t>(m_knn) + 1U, m_region,
                    m_normalize, residentResults, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "eigenvalues path was not used");
            handleCudaStatus(view, *residentResults);
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        std::shared_ptr<const pdg_detail::CudaNeighborhoodResults> results;
        const bool eligible =
            !m_radiusArg->set() && m_stride == 1U && m_knn >= 2 && m_knn < 64;
        const bool usedCuda = requestCuda && eligible &&
                              pdg_detail::tryCudaEigenvalueColumns(
                                  view, static_cast<std::uint32_t>(m_knn) + 1U,
                                  m_region, m_normalize, results, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid eigenvalues path was not used");
        if (usedCuda)
            handleCudaStatus(view, *results);
        else
            computeKd(view);
    }

    int m_knn = 8;
    bool m_normalize = false;
    std::size_t m_stride = 1;
    double m_radius = 0.0;
    Arg* m_radiusArg = nullptr;
    int m_minK = 3;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridEigenvaluesStage),
    "Internal exact PDG shared-index eigenvalues filter", ""};

CREATE_STATIC_STAGE(PdgEigenvaluesFilter, s_info)

} // namespace pdal
