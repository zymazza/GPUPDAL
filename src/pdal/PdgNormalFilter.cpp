#include <pdg/Hybrid.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Eigenvalues>

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
struct NormalValue
{
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double curvature = 0.0;
    std::string message;
};

NormalValue findNormal(const PointView& view, const PointIdList& neighbors)
{
    NormalValue result;
    if (neighbors.size() < 3U)
    {
        result.message = "Not enough neighbors to compute normal.";
        return result;
    }

    const pdg_detail::EigenResult eigen =
        pdg_detail::computeEigenSystem(view, neighbors);
    if (eigen.status == pdg_detail::EigenStatus::CovarianceZero)
    {
        result.message = "Covariance matrix is all zeros. This suggests a "
                         "large number of redundant points.";
        return result;
    }
    if (eigen.status == pdg_detail::EigenStatus::SolverFailure)
    {
        result.message =
            "Cannot perform eigen decomposition during normal calculation.";
        return result;
    }
    const double sum = eigen.system.values[0] + eigen.system.values[1] +
                       eigen.system.values[2];
    result.curvature = sum ? std::fabs(eigen.system.values[0] / sum) : 0.0;
    result.normal =
        Eigen::Vector3d(eigen.system.vectors[0], eigen.system.vectors[3],
                        eigen.system.vectors[6]);
    return result;
}
} // unnamed namespace

// Exact compatibility wrapper for the default kNN normal family. The shared
// index and eigensolver are force/experimental-only until the RTX 4090
// differential, sanitizer, and break-even gates pass.
class PdgNormalFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.normal";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("knn", "k-Nearest Neighbors", m_knn, 8);
        args.add("always_up", "Normals always oriented with positive Z?",
                 m_alwaysUp, true);
        args.add("refine",
                 "Refine normals using minimum spanning tree propagation?",
                 m_refine, false);
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
        layout->registerDims({Dimension::Id::NormalX, Dimension::Id::NormalY,
                              Dimension::Id::NormalZ,
                              Dimension::Id::Curvature});
    }

    void writeNormal(PointRef& point, const Eigen::Vector3d& source,
                     double curvature) const
    {
        Eigen::Vector3d normal = source;
        if (m_alwaysUp && normal[2] < 0.0)
            normal = -normal;
        point.setField(Dimension::Id::NormalX, normal[0]);
        point.setField(Dimension::Id::NormalY, normal[1]);
        point.setField(Dimension::Id::NormalZ, normal[2]);
        point.setField(Dimension::Id::Curvature, curvature);
    }

    void computeKd(PointView& view) const
    {
        KD3Index& index = view.build3dIndex();
        const point_count_t neighbors = static_cast<point_count_t>(m_knn) + 1U;
        for (PointRef point : view)
        {
            const NormalValue result =
                findNormal(view, index.neighbors(point, neighbors));
            if (result.normal.isZero())
            {
                log()->get(LogLevel::Info)
                    << "Skipping point " << point.pointId() << ": "
                    << result.message << "\n";
                continue;
            }
            writeNormal(point, result.normal, result.curvature);
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
                    << ": Covariance matrix is all zeros. This suggests a "
                       "large number of redundant points.\n";
                continue;
            }
            if ((results.status[index] & pdg::KnnEigenFailure) != 0U)
            {
                log()->get(LogLevel::Info)
                    << "Skipping point " << point.pointId()
                    << ": Cannot perform eigen decomposition during normal "
                       "calculation.\n";
                continue;
            }
        }
    }

    void filter(PointView& view) override
    {
        if (m_refine)
            throwError("internal normal replacement does not support refine");
        log()->get(LogLevel::Debug) << "Computing normal vectors\n";
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            std::shared_ptr<const pdg_detail::CudaNeighborhoodResults>
                residentResults;
            const bool usedCuda = pdg_detail::tryCudaNormalColumns(
                view, static_cast<std::uint32_t>(m_knn) + 1U, m_region,
                m_alwaysUp, residentResults, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index normal "
                           "path was not used");
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
        const bool usedCuda =
            requestCuda && pdg_detail::tryCudaNormalColumns(
                               view, static_cast<std::uint32_t>(m_knn) + 1U,
                               m_region, m_alwaysUp, results, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid normal path was not used");
        if (usedCuda)
            handleCudaStatus(view, *results);
        else
            computeKd(view);
    }

    int m_knn = 8;
    bool m_alwaysUp = true;
    bool m_refine = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridNormalStage),
    "Internal exact PDG shared-index normal filter", ""};

CREATE_STATIC_STAGE(PdgNormalFilter, s_info)

} // namespace pdal
