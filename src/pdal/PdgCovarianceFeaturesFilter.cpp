#include <pdg/Hybrid.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace pdal
{
namespace
{
enum class FeatureMode
{
    Raw,
    Sqrt,
    Normalized
};

std::istream& operator>>(std::istream& input, FeatureMode& mode)
{
    std::string value;
    input >> value;
    value = Utils::tolower(value);
    if (value == "raw")
        mode = FeatureMode::Raw;
    else if (value == "sqrt")
        mode = FeatureMode::Sqrt;
    else if (value == "normalized")
        mode = FeatureMode::Normalized;
    else
        input.setstate(std::ios_base::failbit);
    return input;
}

std::ostream& operator<<(std::ostream& output, FeatureMode mode)
{
    if (mode == FeatureMode::Raw)
        output << "raw";
    else if (mode == FeatureMode::Sqrt)
        output << "sqrt";
    else
        output << "normalized";
    return output;
}
} // unnamed namespace

// Exact compatibility wrapper for the shared covariance-feature family. The
// kNN/covariance/eigensolver region is force/experimental-only until the
// device-runtime, sanitizer, residency, and break-even gates pass.
class PdgCovarianceFeaturesFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.covariancefeatures";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("knn", "k-Nearest neighbors", m_knn, 10);
        args.add("threads", "Number of threads used to run this filter",
                 m_threads, 1);
        args.add("feature_set", "Set of features to be computed", m_featureSet,
                 {"dimensionality"});
        args.add("stride", "Compute features on strided neighbors", m_stride,
                 std::size_t(1));
        m_radiusArg =
            &args.add("radius", "Radius for nearest neighbor search", m_radius);
        args.add("min_k", "Minimum number of neighbors in radius", m_minK, 3);
        args.add("mode", "Raw, normalized, or sqrt of eigenvalues", m_mode,
                 FeatureMode::Sqrt);
        args.add("optimized", "Use OptimalKNN or OptimalRadius?", m_optimal,
                 false);
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
        for (std::string feature : m_featureSet)
        {
            feature = Utils::tolower(feature);
            Utils::trim(feature);
            if (feature == "dimensionality")
                m_extraDims.insert(
                    m_extraDims.end(),
                    {Dimension::Id::Linearity, Dimension::Id::Planarity,
                     Dimension::Id::Scattering, Dimension::Id::Verticality});
            else if (feature == "all")
                m_extraDims.insert(
                    m_extraDims.end(),
                    {Dimension::Id::Linearity, Dimension::Id::Planarity,
                     Dimension::Id::Scattering, Dimension::Id::Verticality,
                     Dimension::Id::Omnivariance, Dimension::Id::Anisotropy,
                     Dimension::Id::Eigenentropy, Dimension::Id::EigenvalueSum,
                     Dimension::Id::SurfaceVariation,
                     Dimension::Id::DemantkeVerticality,
                     Dimension::Id::Density});
            else if (feature == "linearity")
                m_extraDims.push_back(Dimension::Id::Linearity);
            else if (feature == "planarity")
                m_extraDims.push_back(Dimension::Id::Planarity);
            else if (feature == "scattering")
                m_extraDims.push_back(Dimension::Id::Scattering);
            else if (feature == "verticality")
                m_extraDims.push_back(Dimension::Id::Verticality);
            else if (feature == "omnivariance")
                m_extraDims.push_back(Dimension::Id::Omnivariance);
            else if (feature == "anisotropy")
                m_extraDims.push_back(Dimension::Id::Anisotropy);
            else if (feature == "eigenentropy")
                m_extraDims.push_back(Dimension::Id::Eigenentropy);
            else if (feature == "eigenvaluesum")
                m_extraDims.push_back(Dimension::Id::EigenvalueSum);
            else if (feature == "surfacevariation")
                m_extraDims.push_back(Dimension::Id::SurfaceVariation);
            else if (feature == "demantkeverticality")
                m_extraDims.push_back(Dimension::Id::DemantkeVerticality);
            else if (feature == "density")
                m_extraDims.push_back(Dimension::Id::Density);
        }
        layout->registerDims(m_extraDims);
    }

    void prepared(PointTableRef table) override
    {
        const PointLayoutPtr layout(table.layout());
        if (std::count(m_extraDims.begin(), m_extraDims.end(),
                       Dimension::Id::Density) &&
            !(layout->hasDim(Dimension::Id::OptimalKNN) &&
              layout->hasDim(Dimension::Id::OptimalRadius)))
            throwError("Density feature requires OptimalKNN and OptimalRadius "
                       "dimensions, which are missing in the input PointView.");
        if (m_optimal && !layout->hasDim(Dimension::Id::OptimalKNN))
            throwError("Missing OptimalKNN dimension in input PointView.");
    }

    void writeFeatures(PointRef& point, const pdg::EigenSystem3d& system) const
    {
        std::array<double, 3> lambda{(std::max)(system.values[2], 0.0),
                                     (std::max)(system.values[1], 0.0),
                                     (std::max)(system.values[0], 0.0)};
        const double sum = std::accumulate(lambda.begin(), lambda.end(), 0.0);
        if (lambda[0] == 0.0)
            throwError("Eigenvalues are all 0. Can't compute local features.");
        if (m_mode == FeatureMode::Sqrt)
            std::transform(lambda.begin(), lambda.end(), lambda.begin(),
                           [](double value) { return std::sqrt(value); });
        else if (m_mode == FeatureMode::Normalized)
            std::transform(lambda.begin(), lambda.end(), lambda.begin(),
                           [sum](double value) { return value / sum; });

        for (Dimension::Id dimension : m_extraDims)
        {
            if (dimension == Dimension::Id::Linearity)
                point.setField(Dimension::Id::Linearity,
                               (lambda[0] - lambda[1]) / lambda[0]);
            if (dimension == Dimension::Id::Planarity)
                point.setField(Dimension::Id::Planarity,
                               (lambda[1] - lambda[2]) / lambda[0]);
            if (dimension == Dimension::Id::Scattering)
                point.setField(Dimension::Id::Scattering,
                               lambda[2] / lambda[0]);
            if (dimension == Dimension::Id::Verticality)
            {
                std::array<double, 3> unary{};
                double norm = 0.0;
                for (std::size_t axis = 0; axis < 3U; ++axis)
                {
                    unary[axis] =
                        lambda[0] * std::fabs(system.vectors[axis * 3U + 2U]) +
                        lambda[1] * std::fabs(system.vectors[axis * 3U + 1U]) +
                        lambda[2] * std::fabs(system.vectors[axis * 3U + 0U]);
                    norm += unary[axis] * unary[axis];
                }
                norm = std::sqrt(norm);
                point.setField(Dimension::Id::Verticality, unary[2] / norm);
            }
            if (dimension == Dimension::Id::Omnivariance)
                point.setField(Dimension::Id::Omnivariance,
                               std::cbrt(lambda[2] * lambda[1] * lambda[0]));
            if (dimension == Dimension::Id::EigenvalueSum)
                point.setField(Dimension::Id::EigenvalueSum, sum);
            if (dimension == Dimension::Id::Eigenentropy)
                point.setField(Dimension::Id::Eigenentropy,
                               -(lambda[2] * std::log(lambda[2]) +
                                 lambda[1] * std::log(lambda[1]) +
                                 lambda[0] * std::log(lambda[0])));
            if (dimension == Dimension::Id::Anisotropy)
                point.setField(Dimension::Id::Anisotropy,
                               (lambda[0] - lambda[2]) / lambda[0]);
            if (dimension == Dimension::Id::SurfaceVariation)
                point.setField(Dimension::Id::SurfaceVariation,
                               lambda[2] / sum);
            if (dimension == Dimension::Id::DemantkeVerticality)
                point.setField(Dimension::Id::DemantkeVerticality,
                               1.0 - std::fabs(system.vectors[2U * 3U + 0U]));
            if (dimension == Dimension::Id::Density)
            {
                const double optimalK =
                    point.getFieldAs<double>(Dimension::Id::OptimalKNN);
                const double optimalRadius =
                    point.getFieldAs<double>(Dimension::Id::OptimalRadius);
                constexpr double Pi = 3.14159265;
                point.setField(Dimension::Id::Density,
                               (optimalK + 1.0) / ((4.0 / 3.0) * Pi *
                                                   std::pow(optimalRadius, 3)));
            }
        }
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
        writeFeatures(point, result.system);
    }

    void computeKd(PointView& view) const
    {
        const KD3Index& index = view.build3dIndex();
        for (PointRef point : view)
        {
            PointIdList ids;
            if (m_optimal)
                ids = index.neighbors(
                    point,
                    point.getFieldAs<std::uint64_t>(Dimension::Id::OptimalKNN),
                    1U);
            else if (m_radiusArg->set())
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
                ids = index.neighbors(
                    point, static_cast<point_count_t>(m_knn) + 1U, m_stride);
            handleResult(point, pdg_detail::computeEigenSystem(view, ids));
        }
    }

    pdg::EigenvalueMode deviceMode() const
    {
        if (m_mode == FeatureMode::Raw)
            return pdg::EigenvalueMode::Raw;
        if (m_mode == FeatureMode::Sqrt)
            return pdg::EigenvalueMode::Sqrt;
        return pdg::EigenvalueMode::Normalized;
    }

    std::uint32_t deviceFeatures() const
    {
        std::uint32_t features = 0;
        for (Dimension::Id dimension : m_extraDims)
        {
            if (dimension == Dimension::Id::Linearity)
                features |= pdg::CovarianceLinearity;
            else if (dimension == Dimension::Id::Planarity)
                features |= pdg::CovariancePlanarity;
            else if (dimension == Dimension::Id::Scattering)
                features |= pdg::CovarianceScattering;
            else if (dimension == Dimension::Id::Verticality)
                features |= pdg::CovarianceVerticality;
            else if (dimension == Dimension::Id::Omnivariance)
                features |= pdg::CovarianceOmnivariance;
            else if (dimension == Dimension::Id::Anisotropy)
                features |= pdg::CovarianceAnisotropy;
            else if (dimension == Dimension::Id::Eigenentropy)
                features |= pdg::CovarianceEigenentropy;
            else if (dimension == Dimension::Id::EigenvalueSum)
                features |= pdg::CovarianceEigenvalueSum;
            else if (dimension == Dimension::Id::SurfaceVariation)
                features |= pdg::CovarianceSurfaceVariation;
            else if (dimension == Dimension::Id::DemantkeVerticality)
                features |= pdg::CovarianceDemantkeVerticality;
        }
        return features;
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
            else if ((results.status[index] & pdg::KnnFeatureInvalid) != 0U)
                throwError("Eigenvalues are all 0. Can't compute local "
                           "features.");
        }
    }

    void filter(PointView& view) override
    {
        log()->get(LogLevel::Debug)
            << "Processing " << view.size() << " points in " << m_threads
            << " threads.\n";
        const bool eligible = !m_optimal && !m_radiusArg->set() &&
                              m_stride == 1U && m_knn >= 2 && m_knn < 64;
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            std::shared_ptr<const pdg_detail::CudaNeighborhoodResults>
                residentResults;
            const bool usedCuda =
                eligible &&
                pdg_detail::tryCudaCovarianceFeatureColumns(
                    view, static_cast<std::uint32_t>(m_knn) + 1U, m_region,
                    deviceMode(), deviceFeatures(), residentResults,
                    /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "covariancefeatures path was not used");
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
            requestCuda && eligible &&
            pdg_detail::tryCudaCovarianceFeatureColumns(
                view, static_cast<std::uint32_t>(m_knn) + 1U, m_region,
                deviceMode(), deviceFeatures(), results, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid covariancefeatures path "
                       "was not used");
        if (usedCuda)
            handleCudaStatus(view, *results);
        else
            computeKd(view);
    }

    int m_knn = 10;
    int m_threads = 1;
    StringList m_featureSet;
    std::vector<Dimension::Id> m_extraDims;
    std::size_t m_stride = 1U;
    double m_radius = 0.0;
    Arg* m_radiusArg = nullptr;
    int m_minK = 3;
    FeatureMode m_mode = FeatureMode::Sqrt;
    bool m_optimal = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    pdg_detail::CudaNeighborhoodRegion m_region;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridCovarianceFeaturesStage),
    "Internal exact PDG shared-index covariance features filter", ""};

CREATE_STATIC_STAGE(PdgCovarianceFeaturesFilter, s_info)

} // namespace pdal
