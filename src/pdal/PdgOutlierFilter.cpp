#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{

namespace
{
constexpr pdg::DimensionId X(pdg::StandardDimension::X);
constexpr pdg::DimensionId Y(pdg::StandardDimension::Y);
constexpr pdg::DimensionId Z(pdg::StandardDimension::Z);

struct Indices
{
    PointIdList inliers;
    PointIdList outliers;
};
} // unnamed namespace

// Exact compatibility wrapper for the pinned OutlierFilter. Radius and
// statistical modes consume the same planner-owned compact spatial index;
// default execution retains nanoflann until same-machine crossover evidence
// qualifies automatic CUDA selection.
class PdgOutlierFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.outlier";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("method", "Method [default: statistical]", m_method,
                 "statistical");
        args.add("min_k", "Minimum number of neighbors in radius", m_minK, 2);
        args.add("radius", "Radius", m_radius, 1.0);
        args.add("mean_k", "Mean number of neighbors", m_meanK, 8);
        args.add("multiplier", "Standard deviation threshold", m_multiplier,
                 2.0);
        args.add("class", "Class to use for noise points", m_class,
                 ClassLabel::LowPoint);
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
        layout->registerDim(Dimension::Id::Classification);
    }

    pdg::PointBatch gather(PointView& view, pdg::MemoryResource& memory) const
    {
        const std::size_t count = static_cast<std::size_t>(view.size());
        pdg::PointBatch batch(
            count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, memory);
        for (pdg::DimensionId dimension : {X, Y, Z})
            batch.materialize(dimension, pdg::DimensionType::Double);
        batch.setSize(count);
        for (PointId point = 0; point < view.size(); ++point)
        {
            const std::size_t offset = static_cast<std::size_t>(point);
            batch.data<double>(X)[offset] =
                view.getFieldAs<double>(Dimension::Id::X, point);
            batch.data<double>(Y)[offset] =
                view.getFieldAs<double>(Dimension::Id::Y, point);
            batch.data<double>(Z)[offset] =
                view.getFieldAs<double>(Dimension::Id::Z, point);
        }
        return batch;
    }

    static Indices classify(const std::vector<std::uint32_t>& counts,
                            int minimumNeighbors)
    {
        Indices indices;
        for (std::size_t point = 0; point < counts.size(); ++point)
        {
            if (static_cast<std::size_t>(counts[point]) >
                static_cast<std::size_t>(minimumNeighbors))
                indices.inliers.push_back(static_cast<PointId>(point));
            else
                indices.outliers.push_back(static_cast<PointId>(point));
        }
        return indices;
    }

    Indices processRadiusKd(PointViewPtr view) const
    {
        KD3Index index(*view);
        index.build();
        Indices indices;
        for (PointId point = 0; point < view->size(); ++point)
        {
            const PointIdList neighbors = index.radius(point, m_radius);
            if (neighbors.size() > static_cast<std::size_t>(m_minK))
                indices.inliers.push_back(point);
            else
                indices.outliers.push_back(point);
        }
        return indices;
    }

    Indices processRadiusHost(PointViewPtr view) const
    {
        try
        {
            pdg::HostMemoryResource memory;
            pdg::PointBatch batch = gather(*view, memory);
            const pdg::UniformGridConfig config =
                pdg::makeUniformGridConfig(batch, 3, m_radius);
            pdg::SpatialIndex index(batch, config);
            index.build();
            std::vector<std::uint32_t> counts(batch.size());
            pdg::radiusCounts(index, m_radius, counts.data());
            return classify(counts, m_minK);
        }
        catch (const std::exception&)
        {
            // Nonpositive/nonfinite radii, nonfinite coordinates, and very
            // large views retain the pinned nanoflann behavior.
            return processRadiusKd(std::move(view));
        }
    }

    Indices classifyStatistical(const std::vector<double>& distances) const
    {
        std::size_t samples = 0;
        double mean = 0.0;
        double secondMoment = 0.0;
        for (double distance : distances)
        {
            const std::size_t previous = samples;
            ++samples;
            const double delta = distance - mean;
            const double normalized = delta / static_cast<double>(samples);
            mean += normalized;
            secondMoment += delta * normalized * static_cast<double>(previous);
        }
        const double variance =
            secondMoment / (static_cast<double>(samples) - 1.0);
        const double threshold = mean + m_multiplier * std::sqrt(variance);

        Indices indices;
        for (std::size_t point = 0; point < distances.size(); ++point)
        {
            if (distances[point] < threshold)
                indices.inliers.push_back(static_cast<PointId>(point));
            else
                indices.outliers.push_back(static_cast<PointId>(point));
        }
        return indices;
    }

    Indices processStatisticalKd(PointViewPtr view) const
    {
        KD3Index index(*view);
        index.build();
        const point_count_t pointCount = view->size();
        std::vector<double> distances(pointCount, 0.0);

        const point_count_t count = m_meanK + 1;
        PointIdList pointIds(count);
        std::vector<double> squaredDistances(count);
        for (PointId point = 0; point < pointCount; ++point)
        {
            index.knnSearch(point, count, &pointIds, &squaredDistances);
            for (std::size_t neighbor = 1; neighbor < count; ++neighbor)
            {
                const double delta =
                    std::sqrt(squaredDistances[neighbor]) - distances[point];
                distances[point] += delta / static_cast<double>(neighbor);
            }
            pointIds.clear();
            pointIds.resize(count);
            squaredDistances.clear();
            squaredDistances.resize(count);
        }
        return classifyStatistical(distances);
    }

#if PDG_HAS_CUDA
    bool tryStatisticalCuda(PointView& view, std::vector<double>& distances,
                            std::vector<std::uint8_t>& status,
                            bool requireCuda) const
    {
        if (m_meanK < 0 || m_meanK >= 64)
            return false;
        const std::uint32_t neighbors =
            static_cast<std::uint32_t>(m_meanK) + 1U;
        if (static_cast<std::size_t>(view.size()) < neighbors)
            return false;
        try
        {
            if (std::getenv("PDG_TEST_R4_OUTLIER_RECOVERABLE_CUDA_FAILURE"))
                throw std::runtime_error(
                    "injected recoverable r4 outlier CUDA failure");
            if (pdg::cudaDevices().empty())
                return false;
            // B0172: coarse phase timing for this hybrid lane. It is
            // stats-only, off unless the variable is set, and printed to
            // stderr so it can never contaminate a byte-compared artifact --
            // the same additive pattern as PDG_DEBUG_FUSED_JIT. B0171 found
            // this lane is a separate implementation from the resident one and
            // carries no telemetry of its own, which is why B0168-B0170 could
            // only narrow the 2M-to-4M cost step by ablation.
            const bool debugPhases =
                std::getenv("PDG_DEBUG_HYBRID_OUTLIER_PHASES") != nullptr;
            using DebugClock = std::chrono::steady_clock;
            auto marked = DebugClock::now();
            const auto mark = [&](const char* label)
            {
                if (!debugPhases)
                    return;
                const auto now = DebugClock::now();
                std::fprintf(
                    stderr, "PDG_HYBRID_OUTLIER %-16s %.6f\n", label,
                    std::chrono::duration<double>(now - marked).count());
                marked = now;
            };
            std::unique_ptr<pdg::MemoryResource> pinnedMemory =
                pdg::makeCudaPinnedMemoryResource();
            mark("pinned_resource");
            pdg::PointBatch host = gather(view, *pinnedMemory);
            mark("host_gather");
            const pdg::UniformGridConfig config =
                pdg_detail::selectCudaKnnConfig(host, 3, neighbors);

            std::unique_ptr<pdg::MemoryResource> deviceMemory =
                pdg::makeCudaMemoryResource();
            mark("device_resource");
            pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                                   *m_dimensions, *deviceMemory);
            for (pdg::DimensionId dimension : {X, Y, Z})
                device.materialize(dimension, pdg::DimensionType::Double);
            device.setSize(host.size());
            const cudaStream_t stream =
                static_cast<cudaStream_t>(device.nativeStreamHandle());
            for (pdg::DimensionId dimension : {X, Y, Z})
                PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                               host.rawData(dimension),
                                               host.size() * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));

            pdg::SpatialIndex index(device, config);
            index.build();
            std::unique_ptr<pdg::Allocation> deviceDistances =
                deviceMemory->allocate(host.size() * sizeof(double),
                                       alignof(double));
            std::unique_ptr<pdg::Allocation> deviceStatus =
                deviceMemory->allocate(host.size(), alignof(std::uint8_t));
            pdg::knnMeanDistances(
                index, neighbors, static_cast<double*>(deviceDistances->data()),
                static_cast<std::uint8_t*>(deviceStatus->data()));
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                distances.data(), deviceDistances->data(),
                host.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
            PDG_CUDA_CHECK(cudaMemcpyAsync(status.data(), deviceStatus->data(),
                                           host.size(), cudaMemcpyDeviceToHost,
                                           stream));
            mark("gather_submit");
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            mark("gather_wait");
            // Incomplete rows are repaired from the compatibility index in
            // upstream's exact operation order; the mean distance has no
            // cross-row dependency, so the repair is per-row.
            std::size_t incompleteRows = 0;
            for (std::uint8_t pointStatus : status)
                incompleteRows += static_cast<std::size_t>(
                    (pointStatus & pdg::KnnSearchIncomplete) != 0U);
            if (debugPhases)
                std::fprintf(stderr,
                             "PDG_HYBRID_OUTLIER %-16s %zu\n",
                             "incomplete_rows", incompleteRows);
            if (incompleteRows != 0U)
            {
                mark("status_scan");
                KD3Index& kdIndex = view.build3dIndex();
                mark("kd3_build");
                const point_count_t count =
                    static_cast<point_count_t>(neighbors);
                for (PointId point = 0; point < view.size(); ++point)
                {
                    if ((status[static_cast<std::size_t>(point)] &
                         pdg::KnnSearchIncomplete) == 0U)
                        continue;
                    PointIdList pointIds(count);
                    std::vector<double> squaredDistances(count);
                    kdIndex.knnSearch(point, count, &pointIds,
                                      &squaredDistances);
                    double& value = distances[static_cast<std::size_t>(point)];
                    value = 0.0;
                    for (std::size_t neighbor = 1; neighbor < count; ++neighbor)
                    {
                        const double delta =
                            std::sqrt(squaredDistances[neighbor]) - value;
                        value += delta / static_cast<double>(neighbor);
                    }
                }
            }
            mark("repair_and_finale");
            return true;
        }
        catch (const pdg::CudaError&)
        {
            if (requireCuda)
                throw;
            return false;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }
#else
    bool tryStatisticalCuda(PointView&, std::vector<double>&,
                            std::vector<std::uint8_t>&, bool) const
    {
        return false;
    }
#endif

    PointViewSet run(PointViewPtr view) override
    {
        PointViewSet viewSet;
        if (view->empty())
            return viewSet;

        if (m_residentContext)
        {
            const bool statistical = Utils::iequals(m_method, "statistical");
            const bool radius = Utils::iequals(m_method, "radius");
            if ((!statistical || m_meanK < 0 || m_meanK >= 64) &&
                (!radius || !std::isfinite(m_radius) || m_radius <= 0.0))
                throwError("planner-selected resident outlier path received "
                           "an unsupported option envelope");
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto executionRegion =
                static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(*view, executionRegion);
            bool usedCuda = false;
            std::size_t inliers = 0U;
            std::size_t outliers = 0U;
            if (statistical)
            {
                pdg_detail::CudaStatisticalOutlierResult result;
                usedCuda = pdg_detail::tryCudaStatisticalOutlier(
                    *view, static_cast<std::uint32_t>(m_meanK) + 1U,
                    m_multiplier, m_class, m_region, result,
                    /*requireCuda=*/true);
                inliers = result.inliers;
                outliers = result.outliers;
            }
            else
            {
                m_region.radiusIndex = true;
                m_region.dimensions = 3U;
                if (m_region.maximumRadius <= 0.0)
                    m_region.maximumRadius = m_radius;
                pdg_detail::CudaRadiusOutlierResult result;
                usedCuda = pdg_detail::tryCudaRadiusOutlier(
                    *view, m_radius, m_minK, m_class, m_region, result,
                    /*requireCuda=*/true);
                inliers = result.inliers;
                outliers = result.outliers;
            }
            if (!usedCuda)
                throwError("planner-selected resident shared-index "
                           "outlier path was not used");
            if (m_region.last)
                context.endDelegatedRegion(*view, executionRegion);

            if (inliers == 0U)
                log()->get(LogLevel::Warning)
                    << "Requested filter would remove all points. Try a "
                       "larger radius/smaller minimum neighbors.\n";
            else if (outliers != 0U)
                log()->get(LogLevel::Debug2)
                    << "Labeled " << outliers << " outliers as noise!\n";
            else
                log()->get(LogLevel::Warning)
                    << "Filtered cloud has no outliers!\n";
            viewSet.insert(view);
            return viewSet;
        }

        Indices indices;
        if (Utils::iequals(m_method, "statistical"))
        {
            const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
            const bool requireAutomatic =
                std::getenv("PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA");
            const bool automaticCuda =
                m_autoCuda && pdg::automaticR4OutlierCudaDeviceQualified();
            const bool requestCuda =
                !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
                (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
                 automaticCuda);
            std::vector<double> distances(
                static_cast<std::size_t>(view->size()));
            std::vector<std::uint8_t> status(
                static_cast<std::size_t>(view->size()));
            const bool usedCuda =
                requestCuda &&
                tryStatisticalCuda(*view, distances, status, requireCuda);
            if (requireCuda && !usedCuda)
                throwError(
                    "required exact CUDA hybrid outlier path was not used");
            if (requireAutomatic && (!m_autoCuda || !usedCuda))
                throwError(
                    "required automatic exact CUDA r4 outlier path was not "
                    "used");
            indices = usedCuda ? classifyStatistical(distances)
                               : processStatisticalKd(view);
        }
        else if (Utils::iequals(m_method, "radius"))
        {
            const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
            const bool requestCuda =
                requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
            std::vector<std::uint32_t> counts(
                static_cast<std::size_t>(view->size()));
            const bool usedCuda =
                requestCuda && pdg_detail::tryCudaRadiusCounts(
                                   *view, m_radius, counts, requireCuda);
            if (requireCuda && !usedCuda)
                throwError(
                    "required exact CUDA hybrid outlier path was not used");
            indices =
                usedCuda ? classify(counts, m_minK) : processRadiusHost(view);
        }
        else
        {
            log()->get(LogLevel::Warning)
                << "Requested method is unrecognized. Please choose from "
                   "\"statistical\" or \"radius\".\n";
            viewSet.insert(view);
            return viewSet;
        }

        if (indices.inliers.empty())
        {
            log()->get(LogLevel::Warning)
                << "Requested filter would remove all points. Try a larger "
                   "radius/smaller minimum neighbors.\n";
            viewSet.insert(view);
            return viewSet;
        }

        if (!indices.outliers.empty())
        {
            log()->get(LogLevel::Debug2)
                << "Labeled " << indices.outliers.size()
                << " outliers as noise!\n";
            for (PointId point : indices.outliers)
                view->setField(Dimension::Id::Classification, point, m_class);
            viewSet.insert(view);
        }
        else
        {
            log()->get(LogLevel::Warning)
                << "Filtered cloud has no outliers!\n";
            viewSet.insert(view);
        }
        return viewSet;
    }

    std::string m_method;
    int m_minK = 2;
    double m_radius = 1.0;
    int m_meanK = 8;
    double m_multiplier = 2.0;
    std::uint8_t m_class = ClassLabel::LowPoint;
    bool m_autoCuda = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
    pdg_detail::CudaNeighborhoodRegion m_region;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions =
        std::make_unique<pdg::DimensionRegistry>();
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridOutlierStage),
    "Internal exact PDG shared-index outlier filter", ""};

CREATE_STATIC_STAGE(PdgOutlierFilter, s_info)

} // namespace pdal
