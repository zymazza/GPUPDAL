#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Histogram.hpp>

#include <filters/private/expr/ConditionalExpression.hpp>
#include <pdal/BatchStreamable.hpp>
#include <pdal/Filter.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pdal
{

// Behaviorally derived from the pinned upstream ExpressionStatsFilter. The
// canonical expression print and ordered maps are observable metadata.
class PdgExpressionStatsFilter final : public Filter, public BatchStreamable
{
    struct Args
    {
        std::vector<expr::ConditionalExpression> expressions;
        Arg* expressionsArg = nullptr;
    };

    struct DeviceColumn
    {
        Dimension::Id pdalId = Dimension::Id::Unknown;
        pdg::DimensionId pdgId;
    };

    struct DeviceExpression
    {
        std::string label;
        pdg::PredicateProgram predicate;
    };

public:
    PdgExpressionStatsFilter() : m_args(std::make_unique<Args>()) {}

    std::string getName() const override
    {
        return "filters.expressionstats";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        m_args->expressionsArg =
            &args.add("expressions",
                      "Conditional expressions describing points to be "
                      "passed to this filter",
                      m_args->expressions)
                 .setPositional();
        args.add("dimension",
                 "Dimension on which apply expression to calculate "
                 "statistics",
                 m_dimensionName)
            .setPositional();
        args.add("pdg_auto_cuda", "Internal automatic CUDA selection marker",
                 m_autoCuda, false);
    }

    void prepared(PointTableRef table) override
    {
        m_dimension = table.layout()->findDim(m_dimensionName);
        if (m_dimension == Dimension::Id::Unknown)
            throwError("Invalid dimension name in 'dimension' option: '" +
                       m_dimensionName + "'.");

        for (expr::ConditionalExpression& expression : m_args->expressions)
        {
            if (!expression.valid())
            {
                std::ostringstream message;
                message << "The expression '" << expression << "' is invalid";
                throwError(message.str());
            }
            const auto status = expression.prepare(table.layout());
            if (!status)
                throwError("Invalid expression: " + status.what());
        }
        // Avoid device-planning work on the exact host fallback. Compilation
        // walks every registered dimension and recompiles every expression,
        // so do it only for an explicitly requested lane or the planner's
        // count-qualified automatic marker.
        if (!std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (std::getenv("PDG_REQUIRE_CUDA_HYBRID") ||
             std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") || m_autoCuda))
            prepareDevice(table);
    }

    void prepareDevice(PointTableRef table) noexcept
    {
#if PDG_HAS_CUDA
        // A Stage is normally prepared once, but clear any prior workspace
        // before replacing the registry whose addresses its batches retain.
        m_deviceBatch.reset();
        m_hostBatch.reset();
        m_deviceMemory.reset();
        m_pinnedMemory.reset();
#endif
        m_deviceProgramsValid = false;
        m_deviceExpressions.clear();
        m_deviceColumns.clear();
        try
        {
            PointLayoutPtr layout(table.layout());
            m_pdgDimensions = std::make_unique<pdg::DimensionRegistry>();
            for (Dimension::Id id : layout->dims())
            {
                const std::string name = layout->dimName(id);
                if (!m_pdgDimensions->find(name))
                    m_pdgDimensions->registerCustom(name,
                                                    pdg::DimensionType::Double);
            }

            const pdg::DimensionDefinition* target =
                m_pdgDimensions->find(m_dimensionName);
            if (!target)
                return;
            m_pdgTarget = target->id;
            std::unordered_set<std::uint32_t> required;
            required.insert(m_pdgTarget.value());
            for (const expr::ConditionalExpression& expression :
                 m_args->expressions)
            {
                const std::string label = expression.print();
                pdg::PredicateProgram predicate =
                    pdg::compilePredicate(label, *m_pdgDimensions);
                for (pdg::DimensionId read : predicate.reads)
                    required.insert(read.value());
                m_deviceExpressions.push_back({label, std::move(predicate)});
            }

            for (Dimension::Id id : layout->dims())
            {
                const pdg::DimensionDefinition* definition =
                    m_pdgDimensions->find(layout->dimName(id));
                if (definition && required.contains(definition->id.value()))
                    m_deviceColumns.push_back({id, definition->id});
            }
            if (m_deviceColumns.size() != required.size())
                return;
            m_deviceProgramsValid = !m_deviceExpressions.empty();
        }
        catch (const std::exception&)
        {
            m_deviceExpressions.clear();
            m_deviceColumns.clear();
            m_deviceProgramsValid = false;
        }
    }

    bool processPoint(PointRef& point)
    {
        const double value = point.getFieldAs<double>(m_dimension);
        for (const expr::ConditionalExpression& expression :
             m_args->expressions)
        {
            auto& bins = m_statistics[expression.print()];
            if (expression.eval(point))
                ++bins[value];
        }
        return true;
    }

    pdg::PointBatch makeBatch(std::size_t capacity,
                              pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_pdgDimensions, memory);
        for (const DeviceColumn& column : m_deviceColumns)
            batch.materialize(column.pdgId, pdg::DimensionType::Double);
        return batch;
    }

    void gather(PointView& view, pdg::PointBatch& batch) const
    {
        batch.setSize(static_cast<std::size_t>(view.size()));
        for (const DeviceColumn& column : m_deviceColumns)
        {
            double* output = batch.data<double>(column.pdgId);
            for (PointId point = 0; point < view.size(); ++point)
                output[static_cast<std::size_t>(point)] =
                    view.getFieldAs<double>(column.pdalId, point);
        }
    }

    void gather(StreamPointTable& table, const std::vector<PointId>& points,
                pdg::PointBatch& batch) const
    {
        batch.setSize(points.size());
        PointRef point(table, 0);
        for (const DeviceColumn& column : m_deviceColumns)
        {
            double* output = batch.data<double>(column.pdgId);
            for (std::size_t offset = 0; offset < points.size(); ++offset)
            {
                point.setPointId(points[offset]);
                output[offset] = point.getFieldAs<double>(column.pdalId);
            }
        }
    }

#if PDG_HAS_CUDA
    void ensureCudaWorkspace(std::size_t capacity)
    {
        if (m_hostBatch && m_deviceBatch &&
            m_hostBatch->capacity() >= capacity &&
            m_deviceBatch->capacity() >= capacity)
            return;

        if (!m_pinnedMemory)
            m_pinnedMemory = pdg::makeCudaPinnedMemoryResource();
        if (!m_deviceMemory)
            m_deviceMemory = pdg::makeCudaMemoryResource();

        // Build both replacements before publishing either one. PointBatch
        // retains the resource addresses, which remain stable behind the
        // unique_ptrs for the lifetime of this stage.
        auto host = std::make_unique<pdg::PointBatch>(
            makeBatch(capacity, *m_pinnedMemory));
        auto device = std::make_unique<pdg::PointBatch>(
            makeBatch(capacity, *m_deviceMemory));
        m_hostBatch = std::move(host);
        m_deviceBatch = std::move(device);
    }

    std::vector<std::vector<pdg::HistogramBin>>
    applyCuda(pdg::PointBatch& host) const
    {
        pdg::PointBatch& device = *m_deviceBatch;
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (const DeviceColumn& column : m_deviceColumns)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                device.rawData(column.pdgId), host.rawData(column.pdgId),
                host.size() * sizeof(double), cudaMemcpyHostToDevice, stream));

        std::vector<std::vector<pdg::HistogramBin>> result;
        result.reserve(m_deviceExpressions.size());
        for (const DeviceExpression& expression : m_deviceExpressions)
            result.push_back(pdg::selectedHistogram(
                device, m_pdgTarget, expression.predicate,
                static_cast<std::uint64_t>(m_processedPoints)));
        return result;
    }

    template <typename Gather>
    bool tryCuda(std::size_t count, Gather&& gatherBatch, bool requireCuda)
    {
        if (!m_deviceProgramsValid)
            return false;
        try
        {
            if (pdg::cudaDevices().empty())
                return false;
            ensureCudaWorkspace(count);
            pdg::PointBatch& host = *m_hostBatch;
            gatherBatch(host);
            for (const DeviceExpression& expression : m_deviceExpressions)
                if (!pdg::histogramMaySupportExactDevice(host, m_pdgTarget,
                                                         expression.predicate))
                    return false;

            const std::vector<std::vector<pdg::HistogramBin>> histograms =
                applyCuda(host);
            for (std::size_t index = 0; index < histograms.size(); ++index)
                insertBins(m_deviceExpressions[index].label, histograms[index]);
            m_processedPoints += static_cast<PointId>(host.size());
            return true;
        }
        catch (const pdg::CudaError&)
        {
            if (requireCuda)
                throw;
            return false;
        }
    }
#else
    template <typename Gather> bool tryCuda(std::size_t, Gather&&, bool)
    {
        return false;
    }
#endif

    void insertBins(const std::string& expression,
                    const std::vector<pdg::HistogramBin>& source)
    {
        auto& destination = m_statistics[expression];
        for (const pdg::HistogramBin& bin : source)
        {
            auto [position, inserted] = destination.emplace(bin.value, 0);
            (void)inserted;
            if (bin.count > static_cast<std::uint64_t>(
                                (std::numeric_limits<point_count_t>::max)() -
                                position->second))
                throwError("Expression statistic count overflow");
            position->second += static_cast<point_count_t>(bin.count);
        }
    }

    void filter(PointView& view) override
    {
        if (std::getenv("PDG_REQUIRE_STREAMING_HYBRID"))
            throwError("required streaming hybrid path was not used");
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             m_autoCuda);
        const bool usedCuda = requestCuda && !view.empty() &&
                              tryCuda(
                                  static_cast<std::size_t>(view.size()),
                                  [&](pdg::PointBatch& batch)
                                  { gather(view, batch); }, requireCuda);
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid expressionstats path was not "
                "used");
        if (usedCuda)
            return;

        PointRef point(view, 0);
        for (PointId id = 0; id < view.size(); ++id)
        {
            point.setPointId(id);
            processPoint(point);
        }
        m_processedPoints += view.size();
    }

    void processBatch(StreamPointTable& table,
                      point_count_t pointLimit) override
    {
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             m_autoCuda);

        if (requestCuda)
        {
            std::vector<PointId> points;
            points.reserve(static_cast<std::size_t>(pointLimit));
            for (PointId point = 0; point < pointLimit; ++point)
                if (!table.skip(point))
                    points.push_back(point);

            const bool usedCuda =
                !points.empty() &&
                tryCuda(
                    points.size(), [&](pdg::PointBatch& batch)
                    { gather(table, points, batch); }, requireCuda);
            if (requireCuda && !usedCuda)
                throwError(
                    "required exact CUDA hybrid expressionstats path was not "
                    "used");
            if (usedCuda)
                return;

            PointRef point(table, 0);
            for (PointId id : points)
            {
                point.setPointId(id);
                processPoint(point);
            }
            m_processedPoints += static_cast<PointId>(points.size());
            return;
        }

        PointRef point(table, 0);
        PointId processed = 0;
        for (PointId id = 0; id < pointLimit; ++id)
        {
            if (table.skip(id))
                continue;
            point.setPointId(id);
            processPoint(point);
            ++processed;
        }
        m_processedPoints += processed;
    }

    void done(PointTableRef table) override
    {
        std::uint32_t position = 0;
        m_metadata.add("dimension", table.layout()->dimName(m_dimension));
        for (auto& [expression, bins] : m_statistics)
        {
            MetadataNode statistic = m_metadata.addList("statistic");
            statistic.add("expression", expression);
            statistic.add("position", position++);
            for (auto& [value, count] : bins)
            {
                MetadataNode bin = statistic.addList("bins");
                bin.add("count", count);
                bin.add("value", value);
            }
        }
    }

    std::unique_ptr<Args> m_args;
    Dimension::Id m_dimension = Dimension::Id::Unknown;
    std::string m_dimensionName;
    std::map<std::string, std::map<double, point_count_t>> m_statistics;
    std::unique_ptr<pdg::DimensionRegistry> m_pdgDimensions;
#if PDG_HAS_CUDA
    // Resources precede batches so reverse member destruction releases every
    // allocation before destroying its allocator/stream state.
    std::unique_ptr<pdg::MemoryResource> m_pinnedMemory;
    std::unique_ptr<pdg::MemoryResource> m_deviceMemory;
    std::unique_ptr<pdg::PointBatch> m_hostBatch;
    std::unique_ptr<pdg::PointBatch> m_deviceBatch;
#endif
    pdg::DimensionId m_pdgTarget;
    std::vector<DeviceColumn> m_deviceColumns;
    std::vector<DeviceExpression> m_deviceExpressions;
    PointId m_processedPoints = 0;
    bool m_deviceProgramsValid = false;
    bool m_autoCuda = false;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridExpressionStatsStage),
    "Internal exact PDG expression statistics", ""};
CREATE_STATIC_STAGE(PdgExpressionStatsFilter, s_info)

} // namespace pdal
