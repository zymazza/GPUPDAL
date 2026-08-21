#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Summary.hpp>

#include <pdal/BatchStreamable.hpp>
#include <pdal/Filter.hpp>
#include <pdal/PDALUtils.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Polygon.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pdal
{

namespace
{
struct SummaryRecord
{
    SummaryRecord(std::string summaryName, pdg::SummaryMode summaryMode,
                  Dimension::Id pdalDimension, pdg::DimensionId pdgDimension)
        : name(std::move(summaryName)), mode(summaryMode),
          pdalId(pdalDimension), pdgId(pdgDimension)
    {
    }

    void insert(double value, bool advanced)
    {
        pdg::insertSummary(state, value, advanced);
        if (mode != pdg::SummaryMode::None)
            ++values[value];
        if (mode == pdg::SummaryMode::Global)
        {
            if (data.capacity() - data.size() < 10000U)
                data.reserve(data.capacity() + state.count);
            data.push_back(value);
        }
    }

    std::string name;
    pdg::SummaryMode mode = pdg::SummaryMode::None;
    Dimension::Id pdalId = Dimension::Id::Unknown;
    pdg::DimensionId pdgId;
    pdg::SummaryState state;
    std::map<double, point_count_t> values;
    std::vector<double> data;
    double median = 0.0;
    double mad = 0.0;
};
} // unnamed namespace

// Behaviorally derived from the pinned upstream StatsFilter. It intentionally
// preserves the upstream dimension ordering, warning strings, online-moment
// arithmetic, metadata schema, and bbox construction.
class PdgStatsFilter final : public Filter, public BatchStreamable
{
public:
    std::string getName() const override
    {
        return "filters.stats";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimensions", "Dimensions on which to calculate statistics",
                 m_dimNames);
        args.add("enumerate", "Dimensions whose values should be enumerated",
                 m_enums);
        args.add("global",
                 "Dimensions to compute global stats (median, mad, mode)",
                 m_global);
        args.add("count", "Dimensions whose values should be counted",
                 m_counts);
        args.add("advanced", "Calculate skewness and kurtosis", m_advanced);
        args.add("commonsrs",
                 "Common SRS to use for normalizing bounding boxes",
                 m_commonSrs, "EPSG:4326");
    }

    void prepared(PointTableRef table) override
    {
        PointLayoutPtr layout(table.layout());
        std::unordered_map<std::string, pdg::SummaryMode> dimensions;
        auto getWarn = [this]() -> std::ostream&
        { return log()->get(LogLevel::Warning); };

        if (m_dimNames.empty())
        {
            for (Dimension::Id id : layout->dims())
                dimensions[layout->dimName(id)] = pdg::SummaryMode::None;
        }
        else
        {
            for (const std::string& name : m_dimNames)
            {
                if (layout->findDim(name) == Dimension::Id::Unknown)
                    getWarn() << "Dimension '" << name
                              << "' listed in --dimensions option does not "
                                 "exist.  Ignoring."
                              << std::endl;
                else
                    dimensions[name] = pdg::SummaryMode::None;
            }
        }

        setModes(dimensions, m_enums, pdg::SummaryMode::Enumerate, "enumerate",
                 getWarn);
        setModes(dimensions, m_counts, pdg::SummaryMode::Count, "count",
                 getWarn);
        setModes(dimensions, m_global, pdg::SummaryMode::Global, "global",
                 getWarn);

        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        for (const auto& [name, mode] : dimensions)
        {
            const Dimension::Id pdalId = layout->findDim(name);
            const pdg::DimensionDefinition* definition =
                m_dimensions->find(name);
            if (!definition)
                definition = &m_dimensions->registerCustom(
                    name, pdg::DimensionType::Double);
            m_summaries.emplace(
                pdalId, SummaryRecord(name, mode, pdalId, definition->id));
        }
    }

    template <typename Warning>
    static void
    setModes(std::unordered_map<std::string, pdg::SummaryMode>& dimensions,
             const StringList& names, pdg::SummaryMode mode, const char* option,
             Warning& warning)
    {
        for (const std::string& name : names)
        {
            const auto position = dimensions.find(name);
            if (position == dimensions.end())
                warning() << "Dimension '" << name << "' listed in --" << option
                          << " option does not exist.  Ignoring." << std::endl;
            else
                position->second = mode;
        }
    }

    pdg::PointBatch makeBatch(std::size_t count,
                              pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch(
            count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, memory);
        for (const auto& [id, summary] : m_summaries)
        {
            (void)id;
            batch.materialize(summary.pdgId, pdg::DimensionType::Double);
        }
        batch.setSize(count);
        return batch;
    }

    pdg::PointBatch gather(PointView& view, pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch =
            makeBatch(static_cast<std::size_t>(view.size()), memory);
        for (const auto& [id, summary] : m_summaries)
        {
            (void)id;
            double* output = batch.data<double>(summary.pdgId);
            for (PointId point = 0; point < view.size(); ++point)
                output[static_cast<std::size_t>(point)] =
                    view.getFieldAs<double>(summary.pdalId, point);
        }
        return batch;
    }

    pdg::PointBatch gather(StreamPointTable& table,
                           const std::vector<PointId>& points,
                           pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch = makeBatch(points.size(), memory);
        PointRef point(table, 0);
        for (const auto& [id, summary] : m_summaries)
        {
            (void)id;
            double* output = batch.data<double>(summary.pdgId);
            for (std::size_t offset = 0; offset < points.size(); ++offset)
            {
                point.setPointId(points[offset]);
                output[offset] = point.getFieldAs<double>(summary.pdalId);
            }
        }
        return batch;
    }

    [[nodiscard]] std::vector<pdg::DimensionId> pdgDimensions() const
    {
        std::vector<pdg::DimensionId> dimensions;
        dimensions.reserve(m_summaries.size());
        for (const auto& [id, summary] : m_summaries)
        {
            (void)id;
            dimensions.push_back(summary.pdgId);
        }
        return dimensions;
    }

    [[nodiscard]] bool deviceModesAreExact() const noexcept
    {
        if (m_advanced)
            return false;
        return std::all_of(
            m_summaries.begin(), m_summaries.end(), [](const auto& item)
            { return item.second.mode == pdg::SummaryMode::None; });
    }

#if PDG_HAS_CUDA
    void applyCuda(pdg::PointBatch& host,
                   const std::vector<pdg::DimensionId>& dimensions)
    {
        const std::size_t count = host.size();
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(count, host.coordinateEncoding(), *m_dimensions,
                               *deviceMemory);
        for (pdg::DimensionId dimension : dimensions)
            device.materialize(dimension, pdg::DimensionType::Double);
        device.setSize(count);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (pdg::DimensionId dimension : dimensions)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                device.rawData(dimension), host.rawData(dimension),
                count * sizeof(double), cudaMemcpyHostToDevice, stream));

        std::vector<pdg::SummaryState> states;
        states.reserve(m_summaries.size());
        for (const auto& [id, summary] : m_summaries)
        {
            (void)id;
            states.push_back(summary.state);
        }
        std::unique_ptr<pdg::Allocation> deviceStates =
            deviceMemory->allocate(states.size() * sizeof(pdg::SummaryState),
                                   alignof(pdg::SummaryState));
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(deviceStates->data(), states.data(),
                            states.size() * sizeof(pdg::SummaryState),
                            cudaMemcpyHostToDevice, stream));
        pdg::updateSummaries(
            device, dimensions,
            static_cast<pdg::SummaryState*>(deviceStates->data()), false);
        PDG_CUDA_CHECK(
            cudaMemcpyAsync(states.data(), deviceStates->data(),
                            states.size() * sizeof(pdg::SummaryState),
                            cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::size_t index = 0;
        for (auto& [id, summary] : m_summaries)
        {
            (void)id;
            summary.state = states[index++];
        }
    }

    template <typename Gather>
    bool tryCuda(Gather&& gatherBatch, bool requireCuda)
    {
        if (!deviceModesAreExact() || m_summaries.empty())
            return false;
        try
        {
            if (pdg::cudaDevices().empty())
                return false;
            std::unique_ptr<pdg::MemoryResource> pinnedMemory =
                pdg::makeCudaPinnedMemoryResource();
            pdg::PointBatch host = gatherBatch(*pinnedMemory);
            const std::vector<pdg::DimensionId> dimensions = pdgDimensions();
            if (!pdg::summariesMaySupportExactDevice(host, dimensions, false))
                return false;
            applyCuda(host, dimensions);
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
    template <typename Gather> bool tryCuda(Gather&&, bool)
    {
        return false;
    }
#endif

    void insertHost(PointView& view)
    {
        for (PointId point = 0; point < view.size(); ++point)
            for (auto& [id, summary] : m_summaries)
            {
                (void)id;
                summary.insert(view.getFieldAs<double>(summary.pdalId, point),
                               m_advanced);
            }
    }

    void insertHost(StreamPointTable& table, const std::vector<PointId>& points)
    {
        PointRef point(table, 0);
        for (PointId id : points)
        {
            point.setPointId(id);
            for (auto& [dimension, summary] : m_summaries)
            {
                (void)dimension;
                summary.insert(point.getFieldAs<double>(summary.pdalId),
                               m_advanced);
            }
        }
    }

    void filter(PointView& view) override
    {
        if (std::getenv("PDG_REQUIRE_STREAMING_HYBRID"))
            throwError("required streaming hybrid path was not used");
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        const bool usedCuda =
            requestCuda && !view.empty() &&
            tryCuda([&](pdg::MemoryResource& memory)
                    { return gather(view, memory); }, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid stats path was not used");
        if (!usedCuda)
            insertHost(view);
    }

    void processBatch(StreamPointTable& table,
                      point_count_t pointLimit) override
    {
        std::vector<PointId> points;
        points.reserve(static_cast<std::size_t>(pointLimit));
        for (PointId point = 0; point < pointLimit; ++point)
            if (!table.skip(point))
                points.push_back(point);

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        const bool usedCuda =
            requestCuda && !points.empty() &&
            tryCuda([&](pdg::MemoryResource& memory)
                    { return gather(table, points, memory); }, requireCuda);
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid stats path was not used");
        if (!usedCuda)
            insertHost(table, points);
    }

    static void computeGlobalStats(SummaryRecord& summary)
    {
        const auto computeMedian = [](std::vector<double> values)
        {
            std::nth_element(values.begin(),
                             values.begin() + values.size() / 2U, values.end());
            return *(values.begin() + values.size() / 2U);
        };
        summary.median = computeMedian(summary.data);
        std::transform(summary.data.begin(), summary.data.end(),
                       summary.data.begin(), [&](double value)
                       { return std::fabs(value - summary.median); });
        summary.mad = computeMedian(summary.data);
    }

    void extractSummaryMetadata(SummaryRecord& summary, MetadataNode& metadata)
    {
        metadata.add("count", static_cast<std::uint32_t>(summary.state.count),
                     "count");
        metadata.add("minimum", summary.state.minimum, "minimum");
        metadata.add("maximum", summary.state.maximum, "maximum");
        metadata.add("average", pdg::summaryAverage(summary.state), "average");

        const double standardDeviation =
            pdg::summarySampleStddev(summary.state);
        if (!std::isinf(standardDeviation) && !std::isnan(standardDeviation))
            metadata.add("stddev", standardDeviation, "standard deviation");
        const double variance = pdg::summarySampleVariance(summary.state);
        if (!std::isinf(variance) && !std::isnan(variance))
            metadata.add("variance", variance, "variance");
        metadata.add("name", summary.name, "name");

        if (m_advanced)
        {
            const double kurtosis =
                pdg::summarySampleExcessKurtosis(summary.state, m_advanced);
            if (!std::isinf(kurtosis) && !std::isnan(kurtosis))
                metadata.add("kurtosis", kurtosis, "kurtosis");
            const double skewness =
                pdg::summarySampleSkewness(summary.state, m_advanced);
            if (!std::isinf(skewness) && !std::isnan(skewness))
                metadata.add(
                    "skewness",
                    pdg::summarySampleSkewness(summary.state, m_advanced),
                    "skewness");
        }

        if (summary.mode == pdg::SummaryMode::Enumerate)
        {
            for (const auto& [value, count] : summary.values)
            {
                (void)count;
                metadata.addList("values", value);
            }
        }
        else if (summary.mode == pdg::SummaryMode::Global)
        {
            computeGlobalStats(summary);
            metadata.add("median", summary.median);
            metadata.add("mad", summary.mad);
        }
        else if (summary.mode == pdg::SummaryMode::Count)
        {
            MetadataNode bins = metadata.add("bins");
            for (const auto& [value, count] : summary.values)
            {
                bins.add(std::to_string(value),
                         static_cast<std::uint64_t>(count));
                metadata.addList("counts", std::to_string(value) + "/" +
                                               std::to_string(count));
            }
        }
    }

    void extractMetadata(PointTableRef table)
    {
        std::uint32_t position = 0;
        bool noPoints = true;
        for (auto& [id, summary] : m_summaries)
        {
            (void)id;
            noPoints = static_cast<bool>(summary.state.count);
            MetadataNode statistic = m_metadata.addList("statistic");
            statistic.add("position", position++);
            extractSummaryMetadata(summary, statistic);
        }

        const auto x = m_summaries.find(Dimension::Id::X);
        const auto y = m_summaries.find(Dimension::Id::Y);
        const auto z = m_summaries.find(Dimension::Id::Z);
        if (x == m_summaries.end() || y == m_summaries.end() ||
            z == m_summaries.end() || !noPoints)
            return;

        BOX3D box(x->second.state.minimum, y->second.state.minimum,
                  z->second.state.minimum, x->second.state.maximum,
                  y->second.state.maximum, z->second.state.maximum);
        Polygon polygon(box);
        MetadataNode boxValues = Utils::toMetadata(box);
        MetadataNode boxMetadata = m_metadata.add("bbox");
        MetadataNode native = boxMetadata.add("native");
        native.addWithType("boundary", polygon.json(), "json",
                           "GeoJSON boundary");
        native.add(boxValues);

        SpatialReference reference = table.anySpatialReference();
        if (reference.empty())
            return;
        polygon.setSpatialReference(reference);
        if (polygon.transform(m_commonSrs))
        {
            BOX3D transformedBox = polygon.bounds();
            MetadataNode transformedValues = Utils::toMetadata(transformedBox);
            MetadataNode transformed = boxMetadata.add(m_commonSrs);
            transformed.add(transformedValues);
            transformed.addWithType("boundary", polygon.json(), "json",
                                    "GeoJSON boundary");
        }
    }

    void done(PointTableRef table) override
    {
        extractMetadata(table);
    }

    StringList m_dimNames;
    StringList m_enums;
    StringList m_counts;
    StringList m_global;
    std::string m_commonSrs = "EPSG:4326";
    bool m_advanced = false;
    std::map<Dimension::Id, SummaryRecord> m_summaries;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridStatsStage),
                                     "Internal exact PDG dimension statistics",
                                     ""};

CREATE_STATIC_STAGE(PdgStatsFilter, s_info)

} // namespace pdal
