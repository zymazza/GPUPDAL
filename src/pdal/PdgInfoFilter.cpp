#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Information.hpp>

#include <pdal/BatchStreamable.hpp>
#include <pdal/Filter.hpp>
#include <pdal/PDALUtils.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{

// Behaviorally derived from the pinned upstream InfoFilter. Historical parser
// spelling and the three-coordinate query quirk are compatibility behavior.
class PdgInfoFilter final : public Filter, public BatchStreamable
{
    struct NearPoint
    {
        NearPoint(PointId id, double distance, std::vector<char>&& data)
            : m_id(id), m_distance(distance), m_data(std::move(data))
        {
        }

        bool operator<(const NearPoint& other) const
        {
            return m_distance < other.m_distance;
        }

        PointId m_id;
        double m_distance;
        std::vector<char> m_data;
    };

public:
    PdgInfoFilter()
        : m_queryCount(10), m_queryZ(std::numeric_limits<double>::quiet_NaN())
    {
    }

    std::string getName() const override
    {
        return "filters.info";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("point,p",
                 "Point to dump\n--point=\"1-5,10,100-200\" (0 indexed)",
                 m_pointSpec);
        args.add("query",
                 "Return points in order of distance from the specified "
                 "location (2D or 3D)\n"
                 "--query Xcoord,Ycoord[,Zcoord][/count]",
                 m_querySpec);
    }

    void parsePointSpec()
    {
        const auto parseInteger = [this](const std::string& value)
        {
            std::uint32_t result;
            if (!Utils::fromString(value, result))
                throwError("Invalid integer '" + value + "in 'point' option");
            return result;
        };
        const auto addRange = [this, &parseInteger](const std::string& begin,
                                                    const std::string& end)
        {
            PointId low = parseInteger(begin);
            const PointId high = parseInteger(end);
            if (low > high)
                throwError("Invalid range in 'point' option: '" + begin + "-" +
                           end);
            while (low <= high)
                m_idList.push_back(low++);
        };

        Utils::trim(m_pointSpec);
        StringList ranges = Utils::split2(m_pointSpec, ',');
        for (std::string& range : ranges)
        {
            const StringList limits = Utils::split(range, '-');
            if (limits.size() == 1U)
                m_idList.push_back(parseInteger(limits[0]));
            else if (limits.size() == 2U)
                addRange(limits[0], limits[1]);
            else
                throwError("Invalid point range in 'point' option: " + range);
        }
    }

    void parseQuerySpec()
    {
        const StringList parts = Utils::split2(m_querySpec, '/');
        if (parts.size() == 2U)
        {
            if (!Utils::fromString(parts[1], m_queryCount))
                throwError("Invalid query count in 'query' option: " +
                           parts[1]);
        }
        else if (parts.size() != 1U)
            throwError("Invalid point location specification. Sytax: "
                       "--query=\"X,Y[/count]\"");

        const auto separators = [](char value)
        { return value == ',' || value == '|' || value == ' '; };
        const StringList tokens = Utils::split2(parts[0], separators);
        if (tokens.size() != 2U && tokens.size() != 3U)
            throwError("Invalid point location specification. Sytax: "
                       "--query=\"X,Y[/count]\"");

        bool valid = true;
        valid &= Utils::fromString(tokens[0], m_queryX);
        valid &= Utils::fromString(tokens[1], m_queryY);
        if (tokens.size() == 3U)
        {
            // Preserve the pinned implementation's historical Y-as-Z parse.
            valid &= Utils::fromString(tokens[1], m_queryZ);
        }
        if (!valid)
            throwError("Invalid point location specification. Sytax: "
                       "--query=\"X,Y[/count]\"");
    }

    void prepared(PointTableRef table) override
    {
        m_dimensions = table.layout()->dimTypes();
        m_pointSize = table.layout()->pointSize();
        if (!m_pointSpec.empty())
            parsePointSpec();
        if (!m_querySpec.empty())
            parseQuerySpec();

        m_pdgDimensions = std::make_unique<pdg::DimensionRegistry>();
    }

    void initialize(PointTableRef table) override
    {
        getMetadata().add(table.layout()->toMetadata());
    }

    void ready(PointTableRef) override
    {
        m_count = 0;
        m_idCurrent = m_idList.begin();
        m_bounds.clear();
    }

    pdg::PointBatch makeBatch(std::size_t count,
                              pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch(
            count, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_pdgDimensions, memory);
        for (pdg::StandardDimension dimension :
             {pdg::StandardDimension::X, pdg::StandardDimension::Y,
              pdg::StandardDimension::Z})
            batch.materialize(pdg::DimensionId(dimension),
                              pdg::DimensionType::Double);
        batch.setSize(count);
        return batch;
    }

    pdg::PointBatch gather(PointView& view, pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch =
            makeBatch(static_cast<std::size_t>(view.size()), memory);
        const std::array<std::pair<Dimension::Id, pdg::DimensionId>, 3>
            dimensions = {
                std::pair{Dimension::Id::X,
                          pdg::DimensionId(pdg::StandardDimension::X)},
                std::pair{Dimension::Id::Y,
                          pdg::DimensionId(pdg::StandardDimension::Y)},
                std::pair{Dimension::Id::Z,
                          pdg::DimensionId(pdg::StandardDimension::Z)}};
        for (const auto& [pdalDimension, pdgDimension] : dimensions)
        {
            double* output = batch.data<double>(pdgDimension);
            for (PointId point = 0; point < view.size(); ++point)
                output[static_cast<std::size_t>(point)] =
                    view.getFieldAs<double>(pdalDimension, point);
        }
        return batch;
    }

    pdg::PointBatch gather(StreamPointTable& table,
                           const std::vector<PointId>& points,
                           pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch = makeBatch(points.size(), memory);
        const std::array<std::pair<Dimension::Id, pdg::DimensionId>, 3>
            dimensions = {
                std::pair{Dimension::Id::X,
                          pdg::DimensionId(pdg::StandardDimension::X)},
                std::pair{Dimension::Id::Y,
                          pdg::DimensionId(pdg::StandardDimension::Y)},
                std::pair{Dimension::Id::Z,
                          pdg::DimensionId(pdg::StandardDimension::Z)}};
        PointRef point(table, 0);
        for (const auto& [pdalDimension, pdgDimension] : dimensions)
        {
            double* output = batch.data<double>(pdgDimension);
            for (std::size_t offset = 0; offset < points.size(); ++offset)
            {
                point.setPointId(points[offset]);
                output[offset] = point.getFieldAs<double>(pdalDimension);
            }
        }
        return batch;
    }

#if PDG_HAS_CUDA
    pdg::BoundsResult applyCuda(pdg::PointBatch& host) const
    {
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_pdgDimensions, *deviceMemory);
        const std::array<pdg::DimensionId, 3> dimensions = {
            pdg::DimensionId(pdg::StandardDimension::X),
            pdg::DimensionId(pdg::StandardDimension::Y),
            pdg::DimensionId(pdg::StandardDimension::Z)};
        for (pdg::DimensionId dimension : dimensions)
            device.materialize(dimension, pdg::DimensionType::Double);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (pdg::DimensionId dimension : dimensions)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                device.rawData(dimension), host.rawData(dimension),
                host.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        return pdg::summarizeBounds(device,
                                    static_cast<std::uint64_t>(m_count));
    }

    template <typename Gather>
    bool tryCuda(Gather&& gatherBatch, bool requireCuda)
    {
        if (!m_pointSpec.empty() || !m_querySpec.empty())
            return false;
        try
        {
            if (pdg::cudaDevices().empty())
                return false;
            std::unique_ptr<pdg::MemoryResource> pinnedMemory =
                pdg::makeCudaPinnedMemoryResource();
            pdg::PointBatch host = gatherBatch(*pinnedMemory);
            insertBounds(applyCuda(host));
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

    void insertBounds(const pdg::BoundsResult& result)
    {
        const auto grow =
            [](double& current, const pdg::LocateResult& value, bool minimum)
        {
            if ((minimum && value.value < current) ||
                (!minimum && value.value > current))
                current = value.value;
        };
        grow(m_bounds.minx, result.minimum[0], true);
        grow(m_bounds.miny, result.minimum[1], true);
        grow(m_bounds.minz, result.minimum[2], true);
        grow(m_bounds.maxx, result.maximum[0], false);
        grow(m_bounds.maxy, result.maximum[1], false);
        grow(m_bounds.maxz, result.maximum[2], false);
        if (result.count > static_cast<std::uint64_t>(
                               (std::numeric_limits<PointId>::max)() - m_count))
            throwError("Point count exceeds the filters.info domain");
        m_count += static_cast<PointId>(result.count);
    }

    bool processPoint(PointRef& point)
    {
        const double x = point.getFieldAs<double>(Dimension::Id::X);
        const double y = point.getFieldAs<double>(Dimension::Id::Y);
        const double z = point.getFieldAs<double>(Dimension::Id::Z);
        m_bounds.grow(x, y, z);

        if (m_idCurrent != m_idList.end() && *m_idCurrent == m_count)
        {
            std::vector<char> data(m_pointSize);
            point.getPackedData(m_dimensions, data.data());
            m_results.emplace_back(m_count, 0.0, std::move(data));
            ++m_idCurrent;
        }
        else if (!m_querySpec.empty() && m_queryCount)
        {
            double distance =
                std::pow(x - m_queryX, 2) + std::pow(y - m_queryY, 2);
            if (!std::isnan(m_queryZ))
                distance += std::pow(z - m_queryZ, 2);
            if (m_results.size() < m_queryCount ||
                distance < m_results.back().m_distance)
            {
                std::vector<char> data(m_pointSize);
                point.getPackedData(m_dimensions, data.data());
                NearPoint nearPoint(m_count, distance, std::move(data));
                m_results.insert(std::upper_bound(m_results.begin(),
                                                  m_results.end(), nearPoint),
                                 std::move(nearPoint));
                if (m_results.size() > m_queryCount)
                    m_results.pop_back();
            }
        }
        ++m_count;
        return true;
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
            throwError("required exact CUDA hybrid info path was not used");
        if (usedCuda)
            return;

        PointRef point(view, 0);
        for (PointId id = 0; id < view.size(); ++id)
        {
            point.setPointId(id);
            processPoint(point);
        }
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
            throwError("required exact CUDA hybrid info path was not used");
        if (usedCuda)
            return;

        PointRef point(table, 0);
        for (PointId id : points)
        {
            point.setPointId(id);
            processPoint(point);
        }
    }

    void done(PointTableRef table) override
    {
        MetadataNode points("points");
        for (NearPoint& nearPoint : m_results)
        {
            MetadataNode node("point");
            const char* data = nearPoint.m_data.data();
            Everything value;
            for (DimType& dimension : m_dimensions)
            {
                const std::size_t size = Dimension::size(dimension.m_type);
                std::copy(data, data + size, reinterpret_cast<char*>(&value));
                node.add(table.layout()->dimName(dimension.m_id),
                         Utils::toDouble(value, dimension.m_type));
                data += size;
            }
            node.add("PointId", nearPoint.m_id);
            points.add(node);
        }
        if (points.hasChildren())
            getMetadata().add(points);

        getMetadata().add(Utils::toMetadata(m_bounds));
        getMetadata().add("num_points", m_count);

        std::string names;
        for (auto dimension = m_dimensions.begin();
             dimension != m_dimensions.end();)
        {
            names += table.layout()->dimName(dimension->m_id);
            if (++dimension != m_dimensions.end())
                names += ", ";
        }
        getMetadata().add("dimensions", names);
        getMetadata().add(table.anySpatialReference().toMetadata());
    }

    std::string m_querySpec;
    point_count_t m_queryCount;
    double m_queryX = 0.0;
    double m_queryY = 0.0;
    double m_queryZ;
    std::list<NearPoint> m_results;
    std::string m_pointSpec;
    PointIdList m_idList;
    PointIdList::const_iterator m_idCurrent;
    DimTypeList m_dimensions;
    std::size_t m_pointSize = 0;
    PointId m_count = 0;
    BOX3D m_bounds;
    std::unique_ptr<pdg::DimensionRegistry> m_pdgDimensions;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridInfoStage),
                                     "Internal exact PDG point information",
                                     ""};
CREATE_STATIC_STAGE(PdgInfoFilter, s_info)

} // namespace pdal
