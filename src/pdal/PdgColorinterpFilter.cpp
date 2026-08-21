#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>

#include <filters/ColorInterpRamps.hpp>
#include <filters/StatsFilter.hpp>
#include <pdal/BatchStreamable.hpp>
#include <pdal/Filter.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/private/gdal/GDALUtils.hpp>
#include <pdal/private/gdal/Raster.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#include <cpl_vsi.h>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pdal
{

namespace
{
void installBuiltInRamp(std::string& filename, const char* name,
                        unsigned char* bytes, std::size_t size)
{
    if (!Utils::iequals(name, filename))
        return;
    filename = "/vsimem/" + std::string(name) + ".png";
    VSILFILE* file = VSIFileFromMemBuffer(
        filename.c_str(), bytes, static_cast<vsi_l_offset>(size), false);
    static_cast<void>(file);
}

std::shared_ptr<gdal::Raster> openPdgRamp(std::string& filename)
{
    installBuiltInRamp(filename, "awesome_green", awesome_green,
                       sizeof(awesome_green));
    installBuiltInRamp(filename, "black_orange", black_orange,
                       sizeof(black_orange));
    installBuiltInRamp(filename, "blue_hue", blue_hue, sizeof(blue_hue));
    installBuiltInRamp(filename, "blue_red", blue_red, sizeof(blue_red));
    installBuiltInRamp(filename, "heat_map", heat_map, sizeof(heat_map));
    installBuiltInRamp(filename, "pestel_shades", pestel_shades,
                       sizeof(pestel_shades));
    installBuiltInRamp(filename, "blue_orange", blue_orange,
                       sizeof(blue_orange));
    return std::make_shared<gdal::Raster>(filename.c_str());
}
} // unnamed namespace

// Behaviorally derived from the pinned upstream ColorinterpFilter. The range
// members deliberately retain upstream's per-view mutation semantics.
class PdgColorinterpFilter final : public Filter, public BatchStreamable
{
public:
    std::string getName() const override
    {
        return "filters.colorinterp";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimension", "Dimension to interpolate", m_dimName, "Z");
        args.add("minimum", "Minimum value to use for scaling", m_minimum,
                 std::numeric_limits<double>::quiet_NaN());
        args.add("maximum", "Maximum value to use for scaling", m_maximum,
                 std::numeric_limits<double>::quiet_NaN());
        args.add("clamp",
                 "Clamp and color values outside the range [minimum, maximum]",
                 m_clamp, false);
        args.add("ramp", "GDAL-readable color ramp image to use", m_rampName,
                 "pestel_shades");
        args.add("invert", "Invert the ramp direction", m_invert, false);
        args.add("mad",
                 "Use Median Absolute Deviation to compute ramp bounds in "
                 "combination with 'k' ",
                 m_useMad, false);
        args.add("mad_multiplier", "MAD threshold multiplier", m_madMultiplier,
                 1.4862);
        args.add("k", "Number of deviations to compute minimum/maximum ", m_k,
                 0.0);
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDims(
            {Dimension::Id::Red, Dimension::Id::Green, Dimension::Id::Blue});
    }

    void prepared(PointTableRef table) override
    {
        PointLayoutPtr layout(table.layout());
        m_pdalDimension = layout->findDim(m_dimName);
        if (m_pdalDimension == Dimension::Id::Unknown)
            throwError("Dimension '" + m_dimName + "' does not exist.");
        if (!std::isnan(m_minimum) && !std::isnan(m_maximum) &&
            m_maximum <= m_minimum)
            throwError("Specified 'minimum' value must be less than "
                       "'maximum' value.");

        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        m_valueDimension =
            m_dimensions
                ->registerCustom("PdgColorValue", pdg::DimensionType::Double)
                .id;
    }

    void ready(PointTableRef) override
    {
        gdal::registerDrivers();
        m_raster = openPdgRamp(m_rampName);
        const gdal::GDALError error = m_raster->open();
        if (error != gdal::GDALError::None &&
            error != gdal::GDALError::NoTransform)
            throwError(m_raster->errorMsg());

        log()->get(LogLevel::Debug)
            << getName() << " raster connection: " << m_raster->filename()
            << std::endl;
        m_raster->readBand(m_redBand, 1);
        m_raster->readBand(m_greenBand, 2);
        m_raster->readBand(m_blueBand, 3);
    }

    void computeRange(PointView& view)
    {
        double median = 0.0;
        double mad = 0.0;
        if (m_k != 0.0)
        {
            std::vector<double> values(view.size());
            stats::Summary summary(Dimension::name(m_pdalDimension),
                                   stats::Summary::NoEnum, false);
            for (PointId point = 0; point < view.size(); ++point)
            {
                const double value =
                    view.getFieldAs<double>(m_pdalDimension, point);
                summary.insert(value);
                values[static_cast<std::size_t>(point)] = value;
            }
            const auto computeMedian = [](std::vector<double> candidates)
            {
                std::nth_element(candidates.begin(),
                                 candidates.begin() + candidates.size() / 2U,
                                 candidates.end());
                return *(candidates.begin() + candidates.size() / 2U);
            };
            median = computeMedian(values);
            if (m_useMad)
            {
                std::transform(values.begin(), values.end(), values.begin(),
                               [median](double value)
                               { return std::fabs(value - median); });
                mad = computeMedian(values);
                const double threshold = mad * m_madMultiplier * m_k;
                m_minimum = median - threshold;
                m_maximum = median + threshold;
                log()->get(LogLevel::Debug)
                    << getName() << " mad " << mad << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " median " << median << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " minimum " << m_minimum << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " maximum " << m_maximum << std::endl;
            }
            else
            {
                const double threshold = m_k * summary.sampleStddev();
                m_minimum = median - threshold;
                m_maximum = median + threshold;
                log()->get(LogLevel::Debug) << getName() << " stddev threshold "
                                            << threshold << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " median " << median << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " minimum " << m_minimum << std::endl;
                log()->get(LogLevel::Debug)
                    << getName() << " maximum " << m_maximum << std::endl;
            }
        }
        else if (std::isnan(m_minimum) || std::isnan(m_maximum))
        {
            stats::Summary summary(Dimension::name(m_pdalDimension),
                                   stats::Summary::NoEnum, false);
            for (PointId point = 0; point < view.size(); ++point)
                summary.insert(view.getFieldAs<double>(m_pdalDimension, point));
            if (std::isnan(m_minimum))
                m_minimum = summary.minimum();
            if (std::isnan(m_maximum))
                m_maximum = summary.maximum();
        }
    }

    bool pipelineStreamable() const override
    {
        if (std::isnan(m_minimum) || std::isnan(m_maximum))
            return false;
        return Streamable::pipelineStreamable();
    }

    void applyHostPoint(PointRef& point) const
    {
        double value = point.getFieldAs<double>(m_pdalDimension);
        if (m_clamp)
            value = Utils::clamp(value, m_minimum, m_maximum);
        if (value < m_minimum || value > m_maximum)
            return;
        const double factor = (value - m_minimum) / (m_maximum - m_minimum);
        const std::size_t width = m_redBand.size();
        std::size_t position = static_cast<std::size_t>(
            std::floor(factor * static_cast<double>(width)));
        position = (std::min)(position, width - 1U);
        if (m_invert)
            position = (width - 1U) - position;
        point.setField(Dimension::Id::Red, m_redBand[position]);
        point.setField(Dimension::Id::Blue, m_blueBand[position]);
        point.setField(Dimension::Id::Green, m_greenBand[position]);
    }

    void applyHost(PointView& view) const
    {
        PointRef point(view, 0);
        for (PointId id = 0; id < view.size(); ++id)
        {
            point.setPointId(id);
            applyHostPoint(point);
        }
    }

    void applyHost(StreamPointTable& table,
                   const std::vector<PointId>& points) const
    {
        PointRef point(table, 0);
        for (PointId id : points)
        {
            point.setPointId(id);
            applyHostPoint(point);
        }
    }

    pdg::PointBatch makeBatch(std::size_t count,
                              pdg::MemoryResource& memory) const
    {
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::PointBatch batch(count, coordinates, *m_dimensions, memory);
        batch.materialize(m_valueDimension, pdg::DimensionType::Double);
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
            batch.materialize(pdg::DimensionId(dimension),
                              pdg::DimensionType::Unsigned16);
        batch.setSize(count);
        return batch;
    }

    pdg::PointBatch gather(PointView& view, pdg::MemoryResource& memory) const
    {
        const std::size_t count = static_cast<std::size_t>(view.size());
        pdg::PointBatch batch = makeBatch(count, memory);
        for (PointId point = 0; point < view.size(); ++point)
        {
            const std::size_t offset = static_cast<std::size_t>(point);
            batch.data<double>(m_valueDimension)[offset] =
                view.getFieldAs<double>(m_pdalDimension, point);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Red))[offset] =
                view.getFieldAs<std::uint16_t>(Dimension::Id::Red, point);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Green))[offset] =
                view.getFieldAs<std::uint16_t>(Dimension::Id::Green, point);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Blue))[offset] =
                view.getFieldAs<std::uint16_t>(Dimension::Id::Blue, point);
        }
        return batch;
    }

    pdg::PointBatch gather(StreamPointTable& table,
                           const std::vector<PointId>& points,
                           pdg::MemoryResource& memory) const
    {
        pdg::PointBatch batch = makeBatch(points.size(), memory);
        PointRef point(table, 0);
        for (std::size_t offset = 0; offset < points.size(); ++offset)
        {
            point.setPointId(points[offset]);
            batch.data<double>(m_valueDimension)[offset] =
                point.getFieldAs<double>(m_pdalDimension);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Red))[offset] =
                point.getFieldAs<std::uint16_t>(Dimension::Id::Red);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Green))[offset] =
                point.getFieldAs<std::uint16_t>(Dimension::Id::Green);
            batch.data<std::uint16_t>(
                pdg::DimensionId(pdg::StandardDimension::Blue))[offset] =
                point.getFieldAs<std::uint16_t>(Dimension::Id::Blue);
        }
        return batch;
    }

    pdg::ColorMapProgram colorMapProgram() const
    {
        pdg::ColorMapProgram program;
        program.value = m_valueDimension;
        program.minimum = m_minimum;
        program.maximum = m_maximum;
        program.clamp = m_clamp;
        program.invert = m_invert;
        return program;
    }

    pdg::ColorRampView colorRamp() const
    {
        return {m_redBand.data(), m_greenBand.data(), m_blueBand.data(),
                m_redBand.size()};
    }

#if PDG_HAS_CUDA
    void applyCuda(pdg::PointBatch& host,
                   const pdg::ColorMapProgram& program) const
    {
        const std::size_t count = host.size();
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(count, host.coordinateEncoding(), *m_dimensions,
                               *deviceMemory);
        device.materialize(m_valueDimension, pdg::DimensionType::Double);
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
            device.materialize(pdg::DimensionId(dimension),
                               pdg::DimensionType::Unsigned16);
        device.setSize(count);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (const pdg::DimensionId dimension :
             {m_valueDimension, pdg::DimensionId(pdg::StandardDimension::Red),
              pdg::DimensionId(pdg::StandardDimension::Green),
              pdg::DimensionId(pdg::StandardDimension::Blue)})
        {
            const std::size_t bytes =
                count *
                pdg::dimensionTypeSize(host.columnInfo(dimension).physicalType);
            PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                           host.rawData(dimension), bytes,
                                           cudaMemcpyHostToDevice, stream));
        }
        std::unique_ptr<pdg::Allocation> deviceRed =
            deviceMemory->allocate(m_redBand.size(), alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceGreen =
            deviceMemory->allocate(m_greenBand.size(), alignof(std::uint8_t));
        std::unique_ptr<pdg::Allocation> deviceBlue =
            deviceMemory->allocate(m_blueBand.size(), alignof(std::uint8_t));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceRed->data(), m_redBand.data(),
                                       m_redBand.size(), cudaMemcpyHostToDevice,
                                       stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceGreen->data(), m_greenBand.data(),
                                       m_greenBand.size(),
                                       cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(deviceBlue->data(), m_blueBand.data(),
                                       m_blueBand.size(),
                                       cudaMemcpyHostToDevice, stream));
        pdg::applyColorMap(
            device, program,
            {static_cast<const std::uint8_t*>(deviceRed->data()),
             static_cast<const std::uint8_t*>(deviceGreen->data()),
             static_cast<const std::uint8_t*>(deviceBlue->data()),
             m_redBand.size()});
        for (const pdg::StandardDimension dimension :
             {pdg::StandardDimension::Red, pdg::StandardDimension::Green,
              pdg::StandardDimension::Blue})
        {
            const pdg::DimensionId id(dimension);
            PDG_CUDA_CHECK(cudaMemcpyAsync(host.rawData(id), device.rawData(id),
                                           count * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, stream));
        }
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
#endif

    void scatter(PointView& view, const pdg::PointBatch& host) const
    {
        for (PointId point = 0; point < view.size(); ++point)
        {
            const std::size_t offset = static_cast<std::size_t>(point);
            view.setField(Dimension::Id::Red, point,
                          host.data<std::uint16_t>(pdg::DimensionId(
                              pdg::StandardDimension::Red))[offset]);
            view.setField(Dimension::Id::Green, point,
                          host.data<std::uint16_t>(pdg::DimensionId(
                              pdg::StandardDimension::Green))[offset]);
            view.setField(Dimension::Id::Blue, point,
                          host.data<std::uint16_t>(pdg::DimensionId(
                              pdg::StandardDimension::Blue))[offset]);
        }
    }

    void scatter(StreamPointTable& table, const std::vector<PointId>& points,
                 const pdg::PointBatch& host) const
    {
        PointRef point(table, 0);
        for (std::size_t offset = 0; offset < points.size(); ++offset)
        {
            point.setPointId(points[offset]);
            point.setField(Dimension::Id::Red,
                           host.data<std::uint16_t>(pdg::DimensionId(
                               pdg::StandardDimension::Red))[offset]);
            point.setField(Dimension::Id::Green,
                           host.data<std::uint16_t>(pdg::DimensionId(
                               pdg::StandardDimension::Green))[offset]);
            point.setField(Dimension::Id::Blue,
                           host.data<std::uint16_t>(pdg::DimensionId(
                               pdg::StandardDimension::Blue))[offset]);
        }
    }

    bool tryCuda(pdg::PointBatch& host, const pdg::ColorMapProgram& program,
                 bool requireCuda) const
    {
#if PDG_HAS_CUDA
        const pdg::ColorRampView ramp = colorRamp();
        if (m_greenBand.size() != m_redBand.size() ||
            m_blueBand.size() != m_redBand.size() ||
            !pdg::colorMapMaySupportExactDevice(host, program, ramp))
            return false;
        try
        {
            if (!pdg::cudaDevices().empty())
            {
                applyCuda(host, program);
                return true;
            }
        }
        catch (const pdg::CudaError&)
        {
            if (requireCuda)
                throw;
        }
#else
        static_cast<void>(host);
        static_cast<void>(program);
        static_cast<void>(requireCuda);
#endif
        return false;
    }

    void processBatch(StreamPointTable& table,
                      point_count_t pointLimit) override
    {
        std::vector<PointId> points;
        points.reserve(static_cast<std::size_t>(pointLimit));
        for (PointId point = 0; point < pointLimit; ++point)
            if (!table.skip(point))
                points.push_back(point);
        if (points.empty())
            return;

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
        if (requestCuda)
        {
            pdg::HostMemoryResource hostMemory;
            pdg::PointBatch host = gather(table, points, hostMemory);
            usedCuda = tryCuda(host, colorMapProgram(), requireCuda);
            if (usedCuda)
                scatter(table, points, host);
        }
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid colorinterp path was not used");
        if (!usedCuda)
            applyHost(table, points);
    }

    void filter(PointView& view) override
    {
        if (std::getenv("PDG_REQUIRE_STREAMING_HYBRID"))
            throwError("required streaming hybrid path was not used");
        computeRange(view);
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
        if (requestCuda && !view.empty())
        {
            pdg::HostMemoryResource hostMemory;
            pdg::PointBatch host = gather(view, hostMemory);
            usedCuda = tryCuda(host, colorMapProgram(), requireCuda);
            if (usedCuda)
                scatter(view, host);
        }
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid colorinterp path was not used");
        if (!usedCuda)
            applyHost(view);
    }

    std::string m_dimName = "Z";
    // Match the upstream constructor so the pre-prepare streamability probe
    // remains optimistic; ProgramArgs installs NaN defaults during prepare.
    double m_minimum = 0.0;
    double m_maximum = 0.0;
    bool m_clamp = false;
    std::string m_rampName = "pestel_shades";
    bool m_invert = false;
    bool m_useMad = false;
    double m_madMultiplier = 1.4862;
    double m_k = 0.0;
    Dimension::Id m_pdalDimension = Dimension::Id::Z;
    std::shared_ptr<gdal::Raster> m_raster;
    std::vector<std::uint8_t> m_redBand;
    std::vector<std::uint8_t> m_greenBand;
    std::vector<std::uint8_t> m_blueBand;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::DimensionId m_valueDimension;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridColorinterpStage),
    "Internal exact PDG dimension color interpolation", ""};

CREATE_STATIC_STAGE(PdgColorinterpFilter, s_info)

} // namespace pdal
