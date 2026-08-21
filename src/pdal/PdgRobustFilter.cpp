#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Robust.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

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
pdg::DimensionType toPdgType(Dimension::Type type)
{
    switch (type)
    {
    case Dimension::Type::Signed8:
        return pdg::DimensionType::Signed8;
    case Dimension::Type::Signed16:
        return pdg::DimensionType::Signed16;
    case Dimension::Type::Signed32:
        return pdg::DimensionType::Signed32;
    case Dimension::Type::Signed64:
        return pdg::DimensionType::Signed64;
    case Dimension::Type::Unsigned8:
        return pdg::DimensionType::Unsigned8;
    case Dimension::Type::Unsigned16:
        return pdg::DimensionType::Unsigned16;
    case Dimension::Type::Unsigned32:
        return pdg::DimensionType::Unsigned32;
    case Dimension::Type::Unsigned64:
        return pdg::DimensionType::Unsigned64;
    case Dimension::Type::Float:
        return pdg::DimensionType::Float;
    case Dimension::Type::Double:
        return pdg::DimensionType::Double;
    case Dimension::Type::None:
        return pdg::DimensionType::None;
    }
    return pdg::DimensionType::None;
}
} // unnamed namespace

class PdgRobustFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridRobustStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("method", "Internal robust statistic", m_method);
        args.add("dimension", "Dimension on which to calculate statistics",
                 m_dimName);
        args.add("k", "Number of deviations", m_multiplier,
                 std::numeric_limits<double>::quiet_NaN());
        args.add("mad_multiplier", "MAD threshold multiplier", m_madMultiplier,
                 1.4862);
    }

    void prepared(PointTableRef table) override
    {
        if (m_method == "iqr")
            m_program.kind = pdg::RobustKind::Iqr;
        else if (m_method == "mad")
            m_program.kind = pdg::RobustKind::Mad;
        else
            throwError("invalid internal robust method: " + m_method);
        if (std::isnan(m_multiplier))
            m_multiplier = m_program.kind == pdg::RobustKind::Iqr ? 1.5 : 2.0;
        m_program.multiplier = m_multiplier;
        m_program.madMultiplier = m_madMultiplier;

        PointLayoutPtr layout(table.layout());
        m_pdalId = layout->findDim(m_dimName);
        if (m_pdalId == Dimension::Id::Unknown)
            throwError("Dimension '" + m_dimName + "' does not exist.");
        m_pdalType = layout->dimType(m_pdalId);
        m_physicalType = toPdgType(m_pdalType);
        if (m_physicalType == pdg::DimensionType::None)
            throwError("Unsupported dimension type for '" + m_dimName + "'.");

        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        const pdg::DimensionDefinition* definition =
            m_dimensions->find(m_dimName);
        if (!definition)
            definition =
                &m_dimensions->registerCustom(m_dimName, m_physicalType);
        m_program.dimension = definition->id;
    }

    void gather(PointView& view, pdg::PointBatch& batch) const
    {
        batch.setSize(static_cast<std::size_t>(view.size()));
        std::byte* destination =
            static_cast<std::byte*>(batch.rawData(m_program.dimension));
        const std::size_t stride = pdg::dimensionTypeSize(m_physicalType);
        for (PointId point = 0; point < view.size(); ++point)
            view.getField(
                reinterpret_cast<char*>(
                    destination + static_cast<std::size_t>(point) * stride),
                m_pdalId, m_pdalType, point);
    }

    PointViewSet run(PointViewPtr view) override
    {
        PointViewSet resultSet;
        PointViewPtr output = view->makeNew();
        if (view->empty())
        {
            resultSet.insert(output);
            return resultSet;
        }

        const std::size_t count = static_cast<std::size_t>(view->size());
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::HostMemoryResource hostMemory;
        pdg::PointBatch host(count, coordinates, *m_dimensions, hostMemory);
        host.materialize(m_program.dimension, m_physicalType);
        gather(*view, host);
        std::vector<std::uint8_t> keep(count);

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && pdg::robustSupportsExactDevice(host, m_program))
        {
            try
            {
                if (!pdg::cudaDevices().empty())
                {
                    std::unique_ptr<pdg::MemoryResource> deviceMemory =
                        pdg::makeCudaMemoryResource();
                    pdg::PointBatch device(count, coordinates, *m_dimensions,
                                           *deviceMemory);
                    device.materialize(m_program.dimension, m_physicalType);
                    device.setSize(count);
                    std::unique_ptr<pdg::Allocation> deviceKeep =
                        deviceMemory->allocate(count, alignof(std::uint8_t));
                    const cudaStream_t stream =
                        static_cast<cudaStream_t>(device.nativeStreamHandle());
                    PDG_CUDA_CHECK(cudaMemcpyAsync(
                        device.rawData(m_program.dimension),
                        host.rawData(m_program.dimension),
                        count * pdg::dimensionTypeSize(m_physicalType),
                        cudaMemcpyHostToDevice, stream));
                    static_cast<void>(pdg::evaluateRobust(
                        device, m_program,
                        static_cast<std::uint8_t*>(deviceKeep->data())));
                    PDG_CUDA_CHECK(
                        cudaMemcpyAsync(keep.data(), deviceKeep->data(), count,
                                        cudaMemcpyDeviceToHost, stream));
                    PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                    usedCuda = true;
                }
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
            }
        }
#else
        (void)requestCuda;
#endif
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid robust path was not used");
        if (!usedCuda)
            static_cast<void>(
                pdg::evaluateRobust(host, m_program, keep.data()));

        for (PointId point = 0; point < view->size(); ++point)
            if (keep[static_cast<std::size_t>(point)])
                output->appendPoint(*view, point);
        resultSet.insert(output);
        return resultSet;
    }

    std::string m_method;
    std::string m_dimName;
    double m_multiplier = std::numeric_limits<double>::quiet_NaN();
    double m_madMultiplier = 1.4862;
    Dimension::Id m_pdalId = Dimension::Id::Unknown;
    Dimension::Type m_pdalType = Dimension::Type::None;
    pdg::DimensionType m_physicalType = pdg::DimensionType::None;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::RobustProgram m_program;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridRobustStage),
    "Internal exact PDG robust-statistics selection", ""};

CREATE_STATIC_STAGE(PdgRobustFilter, s_info)

} // namespace pdal
