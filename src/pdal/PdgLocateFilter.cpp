#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Locate.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

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

std::size_t configuredChunkPoints(std::size_t fallback)
{
    const char* configured = std::getenv("PDG_CUDA_CHUNK_POINTS");
    if (!configured || !*configured)
        return fallback;
    const std::string_view text(configured);
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(
            "PDG_CUDA_CHUNK_POINTS must be a positive integer");
    return value;
}
} // unnamed namespace

class PdgLocateFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridLocateStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimension", "Dimension in which to locate max", m_dimName);
        args.add("minmax", "Whether to search for the minimum or maximum value",
                 m_minmax, "max");
    }

    void prepared(PointTableRef table) override
    {
        PointLayoutPtr layout(table.layout());
        m_pdalId = layout->findDim(m_dimName);
        if (m_pdalId == Dimension::Id::Unknown)
            throwError("Invalid dimension '" + m_dimName + "'.");
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
        if (Utils::iequals("min", m_minmax))
            m_program.kind = pdg::LocateKind::Minimum;
        else if (Utils::iequals("max", m_minmax))
            m_program.kind = pdg::LocateKind::Maximum;
        else
            m_program.kind = pdg::LocateKind::None;
    }

    void gather(PointView& view, std::size_t offset, std::size_t count,
                pdg::PointBatch& batch) const
    {
        batch.setSize(count);
        std::byte* destination =
            static_cast<std::byte*>(batch.rawData(m_program.dimension));
        const std::size_t stride = pdg::dimensionTypeSize(m_physicalType);
        for (std::size_t point = 0; point < count; ++point)
            view.getField(reinterpret_cast<char*>(destination + point * stride),
                          m_pdalId, m_pdalType,
                          static_cast<PointId>(offset + point));
    }

    pdg::LocateResult executeHost(PointView& view, std::size_t capacity) const
    {
        pdg::HostMemoryResource memory;
        pdg::PointBatch batch(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, memory);
        batch.materialize(m_program.dimension, m_physicalType);
        pdg::LocateResult result;
        for (std::size_t offset = 0; offset < view.size(); offset += capacity)
        {
            const std::size_t count =
                (std::min)(capacity,
                           static_cast<std::size_t>(view.size()) - offset);
            gather(view, offset, count, batch);
            result = pdg::mergeLocateResults(
                m_program, result,
                pdg::locateExtreme(batch, m_program,
                                   static_cast<std::uint64_t>(offset)));
        }
        return result;
    }

#if PDG_HAS_CUDA
    pdg::LocateResult executeCuda(PointView& view, std::size_t capacity) const
    {
        std::unique_ptr<pdg::MemoryResource> pinned =
            pdg::makeCudaPinnedMemoryResource();
        std::unique_ptr<pdg::MemoryResource> device =
            pdg::makeCudaMemoryResource(capacity *
                                        pdg::dimensionTypeSize(m_physicalType));
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::PointBatch staging(capacity, coordinates, *m_dimensions, *pinned);
        pdg::PointBatch batch(capacity, coordinates, *m_dimensions, *device);
        staging.materialize(m_program.dimension, m_physicalType);
        batch.materialize(m_program.dimension, m_physicalType);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(batch.nativeStreamHandle());
        pdg::LocateResult result;
        for (std::size_t offset = 0; offset < view.size(); offset += capacity)
        {
            const std::size_t count =
                (std::min)(capacity,
                           static_cast<std::size_t>(view.size()) - offset);
            gather(view, offset, count, staging);
            batch.setSize(count);
            const std::size_t bytes =
                count * pdg::dimensionTypeSize(m_physicalType);
            PDG_CUDA_CHECK(cudaMemcpyAsync(batch.rawData(m_program.dimension),
                                           staging.rawData(m_program.dimension),
                                           bytes, cudaMemcpyHostToDevice,
                                           stream));
            result = pdg::mergeLocateResults(
                m_program, result,
                pdg::locateExtreme(batch, m_program,
                                   static_cast<std::uint64_t>(offset)));
        }
        return result;
    }
#endif

    PointViewSet run(PointViewPtr view) override
    {
        PointViewSet resultSet;
        PointViewPtr output = view->makeNew();
        if (!view->size() || m_program.kind == pdg::LocateKind::None)
        {
            resultSet.insert(output);
            return resultSet;
        }

        const std::size_t capacity =
            (std::min)(static_cast<std::size_t>(view->size()),
                       configuredChunkPoints(1U << 20U));
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
        pdg::LocateResult located;
#if PDG_HAS_CUDA
        if (requestCuda)
        {
            bool available = false;
            try
            {
                available = !pdg::cudaDevices().empty();
                if (available)
                {
                    located = executeCuda(*view, capacity);
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
            throwError("required exact CUDA hybrid locate path was not used");
        if (!usedCuda)
            located = executeHost(*view, capacity);
        if (!located.hasPoints || located.index >= view->size())
            throwError("locate reduction returned an invalid point index");
        output->appendPoint(*view, static_cast<PointId>(located.index));
        resultSet.insert(output);
        return resultSet;
    }

    std::string m_dimName;
    std::string m_minmax = "max";
    Dimension::Id m_pdalId = Dimension::Id::Unknown;
    Dimension::Type m_pdalType = Dimension::Type::None;
    pdg::DimensionType m_physicalType = pdg::DimensionType::None;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::LocateProgram m_program;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridLocateStage),
                                     "Internal exact PDG locate reduction", ""};

CREATE_STATIC_STAGE(PdgLocateFilter, s_info)

} // namespace pdal
