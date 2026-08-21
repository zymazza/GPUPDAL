#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/io/LasTranslate.hpp>
#include <pdg/stages/Ordering.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
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
#if PDG_HAS_CUDA
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#endif

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

class PdgOrderFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridOrderStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimensions", "Dimensions and ordering on which to sort",
                 m_dimNames)
            .setPositional();
        args.addSynonym("dimensions", "dimension");
        args.add("order", "Sort order ASC or DESC", m_order, "ASC");
        args.add("algorithm", "NORMAL or STABLE", m_algorithm, "NORMAL");
        args.add("pdg_resident_context",
                 "Internal planner-owned resident execution marker",
                 m_residentContext, false)
            .setHidden();
        args.add("pdg_execution_region",
                 "Internal planner-owned execution region identifier",
                 m_executionRegion, std::uint64_t(0))
            .setHidden();
    }

    void prepared(PointTableRef table) override
    {
        if (m_dimNames.empty())
            throwError("At least one valid dimension name must be provided!");
        if (Utils::iequals(m_order, "asc"))
            m_program.direction = pdg::OrderingDirection::Ascending;
        else if (Utils::iequals(m_order, "desc"))
            m_program.direction = pdg::OrderingDirection::Descending;
        else
            throwError("Invalid value for argument 'order'.");
        if (Utils::iequals(m_algorithm, "normal"))
            m_program.algorithm = pdg::OrderingAlgorithm::Normal;
        else if (Utils::iequals(m_algorithm, "stable"))
            m_program.algorithm = pdg::OrderingAlgorithm::Stable;
        else
            throwError("Invalid value for argument 'algorithm'.");

        PointLayoutPtr layout(table.layout());
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        for (const std::string& name : m_dimNames)
        {
            const Dimension::Id pdalId = layout->findDim(name);
            if (pdalId == Dimension::Id::Unknown)
                throwError("Cannot sort because dimension '" + name +
                           "' was not found.");
            const Dimension::Type pdalType = layout->dimType(pdalId);
            const pdg::DimensionType physicalType = toPdgType(pdalType);
            if (physicalType == pdg::DimensionType::None)
                throwError("Unsupported sort dimension type for '" + name +
                           "'.");
            const pdg::DimensionDefinition* definition =
                m_dimensions->find(name);
            if (!definition)
                definition = &m_dimensions->registerCustom(name, physicalType);
            m_program.dimensions.push_back(definition->id);
            m_pdalIds.push_back(pdalId);
            m_pdalTypes.push_back(pdalType);
            m_physicalTypes.push_back(physicalType);
        }
    }

    void prerun(const PointViewSet& views) override
    {
        if (!m_residentContext)
            return;
        if (views.size() == 1U && (*views.begin())->empty())
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const std::size_t region =
                static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(**views.begin(), region);
            context.endDelegatedRegion(**views.begin(), region);
        }
    }

    bool gather(PointView& view, pdg::PointBatch& batch,
                bool& mappedSourceUsed) const
    {
        mappedSourceUsed = false;
        batch.setSize(static_cast<std::size_t>(view.size()));
        const pdg::DimensionId z(pdg::StandardDimension::Z);
        const bool strictResidentOrdering =
            m_residentContext &&
            m_program.dimensions == std::vector<pdg::DimensionId>{z} &&
            m_program.direction == pdg::OrderingDirection::Ascending &&
            m_program.algorithm == pdg::OrderingAlgorithm::Normal &&
            m_pdalIds == std::vector<Dimension::Id>{Dimension::Id::Z} &&
            m_pdalTypes ==
                std::vector<Dimension::Type>{Dimension::Type::Double} &&
            m_physicalTypes ==
                std::vector<pdg::DimensionType>{pdg::DimensionType::Double};
        if (m_residentContext && !strictResidentOrdering)
            return false;
        const std::span<const std::byte> mapped =
            pdg_detail::activeCudaLasSource();
        if (strictResidentOrdering && !mapped.empty())
        {
            const pdg::las::FileView input(mapped);
            if (!pdg::las::supportsDefaultTranslation(input) ||
                input.header().pointCount != view.size())
                return false;
            const pdg::CoordinateEncoding inputEncoding =
                input.header().coordinateEncoding();
            double* destination = batch.data<double>(z);
            std::vector<std::uint8_t> seen(
                static_cast<std::size_t>(view.size()), 0U);
            for (PointId point = 0U; point < view.size(); ++point)
            {
                const PointId source =
                    pdg_detail::ResidentPointViewAccess::tableId(view, point);
                if (source >= view.size() ||
                    seen[static_cast<std::size_t>(source)] != 0U)
                    return false;
                seen[static_cast<std::size_t>(source)] = 1U;
                destination[static_cast<std::size_t>(point)] =
                    inputEncoding.decode(
                        2U, input.rawCoordinate(
                                static_cast<std::size_t>(source), 2U));
                if (!std::isfinite(
                        destination[static_cast<std::size_t>(point)]))
                    return false;
            }
            mappedSourceUsed = true;
            return true;
        }
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            std::byte* destination = static_cast<std::byte*>(
                batch.rawData(m_program.dimensions[key]));
            const std::size_t stride =
                pdg::dimensionTypeSize(m_physicalTypes[key]);
            for (PointId point = 0; point < view.size(); ++point)
                view.getField(
                    reinterpret_cast<char*>(
                        destination + static_cast<std::size_t>(point) * stride),
                    m_pdalIds[key], m_pdalTypes[key], point);
        }
        return true;
    }

#if PDG_HAS_CUDA
    bool executeCuda(const pdg::PointBatch& host, PointIdList& permutation,
                     pdg::MemoryResource& deviceMemory) const
    {
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, deviceMemory);
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const pdg::DimensionId dimension = m_program.dimensions[key];
            if (device.has(dimension))
                continue;
            device.materialize(dimension, m_physicalTypes[key]);
        }
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const pdg::DimensionId dimension = m_program.dimensions[key];
            if (key && dimension == m_program.dimensions[key - 1U])
                continue;
            const std::size_t bytes =
                host.size() * pdg::dimensionTypeSize(m_physicalTypes[key]);
            PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                           host.rawData(dimension), bytes,
                                           cudaMemcpyHostToDevice, stream));
        }
        if (m_residentContext)
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::HostToDevice,
                static_cast<std::size_t>(m_executionRegion),
                host.size() * sizeof(double));
        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory.allocate(host.size() * sizeof(std::uint64_t),
                                  alignof(std::uint64_t));
        NvtxRange range("pdg::filters.sort.order");
        const pdg::OrderingResult ordered = pdg::orderPoints(
            device, m_program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        if (!ordered.exact)
            return false;
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation.data(),
                                       devicePermutation->data(),
                                       host.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        if (m_residentContext)
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::DeviceToHost,
                static_cast<std::size_t>(m_executionRegion),
                host.size() * sizeof(std::uint64_t));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
#endif

    void filter(PointView& view) override
    {
        if (view.empty())
            return;
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().beginDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));
        pdg_detail::ResidentExecutionContext* resident =
            m_residentContext ? &pdg_detail::requireResidentExecutionContext()
                              : nullptr;
        const std::size_t executionRegion =
            static_cast<std::size_t>(m_executionRegion);
        const std::size_t count = static_cast<std::size_t>(view.size());
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        std::unique_ptr<pdg::MemoryResource> ownedHostMemory;
        pdg::MemoryResource* hostMemory = nullptr;
#if PDG_HAS_CUDA
        if (resident)
            hostMemory =
                &resident->delegatedStagingMemory(view, executionRegion);
        else
#endif
        {
            ownedHostMemory = std::make_unique<pdg::HostMemoryResource>();
            hostMemory = ownedHostMemory.get();
        }
        pdg::PointBatch host(count, coordinates, *m_dimensions, *hostMemory);
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const pdg::DimensionId dimension = m_program.dimensions[key];
            if (!host.has(dimension))
                host.materialize(dimension, m_physicalTypes[key]);
        }
        bool mappedSourceUsed = false;
        const bool gathered = gather(view, host, mappedSourceUsed);

        PointIdList permutation(count);
        const bool requireCuda =
            m_residentContext || std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (gathered && requestCuda &&
            pdg::orderingMaySupportExactDevice(host, m_program))
        {
            try
            {
                if (!pdg::cudaDevices().empty())
                {
                    std::unique_ptr<pdg::MemoryResource> ownedDeviceMemory;
                    pdg::MemoryResource* deviceMemory = nullptr;
                    if (resident)
                        deviceMemory = &resident->delegatedDeviceMemory(
                            view, executionRegion);
                    else
                    {
                        ownedDeviceMemory = pdg::makeCudaMemoryResource();
                        deviceMemory = ownedDeviceMemory.get();
                    }
                    usedCuda = executeCuda(host, permutation, *deviceMemory);
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
            throwError("required exact CUDA hybrid ordering path was not used");
        if (!usedCuda)
            static_cast<void>(
                pdg::orderPoints(host, m_program, permutation.data()));
        else if (mappedSourceUsed)
            pdg_detail::markActiveCudaLasSourceUsed();
        view.applyPermutation(permutation);
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().endDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));
    }

    StringList m_dimNames;
    std::string m_order = "ASC";
    std::string m_algorithm = "NORMAL";
    std::vector<Dimension::Id> m_pdalIds;
    std::vector<Dimension::Type> m_pdalTypes;
    std::vector<pdg::DimensionType> m_physicalTypes;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::OrderingProgram m_program;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridOrderStage),
                                     "Internal exact PDG point ordering", ""};

CREATE_STATIC_STAGE(PdgOrderFilter, s_info)

} // namespace pdal
