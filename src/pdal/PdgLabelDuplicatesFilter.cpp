#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/LabelDuplicates.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

// Exact compatibility wrapper for the pinned LabelDuplicatesFilter. Upstream
// does not sort: it compares each row only with its immediate predecessor
// after converting every selected field to double.
class PdgLabelDuplicatesFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.label_duplicates";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimensions",
                 "Dimensions to use to declare points as duplicate",
                 m_dimNames);
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
        args.add("pdg_region_radius",
                 "Internal resident-region radius envelope",
                 m_region.maximumRadius, 0.0)
            .setHidden();
        args.add("pdg_region_dimensions",
                 "Internal resident-region spatial dimensions",
                 m_region.dimensions, std::uint32_t(3))
            .setHidden();
        args.add("pdg_region_index_required",
                 "Internal resident-region index requirement",
                 m_region.indexRequired, false)
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
        layout->registerDim(Dimension::Id::Duplicate);
    }

    void prepared(PointTableRef table) override
    {
        PointLayoutPtr layout = table.layout();
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        m_program = {};
        m_pdalIds.clear();
        m_pdalTypes.clear();
        m_physicalTypes.clear();
        for (const std::string& name : m_dimNames)
        {
            const Dimension::Id pdalId = layout->findDim(name);
            if (pdalId == Dimension::Id::Unknown)
                throwError("Dimension '" + name +
                           "' specified in "
                           "'dimensions' option not found in layout.");
            const Dimension::Type pdalType = layout->dimType(pdalId);
            const pdg::DimensionType physicalType = toPdgType(pdalType);
            if (physicalType == pdg::DimensionType::None)
                throwError("Dimension '" + name +
                           "' specified in "
                           "'dimensions' option not found in layout.");
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

    void computeHost(PointView& view)
    {
        if (view.size() < 2U)
            return;
        for (PointId point = 1U; point < view.size(); ++point)
        {
            bool duplicate = true;
            for (Dimension::Id dimension : m_pdalIds)
                if (view.getFieldAs<double>(dimension, point) !=
                    view.getFieldAs<double>(dimension, point - 1U))
                {
                    duplicate = false;
                    break;
                }
            view.setField(Dimension::Id::Duplicate, point,
                          static_cast<std::uint8_t>(duplicate));
        }
    }

#if PDG_HAS_CUDA
    bool computeCuda(PointView& view)
    {
        if (pdg::cudaDevices().empty())
            return false;
        const std::size_t count = static_cast<std::size_t>(view.size());
        if (count < 2U)
            return true;
        std::unique_ptr<pdg::MemoryResource> pinnedMemory =
            pdg::makeCudaPinnedMemoryResource();
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::PointBatch host(count, coordinates, *m_dimensions, *pinnedMemory);
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const pdg::DimensionId dimension = m_program.dimensions[key];
            if (!host.has(dimension))
                host.materialize(dimension, m_physicalTypes[key]);
        }
        host.setSize(count);
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const std::size_t stride =
                pdg::dimensionTypeSize(m_physicalTypes[key]);
            auto* destination = static_cast<std::byte*>(
                host.rawData(m_program.dimensions[key]));
            for (PointId point = 0; point < view.size(); ++point)
                view.getField(
                    reinterpret_cast<char*>(
                        destination + static_cast<std::size_t>(point) * stride),
                    m_pdalIds[key], m_pdalTypes[key], point);
        }
        if (!pdg::labelDuplicatesMaySupportExactDevice(host, m_program))
            return false;

        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(count, coordinates, *m_dimensions,
                               *deviceMemory);
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        for (std::size_t key = 0; key < m_program.dimensions.size(); ++key)
        {
            const pdg::DimensionId dimension = m_program.dimensions[key];
            if (device.has(dimension))
                continue;
            device.materialize(dimension, m_physicalTypes[key]);
            const std::size_t bytes =
                count * pdg::dimensionTypeSize(m_physicalTypes[key]);
            PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(dimension),
                                           host.rawData(dimension), bytes,
                                           cudaMemcpyHostToDevice, stream));
        }
        device.setSize(count);
        std::unique_ptr<pdg::Allocation> deviceOutput =
            deviceMemory->allocate(count, alignof(std::uint8_t));
        PDG_CUDA_CHECK(cudaMemsetAsync(deviceOutput->data(), 0, count, stream));
        pdg::labelDuplicates(device, m_program,
                             static_cast<std::uint8_t*>(deviceOutput->data()));
        std::vector<std::uint8_t> output(count);
        PDG_CUDA_CHECK(cudaMemcpyAsync(output.data(), deviceOutput->data(),
                                       count, cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (PointId point = 1U; point < view.size(); ++point)
            view.setField(Dimension::Id::Duplicate, point,
                          output[static_cast<std::size_t>(point)]);
        return true;
    }
#endif

    void filter(PointView& view) override
    {
        log()->get(LogLevel::Debug) << "Finding duplicates...\n";
        if (m_residentContext)
        {
            m_region.radiusIndex = m_region.maximumRadius > 0.0;
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            const auto region = static_cast<std::size_t>(m_executionRegion);
            context.beginDelegatedRegion(view, region);
            const bool usedCuda = pdg_detail::tryCudaLabelDuplicatesColumn(
                view, m_program, *m_dimensions, m_region,
                /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident label_duplicates path "
                           "was not used");
            if (m_region.last)
                context.endDelegatedRegion(view, region);
            return;
        }
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requireAutomatic =
            std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID");
        const bool automaticCuda =
            m_autoCuda && pdg::automaticLabelNnDistanceCudaDeviceQualified();
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             automaticCuda);
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda &&
            !std::getenv("PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE"))
        {
            try
            {
                usedCuda = computeCuda(view);
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
            }
        }
#else
        static_cast<void>(requestCuda);
#endif
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid label_duplicates path was not "
                "used");
        if (requireAutomatic && (!m_autoCuda || !usedCuda))
            throwError(
                "required automatic exact CUDA label/NNDistance hybrid path "
                "was not used");
        if (!usedCuda)
            computeHost(view);
    }

    StringList m_dimNames;
    std::vector<Dimension::Id> m_pdalIds;
    std::vector<Dimension::Type> m_pdalTypes;
    std::vector<pdg::DimensionType> m_physicalTypes;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::LabelDuplicatesProgram m_program;
    pdg_detail::CudaNeighborhoodRegion m_region;
    bool m_autoCuda = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridLabelDuplicatesStage),
    "Internal exact PDG adjacent duplicate labeling filter", ""};

CREATE_STATIC_STAGE(PdgLabelDuplicatesFilter, s_info)

} // namespace pdal
