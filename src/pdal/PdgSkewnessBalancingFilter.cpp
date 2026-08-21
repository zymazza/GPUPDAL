#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>
#include <pdg/stages/Skewness.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/io/LasTranslate.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

#include <cmath>
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
} // unnamed namespace

class PdgSkewnessBalancingFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.skewnessbalancing";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("ground_class",
                 "Classification value of ground points."
                 " [Default: 2]",
                 m_program.groundClass, std::uint8_t(2));
        args.add("other_class",
                 "Classification value of non-ground points."
                 " [Default: 1]",
                 m_program.otherClass, std::uint8_t(1));
        args.add("only_ground",
                 "Set to true to only modify the CLassification"
                 " value of detected ground points. [Default: false]",
                 m_program.onlyGround, false);
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

    void prepared(PointTableRef table) override
    {
        if (!pdg::skewnessProgramValid(m_program))
            throwError("Ground and non-ground class cannot be"
                       "equal when only_ground is false.");
        m_zType = table.layout()->dimType(Dimension::Id::Z);
    }

    void prerun(const PointViewSet& views) override
    {
        if (m_residentContext)
        {
            if (views.size() == 1U && (*views.begin())->empty())
            {
                pdg_detail::ResidentExecutionContext& context =
                    pdg_detail::requireResidentExecutionContext();
                const std::size_t region =
                    static_cast<std::size_t>(m_executionRegion);
                context.beginDelegatedRegion(**views.begin(), region);
                context.endDelegatedRegion(**views.begin(), region);
            }
            return;
        }
        if (!std::getenv("PDG_REQUIRE_CUDA_HYBRID"))
            return;
        for (const PointViewPtr& view : views)
            if (!view->empty())
                return;
        throwError(RequiredCudaError);
    }

#if PDG_HAS_CUDA
    bool orderCuda(const PointView& view, PointIdList& permutation) const
    {
        if (m_zType != Dimension::Type::Double ||
            !pdg::skewnessOrderingSizeWithinDeviceEnvelope(
                static_cast<std::size_t>(view.size())))
            return false;
        permutation.resize(static_cast<std::size_t>(view.size()));

        pdg::DimensionRegistry dimensions;
        const pdg::DimensionId zId(pdg::StandardDimension::Z);
        std::unique_ptr<pdg::MemoryResource> pinned =
            pdg::makeCudaPinnedMemoryResource();
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::PointBatch host(static_cast<std::size_t>(view.size()), coordinates,
                             dimensions, *pinned);
        host.materialize(zId, pdg::DimensionType::Double);
        host.setSize(static_cast<std::size_t>(view.size()));
        double* z = host.data<double>(zId);
        const std::span<const std::byte> mapped =
            pdg_detail::activeCudaLasSource();
        if (!mapped.empty())
        {
            const pdg::las::FileView input(mapped);
            if (!pdg::las::supportsDefaultTranslation(input) ||
                input.header().pointCount != view.size())
                return false;
            const pdg::CoordinateEncoding inputEncoding =
                input.header().coordinateEncoding();
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
                z[static_cast<std::size_t>(point)] = inputEncoding.decode(
                    2U, input.rawCoordinate(static_cast<std::size_t>(source),
                                            2U));
                if (!std::isfinite(z[static_cast<std::size_t>(point)]))
                    return false;
            }
        }
        else
        {
            for (PointId point = 0U; point < view.size(); ++point)
            {
                z[static_cast<std::size_t>(point)] =
                    view.getFieldAs<double>(Dimension::Id::Z, point);
                if (!std::isfinite(z[static_cast<std::size_t>(point)]))
                    return false;
            }
        }

        pdg::OrderingProgram ordering;
        ordering.dimensions = {zId};
        if (!pdg::orderingMaySupportExactDevice(host, ordering))
            return false;

        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), coordinates, dimensions,
                               *deviceMemory);
        device.materialize(zId, pdg::DimensionType::Double);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(zId), host.rawData(zId),
                                       host.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::HostToDevice,
            static_cast<std::size_t>(m_executionRegion),
            host.size() * sizeof(double));
        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory->allocate(host.size() * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
        NvtxRange range("pdg::filters.skewnessbalancing.order");
        const pdg::OrderingResult result = pdg::orderPoints(
            device, ordering,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        if (!result.exact)
            return false;
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation.data(),
                                       devicePermutation->data(),
                                       host.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceToHost,
            static_cast<std::size_t>(m_executionRegion),
            host.size() * sizeof(std::uint64_t));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!mapped.empty())
            pdg_detail::markActiveCudaLasSourceUsed();
        return true;
    }
#endif

    void classifySorted(PointView& view) const
    {
        const std::size_t size = static_cast<std::size_t>(view.size());
        std::vector<double> z(size);
        std::vector<std::uint8_t> classification(size);
        for (PointId point = 0U; point < view.size(); ++point)
        {
            const std::size_t index = static_cast<std::size_t>(point);
            z[index] = view.getFieldAs<double>(Dimension::Id::Z, point);
            classification[index] = view.getFieldAs<std::uint8_t>(
                Dimension::Id::Classification, point);
        }
        static_cast<void>(pdg::classifySkewnessSorted(
            z.data(), classification.data(), size, m_program));
        for (PointId point = 0U; point < view.size(); ++point)
            view.setField(Dimension::Id::Classification, point,
                          classification[static_cast<std::size_t>(point)]);
    }

    PointViewSet run(PointViewPtr view) override
    {
        PointViewSet result{view};
        if (log()->getLevel() > LogLevel::Debug1)
            log()->floatPrecision(8);
        if (view->empty())
            return result;

        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().beginDelegatedRegion(
                *view, static_cast<std::size_t>(m_executionRegion));

        const bool requireCuda =
            m_residentContext || std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        bool usedCuda = false;
        PointIdList permutation;
#if PDG_HAS_CUDA
        if (requestCuda)
        {
            try
            {
                if (!pdg::cudaDevices().empty())
                    usedCuda = orderCuda(*view, permutation);
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
            throwError(RequiredCudaError);
        if (usedCuda)
            view->applyPermutation(permutation);
        else
            view->sort(Dimension::Id::Z);
        classifySorted(*view);
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().endDelegatedRegion(
                *view, static_cast<std::size_t>(m_executionRegion));
        return result;
    }

    pdg::SkewnessProgram m_program;
    Dimension::Type m_zType = Dimension::Type::None;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
    static constexpr const char* RequiredCudaError =
        "required exact CUDA hybrid skewnessbalancing path was not used";
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridSkewnessStage),
    "Internal exact PDG skewness balancing filter", ""};

CREATE_STATIC_STAGE(PdgSkewnessBalancingFilter, s_info)

} // namespace pdal
