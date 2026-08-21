#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/PointView.hpp>
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
#include <utility>

namespace pdal
{

class PdgElmFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.elm";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("cell", "Cell size", m_program.cell, 10.0);
        args.add("class", "Class to use for noise points",
                 m_program.classification, std::uint8_t(7));
        args.add("threshold", "Threshold value", m_program.threshold, 1.0);
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

    void prerun(const PointViewSet& views) override
    {
        if (!m_residentContext || views.size() != 1U ||
            !(*views.begin())->empty())
            return;
        pdg_detail::ResidentExecutionContext& context =
            pdg_detail::requireResidentExecutionContext();
        const std::size_t region = static_cast<std::size_t>(m_executionRegion);
        context.beginDelegatedRegion(**views.begin(), region);
        context.endDelegatedRegion(**views.begin(), region);
    }

    void filter(PointView& view) override
    {
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().beginDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));

        const bool requireCuda =
            m_residentContext || std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
        pdg::DimensionRegistry dimensions;
        std::unique_ptr<pdg::MemoryResource> memory;
#if PDG_HAS_CUDA
        if (requestCuda)
            memory = pdg::makeCudaPinnedMemoryResource();
        else
#endif
            memory = std::make_unique<pdg::HostMemoryResource>();
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::PointBatch batch(static_cast<std::size_t>(view.size()),
                              coordinates, dimensions, *memory);
        const pdg::DimensionId xId(pdg::StandardDimension::X);
        const pdg::DimensionId yId(pdg::StandardDimension::Y);
        const pdg::DimensionId zId(pdg::StandardDimension::Z);
        const pdg::DimensionId classificationId(
            pdg::StandardDimension::Classification);
        batch.materialize(xId, pdg::DimensionType::Double);
        batch.materialize(yId, pdg::DimensionType::Double);
        batch.materialize(zId, pdg::DimensionType::Double);
        batch.materialize(classificationId, pdg::DimensionType::Unsigned8);
        batch.setSize(static_cast<std::size_t>(view.size()));
        double* x = batch.data<double>(xId);
        double* y = batch.data<double>(yId);
        double* z = batch.data<double>(zId);
        std::uint8_t* classification =
            batch.data<std::uint8_t>(classificationId);
        for (PointId point = 0U; point < view.size(); ++point)
        {
            const std::size_t index = static_cast<std::size_t>(point);
            x[index] = view.getFieldAs<double>(Dimension::Id::X, point);
            y[index] = view.getFieldAs<double>(Dimension::Id::Y, point);
            z[index] = view.getFieldAs<double>(Dimension::Id::Z, point);
            classification[index] = view.getFieldAs<std::uint8_t>(
                Dimension::Id::Classification, point);
        }

        pdg::ElmResult result;
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && !pdg::cudaDevices().empty() &&
            pdg::elmSupportsExactDevice(batch, m_program))
        {
            try
            {
                std::unique_ptr<pdg::MemoryResource> deviceMemory =
                    pdg::makeCudaMemoryResource();
                pdg::PointBatch device(batch.size(), coordinates, dimensions,
                                       *deviceMemory);
                for (const auto [id, type] :
                     {std::pair{xId, pdg::DimensionType::Double},
                      std::pair{yId, pdg::DimensionType::Double},
                      std::pair{zId, pdg::DimensionType::Double}, std::pair {
                          classificationId,
                          pdg::DimensionType::Unsigned8
                      }})
                    device.materialize(id, type);
                device.setSize(batch.size());
                const cudaStream_t stream =
                    static_cast<cudaStream_t>(device.nativeStreamHandle());
                for (const auto [id, bytes] :
                     {std::pair{xId, batch.size() * sizeof(double)},
                      std::pair{yId, batch.size() * sizeof(double)},
                      std::pair{zId, batch.size() * sizeof(double)}, std::pair {
                          classificationId,
                          batch.size() * sizeof(std::uint8_t)
                      }})
                    PDG_CUDA_CHECK(
                        cudaMemcpyAsync(device.rawData(id), batch.rawData(id),
                                        bytes, cudaMemcpyHostToDevice, stream));
                result = pdg::classifyElm(device, m_program);
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(batch.rawData(classificationId),
                                    device.rawData(classificationId),
                                    batch.size() * sizeof(std::uint8_t),
                                    cudaMemcpyDeviceToHost, stream));
                PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
                usedCuda = true;
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
                m_residentContext
                    ? "planner-selected resident elm path was not used"
                    : "required exact CUDA hybrid elm path was not used");
        if (!usedCuda)
            result = pdg::classifyElm(batch, m_program);
        for (PointId point = 0U; point < view.size(); ++point)
            view.setField(Dimension::Id::Classification, point,
                          classification[static_cast<std::size_t>(point)]);
        log()->get(LogLevel::Info)
            << "Classified " << result.classifiedPoints
            << " points as noise by Extended Local Minimum (ELM).\n";
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().endDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));
    }

    pdg::ElmProgram m_program;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridElmStage),
                                     "Internal exact PDG ELM filter", ""};

CREATE_STATIC_STAGE(PdgElmFilter, s_info)

} // namespace pdal
