#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Csf.hpp>

#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{

class PdgCsfFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.csf";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("smooth", "Slope postprocessing?", m_program.smooth, true);
        args.add("step", "Time step", m_program.timeStep, 0.65);
        args.add("threshold", "Classification threshold",
                 m_program.classThreshold, 0.5);
        args.add("hdiff", "Height difference threshold",
                 m_program.heightThreshold, 0.3);
        args.add("resolution", "Cloth resolution", m_program.resolution, 1.0);
        args.add("rigidness", "Rigidness", m_program.rigidness, 3);
        args.add("iterations", "Max iterations", m_program.iterations, 500);
        args.add("returns", "Include only returns?", m_returns,
                 {"last", "only"});
        args.add("ground_class", "Classification value of ground points.",
                 m_program.groundClass, std::uint8_t(2));
        args.add("other_class", "Classification value of non-ground points.",
                 m_program.otherClass, std::uint8_t(1));
        args.add("only_ground",
                 "Only modify classifications of detected ground points.",
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
        if (m_program.groundClass == m_program.otherClass &&
            !m_program.onlyGround)
            throwError("Ground and non-ground class cannot beequal when "
                       "only_ground is false.");

        for (std::string& value : m_returns)
        {
            Utils::trim(value);
            if (value != "first" && value != "intermediate" &&
                value != "last" && value != "only")
                throwError("Unrecognized 'returns' value: '" + value + "'.");
        }

        PointLayoutPtr layout(table.layout());
        m_hasReturnNumber = layout->hasDim(Dimension::Id::ReturnNumber);
        m_hasNumberOfReturns = layout->hasDim(Dimension::Id::NumberOfReturns);
        if (!m_returns.empty() && (!m_hasReturnNumber || !m_hasNumberOfReturns))
        {
            log()->get(LogLevel::Warning)
                << "Could not find ReturnNumber and NumberOfReturns. "
                   "Skipping segmentation of last returns and proceeding "
                   "with all returns.\n";
            m_returns.clear();
        }
    }

    bool returnSelected(std::uint8_t number, std::uint8_t count) const noexcept
    {
        if (m_returns.empty())
            return true;
        for (const std::string& value : m_returns)
        {
            if ((value == "first" && number == 1U && count > 1U) ||
                (value == "intermediate" && number > 1U && number < count) ||
                (value == "last" && number == count && count > 1U) ||
                (value == "only" && count == 1U))
                return true;
        }
        return false;
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

    std::vector<PointId> selectedPoints(PointView& view)
    {
        bool numberOneZero = false;
        bool returnOneZero = false;
        bool numberAllZero = true;
        bool returnAllZero = true;
        for (PointId point = 0U; point < view.size(); ++point)
        {
            const std::uint8_t count =
                m_hasNumberOfReturns
                    ? view.getFieldAs<std::uint8_t>(
                          Dimension::Id::NumberOfReturns, point)
                    : std::uint8_t{0U};
            const std::uint8_t number =
                m_hasReturnNumber ? view.getFieldAs<std::uint8_t>(
                                        Dimension::Id::ReturnNumber, point)
                                  : std::uint8_t{0U};
            numberOneZero = numberOneZero || count == 0U;
            returnOneZero = returnOneZero || number == 0U;
            numberAllZero = numberAllZero && count == 0U;
            returnAllZero = returnAllZero && number == 0U;
        }

        if ((numberOneZero || returnOneZero) &&
            !(numberAllZero && returnAllZero))
            throwError("Some NumberOfReturns or ReturnNumber values were 0, "
                       "but not all. Check that all values in the input file "
                       "are >= 1.");

        const bool allZero = numberAllZero && returnAllZero;
        if (allZero)
            log()->get(LogLevel::Warning)
                << "Both NumberOfReturns and ReturnNumber are filled with "
                   "0's. Proceeding without any further return filtering.\n";

        std::vector<PointId> selected;
        selected.reserve(static_cast<std::size_t>(view.size()));
        for (PointId point = 0U; point < view.size(); ++point)
        {
            if (allZero || m_returns.empty() ||
                returnSelected(view.getFieldAs<std::uint8_t>(
                                   Dimension::Id::ReturnNumber, point),
                               view.getFieldAs<std::uint8_t>(
                                   Dimension::Id::NumberOfReturns, point)))
                selected.push_back(point);
        }
        return selected;
    }

    void filter(PointView& view) override
    {
        std::vector<PointId> selected = selectedPoints(view);
        if (selected.empty())
            throwError("No returns to process.");
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
        pdg::PointBatch batch(selected.size(), coordinates, dimensions,
                              *memory);
        const pdg::DimensionId xId(pdg::StandardDimension::X);
        const pdg::DimensionId yId(pdg::StandardDimension::Y);
        const pdg::DimensionId zId(pdg::StandardDimension::Z);
        const pdg::DimensionId classificationId(
            pdg::StandardDimension::Classification);
        batch.materialize(xId, pdg::DimensionType::Double);
        batch.materialize(yId, pdg::DimensionType::Double);
        batch.materialize(zId, pdg::DimensionType::Double);
        batch.materialize(classificationId, pdg::DimensionType::Unsigned8);
        batch.setSize(selected.size());
        double* x = batch.data<double>(xId);
        double* y = batch.data<double>(yId);
        double* z = batch.data<double>(zId);
        std::uint8_t* classification =
            batch.data<std::uint8_t>(classificationId);
        for (std::size_t index = 0U; index < selected.size(); ++index)
        {
            const PointId point = selected[index];
            x[index] = view.getFieldAs<double>(Dimension::Id::X, point);
            y[index] = view.getFieldAs<double>(Dimension::Id::Y, point);
            z[index] = view.getFieldAs<double>(Dimension::Id::Z, point);
            classification[index] = view.getFieldAs<std::uint8_t>(
                Dimension::Id::Classification, point);
        }

        pdg::CsfResult result;
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && !pdg::cudaDevices().empty() &&
            pdg::csfSupportsExactDevice(batch, m_program))
        {
            try
            {
                std::unique_ptr<pdg::MemoryResource> deviceMemory =
                    pdg::makeCudaMemoryResource();
                pdg::PointBatch device(selected.size(), coordinates, dimensions,
                                       *deviceMemory);
                for (const auto [id, type] :
                     {std::pair{xId, pdg::DimensionType::Double},
                      std::pair{yId, pdg::DimensionType::Double},
                      std::pair{zId, pdg::DimensionType::Double}, std::pair {
                          classificationId,
                          pdg::DimensionType::Unsigned8
                      }})
                    device.materialize(id, type);
                device.setSize(selected.size());
                const cudaStream_t stream =
                    static_cast<cudaStream_t>(device.nativeStreamHandle());
                for (const auto [id, bytes] :
                     {std::pair{xId, selected.size() * sizeof(double)},
                      std::pair{yId, selected.size() * sizeof(double)},
                      std::pair{zId, selected.size() * sizeof(double)},
                      std::pair {
                          classificationId,
                          selected.size() * sizeof(std::uint8_t)
                      }})
                    PDG_CUDA_CHECK(
                        cudaMemcpyAsync(device.rawData(id), batch.rawData(id),
                                        bytes, cudaMemcpyHostToDevice, stream));
                result = pdg::classifyCsf(device, m_program);
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(batch.rawData(classificationId),
                                    device.rawData(classificationId),
                                    selected.size() * sizeof(std::uint8_t),
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
                    ? "planner-selected resident csf path was not used"
                    : "required exact CUDA hybrid csf path was not used");
        if (!usedCuda)
            result = pdg::classifyCsf(batch, m_program);
        for (std::size_t index = 0U; index < selected.size(); ++index)
            view.setField(Dimension::Id::Classification, selected[index],
                          classification[index]);
        const auto [minimumX, maximumX] =
            std::minmax_element(x, x + selected.size());
        log()->get(LogLevel::Debug)
            << "setPointCloud: " << selected.size() << '\n';
        log()->get(LogLevel::Debug) << "[0] Configuring terrain...\n";
        log()->get(LogLevel::Debug) << *maximumX << ", " << *minimumX << '\n';
        log()->get(LogLevel::Debug) << "[0] Configuring cloth...\n";
        log()->get(LogLevel::Debug) << "[0]  - width: " << result.width
                                    << " height: " << result.height << '\n';
        log()->get(LogLevel::Debug) << "[0] Rasterizing...\n";
        log()->get(LogLevel::Debug) << "[0] Simulating...\n";
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            context.endDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));
        }
    }

    pdg::CsfProgram m_program;
    StringList m_returns;
    bool m_hasReturnNumber = false;
    bool m_hasNumberOfReturns = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridCsfStage),
    "Internal exact PDG cloth simulation filter", ""};

CREATE_STATIC_STAGE(PdgCsfFilter, s_info)

} // namespace pdal
