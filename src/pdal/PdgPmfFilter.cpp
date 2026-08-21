#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Pmf.hpp>

#include "PdgResidentContext.hpp"

#include <io/BufferReader.hpp>
#include <pdal/Filter.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{
namespace
{
class ResidentRegionCompletion
{
public:
    ResidentRegionCompletion(pdg_detail::ResidentExecutionContext* context,
                             PointView& view, std::size_t region) noexcept
        : m_context(context), m_view(&view), m_region(region),
          m_active(context != nullptr)
    {
    }

    ~ResidentRegionCompletion() noexcept
    {
        if (!m_active)
            return;
        try
        {
            complete();
        }
        catch (...)
        {
        }
    }

    ResidentRegionCompletion(const ResidentRegionCompletion&) = delete;
    ResidentRegionCompletion&
    operator=(const ResidentRegionCompletion&) = delete;

    void complete()
    {
        if (!m_active)
            return;
        m_context->endDelegatedRegion(*m_view, m_region);
        m_active = false;
    }

    void keepOpen() noexcept
    {
        m_active = false;
    }

private:
    pdg_detail::ResidentExecutionContext* m_context = nullptr;
    PointView* m_view = nullptr;
    std::size_t m_region = 0U;
    bool m_active = false;
};
} // unnamed namespace

class PdgPmfFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.pmf";
    }

private:
    bool requireCuda() const noexcept
    {
        return m_residentContext || std::getenv("PDG_REQUIRE_CUDA_HYBRID");
    }

    bool requestCuda() const noexcept
    {
        return !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
               (requireCuda() || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID"));
    }

    void addArgs(ProgramArgs& args) override
    {
        args.add("cell_size", "Cell size", m_program.cellSize, 1.0);
        args.add("exponential", "Exponential growth of window size?",
                 m_program.exponential, true);
        args.add("initial_distance", "Initial distance",
                 m_program.initialDistance, 0.15);
        args.add("returns", "Include only returns?", m_returns,
                 {"last", "only"});
        args.add("max_distance", "Maximum distance", m_program.maxDistance,
                 2.5);
        args.add("max_window_size", "Maximum window size",
                 m_program.maxWindowSize, 33.0);
        args.add("slope", "Slope", m_program.slope, 1.0);
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
        args.add("pdg_grid_reuse",
                 "Internal planner-owned raster allocation reuse marker",
                 m_gridReuse, false)
            .setHidden();
        args.add("pdg_grid_region_last",
                 "Internal planner-owned final raster stage marker",
                 m_gridRegionLast, true)
            .setHidden();
    }

    void addDimensions(PointLayoutPtr layout) override
    {
        layout->registerDim(Dimension::Id::Classification);
    }

    void prepared(PointTableRef table) override
    {
        // In default and merely experimental compatibility modes the original
        // stage owns option validation and diagnostics. The internal wrapper
        // prepares its reduced envelope only for an explicitly required or
        // planner-owned resident CUDA execution.
        if (!requireCuda())
            return;

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
        m_returnDimensionsPresent =
            layout->hasDim(Dimension::Id::ReturnNumber) &&
            layout->hasDim(Dimension::Id::NumberOfReturns);
        if (!m_returns.empty() && !m_returnDimensionsPresent)
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

    std::vector<PointId> selectedPoints(PointView& view)
    {
        bool numberOneZero = false;
        bool returnOneZero = false;
        bool numberAllZero = true;
        bool returnAllZero = true;
        if (m_returnDimensionsPresent)
            for (PointId point = 0U; point < view.size(); ++point)
            {
                const std::uint8_t count = view.getFieldAs<std::uint8_t>(
                    Dimension::Id::NumberOfReturns, point);
                const std::uint8_t number = view.getFieldAs<std::uint8_t>(
                    Dimension::Id::ReturnNumber, point);
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
        m_returnValuesAllZero = allZero;
        if (allZero)
            log()->get(LogLevel::Warning)
                << "Both NumberOfReturns and ReturnNumber are filled with "
                   "0's. Proceeding without any further return filtering.\n";

        std::vector<PointId> selected;
        selected.reserve(static_cast<std::size_t>(view.size()));
        for (PointId point = 0U; point < view.size(); ++point)
        {
            if (allZero ||
                returnSelected(view.getFieldAs<std::uint8_t>(
                                   Dimension::Id::ReturnNumber, point),
                               view.getFieldAs<std::uint8_t>(
                                   Dimension::Id::NumberOfReturns, point)))
                selected.push_back(point);
        }
        return selected;
    }

    void runUpstreamFallback(PointView& view,
                             bool returnDiagnosticsAlreadyEmitted = false)
    {
        PointTable table;
        PointLayoutPtr layout(table.layout());
        layout->registerDims({Dimension::Id::X, Dimension::Id::Y,
                              Dimension::Id::Z, Dimension::Id::Classification});
        const bool copyReturns =
            view.layout()->hasDim(Dimension::Id::ReturnNumber) &&
            view.layout()->hasDim(Dimension::Id::NumberOfReturns);
        const bool normalizeReturns =
            returnDiagnosticsAlreadyEmitted &&
            (!m_returnDimensionsPresent || m_returnValuesAllZero);
        if (copyReturns || normalizeReturns)
            layout->registerDims(
                {Dimension::Id::ReturnNumber, Dimension::Id::NumberOfReturns});
        PointViewPtr fallbackView(new PointView(table));
        BufferReader reader;
        reader.addView(fallbackView);
        StageFactory factory;
        Stage* filter = factory.createStage("filters.pmf");
        if (!filter)
            throwError("upstream filters.pmf fallback is unavailable");
        Options options;
        options.add("cell_size", m_program.cellSize);
        options.add("exponential", m_program.exponential);
        options.add("initial_distance", m_program.initialDistance);
        if (normalizeReturns || m_returns.empty())
            for (const char* value : {"first", "intermediate", "last", "only"})
                options.add("returns", value);
        else
            for (const std::string& value : m_returns)
                options.add("returns", value);
        options.add("max_distance", m_program.maxDistance);
        options.add("max_window_size", m_program.maxWindowSize);
        options.add("slope", m_program.slope);
        options.add("ground_class", m_program.groundClass);
        options.add("other_class", m_program.otherClass);
        options.add("only_ground", m_program.onlyGround);
        filter->setOptions(options);
        filter->setLog(log());
        filter->setInput(reader);
        // Replace the wrapper's log leader with the original stage's leader
        // while it owns compatibility diagnostics.
        stopLogging();
        try
        {
            filter->prepare(table);
            for (PointId point = 0U; point < view.size(); ++point)
            {
                fallbackView->setField(
                    Dimension::Id::X, point,
                    view.getFieldAs<double>(Dimension::Id::X, point));
                fallbackView->setField(
                    Dimension::Id::Y, point,
                    view.getFieldAs<double>(Dimension::Id::Y, point));
                fallbackView->setField(
                    Dimension::Id::Z, point,
                    view.getFieldAs<double>(Dimension::Id::Z, point));
                fallbackView->setField(
                    Dimension::Id::Classification, point,
                    view.getFieldAs<std::uint8_t>(Dimension::Id::Classification,
                                                  point));
                if (copyReturns || normalizeReturns)
                {
                    fallbackView->setField(
                        Dimension::Id::ReturnNumber, point,
                        normalizeReturns
                            ? std::uint8_t{1U}
                            : view.getFieldAs<std::uint8_t>(
                                  Dimension::Id::ReturnNumber, point));
                    fallbackView->setField(
                        Dimension::Id::NumberOfReturns, point,
                        normalizeReturns
                            ? std::uint8_t{1U}
                            : view.getFieldAs<std::uint8_t>(
                                  Dimension::Id::NumberOfReturns, point));
                }
            }
            const PointViewSet output = filter->execute(table);
            if (output.size() != 1U || (*output.begin())->size() != view.size())
                throwError(
                    "upstream filters.pmf fallback changed view topology");
            const PointView& result = **output.begin();
            for (PointId point = 0U; point < view.size(); ++point)
                view.setField(Dimension::Id::Classification, point,
                              result.getFieldAs<std::uint8_t>(
                                  Dimension::Id::Classification, point));
        }
        catch (...)
        {
            startLogging();
            throw;
        }
        startLogging();
    }

    void filter(PointView& view) override
    {
        const bool requireCuda = this->requireCuda();
        if (!requireCuda)
        {
            runUpstreamFallback(view);
            return;
        }
        const bool requestCuda = this->requestCuda();

        std::vector<PointId> selected = selectedPoints(view);
        if (selected.empty())
            throwError("No returns to process.");
        pdg_detail::ResidentExecutionContext* resident =
            m_residentContext ? &pdg_detail::requireResidentExecutionContext()
                              : nullptr;
        const std::size_t executionRegion =
            static_cast<std::size_t>(m_executionRegion);
        if (resident)
            resident->beginDelegatedRegion(view, executionRegion);
        ResidentRegionCompletion regionCompletion(resident, view,
                                                  executionRegion);

        pdg::DimensionRegistry dimensions;
        std::unique_ptr<pdg::MemoryResource> ownedMemory;
        pdg::MemoryResource* memory = nullptr;
#if PDG_HAS_CUDA
        if (resident)
            memory = &resident->delegatedStagingMemory(
                view, static_cast<std::size_t>(m_executionRegion));
        else if (requestCuda)
        {
            ownedMemory = pdg::makeCudaPinnedMemoryResource();
            memory = ownedMemory.get();
        }
        else
#endif
        {
            ownedMemory = std::make_unique<pdg::HostMemoryResource>();
            memory = ownedMemory.get();
        }
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

        pdg::PmfResult result;
        pdg::PmfTiledExecutionFacts tiledFacts;
        bool usedCuda = false;
#if PDG_HAS_CUDA
        const bool exactDeviceInput =
            resident ? pdg::pmfSupportsExactTiledDevice(batch, m_program)
                     : pdg::pmfSupportsExactDevice(batch, m_program);
        if (requestCuda && !pdg::cudaDevices().empty() && exactDeviceInput)
        {
            pdg::RasterGridProduct* gridProduct = nullptr;
            if (resident)
            {
                const pdg::PmfRasterFrame frame =
                    pdg::pmfRasterFrame(batch, m_program);
                gridProduct = &resident->acquireRasterGridProduct(
                    view, executionRegion,
                    {frame.minimumX, frame.minimumY, frame.rows, frame.columns,
                     m_program.cellSize, pdg::RasterGridFramePolicy::PmfV1},
                    m_gridReuse);
            }
            std::unique_ptr<pdg::MemoryResource> ownedDeviceMemory;
            pdg::MemoryResource* deviceMemory = nullptr;
            if (resident)
                deviceMemory =
                    &resident->delegatedDeviceMemory(view, executionRegion);
            else
            {
                ownedDeviceMemory = pdg::makeCudaMemoryResource();
                deviceMemory = ownedDeviceMemory.get();
            }
            pdg::PointBatch device(selected.size(), coordinates, dimensions,
                                   *deviceMemory);
            const bool deviceRasterProof =
                resident &&
                gridProduct->frame().size() <=
                    static_cast<std::size_t>(
                        (std::numeric_limits<unsigned int>::max)()) &&
                gridProduct->canMaterializeDeviceProofWorkspace();
            if (resident && !deviceRasterProof)
            {
                pdg::buildPmfTiledRaster(batch, m_program, *gridProduct);
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::RasterBuild, executionRegion,
                    gridProduct->backingBytes());
                static_cast<void>(
                    gridProduct->materializeResidentDeviceBackings());
            }
            for (const auto [id, type] :
                 {std::pair{xId, pdg::DimensionType::Double},
                  std::pair{yId, pdg::DimensionType::Double}, std::pair {
                      zId,
                      pdg::DimensionType::Double
                  }})
                device.materialize(id, type);
            if (!deviceRasterProof)
                device.materialize(classificationId,
                                   pdg::DimensionType::Unsigned8);
            device.setSize(selected.size());
            const cudaStream_t stream =
                static_cast<cudaStream_t>(device.nativeStreamHandle());
            for (const auto [id, bytes] :
                 {std::pair{xId, selected.size() * sizeof(double)},
                  std::pair{yId, selected.size() * sizeof(double)}, std::pair {
                      zId,
                      selected.size() * sizeof(double)
                  }})
                PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(id),
                                               batch.rawData(id), bytes,
                                               cudaMemcpyHostToDevice, stream));
            if (deviceRasterProof)
            {
                try
                {
                    pdg::buildPmfTiledRasterDevice(device, m_program,
                                                   *gridProduct);
                }
                catch (const pdg::PmfRasterTieError&)
                {
                    // The proof rejected before device Classification storage
                    // or any point mutation. Compatibility mode can therefore
                    // hand this one stage to the pinned implementation without
                    // exposing a partially published resident product.
                    runUpstreamFallback(view, true);
                    if (m_gridRegionLast)
                        regionCompletion.complete();
                    else
                        regionCompletion.keepOpen();
                    return;
                }
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::RasterBuild, executionRegion,
                    gridProduct->backingBytes());
                static_cast<void>(
                    gridProduct->materializeResidentDeviceBackings());
                device.materialize(classificationId,
                                   pdg::DimensionType::Unsigned8);
            }
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(device.rawData(classificationId),
                                batch.rawData(classificationId),
                                selected.size() * sizeof(std::uint8_t),
                                cudaMemcpyHostToDevice, stream));
            if (resident)
                result = pdg::classifyPmfTiledDevice(device, m_program,
                                                     *gridProduct, &tiledFacts);
            else
            {
                try
                {
                    result = pdg::classifyPmf(device, m_program);
                }
                catch (const pdg::PmfRasterTieError&)
                {
                    // The bounded primitive proves this ambiguity before its
                    // Classification kernel. Preserve exact compatibility by
                    // executing the original stage on the untouched view.
                    runUpstreamFallback(view, true);
                    return;
                }
            }
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(batch.rawData(classificationId),
                                device.rawData(classificationId),
                                selected.size() * sizeof(std::uint8_t),
                                cudaMemcpyDeviceToHost, stream));
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            if (resident && tiledFacts.rasterHostToDeviceTransfers != 0U)
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::RasterUpload, executionRegion,
                    tiledFacts.rasterHostToDeviceBytes);
            if (resident && tiledFacts.rasterDeviceToHostTransfers != 0U)
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::RasterDownload, executionRegion,
                    tiledFacts.rasterDeviceToHostBytes);
            usedCuda = true;
        }
#else
        static_cast<void>(requestCuda);
#endif
        if (!usedCuda)
            throwError(
                m_residentContext
                    ? "planner-selected resident pmf path was not used"
                    : "required exact CUDA hybrid pmf path was not used");
        for (std::size_t index = 0U; index < selected.size(); ++index)
            view.setField(Dimension::Id::Classification, selected[index],
                          classification[index]);
        log()->get(LogLevel::Debug2)
            << "Labeled " << result.groundPoints << " ground returns!\n";
        if (m_residentContext)
        {
            if (m_gridRegionLast)
                regionCompletion.complete();
            else
                regionCompletion.keepOpen();
        }
    }

    pdg::PmfProgram m_program;
    StringList m_returns;
    bool m_returnDimensionsPresent = false;
    bool m_returnValuesAllZero = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
    bool m_gridReuse = false;
    bool m_gridRegionLast = true;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridPmfStage),
    "Internal exact PDG progressive morphological filter", ""};

CREATE_STATIC_STAGE(PdgPmfFilter, s_info)

} // namespace pdal
