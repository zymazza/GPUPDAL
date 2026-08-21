#include <pdg/Cuda.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Smrf.hpp>

#include "PdgResidentContext.hpp"

#include <pdal/Filter.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

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
#include <utility>
#include <vector>

namespace pdal
{

// Exact compatibility wrapper for the pinned SMRFilter. Selection stays in
// PointView space so returns outside the configured domain remain untouched;
// the raster primitive consumes only the selected logical-double columns.
class PdgSmrfFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.smrf";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("cell", "Cell size?", m_program.cell, 1.0);
        args.add("slope", "Percent slope?", m_program.slope, 0.15);
        args.add("scalar", "Elevation scalar?", m_program.scalar, 1.25);
        args.add("threshold", "Elevation threshold?", m_program.threshold, 0.5);
        args.add("cut", "Cut net size?", m_program.cut, 0.0);
        args.add("returns", "Include last returns?", m_returns,
                 {"last", "only"});
        m_windowArg = &args.add("window", "Max window size?", m_program.window);
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
        if (!m_windowArg->set())
            m_program.window = 18.0 * m_program.cell;
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

    void warnSmallRaster(PointView& view,
                         const std::vector<PointId>& selected) const
    {
        double minimumX =
            view.getFieldAs<double>(Dimension::Id::X, selected.front());
        double maximumX = minimumX;
        double minimumY =
            view.getFieldAs<double>(Dimension::Id::Y, selected.front());
        double maximumY = minimumY;
        for (PointId point : selected)
        {
            const double x = view.getFieldAs<double>(Dimension::Id::X, point);
            const double y = view.getFieldAs<double>(Dimension::Id::Y, point);
            minimumX = (std::min)(minimumX, x);
            maximumX = (std::max)(maximumX, x);
            minimumY = (std::min)(minimumY, y);
            maximumY = (std::max)(maximumY, y);
        }
        const int columns =
            static_cast<int>(((maximumX - minimumX) / m_program.cell) + 1.0);
        const int rows =
            static_cast<int>(((maximumY - minimumY) / m_program.cell) + 1.0);
        if (columns * rows < 10000)
            log()->get(LogLevel::Warning)
                << "SMRF running with a small number of cells ("
                << (columns * rows) << ").  Consider changing cell size.\n";
    }

    // B0216: fill raster voids exactly as the pinned oracle does.
    //
    // Upstream's `knnfill` builds a 2D KD-tree over the non-void cell centres
    // in world coordinates and averages the eight nearest with a Welford
    // recurrence. pdg's own fill picks neighbours by integer cell distance and
    // breaks ties by lowest cell index, which B0214 proved disagrees with the
    // oracle in exactly the cells where a tie at the eighth neighbour has to
    // be broken -- 3 cells of 1,880 at `cell=8`, enough to misclassify 99 of
    // 1,000,000 points at `cell=1`.
    //
    // B0215 established that upstream's resolution is its KD-tree's traversal
    // order and reduces to no short rule. So this does not reimplement the
    // choice: it builds the same index over the same points in the same order
    // and asks it the same question, which is exact by construction and cannot
    // drift as long as both call the same PDAL index.
    void fillRasterLikeOracle(PointView& view, std::vector<double>& values,
                              std::size_t rows, std::size_t columns,
                              double minimumX, double minimumY,
                              double cell) const
    {
        PointViewPtr temp = view.makeNew();
        PointId next(0);
        for (std::size_t column = 0U; column < columns; ++column)
            for (std::size_t row = 0U; row < rows; ++row)
            {
                const double value = values[column * rows + row];
                if (std::isnan(value))
                    continue;
                PointRef point = temp->point(next++);
                point.setField(Dimension::Id::X,
                               minimumX + (static_cast<double>(column) + 0.5) *
                                              cell);
                point.setField(Dimension::Id::Y,
                               minimumY +
                                   (static_cast<double>(row) + 0.5) * cell);
                point.setField(Dimension::Id::Z, value);
            }
        // Upstream returns early on an all-void raster rather than dividing by
        // zero; preserve that instead of leaving NaNs behind differently.
        if (!temp->size())
            return;

        const KD2Index& index = temp->build2dIndex();
        for (std::size_t column = 0U; column < columns; ++column)
            for (std::size_t row = 0U; row < rows; ++row)
            {
                const std::size_t id = column * rows + row;
                if (!std::isnan(values[id]))
                    continue;
                const double x =
                    minimumX + (static_cast<double>(column) + 0.5) * cell;
                const double y =
                    minimumY + (static_cast<double>(row) + 0.5) * cell;
                const PointIdList neighbors = index.neighbors(x, y, 8);
                double mean(0.0);
                std::size_t seen(0);
                for (const PointId neighbor : neighbors)
                {
                    ++seen;
                    const double delta =
                        temp->getFieldAs<double>(Dimension::Id::Z, neighbor) -
                        mean;
                    mean += delta / static_cast<double>(seen);
                }
                values[id] = mean;
            }
    }

    void filter(PointView& view) override
    {
        std::vector<PointId> selected = selectedPoints(view);
        if (selected.empty())
            throwError("No returns to process.");
        warnSmallRaster(view, selected);
        if (m_residentContext)
            pdg_detail::requireResidentExecutionContext().beginDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));

        const bool requireCuda =
            m_residentContext || std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            pdg::SmrfExactDeviceQualified &&
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

        pdg::SmrfResult result;
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && !pdg::cudaDevices().empty() &&
            pdg::smrfSupportsExactDevice(batch, m_program))
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
                result = pdg::classifySmrf(device, m_program);
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
            throwError(m_residentContext
                           ? "planner-selected resident smrf path was not used"
                           : "required exact CUDA hybrid smrf path was not "
                             "used");
        if (!usedCuda)
            result = pdg::classifySmrf(
                batch, m_program,
                [this, &view](std::vector<double>& values, std::size_t rows,
                              std::size_t columns, double minimumX,
                              double minimumY, double cell)
                {
                    fillRasterLikeOracle(view, values, rows, columns, minimumX,
                                         minimumY, cell);
                });
        for (std::size_t index = 0U; index < selected.size(); ++index)
            view.setField(Dimension::Id::Classification, selected[index],
                          classification[index]);
        log()->get(LogLevel::Debug)
            << "Identified " << result.groundPoints << " ground returns ("
            << (100.0 * static_cast<double>(result.groundPoints) /
                static_cast<double>(selected.size()))
            << "%)\n";
        if (m_residentContext)
        {
            pdg_detail::ResidentExecutionContext& context =
                pdg_detail::requireResidentExecutionContext();
            context.endDelegatedRegion(
                view, static_cast<std::size_t>(m_executionRegion));
        }
    }

    pdg::SmrfProgram m_program;
    StringList m_returns;
    Arg* m_windowArg = nullptr;
    bool m_returnDimensionsPresent = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0U;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridSmrfStage),
    "Internal exact PDG simple morphological ground filter", ""};

CREATE_STATIC_STAGE(PdgSmrfFilter, s_info)

} // namespace pdal
