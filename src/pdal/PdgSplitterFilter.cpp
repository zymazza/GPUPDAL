#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

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
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pdal
{

// Behaviorally derived from the pinned upstream filters/SplitterFilter.cpp;
// see NOTICE. Cell identities persist across incoming views, and first
// encounter controls PointView publication order.
class PdgSplitterFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.splitter";
    }

private:
    using Coordinate = std::pair<int, int>;

    void addArgs(ProgramArgs& args) override
    {
        args.add("length", "Edge length of cell", m_length, 1000.0);
        args.add("origin_x", "X origin for a cell", m_xOrigin,
                 (std::numeric_limits<double>::quiet_NaN)());
        args.add("origin_y", "Y origin for a cell", m_yOrigin,
                 (std::numeric_limits<double>::quiet_NaN)());
        args.add("buffer", "Size of buffer (overlap) around each tile",
                 m_buffer, 0.0);
    }

    void initialize() override
    {
        if (m_buffer >= m_length / 2.0)
        {
            std::ostringstream message;
            message << "Buffer (" << m_buffer
                    << ") must be less than half of length (" << m_length
                    << ")";
            throwError(message.str());
        }
    }

    void prepared(PointTableRef) override
    {
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
    }

    void append(PointView& input, PointId point, int xCell, int yCell)
    {
        PointViewPtr& output = m_outputs[Coordinate{xCell, yCell}];
        if (!output)
            output = input.makeNew();
        output->appendPoint(input, point);
    }

    bool squareContains(int xCell, int yCell, double x, double y) const
    {
        const double minimumX = m_xOrigin + xCell * m_length - m_buffer;
        const double maximumX = minimumX + m_length + 2.0 * m_buffer;
        const double minimumY = m_yOrigin + yCell * m_length - m_buffer;
        const double maximumY = minimumY + m_length + 2.0 * m_buffer;
        return minimumX < x && x < maximumX && minimumY < y && y < maximumY;
    }

    void appendHost(PointView& input, PointId point)
    {
        const double x = input.getFieldAs<double>(Dimension::Id::X, point);
        const double dx = x - m_xOrigin;
        int xCell = static_cast<int>(dx / m_length);
        if (dx < 0.0)
            --xCell;

        const double y = input.getFieldAs<double>(Dimension::Id::Y, point);
        const double dy = y - m_yOrigin;
        int yCell = static_cast<int>(dy / m_length);
        if (dy < 0.0)
            --yCell;

        append(input, point, xCell, yCell);
        if (m_buffer <= 0.0)
            return;
        if (squareContains(xCell - 1, yCell, x, y))
            append(input, point, xCell - 1, yCell);
        else if (squareContains(xCell + 1, yCell, x, y))
            append(input, point, xCell + 1, yCell);

        if (squareContains(xCell, yCell - 1, x, y))
            append(input, point, xCell, yCell - 1);
        else if (squareContains(xCell, yCell + 1, x, y))
            append(input, point, xCell, yCell + 1);

        if (squareContains(xCell - 1, yCell - 1, x, y))
            append(input, point, xCell - 1, yCell - 1);
        else if (squareContains(xCell - 1, yCell + 1, x, y))
            append(input, point, xCell - 1, yCell + 1);
        else if (squareContains(xCell + 1, yCell - 1, x, y))
            append(input, point, xCell + 1, yCell - 1);
        else if (squareContains(xCell + 1, yCell + 1, x, y))
            append(input, point, xCell + 1, yCell + 1);
    }

#if PDG_HAS_CUDA
    bool executeCuda(const pdg::PointBatch& host,
                     std::vector<std::int32_t>& xCells,
                     std::vector<std::int32_t>& yCells,
                     const pdg::SplitterProgram& program) const
    {
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, *deviceMemory);
        const pdg::DimensionId x(pdg::StandardDimension::X);
        const pdg::DimensionId y(pdg::StandardDimension::Y);
        device.materialize(x, pdg::DimensionType::Double);
        device.materialize(y, pdg::DimensionType::Double);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        const std::size_t bytes = host.size() * sizeof(double);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(x), host.rawData(x),
                                       bytes, cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(y), host.rawData(y),
                                       bytes, cudaMemcpyHostToDevice, stream));
        const std::size_t cellBytes = host.size() * sizeof(std::int32_t);
        std::unique_ptr<pdg::Allocation> deviceX =
            deviceMemory->allocate(cellBytes, alignof(std::int32_t));
        std::unique_ptr<pdg::Allocation> deviceY =
            deviceMemory->allocate(cellBytes, alignof(std::int32_t));
        pdg::computeSplitterCells(device, program,
                                  static_cast<std::int32_t*>(deviceX->data()),
                                  static_cast<std::int32_t*>(deviceY->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(xCells.data(), deviceX->data(),
                                       cellBytes, cudaMemcpyDeviceToHost,
                                       stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(yCells.data(), deviceY->data(),
                                       cellBytes, cudaMemcpyDeviceToHost,
                                       stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
#endif

    PointViewSet run(PointViewPtr input) override
    {
        // Deliberately perform the same point-zero access as upstream for an
        // empty first view; its behavior is part of the pinned oracle.
        if (m_xOrigin != m_xOrigin)
            m_xOrigin = input->getFieldAs<double>(Dimension::Id::X, 0);
        if (m_yOrigin != m_yOrigin)
            m_yOrigin = input->getFieldAs<double>(Dimension::Id::Y, 0);

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
        const std::size_t count = static_cast<std::size_t>(input->size());
        std::vector<std::int32_t> xCells(count);
        std::vector<std::int32_t> yCells(count);

        if (requestCuda && count)
        {
            const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                      {0.0, 0.0, 0.0});
            pdg::HostMemoryResource hostMemory;
            pdg::PointBatch host(count, coordinates, *m_dimensions, hostMemory);
            const pdg::DimensionId x(pdg::StandardDimension::X);
            const pdg::DimensionId y(pdg::StandardDimension::Y);
            host.materialize(x, pdg::DimensionType::Double);
            host.materialize(y, pdg::DimensionType::Double);
            host.setSize(count);
            for (PointId point = 0; point < input->size(); ++point)
            {
                const std::size_t position = static_cast<std::size_t>(point);
                host.data<double>(x)[position] =
                    input->getFieldAs<double>(Dimension::Id::X, point);
                host.data<double>(y)[position] =
                    input->getFieldAs<double>(Dimension::Id::Y, point);
            }
            pdg::SplitterProgram program;
            program.length = m_length;
            program.originX = m_xOrigin;
            program.originY = m_yOrigin;
            program.buffer = m_buffer;
#if PDG_HAS_CUDA
            try
            {
                if (pdg::splitterCellsMaySupportExactDevice(host, program) &&
                    !pdg::cudaDevices().empty())
                    usedCuda = executeCuda(host, xCells, yCells, program);
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
            }
#else
            (void)program;
#endif
        }
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid splitter path was not used");

        for (PointId point = 0; point < input->size(); ++point)
        {
            if (usedCuda)
            {
                const std::size_t position = static_cast<std::size_t>(point);
                append(*input, point, static_cast<int>(xCells[position]),
                       static_cast<int>(yCells[position]));
            }
            else
                appendHost(*input, point);
        }

        PointViewSet result;
        for (const auto& entry : m_outputs)
            result.insert(entry.second);
        return result;
    }

    double m_length = 1000.0;
    double m_xOrigin = (std::numeric_limits<double>::quiet_NaN)();
    double m_yOrigin = (std::numeric_limits<double>::quiet_NaN)();
    double m_buffer = 0.0;
    std::map<Coordinate, PointViewPtr> m_outputs;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridSplitterStage),
    "Internal exact PDG XY point-view splitter", ""};

CREATE_STATIC_STAGE(PdgSplitterFilter, s_info)

} // namespace pdal
