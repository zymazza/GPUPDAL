#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Partition.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

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

// Behaviorally derived from the pinned upstream filters/DividerFilter.cpp;
// see NOTICE. Requested empty views and source order within each view are
// observable through numbered writers.
class PdgDividerFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return "filters.divider";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("mode", "Divider mode", m_mode, "partition");
        args.add("count", "Number of output views", m_count);
    }

    void initialize() override
    {
        m_mode = Utils::tolower(m_mode);
        if (m_mode == "partition")
            m_program.mode = pdg::DividerMode::Partition;
        else if (m_mode == "round_robin")
            m_program.mode = pdg::DividerMode::RoundRobin;
        else
            throwError("Invalid internal divider mode");
        if (m_count < 2U || m_count > 1000U)
            throwError("Option 'count' must be in the range [2, 1000].");
        m_program.count = static_cast<std::uint32_t>(m_count);
    }

    void prepared(PointTableRef) override
    {
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
    }

#if PDG_HAS_CUDA
    bool executeCuda(pdg::PointBatch& host,
                     std::vector<std::uint64_t>& permutation,
                     pdg::DividerPartitionResult& partition) const
    {
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, *deviceMemory);
        device.setSize(host.size());
        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory->allocate(host.size() * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
        partition = pdg::partitionDivider(
            device, m_program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation.data(),
                                       devicePermutation->data(),
                                       host.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
#endif

    PointViewSet run(PointViewPtr input) override
    {
        PointViewSet result;
        std::vector<PointViewPtr> outputs;
        outputs.reserve(static_cast<std::size_t>(m_program.count));
        for (std::uint32_t view = 0; view < m_program.count; ++view)
        {
            PointViewPtr output = input->makeNew();
            outputs.push_back(output);
            result.insert(output);
        }

        const std::size_t count = static_cast<std::size_t>(input->size());
        const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                  {0.0, 0.0, 0.0});
        pdg::HostMemoryResource hostMemory;
        pdg::PointBatch host(count, coordinates, *m_dimensions, hostMemory);
        host.setSize(count);
        std::vector<std::uint64_t> permutation(count);
        pdg::DividerPartitionResult partition;

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && count &&
            pdg::dividerMaySupportExactDevice(host, m_program))
        {
            try
            {
                if (!pdg::cudaDevices().empty())
                    usedCuda = executeCuda(host, permutation, partition);
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
            throwError("required exact CUDA hybrid divider path was not used");
        if (!usedCuda)
            partition =
                pdg::partitionDivider(host, m_program, permutation.data());

        std::size_t position = 0;
        for (std::size_t view = 0; view < outputs.size(); ++view)
        {
            const std::size_t end =
                position + static_cast<std::size_t>(partition.counts[view]);
            for (; position < end; ++position)
                outputs[view]->appendPoint(
                    *input, static_cast<PointId>(permutation[position]));
        }
        return result;
    }

    std::string m_mode = "partition";
    point_count_t m_count = 0;
    pdg::DividerProgram m_program;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridDividerStage),
    "Internal exact PDG count-based point-view divider", ""};

CREATE_STATIC_STAGE(PdgDividerFilter, s_info)

} // namespace pdal
