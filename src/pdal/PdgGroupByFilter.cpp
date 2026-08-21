#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ordering.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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

// Behaviorally derived from the pinned upstream filters/GroupByFilter.cpp;
// see NOTICE. The output-view creation sequence is part of writer numbering.
class PdgGroupByFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridGroupByStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("dimension", "Dimension containing data to be grouped",
                 m_dimensionName);
    }

    void prepared(PointTableRef table) override
    {
        m_pdalDimension = table.layout()->findDim(m_dimensionName);
        if (m_pdalDimension == Dimension::Id::Unknown)
            throwError("Invalid dimension name '" + m_dimensionName + "'.");
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        m_key =
            m_dimensions
                ->registerCustom("PdgGroupKey", pdg::DimensionType::Signed64)
                .id;
    }

    PointViewPtr outputFor(PointView& source, std::int64_t key)
    {
        PointViewPtr& output = m_outputs[key];
        if (!output)
            output = source.makeNew();
        return output;
    }

#if PDG_HAS_CUDA
    bool executeCuda(const pdg::PointBatch& host,
                     std::vector<std::uint64_t>& permutation) const
    {
        NvtxRange range("pdg::filters.groupby");
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, *deviceMemory);
        device.materialize(m_key, pdg::DimensionType::Signed64);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(m_key),
                                       host.rawData(m_key),
                                       host.size() * sizeof(std::int64_t),
                                       cudaMemcpyHostToDevice, stream));

        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory->allocate(host.size() * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
        pdg::OrderingProgram ordering;
        ordering.dimensions = {m_key};
        ordering.algorithm = pdg::OrderingAlgorithm::Stable;
        const pdg::OrderingResult result = pdg::orderPoints(
            device, ordering,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        if (!result.exact)
            return false;
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation.data(),
                                       devicePermutation->data(),
                                       host.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
#endif

    PointViewSet run(PointViewPtr view) override
    {
        PointViewSet result;
        const std::size_t count = static_cast<std::size_t>(view->size());
        if (!count)
        {
            for (const auto& entry : m_outputs)
                result.insert(entry.second);
            return result;
        }

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");

        if (!requestCuda)
        {
            for (PointId point = 0; point < view->size(); ++point)
            {
                const std::int64_t key =
                    view->getFieldAs<std::int64_t>(m_pdalDimension, point);
                outputFor(*view, key)->appendPoint(*view, point);
            }
            for (const auto& entry : m_outputs)
                result.insert(entry.second);
            return result;
        }

        std::vector<std::int64_t> keys(count);
        for (PointId point = 0; point < view->size(); ++point)
            keys[static_cast<std::size_t>(point)] =
                view->getFieldAs<std::int64_t>(m_pdalDimension, point);

        std::vector<std::uint64_t> permutation(count);
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda && count <= static_cast<std::size_t>(
                                        (std::numeric_limits<int>::max)()))
        {
            const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                      {0.0, 0.0, 0.0});
            pdg::HostMemoryResource hostMemory;
            pdg::PointBatch host(count, coordinates, *m_dimensions, hostMemory);
            host.materialize(m_key, pdg::DimensionType::Signed64);
            host.setSize(count);
            std::copy(keys.begin(), keys.end(),
                      host.hostSpan<std::int64_t>(m_key).begin());
            try
            {
                if (!pdg::cudaDevices().empty())
                    usedCuda = executeCuda(host, permutation);
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
            throwError("required exact CUDA hybrid groupby path was not used");

        if (usedCuda)
        {
            // PointView ids define PointViewSet order. Create views in source
            // first-occurrence order, exactly as the upstream map loop does.
            for (std::size_t point = 0; point < count; ++point)
                static_cast<void>(outputFor(*view, keys[point]));
            for (std::uint64_t point : permutation)
                outputFor(*view, keys[static_cast<std::size_t>(point)])
                    ->appendPoint(*view, static_cast<PointId>(point));
        }
        else
        {
            for (PointId point = 0; point < view->size(); ++point)
                outputFor(*view, keys[static_cast<std::size_t>(point)])
                    ->appendPoint(*view, point);
        }

        for (const auto& entry : m_outputs)
            result.insert(entry.second);
        return result;
    }

    std::string m_dimensionName;
    Dimension::Id m_pdalDimension = Dimension::Id::Unknown;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::DimensionId m_key;
    std::map<std::int64_t, PointViewPtr> m_outputs;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridGroupByStage),
                                     "Internal exact PDG categorical grouping",
                                     ""};

CREATE_STATIC_STAGE(PdgGroupByFilter, s_info)

} // namespace pdal
