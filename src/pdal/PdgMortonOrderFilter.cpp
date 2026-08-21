#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Morton.hpp>
#include <pdg/stages/Ordering.hpp>

#include <pdal/Filter.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace pdal
{

namespace
{
using Coordinate = std::pair<double, double>;

// Behaviorally derived from the pinned upstream
// filters/MortonOrderFilter.cpp; see NOTICE. Keeping the comparator and
// reverse-code arithmetic here makes degenerate or nonfinite host fallback
// behavior identical even when no exact radix key can be established.
bool lessMostSignificantBit(const int& left, const int& right)
{
    return left < right && left < (left ^ right);
}

class ZOrderCompare
{
public:
    bool operator()(const Coordinate& left, const Coordinate& right) const
    {
        const int a[2] = {static_cast<int>(left.first * INT_MAX),
                          static_cast<int>(left.second * INT_MAX)};
        const int b[2] = {static_cast<int>(right.first * INT_MAX),
                          static_cast<int>(right.second * INT_MAX)};
        int selected = 0;
        int difference = 0;
        for (int axis = 0; axis < 2; ++axis)
        {
            const int candidate = a[axis] ^ b[axis];
            if (lessMostSignificantBit(difference, candidate))
            {
                selected = axis;
                difference = candidate;
            }
        }
        return (a[selected] - b[selected]) < 0;
    }
};

std::uint32_t part1By1(std::uint32_t value)
{
    value &= 0x0000ffffU;
    value = (value ^ (value << 8U)) & 0x00ff00ffU;
    value = (value ^ (value << 4U)) & 0x0f0f0f0fU;
    value = (value ^ (value << 2U)) & 0x33333333U;
    value = (value ^ (value << 1U)) & 0x55555555U;
    return value;
}

std::uint32_t reverseBits(std::uint32_t value)
{
    value = ((value >> 1U) & 0x55555555U) | ((value & 0x55555555U) << 1U);
    value = ((value >> 2U) & 0x33333333U) | ((value & 0x33333333U) << 2U);
    value = ((value >> 4U) & 0x0f0f0f0fU) | ((value & 0x0f0f0f0fU) << 4U);
    value = ((value >> 8U) & 0x00ff00ffU) | ((value & 0x00ff00ffU) << 8U);
    return (value >> 16U) | (value << 16U);
}

std::uint32_t reverseCode(std::uint32_t x, std::uint32_t y)
{
    return reverseBits((part1By1(y) << 1U) + part1By1(x));
}
} // unnamed namespace

class PdgMortonOrderFilter final : public Filter
{
public:
    std::string getName() const override
    {
        return std::string(pdg::HybridMortonOrderStage);
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("reverse", "Reverse Morton", m_reverse, false);
    }

    void prepared(PointTableRef) override
    {
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        m_key =
            m_dimensions
                ->registerCustom("PdgMortonKey", pdg::DimensionType::Unsigned64)
                .id;
    }

    pdg::MortonProgram program(const BOX2D& bounds) const
    {
        pdg::MortonProgram result;
        result.bounds = {bounds.minx, bounds.miny, bounds.maxx, bounds.maxy};
        result.reverse = m_reverse;
        return result;
    }

    void gather(PointView& view, pdg::PointBatch& batch) const
    {
        const pdg::DimensionId x(pdg::StandardDimension::X);
        const pdg::DimensionId y(pdg::StandardDimension::Y);
        batch.setSize(static_cast<std::size_t>(view.size()));
        auto xValues = batch.hostSpan<double>(x);
        auto yValues = batch.hostSpan<double>(y);
        for (PointId point = 0; point < view.size(); ++point)
        {
            xValues[static_cast<std::size_t>(point)] =
                view.getFieldAs<double>(Dimension::Id::X, point);
            yValues[static_cast<std::size_t>(point)] =
                view.getFieldAs<double>(Dimension::Id::Y, point);
        }
    }

    PointIdList hostPermutation(PointView& view, const BOX2D& bounds) const
    {
        PointIdList permutation;
        permutation.reserve(static_cast<std::size_t>(view.size()));
        const double xRange = bounds.maxx - bounds.minx;
        const double yRange = bounds.maxy - bounds.miny;
        if (m_reverse)
        {
            const std::int32_t cell =
                static_cast<std::int32_t>(std::sqrt(view.size()));
            const double cellWidth = xRange / static_cast<double>(cell);
            const double cellHeight = yRange / static_cast<double>(cell);
            std::multimap<std::uint32_t, PointId> sorted;
            for (PointId point = 0; point < view.size(); ++point)
            {
                const double x =
                    view.getFieldAs<double>(Dimension::Id::X, point);
                const double y =
                    view.getFieldAs<double>(Dimension::Id::Y, point);
                const auto xPosition = static_cast<std::int32_t>(
                    std::floor((x - bounds.minx) / cellWidth));
                const auto yPosition = static_cast<std::int32_t>(
                    std::floor((y - bounds.miny) / cellHeight));
                sorted.emplace(
                    reverseCode(static_cast<std::uint32_t>(xPosition),
                                static_cast<std::uint32_t>(yPosition)),
                    point);
            }
            for (const auto& entry : sorted)
                permutation.push_back(entry.second);
            return permutation;
        }

        std::multimap<Coordinate, PointId, ZOrderCompare> sorted;
        for (PointId point = 0; point < view.size(); ++point)
        {
            const double x = (view.getFieldAs<double>(Dimension::Id::X, point) -
                              bounds.minx) /
                             xRange;
            const double y = (view.getFieldAs<double>(Dimension::Id::Y, point) -
                              bounds.miny) /
                             yRange;
            sorted.emplace(Coordinate{x, y}, point);
        }
        for (const auto& entry : sorted)
            permutation.push_back(entry.second);
        return permutation;
    }

#if PDG_HAS_CUDA
    bool executeCuda(const pdg::PointBatch& host,
                     const pdg::MortonProgram& morton,
                     PointIdList& permutation) const
    {
        const pdg::DimensionId x(pdg::StandardDimension::X);
        const pdg::DimensionId y(pdg::StandardDimension::Y);
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, *deviceMemory);
        device.materialize(x, pdg::DimensionType::Double);
        device.materialize(y, pdg::DimensionType::Double);
        device.materialize(m_key, pdg::DimensionType::Unsigned64);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        const std::size_t coordinateBytes = host.size() * sizeof(double);
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(x), host.rawData(x),
                                       coordinateBytes, cudaMemcpyHostToDevice,
                                       stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(y), host.rawData(y),
                                       coordinateBytes, cudaMemcpyHostToDevice,
                                       stream));
        pdg::generateMortonKeys(
            device, morton, static_cast<std::uint64_t*>(device.rawData(m_key)));

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
        PointViewSet output;
        if (view->empty())
        {
            if (m_reverse)
                output.insert(view->makeNew());
            return output;
        }

        BOX2D bounds;
        view->calculateBounds(bounds);
        const pdg::MortonProgram morton = program(bounds);
        const std::size_t count = static_cast<std::size_t>(view->size());
        PointIdList permutation(count);
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool cudaDisabled = std::getenv("PDG_DISABLE_CUDA_HYBRID");
        const bool automaticCuda =
            !cudaDisabled && pdg::preferDefaultCudaMorton(count, morton);
        const bool requestCuda =
            !cudaDisabled &&
            (requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
             automaticCuda);
        bool usedCuda = false;
#if PDG_HAS_CUDA
        if (requestCuda)
        {
            try
            {
                if (!pdg::cudaDevices().empty())
                {
                    const pdg::CoordinateEncoding coordinates(
                        {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
                    pdg::HostMemoryResource hostMemory;
                    pdg::PointBatch host(count, coordinates, *m_dimensions,
                                         hostMemory);
                    const pdg::DimensionId x(pdg::StandardDimension::X);
                    const pdg::DimensionId y(pdg::StandardDimension::Y);
                    host.materialize(x, pdg::DimensionType::Double);
                    host.materialize(y, pdg::DimensionType::Double);
                    gather(*view, host);

                    if (pdg::mortonMaySupportExactDevice(host, morton))
                        usedCuda = executeCuda(host, morton, permutation);
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
            throwError("required exact CUDA hybrid Morton path was not used");
        if (!usedCuda)
            permutation = hostPermutation(*view, bounds);

        PointViewPtr reordered = view->makeNew();
        for (PointId point : permutation)
            reordered->appendPoint(*view, point);
        output.insert(reordered);
        return output;
    }

    bool m_reverse = false;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::DimensionId m_key;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridMortonOrderStage),
                                     "Internal exact PDG Morton ordering", ""};

CREATE_STATIC_STAGE(PdgMortonOrderFilter, s_info)

} // namespace pdal
