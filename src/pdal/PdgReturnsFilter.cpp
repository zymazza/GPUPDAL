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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pdal
{

// Behaviorally derived from the pinned upstream filters/ReturnsFilter.cpp;
// see NOTICE. View creation and publication order are observable behavior.
class PdgReturnsFilter final : public Filter
{
public:
    std::string getName() const override
    {
        // Preserve upstream log and diagnostic attribution. Registration is
        // still under the private stage name below.
        return "filters.returns";
    }

private:
    void addArgs(ProgramArgs& args) override
    {
        args.add("groups",
                 "Comma-separated list of return number groupings ('first', "
                 "'last', 'intermediate', or 'only')",
                 m_returnsString, {"last"});
    }

    void prepared(PointTableRef table) override
    {
        const PointLayoutPtr layout(table.layout());
        if (!layout->hasDim(Dimension::Id::ReturnNumber) ||
            !layout->hasDim(Dimension::Id::NumberOfReturns))
        {
            log()->get(LogLevel::Warning)
                << "Could not find ReturnNumber or "
                   "NumberOfReturns. Proceeding with all returns.\n";
        }
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
    }

    void parseGroups()
    {
        m_program.groups = 0U;
        for (std::string& group : m_returnsString)
        {
            Utils::trim(group);
            if (group == "first")
                m_program.groups |= pdg::ReturnFirst;
            else if (group == "intermediate")
                m_program.groups |= pdg::ReturnIntermediate;
            else if (group == "last")
                m_program.groups |= pdg::ReturnLast;
            else if (group == "only")
                m_program.groups |= pdg::ReturnOnly;
            else
                throwError("Invalid output type: '" + group + "'.");
        }
    }

    static void appendHost(PointView& input,
                           const std::array<PointViewPtr, 4>& outputs,
                           std::uint8_t groups)
    {
        for (PointId point = 0; point < input.size(); ++point)
        {
            PointRef source = input.point(point);
            const std::uint8_t returnNumber =
                source.getFieldAs<std::uint8_t>(Dimension::Id::ReturnNumber);
            const std::uint8_t numberOfReturns =
                source.getFieldAs<std::uint8_t>(Dimension::Id::NumberOfReturns);
            if ((groups & pdg::ReturnFirst) && returnNumber == 1U &&
                numberOfReturns > 1U)
                outputs[0]->appendPoint(input, point);
            if ((groups & pdg::ReturnIntermediate) && returnNumber > 1U &&
                returnNumber < numberOfReturns && numberOfReturns > 2U)
                outputs[1]->appendPoint(input, point);
            if ((groups & pdg::ReturnLast) && returnNumber == numberOfReturns &&
                numberOfReturns > 1U)
                outputs[2]->appendPoint(input, point);
            if ((groups & pdg::ReturnOnly) && numberOfReturns == 1U)
                outputs[3]->appendPoint(input, point);
        }
    }

#if PDG_HAS_CUDA
    bool executeCuda(const pdg::PointBatch& host,
                     std::vector<std::uint64_t>& permutation,
                     pdg::ReturnsPartitionResult& result) const
    {
        std::unique_ptr<pdg::MemoryResource> deviceMemory =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch device(host.size(), host.coordinateEncoding(),
                               *m_dimensions, *deviceMemory);
        const pdg::DimensionId returnNumber(
            pdg::StandardDimension::ReturnNumber);
        const pdg::DimensionId numberOfReturns(
            pdg::StandardDimension::NumberOfReturns);
        device.materialize(returnNumber, pdg::DimensionType::Unsigned8);
        device.materialize(numberOfReturns, pdg::DimensionType::Unsigned8);
        device.setSize(host.size());
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device.nativeStreamHandle());
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(returnNumber),
                                       host.rawData(returnNumber),
                                       host.size() * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice, stream));
        PDG_CUDA_CHECK(cudaMemcpyAsync(device.rawData(numberOfReturns),
                                       host.rawData(numberOfReturns),
                                       host.size() * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice, stream));
        std::unique_ptr<pdg::Allocation> devicePermutation =
            deviceMemory->allocate(host.size() * sizeof(std::uint64_t),
                                   alignof(std::uint64_t));
        result = pdg::partitionReturns(
            device, m_program,
            static_cast<std::uint64_t*>(devicePermutation->data()));
        PDG_CUDA_CHECK(cudaMemcpyAsync(permutation.data(),
                                       devicePermutation->data(),
                                       host.size() * sizeof(std::uint64_t),
                                       cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
#endif

    void publish(PointViewSet& result,
                 const std::array<PointViewPtr, 4>& outputs) const
    {
        static constexpr std::array<const char*, 4> Names = {
            "first", "intermediate", "last", "only"};
        static constexpr std::array<std::uint8_t, 4> Masks = {
            pdg::ReturnFirst, pdg::ReturnIntermediate, pdg::ReturnLast,
            pdg::ReturnOnly};
        for (std::size_t group = 0; group < outputs.size(); ++group)
        {
            if (!(m_program.groups & Masks[group]))
                continue;
            if (outputs[group]->size())
                result.insert(outputs[group]);
            else
                log()->get(LogLevel::Warning) << "Requested returns group '"
                                              << Names[group] << "' is empty\n";
        }
    }

    PointViewSet run(PointViewPtr view) override
    {
        parseGroups();
        std::array<PointViewPtr, 4> outputs = {
            view->makeNew(), view->makeNew(), view->makeNew(), view->makeNew()};

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requestCuda =
            requireCuda || std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        bool usedCuda = false;
        if (requestCuda && view->size() &&
            static_cast<std::size_t>(view->size()) <=
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            const std::size_t count = static_cast<std::size_t>(view->size());
            const pdg::DimensionId returnNumber(
                pdg::StandardDimension::ReturnNumber);
            const pdg::DimensionId numberOfReturns(
                pdg::StandardDimension::NumberOfReturns);
            const pdg::CoordinateEncoding coordinates({1.0, 1.0, 1.0},
                                                      {0.0, 0.0, 0.0});
            pdg::HostMemoryResource hostMemory;
            pdg::PointBatch host(count, coordinates, *m_dimensions, hostMemory);
            host.materialize(returnNumber, pdg::DimensionType::Unsigned8);
            host.materialize(numberOfReturns, pdg::DimensionType::Unsigned8);
            host.setSize(count);
            auto returnNumbers = host.hostSpan<std::uint8_t>(returnNumber);
            auto numbersOfReturns =
                host.hostSpan<std::uint8_t>(numberOfReturns);
            for (PointId point = 0; point < view->size(); ++point)
            {
                const std::size_t position = static_cast<std::size_t>(point);
                returnNumbers[position] = view->getFieldAs<std::uint8_t>(
                    Dimension::Id::ReturnNumber, point);
                numbersOfReturns[position] = view->getFieldAs<std::uint8_t>(
                    Dimension::Id::NumberOfReturns, point);
            }

            std::vector<std::uint64_t> permutation(count);
            pdg::ReturnsPartitionResult partition;
#if PDG_HAS_CUDA
            try
            {
                if (pdg::returnsMaySupportExactDevice(host, m_program) &&
                    !pdg::cudaDevices().empty())
                    usedCuda = executeCuda(host, permutation, partition);
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
            }
#endif
            if (usedCuda)
            {
                std::size_t position = 0;
                for (std::size_t group = 0; group < outputs.size(); ++group)
                {
                    const std::size_t end =
                        position +
                        static_cast<std::size_t>(partition.counts[group]);
                    for (; position < end; ++position)
                        outputs[group]->appendPoint(
                            *view, static_cast<PointId>(permutation[position]));
                }
            }
        }
        if (requireCuda && !usedCuda)
            throwError("required exact CUDA hybrid returns path was not used");
        if (!usedCuda)
            appendHost(*view, outputs, m_program.groups);

        PointViewSet result;
        publish(result, outputs);
        return result;
    }

    StringList m_returnsString;
    pdg::ReturnsProgram m_program;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
};

static StaticPluginInfo const s_info{
    std::string(pdg::HybridReturnsStage),
    "Internal exact PDG return-order partition", ""};

CREATE_STATIC_STAGE(PdgReturnsFilter, s_info)

} // namespace pdal
