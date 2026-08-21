#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Ferry.hpp>

#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr std::size_t MaximumBindingsPerLaunch = 16;

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

struct DeviceBinding
{
    const void* source = nullptr;
    void* destination = nullptr;
    DimensionType sourceType = DimensionType::None;
    DimensionType destinationType = DimensionType::None;
    bool hasSource = false;
};

struct DeviceProgram
{
    DeviceBinding bindings[MaximumBindingsPerLaunch];
    std::size_t count = 0;
};

__device__ double loadPhysical(const void* data, DimensionType type,
                               std::size_t index)
{
    switch (type)
    {
    case DimensionType::Signed8:
        return static_cast<const std::int8_t*>(data)[index];
    case DimensionType::Signed16:
        return static_cast<const std::int16_t*>(data)[index];
    case DimensionType::Signed32:
        return static_cast<const std::int32_t*>(data)[index];
    case DimensionType::Signed64:
        return static_cast<double>(
            static_cast<const std::int64_t*>(data)[index]);
    case DimensionType::Unsigned8:
        return static_cast<const std::uint8_t*>(data)[index];
    case DimensionType::Unsigned16:
        return static_cast<const std::uint16_t*>(data)[index];
    case DimensionType::Unsigned32:
        return static_cast<const std::uint32_t*>(data)[index];
    case DimensionType::Unsigned64:
        return static_cast<double>(
            static_cast<const std::uint64_t*>(data)[index]);
    case DimensionType::Float:
        return static_cast<const float*>(data)[index];
    case DimensionType::Double:
        return static_cast<const double*>(data)[index];
    case DimensionType::None:
        break;
    }
    return 0.0;
}

__device__ double symmetricRound(double value)
{
    return value > 0.0 ? floor(value + 0.5) : ceil(value - 0.5);
}

template <typename T>
__device__ void storeIntegral(void* data, std::size_t index, double value,
                              double lowest, double maximum)
{
    const double rounded = symmetricRound(value);
    if (rounded >= lowest && rounded <= maximum)
        static_cast<T*>(data)[index] = static_cast<T>(rounded);
}

__device__ void storePhysical(void* data, DimensionType type, std::size_t index,
                              double value)
{
    switch (type)
    {
    case DimensionType::Signed8:
        storeIntegral<std::int8_t>(data, index, value, -128.0, 127.0);
        return;
    case DimensionType::Signed16:
        storeIntegral<std::int16_t>(data, index, value, -32768.0, 32767.0);
        return;
    case DimensionType::Signed32:
        storeIntegral<std::int32_t>(data, index, value, -2147483648.0,
                                    2147483647.0);
        return;
    case DimensionType::Signed64:
        storeIntegral<std::int64_t>(data, index, value, -9223372036854775808.0,
                                    9223372036854775808.0);
        return;
    case DimensionType::Unsigned8:
        storeIntegral<std::uint8_t>(data, index, value, 0.0, 255.0);
        return;
    case DimensionType::Unsigned16:
        storeIntegral<std::uint16_t>(data, index, value, 0.0, 65535.0);
        return;
    case DimensionType::Unsigned32:
        storeIntegral<std::uint32_t>(data, index, value, 0.0, 4294967295.0);
        return;
    case DimensionType::Unsigned64:
        storeIntegral<std::uint64_t>(data, index, value, 0.0,
                                     18446744073709551616.0);
        return;
    case DimensionType::Float:
        if (isnan(value) ||
            (value >= -0x1.fffffep+127 && value <= 0x1.fffffep+127))
            static_cast<float*>(data)[index] = static_cast<float>(value);
        return;
    case DimensionType::Double:
        static_cast<double*>(data)[index] = value;
        return;
    case DimensionType::None:
        return;
    }
}

__global__ void ferryKernel(std::size_t pointCount, DeviceProgram program)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < pointCount; index += grid)
        for (std::size_t operation = 0; operation < program.count; ++operation)
        {
            const DeviceBinding& binding = program.bindings[operation];
            if (binding.hasSource)
                storePhysical(
                    binding.destination, binding.destinationType, index,
                    loadPhysical(binding.source, binding.sourceType, index));
        }
}

int gridSize(std::size_t count)
{
    const std::size_t blocks =
        count / BlockSize + static_cast<std::size_t>(count % BlockSize != 0);
    return static_cast<int>(
        std::min<std::size_t>(blocks, std::numeric_limits<int>::max()));
}

bool isCoordinate(DimensionId id) noexcept
{
    return id == DimensionId(StandardDimension::X) ||
           id == DimensionId(StandardDimension::Y) ||
           id == DimensionId(StandardDimension::Z);
}
} // unnamed namespace

void executeFerryDevice(PointBatch& batch, const FerryProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA ferry requires a device PointBatch");
    for (const FerryCopy& copy : program.copies)
        if (isCoordinate(copy.destination) ||
            (copy.hasSource && isCoordinate(copy.source)))
            throw std::invalid_argument(
                "CUDA ferry coordinate conversion is not yet in the exact "
                "device envelope");

    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    for (const FerryCopy& copy : program.copies)
    {
        const bool destinationPresent = batch.has(copy.destination);
        if (!destinationPresent)
            batch.materialize(copy.destination);
        if (copy.destinationCreated || !destinationPresent)
        {
            const ColumnInfo& destination = batch.columnInfo(copy.destination);
            PDG_CUDA_CHECK(cudaMemsetAsync(
                batch.rawData(copy.destination), 0,
                batch.size() * dimensionTypeSize(destination.physicalType),
                stream));
        }
        if (copy.hasSource && !batch.has(copy.source))
            throw std::invalid_argument(
                "ferry source column is not materialized");
    }
    if (batch.size() == 0 || program.copies.empty())
        return;

    NvtxRange range("pdg::filters.ferry");
    for (std::size_t begin = 0; begin < program.copies.size();
         begin += MaximumBindingsPerLaunch)
    {
        DeviceProgram deviceProgram;
        deviceProgram.count =
            std::min(MaximumBindingsPerLaunch, program.copies.size() - begin);
        for (std::size_t index = 0; index < deviceProgram.count; ++index)
        {
            const FerryCopy& copy = program.copies[begin + index];
            DeviceBinding& binding = deviceProgram.bindings[index];
            binding.hasSource = copy.hasSource;
            binding.destination = batch.rawData(copy.destination);
            binding.destinationType =
                batch.columnInfo(copy.destination).physicalType;
            if (copy.hasSource)
            {
                binding.source = batch.rawData(copy.source);
                binding.sourceType = batch.columnInfo(copy.source).physicalType;
            }
        }
        ferryKernel<<<gridSize(batch.size()), BlockSize, 0, stream>>>(
            batch.size(), deviceProgram);
        PDG_CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace pdg
