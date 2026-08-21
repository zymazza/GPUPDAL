#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;
constexpr DimensionId Red(StandardDimension::Red);
constexpr DimensionId Green(StandardDimension::Green);
constexpr DimensionId Blue(StandardDimension::Blue);

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

unsigned int launchBlocks(std::size_t size)
{
    const std::size_t natural =
        (size - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    return static_cast<unsigned int>(
        (std::min)(natural, static_cast<std::size_t>(65535)));
}

__global__ void colorMapKernel(const double* values, std::size_t count,
                               double minimum, double maximum, bool clamp,
                               bool invert, const std::uint8_t* rampRed,
                               const std::uint8_t* rampGreen,
                               const std::uint8_t* rampBlue,
                               std::uint32_t rampSize, std::uint16_t* red,
                               std::uint16_t* green, std::uint16_t* blue)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t point = thread; point < count; point += grid)
    {
        double value = values[point];
        if (clamp)
            value =
                value < minimum ? minimum : (value > maximum ? maximum : value);
        if (value < minimum || value > maximum)
            continue;
        const double numerator = __dsub_rn(value, minimum);
        const double denominator = __dsub_rn(maximum, minimum);
        const double factor = __ddiv_rn(numerator, denominator);
        const double scaled = __dmul_rn(factor, static_cast<double>(rampSize));
        std::uint32_t position = static_cast<std::uint32_t>(floor(scaled));
        position = position < rampSize ? position : rampSize - 1U;
        if (invert)
            position = (rampSize - 1U) - position;
        red[point] = static_cast<std::uint16_t>(rampRed[position]);
        green[point] = static_cast<std::uint16_t>(rampGreen[position]);
        blue[point] = static_cast<std::uint16_t>(rampBlue[position]);
    }
}
} // unnamed namespace

void applyColorMapDevice(PointBatch& batch, const ColorMapProgram& program,
                         ColorRampView ramp)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument("CUDA color map requires a device batch");
    if (!batch.size())
        return;
    NvtxRange range("pdg::filters.colorinterp");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    colorMapKernel<<<launchBlocks(batch.size()), BlockSize, 0, stream>>>(
        batch.data<double>(program.value), batch.size(), program.minimum,
        program.maximum, program.clamp, program.invert, ramp.red, ramp.green,
        ramp.blue, static_cast<std::uint32_t>(ramp.size),
        batch.data<std::uint16_t>(Red), batch.data<std::uint16_t>(Green),
        batch.data<std::uint16_t>(Blue));
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
