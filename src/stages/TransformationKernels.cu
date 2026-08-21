#include <pdg/Cuda.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Transformation.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr int BlockSize = 256;

struct DeviceTransformationProgram
{
    double matrix[16];
};

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

__global__ void transformationKernel(double* xValues, double* yValues,
                                     double* zValues, std::size_t size,
                                     DeviceTransformationProgram program)
{
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const double* matrix = program.matrix;
    for (std::size_t point = thread; point < size; point += grid)
    {
        const double x = xValues[point];
        const double y = yValues[point];
        const double z = zValues[point];

        double scale = __dmul_rn(x, matrix[12]);
        scale = __dadd_rn(scale, __dmul_rn(y, matrix[13]));
        scale = __dadd_rn(scale, __dmul_rn(z, matrix[14]));
        scale = __dadd_rn(scale, matrix[15]);

        double outputX = __dmul_rn(x, matrix[0]);
        outputX = __dadd_rn(outputX, __dmul_rn(y, matrix[1]));
        outputX = __dadd_rn(outputX, __dmul_rn(z, matrix[2]));
        outputX = __dadd_rn(outputX, matrix[3]);
        outputX = __ddiv_rn(outputX, scale);

        double outputY = __dmul_rn(x, matrix[4]);
        outputY = __dadd_rn(outputY, __dmul_rn(y, matrix[5]));
        outputY = __dadd_rn(outputY, __dmul_rn(z, matrix[6]));
        outputY = __dadd_rn(outputY, matrix[7]);
        outputY = __ddiv_rn(outputY, scale);

        double outputZ = __dmul_rn(x, matrix[8]);
        outputZ = __dadd_rn(outputZ, __dmul_rn(y, matrix[9]));
        outputZ = __dadd_rn(outputZ, __dmul_rn(z, matrix[10]));
        outputZ = __dadd_rn(outputZ, matrix[11]);
        outputZ = __ddiv_rn(outputZ, scale);

        xValues[point] = outputX;
        yValues[point] = outputY;
        zValues[point] = outputZ;
    }
}
} // unnamed namespace

void executeTransformationDevice(PointBatch& batch,
                                 const TransformationProgram& program)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "CUDA transformation requires a device batch");
    if (!transformationSupportsExactDevice(batch, program))
        throw std::invalid_argument(
            "transformation is outside the exact CUDA affine envelope");
    if (!batch.size())
        return;
    const std::size_t naturalBlocks =
        (batch.size() - 1U) / static_cast<std::size_t>(BlockSize) + 1U;
    const int blocks = static_cast<int>((
        std::min<std::size_t>)(naturalBlocks, static_cast<std::size_t>(65535)));
    const cudaStream_t stream =
        static_cast<cudaStream_t>(batch.nativeStreamHandle());
    DeviceTransformationProgram deviceProgram{};
    std::copy(program.matrix.begin(), program.matrix.end(),
              deviceProgram.matrix);
    NvtxRange range("pdg::filters.transformation");
    transformationKernel<<<blocks, BlockSize, 0, stream>>>(
        batch.data<double>(DimensionId(StandardDimension::X)),
        batch.data<double>(DimensionId(StandardDimension::Y)),
        batch.data<double>(DimensionId(StandardDimension::Z)), batch.size(),
        deviceProgram);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg
