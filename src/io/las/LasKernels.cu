#include <pdg/io/LasCuda.hpp>

#include <pdg/PointBatch.hpp>

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace pdg::las
{

namespace
{
constexpr int BlockSize = 256;
constexpr std::size_t CanonicalPointBytes = 36;
constexpr std::size_t MaximumCanonicalColumns = 21;

__host__ __device__ constexpr std::uint32_t
id(StandardDimension dimension) noexcept
{
    return static_cast<std::uint16_t>(dimension);
}

struct DeviceColumn
{
    void* data = nullptr;
    std::uint32_t dimension = 0;
};

struct DeviceColumnSet
{
    DeviceColumn columns[MaximumCanonicalColumns]{};
    std::uint32_t count = 0;
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

__device__ std::int32_t loadLe32(const std::uint8_t* bytes)
{
    const std::uint32_t value = static_cast<std::uint32_t>(bytes[0]) |
                                (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                (static_cast<std::uint32_t>(bytes[3]) << 24U);
    if (value <= 0x7fffffffU)
        return static_cast<std::int32_t>(value);
    return -1 - static_cast<std::int32_t>(0xffffffffU - value);
}

__device__ std::uint16_t loadLe16(const std::uint8_t* bytes)
{
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1])
                                      << 8U);
}

__device__ std::int16_t loadLeSigned16(const std::uint8_t* bytes)
{
    const std::uint16_t value = loadLe16(bytes);
    if (value <= 0x7fffU)
        return static_cast<std::int16_t>(value);
    return static_cast<std::int16_t>(
        -1 - static_cast<std::int32_t>(0xffffU - value));
}

__device__ double loadLeDouble(const std::uint8_t* bytes)
{
    unsigned long long bits = 0;
#pragma unroll
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        bits |= static_cast<unsigned long long>(bytes[shift / 8U]) << shift;
    return __longlong_as_double(static_cast<long long>(bits));
}

__device__ void storeLe32(std::uint8_t* bytes, std::int32_t signedValue)
{
    const std::uint32_t value = static_cast<std::uint32_t>(signedValue);
    bytes[0] = static_cast<std::uint8_t>(value);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
    bytes[2] = static_cast<std::uint8_t>(value >> 16U);
    bytes[3] = static_cast<std::uint8_t>(value >> 24U);
}

__device__ void storeLe16(std::uint8_t* bytes, std::uint16_t value)
{
    bytes[0] = static_cast<std::uint8_t>(value);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

__device__ void storeLeDouble(std::uint8_t* bytes, double value)
{
    const unsigned long long bits =
        static_cast<unsigned long long>(__double_as_longlong(value));
#pragma unroll
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        bytes[shift / 8U] = static_cast<std::uint8_t>(bits >> shift);
}

__device__ void decodeCanonicalColumn(const std::uint8_t* record,
                                      const DeviceColumn& column,
                                      std::size_t index)
{
    const std::uint32_t dimension = column.dimension;
    if (dimension >= id(StandardDimension::X) &&
        dimension <= id(StandardDimension::Z))
    {
        const std::size_t axis = dimension - id(StandardDimension::X);
        static_cast<std::int32_t*>(column.data)[index] =
            loadLe32(record + axis * sizeof(std::int32_t));
        return;
    }
    if (dimension == id(StandardDimension::Intensity) ||
        dimension == id(StandardDimension::PointSourceId) ||
        dimension == id(StandardDimension::Red) ||
        dimension == id(StandardDimension::Green) ||
        dimension == id(StandardDimension::Blue))
    {
        std::size_t offset = 12;
        if (dimension == id(StandardDimension::PointSourceId))
            offset = 20;
        else if (dimension == id(StandardDimension::Red))
            offset = 30;
        else if (dimension == id(StandardDimension::Green))
            offset = 32;
        else if (dimension == id(StandardDimension::Blue))
            offset = 34;
        static_cast<std::uint16_t*>(column.data)[index] =
            loadLe16(record + offset);
        return;
    }
    if (dimension == id(StandardDimension::GpsTime))
    {
        static_cast<double*>(column.data)[index] = loadLeDouble(record + 22);
        return;
    }
    if (dimension == id(StandardDimension::ScanAngleRank))
    {
        const double scaled =
            __dmul_rn(static_cast<double>(loadLeSigned16(record + 18)), 0.006);
        static_cast<float*>(column.data)[index] = __double2float_rn(scaled);
        return;
    }

    const std::uint8_t returns = record[14];
    const std::uint8_t flags = record[15];
    std::uint8_t value = 0;
    if (dimension == id(StandardDimension::ReturnNumber))
        value = returns & 0x0fU;
    else if (dimension == id(StandardDimension::NumberOfReturns))
        value = (returns >> 4U) & 0x0fU;
    else if (dimension == id(StandardDimension::Synthetic))
        value = flags & 0x01U;
    else if (dimension == id(StandardDimension::KeyPoint))
        value = (flags >> 1U) & 0x01U;
    else if (dimension == id(StandardDimension::Withheld))
        value = (flags >> 2U) & 0x01U;
    else if (dimension == id(StandardDimension::Overlap))
        value = (flags >> 3U) & 0x01U;
    else if (dimension == id(StandardDimension::ScanChannel))
        value = (flags >> 4U) & 0x03U;
    else if (dimension == id(StandardDimension::ScanDirectionFlag))
        value = (flags >> 6U) & 0x01U;
    else if (dimension == id(StandardDimension::EdgeOfFlightLine))
        value = (flags >> 7U) & 0x01U;
    else if (dimension == id(StandardDimension::Classification))
        value = record[16];
    else if (dimension == id(StandardDimension::UserData))
        value = record[17];
    static_cast<std::uint8_t*>(column.data)[index] = value;
}

__device__ void packCanonicalColumn(std::uint8_t* record,
                                    const DeviceColumn& column,
                                    std::size_t index)
{
    const std::uint32_t dimension = column.dimension;
    if (dimension >= id(StandardDimension::X) &&
        dimension <= id(StandardDimension::Z))
    {
        const std::size_t axis = dimension - id(StandardDimension::X);
        storeLe32(record + axis * sizeof(std::int32_t),
                  static_cast<const std::int32_t*>(column.data)[index]);
        return;
    }
    if (dimension == id(StandardDimension::Intensity) ||
        dimension == id(StandardDimension::PointSourceId) ||
        dimension == id(StandardDimension::Red) ||
        dimension == id(StandardDimension::Green) ||
        dimension == id(StandardDimension::Blue))
    {
        std::size_t offset = 12;
        if (dimension == id(StandardDimension::PointSourceId))
            offset = 20;
        else if (dimension == id(StandardDimension::Red))
            offset = 30;
        else if (dimension == id(StandardDimension::Green))
            offset = 32;
        else if (dimension == id(StandardDimension::Blue))
            offset = 34;
        storeLe16(record + offset,
                  static_cast<const std::uint16_t*>(column.data)[index]);
        return;
    }
    if (dimension == id(StandardDimension::GpsTime))
    {
        storeLeDouble(record + 22,
                      static_cast<const double*>(column.data)[index]);
        return;
    }

    const std::uint8_t value =
        static_cast<const std::uint8_t*>(column.data)[index];
    if (dimension == id(StandardDimension::ReturnNumber))
        record[14] =
            static_cast<std::uint8_t>((record[14] & 0xf0U) | (value & 0x0fU));
    else if (dimension == id(StandardDimension::NumberOfReturns))
        record[14] = static_cast<std::uint8_t>((record[14] & 0x0fU) |
                                               ((value & 0x0fU) << 4U));
    else if (dimension == id(StandardDimension::Classification))
        record[16] = value;
    else if (dimension == id(StandardDimension::UserData))
        record[17] = value;
    else
    {
        unsigned shift = 0;
        std::uint8_t mask = 0x01U;
        if (dimension == id(StandardDimension::KeyPoint))
            shift = 1;
        else if (dimension == id(StandardDimension::Withheld))
            shift = 2;
        else if (dimension == id(StandardDimension::Overlap))
            shift = 3;
        else if (dimension == id(StandardDimension::ScanChannel))
        {
            shift = 4;
            mask = 0x03U;
        }
        else if (dimension == id(StandardDimension::ScanDirectionFlag))
            shift = 6;
        else if (dimension == id(StandardDimension::EdgeOfFlightLine))
            shift = 7;
        const std::uint8_t shiftedMask =
            static_cast<std::uint8_t>(mask << shift);
        record[15] = static_cast<std::uint8_t>(
            (record[15] & static_cast<std::uint8_t>(~shiftedMask)) |
            static_cast<std::uint8_t>((value & mask) << shift));
    }
}

__global__ void decodeCoordinatesKernel(const std::uint8_t* records,
                                        std::size_t stride, std::size_t count,
                                        std::int32_t* x, std::int32_t* y,
                                        std::int32_t* z)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < count; index += grid)
    {
        const std::uint8_t* record = records + index * stride;
        x[index] = loadLe32(record);
        y[index] = loadLe32(record + 4);
        z[index] = loadLe32(record + 8);
    }
}

__global__ void packCoordinatesKernel(const std::int32_t* x,
                                      const std::int32_t* y,
                                      const std::int32_t* z, std::size_t count,
                                      std::uint8_t* records, std::size_t stride)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < count; index += grid)
    {
        std::uint8_t* record = records + index * stride;
        storeLe32(record, x[index]);
        storeLe32(record + 4, y[index]);
        storeLe32(record + 8, z[index]);
    }
}

__global__ void expandCoordinatesKernel(
    const std::int32_t* rawX, const std::int32_t* rawY,
    const std::int32_t* rawZ, std::size_t count, double scaleX,
    double scaleY, double scaleZ, double offsetX, double offsetY,
    double offsetZ, double* x, double* y, double* z)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < count; index += grid)
    {
        x[index] = static_cast<double>(rawX[index]) * scaleX + offsetX;
        y[index] = static_cast<double>(rawY[index]) * scaleY + offsetY;
        z[index] = static_cast<double>(rawZ[index]) * scaleZ + offsetZ;
    }
}

__global__ void decodeCanonicalColumnsKernel(const std::uint8_t* records,
                                             std::size_t count,
                                             DeviceColumnSet columns)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < count; index += grid)
    {
        const std::uint8_t* record = records + index * CanonicalPointBytes;
        for (std::uint32_t column = 0; column < columns.count; ++column)
            decodeCanonicalColumn(record, columns.columns[column], index);
    }
}

__global__ void packCanonicalColumnsKernel(std::uint8_t* records,
                                           std::size_t count,
                                           DeviceColumnSet columns)
{
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(blockDim.x) +
                               static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid = static_cast<std::size_t>(blockDim.x) *
                             static_cast<std::size_t>(gridDim.x);
    for (std::size_t index = thread; index < count; index += grid)
    {
        std::uint8_t* record = records + index * CanonicalPointBytes;
        for (std::uint32_t column = 0; column < columns.count; ++column)
            packCanonicalColumn(record, columns.columns[column], index);
    }
}

int gridSize(std::size_t count)
{
    const std::size_t blocks =
        count / BlockSize + static_cast<std::size_t>(count % BlockSize != 0);
    return static_cast<int>(
        std::min<std::size_t>(blocks, std::numeric_limits<int>::max()));
}

void validate(const void* records, std::size_t stride, std::size_t count,
              const void* x, const void* y, const void* z)
{
    if (stride < 12)
        throw std::invalid_argument("LAS record stride is too small for XYZ");
    if (count > std::numeric_limits<std::size_t>::max() / stride)
        throw std::overflow_error("LAS CUDA record extent overflows size_t");
    if (count && (!records || !x || !y || !z))
        throw std::invalid_argument(
            "LAS CUDA decode/pack received a null pointer");
}

DimensionType canonicalPhysicalType(DimensionId dimension) noexcept
{
    if (dimension == DimensionId(StandardDimension::X) ||
        dimension == DimensionId(StandardDimension::Y) ||
        dimension == DimensionId(StandardDimension::Z))
        return DimensionType::Signed32;
    if (dimension == DimensionId(StandardDimension::Intensity) ||
        dimension == DimensionId(StandardDimension::PointSourceId) ||
        dimension == DimensionId(StandardDimension::Red) ||
        dimension == DimensionId(StandardDimension::Green) ||
        dimension == DimensionId(StandardDimension::Blue))
        return DimensionType::Unsigned16;
    if (dimension == DimensionId(StandardDimension::GpsTime))
        return DimensionType::Double;
    if (dimension == DimensionId(StandardDimension::ScanAngleRank))
        return DimensionType::Float;
    if (dimension == DimensionId(StandardDimension::ReturnNumber) ||
        dimension == DimensionId(StandardDimension::NumberOfReturns) ||
        dimension == DimensionId(StandardDimension::ScanDirectionFlag) ||
        dimension == DimensionId(StandardDimension::EdgeOfFlightLine) ||
        dimension == DimensionId(StandardDimension::Classification) ||
        dimension == DimensionId(StandardDimension::UserData) ||
        dimension == DimensionId(StandardDimension::ScanChannel) ||
        dimension == DimensionId(StandardDimension::Synthetic) ||
        dimension == DimensionId(StandardDimension::KeyPoint) ||
        dimension == DimensionId(StandardDimension::Withheld) ||
        dimension == DimensionId(StandardDimension::Overlap))
        return DimensionType::Unsigned8;
    return DimensionType::None;
}

DeviceColumnSet bindCanonicalColumns(const PointBatch& batch,
                                     std::span<const DimensionId> dimensions,
                                     bool packing)
{
    if (batch.memoryKind() != MemoryKind::Device)
        throw std::invalid_argument(
            "canonical LAS CUDA columns require a device PointBatch");
    if (dimensions.size() > MaximumCanonicalColumns)
        throw std::invalid_argument("too many canonical LAS CUDA columns");
    DeviceColumnSet result;
    for (const DimensionId dimension : dimensions)
    {
        const DimensionType expected = canonicalPhysicalType(dimension);
        if (expected == DimensionType::None ||
            (packing &&
             dimension == DimensionId(StandardDimension::ScanAngleRank)))
            throw std::invalid_argument(
                "unsupported canonical LAS CUDA column");
        if (!batch.has(dimension) ||
            batch.columnInfo(dimension).physicalType != expected)
            throw std::invalid_argument(
                "canonical LAS CUDA column is not materialized with its "
                "physical type");
        DeviceColumn& target = result.columns[result.count++];
        target.data = const_cast<void*>(batch.rawData(dimension));
        target.dimension = dimension.value();
    }
    return result;
}

void validateCanonical(const void* records, std::size_t count,
                       const PointBatch& batch, cudaStream_t stream)
{
    if (count > batch.size())
        throw std::invalid_argument(
            "canonical LAS CUDA column count exceeds batch size");
    if (count > std::numeric_limits<std::size_t>::max() / CanonicalPointBytes)
        throw std::overflow_error(
            "canonical LAS CUDA record extent overflows size_t");
    if (count && !records)
        throw std::invalid_argument(
            "canonical LAS CUDA columns received a null record pointer");
    if (stream != static_cast<cudaStream_t>(batch.nativeStreamHandle()))
        throw std::invalid_argument(
            "canonical LAS CUDA columns require the batch allocator stream");
}
} // unnamed namespace

void decodeCoordinatesAsync(const void* interleavedRecords,
                            std::size_t recordStride, std::size_t count,
                            std::int32_t* x, std::int32_t* y, std::int32_t* z,
                            cudaStream_t stream)
{
    validate(interleavedRecords, recordStride, count, x, y, z);
    if (!count)
        return;
    NvtxRange range("pdg::las::decodeCoordinates");
    decodeCoordinatesKernel<<<gridSize(count), BlockSize, 0, stream>>>(
        static_cast<const std::uint8_t*>(interleavedRecords), recordStride,
        count, x, y, z);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void packCoordinatesAsync(const std::int32_t* x, const std::int32_t* y,
                          const std::int32_t* z, std::size_t count,
                          void* interleavedRecords, std::size_t recordStride,
                          cudaStream_t stream)
{
    validate(interleavedRecords, recordStride, count, x, y, z);
    if (!count)
        return;
    NvtxRange range("pdg::las::packCoordinates");
    packCoordinatesKernel<<<gridSize(count), BlockSize, 0, stream>>>(
        x, y, z, count, static_cast<std::uint8_t*>(interleavedRecords),
        recordStride);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void expandCoordinatesAsync(const std::int32_t* rawX,
                            const std::int32_t* rawY,
                            const std::int32_t* rawZ, std::size_t count,
                            const CoordinateEncoding& encoding, double* x,
                            double* y, double* z, cudaStream_t stream)
{
    if (count && (!rawX || !rawY || !rawZ || !x || !y || !z))
        throw std::invalid_argument(
            "LAS CUDA coordinate expansion received a null pointer");
    if (!count)
        return;
    NvtxRange range("pdg::las::expandCoordinates");
    const std::array<double, 3>& scale = encoding.scale();
    const std::array<double, 3>& offset = encoding.offset();
    expandCoordinatesKernel<<<gridSize(count), BlockSize, 0, stream>>>(
        rawX, rawY, rawZ, count, scale[0], scale[1], scale[2], offset[0],
        offset[1], offset[2], x, y, z);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void decodeCanonicalColumnsAsync(const void* canonicalRecords,
                                 std::size_t count, PointBatch& batch,
                                 std::span<const DimensionId> dimensions,
                                 cudaStream_t stream)
{
    validateCanonical(canonicalRecords, count, batch, stream);
    const DeviceColumnSet columns =
        bindCanonicalColumns(batch, dimensions, false);
    if (!count || !columns.count)
        return;
    NvtxRange range("pdg::las::decodeCanonicalColumns");
    decodeCanonicalColumnsKernel<<<gridSize(count), BlockSize, 0, stream>>>(
        static_cast<const std::uint8_t*>(canonicalRecords), count, columns);
    PDG_CUDA_CHECK(cudaGetLastError());
}

void packCanonicalColumnsAsync(const PointBatch& batch,
                               std::span<const DimensionId> dimensions,
                               std::size_t count, void* canonicalRecords,
                               cudaStream_t stream)
{
    validateCanonical(canonicalRecords, count, batch, stream);
    const DeviceColumnSet columns =
        bindCanonicalColumns(batch, dimensions, true);
    if (!count || !columns.count)
        return;
    NvtxRange range("pdg::las::packCanonicalColumns");
    packCanonicalColumnsKernel<<<gridSize(count), BlockSize, 0, stream>>>(
        static_cast<std::uint8_t*>(canonicalRecords), count, columns);
    PDG_CUDA_CHECK(cudaGetLastError());
}

} // namespace pdg::las
