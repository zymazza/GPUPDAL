#pragma once

#include <pdg/Cuda.hpp>
#include <pdg/Coordinate.hpp>
#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace pdg
{
class PointBatch;
}

#if PDG_HAS_CUDA
namespace pdg::las
{

void decodeCoordinatesAsync(const void* interleavedRecords,
                            std::size_t recordStride, std::size_t count,
                            std::int32_t* x, std::int32_t* y, std::int32_t* z,
                            cudaStream_t stream);
void packCoordinatesAsync(const std::int32_t* x, const std::int32_t* y,
                          const std::int32_t* z, std::size_t count,
                          void* interleavedRecords, std::size_t recordStride,
                          cudaStream_t stream);
// Expand raw LAS coordinates into the logical binary64 columns consumed by
// the resident spatial-index path. Multiply and add remain separate under the
// product target's exact CUDA flags, matching CoordinateEncoding::decode.
void expandCoordinatesAsync(const std::int32_t* rawX,
                            const std::int32_t* rawY,
                            const std::int32_t* rawZ, std::size_t count,
                            const CoordinateEncoding& encoding, double* x,
                            double* y, double* z, cudaStream_t stream);

// Transpose the canonical 36-byte LAS 1.4 format-7 representation used by the
// exact default writer. Only standard dimensions are transferred; custom
// point-program intermediates remain allocator-owned SoA columns.
void decodeCanonicalColumnsAsync(const void* canonicalRecords,
                                 std::size_t count, PointBatch& batch,
                                 std::span<const DimensionId> dimensions,
                                 cudaStream_t stream);
void packCanonicalColumnsAsync(const PointBatch& batch,
                               std::span<const DimensionId> dimensions,
                               std::size_t count, void* canonicalRecords,
                               cudaStream_t stream);

} // namespace pdg::las
#endif
