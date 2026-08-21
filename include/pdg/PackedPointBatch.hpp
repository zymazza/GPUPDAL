#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace pdg
{

class PointBatch;

// Describes one physical field in a packed PDAL point record. Offsets and
// types refer to the in-memory representation, so transfers preserve every
// bit rather than converting through a logical value.
struct PackedPointColumn
{
    DimensionId id;
    DimensionType physicalType = DimensionType::None;
    std::size_t offset = 0;
    bool written = false;
};

// Transposes selected fields from device-resident packed records into a
// device PointBatch. pointIds is an optional device mapping from dense batch
// positions to packed-record positions. When originalIndexes is non-null the
// same mapping is retained for stable filtering and exact repack.
void unpackPackedPointBatchDevice(const void* packed, std::size_t pointStride,
                                  const std::uint64_t* pointIds,
                                  std::size_t pointCount,
                                  std::span<const PackedPointColumn> columns,
                                  PointBatch& destination,
                                  std::uint64_t* originalIndexes = nullptr);

// Writes only columns marked as written back to device-resident packed
// records. pointIds has one device-resident destination record index for each
// dense source point and may be null for the identity mapping.
void repackPackedPointBatchDevice(const PointBatch& source, void* packed,
                                  std::size_t pointStride,
                                  const std::uint64_t* pointIds,
                                  std::span<const PackedPointColumn> columns);

} // namespace pdg
