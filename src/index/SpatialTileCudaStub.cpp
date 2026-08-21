#include <pdg/index/SpatialTile.hpp>

#include <stdexcept>

namespace pdg::detail
{

void tiledRadiusCountsDevice(const PointBatch&, const SpatialTileSet&,
                             std::uint8_t, double, DimensionRegistry&,
                             MemoryResource&, MemoryResource&,
                             std::span<std::uint32_t>)
{
    throw std::logic_error(
        "CUDA tiled radius execution is not compiled in this build");
}

void tiledRadiusScaledValuesDevice(const PointBatch&, const SpatialTileSet&,
                                   std::uint8_t, double, double,
                                   DimensionRegistry&, MemoryResource&,
                                   MemoryResource&, std::span<double>)
{
    throw std::logic_error(
        "CUDA tiled radius execution is not compiled in this build");
}

} // namespace pdg::detail
