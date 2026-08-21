#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace pdg
{

class PointBatch;

struct ColorMapProgram
{
    DimensionId value;
    double minimum = 0.0;
    double maximum = 1.0;
    bool clamp = false;
    bool invert = false;
};

struct ColorinterpProgram
{
    DimensionId dimension;
    double minimum = std::numeric_limits<double>::quiet_NaN();
    double maximum = std::numeric_limits<double>::quiet_NaN();
    bool clamp = false;
    std::string ramp = "pestel_shades";
    bool invert = false;
    bool mad = false;
    double madMultiplier = 1.4862;
    double k = 0.0;
};

struct ColorRampView
{
    const std::uint8_t* red = nullptr;
    const std::uint8_t* green = nullptr;
    const std::uint8_t* blue = nullptr;
    std::size_t size = 0;
};

[[nodiscard]] bool colorMapMaySupportExactDevice(const PointBatch& hostBatch,
                                                 const ColorMapProgram& program,
                                                 ColorRampView ramp) noexcept;

// Maps a materialized Double value column into the standard Unsigned16 RGB
// columns. Points outside an unclamped range retain their existing colors.
void applyColorMap(PointBatch& batch, const ColorMapProgram& program,
                   ColorRampView ramp);

} // namespace pdg
