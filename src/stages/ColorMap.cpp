#include <pdg/PointBatch.hpp>
#include <pdg/stages/ColorMap.hpp>

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
constexpr DimensionId Red(StandardDimension::Red);
constexpr DimensionId Green(StandardDimension::Green);
constexpr DimensionId Blue(StandardDimension::Blue);

bool validColumns(const PointBatch& batch,
                  const ColorMapProgram& program) noexcept
{
    try
    {
        return batch.has(program.value) && batch.has(Red) && batch.has(Green) &&
               batch.has(Blue) &&
               batch.columnInfo(program.value).physicalType ==
                   DimensionType::Double &&
               batch.columnInfo(Red).physicalType ==
                   DimensionType::Unsigned16 &&
               batch.columnInfo(Green).physicalType ==
                   DimensionType::Unsigned16 &&
               batch.columnInfo(Blue).physicalType == DimensionType::Unsigned16;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool validProgram(const ColorMapProgram& program, ColorRampView ramp) noexcept
{
    return std::isfinite(program.minimum) && std::isfinite(program.maximum) &&
           program.maximum > program.minimum && ramp.size && ramp.red &&
           ramp.green && ramp.blue;
}
} // unnamed namespace

void applyColorMapDevice(PointBatch& batch, const ColorMapProgram& program,
                         ColorRampView ramp);

bool colorMapMaySupportExactDevice(const PointBatch& hostBatch,
                                   const ColorMapProgram& program,
                                   ColorRampView ramp) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !validColumns(hostBatch, program) || !validProgram(program, ramp) ||
        ramp.size > static_cast<std::size_t>(
                        (std::numeric_limits<std::uint32_t>::max)()))
        return false;

    const auto* values = hostBatch.data<double>(program.value);
    for (std::size_t point = 0; point < hostBatch.size(); ++point)
        if (!std::isfinite(values[point]))
            return false;
    return true;
}

void applyColorMap(PointBatch& batch, const ColorMapProgram& program,
                   ColorRampView ramp)
{
    if (!validProgram(program, ramp))
        throw std::invalid_argument("invalid color-map program or ramp");
    if (!validColumns(batch, program))
        throw std::invalid_argument(
            "color-map value and RGB columns have invalid physical types");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        applyColorMapDevice(batch, program, ramp);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported color-map memory kind");

    const auto* values = batch.data<double>(program.value);
    auto* red = batch.data<std::uint16_t>(Red);
    auto* green = batch.data<std::uint16_t>(Green);
    auto* blue = batch.data<std::uint16_t>(Blue);
    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        double value = values[point];
        if (program.clamp)
            value = value < program.minimum
                        ? program.minimum
                        : (value > program.maximum ? program.maximum : value);
        if (value < program.minimum || value > program.maximum)
            continue;
        const double factor =
            (value - program.minimum) / (program.maximum - program.minimum);
        std::size_t position = static_cast<std::size_t>(
            std::floor(factor * static_cast<double>(ramp.size)));
        position = (std::min)(position, ramp.size - 1U);
        if (program.invert)
            position = (ramp.size - 1U) - position;
        red[point] = static_cast<std::uint16_t>(ramp.red[position]);
        green[point] = static_cast<std::uint16_t>(ramp.green[position]);
        blue[point] = static_cast<std::uint16_t>(ramp.blue[position]);
    }
}

} // namespace pdg
