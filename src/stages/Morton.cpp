#include <pdg/PointBatch.hpp>
#include <pdg/stages/Morton.hpp>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

namespace
{
constexpr DimensionId X(StandardDimension::X);
constexpr DimensionId Y(StandardDimension::Y);

std::uint64_t forwardCode(std::uint32_t x, std::uint32_t y) noexcept
{
    std::uint64_t code = 0;
    for (unsigned int bit = 0; bit < 31U; ++bit)
    {
        code |= static_cast<std::uint64_t>((y >> bit) & 1U) << (2U * bit);
        code |= static_cast<std::uint64_t>((x >> bit) & 1U) << (2U * bit + 1U);
    }
    return code;
}

std::uint32_t part1By1(std::uint32_t value) noexcept
{
    value &= 0x0000ffffU;
    value = (value ^ (value << 8U)) & 0x00ff00ffU;
    value = (value ^ (value << 4U)) & 0x0f0f0f0fU;
    value = (value ^ (value << 2U)) & 0x33333333U;
    value = (value ^ (value << 1U)) & 0x55555555U;
    return value;
}

std::uint32_t reverseBits(std::uint32_t value) noexcept
{
    value = ((value >> 1U) & 0x55555555U) | ((value & 0x55555555U) << 1U);
    value = ((value >> 2U) & 0x33333333U) | ((value & 0x33333333U) << 2U);
    value = ((value >> 4U) & 0x0f0f0f0fU) | ((value & 0x0f0f0f0fU) << 4U);
    value = ((value >> 8U) & 0x00ff00ffU) | ((value & 0x00ff00ffU) << 8U);
    value = (value >> 16U) | (value << 16U);
    return value;
}

std::uint32_t reverseCode(std::uint32_t x, std::uint32_t y) noexcept
{
    return reverseBits((part1By1(y) << 1U) + part1By1(x));
}

bool finiteBounds(const MortonBounds& bounds) noexcept
{
    return std::isfinite(bounds.minX) && std::isfinite(bounds.minY) &&
           std::isfinite(bounds.maxX) && std::isfinite(bounds.maxY) &&
           bounds.maxX > bounds.minX && bounds.maxY > bounds.minY;
}

void validateColumns(const PointBatch& batch)
{
    if (!batch.has(X) || !batch.has(Y) ||
        batch.columnInfo(X).physicalType != DimensionType::Double ||
        batch.columnInfo(Y).physicalType != DimensionType::Double)
        throw std::invalid_argument(
            "Morton ordering requires logical double X and Y columns");
}

void generateMortonKeysHost(PointBatch& batch, const MortonProgram& program,
                            std::uint64_t* keys)
{
    const auto x = batch.hostSpan<double>(X);
    const auto y = batch.hostSpan<double>(Y);
    const double xRange = program.bounds.maxX - program.bounds.minX;
    const double yRange = program.bounds.maxY - program.bounds.minY;
    if (program.reverse)
    {
        const std::int32_t cell =
            static_cast<std::int32_t>(std::sqrt(batch.size()));
        const double cellWidth = xRange / static_cast<double>(cell);
        const double cellHeight = yRange / static_cast<double>(cell);
        for (std::size_t point = 0; point < batch.size(); ++point)
        {
            const std::int32_t xPosition = static_cast<std::int32_t>(
                std::floor((x[point] - program.bounds.minX) / cellWidth));
            const std::int32_t yPosition = static_cast<std::int32_t>(
                std::floor((y[point] - program.bounds.minY) / cellHeight));
            keys[point] = reverseCode(static_cast<std::uint32_t>(xPosition),
                                      static_cast<std::uint32_t>(yPosition));
        }
        return;
    }

    for (std::size_t point = 0; point < batch.size(); ++point)
    {
        const int xPosition = static_cast<int>(
            ((x[point] - program.bounds.minX) / xRange) * INT_MAX);
        const int yPosition = static_cast<int>(
            ((y[point] - program.bounds.minY) / yRange) * INT_MAX);
        keys[point] = forwardCode(static_cast<std::uint32_t>(xPosition),
                                  static_cast<std::uint32_t>(yPosition));
    }
}
} // unnamed namespace

void generateMortonKeysDevice(PointBatch& batch, const MortonProgram& program,
                              std::uint64_t* keys);

bool preferDefaultCudaMorton(std::size_t points,
                             const MortonProgram& program) noexcept
{
    constexpr std::size_t MinimumPoints = 2'000'000;
    return points >= MinimumPoints && finiteBounds(program.bounds);
}

bool mortonMaySupportExactDevice(const PointBatch& hostBatch,
                                 const MortonProgram& program) noexcept
{
    if ((hostBatch.memoryKind() != MemoryKind::Host &&
         hostBatch.memoryKind() != MemoryKind::PinnedHost) ||
        !hostBatch.size() ||
        hostBatch.size() >
            static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
        !finiteBounds(program.bounds))
        return false;
    try
    {
        validateColumns(hostBatch);
        const double* x = hostBatch.data<double>(X);
        const double* y = hostBatch.data<double>(Y);
        for (std::size_t point = 0; point < hostBatch.size(); ++point)
            if (!std::isfinite(x[point]) || !std::isfinite(y[point]) ||
                x[point] < program.bounds.minX ||
                x[point] > program.bounds.maxX ||
                y[point] < program.bounds.minY ||
                y[point] > program.bounds.maxY)
                return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

void generateMortonKeys(PointBatch& batch, const MortonProgram& program,
                        std::uint64_t* keys)
{
    validateColumns(batch);
    if (batch.size() && !keys)
        throw std::invalid_argument("Morton key output is null");
    if (!batch.size())
        return;
    if (!finiteBounds(program.bounds))
        throw std::invalid_argument(
            "Morton device key bounds must be finite and nondegenerate");
    if (batch.memoryKind() == MemoryKind::Device)
    {
        generateMortonKeysDevice(batch, program, keys);
        return;
    }
    if (batch.memoryKind() != MemoryKind::Host &&
        batch.memoryKind() != MemoryKind::PinnedHost)
        throw std::invalid_argument("unsupported Morton key memory kind");
    generateMortonKeysHost(batch, program, keys);
}

} // namespace pdg
