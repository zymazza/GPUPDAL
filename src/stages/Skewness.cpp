/*
 * This file derives from PDAL's BSD-licensed filters.skewnessbalancing
 * implementation. See NOTICE for the retained copyright and attribution.
 */

#include <pdg/stages/Skewness.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pdg
{

bool skewnessProgramValid(const SkewnessProgram& program) noexcept
{
    return program.onlyGround || program.groundClass != program.otherClass;
}

bool skewnessOrderingSizeWithinDeviceEnvelope(std::size_t size) noexcept
{
    return size &&
           size <= static_cast<std::size_t>((std::numeric_limits<int>::max)());
}

SkewnessResult classifySkewnessSorted(const double* z,
                                      std::uint8_t* classification,
                                      std::size_t size,
                                      const SkewnessProgram& program)
{
    if (!skewnessProgramValid(program))
        throw std::invalid_argument(
            "ground and non-ground classes must differ unless only_ground "
            "is true");
    if (size && (!z || !classification))
        throw std::invalid_argument("skewness input columns are null");
    if (!size)
        return {};

    const auto setClass = [classification](std::size_t first, std::size_t last,
                                           std::uint8_t value)
    {
        for (std::size_t point = first; point <= last; ++point)
            classification[point] = value;
    };

    std::uint64_t n = 0U;
    std::uint64_t n1 = 0U;
    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;
    std::size_t lastPositive = 0U;
    double skewness = 0.0;
    double lastSkewness = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t point = 0U; point < size; ++point)
    {
        n1 = n;
        ++n;
        const double delta = z[point] - m1;
        const double deltaN = delta / static_cast<double>(n);
        const double term1 = delta * deltaN * static_cast<double>(n1);
        m1 += deltaN;
        m3 += term1 * deltaN * static_cast<double>(n - 2U) - 3.0 * deltaN * m2;
        m2 += term1;
        skewness = std::sqrt(n) * m3 / std::pow(m2, 1.5);
        if (skewness > 0.0 && lastSkewness <= 0.0)
        {
            setClass(lastPositive, point - 1U, program.groundClass);
            lastPositive = point;
        }
        lastSkewness = skewness;
    }

    if (lastPositive == 0U && skewness <= 0.0)
        setClass(0U, size - 1U, program.groundClass);
    else if (!program.onlyGround)
        setClass(lastPositive, size - 1U, program.otherClass);

    SkewnessResult result;
    for (std::size_t point = 0U; point < size; ++point)
    {
        if (classification[point] == program.groundClass)
            ++result.groundPoints;
        else if (!program.onlyGround &&
                 classification[point] == program.otherClass)
            ++result.otherPoints;
    }
    return result;
}

} // namespace pdg
