#include <pdg/Coordinate.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pdg
{

namespace
{
void checkAxis(std::size_t axis)
{
    if (axis >= 3)
        throw std::out_of_range("coordinate axis must be 0, 1, or 2");
}
} // unnamed namespace

CoordinateEncoding::CoordinateEncoding(std::array<double, 3> scale,
                                       std::array<double, 3> offset)
    : m_scale(scale), m_offset(offset)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(m_scale[axis]) || m_scale[axis] <= 0.0)
            throw std::invalid_argument(
                "coordinate scales must be finite and positive");
        if (!std::isfinite(m_offset[axis]))
            throw std::invalid_argument("coordinate offsets must be finite");
    }
}

const std::array<double, 3>& CoordinateEncoding::scale() const noexcept
{
    return m_scale;
}

const std::array<double, 3>& CoordinateEncoding::offset() const noexcept
{
    return m_offset;
}

double CoordinateEncoding::decode(std::size_t axis, std::int32_t raw) const
{
    checkAxis(axis);
    // Keep multiply and add separate to match LAS/PDAL evaluation order.
    return static_cast<double>(raw) * m_scale[axis] + m_offset[axis];
}

std::int32_t CoordinateEncoding::encode(std::size_t axis, double value) const
{
    checkAxis(axis);
    if (!std::isfinite(value))
        throw std::invalid_argument("coordinate values must be finite");

    const double scaled = (value - m_offset[axis]) / m_scale[axis];
    constexpr double minimum =
        static_cast<double>(std::numeric_limits<std::int32_t>::min()) - 0.5;
    constexpr double maximum =
        static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 0.5;
    if (scaled <= minimum || scaled >= maximum)
        throw std::overflow_error(
            "coordinate does not fit the LAS int32 domain");
    return static_cast<std::int32_t>(std::llround(scaled));
}

TileFrame::TileFrame(CoordinateEncoding encoding, std::array<double, 3> origin)
    : m_encoding(std::move(encoding)), m_origin(origin)
{
    for (double value : m_origin)
        if (!std::isfinite(value))
            throw std::invalid_argument("tile origin must be finite");
}

float TileFrame::local(std::size_t axis, std::int32_t raw) const
{
    checkAxis(axis);
    const double value = m_encoding.decode(axis, raw) - m_origin[axis];
    if (value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value > static_cast<double>(std::numeric_limits<float>::max()))
        throw std::overflow_error("local coordinate does not fit fp32");
    return static_cast<float>(value);
}

const std::array<double, 3>& TileFrame::origin() const noexcept
{
    return m_origin;
}

} // namespace pdg
