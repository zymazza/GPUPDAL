#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pdg
{

class CoordinateEncoding
{
public:
    CoordinateEncoding(std::array<double, 3> scale,
                       std::array<double, 3> offset);

    [[nodiscard]] const std::array<double, 3>& scale() const noexcept;
    [[nodiscard]] const std::array<double, 3>& offset() const noexcept;

    [[nodiscard]] double decode(std::size_t axis, std::int32_t raw) const;
    [[nodiscard]] std::int32_t encode(std::size_t axis, double value) const;

private:
    std::array<double, 3> m_scale;
    std::array<double, 3> m_offset;
};

class TileFrame
{
public:
    TileFrame(CoordinateEncoding encoding, std::array<double, 3> origin);

    [[nodiscard]] float local(std::size_t axis, std::int32_t raw) const;
    [[nodiscard]] const std::array<double, 3>& origin() const noexcept;

private:
    CoordinateEncoding m_encoding;
    std::array<double, 3> m_origin;
};

} // namespace pdg
