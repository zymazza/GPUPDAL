#include <pdg/SyntheticCloud.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace pdg
{
namespace
{
// splitmix64: a small, deterministic, well-distributed hash of (seed, index).
[[nodiscard]] std::uint64_t mix(std::uint64_t value) noexcept
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

[[nodiscard]] double unit(std::uint64_t seed, std::size_t index,
                          std::uint64_t salt) noexcept
{
    const std::uint64_t bits = mix(seed ^ (index * 0x9E3779B97F4A7C15ULL) ^
                                   (salt * 0xD1B54A32D192ED03ULL));
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

[[nodiscard]] double terrain(double x, double y) noexcept
{
    return 20.0 + 5.0 * std::sin(x / 60.0) + 4.0 * std::cos(y / 45.0) +
           2.0 * std::sin((x + y) / 17.0) + 0.5 * std::sin(x / 3.5) *
                                                std::cos(y / 4.25);
}

void put16(std::vector<unsigned char>& out, std::size_t at, std::uint16_t v)
{
    out[at] = static_cast<unsigned char>(v & 0xFFU);
    out[at + 1] = static_cast<unsigned char>(v >> 8);
}
void put32(std::vector<unsigned char>& out, std::size_t at, std::uint32_t v)
{
    for (int i = 0; i < 4; ++i)
        out[at + static_cast<std::size_t>(i)] =
            static_cast<unsigned char>((v >> (8 * i)) & 0xFFU);
}
void put64(std::vector<unsigned char>& out, std::size_t at, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        out[at + static_cast<std::size_t>(i)] =
            static_cast<unsigned char>((v >> (8 * i)) & 0xFFU);
}
void putDouble(std::vector<unsigned char>& out, std::size_t at, double v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    put64(out, at, bits);
}
} // unnamed namespace

SyntheticCloudGenerator::SyntheticCloudGenerator(
    SyntheticCloudParameters parameters)
    : m_parameters(parameters)
{
    if (m_parameters.points == 0U)
        throw std::invalid_argument("synthetic cloud needs at least one point");
    if (!(m_parameters.pointsPerSquareMetre > 0.0) ||
        !(m_parameters.stripWidthMetres > 0.0))
        throw std::invalid_argument("synthetic cloud density and strip width "
                                    "must be positive");
    m_side = std::sqrt(static_cast<double>(m_parameters.points) /
                       m_parameters.pointsPerSquareMetre);
    m_strips = static_cast<std::size_t>(
        std::max(1.0, std::ceil(m_side / m_parameters.stripWidthMetres)));
    m_pointsPerStrip = (m_parameters.points + m_strips - 1U) / m_strips;
    // Points across one scan line: strip width times density times the
    // along-track advance of one scan (~0.5 m).
    m_pointsPerScan = std::max(
        8.0, m_parameters.stripWidthMetres * m_parameters.pointsPerSquareMetre *
                 0.5);
}

SyntheticPoint SyntheticCloudGenerator::point(std::size_t index) const noexcept
{
    const std::uint64_t seed = m_parameters.seed;
    const std::size_t strip = index / m_pointsPerStrip;
    const std::size_t within = index % m_pointsPerStrip;
    const double along =
        static_cast<double>(within) / static_cast<double>(m_pointsPerStrip);
    // Alternate strip direction like real flight lines.
    const double y = (strip % 2U == 0U ? along : 1.0 - along) * m_side;
    const double stripX0 =
        static_cast<double>(strip) * m_parameters.stripWidthMetres;
    const double phase = 2.0 * std::numbers::pi *
                         static_cast<double>(within) / m_pointsPerScan;
    const double across = 0.5 + 0.5 * std::sin(phase);
    const double x =
        std::min(m_side, stripX0 + across * m_parameters.stripWidthMetres) +
        (unit(seed, index, 1) - 0.5) * 0.6;
    const double yj = y + (unit(seed, index, 2) - 0.5) * 0.6;
    SyntheticPoint result;
    result.x = std::clamp(x, 0.0, m_side);
    result.y = std::clamp(yj, 0.0, m_side);
    const double ground = terrain(result.x, result.y);
    const bool vegetation = unit(seed, index, 3) < m_parameters.vegetationFraction;
    if (vegetation)
    {
        result.z = ground + 1.5 + 12.0 * unit(seed, index, 4);
        result.returnNumber = 1;
        result.numberOfReturns = 2;
        result.classification = 1;
        result.intensity = static_cast<std::uint16_t>(
            60.0 + 120.0 * unit(seed, index, 5));
    }
    else
    {
        result.z = ground + (unit(seed, index, 4) - 0.5) * 0.06;
        const bool lastOfTwo = unit(seed, index, 6) < 0.3;
        result.returnNumber = lastOfTwo ? 2 : 1;
        result.numberOfReturns = lastOfTwo ? 2 : 1;
        result.classification = 2;
        result.intensity = static_cast<std::uint16_t>(
            900.0 + 700.0 * unit(seed, index, 5));
    }
    result.gpsTime = 300000.0 + static_cast<double>(index) * 2.0e-6;
    const auto channel = [&](double v)
    {
        return static_cast<std::uint16_t>(
            std::clamp(v, 0.0, 65535.0));
    };
    result.rgb = {channel(2000.0 * (result.z - 10.0)),
                  channel(vegetation ? 30000.0 : 12000.0),
                  channel(65535.0 * unit(seed, index, 7) * 0.2)};
    return result;
}

std::size_t writeSyntheticLas(const std::filesystem::path& path,
                              const SyntheticCloudParameters& parameters)
{
    const SyntheticCloudGenerator generator(parameters);
    constexpr std::size_t HeaderSize = 375;
    constexpr std::size_t RecordSize = 36;
    constexpr double Scale = 0.01;
    const std::size_t count = parameters.points;

    std::vector<unsigned char> header(HeaderSize, 0);
    std::memcpy(header.data(), "LASF", 4);
    // file source id (4-5); global encoding (6-7): GPS standard time and,
    // as the 1.4 specification requires for record formats 6-10, WKT.
    put16(header, 6, 0x0011);
    // GUID zeroed (8-23); version 1.4
    header[24] = 1;
    header[25] = 4;
    std::memcpy(header.data() + 26, "gpupal calibrate", 16);
    std::memcpy(header.data() + 58, "gpupal calibrate synthetic", 26);
    put16(header, 90, 1);    // creation day
    put16(header, 92, 2026); // creation year
    put16(header, 94, HeaderSize);
    put32(header, 96, HeaderSize); // offset to point data
    put32(header, 100, 0);          // number of VLRs
    header[104] = 7;                 // point data record format
    put16(header, 105, RecordSize);
    // Legacy point counts (107-130) must be zero for record formats 6-10.
    putDouble(header, 131, Scale);
    putDouble(header, 139, Scale);
    putDouble(header, 147, Scale);
    putDouble(header, 155, 0.0);
    putDouble(header, 163, 0.0);
    putDouble(header, 171, 0.0);
    // bounds filled after the sweep (179-226); waveform offsets zero
    put64(header, 235, 0);   // start of first EVLR
    put32(header, 243, 0);   // number of EVLRs
    put64(header, 247, count);
    put64(header, 255, count); // number of points by return[0]

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        throw std::runtime_error("unable to write synthetic LAS: " +
                                 path.string());
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));

    double minX = 0.0, minY = 0.0, minZ = 0.0, maxX = 0.0, maxY = 0.0,
           maxZ = 0.0;
    std::vector<unsigned char> buffer;
    constexpr std::size_t Chunk = 65536;
    buffer.resize(Chunk * RecordSize);
    std::size_t returnsByCount[15] = {};
    for (std::size_t base = 0; base < count; base += Chunk)
    {
        const std::size_t n = std::min(Chunk, count - base);
        for (std::size_t i = 0; i < n; ++i)
        {
            const SyntheticPoint p = generator.point(base + i);
            const auto quantize = [](double v)
            { return static_cast<std::int32_t>(std::llround(v / Scale)); };
            const std::int32_t qx = quantize(p.x);
            const std::int32_t qy = quantize(p.y);
            const std::int32_t qz = quantize(p.z);
            const double fx = qx * Scale, fy = qy * Scale, fz = qz * Scale;
            if (base + i == 0)
            {
                minX = maxX = fx;
                minY = maxY = fy;
                minZ = maxZ = fz;
            }
            else
            {
                minX = std::min(minX, fx); maxX = std::max(maxX, fx);
                minY = std::min(minY, fy); maxY = std::max(maxY, fy);
                minZ = std::min(minZ, fz); maxZ = std::max(maxZ, fz);
            }
            const std::size_t at = i * RecordSize;
            put32(buffer, at + 0, static_cast<std::uint32_t>(qx));
            put32(buffer, at + 4, static_cast<std::uint32_t>(qy));
            put32(buffer, at + 8, static_cast<std::uint32_t>(qz));
            put16(buffer, at + 12, p.intensity);
            buffer[at + 14] = static_cast<unsigned char>(
                (p.returnNumber & 0x0F) | ((p.numberOfReturns & 0x0F) << 4));
            buffer[at + 15] = 0; // classification flags / scanner channel / dir / edge
            buffer[at + 16] = p.classification;
            buffer[at + 17] = 0; // user data
            put16(buffer, at + 18, 0); // scan angle
            put16(buffer, at + 20, static_cast<std::uint16_t>(1U + (base + i) / std::max<std::size_t>(1, count / 4 + 1))); // point source id
            putDouble(buffer, at + 22, p.gpsTime);
            put16(buffer, at + 30, p.rgb[0]);
            put16(buffer, at + 32, p.rgb[1]);
            put16(buffer, at + 34, p.rgb[2]);
            if (p.returnNumber >= 1 && p.returnNumber <= 15)
                ++returnsByCount[p.returnNumber - 1];
        }
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(n * RecordSize));
    }
    // Patch bounds and return counts.
    putDouble(header, 179, maxX);
    putDouble(header, 187, minX);
    putDouble(header, 195, maxY);
    putDouble(header, 203, minY);
    putDouble(header, 211, maxZ);
    putDouble(header, 219, minZ);
    for (std::size_t r = 0; r < 15; ++r)
        put64(header, 255 + r * 8, returnsByCount[r]);
    out.seekp(0);
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    if (!out)
        throw std::runtime_error("unable to finish synthetic LAS: " +
                                 path.string());
    return count;
}

} // namespace pdg
