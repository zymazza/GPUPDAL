#pragma once

// D0277: the deterministic synthetic lidar-like cloud used by `gpupal
// calibrate` when the maintainer supplies no input.  It is a stand-in for real
// data: scan-ordered strips over a rolling terrain with a vegetation fraction,
// at a nominal density, so kNN structures and traversal see spatially coherent
// input with realistic ties.  The generator is header-declared here so the
// fixture writer (engine) and the index probe (core) draw the same points.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace pdg
{

struct SyntheticCloudParameters
{
    std::size_t points = 0;
    std::uint64_t seed = 0x5eedULL;
    double pointsPerSquareMetre = 15.0;
    double stripWidthMetres = 40.0;
    double vegetationFraction = 0.25;
};

struct SyntheticPoint
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    std::uint16_t intensity = 0;
    std::uint8_t returnNumber = 1;
    std::uint8_t numberOfReturns = 1;
    std::uint8_t classification = 2;
    double gpsTime = 0.0;
    std::array<std::uint16_t, 3> rgb{0, 0, 0};
};

class SyntheticCloudGenerator
{
public:
    explicit SyntheticCloudGenerator(SyntheticCloudParameters parameters);
    [[nodiscard]] const SyntheticCloudParameters& parameters() const noexcept
    {
        return m_parameters;
    }
    // Side length of the square area covered, metres.
    [[nodiscard]] double sideMetres() const noexcept { return m_side; }
    // The i-th point of the deterministic sequence, i < parameters().points.
    [[nodiscard]] SyntheticPoint point(std::size_t index) const noexcept;

private:
    SyntheticCloudParameters m_parameters;
    double m_side = 0.0;
    std::size_t m_strips = 1;
    std::size_t m_pointsPerStrip = 1;
    double m_pointsPerScan = 1.0;
};

// Writes the cloud as an uncompressed LAS 1.4, point data record format 7
// (36-byte records: XYZ, intensity, returns, classification, user data, scan
// angle, point source id, GPS time, RGB), scale 0.01 m, offsets at the
// origin, WKT-less.  Returns the number of points written.
std::size_t writeSyntheticLas(const std::filesystem::path& path,
                              const SyntheticCloudParameters& parameters);

} // namespace pdg
