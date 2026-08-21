#pragma once

#include <pdg/Dimension.hpp>
#include <pdg/io/Las.hpp>
#include <pdg/stages/Ferry.hpp>

#include <cstddef>
#include <span>

namespace pdg::las
{

// The first exact pipeline envelope operates on the canonical format-7 output
// produced by translateDefaultInto. Unsupported layouts remain on PDAL.
[[nodiscard]] bool
supportsDefaultFerry(const FileView& input, const FerryProgram& program,
                     const DimensionRegistry& dimensions) noexcept;

void applyDefaultFerry(std::span<std::byte> canonicalOutput,
                       const FileView& input, const FerryProgram& program,
                       DimensionRegistry& dimensions,
                       std::size_t maximumHostWorkers = 0);

} // namespace pdg::las
