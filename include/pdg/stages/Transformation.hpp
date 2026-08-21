#pragma once

#include <array>
#include <string_view>

namespace pdg
{

class PointBatch;

struct TransformationProgram
{
    std::array<double, 16> matrix{};
};

[[nodiscard]] TransformationProgram
compileTransformation(std::string_view specification);

// The first device envelope covers the ordinary affine form whose homogeneous
// denominator is one. Host execution preserves the full projective form.
[[nodiscard]] bool transformationSupportsExactDevice(
    const TransformationProgram& program) noexcept;
[[nodiscard]] bool transformationSupportsExactDevice(
    const PointBatch& batch, const TransformationProgram& program) noexcept;

void executeTransformation(PointBatch& batch,
                           const TransformationProgram& program);

} // namespace pdg
