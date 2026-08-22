#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>

namespace pdg::detail
{

Eigen::Matrix3d
pinnedCovariance(const std::array<const double*, 3>& coordinates,
                 const std::uint32_t* pointIds, std::size_t count);

} // namespace pdg::detail
