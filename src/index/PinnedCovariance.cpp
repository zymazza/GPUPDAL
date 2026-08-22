/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the names
*       of its contributors may be used to endorse or promote products derived
*       from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
****************************************************************************/

#include "PinnedCovariance.hpp"

#include <stdexcept>

namespace pdg::detail
{

// Compatibility implementation of pdal::math::computeCovariance() from the
// upstream source revision pinned by cmake/pdg-oracle.cmake.
Eigen::Matrix3d
pinnedCovariance(const std::array<const double*, 3>& coordinates,
                 const std::uint32_t* pointIds, std::size_t count)
{
    if (!pointIds || count < 2U)
        throw std::invalid_argument(
            "pinned covariance requires at least two point IDs");

    double meanX = 0.0;
    double meanY = 0.0;
    double meanZ = 0.0;
    for (std::size_t item = 0U; item < count; ++item)
    {
        const std::uint32_t point = pointIds[item];
        const double divisor = static_cast<double>(item + 1U);
        meanX += (coordinates[0][point] - meanX) / divisor;
        meanY += (coordinates[1][point] - meanY) / divisor;
        meanZ += (coordinates[2][point] - meanZ) / divisor;
    }

    Eigen::Vector3d centroid;
    centroid << meanX, meanY, meanZ;
    Eigen::MatrixXd demeaned(3, static_cast<Eigen::Index>(count));
    for (std::size_t item = 0U; item < count; ++item)
    {
        const std::uint32_t point = pointIds[item];
        demeaned(0, static_cast<Eigen::Index>(item)) =
            static_cast<float>(coordinates[0][point] - centroid[0]);
        demeaned(1, static_cast<Eigen::Index>(item)) =
            static_cast<float>(coordinates[1][point] - centroid[1]);
        demeaned(2, static_cast<Eigen::Index>(item)) =
            static_cast<float>(coordinates[2][point] - centroid[2]);
    }

    return demeaned * demeaned.transpose() /
           static_cast<double>(count - 1U);
}

} // namespace pdg::detail
