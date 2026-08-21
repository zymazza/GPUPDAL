/******************************************************************************
 * Copyright (c) 2016, 2020 Bradley J Chambers (brad.chambers@gmail.com)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 ****************************************************************************/

#pragma once

#include <pdal/Filter.hpp>
#include <pdal/Streamable.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace pdal
{

class PDAL_EXPORT SampleFilter : public Filter, public Streamable
{
    using Voxel = std::tuple<int, int, int>;
    using Coord = std::tuple<double, double, double>;
    using CoordList = std::vector<Coord>;
    // B0267/D0267: the voxel table is only ever probed by key (never
    // iterated), so an unordered map replaces the ordered map without any
    // change to the greedy decisions; the hash mixes the three cell indices.
    struct VoxelHash
    {
        std::size_t operator()(const Voxel& v) const noexcept
        {
            const std::uint64_t x = static_cast<std::uint32_t>(std::get<0>(v));
            const std::uint64_t y = static_cast<std::uint32_t>(std::get<1>(v));
            const std::uint64_t z = static_cast<std::uint32_t>(std::get<2>(v));
            std::uint64_t h = x * 0x9E3779B185EBCA87ULL;
            h ^= (y + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL;
            h ^= (z + 0x165667B19E3779F9ULL) * 0x27D4EB2F165667C5ULL;
            h ^= h >> 31;
            return static_cast<std::size_t>(h);
        }
    };

public:
    SampleFilter() : Filter() {}
    SampleFilter& operator=(const SampleFilter&) = delete;
    SampleFilter(const SampleFilter&) = delete;

    std::string getName() const;

private:
    double m_cell;
    Arg* m_cellArg;
    double m_radius;
    double m_radiusSqr;
    Arg* m_radiusArg;
    double m_originX;
    double m_originY;
    double m_originZ;
    Arg* m_dimensionArg;
    std::string m_dimensionName;
    pdal::Dimension::Id m_dimension;
    Arg* m_originXArg;
    Arg* m_originYArg;
    Arg* m_originZArg;
    std::unordered_map<Voxel, CoordList, VoxelHash> m_populatedVoxels;

    virtual void addArgs(ProgramArgs& args);
    virtual void prepared(PointTableRef table);
    virtual bool processOne(PointRef& point);
    virtual void ready(PointTableRef);
    virtual PointViewSet run(PointViewPtr view);

    bool keepPoint(PointRef& point);

    bool voxelize(PointRef& point);
};

} // namespace pdal
