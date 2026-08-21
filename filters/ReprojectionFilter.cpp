/******************************************************************************
* Copyright (c) 2011, Michael P. Gerlek (mpg@flaxen.com)
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

#include "ReprojectionFilter.hpp"

#include <pdal/PointView.hpp>
#include <pdal/private/HostNeighborhoodWorkers.hpp>
#include <pdal/private/HostSlotPool.hpp>

#include <cpl_error.h>
#include <pdal/private/SrsTransform.hpp>
#include <pdal/util/ProgramArgs.hpp>

//#include <algorithm>
#include <memory>
#include <string>

namespace pdal
{

static StaticPluginInfo const s_info
{
    "filters.reprojection",
    "Reproject data using GDAL from one coordinate system to another.",
    "https://pdal.org/stages/filters.reprojection.html"
};

CREATE_STATIC_STAGE(ReprojectionFilter, s_info)

std::string ReprojectionFilter::getName() const { return s_info.name; }

ReprojectionFilter::ReprojectionFilter() : m_inferInputSRS(true)
{}


ReprojectionFilter::~ReprojectionFilter()
{}


void ReprojectionFilter::addArgs(ProgramArgs& args)
{
    args.add("out_srs", "Output spatial reference", m_outSRS).setPositional();
    args.add("in_srs", "Input spatial reference", m_inSRS);
    args.add("in_axis_ordering", "Axis ordering override for in_srs", m_inAxisOrderingArg, {} );
    args.add("out_axis_ordering", "Axis ordering override for out_srs", m_outAxisOrderingArg, {} );
    args.add("in_coord_epoch", "Input coordinate epoch for transformation", m_inCoordEpochArg);
    args.add("out_coord_epoch", "Output coordinate epoch for transformation", m_outCoordEpochArg);
    args.add("error_on_failure", "Throw an exception if we can't reproject any point",
        m_errorOnFailure);
}


void ReprojectionFilter::initialize()
{
    m_inferInputSRS = m_inSRS.empty();
    setSpatialReference(m_outSRS);
}


void ReprojectionFilter::spatialReferenceChanged(const SpatialReference& srs)
{
    createTransform(srs);
}

void ReprojectionFilter::prepared(PointTableRef table)
{

    // convert string args to integers and throw if
    // we can't
    auto convert = [] (const std::vector<std::string>& in)
    {
        std::vector<int> output;
        for (auto& str: in)
        {
           try
           {
                output.push_back(std::stoi(str));
           } catch (std::invalid_argument&)
           {
                throw pdal_error("Unable to convert axis ordering to integer");
           }

        }
        return output;

    };

    // Check that the sorted vector is 1,2 or 1,2,3
    auto check = [this] (const std::vector<int>& in)
    {
        auto test = in;
        std::sort(test.begin(), test.end());
        if (test.size() > 3)
            throwError("Axis ordering vector is too long");
        if (test.at(0) != 1 && test.at(1) != 2)
            throwError("Axis ordering is invalid");
        if (test.size() > 2)
            if (test.at(2) != 3)
                throwError("Axis ordering for 3rd dimension is invalid");

    };

    if (m_inAxisOrderingArg.size())
    {
        m_inAxisOrdering = convert(m_inAxisOrderingArg);
        check(m_inAxisOrdering);
    }

    if (m_outAxisOrderingArg.size())
    {
        m_outAxisOrdering = convert(m_outAxisOrderingArg);
        check(m_outAxisOrdering);
    }

}

void ReprojectionFilter::createTransform(const SpatialReference& srsSRS)
{
    if (m_inferInputSRS)
    {
        m_inSRS = srsSRS;
        if (m_inSRS.empty())
            throwError("source data has no spatial reference and "
                "none is specified with the 'in_srs' option.");
    }


    // If either vector is empty, GDAL's default ordering is used.
    if (m_inAxisOrdering.size() || m_outAxisOrdering.size())
    {

        m_transform.reset(new SrsTransform(m_inSRS,
                                           m_inAxisOrdering,
                                           m_outSRS,
                                           m_outAxisOrdering));
    } else {
        m_transform.reset(new SrsTransform(m_inSRS, m_outSRS));
    }


    if (!pdal::Utils::compare_approx(m_inCoordEpochArg, 0.0f, 0.00f))
    {
        m_transform->setSrcEpoch(m_inCoordEpochArg);
    }

    if (!pdal::Utils::compare_approx(m_outCoordEpochArg, 0.0f, 0.00f))
    {
        m_transform->setDstEpoch(m_outCoordEpochArg);
    }

}


// D0266/B0266: exact parallel reprojection. Each point's result depends only
// on its own coordinates, so fixed contiguous chunks run on a small pool of
// fixed worker slots, each slot owning a clone of the coordinate
// transformation created lazily on that slot's thread (GDAL transformations
// are not thread-safe; a clone bound to one thread produces the same
// per-point results). Kept ids are appended in slot order afterwards, which
// is the original order, and the first failing point in original order
// surfaces first when error_on_failure is set. The pool is created on first
// use and released in done(); small batches, absent transforms, `where`
// expressions, and clone failures keep the serial upstream loop.
struct ReprojectionFilter::Workers
{
    hostworkers::HostSlotPool pool;
    std::vector<std::unique_ptr<SrsTransform>> transforms;
    // The transformation the clones were taken from; a new source (SRS
    // change between streams) invalidates them.
    const SrsTransform* source = nullptr;

    explicit Workers(std::size_t slots) : pool(slots), transforms(slots)
    {}
};

namespace
{
constexpr std::size_t ReprojectionWorkerCap = 8U;
constexpr std::size_t ReprojectionMinimumRows = 2048U;

// GDAL reports a failed transformation through its (process-global) error
// handler with per-transformation-object rate limiting, and PDAL routes those
// reports to the stage log. Worker slots therefore transform into scratch
// under a quiet thread-local handler; when no row failed, the slots commit
// the results (nothing was reported, exactly as in the serial loop). When any
// row failed, the scratch is discarded and the rows are re-run serially
// through the filter's own transformation, so every diagnostic, its order,
// GDAL's per-object error accounting, dropped-point positions, and the
// error_on_failure first-failure message are pinned PDAL's. error_on_failure
// itself keeps the serial loop.
class ScopedQuietGdalErrors
{
public:
    ScopedQuietGdalErrors()
    {
        CPLPushErrorHandler(CPLQuietErrorHandler);
    }
    ~ScopedQuietGdalErrors()
    {
        CPLPopErrorHandler();
    }
    ScopedQuietGdalErrors(const ScopedQuietGdalErrors&) = delete;
    ScopedQuietGdalErrors& operator=(const ScopedQuietGdalErrors&) = delete;
};

struct ReprojectionScratch
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<unsigned char> ok;
    void resize(std::size_t rows)
    {
        x.resize(rows);
        y.resize(rows);
        z.resize(rows);
        ok.assign(rows, 0U);
    }
};
} // unnamed namespace

ReprojectionFilter::Workers& ReprojectionFilter::workers()
{
    if (!m_workers)
    {
        // Measured on the 1M reference: 8 slots reach about 4x per-point
        // throughput; PROJ/GDAL serialize beyond that.
        std::size_t slots = (std::min)(ReprojectionWorkerCap,
            hostworkers::rowWorkerCount(
                ReprojectionWorkerCap * hostworkers::MinimumRowsPerWorker));
        m_workers.reset(new Workers(slots));
    }
    if (m_workers->source != m_transform.get())
    {
        for (std::unique_ptr<SrsTransform>& transform : m_workers->transforms)
            transform.reset();
        m_workers->source = m_transform.get();
    }
    return *m_workers;
}

void ReprojectionFilter::releaseWorkers()
{
    m_workers.reset();
}

void ReprojectionFilter::done(PointTableRef)
{
    releaseWorkers();
}

PointViewSet ReprojectionFilter::run(PointViewPtr view)
{
    PointViewSet viewSet;
    PointViewPtr outView = view->makeNew();

    createTransform(view->spatialReference());

    const std::size_t rows = static_cast<std::size_t>(view->size());
    Workers* w = nullptr;
    if (rows >= ReprojectionMinimumRows && m_transform->valid() &&
        !m_errorOnFailure && hostworkers::rowWorkerCount(rows) > 1U)
    {
        w = &workers();
        if (w->pool.slots() <= 1U)
            w = nullptr;
    }
    if (!w)
    {
        PointRef point(*view, 0);
        for (PointId id = 0; id < view->size(); ++id)
        {
            point.setPointId(id);
            if (processOne(point))
                outView->appendPoint(*view, id);
        }
        viewSet.insert(outView);
        return viewSet;
    }

    const std::size_t slots = w->pool.slots();
    ReprojectionScratch scratch;
    scratch.resize(rows);
    std::vector<unsigned char> slotFailed(slots, 0U);
    w->pool.run([&](std::size_t slot)
    {
        ScopedQuietGdalErrors quiet;
        std::unique_ptr<SrsTransform>& local = w->transforms[slot];
        if (!local)
        {
            local = m_transform->clone();
            if (!local)
                throw pdal_error("reprojection: unable to clone the "
                                 "coordinate transformation for a worker");
        }
        std::size_t begin = 0U;
        std::size_t end = 0U;
        hostworkers::slotRange(rows, slots, slot, begin, end);
        PointRef point(*view, 0);
        for (std::size_t id = begin; id < end; ++id)
        {
            point.setPointId(id);
            double x(point.getFieldAs<double>(Dimension::Id::X));
            double y(point.getFieldAs<double>(Dimension::Id::Y));
            double z(point.getFieldAs<double>(Dimension::Id::Z));
            if (local->transform(x, y, z))
            {
                scratch.x[id] = x;
                scratch.y[id] = y;
                scratch.z[id] = z;
                scratch.ok[id] = 1U;
            }
            else
                slotFailed[slot] = 1U;
        }
    });
    if (std::find(slotFailed.begin(), slotFailed.end(), 1U) != slotFailed.end())
    {
        // Rare: reproduce every report and drop through the pinned loop.
        PointRef point(*view, 0);
        for (PointId id = 0; id < view->size(); ++id)
        {
            point.setPointId(id);
            if (processOne(point))
                outView->appendPoint(*view, id);
        }
        viewSet.insert(outView);
        return viewSet;
    }
    w->pool.run([&](std::size_t slot)
    {
        std::size_t begin = 0U;
        std::size_t end = 0U;
        hostworkers::slotRange(rows, slots, slot, begin, end);
        PointRef point(*view, 0);
        for (std::size_t id = begin; id < end; ++id)
        {
            point.setPointId(id);
            point.setField(Dimension::Id::X, scratch.x[id]);
            point.setField(Dimension::Id::Y, scratch.y[id]);
            point.setField(Dimension::Id::Z, scratch.z[id]);
        }
    });
    for (PointId id = 0; id < view->size(); ++id)
        outView->appendPoint(*view, id);

    viewSet.insert(outView);
    return viewSet;
}

bool ReprojectionFilter::processStreamBatch(StreamPointTable& table,
    point_count_t pointLimit, const expr::ConditionalExpression* where)
{
    if (where || m_errorOnFailure || !m_transform || !m_transform->valid() ||
        pointLimit < ReprojectionMinimumRows ||
        hostworkers::rowWorkerCount(pointLimit) <= 1U)
        return false;
    Workers& w = workers();
    const std::size_t slots = w.pool.slots();
    if (slots <= 1U)
        return false;
    ReprojectionScratch scratch;
    scratch.resize(pointLimit);
    std::vector<unsigned char> slotFailed(slots, 0U);
    w.pool.run([&](std::size_t slot)
    {
        ScopedQuietGdalErrors quiet;
        std::unique_ptr<SrsTransform>& local = w.transforms[slot];
        if (!local)
        {
            local = m_transform->clone();
            if (!local)
                throw pdal_error("reprojection: unable to clone the "
                                 "coordinate transformation for a worker");
        }
        std::size_t begin = 0U;
        std::size_t end = 0U;
        hostworkers::slotRange(pointLimit, slots, slot, begin, end);
        PointRef point(table, 0);
        for (std::size_t idx = begin; idx < end; ++idx)
        {
            if (table.skip(idx))
                continue;
            point.setPointId(idx);
            double x(point.getFieldAs<double>(Dimension::Id::X));
            double y(point.getFieldAs<double>(Dimension::Id::Y));
            double z(point.getFieldAs<double>(Dimension::Id::Z));
            if (local->transform(x, y, z))
            {
                scratch.x[idx] = x;
                scratch.y[idx] = y;
                scratch.z[idx] = z;
                scratch.ok[idx] = 1U;
            }
            else
                slotFailed[slot] = 1U;
        }
    });
    if (std::find(slotFailed.begin(), slotFailed.end(), 1U) != slotFailed.end())
    {
        // Rare: reproduce every report and skip through the pinned loop.
        PointRef point(table, 0);
        for (PointId idx = 0; idx < pointLimit; ++idx)
        {
            if (table.skip(idx))
                continue;
            point.setPointId(idx);
            if (!processOne(point))
                table.setSkip(idx);
        }
        return true;
    }
    w.pool.run([&](std::size_t slot)
    {
        std::size_t begin = 0U;
        std::size_t end = 0U;
        hostworkers::slotRange(pointLimit, slots, slot, begin, end);
        PointRef point(table, 0);
        for (std::size_t idx = begin; idx < end; ++idx)
        {
            if (!scratch.ok[idx])
                continue;
            point.setPointId(idx);
            point.setField(Dimension::Id::X, scratch.x[idx]);
            point.setField(Dimension::Id::Y, scratch.y[idx]);
            point.setField(Dimension::Id::Z, scratch.z[idx]);
        }
    });
    return true;
}


bool ReprojectionFilter::processOne(PointRef& point)
{
    return processOne(point, *m_transform);
}


bool ReprojectionFilter::processOne(PointRef& point,
    const SrsTransform& transform)
{
    double x(point.getFieldAs<double>(Dimension::Id::X));
    double y(point.getFieldAs<double>(Dimension::Id::Y));
    double z(point.getFieldAs<double>(Dimension::Id::Z));

    bool ok = transform.transform(x, y, z);
    if (ok)
    {
        point.setField(Dimension::Id::X, x);
        point.setField(Dimension::Id::Y, y);
        point.setField(Dimension::Id::Z, z);
    }
    else if (m_errorOnFailure)
        throwError("Couldn't reproject point with X/Y/Z coordinates of (" +
            std::to_string(point.getFieldAs<double>(Dimension::Id::X)) + ", " +
            std::to_string(point.getFieldAs<double>(Dimension::Id::Y)) + ", " +
            std::to_string(point.getFieldAs<double>(Dimension::Id::Z)) + ").");
    return ok;
}

} // namespace pdal
