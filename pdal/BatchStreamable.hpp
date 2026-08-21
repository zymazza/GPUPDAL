/******************************************************************************
 * Copyright (c) 2026 PDAL-GPU contributors
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
 *     * Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
 ******************************************************************************/

#pragma once

#include <pdal/Streamable.hpp>

namespace pdal
{

// Extension point for filters that retain streaming semantics but operate on
// the populated portion of a StreamPointTable as one bounded batch. The core
// streaming executor calls processBatch once per reader fill instead of
// processOne once per point. Implementations may update fields and mark
// rejected points with StreamPointTable::setSkip().
class PDAL_EXPORT BatchStreamable : public virtual Streamable
{
public:
    virtual void processBatch(StreamPointTable& table,
                              point_count_t pointLimit) = 0;

private:
    bool processOne(PointRef&) final
    {
        // The streaming executor always dispatches BatchStreamable through
        // processBatch. Returning true keeps accidental direct calls benign.
        return true;
    }
};

} // namespace pdal
