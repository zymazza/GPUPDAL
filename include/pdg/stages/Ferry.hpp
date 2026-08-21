#pragma once

#include <pdg/Dimension.hpp>

#include <cstddef>
#include <vector>

namespace pdg
{

class PointBatch;

struct FerryCopy
{
    bool hasSource = false;
    DimensionId source;
    DimensionId destination;
    bool destinationCreated = false;
};

struct FerryProgram
{
    std::vector<FerryCopy> copies;
};

// Reports whether the generic PDAL point-program wrapper can execute the
// ferry without changing the prepared PointLayout. Blank-source ferries and
// copies that widen an existing destination remain on the untouched host
// path even though the lower-level FerryProgram is valid.
[[nodiscard]] bool
ferrySupportsExactPointProgram(const FerryProgram& program,
                               const DimensionRegistry& dimensions) noexcept;

// Host and pinned-host batches complete before returning. Device batches are
// enqueued on the batch allocator's stream and preserve program order.
void executeFerry(PointBatch& batch, const FerryProgram& program,
                  std::size_t maximumHostWorkers = 0);

} // namespace pdg
