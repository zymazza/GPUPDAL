#pragma once

#include <pdg/Expression.hpp>
#include <pdg/stages/Ferry.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace pdg
{

class PointBatch;

struct PointAssignment
{
    DimensionId destination;
    CompiledExpression value;
    CompiledExpression condition;
    bool destinationCreated = false;
};

struct AssignProgram
{
    std::vector<PointAssignment> assignments;
    std::vector<DimensionId> reads;
    std::vector<DimensionId> writes;
};

[[nodiscard]] AssignProgram
compileAssignments(std::span<const std::string> specifications,
                   DimensionRegistry& dimensions);

void appendAssignments(AssignProgram& destination, const AssignProgram& source);
void appendFerry(AssignProgram& destination, const FerryProgram& source);

// Reports whether the program fits the deliberately small device subset whose
// evaluation order and operations can match the host VM exactly. Other valid
// programs remain native and execute on the host.
[[nodiscard]] bool
assignSupportsExactDevice(const AssignProgram& program) noexcept;

// Generic PDAL PointViews store XYZ as physical doubles instead of LAS
// scaled integers. This overload accepts coordinate expressions only when
// every referenced/written coordinate column has that exact representation.
[[nodiscard]] bool
assignSupportsExactDevice(const PointBatch& batch,
                          const AssignProgram& program) noexcept;

// Host and pinned-host batches complete before returning. Device batches are
// enqueued on the allocator stream. Statements and expressions retain PDAL's
// declaration/evaluation order.
void executeAssign(PointBatch& batch, const AssignProgram& program,
                   std::size_t maximumHostWorkers = 0);

} // namespace pdg
