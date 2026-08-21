#pragma once

#include <pdg/Cuda.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslate.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#if PDG_HAS_CUDA
namespace pdg::las
{

[[nodiscard]] bool
supportsDefaultCudaTranslation(const FileView& input) noexcept;
[[nodiscard]] std::vector<std::byte> translateDefaultCuda(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    std::size_t chunkPoints = 1U << 20U, std::size_t schedulerLanes = 0);
void translateDefaultCudaInto(const FileView& input,
                              const DefaultTranslationMetadata& metadata,
                              std::span<std::byte> output,
                              std::size_t chunkPoints = 1U << 20U,
                              std::size_t schedulerLanes = 0);

[[nodiscard]] bool
supportsDefaultCudaPointProgram(const FileView& input,
                                const AssignProgram& program,
                                const DimensionRegistry& dimensions) noexcept;
// Stronger than supportsDefaultCudaPointProgram: the complete default LAS
// translation, ordered assign/ferry VM, canonical repack, and summary execute
// in one kernel. Programs outside this bounded fusion envelope retain the
// resident multi-kernel CUDA path.
[[nodiscard]] bool supportsDefaultFusedCudaPointProgram(
    const FileView& input, const AssignProgram& program,
    const DimensionRegistry& dimensions) noexcept;
[[nodiscard]] bool
supportsDefaultCudaPointProgram(const FileView& input,
                                const OrderedPointProgram& program,
                                const DimensionRegistry& dimensions) noexcept;
[[nodiscard]] bool
preferDefaultCudaPointProgram(std::uint64_t pointCount,
                              const AssignProgram& program) noexcept;
[[nodiscard]] std::vector<std::byte> translateDefaultPointProgramCuda(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    std::size_t chunkPoints = 1U << 17U, std::size_t schedulerLanes = 0);
void translateDefaultPointProgramCudaInto(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    std::span<std::byte> output, std::size_t chunkPoints = 1U << 17U,
    std::size_t schedulerLanes = 0);

// Invokes the sink synchronously with nonoverlapping regions of the final LAS
// file. Point chunks may arrive out of offset order; the 375-byte header is
// emitted last after exact bounds and return counts are known. The sink must
// consume each span before returning.
using CudaTranslationSink =
    std::function<void(std::size_t, std::span<const std::byte>)>;

struct CudaTranslationMetrics
{
    // Successful aggregate cudaMemcpy payloads, including one summary transfer
    // in each direction per scheduled tile. Host sink writes are not included.
    std::size_t hostToDeviceBytes = 0;
    std::size_t deviceToHostBytes = 0;
};

TiledSchedule translateDefaultPointProgramCudaToSink(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const AssignProgram& program, DimensionRegistry& dimensions,
    const CudaTranslationSink& sink, std::size_t chunkPoints = 1U << 17U,
    std::size_t schedulerLanes = 0, std::size_t memoryBudgetBytes = 0,
    CudaTranslationMetrics* metrics = nullptr);

// Executes an ordered assign/ferry/predicate program, returning the exact
// number of stable survivors written to the sink. The caller may allocate the
// input-sized upper bound and truncate after the 375-byte header is emitted.
// A nonzero memoryBudgetBytes bounds the scheduler exactly as the fused
// variant does. Metrics accumulate the explicit cudaMemcpy payloads issued by
// this function; survivor-dependent output transfers make them run-observed
// rather than analytic, and stable-compaction internals are not included.
[[nodiscard]] std::uint64_t translateDefaultOrderedPointProgramCudaToSink(
    const FileView& input, const DefaultTranslationMetadata& metadata,
    const OrderedPointProgram& program, DimensionRegistry& dimensions,
    const CudaTranslationSink& sink, std::size_t chunkPoints = 1U << 17U,
    std::size_t schedulerLanes = 0, std::size_t memoryBudgetBytes = 0,
    CudaTranslationMetrics* metrics = nullptr,
    TiledSchedule* scheduleOut = nullptr);

} // namespace pdg::las
#endif
