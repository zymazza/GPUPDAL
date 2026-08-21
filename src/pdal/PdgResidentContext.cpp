#include "PdgResidentContext.hpp"

#include <pdg/Cuda.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Memory.hpp>
#include <pdg/stages/Ordering.hpp>

#include <pdal/Dimension.hpp>
#include <pdal/PointLayout.hpp>
#include <pdal/PointView.hpp>

#if PDG_HAS_CUDA
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pdal::pdg_detail
{
namespace
{

thread_local ResidentExecutionContext* ActiveResidentContext = nullptr;

// Executor-declared per-point device scratch of the whole-view shared-index
// machinery on top of the planner's column/index estimate: the 96-byte cached
// eigensystem, the 48-byte transient covariance build scratch, and two
// one-byte status streams.
constexpr std::size_t NeighborhoodExecutorScratchBytesPerPoint = 146U;

#if PDG_HAS_CUDA
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
        nvtxRangePushA(name);
    }

    ~NvtxRange()
    {
        nvtxRangePop();
    }
};
#endif

std::size_t checkedProduct(std::size_t left, std::size_t right,
                           const char* message)
{
    if (left && right > (std::numeric_limits<std::size_t>::max)() / left)
        throw std::overflow_error(message);
    return left * right;
}

std::size_t checkedAdd(std::size_t left, std::size_t right, const char* message)
{
    if (left > (std::numeric_limits<std::size_t>::max)() - right)
        throw std::overflow_error(message);
    return left + right;
}

pdg::DimensionType toPdgType(Dimension::Type type)
{
    switch (type)
    {
    case Dimension::Type::Signed8:
        return pdg::DimensionType::Signed8;
    case Dimension::Type::Signed16:
        return pdg::DimensionType::Signed16;
    case Dimension::Type::Signed32:
        return pdg::DimensionType::Signed32;
    case Dimension::Type::Signed64:
        return pdg::DimensionType::Signed64;
    case Dimension::Type::Unsigned8:
        return pdg::DimensionType::Unsigned8;
    case Dimension::Type::Unsigned16:
        return pdg::DimensionType::Unsigned16;
    case Dimension::Type::Unsigned32:
        return pdg::DimensionType::Unsigned32;
    case Dimension::Type::Unsigned64:
        return pdg::DimensionType::Unsigned64;
    case Dimension::Type::Float:
        return pdg::DimensionType::Float;
    case Dimension::Type::Double:
        return pdg::DimensionType::Double;
    case Dimension::Type::None:
        return pdg::DimensionType::None;
    }
    return pdg::DimensionType::None;
}

void addPhaseSeconds(double* accumulator,
                     std::chrono::steady_clock::time_point started)
{
    if (!accumulator)
        return;
    *accumulator += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count();
}

double phaseSecondsBetween(std::chrono::steady_clock::time_point started,
                           std::chrono::steady_clock::time_point ended) noexcept
{
    return std::chrono::duration<double>(ended - started).count();
}

bool sameLayout(const DimTypeList& left, const DimTypeList& right) noexcept
{
    if (left.size() != right.size())
        return false;
    for (std::size_t index = 0; index < left.size(); ++index)
        if (left[index].m_id != right[index].m_id ||
            left[index].m_type != right[index].m_type)
            return false;
    return true;
}

} // unnamed namespace

class ResidentExecutionContext::Impl
{
    struct ManagerTimingState
    {
        std::chrono::steady_clock::time_point managerStarted{};
        std::chrono::steady_clock::time_point executeStarted{};
        std::chrono::steady_clock::time_point residentStarted{};
        std::chrono::steady_clock::time_point spillEnded{};
        double readerRowTableMaterialization = 0.0;
        double residentWrapperIndexFilter = 0.0;
        ResidentManagerDetailSeconds detail;
        bool uploadSeen = false;
        bool spillSeen = false;
        bool invalid = false;
    };

public:
    Impl(const pdg::Plan& plan, const pdg::DimensionRegistry& planDimensions,
         std::size_t deviceMemoryBudgetBytes, std::size_t tilePointCapacity)
        : m_plan(&plan), m_planDimensions(&planDimensions),
          m_deviceMemoryBudgetBytes(deviceMemoryBudgetBytes),
          m_tilePointCapacity(tilePointCapacity)
    {
        if (!m_deviceMemoryBudgetBytes)
            throw std::invalid_argument(
                "resident execution requires a nonzero VRAM budget");
        if (!m_tilePointCapacity)
            throw std::invalid_argument(
                "resident execution tile capacity is zero");
    }

    ~Impl() = default;

    void preflight(PointLayout& layout, std::size_t pointCount,
                   std::span<const std::size_t> selectedRegions)
    {
#if PDG_HAS_CUDA
        if (m_state != State::Empty || m_view || m_layoutBound)
            throw std::logic_error(
                "resident execution preflight may run only once");
        if (selectedRegions.empty())
            throw std::invalid_argument(
                "resident execution preflight selected no regions");
        bindLayout(layout, pointCount);
        m_selectedRegions.assign(selectedRegions.begin(),
                                 selectedRegions.end());
        std::sort(m_selectedRegions.begin(), m_selectedRegions.end());
        if (std::adjacent_find(m_selectedRegions.begin(),
                               m_selectedRegions.end()) !=
            m_selectedRegions.end())
            throw std::invalid_argument(
                "resident execution preflight selected a duplicate region");
        std::optional<std::size_t> cardinalityChangingRegion;
        bool delegatedPlan = false;
        std::size_t delegatedRegions = 0;
        std::size_t delegatedScratchBytesPerPoint = 0;
        std::size_t delegatedFixedScratchBytes = 0;
        std::size_t delegatedHostBytesPerPoint = 0;
        for (std::size_t region : m_selectedRegions)
        {
            if (region >= m_plan->summary().residentRegions)
                throw std::invalid_argument(
                    "resident execution preflight selected an unknown region");
            const std::vector<const pdg::PlannedStage*> stages =
                regionStages(region);
            if (stages.empty())
                throw std::invalid_argument(
                    "resident execution preflight selected an empty region");

            pdg::HostMemoryResource memory;
            pdg::PointBatch probe(
                1U, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                m_dimensions, memory);
            probe.setSize(1U);
            std::size_t declaredCardinalityChanges = 0U;
            bool regionDelegated = false;
            for (const pdg::PlannedStage* stage : stages)
            {
                // The admission contract is descriptor-declared: purity,
                // determinism, order, cardinality, and where semantics come
                // from the stage contract, never from stage names.
                if (!stage->native ||
                    stage->preferredResidency != pdg::MemoryKind::Device ||
                    !stage->descriptor.fusion.pure ||
                    !stage->descriptor.fusion.deterministicSafe ||
                    stage->descriptor.mutatesCoordinates ||
                    stage->descriptor.fusion.hasWhere ||
                    stage->descriptor.fusion.whereMerge !=
                        pdg::WhereMergeMode::NotApplicable)
                    throw std::invalid_argument(
                        "resident execution preflight rejected stage " +
                        std::to_string(stage->id));
                if (stage->descriptor.fusion.cardinalityPreserving ==
                    std::holds_alternative<pdg::PredicateProgram>(
                        stage->payload))
                    throw std::invalid_argument(
                        "resident execution preflight rejected stage " +
                        std::to_string(stage->id));
                if (!stage->descriptor.fusion.cardinalityPreserving)
                {
                    if (!stage->descriptor.preservesOrder ||
                        ++declaredCardinalityChanges > 1U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared cardinality change of stage " +
                            std::to_string(stage->id));
                }
                pdg::preparePlannedDeviceColumns(probe, *stage);
                if (const auto* assign =
                        std::get_if<pdg::AssignProgram>(&stage->payload))
                {
                    if (!pdg::assignSupportsExactDevice(probe, *assign))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "assignment envelope");
                }
                else if (const auto* ferry =
                             std::get_if<pdg::FerryProgram>(&stage->payload))
                {
                    if (!pdg::ferrySupportsExactPointProgram(*ferry,
                                                             m_dimensions))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "ferry envelope");
                    pdg::AssignProgram program;
                    pdg::appendFerry(program, *ferry);
                    if (!pdg::assignSupportsExactDevice(probe, program))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "ferry assignment envelope");
                }
                else if (const auto* predicate =
                             std::get_if<pdg::PredicateProgram>(
                                 &stage->payload))
                {
                    if (!pdg::predicateSupportsExactDevice(probe, *predicate))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "predicate envelope");
                }
                else if (const auto* labelDuplicates =
                             std::get_if<pdg::LabelDuplicatesProgram>(
                                 &stage->payload))
                {
                    if (stage->descriptor.index.kind != pdg::IndexKind::None ||
                        !pdg::labelDuplicatesMaySupportExactDevice(
                            probe, *labelDuplicates))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "label_duplicates envelope");
                    // The Duplicate output is a planner-declared resident
                    // column, so this whole-view adjacent pass requires no
                    // additional per-point scratch and no private index.
                    regionDelegated = true;
                }
                else if (const auto* smrf =
                             std::get_if<pdg::SmrfProgram>(&stage->payload))
                {
                    const double objectRadius =
                        std::ceil(smrf->window / smrf->cell);
                    const double netRadius =
                        smrf->cut > 0.0
                            ? 2.0 * std::ceil(smrf->cut / smrf->cell)
                            : 0.0;
                    if (stage->descriptor.kind != pdg::StageKind::Grid ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        !std::isfinite(smrf->cell) || smrf->cell <= 0.0 ||
                        !std::isfinite(smrf->slope) ||
                        !std::isfinite(smrf->scalar) ||
                        !std::isfinite(smrf->threshold) ||
                        !std::isfinite(smrf->window) || smrf->window < 0.0 ||
                        !std::isfinite(smrf->cut) || smrf->cut < 0.0 ||
                        objectRadius >
                            pdg::SmrfExactDeviceMaximumMorphologyRadius ||
                        netRadius >
                            pdg::SmrfExactDeviceMaximumMorphologyRadius ||
                        (!smrf->onlyGround &&
                         smrf->groundClass == smrf->otherClass))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "smrf grid envelope");
                    regionDelegated = true;
                    delegatedFixedScratchBytes =
                        (std::max)(delegatedFixedScratchBytes,
                                   pdg::
                                       SmrfExactDeviceMaximumFixedScratchBytes);
                }
                else if (const auto* pmf =
                             std::get_if<pdg::PmfProgram>(&stage->payload))
                {
                    const pdg::GridRequest& grid = stage->descriptor.grid;
                    if (stage->descriptor.kind != pdg::StageKind::Grid ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        !pdg::pmfProgramWithinExactDeviceEnvelope(*pmf) ||
                        grid.framePolicy !=
                            pdg::GridFramePolicy::PmfInitialLookupV1 ||
                        grid.cellSize != pmf->cellSize ||
                        grid.deviceBytesPerCell !=
                            pdg::PmfTiledDeviceBytesPerCell ||
                        grid.deviceBackingCount != 2U ||
                        grid.deviceProofBytesPerCell !=
                            pdg::PmfTiledDeviceProofBytesPerCell ||
                        grid.deviceFixedBytes !=
                            pdg::PmfTiledDeviceFixedScratchBytes ||
                        grid.hostBytesPerPoint !=
                            pdg::PmfTiledHostStagingBytesPerPoint ||
                        grid.hostBytesPerCell !=
                            pdg::PmfTiledHostBytesPerCell ||
                        grid.hostTileBytesPerExpandedCell != sizeof(double) ||
                        grid.maximumHaloCells != 1U || !grid.phaseSynchronized)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "pmf grid envelope");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   pdg::PmfExactDeviceScratchBytesPerPoint);
                    delegatedHostBytesPerPoint =
                        (std::max)(delegatedHostBytesPerPoint,
                                   grid.hostBytesPerPoint);
                }
                else if (const auto* csf =
                             std::get_if<pdg::CsfProgram>(&stage->payload))
                {
                    if (stage->descriptor.kind != pdg::StageKind::Grid ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        !pdg::csfProgramWithinExactDeviceEnvelope(*csf))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "csf grid envelope");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   pdg::CsfExactDeviceScratchBytesPerPoint);
                    delegatedFixedScratchBytes =
                        (std::max)(delegatedFixedScratchBytes,
                                   pdg::CsfExactDeviceMaximumFixedScratchBytes);
                }
                else if (const auto* elm =
                             std::get_if<pdg::ElmProgram>(&stage->payload))
                {
                    if (stage->descriptor.kind != pdg::StageKind::Grid ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        !pdg::elmProgramWithinExactDeviceEnvelope(*elm))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "elm grid envelope");
                    regionDelegated = true;
                    delegatedFixedScratchBytes =
                        (std::max)(delegatedFixedScratchBytes,
                                   pdg::elmExactDeviceScratchBytes(
                                       m_pointCount));
                }
                else if (const auto* skewness =
                             std::get_if<pdg::SkewnessProgram>(&stage->payload))
                {
                    // The bounded whole-view lane owns the measured exact
                    // one-key ordering reservation: planned Z plus caller and
                    // alternate permutations, two radix-key buffers, CUB
                    // temporary storage, and the duplicate flag. The pinned
                    // recurrence then owns sorted Z, Classification, and
                    // source-order host columns after device scratch releases.
                    if (stage->descriptor.kind != pdg::StageKind::Global ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        stage->descriptor.preservesOrder ||
                        !stage->descriptor.fusion.cardinalityPreserving ||
                        !pdg::skewnessProgramValid(*skewness))
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "skewness ordering envelope");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   pdg::
                                       SkewnessExactDeviceScratchBytesPerPoint);
                    delegatedHostBytesPerPoint =
                        (std::max)(delegatedHostBytesPerPoint,
                                   sizeof(double) + sizeof(std::uint8_t) +
                                       sizeof(std::uint64_t));
                }
                else if (const auto* ordering =
                             std::get_if<pdg::OrderingProgram>(&stage->payload))
                {
                    const pdg::DimensionId z(pdg::StandardDimension::Z);
                    if (stage->descriptor.kind != pdg::StageKind::Global ||
                        stage->descriptor.index.kind != pdg::IndexKind::None ||
                        stage->descriptor.preservesOrder ||
                        !stage->descriptor.fusion.cardinalityPreserving ||
                        ordering->dimensions !=
                            std::vector<pdg::DimensionId>{z} ||
                        ordering->direction !=
                            pdg::OrderingDirection::Ascending ||
                        ordering->algorithm != pdg::OrderingAlgorithm::Normal)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the exact "
                            "ordering envelope");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   pdg::
                                       OrderingExactDeviceScratchBytesPerPoint);
                    delegatedHostBytesPerPoint =
                        (std::max)(delegatedHostBytesPerPoint,
                                   sizeof(double) + sizeof(std::uint64_t));
                }
                else if (std::holds_alternative<
                             pdg::ApproximateCoplanarProgram>(stage->payload))
                {
                    // A whole-view shared-index payload; its region-level
                    // envelope is verified below.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 3U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   NeighborhoodExecutorScratchBytesPerPoint);
                }
                else if (std::holds_alternative<pdg::LofProgram>(
                             stage->payload))
                {
                    // A whole-view shared-kNN payload retaining its full
                    // adjacency: per point, `neighbors` id/squared-distance
                    // pairs plus the three status streams.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 2U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   static_cast<std::size_t>(
                                       stage->descriptor.index.neighbors) *
                                           (sizeof(std::uint32_t) +
                                            sizeof(double)) +
                                       3U);
                }
                else if (std::holds_alternative<pdg::NormalProgram>(
                             stage->payload) ||
                         std::holds_alternative<pdg::EigenvaluesProgram>(
                             stage->payload) ||
                         std::holds_alternative<pdg::CovarianceFeaturesProgram>(
                             stage->payload) ||
                         std::holds_alternative<pdg::EstimateRankProgram>(
                             stage->payload))
                {
                    // The eigen family shares the coplanar executor's cached
                    // eigensystem, covariance scratch, and status streams.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 3U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   NeighborhoodExecutorScratchBytesPerPoint);
                }
                else if (std::holds_alternative<
                             pdg::OptimalNeighborhoodProgram>(stage->payload))
                {
                    // The whole-view sweep retains its full adjacency: per
                    // point, max_k id/squared-distance pairs, the two output
                    // columns, and a status byte.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 1U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   static_cast<std::size_t>(
                                       stage->descriptor.index.neighbors) *
                                           (sizeof(std::uint32_t) +
                                            sizeof(double)) +
                                       sizeof(std::uint64_t) + sizeof(double) +
                                       1U);
                }
                else if (std::holds_alternative<pdg::NeighborClassifierProgram>(
                             stage->payload))
                {
                    // One staged result byte plus a status byte on top of
                    // the planner's Classification column estimate.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 1U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   static_cast<std::size_t>(
                                       stage->descriptor.index.neighbors) *
                                           (sizeof(std::uint32_t) +
                                            sizeof(double)) +
                                       2U);
                }
                else if (std::holds_alternative<pdg::RadialDensityProgram>(
                             stage->payload))
                {
                    // The shared radius index writes the declared binary64
                    // density column directly. No private query product is
                    // retained beyond that planner-owned output column.
                    if (stage->descriptor.index.kind !=
                            pdg::IndexKind::Radius ||
                        stage->descriptor.index.dimensions != 3U ||
                        !std::isfinite(stage->descriptor.index.radius) ||
                        stage->descriptor.index.radius <= 0.0)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "radial-density radius index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   sizeof(double));
                }
                else if (const auto* radiusAssign =
                             std::get_if<pdg::RadiusAssignProgram>(
                                 &stage->payload))
                {
                    // Source, reference, and match masks are the only device
                    // scratch beyond the planner-declared resident columns
                    // and shared radius index. Ordered assignment evaluation
                    // is the exact host finale and written columns are
                    // uploaded before a downstream resident bridge runs.
                    if (stage->descriptor.index.kind !=
                            pdg::IndexKind::Radius ||
                        !std::isfinite(stage->descriptor.index.radius) ||
                        stage->descriptor.index.radius <= 0.0 ||
                        radiusAssign->updates.writes.empty())
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared radius index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   std::size_t{3U});
                }
                else if (std::holds_alternative<pdg::NnDistanceProgram>(
                             stage->payload))
                {
                    // One double distance column plus a one-byte status
                    // stream.
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors < 2U ||
                        stage->descriptor.index.neighbors > 64U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared kNN index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   sizeof(double) + 1U);
                }
                else if (const auto* outlier =
                             std::get_if<pdg::OutlierProgram>(&stage->payload))
                {
                    // Statistical outlier retains its exact per-row mean and
                    // status. Radius outlier retains one source-ordered count.
                    // Both classification finales remain in pinned upstream
                    // host order.
                    const bool statistical =
                        outlier->method == pdg::OutlierMethod::Statistical &&
                        outlier->meanNeighbors >= 0 &&
                        outlier->meanNeighbors < 64 &&
                        stage->descriptor.index.kind == pdg::IndexKind::Knn &&
                        stage->descriptor.index.neighbors ==
                            static_cast<std::uint32_t>(outlier->meanNeighbors) +
                                1U &&
                        (m_pointCount == 0U ||
                         m_pointCount >= stage->descriptor.index.neighbors);
                    const bool radius =
                        outlier->method == pdg::OutlierMethod::Radius &&
                        std::isfinite(outlier->radius) &&
                        outlier->radius > 0.0 &&
                        stage->descriptor.index.kind ==
                            pdg::IndexKind::Radius &&
                        stage->descriptor.index.radius == outlier->radius &&
                        stage->descriptor.index.dimensions == 3U;
                    if (!statistical && !radius)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "outlier shared-index envelope");
                    regionDelegated = true;
                    if (statistical)
                        delegatedScratchBytesPerPoint =
                            (std::max)(delegatedScratchBytesPerPoint,
                                       sizeof(double) + 1U);
                }
                else if (std::holds_alternative<pdg::HagNnProgram>(
                             stage->payload))
                {
                    // Ground/non-ground masks, up to 64 id/distance
                    // entries, and one exactness status are stage-local query
                    // scratch.
                    const auto& program =
                        std::get<pdg::HagNnProgram>(stage->payload);
                    if (stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        program.count < 1U || program.count > 64U ||
                        stage->descriptor.index.neighbors != program.count ||
                        stage->descriptor.index.dimensions != 2U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "declared HAG nearest-neighbor index request");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   3U * sizeof(std::uint8_t) +
                                       program.count * (sizeof(std::uint32_t) +
                                                        sizeof(double)));
                }
                else if (std::holds_alternative<pdg::HagDelaunayProgram>(
                             stage->payload))
                {
                    const auto& program =
                        std::get<pdg::HagDelaunayProgram>(stage->payload);
                    if (program.count != 3U ||
                        stage->descriptor.index.kind != pdg::IndexKind::Knn ||
                        stage->descriptor.index.neighbors != program.count ||
                        stage->descriptor.index.dimensions != 2U)
                        throw std::invalid_argument(
                            "resident execution preflight rejected the "
                            "HAG Delaunay index contract");
                    regionDelegated = true;
                    delegatedScratchBytesPerPoint =
                        (std::max)(delegatedScratchBytesPerPoint,
                                   3U * sizeof(std::uint8_t) +
                                       program.count * (sizeof(std::uint32_t) +
                                                        sizeof(double)));
                }
                else
                    throw std::invalid_argument(
                        "resident execution preflight found no supported "
                        "point-program payload");
                pdg::releasePlannedDeviceColumns(probe, *stage);
            }
            if (declaredCardinalityChanges)
                cardinalityChangingRegion = region;
            if (regionDelegated)
            {
                delegatedPlan = true;
                ++delegatedRegions;
            }
            pdg::releaseSpilledDeviceColumns(probe, spillBoundary(region));
        }
        // Later tiles are addressed by original-view offsets, so a shrunken
        // view cannot feed another resident region afterwards.
        if (cardinalityChangingRegion &&
            *cardinalityChangingRegion != m_selectedRegions.back())
            throw std::invalid_argument(
                "resident cardinality change must terminate the selected "
                "resident chain");

        if (delegatedPlan)
        {
            // Whole-view shared-index execution: no context tiles. The
            // planner estimate carries columns, persistent index bytes, and
            // any explicitly declared max-k query product; stage-local
            // eigensystems, projections, and statuses are the conservative
            // executor scratch on top of it. Regions run sequentially through
            // their planned
            // upload/spill boundaries, so a plan mixing delegated and tiled
            // point-program regions stays fail-closed.
            if (delegatedRegions != m_selectedRegions.size())
                throw std::invalid_argument(
                    "resident shared-index execution cannot mix delegated "
                    "and tiled point-program regions");
            if (cardinalityChangingRegion)
                throw std::invalid_argument(
                    "resident shared-index execution cannot change "
                    "cardinality");
            const std::size_t scratchBytes = checkedProduct(
                m_pointCount, delegatedScratchBytesPerPoint,
                "resident shared-index scratch estimate overflows");
            const std::size_t wholeViewBytes = checkedAdd(
                checkedAdd(m_plan->estimatedDeviceBytes(m_pointCount),
                           scratchBytes,
                           "resident delegated working set overflows"),
                delegatedFixedScratchBytes,
                "resident shared-index working set overflows");
            if (wholeViewBytes > m_deviceMemoryBudgetBytes)
                throw std::invalid_argument(
                    "resident shared-index working set exceeds the planner "
                    "VRAM budget");
            const std::size_t wholeViewHostBytes = checkedProduct(
                m_pointCount, delegatedHostBytesPerPoint,
                "resident delegated pinned-host estimate overflows");
            if (wholeViewHostBytes > m_deviceMemoryBudgetBytes)
                throw std::invalid_argument(
                    "resident delegated staging exceeds the conservative "
                    "pinned-host budget");
            m_delegatedPlan = true;
            m_delegatedBaseDeviceBytes = wholeViewBytes;
            m_delegatedBaseHostBytes = wholeViewHostBytes;
            m_delegatedStagingMemory = pdg::makeCudaPinnedMemoryResource();
            m_delegatedDeviceMemory = pdg::makeCudaMemoryResource();
            m_schedule = {};
            m_schedule.itemCount = m_pointCount;
            m_schedule.tileItemCapacity = m_pointCount;
            m_schedule.tileCount = m_pointCount ? 1U : 0U;
            m_schedule.configuredLaneCount = 1U;
            m_schedule.activeLaneCount = m_pointCount ? 1U : 0U;
            m_schedule.peakLaneBytes = wholeViewBytes;
            return;
        }

        // Tiled execution requires at least one input row. Whole-view
        // delegated regions above are different: on an empty input their
        // wrappers still have to execute the planned upload/region/spill
        // lifecycle as an exact no-op, but no lane or allocation is needed.
        if (!m_pointCount)
            throw std::invalid_argument(
                "resident execution requires a nonempty point set");

        initializeLanes();
        for (const std::unique_ptr<Lane>& lane : m_lanes)
        {
            for (std::size_t region : m_selectedRegions)
                lane->probeAllocations(regionStages(region),
                                       spillBoundary(region), m_bytesPerLane);
            lane->resetPeakDeviceBytes();
        }
#else
        (void)layout;
        (void)pointCount;
        (void)selectedRegions;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    void enterBoundary(PointView& view, std::size_t boundaryId,
                       ResidentBoundaryDirection direction,
                       std::size_t residentRegion, bool requiresFullPointRecord)
    {
        if (boundaryId >= m_plan->summary().residencyBoundaries.size())
            throw std::out_of_range(
                "resident boundary identifier is outside the plan");
        const pdg::ResidencyBoundary& boundary =
            m_plan->summary().residencyBoundaries[boundaryId];
        const pdg::ResidencyBoundaryKind expectedKind =
            direction == ResidentBoundaryDirection::Upload
                ? pdg::ResidencyBoundaryKind::Upload
                : pdg::ResidencyBoundaryKind::Spill;
        if (boundary.kind != expectedKind ||
            boundary.requiresFullPointRecord != requiresFullPointRecord)
            throw std::invalid_argument(
                "resident boundary marker differs from the planned boundary");
        const std::size_t deviceStage =
            direction == ResidentBoundaryDirection::Upload ? boundary.consumer
                                                           : boundary.producer;
        if (deviceStage >= m_plan->stages().size() ||
            m_plan->stages()[deviceStage].residentRegion != residentRegion)
            throw std::invalid_argument(
                "resident boundary marker has the wrong execution region");

        if (direction == ResidentBoundaryDirection::Upload)
            markManagerUpload();

        ensureView(view);
        if (direction == ResidentBoundaryDirection::Upload)
        {
            if (m_state != State::Empty && m_state != State::HostMaterialized)
                throw std::logic_error(
                    "resident upload boundary is out of execution order");
            if (m_cardinalityChanged)
                throw std::logic_error(
                    "resident cardinality change must terminate resident "
                    "execution");
            // A delegated shared-index region transfers its planned logical
            // live columns rather than complete physical rows.
            const std::size_t packedBytes = checkedProduct(
                m_pointCount,
                m_delegatedPlan ? boundary.bytesPerPoint : m_pointStride,
                "resident full-record transfer byte count overflows");
            m_activeRegion = residentRegion;
            m_state = State::AwaitingRegion;
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::BoundaryUpload, boundaryId,
                packedBytes);
            if (boundary.fallback)
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::FallbackBoundary, boundaryId,
                    packedBytes);
            return;
        }

        if (m_state != State::RegionComplete ||
            residentRegion != m_activeRegion)
            throw std::logic_error(
                "resident spill boundary is out of execution order");
        // A declared cardinality change still transfers the complete input
        // tile plus its one-byte keep mask, so the spill accounting stays
        // input-cardinality-based and exactly predictable.
        const std::size_t spillPoints = m_completedRegionHadPredicate
                                            ? m_completedRegionInputCount
                                            : m_pointCount;
        std::size_t packedBytes = checkedProduct(
            spillPoints,
            m_delegatedPlan ? boundary.bytesPerPoint : m_pointStride,
            "resident full-record transfer byte count overflows");
        if (m_completedRegionHadPredicate)
            packedBytes =
                checkedAdd(packedBytes, spillPoints,
                           "resident keep-mask transfer byte count overflows");
        for (const std::unique_ptr<Lane>& lane : m_lanes)
            lane->releaseBoundary(boundary);
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::BoundarySpill, boundaryId, packedBytes);
        if (boundary.fallback)
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::FallbackBoundary, boundaryId,
                packedBytes);
        // A Grid product belongs to exactly one delegated region. Compatible
        // PMF stages may reuse its allocation inside that region; their
        // kernels have synchronized before endDelegatedRegion(), so the
        // matching spill is the event-governed release point. A later region
        // must build and validate its own frame.
        m_rasterGridProduct.reset();
        m_state = State::HostMaterialized;
        // Every device result is now published to host rows, so the context
        // is view-agnostic until the next upload boundary rebinds. A host
        // stage between regions may legitimately republish the same points
        // through a new PointView (upstream filters often do); identity
        // stays enforced within each upload/region/spill span.
        m_view = nullptr;
        markManagerSpill();
    }

    void beginDelegatedRegion(PointView& view, std::size_t residentRegion)
    {
        ensureView(view);
        if (!m_delegatedPlan)
            throw std::logic_error(
                "resident delegated region was not accepted by preflight");
        if (m_state == State::RegionActive && residentRegion == m_activeRegion)
            return;
        if (m_state != State::AwaitingRegion ||
            residentRegion != m_activeRegion)
            throw std::logic_error(
                "resident delegated region has no matching upload boundary");
        if (!std::binary_search(m_selectedRegions.begin(),
                                m_selectedRegions.end(), residentRegion))
            throw std::invalid_argument(
                "resident delegated region was not accepted by preflight");
        m_state = State::RegionActive;
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionBegin, residentRegion);
    }

    void endDelegatedRegion(PointView& view, std::size_t residentRegion)
    {
        ensureView(view);
        if (!m_delegatedPlan || m_state != State::RegionActive ||
            residentRegion != m_activeRegion)
            throw std::logic_error(
                "resident delegated region ended out of execution order");
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionEnd, residentRegion);
        m_state = State::RegionComplete;
    }

    pdg::MemoryResource& delegatedStagingMemory(PointView& view,
                                                std::size_t residentRegion)
    {
        requireActiveDelegatedRegion(view, residentRegion);
        return *m_delegatedStagingMemory;
    }

    pdg::MemoryResource& delegatedDeviceMemory(PointView& view,
                                               std::size_t residentRegion)
    {
        requireActiveDelegatedRegion(view, residentRegion);
        return *m_delegatedDeviceMemory;
    }

    pdg::RasterGridProduct&
    acquireRasterGridProduct(PointView& view, std::size_t residentRegion,
                             const pdg::RasterGridFrame& frame,
                             bool reuseExpected)
    {
        requireActiveDelegatedRegion(view, residentRegion);
        const std::vector<const pdg::PlannedStage*> stages =
            regionStages(residentRegion);
        if (stages.empty())
            throw std::logic_error(
                "resident raster grid requires a selected stage");
        const pdg::GridRequest& request = stages.front()->descriptor.grid;
        for (const pdg::PlannedStage* stage : stages)
        {
            const pdg::GridRequest& stageRequest = stage->descriptor.grid;
            if (!std::holds_alternative<pdg::PmfProgram>(stage->payload) ||
                stageRequest.framePolicy !=
                    pdg::GridFramePolicy::PmfInitialLookupV1 ||
                frame.policy != pdg::RasterGridFramePolicy::PmfV1 ||
                stageRequest.cellSize != frame.cellSize ||
                stageRequest.deviceBytesPerCell !=
                    pdg::PmfTiledDeviceBytesPerCell ||
                stageRequest.deviceBackingCount != 2U ||
                stageRequest.deviceProofBytesPerCell !=
                    pdg::PmfTiledDeviceProofBytesPerCell ||
                stageRequest.deviceFixedBytes !=
                    pdg::PmfTiledDeviceFixedScratchBytes ||
                stageRequest.hostBytesPerPoint !=
                    pdg::PmfTiledHostStagingBytesPerPoint ||
                stageRequest.hostBytesPerCell !=
                    pdg::PmfTiledHostBytesPerCell ||
                stageRequest.hostTileBytesPerExpandedCell != sizeof(double) ||
                stageRequest.maximumHaloCells != 1U ||
                !stageRequest.phaseSynchronized)
                throw std::invalid_argument(
                    "resident raster grid differs from the planned product");
        }
        if (m_rasterGridProduct)
        {
            if (!reuseExpected)
                throw std::logic_error(
                    "resident raster grid reuse marker was not declared");
            const pdg::RasterGridFrame& existing = m_rasterGridProduct->frame();
            if (existing.minimumX != frame.minimumX ||
                existing.minimumY != frame.minimumY ||
                existing.rows != frame.rows ||
                existing.columns != frame.columns ||
                existing.cellSize != frame.cellSize ||
                existing.policy != frame.policy)
                throw std::logic_error(
                    "resident raster grid product changed within its region");
            return *m_rasterGridProduct;
        }
        if (reuseExpected)
            throw std::logic_error(
                "resident raster grid reuse has no prior product");
        m_rasterGridProduct = std::make_unique<pdg::RasterGridProduct>(
            frame,
            pdg::RasterGridProductConfig{
                .haloCells = request.maximumHaloCells,
                .deviceBytesPerExpandedCell = request.deviceBytesPerCell,
                .deviceBackingCount = request.deviceBackingCount,
                .deviceProofBytesPerCell = request.deviceProofBytesPerCell,
                .hostBytesPerCell = request.hostBytesPerCell,
                .hostTileBytesPerExpandedCell =
                    request.hostTileBytesPerExpandedCell,
                .hostBackingCount = 2U,
                .baseDeviceBytes = m_delegatedBaseDeviceBytes,
                .baseHostBytes = m_delegatedBaseHostBytes,
                .deviceMemoryBudgetBytes = m_deviceMemoryBudgetBytes,
                .hostMemoryBudgetBytes = m_deviceMemoryBudgetBytes},
            *m_delegatedStagingMemory, *m_delegatedDeviceMemory);
        m_schedule = m_rasterGridProduct->schedule();
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::GridBuild, residentRegion,
            m_rasterGridProduct->schedule().peakLaneBytes);
        return *m_rasterGridProduct;
    }

    void beginRegion(PointView& view, std::size_t residentRegion,
                     std::span<const pdg::PackedPointColumn> columns,
                     PointView* compactionOutput)
    {
#if PDG_HAS_CUDA
        ensureView(view);
        if (m_delegatedPlan)
            throw std::logic_error(
                "resident delegated plan has no tile regions");
        if (m_state != State::AwaitingRegion ||
            residentRegion != m_activeRegion)
            throw std::logic_error(
                "resident device region has no matching upload boundary");
        if (regionDeclaresCardinalityChange(residentRegion) !=
            (compactionOutput != nullptr))
            throw std::invalid_argument(
                "resident region compaction output does not match the "
                "declared cardinality contract");
        if (compactionOutput &&
            (compactionOutput == &view || compactionOutput->size() != 0U))
            throw std::invalid_argument(
                "resident cardinality change requires a fresh empty output "
                "view");
        if (!m_selectedRegions.empty() &&
            !std::binary_search(m_selectedRegions.begin(),
                                m_selectedRegions.end(), residentRegion))
            throw std::invalid_argument(
                "resident device region was not accepted by preflight");
        if (columns.empty())
            throw std::invalid_argument(
                "resident point-program region has no device columns");
        const std::vector<const pdg::PlannedStage*> stages =
            regionStages(residentRegion);
        if (stages.empty())
            throw std::invalid_argument(
                "resident execution region contains no planned stages");
        std::vector<pdg::PackedPointColumn> plannedColumns(columns.begin(),
                                                           columns.end());
        for (const pdg::PlannedStage* stage : stages)
            for (pdg::DimensionId id : stage->deviceMaterialize)
            {
                const auto existing =
                    std::find_if(plannedColumns.begin(), plannedColumns.end(),
                                 [&](const pdg::PackedPointColumn& candidate)
                                 { return candidate.id == id; });
                if (existing != plannedColumns.end())
                    continue;
                const pdg::DimensionDefinition& definition =
                    m_dimensions.require(id);
                const Dimension::Id pdalId =
                    view.layout()->findDim(definition.name);
                if (pdalId == Dimension::Id::Unknown)
                    throw std::invalid_argument(
                        "planned resident dimension is absent from the "
                        "PointView");
                plannedColumns.push_back(
                    {id, toPdgType(view.layout()->dimType(pdalId)), 0U, false});
            }
        m_activeColumns = normalizeColumns(view, plannedColumns);
        initializeLanes();
        m_activeStages = stages;
        m_activeOutput = compactionOutput;
        m_activeSpillBoundary = &spillBoundary(residentRegion);
        m_nextTile = 0U;
        m_submittedTiles = 0U;
        m_hostToDeviceBytes = 0U;
        m_deviceToHostBytes = 0U;
        m_hostToDevicePackingBytes = 0U;
        m_deviceToHostPackingBytes = 0U;
        m_state = State::RegionActive;
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionBegin, residentRegion);
#else
        (void)view;
        (void)residentRegion;
        (void)columns;
        (void)compactionOutput;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    [[nodiscard]] std::size_t tileCount() const noexcept
    {
        return m_schedule.tileCount;
    }

    pdg::PointBatch& acquireTile(PointView& view, std::size_t tileIndex)
    {
#if PDG_HAS_CUDA
        ensureView(view);
        if (m_state != State::RegionActive || tileIndex != m_nextTile ||
            tileIndex >= m_schedule.tileCount)
            throw std::logic_error(
                "resident tile acquisition is out of execution order");
        Lane& lane = *m_lanes[tileIndex % m_lanes.size()];
        lane.drain(view, m_dimensions, m_activeOutput);
        lane.releaseBoundary(*m_activeSpillBoundary);
        lane.acquire(view, m_physicalColumns, m_pointStride,
                     tileIndex * m_tilePointCapacity, m_activeStages.size(),
                     tileIndex);
        m_hostToDeviceBytes =
            checkedAdd(m_hostToDeviceBytes, lane.currentUploadBytes(),
                       "resident H2D byte count overflows");
        m_hostToDevicePackingBytes = checkedAdd(
            m_hostToDevicePackingBytes, lane.currentUploadPackingBytes(),
            "resident H2D packing byte count overflows");
        ++m_nextTile;
        return lane.batch();
#else
        (void)view;
        (void)tileIndex;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    [[nodiscard]] std::uint8_t* tileKeepMask(std::size_t tileIndex)
    {
#if PDG_HAS_CUDA
        if (m_state != State::RegionActive || !m_activeOutput ||
            tileIndex >= m_nextTile || tileIndex != m_submittedTiles)
            throw std::logic_error(
                "resident keep-mask access is out of execution order");
        std::uint8_t* mask =
            m_lanes[tileIndex % m_lanes.size()]->deviceKeepMask();
        if (!mask)
            throw std::logic_error("resident lane has no keep-mask allocation");
        return mask;
#else
        (void)tileIndex;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    void beginStage(std::size_t tileIndex, std::size_t stageIndex)
    {
#if PDG_HAS_CUDA
        if (m_state != State::RegionActive || tileIndex >= m_nextTile ||
            tileIndex != m_submittedTiles ||
            stageIndex >= m_activeStages.size())
            throw std::logic_error(
                "resident stage start is out of execution order");
        Lane& lane = *m_lanes[tileIndex % m_lanes.size()];
        lane.beginStage(*m_activeStages[stageIndex], stageIndex,
                        m_activeColumns);
        if (lane.currentDeviceBytes() > m_bytesPerLane)
            throw std::runtime_error(
                "resident lane exceeded the planner VRAM estimate");
        std::size_t observed = 0U;
        for (const std::unique_ptr<Lane>& candidate : m_lanes)
            observed = checkedAdd(
                observed, candidate->currentDeviceBytes(),
                "resident observed device peak byte count overflows");
        m_schedule.observedPeakLaneBytes =
            (std::max)(m_schedule.observedPeakLaneBytes, observed);
        if (m_schedule.observedPeakLaneBytes > m_schedule.peakLaneBytes)
            throw std::runtime_error(
                "resident execution exceeded the planner VRAM estimate");
#else
        (void)tileIndex;
        (void)stageIndex;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    void endStage(std::size_t tileIndex, std::size_t stageIndex)
    {
#if PDG_HAS_CUDA
        if (m_state != State::RegionActive || tileIndex >= m_nextTile ||
            tileIndex != m_submittedTiles ||
            stageIndex >= m_activeStages.size())
            throw std::logic_error(
                "resident stage end is out of execution order");
        Lane& lane = *m_lanes[tileIndex % m_lanes.size()];
        lane.endStage(*m_activeStages[stageIndex], stageIndex);
#else
        (void)tileIndex;
        (void)stageIndex;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    void submitTile(PointView& view, std::size_t tileIndex,
                    pdg::PointBatch& result)
    {
#if PDG_HAS_CUDA
        ensureView(view);
        if (m_state != State::RegionActive ||
            tileIndex >= m_schedule.tileCount || tileIndex >= m_nextTile ||
            tileIndex != m_submittedTiles)
            throw std::logic_error(
                "resident tile submission is out of execution order");
        Lane& lane = *m_lanes[tileIndex % m_lanes.size()];
        lane.submit(result, m_pointStride, m_activeColumns,
                    m_activeOutput != nullptr);
        m_deviceToHostBytes =
            checkedAdd(m_deviceToHostBytes, lane.currentSpillBytes(),
                       "resident D2H byte count overflows");
        m_deviceToHostPackingBytes = checkedAdd(
            m_deviceToHostPackingBytes, lane.currentSpillPackingBytes(),
            "resident D2H packing byte count overflows");
        ++m_submittedTiles;
#else
        (void)view;
        (void)tileIndex;
        (void)result;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    void endRegion(PointView& view, std::size_t residentRegion)
    {
#if PDG_HAS_CUDA
        ensureView(view);
        if (m_state != State::RegionActive ||
            residentRegion != m_activeRegion ||
            m_nextTile != m_schedule.tileCount ||
            m_submittedTiles != m_schedule.tileCount)
            throw std::logic_error(
                "resident device region ended before all tiles completed");
        // Surviving points append to the output view during drains, so the
        // remaining pending lanes must drain in tile order to preserve the
        // stable source order across lanes.
        std::vector<Lane*> pending;
        pending.reserve(m_lanes.size());
        for (const std::unique_ptr<Lane>& lane : m_lanes)
            if (lane->hasPendingTile())
                pending.push_back(lane.get());
        std::sort(
            pending.begin(), pending.end(),
            [](const Lane* left, const Lane* right)
            { return left->pendingTileIndex() < right->pendingTileIndex(); });
        for (Lane* lane : pending)
            lane->drain(view, m_dimensions, m_activeOutput);
        for (const std::unique_ptr<Lane>& lane : m_lanes)
            lane->releaseBoundary(*m_activeSpillBoundary);
        view.invalidateProducts();
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::HostToDevice, residentRegion,
            m_hostToDeviceBytes, m_hostToDevicePackingBytes);
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceToHost, residentRegion,
            m_deviceToHostBytes, m_deviceToHostPackingBytes);
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionEnd, residentRegion);
        if (m_activeOutput)
        {
            m_completedRegionHadPredicate = true;
            m_completedRegionInputCount = m_pointCount;
            m_cardinalityChanged = true;
            m_pointCount = static_cast<std::size_t>(m_activeOutput->size());
            m_view = m_activeOutput;
        }
        else
            m_completedRegionHadPredicate = false;
        m_activeOutput = nullptr;
        m_activeColumns.clear();
        m_activeStages.clear();
        m_state = State::RegionComplete;
#else
        (void)view;
        (void)residentRegion;
        throw std::runtime_error(
            "resident execution requires a CUDA-enabled build");
#endif
    }

    [[nodiscard]] const pdg::TiledSchedule& schedule() const noexcept
    {
        return m_schedule;
    }

    [[nodiscard]] std::size_t observedOutputPointCount() const noexcept
    {
        return m_pointCount;
    }

    [[nodiscard]] const ResidentPhaseSeconds& phaseSeconds() const noexcept
    {
        return m_phaseSeconds;
    }

    [[nodiscard]] ResidentPhaseSeconds& phaseAccumulator() noexcept
    {
        return m_phaseSeconds;
    }

    [[nodiscard]] ResidentManagerDetailSeconds*
    managerDetailAccumulator() noexcept
    {
        return m_managerTiming && !m_managerTiming->invalid
                   ? &m_managerTiming->detail
                   : nullptr;
    }

    void beginManagerPhaseTiming(
        std::chrono::steady_clock::time_point managerStarted,
        std::chrono::steady_clock::time_point executeStarted) noexcept
    {
        m_managerTiming = ManagerTimingState{.managerStarted = managerStarted,
                                             .executeStarted = executeStarted};
    }

    [[nodiscard]] ResidentManagerPhaseSeconds finishManagerPhaseTiming(
        std::chrono::steady_clock::time_point executeEnded) noexcept
    {
        ResidentManagerPhaseSeconds result;
        if (m_managerTiming && !m_managerTiming->invalid &&
            m_managerTiming->uploadSeen && m_managerTiming->spillSeen)
        {
            const ManagerTimingState& timing = *m_managerTiming;
            result.managerGraphAndPrepare = phaseSecondsBetween(
                timing.managerStarted, timing.executeStarted);
            result.readerRowTableMaterialization =
                timing.readerRowTableMaterialization;
            result.residentWrapperIndexFilter =
                timing.residentWrapperIndexFilter;
            result.postSpillStageControl =
                phaseSecondsBetween(timing.spillEnded, executeEnded);
            result.total =
                phaseSecondsBetween(timing.managerStarted, executeEnded);
            result.detail = timing.detail;
            result.complete = true;
        }
        m_managerTiming.reset();
        return result;
    }

private:
    void markManagerUpload() noexcept
    {
        if (!m_managerTiming)
            return;
        if (m_managerTiming->uploadSeen || m_managerTiming->spillSeen)
        {
            m_managerTiming->invalid = true;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        m_managerTiming->readerRowTableMaterialization =
            phaseSecondsBetween(m_managerTiming->executeStarted, now);
        m_managerTiming->residentStarted = now;
        m_managerTiming->uploadSeen = true;
    }

    void markManagerSpill() noexcept
    {
        if (!m_managerTiming)
            return;
        if (!m_managerTiming->uploadSeen || m_managerTiming->spillSeen)
        {
            m_managerTiming->invalid = true;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        m_managerTiming->residentWrapperIndexFilter =
            phaseSecondsBetween(m_managerTiming->residentStarted, now);
        m_managerTiming->spillEnded = now;
        m_managerTiming->spillSeen = true;
    }

    enum class State
    {
        Empty,
        AwaitingRegion,
        RegionActive,
        RegionComplete,
        HostMaterialized
    };

    // One physical PointView dimension: the packed tile layout is the row
    // layout, so these offsets serve both whole-row copies and the
    // field-by-field fallback for tables without row storage.
    struct PhysicalColumn
    {
        Dimension::Id id = Dimension::Id::Unknown;
        Dimension::Type type = Dimension::Type::None;
        std::size_t offset = 0U;
        std::size_t size = 0U;
    };

#if PDG_HAS_CUDA
    class Lane
    {
    public:
        Lane(std::size_t capacity, std::size_t pointStride,
             pdg::DimensionRegistry& dimensions,
             std::size_t releaseThresholdBytes, std::size_t maskCapacity)
            : m_pinnedMemory(pdg::makeCudaPinnedMemoryResource()),
              m_deviceMemory(
                  pdg::makeCudaMemoryResource(releaseThresholdBytes)),
              m_hostPacked(m_pinnedMemory->allocate(
                  checkedProduct(capacity, pointStride,
                                 "resident pinned tile size overflows"),
                  alignof(std::max_align_t))),
              m_devicePacked(m_deviceMemory->allocate(
                  checkedProduct(capacity, pointStride,
                                 "resident device tile size overflows"),
                  alignof(std::max_align_t))),
              m_hostMask(maskCapacity
                             ? m_pinnedMemory->allocate(
                                   maskCapacity, alignof(std::max_align_t))
                             : nullptr),
              m_deviceMask(maskCapacity
                               ? m_deviceMemory->allocate(
                                     maskCapacity, alignof(std::max_align_t))
                               : nullptr),
              m_batch(std::make_unique<pdg::PointBatch>(
                  capacity,
                  pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                  dimensions, *m_deviceMemory)),
              m_stream(static_cast<cudaStream_t>(
                  m_deviceMemory->nativeStreamHandle()))
        {
            if (!m_stream)
                throw std::invalid_argument(
                    "resident device lane has no CUDA stream");
            PDG_CUDA_CHECK(cudaEventCreateWithFlags(&m_completion,
                                                    cudaEventDisableTiming));
        }

        ~Lane()
        {
            if (m_workQueued && m_stream)
                PDG_CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(m_stream));
            if (m_completion)
                PDG_CUDA_CHECK_NOEXCEPT(cudaEventDestroy(m_completion));
        }

        Lane(const Lane&) = delete;
        Lane& operator=(const Lane&) = delete;

        void
        probeAllocations(const std::vector<const pdg::PlannedStage*>& stages,
                         const pdg::ResidencyBoundary& boundary,
                         std::size_t bytesPerLane)
        {
            if (m_pending || m_acquired || m_batch->allocatedBytes())
                throw std::logic_error(
                    "resident lane allocation probe requires an empty lane");
            m_batch->setSize(m_batch->capacity());
            for (const pdg::PlannedStage* stage : stages)
            {
                pdg::preparePlannedDeviceColumns(*m_batch, *stage);
                for (pdg::DimensionId id : stage->deviceLiveBefore)
                    if (!m_batch->has(id))
                        throw std::logic_error(
                            "resident allocation probe found a missing live "
                            "input");
                m_peakDeviceBytes =
                    (std::max)(m_peakDeviceBytes,
                               checkedAdd(
                                   fixedDeviceBytes(),
                                   m_batch->allocatedBytes(),
                                   "resident probed lane peak overflows"));
                if (m_peakDeviceBytes > bytesPerLane)
                    throw std::runtime_error(
                        "resident lane allocation probe exceeds the planner "
                        "VRAM "
                        "estimate");
                pdg::releasePlannedDeviceColumns(*m_batch, *stage);
            }
            releaseBoundary(boundary);
            if (m_batch->allocatedBytes())
                throw std::logic_error(
                    "resident lane allocation probe left unowned device "
                    "columns");
        }

        void resetPeakDeviceBytes() noexcept
        {
            m_peakDeviceBytes = 0U;
        }

        void releaseBoundary(const pdg::ResidencyBoundary& boundary)
        {
            pdg::releaseSpilledDeviceColumns(*m_batch, boundary);
        }

        void acquire(PointView& view,
                     const std::vector<PhysicalColumn>& physicalColumns,
                     std::size_t pointStride, std::size_t offset,
                     std::size_t stageCount, std::size_t tileIndex)
        {
            NvtxRange range("pdg.resident.acquire");
            if (m_pending || m_acquired)
                throw std::logic_error(
                    "resident lane reused before it was drained");
            m_offset = offset;
            m_pendingTileIndex = tileIndex;
            m_maskValid = false;
            m_count =
                (std::min)(m_batch->capacity(),
                           static_cast<std::size_t>(view.size()) - offset);
            m_bytes =
                checkedProduct(m_count, pointStride,
                               "resident tile transfer byte count overflows");
            m_spillBytes = 0U;
            // Serializing one complete physical record for every transferred
            // point is explicit packing work; count its bytes independently
            // from PCIe volume.
            m_uploadPackingBytes = m_bytes;
            m_spillPackingBytes = 0U;
            auto* packed = static_cast<std::byte*>(m_hostPacked->data());
            const auto packStarted = std::chrono::steady_clock::now();
            for (std::size_t point = 0; point < m_count; ++point)
            {
                const auto pointId = static_cast<PointId>(offset + point);
                std::byte* destination = packed + point * pointStride;
                // The packed tile layout is the physical row layout, so a
                // row-backed table serializes with one copy. Tables without
                // row storage fall back to field-by-field access at the same
                // physical offsets.
                if (const char* row = view.getPoint(pointId))
                    std::memcpy(destination, row, pointStride);
                else
                    for (const PhysicalColumn& column : physicalColumns)
                        view.getField(reinterpret_cast<char*>(destination +
                                                              column.offset),
                                      column.id, column.type, pointId);
            }
            addPhaseSeconds(m_phases ? &m_phases->uploadPack : nullptr,
                            packStarted);
            PDG_CUDA_CHECK(cudaMemcpyAsync(m_devicePacked->data(),
                                           m_hostPacked->data(), m_bytes,
                                           cudaMemcpyHostToDevice, m_stream));
            m_workQueued = true;
            m_batch->setSize(m_count);
            m_stageCount = stageCount;
            m_nextStage = 0U;
            m_acquired = true;
        }

        void beginStage(const pdg::PlannedStage& stage, std::size_t stageIndex,
                        std::span<const pdg::PackedPointColumn> columns)
        {
            if (!m_acquired || stageIndex != m_nextStage ||
                stageIndex >= m_stageCount)
                throw std::logic_error(
                    "resident lane stage start is out of execution order");

            std::vector<pdg::PackedPointColumn> unpack;
            unpack.reserve(stage.deviceMaterialize.size());
            for (pdg::DimensionId id : stage.deviceMaterialize)
            {
                if (m_batch->has(id))
                    continue;
                const auto column =
                    std::find_if(columns.begin(), columns.end(),
                                 [&](const pdg::PackedPointColumn& candidate)
                                 { return candidate.id == id; });
                if (column == columns.end())
                    throw std::invalid_argument(
                        "planned resident materialization is absent from the "
                        "packed point record: dimension " +
                        std::to_string(id.value()));
                unpack.push_back(*column);
            }
            pdg::preparePlannedDeviceColumns(*m_batch, stage);
            if (!unpack.empty())
                pdg::unpackPackedPointBatchDevice(m_devicePacked->data(),
                                                  m_pointStride, nullptr,
                                                  m_count, unpack, *m_batch);
            for (pdg::DimensionId id : stage.deviceLiveBefore)
                if (!m_batch->has(id))
                    throw std::logic_error(
                        "planned resident live input is not materialized");
            m_peakDeviceBytes =
                (std::max)(m_peakDeviceBytes,
                           checkedAdd(
                               fixedDeviceBytes(), m_batch->allocatedBytes(),
                               "resident lane allocation peak overflows"));
        }

        void endStage(const pdg::PlannedStage& stage, std::size_t stageIndex)
        {
            if (!m_acquired || stageIndex != m_nextStage ||
                stageIndex >= m_stageCount)
                throw std::logic_error(
                    "resident lane stage end is out of execution order");
            pdg::releasePlannedDeviceColumns(*m_batch, stage);
            ++m_nextStage;
        }

        void submit(pdg::PointBatch& result, std::size_t pointStride,
                    std::span<const pdg::PackedPointColumn> columns,
                    bool maskActive)
        {
            NvtxRange range("pdg.resident.submit");
            if (!m_acquired || &result != m_batch.get() ||
                result.size() != m_count || m_nextStage != m_stageCount)
                throw std::logic_error(
                    "resident point program changed tile ownership or "
                    "cardinality");
            if (maskActive && !m_deviceMask)
                throw std::logic_error(
                    "resident lane has no keep-mask allocation");
            std::vector<pdg::PackedPointColumn> written;
            written.reserve(columns.size());
            std::size_t writtenBytesPerPoint = 0U;
            for (const pdg::PackedPointColumn& column : columns)
                if (column.written && result.has(column.id))
                {
                    written.push_back(column);
                    writtenBytesPerPoint =
                        checkedAdd(writtenBytesPerPoint,
                                   pdg::dimensionTypeSize(column.physicalType),
                                   "resident spill packing width overflows");
                }
            m_spillPackingBytes =
                checkedProduct(m_count, writtenBytesPerPoint,
                               "resident spill packing byte count overflows");
            pdg::repackPackedPointBatchDevice(result, m_devicePacked->data(),
                                              pointStride, nullptr, written);
            PDG_CUDA_CHECK(cudaMemcpyAsync(m_hostPacked->data(),
                                           m_devicePacked->data(), m_bytes,
                                           cudaMemcpyDeviceToHost, m_stream));
            m_spillBytes = m_bytes;
            if (maskActive)
            {
                PDG_CUDA_CHECK(
                    cudaMemcpyAsync(m_hostMask->data(), m_deviceMask->data(),
                                    m_count, cudaMemcpyDeviceToHost, m_stream));
                m_spillBytes =
                    checkedAdd(m_spillBytes, m_count,
                               "resident keep-mask transfer overflows");
            }
            PDG_CUDA_CHECK(cudaEventRecord(m_completion, m_stream));
            m_writtenColumns = std::move(written);
            m_maskValid = maskActive;
            m_pending = true;
            m_acquired = false;
        }

        void drain(PointView& view, const pdg::DimensionRegistry& dimensions,
                   PointView* output)
        {
            if (!m_pending)
                return;
            NvtxRange range("pdg.resident.drain");
            const auto waitStarted = std::chrono::steady_clock::now();
            PDG_CUDA_CHECK(cudaEventSynchronize(m_completion));
            addPhaseSeconds(m_phases ? &m_phases->spillWait : nullptr,
                            waitStarted);
            if (m_maskValid != (output != nullptr))
                throw std::logic_error(
                    "resident drain output does not match the submitted keep "
                    "mask");
            const auto* packed =
                static_cast<const std::byte*>(m_hostPacked->data());
            const auto* mask =
                m_maskValid
                    ? static_cast<const std::uint8_t*>(m_hostMask->data())
                    : nullptr;
            struct BoundColumn
            {
                Dimension::Id pdalId = Dimension::Id::Unknown;
                Dimension::Type pdalType = Dimension::Type::None;
                std::size_t offset = 0U;
                std::size_t size = 0U;
            };
            std::vector<BoundColumn> bound;
            bound.reserve(m_writtenColumns.size());
            for (const pdg::PackedPointColumn& column : m_writtenColumns)
            {
                if (!column.written)
                    continue;
                const pdg::DimensionDefinition& definition =
                    dimensions.require(column.id);
                const Dimension::Id pdalId =
                    view.layout()->findDim(definition.name);
                if (pdalId == Dimension::Id::Unknown)
                    throw std::invalid_argument(
                        "resident result dimension is absent from PointView");
                const Dimension::Type pdalType = view.layout()->dimType(pdalId);
                if (toPdgType(pdalType) != column.physicalType)
                    throw std::invalid_argument(
                        "resident result dimension changed physical type");
                bound.push_back({pdalId, pdalType, column.offset,
                                 pdg::dimensionTypeSize(column.physicalType)});
            }
            const auto publishStarted = std::chrono::steady_clock::now();
            for (std::size_t point = 0; point < m_count; ++point)
            {
                // Dropped points publish nothing: they leave the pipeline at
                // this declared cardinality change and no later stage can
                // observe them.
                if (mask && !mask[point])
                    continue;
                const PointId viewPoint =
                    static_cast<PointId>(m_offset + point);
                // The packed column offset equals the physical row offset,
                // so written columns copy directly into a row-backed table;
                // tables without row storage keep the field-wise path.
                if (char* row = view.getPoint(viewPoint))
                    for (const BoundColumn& column : bound)
                        std::memcpy(row + column.offset,
                                    packed + point * m_pointStride +
                                        column.offset,
                                    column.size);
                else
                    for (const BoundColumn& column : bound)
                        view.setField(column.pdalId, column.pdalType, viewPoint,
                                      packed + point * m_pointStride +
                                          column.offset);
                if (output)
                    output->appendPoint(view, viewPoint);
            }
            addPhaseSeconds(m_phases ? &m_phases->spillPublish : nullptr,
                            publishStarted);
            m_writtenColumns.clear();
            m_maskValid = false;
            m_pending = false;
        }

        [[nodiscard]] pdg::PointBatch& batch() noexcept
        {
            return *m_batch;
        }

        [[nodiscard]] std::uint8_t* deviceKeepMask() noexcept
        {
            return m_deviceMask
                       ? static_cast<std::uint8_t*>(m_deviceMask->data())
                       : nullptr;
        }

        [[nodiscard]] bool hasPendingTile() const noexcept
        {
            return m_pending;
        }

        [[nodiscard]] std::size_t pendingTileIndex() const noexcept
        {
            return m_pendingTileIndex;
        }

        [[nodiscard]] std::size_t currentUploadBytes() const noexcept
        {
            return m_bytes;
        }

        [[nodiscard]] std::size_t currentSpillBytes() const noexcept
        {
            return m_spillBytes;
        }

        [[nodiscard]] std::size_t currentUploadPackingBytes() const noexcept
        {
            return m_uploadPackingBytes;
        }

        [[nodiscard]] std::size_t currentSpillPackingBytes() const noexcept
        {
            return m_spillPackingBytes;
        }

        [[nodiscard]] std::size_t peakDeviceBytes() const noexcept
        {
            return m_peakDeviceBytes;
        }

        [[nodiscard]] std::size_t currentDeviceBytes() const
        {
            return checkedAdd(fixedDeviceBytes(), m_batch->allocatedBytes(),
                              "resident current lane allocation overflows");
        }

        void setPointStride(std::size_t value) noexcept
        {
            m_pointStride = value;
        }

        void setPhaseAccumulator(ResidentPhaseSeconds* value) noexcept
        {
            m_phases = value;
        }

    private:
        [[nodiscard]] std::size_t fixedDeviceBytes() const
        {
            return checkedAdd(m_devicePacked->size(),
                              m_deviceMask ? m_deviceMask->size() : 0U,
                              "resident fixed lane allocation overflows");
        }

        // Resources precede allocations and the PointBatch so reverse member
        // destruction releases every allocation before its owning stream.
        std::unique_ptr<pdg::MemoryResource> m_pinnedMemory;
        std::unique_ptr<pdg::MemoryResource> m_deviceMemory;
        std::unique_ptr<pdg::Allocation> m_hostPacked;
        std::unique_ptr<pdg::Allocation> m_devicePacked;
        std::unique_ptr<pdg::Allocation> m_hostMask;
        std::unique_ptr<pdg::Allocation> m_deviceMask;
        std::unique_ptr<pdg::PointBatch> m_batch;
        cudaStream_t m_stream = nullptr;
        cudaEvent_t m_completion = nullptr;
        std::vector<pdg::PackedPointColumn> m_writtenColumns;
        ResidentPhaseSeconds* m_phases = nullptr;
        std::size_t m_pointStride = 0U;
        std::size_t m_offset = 0U;
        std::size_t m_pendingTileIndex = 0U;
        std::size_t m_count = 0U;
        std::size_t m_bytes = 0U;
        std::size_t m_spillBytes = 0U;
        std::size_t m_uploadPackingBytes = 0U;
        std::size_t m_spillPackingBytes = 0U;
        std::size_t m_stageCount = 0U;
        std::size_t m_nextStage = 0U;
        std::size_t m_peakDeviceBytes = 0U;
        bool m_pending = false;
        bool m_acquired = false;
        bool m_maskValid = false;
        bool m_workQueued = false;
    };
#else
    class Lane
    {
    public:
        void releaseBoundary(const pdg::ResidencyBoundary&) noexcept {}
    };
#endif

    // True when a planned stage reads or writes the dimension named `name`

    // (by the plan registry's naming); such dimensions must keep the

    // planned type on the device.

    [[nodiscard]] bool planTouchesDimension(std::string_view name) const

    {

        if (!m_plan || !m_planDimensions)

            return true;

        for (pdg::DimensionId id : m_plan->summary().touchedDimensions)

            if (const pdg::DimensionDefinition* planned =

                    m_planDimensions->find(id);

                planned && planned->name == name)

                return true;

        return false;

    }


    void bindLayout(PointLayout& layout, std::size_t pointCount)
    {
        if (m_layoutBound)
        {
            // A declared cardinality change updates m_pointCount when its
            // region completes; a zero count is then a legal empty survivor
            // set rather than a binding error.
            if (m_pointCount != pointCount ||
                m_pointStride != layout.pointSize() ||
                !sameLayout(m_dimTypes, layout.dimTypes()))
                throw std::invalid_argument(
                    "resident execution cardinality or layout changed");
            return;
        }
        m_pointCount = pointCount;
        m_pointStride = layout.pointSize();
        m_dimTypes = layout.dimTypes();
        if (!m_pointStride || m_dimTypes.empty())
            throw std::invalid_argument(
                "resident PointView has an empty physical layout");
        std::size_t packedStride = 0U;
        m_physicalColumns.clear();
        m_physicalColumns.reserve(m_dimTypes.size());
        m_fileTypedDimensions.clear();
        for (const DimType& dim : m_dimTypes)
        {
            const pdg::DimensionType type = toPdgType(dim.m_type);
            if (type == pdg::DimensionType::None)
                throw std::invalid_argument(
                    "resident PointView contains an unsupported dimension "
                    "type");
            const std::size_t size = Dimension::size(dim.m_type);
            packedStride = checkedAdd(packedStride, size,
                                      "resident packed point stride overflows");
            const std::size_t physicalOffset = layout.dimOffset(dim.m_id);
            if (physicalOffset > layout.pointSize() ||
                size > layout.pointSize() - physicalOffset)
                throw std::invalid_argument(
                    "resident PointView dimension lies outside its record");
            m_physicalColumns.push_back(
                {dim.m_id, dim.m_type, physicalOffset, size});
            const std::string name = layout.dimName(dim.m_id);
            const pdg::DimensionDefinition* existing = m_dimensions.find(name);
            if (!existing)
                m_dimensions.registerCustom(name, type);
            else if (existing->type != type)
            {
                // D0278: a file's extra-bytes VLR may declare a standard
                // dimension name with another concrete type (AHN4 tiles carry
                // Deviation as uint16 where PDAL's standard is double). Stock
                // PDAL follows the file; so does the resident path for a
                // dimension no planned device stage reads or writes — it
                // travels in the physical row and is never interpreted
                // through the PDG definition. A dimension the plan touches
                // keeps the refusal: device stages compute through the
                // planned type, and this is exactly the case the runtime
                // mapping check below would reject.
                if (planTouchesDimension(name))
                    throw std::invalid_argument(
                        "resident PointView dimension type differs from PDG "
                        "for a device-computed dimension: " + name);
                m_dimensions.redefineType(existing->id, type);
                m_fileTypedDimensions.push_back(name);
                if (std::getenv("PDG_DEBUG_HYBRID"))
                    std::cerr
                        << "gpupdal: resident layout follows the file's type "
                           "for dimension "
                        << name << '\n';
            }
        }
        if (packedStride != m_pointStride)
            throw std::invalid_argument(
                "resident PointView packed stride is inconsistent");
        // The packed tile layout is the physical row layout, so the declared
        // dimension offsets must partition the record exactly. A layout whose
        // offsets are not byte offsets — for example a column table that
        // rewrote them into per-dimension ordinals — must fail closed here,
        // before any boundary executes.
        std::vector<PhysicalColumn> ordered = m_physicalColumns;
        std::sort(ordered.begin(), ordered.end(),
                  [](const PhysicalColumn& left, const PhysicalColumn& right)
                  { return left.offset < right.offset; });
        std::size_t expectedOffset = 0U;
        for (const PhysicalColumn& column : ordered)
        {
            if (column.offset != expectedOffset)
                throw std::invalid_argument(
                    "resident PointView dimension offsets are not physical "
                    "row offsets");
            expectedOffset += column.size;
        }
        if (expectedOffset != m_pointStride)
            throw std::invalid_argument(
                "resident PointView dimension offsets are not physical row "
                "offsets");
        for (pdg::DimensionId id : m_plan->summary().touchedDimensions)
        {
            const pdg::DimensionDefinition& planned =
                m_planDimensions->require(id);
            const pdg::DimensionDefinition* runtime =
                m_dimensions.find(planned.name);
            if (!runtime || runtime->id != planned.id ||
                runtime->type != planned.type)
                throw std::invalid_argument(
                    "resident PointView custom dimension mapping differs from "
                    "the plan");
        }
        m_layoutBound = true;
    }

    void ensureView(PointView& view)
    {
        bindLayout(*view.layout(), static_cast<std::size_t>(view.size()));
        if (!m_view)
        {
            m_view = &view;
            return;
        }
        if (m_view != &view)
            throw std::invalid_argument(
                "resident execution PointView identity changed");
    }

    void requireActiveDelegatedRegion(PointView& view,
                                      std::size_t residentRegion)
    {
        ensureView(view);
        if (!m_delegatedPlan || m_state != State::RegionActive ||
            residentRegion != m_activeRegion || !m_delegatedStagingMemory ||
            !m_delegatedDeviceMemory)
            throw std::logic_error(
                "resident delegated resource requested outside its region");
    }

    std::vector<pdg::PackedPointColumn>
    normalizeColumns(PointView& view,
                     std::span<const pdg::PackedPointColumn> columns) const
    {
        std::vector<pdg::PackedPointColumn> result;
        result.reserve(columns.size());
        for (const pdg::PackedPointColumn& column : columns)
        {
            const pdg::DimensionDefinition& definition =
                m_dimensions.require(column.id);
            const Dimension::Id pdalId =
                view.layout()->findDim(definition.name);
            if (pdalId == Dimension::Id::Unknown ||
                toPdgType(view.layout()->dimType(pdalId)) !=
                    column.physicalType)
                throw std::invalid_argument(
                    "resident point-program packed layout differs from the "
                    "PointView");
            // The packed tile layout is the PointView's physical row layout,
            // so a row-backed table can serialize and publish records with
            // whole-row copies instead of field-by-field access.
            const std::size_t packedOffset = view.layout()->dimOffset(pdalId);
            const std::size_t size =
                pdg::dimensionTypeSize(column.physicalType);
            if (!size || packedOffset > m_pointStride ||
                size > m_pointStride - packedOffset)
                throw std::invalid_argument(
                    "resident point-program column lies outside the record");
            result.push_back(
                {column.id, column.physicalType, packedOffset, column.written});
        }
        return result;
    }

    std::vector<const pdg::PlannedStage*>
    regionStages(std::size_t residentRegion) const
    {
        std::vector<const pdg::PlannedStage*> result;
        for (const pdg::PlannedStage& stage : m_plan->stages())
            if (stage.residentRegion == residentRegion)
                result.push_back(&stage);
        return result;
    }

    [[nodiscard]] bool
    regionDeclaresCardinalityChange(std::size_t residentRegion) const
    {
        for (const pdg::PlannedStage& stage : m_plan->stages())
            if (stage.residentRegion == residentRegion &&
                !stage.descriptor.fusion.cardinalityPreserving)
                return true;
        return false;
    }

    [[nodiscard]] bool keepMaskNeeded() const
    {
        if (m_selectedRegions.empty())
        {
            for (const pdg::PlannedStage& stage : m_plan->stages())
                if (stage.residentRegion != pdg::NoResidentRegion &&
                    !stage.descriptor.fusion.cardinalityPreserving)
                    return true;
            return false;
        }
        for (std::size_t region : m_selectedRegions)
            if (regionDeclaresCardinalityChange(region))
                return true;
        return false;
    }

    const pdg::ResidencyBoundary&
    spillBoundary(std::size_t residentRegion) const
    {
        const pdg::ResidencyBoundary* result = nullptr;
        for (const pdg::ResidencyBoundary& boundary :
             m_plan->summary().residencyBoundaries)
        {
            if (boundary.kind != pdg::ResidencyBoundaryKind::Spill ||
                boundary.producer >= m_plan->stages().size() ||
                m_plan->stages()[boundary.producer].residentRegion !=
                    residentRegion)
                continue;
            if (result)
                throw std::invalid_argument(
                    "resident execution region has multiple spill boundaries");
            result = &boundary;
        }
        if (!result)
            throw std::invalid_argument(
                "resident execution region has no spill boundary");
        return *result;
    }

    void initializeLanes()
    {
#if PDG_HAS_CUDA
        if (!m_lanes.empty())
            return;
        const std::size_t capacity =
            (std::min)(m_tilePointCapacity, m_pointCount);
        const bool maskNeeded = keepMaskNeeded();
        const std::size_t packedBytes =
            checkedProduct(capacity, m_pointStride,
                           "resident packed device tile estimate overflows");
        const std::size_t columnBytes = m_plan->estimatedDeviceBytes(capacity);
        std::size_t bytesPerLane =
            checkedAdd(packedBytes, columnBytes,
                       "resident device lane estimate overflows");
        if (maskNeeded)
            bytesPerLane =
                checkedAdd(bytesPerLane, capacity,
                           "resident keep-mask lane estimate overflows");
        m_bytesPerLane = bytesPerLane;
        m_schedule = pdg::makeTiledSchedule(
            {.pipelineClass = maskNeeded
                                  ? pdg::PipelineClass::OrderedPointProgram
                                  : pdg::PipelineClass::FusedPointProgram,
             .itemCount = m_pointCount,
             .tileItems = capacity,
             .bytesPerLane = bytesPerLane,
             .memoryBudgetBytes = m_deviceMemoryBudgetBytes});
        if (!m_schedule.activeLaneCount)
            throw std::runtime_error(
                "resident scheduler selected no active CUDA lanes");
        m_lanes.reserve(m_schedule.activeLaneCount);
        for (std::size_t lane = 0; lane < m_schedule.activeLaneCount; ++lane)
        {
            auto instance = std::make_unique<Lane>(capacity, m_pointStride,
                                                   m_dimensions, bytesPerLane,
                                                   maskNeeded ? capacity : 0U);
            instance->setPointStride(m_pointStride);
            instance->setPhaseAccumulator(&m_phaseSeconds);
            m_lanes.push_back(std::move(instance));
        }
#endif
    }

    const pdg::Plan* m_plan = nullptr;
    const pdg::DimensionRegistry* m_planDimensions = nullptr;
    std::size_t m_deviceMemoryBudgetBytes = 0U;
    std::size_t m_tilePointCapacity = 0U;
    pdg::DimensionRegistry m_dimensions;
    // Names whose runtime type was taken from the file (D0278), for
    // diagnostics; empty for every layout that agrees with the registry.
    std::vector<std::string> m_fileTypedDimensions;
    PointView* m_view = nullptr;
    DimTypeList m_dimTypes;
    std::vector<pdg::PackedPointColumn> m_activeColumns;
    std::vector<const pdg::PlannedStage*> m_activeStages;
    std::vector<std::size_t> m_selectedRegions;
    std::size_t m_pointCount = 0U;
    std::size_t m_pointStride = 0U;
    std::size_t m_bytesPerLane = 0U;
    std::vector<std::unique_ptr<Lane>> m_lanes;
    pdg::TiledSchedule m_schedule;
    const pdg::ResidencyBoundary* m_activeSpillBoundary = nullptr;
    PointView* m_activeOutput = nullptr;
    State m_state = State::Empty;
    std::size_t m_activeRegion = pdg::NoResidentRegion;
    std::size_t m_nextTile = 0U;
    std::size_t m_submittedTiles = 0U;
    std::size_t m_hostToDeviceBytes = 0U;
    std::size_t m_deviceToHostBytes = 0U;
    std::size_t m_hostToDevicePackingBytes = 0U;
    std::size_t m_deviceToHostPackingBytes = 0U;
    std::size_t m_completedRegionInputCount = 0U;
    std::vector<PhysicalColumn> m_physicalColumns;
    ResidentPhaseSeconds m_phaseSeconds;
    std::optional<ManagerTimingState> m_managerTiming;
    bool m_delegatedPlan = false;
    std::size_t m_delegatedBaseDeviceBytes = 0U;
    std::size_t m_delegatedBaseHostBytes = 0U;
    std::unique_ptr<pdg::MemoryResource> m_delegatedStagingMemory;
    std::unique_ptr<pdg::MemoryResource> m_delegatedDeviceMemory;
    std::unique_ptr<pdg::RasterGridProduct> m_rasterGridProduct;
    bool m_completedRegionHadPredicate = false;
    bool m_cardinalityChanged = false;
    bool m_layoutBound = false;
};

ResidentExecutionContext::ResidentExecutionContext(
    const pdg::Plan& plan, const pdg::DimensionRegistry& planDimensions,
    std::size_t deviceMemoryBudgetBytes, std::size_t tilePointCapacity)
    : m_impl(std::make_unique<Impl>(plan, planDimensions,
                                    deviceMemoryBudgetBytes, tilePointCapacity))
{
}

ResidentExecutionContext::~ResidentExecutionContext() = default;

void ResidentExecutionContext::preflight(
    PointLayout& layout, std::size_t pointCount,
    std::span<const std::size_t> selectedRegions)
{
    m_impl->preflight(layout, pointCount, selectedRegions);
}

void ResidentExecutionContext::enterBoundary(
    PointView& view, std::size_t boundaryId,
    ResidentBoundaryDirection direction, std::size_t residentRegion,
    bool requiresFullPointRecord)
{
    m_impl->enterBoundary(view, boundaryId, direction, residentRegion,
                          requiresFullPointRecord);
}

void ResidentExecutionContext::beginRegion(
    PointView& view, std::size_t residentRegion,
    std::span<const pdg::PackedPointColumn> columns,
    PointView* compactionOutput)
{
    m_impl->beginRegion(view, residentRegion, columns, compactionOutput);
}

std::size_t ResidentExecutionContext::tileCount() const noexcept
{
    return m_impl->tileCount();
}

pdg::PointBatch& ResidentExecutionContext::acquireTile(PointView& view,
                                                       std::size_t tileIndex)
{
    return m_impl->acquireTile(view, tileIndex);
}

std::uint8_t* ResidentExecutionContext::tileKeepMask(std::size_t tileIndex)
{
    return m_impl->tileKeepMask(tileIndex);
}

void ResidentExecutionContext::beginDelegatedRegion(PointView& view,
                                                    std::size_t residentRegion)
{
    m_impl->beginDelegatedRegion(view, residentRegion);
}

void ResidentExecutionContext::endDelegatedRegion(PointView& view,
                                                  std::size_t residentRegion)
{
    m_impl->endDelegatedRegion(view, residentRegion);
}

pdg::MemoryResource&
ResidentExecutionContext::delegatedStagingMemory(PointView& view,
                                                 std::size_t residentRegion)
{
    return m_impl->delegatedStagingMemory(view, residentRegion);
}

pdg::MemoryResource&
ResidentExecutionContext::delegatedDeviceMemory(PointView& view,
                                                std::size_t residentRegion)
{
    return m_impl->delegatedDeviceMemory(view, residentRegion);
}

pdg::RasterGridProduct& ResidentExecutionContext::acquireRasterGridProduct(
    PointView& view, std::size_t residentRegion,
    const pdg::RasterGridFrame& frame, bool reuseExpected)
{
    return m_impl->acquireRasterGridProduct(view, residentRegion, frame,
                                            reuseExpected);
}

void ResidentExecutionContext::beginStage(std::size_t tileIndex,
                                          std::size_t stageIndex)
{
    m_impl->beginStage(tileIndex, stageIndex);
}

void ResidentExecutionContext::endStage(std::size_t tileIndex,
                                        std::size_t stageIndex)
{
    m_impl->endStage(tileIndex, stageIndex);
}

void ResidentExecutionContext::submitTile(PointView& view,
                                          std::size_t tileIndex,
                                          pdg::PointBatch& result)
{
    m_impl->submitTile(view, tileIndex, result);
}

void ResidentExecutionContext::endRegion(PointView& view,
                                         std::size_t residentRegion)
{
    m_impl->endRegion(view, residentRegion);
}

const pdg::TiledSchedule& ResidentExecutionContext::schedule() const noexcept
{
    return m_impl->schedule();
}

const ResidentPhaseSeconds&
ResidentExecutionContext::phaseSeconds() const noexcept
{
    return m_impl->phaseSeconds();
}

ResidentPhaseSeconds& ResidentExecutionContext::phaseAccumulator() noexcept
{
    return m_impl->phaseAccumulator();
}

ResidentManagerDetailSeconds*
ResidentExecutionContext::managerDetailAccumulator() noexcept
{
    return m_impl->managerDetailAccumulator();
}

void ResidentExecutionContext::beginManagerPhaseTiming(
    std::chrono::steady_clock::time_point managerStarted,
    std::chrono::steady_clock::time_point executeStarted) noexcept
{
    m_impl->beginManagerPhaseTiming(managerStarted, executeStarted);
}

ResidentManagerPhaseSeconds ResidentExecutionContext::finishManagerPhaseTiming(
    std::chrono::steady_clock::time_point executeEnded) noexcept
{
    return m_impl->finishManagerPhaseTiming(executeEnded);
}

std::size_t ResidentExecutionContext::observedOutputPointCount() const noexcept
{
    return m_impl->observedOutputPointCount();
}

ResidentExecutionScope::ResidentExecutionScope(
    const pdg::Plan& plan, const pdg::DimensionRegistry& planDimensions,
    std::size_t deviceMemoryBudgetBytes, std::size_t tilePointCapacity)
    : m_context(plan, planDimensions, deviceMemoryBudgetBytes,
                tilePointCapacity),
      m_previous(ActiveResidentContext)
{
    if (m_previous)
        throw std::logic_error("resident execution scopes cannot be nested");
    ActiveResidentContext = &m_context;
}

ResidentExecutionScope::~ResidentExecutionScope()
{
    if (ActiveResidentContext == &m_context)
        ActiveResidentContext = m_previous;
}

void ResidentExecutionScope::preflight(
    PointLayout& layout, std::size_t pointCount,
    std::span<const std::size_t> selectedRegions)
{
    m_context.preflight(layout, pointCount, selectedRegions);
}

ResidentExecutionContext& ResidentExecutionScope::context() noexcept
{
    return m_context;
}

ResidentExecutionContext& requireResidentExecutionContext()
{
    if (!ActiveResidentContext)
        throw std::logic_error(
            "resident execution marker has no active planner scope");
    return *ActiveResidentContext;
}

ResidentPhaseSeconds* activeResidentPhaseSeconds() noexcept
{
    return ActiveResidentContext ? &ActiveResidentContext->phaseAccumulator()
                                 : nullptr;
}

ResidentManagerDetailSeconds* activeResidentManagerDetailSeconds() noexcept
{
    return ActiveResidentContext
               ? ActiveResidentContext->managerDetailAccumulator()
               : nullptr;
}

} // namespace pdal::pdg_detail
