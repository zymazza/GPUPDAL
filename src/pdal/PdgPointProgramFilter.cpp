#include <pdg/Compaction.hpp>
#include <pdg/Cuda.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/ExecutionStats.hpp>
#include <pdg/Hybrid.hpp>
#include <pdg/Memory.hpp>
#include <pdg/PackedPointBatch.hpp>
#include <pdg/Plan.hpp>
#include <pdg/PointBatch.hpp>
#include <pdg/stages/Assign.hpp>
#include <pdg/stages/Expression.hpp>
#include <pdg/stages/Ordinal.hpp>
#include <pdg/stages/Transformation.hpp>

#include <pdal/Filter.hpp>
#include <pdal/PointRef.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <pdal/BatchStreamable.hpp>

#include "PdgNeighborhood.hpp"
#include "PdgResidentContext.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pdal
{

namespace
{
using Json = nlohmann::json;

class UnsupportedDeviceProgram final : public std::runtime_error
{
public:
    UnsupportedDeviceProgram()
        : std::runtime_error("point program is outside the exact CUDA "
                             "execution envelope")
    {
    }
};

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

Dimension::Type toPdalType(pdg::DimensionType type)
{
    switch (type)
    {
    case pdg::DimensionType::Signed8:
        return Dimension::Type::Signed8;
    case pdg::DimensionType::Signed16:
        return Dimension::Type::Signed16;
    case pdg::DimensionType::Signed32:
        return Dimension::Type::Signed32;
    case pdg::DimensionType::Signed64:
        return Dimension::Type::Signed64;
    case pdg::DimensionType::Unsigned8:
        return Dimension::Type::Unsigned8;
    case pdg::DimensionType::Unsigned16:
        return Dimension::Type::Unsigned16;
    case pdg::DimensionType::Unsigned32:
        return Dimension::Type::Unsigned32;
    case pdg::DimensionType::Unsigned64:
        return Dimension::Type::Unsigned64;
    case pdg::DimensionType::Float:
        return Dimension::Type::Float;
    case pdg::DimensionType::Double:
        return Dimension::Type::Double;
    case pdg::DimensionType::None:
        return Dimension::Type::None;
    }
    return Dimension::Type::None;
}

std::size_t configuredChunkPoints(std::size_t fallback)
{
    const char* configured = std::getenv("PDG_CUDA_CHUNK_POINTS");
    if (!configured || !*configured)
        return fallback;
    const std::string_view text(configured);
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(
            "PDG_CUDA_CHUNK_POINTS must be a positive integer");
    return value;
}

std::size_t instructionCount(const pdg::AssignProgram& program)
{
    std::size_t result = 0;
    for (const pdg::PointAssignment& assignment : program.assignments)
        result += assignment.value.instructions.size() +
                  assignment.condition.instructions.size();
    return result;
}

std::size_t touchedCount(const pdg::AssignProgram& program)
{
    std::vector<pdg::DimensionId> touched = program.reads;
    for (pdg::DimensionId id : program.writes)
        if (std::find(touched.begin(), touched.end(), id) == touched.end())
            touched.push_back(id);
    return touched.size();
}

bool preferCuda(std::size_t points, const pdg::AssignProgram& program)
{
    return points >= 16'000'000U && program.assignments.size() >= 5U &&
           instructionCount(program) >= 28U && program.writes.size() <= 5U &&
           touchedCount(program) <= 6U;
}
} // unnamed namespace

class PdgPointProgramFilter final : public Filter, public BatchStreamable
{
public:
    std::string getName() const override
    {
        // A one-stage region must retain the public stage name so pipeline
        // metadata and log attribution remain byte-for-byte compatible. The
        // private registration name is used only for genuinely fused regions.
        try
        {
            if (!m_programText.empty())
            {
                const Json stages = Json::parse(m_programText);
                if (stages.is_array() && stages.size() == 1U &&
                    stages.front().is_object() &&
                    stages.front().contains("type") &&
                    stages.front().at("type").is_string())
                    return stages.front().at("type").get<std::string>();
            }
        }
        catch (const Json::exception&)
        {
            // initialize() owns the exact diagnostic for malformed programs.
        }
        return std::string(pdg::HybridPointProgramStage);
    }

private:
    struct Binding
    {
        pdg::DimensionId pdgId;
        Dimension::Id pdalId = Dimension::Id::Unknown;
        Dimension::Type pdalType = Dimension::Type::None;
        pdg::DimensionType physicalType = pdg::DimensionType::None;
        std::size_t pdalOffset = 0;
        bool written = false;
    };

    using Operation =
        std::variant<pdg::AssignProgram, pdg::PredicateProgram,
                     pdg::OrdinalProgram, pdg::TransformationProgram>;
    using OperationStates = std::vector<std::optional<pdg::OrdinalState>>;

    void addArgs(ProgramArgs& args) override
    {
        args.add("program", "Internal fused PDG point program", m_programText);
        args.add("pdg_neighborhood_region_id",
                 "Internal resident neighborhood region identifier",
                 m_neighborhoodRegion, std::uint64_t(0))
            .setHidden();
        args.add("pdg_neighborhood_region_last",
                 "Internal terminal resident neighborhood bridge marker",
                 m_neighborhoodRegionLast, true)
            .setHidden();
        args.add("pdg_plan_cuda", "Internal plan-selected resident CUDA marker",
                 m_planCuda, false)
            .setHidden();
        args.add("pdg_auto_cuda", "Internal automatic CUDA selection marker",
                 m_autoCuda, false)
            .setHidden();
        args.add("pdg_resident_context",
                 "Internal planner-owned resident execution marker",
                 m_residentContext, false)
            .setHidden();
        args.add("pdg_execution_region",
                 "Internal planner-owned execution region identifier",
                 m_executionRegion, std::uint64_t(0))
            .setHidden();
    }

    void initialize() override
    {
        try
        {
            m_stageNodes = Json::parse(m_programText);
        }
        catch (const Json::parse_error& error)
        {
            throwError(std::string("invalid internal point program: ") +
                       error.what());
        }
        if (!m_stageNodes.is_array() || m_stageNodes.empty())
            throwError("internal point program must be a nonempty array");
    }

    void addDimensions(PointLayoutPtr) override
    {
        // Upstream assign registers new destinations from prepared(), while
        // ferry registers them from addDimensions(). A fused region can
        // contain either and can depend on a custom dimension created by an
        // earlier region during prepared(). Defer compilation and binding to
        // prepared() so earlier regions are visible without exposing this
        // region's new dimensions to intervening initialize() metadata.
    }

    void compileProgram(PointLayoutPtr layout)
    {
        m_dimensions = std::make_unique<pdg::DimensionRegistry>();
        for (Dimension::Id id : layout->dims())
        {
            const std::string name = layout->dimName(id);
            const pdg::DimensionDefinition* existing = m_dimensions->find(name);
            if (existing && !existing->standard && existing->name != name)
                throwError("case-distinct proprietary dimensions remain on "
                           "the PDAL path");
            if (!existing)
            {
                const pdg::DimensionType type = toPdgType(layout->dimType(id));
                if (type == pdg::DimensionType::None)
                    throwError("unsupported input dimension type: " + name);
                m_dimensions->registerCustom(name, type);
            }
        }

        Json pipeline = Json::array();
        pipeline.push_back(
            {{"type", "readers.las"}, {"filename", "pdg-input.las"}});
        for (const Json& stage : m_stageNodes)
            pipeline.push_back(stage);
        pipeline.push_back(
            {{"type", "writers.las"}, {"filename", "pdg-output.las"}});

        pdg::Plan plan({}, {});
        try
        {
            plan = pdg::compilePipeline(pipeline.dump(), *m_dimensions);
        }
        catch (const std::exception& error)
        {
            throwError(error.what());
        }

        m_program = {};
        m_operations.clear();
        m_residentOperations.clear();
        m_hasPredicate = false;
        m_hasOrdinal = false;
        m_hasTransformation = false;
        pdg::AssignProgram pendingAssignments;
        const auto flushAssignments = [&]
        {
            if (!pendingAssignments.assignments.empty())
            {
                m_operations.emplace_back(std::move(pendingAssignments));
                pendingAssignments = {};
            }
        };
        const std::vector<pdg::PlannedStage>& stages = plan.stages();
        for (std::size_t index = 1; index + 1U < stages.size(); ++index)
        {
            if (const auto* assign =
                    std::get_if<pdg::AssignProgram>(&stages[index].payload))
            {
                m_residentOperations.emplace_back(*assign);
                pdg::appendAssignments(pendingAssignments, *assign);
                pdg::appendAssignments(m_program, *assign);
            }
            else if (const auto* ferry =
                         std::get_if<pdg::FerryProgram>(&stages[index].payload))
            {
                if (!pdg::ferrySupportsExactPointProgram(*ferry, *m_dimensions))
                    throwError("ferry remains on the exact PDAL host path");
                pdg::AssignProgram residentProgram;
                pdg::appendFerry(residentProgram, *ferry);
                m_residentOperations.emplace_back(std::move(residentProgram));
                pdg::appendFerry(pendingAssignments, *ferry);
                pdg::appendFerry(m_program, *ferry);
            }
            else if (const auto* predicate = std::get_if<pdg::PredicateProgram>(
                         &stages[index].payload))
            {
                flushAssignments();
                m_operations.emplace_back(*predicate);
                m_residentOperations.emplace_back(*predicate);
                m_hasPredicate = true;
            }
            else if (const auto* ordinal = std::get_if<pdg::OrdinalProgram>(
                         &stages[index].payload))
            {
                flushAssignments();
                m_operations.emplace_back(*ordinal);
                m_hasPredicate = true;
                m_hasOrdinal = true;
            }
            else if (const auto* transformation =
                         std::get_if<pdg::TransformationProgram>(
                             &stages[index].payload))
            {
                flushAssignments();
                m_operations.emplace_back(*transformation);
                m_hasTransformation = true;
            }
            else
                throwError("internal point program contains an unsupported "
                           "stage");
        }
        flushAssignments();

        // Upstream assign/ferry stages register newly written dimensions from
        // prepared(), after every stage has completed initialize(). Defer the
        // layout mutation to the same phase so an intervening metadata stage
        // observes the original schema during initialize().
    }

    void prepared(PointTableRef table) override
    {
        compileProgram(table.layout());
        bindDimensions(table.layout());
        if (m_residentContext)
            preparePackedColumns(table.layout());
    }

    void bindDimensions(PointLayoutPtr layout)
    {

        for (pdg::DimensionId id : m_program.writes)
        {
            const pdg::DimensionDefinition& definition =
                m_dimensions->require(id);
            if (layout->findDim(definition.name) == Dimension::Id::Unknown)
                layout->registerOrAssignDim(definition.name,
                                            toPdalType(definition.type));
        }

        m_bindings.clear();
        const auto appendBinding = [&](pdg::DimensionId id, bool written)
        {
            const auto position = std::find_if(
                m_bindings.begin(), m_bindings.end(),
                [&](const Binding& binding) { return binding.pdgId == id; });
            if (position != m_bindings.end())
            {
                position->written = position->written || written;
                return;
            }
            const pdg::DimensionDefinition& definition =
                m_dimensions->require(id);
            const Dimension::Id pdalId = layout->findDim(definition.name);
            if (pdalId == Dimension::Id::Unknown)
                throwError("point program source dimension does not exist: " +
                           definition.name);
            const Dimension::Type pdalType = layout->dimType(pdalId);
            const pdg::DimensionType physicalType = toPdgType(pdalType);
            if (physicalType == pdg::DimensionType::None)
                throwError("unsupported point-program dimension type: " +
                           definition.name);
            m_bindings.push_back(
                {id, pdalId, pdalType, physicalType, 0, written});
        };
        for (pdg::DimensionId id : m_program.reads)
            appendBinding(id, false);
        for (pdg::DimensionId id : m_program.writes)
            appendBinding(id, true);
        for (const Operation& operation : m_operations)
        {
            if (const auto* predicate =
                    std::get_if<pdg::PredicateProgram>(&operation))
                for (pdg::DimensionId id : predicate->reads)
                    appendBinding(id, false);
            else if (std::holds_alternative<pdg::TransformationProgram>(
                         operation))
            {
                for (const pdg::StandardDimension dimension :
                     {pdg::StandardDimension::X, pdg::StandardDimension::Y,
                      pdg::StandardDimension::Z})
                {
                    const pdg::DimensionId id(dimension);
                    appendBinding(id, true);
                    const auto binding =
                        std::find_if(m_bindings.begin(), m_bindings.end(),
                                     [&](const Binding& candidate)
                                     { return candidate.pdgId == id; });
                    if (binding->physicalType != pdg::DimensionType::Double)
                        throwError("transformation requires double XYZ "
                                   "columns");
                }
            }
        }

        m_compactionColumns.clear();
        for (const Binding& binding : m_bindings)
            m_compactionColumns.push_back(binding.pdgId);
        if (m_hasPredicate)
        {
            std::string indexName = "PdgInternalOriginalIndex";
            std::size_t suffix = 0;
            while (m_dimensions->find(indexName))
                indexName =
                    "PdgInternalOriginalIndex" + std::to_string(++suffix);
            m_indexId =
                m_dimensions
                    ->registerCustom(indexName, pdg::DimensionType::Unsigned64)
                    .id;
            m_compactionColumns.push_back(m_indexId);
        }

        const pdg::DimensionId x(pdg::StandardDimension::X);
        const pdg::DimensionId y(pdg::StandardDimension::Y);
        const pdg::DimensionId z(pdg::StandardDimension::Z);
        m_writesCoordinates =
            std::find(m_program.writes.begin(), m_program.writes.end(), x) !=
                m_program.writes.end() ||
            std::find(m_program.writes.begin(), m_program.writes.end(), y) !=
                m_program.writes.end() ||
            std::find(m_program.writes.begin(), m_program.writes.end(), z) !=
                m_program.writes.end() ||
            m_hasTransformation;
        m_hasWrites =
            std::any_of(m_bindings.begin(), m_bindings.end(),
                        [](const Binding& binding) { return binding.written; });
    }

    void preparePackedColumns(PointLayoutPtr layout)
    {
        m_packedColumns.clear();
        m_packedColumns.reserve(m_bindings.size());
        for (Binding& binding : m_bindings)
        {
            binding.pdalOffset = layout->dimOffset(binding.pdalId);
            m_packedColumns.push_back({binding.pdgId, binding.physicalType,
                                       binding.pdalOffset, binding.written});
        }
    }

    void materialize(pdg::PointBatch& batch) const
    {
        for (const Binding& binding : m_bindings)
            batch.materialize(binding.pdgId, binding.physicalType);
        if (m_hasPredicate)
            batch.materialize(m_indexId, pdg::DimensionType::Unsigned64);
    }

    void gather(PointView& view, std::size_t offset, std::size_t count,
                pdg::PointBatch& batch) const
    {
        batch.setSize(count);
        for (const Binding& binding : m_bindings)
        {
            std::byte* destination =
                static_cast<std::byte*>(batch.rawData(binding.pdgId));
            const std::size_t stride =
                pdg::dimensionTypeSize(binding.physicalType);
            for (std::size_t point = 0; point < count; ++point)
                view.getField(
                    reinterpret_cast<char*>(destination + point * stride),
                    binding.pdalId, binding.pdalType, offset + point);
        }
    }

    void gather(StreamPointTable& table, std::span<const PointId> points,
                pdg::PointBatch& batch) const
    {
        batch.setSize(points.size());
        PointRef point(table);
        for (const Binding& binding : m_bindings)
        {
            std::byte* destination =
                static_cast<std::byte*>(batch.rawData(binding.pdgId));
            const std::size_t stride =
                pdg::dimensionTypeSize(binding.physicalType);
            for (std::size_t index = 0; index < points.size(); ++index)
            {
                point.setPointId(points[index]);
                point.getField(
                    reinterpret_cast<char*>(destination + index * stride),
                    binding.pdalId, binding.pdalType);
            }
        }
    }

    void initializeIndexes(pdg::PointBatch& batch, std::size_t count) const
    {
        if (!m_hasPredicate)
            return;
        auto* indexes = batch.data<std::uint64_t>(m_indexId);
        for (std::size_t point = 0; point < count; ++point)
            indexes[point] = point;
    }

    void initializeIndexes(pdg::PointBatch& batch,
                           std::span<const PointId> points) const
    {
        if (!m_hasPredicate)
            return;
        auto* indexes = batch.data<std::uint64_t>(m_indexId);
        for (std::size_t point = 0; point < points.size(); ++point)
            indexes[point] = points[point];
    }

    void scatter(PointView& view, std::size_t offset, std::size_t originalCount,
                 const pdg::PointBatch& batch, PointView* output) const
    {
        const std::uint64_t* indexes =
            m_hasPredicate ? batch.data<std::uint64_t>(m_indexId) : nullptr;
        for (std::size_t point = 0; point < batch.size(); ++point)
        {
            const std::size_t local =
                indexes ? static_cast<std::size_t>(indexes[point]) : point;
            if (local >= originalCount)
                throw std::runtime_error(
                    "compacted point index exceeds its input chunk");
            const PointId viewPoint = static_cast<PointId>(offset + local);
            for (const Binding& binding : m_bindings)
            {
                if (!binding.written)
                    continue;
                const std::byte* source =
                    static_cast<const std::byte*>(batch.rawData(binding.pdgId));
                const std::size_t stride =
                    pdg::dimensionTypeSize(binding.physicalType);
                view.setField(binding.pdalId, binding.pdalType, viewPoint,
                              source + point * stride);
            }
            if (output)
                output->appendPoint(view, viewPoint);
        }
    }

    void scatter(StreamPointTable& table, point_count_t pointLimit,
                 std::span<const PointId> inputPoints,
                 const pdg::PointBatch& batch) const
    {
        const std::uint64_t* indexes =
            m_hasPredicate ? batch.data<std::uint64_t>(m_indexId) : nullptr;
        std::vector<std::uint8_t> selected;
        if (m_hasPredicate)
            selected.resize(static_cast<std::size_t>(pointLimit));
        PointRef point(table);
        for (std::size_t index = 0; index < batch.size(); ++index)
        {
            const PointId pointId = indexes
                                        ? static_cast<PointId>(indexes[index])
                                        : inputPoints[index];
            if (pointId >= pointLimit || table.skip(pointId))
                throw std::runtime_error(
                    "streamed compacted point index is invalid");
            point.setPointId(pointId);
            for (const Binding& binding : m_bindings)
            {
                if (!binding.written)
                    continue;
                const std::byte* source =
                    static_cast<const std::byte*>(batch.rawData(binding.pdgId));
                const std::size_t stride =
                    pdg::dimensionTypeSize(binding.physicalType);
                point.setField(binding.pdalId, binding.pdalType,
                               source + index * stride);
            }
            if (m_hasPredicate)
                selected[static_cast<std::size_t>(pointId)] = 1U;
        }
        if (m_hasPredicate)
            for (PointId pointId : inputPoints)
                if (!selected[static_cast<std::size_t>(pointId)])
                    table.setSkip(pointId);
    }

    OperationStates makeOrdinalStates(pdg::OrdinalMode mode,
                                      std::uint64_t inputTotal) const
    {
        OperationStates states(m_operations.size());
        std::uint64_t currentTotal = inputTotal;
        bool dataDependentCount = false;
        for (std::size_t index = 0; index < m_operations.size(); ++index)
        {
            const Operation& operation = m_operations[index];
            if (std::holds_alternative<pdg::PredicateProgram>(operation))
            {
                dataDependentCount = true;
                continue;
            }
            const auto* ordinal = std::get_if<pdg::OrdinalProgram>(&operation);
            if (!ordinal)
                continue;
            if (mode == pdg::OrdinalMode::Standard && dataDependentCount)
                throw std::invalid_argument(
                    "standard ordinal stage follows a data-dependent "
                    "predicate inside one fused region");
            states[index] = pdg::makeOrdinalState(*ordinal, mode, currentTotal);
            if (mode == pdg::OrdinalMode::Standard)
                currentTotal =
                    pdg::ordinalStandardOutputCount(*ordinal, currentTotal);
        }
        return states;
    }

    void emitTailWarnings(std::uint64_t inputTotal) const
    {
        std::uint64_t currentTotal = inputTotal;
        bool dataDependentCount = false;
        for (const Operation& operation : m_operations)
        {
            if (std::holds_alternative<pdg::PredicateProgram>(operation))
            {
                dataDependentCount = true;
                continue;
            }
            const auto* ordinal = std::get_if<pdg::OrdinalProgram>(&operation);
            if (!ordinal)
                continue;
            if (dataDependentCount)
                throw std::invalid_argument(
                    "tail warning requires a known fused-region point count");
            if (ordinal->kind == pdg::OrdinalKind::Tail &&
                ordinal->count > currentTotal)
                std::cerr << "(pdal pipeline filters.tail Warning) Requested "
                             "number of points (count="
                          << ordinal->count
                          << ") exceeds number of available points.\n";
            currentTotal =
                pdg::ordinalStandardOutputCount(*ordinal, currentTotal);
        }
    }

    pdg::PointBatch* executeHostOperations(pdg::PointBatch& first,
                                           pdg::PointBatch* second,
                                           std::uint8_t* keep,
                                           OperationStates& states) const
    {
        pdg::PointBatch* active = &first;
        pdg::PointBatch* spare = second;
        for (std::size_t index = 0; index < m_operations.size(); ++index)
        {
            const Operation& operation = m_operations[index];
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
                pdg::executeAssign(*active, *assignments);
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
            {
                pdg::evaluatePredicate(*active, *predicate, keep);
                static_cast<void>(pdg::compactPointBatch(
                    *active, *spare, m_compactionColumns, keep));
                std::swap(active, spare);
            }
            else if (const auto* ordinal =
                         std::get_if<pdg::OrdinalProgram>(&operation))
            {
                if (!states[index])
                    throw std::logic_error(
                        "ordinal operation state is missing");
                pdg::evaluateOrdinal(*active, *ordinal, *states[index], keep);
                static_cast<void>(pdg::compactPointBatch(
                    *active, *spare, m_compactionColumns, keep));
                std::swap(active, spare);
            }
            else if (const auto* transformation =
                         std::get_if<pdg::TransformationProgram>(&operation))
                pdg::executeTransformation(*active, *transformation);
            else
                throw std::logic_error("unknown host point operation");
        }
        return active;
    }

    void executeHost(PointView& view, PointView* output,
                     OperationStates& states)
    {
        const std::size_t capacity =
            std::min<std::size_t>(view.size(), 1U << 20U);
        pdg::HostMemoryResource memory;
        pdg::PointBatch first(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, memory);
        materialize(first);
        std::unique_ptr<pdg::PointBatch> second;
        std::vector<std::uint8_t> keep;
        if (m_hasPredicate)
        {
            second = std::make_unique<pdg::PointBatch>(
                capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                *m_dimensions, memory);
            materialize(*second);
            keep.resize(capacity);
        }
        for (std::size_t offset = 0; offset < view.size(); offset += capacity)
        {
            const std::size_t count =
                std::min<std::size_t>(capacity, view.size() - offset);
            gather(view, offset, count, first);
            initializeIndexes(first, count);
            pdg::PointBatch* active = executeHostOperations(
                first, second.get(), keep.empty() ? nullptr : keep.data(),
                states);
            scatter(view, offset, count, *active, output);
        }
    }

#if PDG_HAS_CUDA
    void executeResident(PointView& view, PointView* output,
                         OperationStates& states)
    {
        (void)states;
        if (m_hasOrdinal || m_hasTransformation || m_writesCoordinates)
            throw UnsupportedDeviceProgram();
        if (m_hasPredicate == (output == nullptr))
            throw UnsupportedDeviceProgram();
        pdg_detail::ResidentExecutionContext& context =
            pdg_detail::requireResidentExecutionContext();
        const std::size_t region = static_cast<std::size_t>(m_executionRegion);
        context.beginRegion(view, region, m_packedColumns, output);
        if (m_residentOperations.empty())
            throw UnsupportedDeviceProgram();
        bool checkedEnvelope = false;
        for (std::size_t tile = 0; tile < context.tileCount(); ++tile)
        {
            pdg::PointBatch& batch = context.acquireTile(view, tile);
            for (std::size_t stage = 0; stage < m_residentOperations.size();
                 ++stage)
            {
                context.beginStage(tile, stage);
                if (const auto* program = std::get_if<pdg::AssignProgram>(
                        &m_residentOperations[stage]))
                {
                    if (!checkedEnvelope &&
                        !pdg::assignSupportsExactDevice(batch, *program))
                        throw UnsupportedDeviceProgram();
                    pdg::executeAssign(batch, *program);
                }
                else
                {
                    const auto& predicate = std::get<pdg::PredicateProgram>(
                        m_residentOperations[stage]);
                    if (!checkedEnvelope &&
                        !pdg::predicateSupportsExactDevice(batch, predicate))
                        throw UnsupportedDeviceProgram();
                    pdg::evaluatePredicate(batch, predicate,
                                           context.tileKeepMask(tile));
                }
                context.endStage(tile, stage);
            }
            checkedEnvelope = true;
            context.submitTile(view, tile, batch);
        }
        context.endRegion(view, region);
    }

    pdg::PointBatch* executeDeviceOperations(pdg::PointBatch& first,
                                             pdg::PointBatch* second,
                                             std::uint8_t* keep,
                                             OperationStates& states) const
    {
        pdg::PointBatch* active = &first;
        pdg::PointBatch* spare = second;
        for (std::size_t index = 0; index < m_operations.size(); ++index)
        {
            const Operation& operation = m_operations[index];
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
                pdg::executeAssign(*active, *assignments);
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
            {
                pdg::evaluatePredicate(*active, *predicate, keep);
                static_cast<void>(pdg::compactPointBatch(
                    *active, *spare, m_compactionColumns, keep));
                std::swap(active, spare);
            }
            else if (const auto* ordinal =
                         std::get_if<pdg::OrdinalProgram>(&operation))
            {
                if (!states[index])
                    throw std::logic_error(
                        "ordinal operation state is missing");
                pdg::evaluateOrdinal(*active, *ordinal, *states[index], keep);
                static_cast<void>(pdg::compactPointBatch(
                    *active, *spare, m_compactionColumns, keep));
                std::swap(active, spare);
            }
            else if (const auto* transformation =
                         std::get_if<pdg::TransformationProgram>(&operation))
                pdg::executeTransformation(*active, *transformation);
            else
                throw std::logic_error("unknown device point operation");
        }
        return active;
    }

    void executeCuda(PointView& view, PointView* output,
                     OperationStates& states)
    {
        bool observedRegion = false;
        const std::size_t capacity = std::min<std::size_t>(
            view.size(), configuredChunkPoints(1U << 17U));
        std::unique_ptr<pdg::MemoryResource> pinned =
            pdg::makeCudaPinnedMemoryResource();
        std::unique_ptr<pdg::MemoryResource> device =
            pdg::makeCudaMemoryResource();
        pdg::PointBatch staging(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, *pinned);
        pdg::PointBatch first(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, *device);
        materialize(staging);
        materialize(first);
        std::unique_ptr<pdg::PointBatch> second;
        std::unique_ptr<pdg::Allocation> keep;
        if (m_hasPredicate)
        {
            second = std::make_unique<pdg::PointBatch>(
                capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                *m_dimensions, *device);
            materialize(*second);
            keep = device->allocate(capacity, alignof(std::uint8_t));
        }
        for (const Operation& operation : m_operations)
        {
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
            {
                if (!pdg::assignSupportsExactDevice(first, *assignments))
                    throw UnsupportedDeviceProgram();
            }
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
            {
                if (!pdg::predicateSupportsExactDevice(first, *predicate))
                    throw UnsupportedDeviceProgram();
            }
            else if (const auto* transformation =
                         std::get_if<pdg::TransformationProgram>(&operation))
            {
                if (!pdg::transformationSupportsExactDevice(first,
                                                            *transformation))
                    throw UnsupportedDeviceProgram();
            }
        }
        const cudaStream_t stream =
            static_cast<cudaStream_t>(device->nativeStreamHandle());
        const auto recordTransfer =
            [&](pdg::ExecutionEventKind kind, std::size_t bytes)
        {
            if (!m_planCuda)
                return;
            if (!observedRegion)
            {
                pdg::ExecutionObservationScope::record(
                    pdg::ExecutionEventKind::DeviceRegionBegin,
                    static_cast<std::size_t>(m_executionRegion));
                observedRegion = true;
            }
            pdg::ExecutionObservationScope::record(
                kind, static_cast<std::size_t>(m_executionRegion), bytes);
        };

        for (std::size_t offset = 0; offset < view.size(); offset += capacity)
        {
            const std::size_t count =
                std::min<std::size_t>(capacity, view.size() - offset);
            gather(view, offset, count, staging);
            initializeIndexes(staging, count);
            first.setSize(count);
            for (const Binding& binding : m_bindings)
            {
                const std::size_t bytes =
                    count * pdg::dimensionTypeSize(binding.physicalType);
                PDG_CUDA_CHECK(cudaMemcpyAsync(first.rawData(binding.pdgId),
                                               staging.rawData(binding.pdgId),
                                               bytes, cudaMemcpyHostToDevice,
                                               stream));
                recordTransfer(pdg::ExecutionEventKind::HostToDevice, bytes);
            }
            if (m_hasPredicate)
            {
                PDG_CUDA_CHECK(cudaMemcpyAsync(first.rawData(m_indexId),
                                               staging.rawData(m_indexId),
                                               count * sizeof(std::uint64_t),
                                               cudaMemcpyHostToDevice, stream));
                recordTransfer(pdg::ExecutionEventKind::HostToDevice,
                               count * sizeof(std::uint64_t));
            }

            auto* keepBytes =
                keep ? static_cast<std::uint8_t*>(keep->data()) : nullptr;
            pdg::PointBatch* active =
                executeDeviceOperations(first, second.get(), keepBytes, states);
            staging.setSize(active->size());
            for (const Binding& binding : m_bindings)
            {
                if (!binding.written)
                    continue;
                const std::size_t bytes =
                    active->size() *
                    pdg::dimensionTypeSize(binding.physicalType);
                PDG_CUDA_CHECK(cudaMemcpyAsync(staging.rawData(binding.pdgId),
                                               active->rawData(binding.pdgId),
                                               bytes, cudaMemcpyDeviceToHost,
                                               stream));
                recordTransfer(pdg::ExecutionEventKind::DeviceToHost, bytes);
            }
            if (m_hasPredicate)
            {
                PDG_CUDA_CHECK(cudaMemcpyAsync(
                    staging.rawData(m_indexId), active->rawData(m_indexId),
                    active->size() * sizeof(std::uint64_t),
                    cudaMemcpyDeviceToHost, stream));
                recordTransfer(pdg::ExecutionEventKind::DeviceToHost,
                               active->size() * sizeof(std::uint64_t));
            }
            PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
            scatter(view, offset, count, staging, output);
        }
        if (observedRegion)
            pdg::ExecutionObservationScope::record(
                pdg::ExecutionEventKind::DeviceRegionEnd,
                static_cast<std::size_t>(m_executionRegion));
    }
#endif

    void resetStreamContexts() noexcept
    {
        m_streamPointIdsDevice.reset();
        m_streamPointIdsHost.reset();
        m_streamPackedDevice.reset();
        m_streamPackedHost.reset();
        m_streamDeviceKeep.reset();
        m_streamDeviceSecond.reset();
        m_streamDeviceFirst.reset();
        m_streamStaging.reset();
        m_streamDeviceMemory.reset();
        m_streamPinnedMemory.reset();
        m_streamHostSecond.reset();
        m_streamHostFirst.reset();
        m_streamHostMemory.reset();
        m_streamHostKeep.clear();
        m_streamOrdinalStates.clear();
        m_streamPointStride = 0;
        m_streamReady = false;
        m_streamUseCuda = false;
        m_streamUsePackedCuda = false;
    }

    void prepareHostStream(std::size_t capacity)
    {
        m_streamHostMemory = std::make_unique<pdg::HostMemoryResource>();
        m_streamHostFirst = std::make_unique<pdg::PointBatch>(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, *m_streamHostMemory);
        materialize(*m_streamHostFirst);
        if (m_hasPredicate)
        {
            m_streamHostSecond = std::make_unique<pdg::PointBatch>(
                capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                *m_dimensions, *m_streamHostMemory);
            materialize(*m_streamHostSecond);
            m_streamHostKeep.resize(capacity);
        }
    }

#if PDG_HAS_CUDA
    bool prepareCudaStream(std::size_t capacity, std::size_t pointStride,
                           bool usePackedRecords)
    {
        m_streamPinnedMemory = pdg::makeCudaPinnedMemoryResource();
        m_streamDeviceMemory = pdg::makeCudaMemoryResource();
        m_streamDeviceFirst = std::make_unique<pdg::PointBatch>(
            capacity, pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
            *m_dimensions, *m_streamDeviceMemory);
        materialize(*m_streamDeviceFirst);
        if (usePackedRecords)
        {
            if (!pointStride ||
                capacity >
                    std::numeric_limits<std::size_t>::max() / pointStride)
                throw std::overflow_error(
                    "streaming packed point allocation size overflow");
            const std::size_t packedBytes = capacity * pointStride;
            m_streamPackedHost = m_streamPinnedMemory->allocate(
                packedBytes, alignof(std::max_align_t));
            m_streamPackedDevice = m_streamDeviceMemory->allocate(
                packedBytes, alignof(std::max_align_t));
            m_streamPointIdsHost = m_streamPinnedMemory->allocate(
                capacity * sizeof(std::uint64_t), alignof(std::uint64_t));
            m_streamPointIdsDevice = m_streamDeviceMemory->allocate(
                capacity * sizeof(std::uint64_t), alignof(std::uint64_t));
            m_streamPointStride = pointStride;
        }
        else
        {
            m_streamStaging = std::make_unique<pdg::PointBatch>(
                capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                *m_dimensions, *m_streamPinnedMemory);
            materialize(*m_streamStaging);
        }
        if (m_hasPredicate)
        {
            m_streamDeviceSecond = std::make_unique<pdg::PointBatch>(
                capacity,
                pdg::CoordinateEncoding({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}),
                *m_dimensions, *m_streamDeviceMemory);
            materialize(*m_streamDeviceSecond);
            m_streamDeviceKeep =
                m_streamDeviceMemory->allocate(capacity, alignof(std::uint8_t));
        }
        for (const Operation& operation : m_operations)
        {
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
            {
                if (!pdg::assignSupportsExactDevice(*m_streamDeviceFirst,
                                                    *assignments))
                    return false;
            }
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
            {
                if (!pdg::predicateSupportsExactDevice(*m_streamDeviceFirst,
                                                       *predicate))
                    return false;
            }
            else if (const auto* transformation =
                         std::get_if<pdg::TransformationProgram>(&operation))
            {
                if (!pdg::transformationSupportsExactDevice(
                        *m_streamDeviceFirst, *transformation))
                    return false;
            }
        }
        return true;
    }

    void executeCudaStreamBatch(StreamPointTable& table,
                                point_count_t pointLimit,
                                std::span<const PointId> points)
    {
        pdg::PointBatch& staging = *m_streamStaging;
        pdg::PointBatch& first = *m_streamDeviceFirst;
        gather(table, points, staging);
        initializeIndexes(staging, points);
        first.setSize(points.size());
        const cudaStream_t stream = static_cast<cudaStream_t>(
            m_streamDeviceMemory->nativeStreamHandle());
        for (const Binding& binding : m_bindings)
        {
            const std::size_t bytes =
                points.size() * pdg::dimensionTypeSize(binding.physicalType);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                first.rawData(binding.pdgId), staging.rawData(binding.pdgId),
                bytes, cudaMemcpyHostToDevice, stream));
        }
        if (m_hasPredicate)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                first.rawData(m_indexId), staging.rawData(m_indexId),
                points.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                stream));

        auto* keep =
            m_streamDeviceKeep
                ? static_cast<std::uint8_t*>(m_streamDeviceKeep->data())
                : nullptr;
        pdg::PointBatch* active = executeDeviceOperations(
            first, m_streamDeviceSecond.get(), keep, m_streamOrdinalStates);
        staging.setSize(active->size());
        for (const Binding& binding : m_bindings)
        {
            if (!binding.written)
                continue;
            const std::size_t bytes =
                active->size() * pdg::dimensionTypeSize(binding.physicalType);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                staging.rawData(binding.pdgId), active->rawData(binding.pdgId),
                bytes, cudaMemcpyDeviceToHost, stream));
        }
        if (m_hasPredicate)
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                staging.rawData(m_indexId), active->rawData(m_indexId),
                active->size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));
        scatter(table, pointLimit, points, staging);
    }

    void executeCudaPackedStreamBatch(FixedPointTable& table,
                                      point_count_t pointLimit,
                                      std::span<const PointId> points,
                                      bool sequentialPoints)
    {
        const std::size_t activeCount =
            sequentialPoints ? static_cast<std::size_t>(pointLimit)
                             : points.size();
        if (!activeCount)
            return;
        if (!m_streamPointStride || !m_streamPackedHost ||
            !m_streamPackedDevice || !m_streamPointIdsHost ||
            !m_streamPointIdsDevice)
            throwError("packed streaming CUDA context is incomplete");
        if (static_cast<std::size_t>(pointLimit) >
            std::numeric_limits<std::size_t>::max() / m_streamPointStride)
            throwError("packed streaming byte count overflow");

        const std::size_t packedBytes =
            static_cast<std::size_t>(pointLimit) * m_streamPointStride;
        std::memcpy(m_streamPackedHost->data(), table.data(), packedBytes);

        auto* hostPointIds =
            static_cast<std::uint64_t*>(m_streamPointIdsHost->data());
        const auto* devicePointIds =
            static_cast<const std::uint64_t*>(m_streamPointIdsDevice->data());
        const cudaStream_t stream = static_cast<cudaStream_t>(
            m_streamDeviceMemory->nativeStreamHandle());
        if (!sequentialPoints)
        {
            for (std::size_t index = 0; index < activeCount; ++index)
                hostPointIds[index] = points[index];
            PDG_CUDA_CHECK(cudaMemcpyAsync(m_streamPointIdsDevice->data(),
                                           m_streamPointIdsHost->data(),
                                           activeCount * sizeof(std::uint64_t),
                                           cudaMemcpyHostToDevice, stream));
        }
        else
            devicePointIds = nullptr;

        PDG_CUDA_CHECK(cudaMemcpyAsync(m_streamPackedDevice->data(),
                                       m_streamPackedHost->data(), packedBytes,
                                       cudaMemcpyHostToDevice, stream));

        pdg::PointBatch& first = *m_streamDeviceFirst;
        std::uint64_t* originalIndexes =
            m_hasPredicate ? first.data<std::uint64_t>(m_indexId) : nullptr;
        pdg::unpackPackedPointBatchDevice(
            m_streamPackedDevice->data(), m_streamPointStride, devicePointIds,
            activeCount, m_packedColumns, first, originalIndexes);

        auto* keep =
            m_streamDeviceKeep
                ? static_cast<std::uint8_t*>(m_streamDeviceKeep->data())
                : nullptr;
        pdg::PointBatch* active = executeDeviceOperations(
            first, m_streamDeviceSecond.get(), keep, m_streamOrdinalStates);
        const std::uint64_t* outputPointIds =
            m_hasPredicate ? active->data<std::uint64_t>(m_indexId)
                           : devicePointIds;
        if (m_hasWrites)
        {
            pdg::repackPackedPointBatchDevice(
                *active, m_streamPackedDevice->data(), m_streamPointStride,
                outputPointIds, m_packedColumns);
            PDG_CUDA_CHECK(cudaMemcpyAsync(
                m_streamPackedHost->data(), m_streamPackedDevice->data(),
                packedBytes, cudaMemcpyDeviceToHost, stream));
        }
        if (m_hasPredicate && active->size())
            PDG_CUDA_CHECK(
                cudaMemcpyAsync(m_streamPointIdsHost->data(), outputPointIds,
                                active->size() * sizeof(std::uint64_t),
                                cudaMemcpyDeviceToHost, stream));
        PDG_CUDA_CHECK(cudaStreamSynchronize(stream));

        if (m_hasWrites)
            std::memcpy(table.data(), m_streamPackedHost->data(), packedBytes);
        if (!m_hasPredicate)
            return;

        std::vector<std::uint8_t> selected(
            static_cast<std::size_t>(pointLimit));
        for (std::size_t index = 0; index < active->size(); ++index)
        {
            const std::uint64_t selectedPoint = hostPointIds[index];
            if (selectedPoint >= pointLimit || table.skip(selectedPoint))
                throwError("packed CUDA predicate returned an invalid point "
                           "index");
            selected[static_cast<std::size_t>(selectedPoint)] = 1U;
        }
        if (sequentialPoints)
        {
            for (PointId point = 0; point < pointLimit; ++point)
                if (!selected[static_cast<std::size_t>(point)])
                    table.setSkip(point);
        }
        else
        {
            for (PointId point : points)
                if (!selected[static_cast<std::size_t>(point)])
                    table.setSkip(point);
        }
    }
#endif

    void ready(PointTableRef table) override
    {
        resetStreamContexts();
        auto* streamTable = dynamic_cast<StreamPointTable*>(&table);
        if (!streamTable)
            return;
        if (m_residentContext)
            throwError(
                "planner-owned resident execution does not support streaming");
        m_streamOrdinalStates =
            makeOrdinalStates(pdg::OrdinalMode::Streaming, 0);
        const std::size_t capacity =
            static_cast<std::size_t>(streamTable->capacity());
        if (!capacity)
            throwError("streaming point-program capacity is zero");

        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requireAutomatic =
            std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID");
        const bool requestCuda = requireCuda ||
                                 std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID") ||
                                 m_autoCuda;
#if PDG_HAS_CUDA
        auto* fixedTable = dynamic_cast<FixedPointTable*>(streamTable);
        if (fixedTable)
            preparePackedColumns(table.layout());
        if (requestCuda)
        {
            try
            {
                if (!pdg::cudaDevices().empty() &&
                    prepareCudaStream(
                        capacity, fixedTable ? table.layout()->pointSize() : 0,
                        fixedTable != nullptr))
                {
                    m_streamUseCuda = true;
                    m_streamUsePackedCuda = fixedTable != nullptr;
                }
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
            }
            if (!m_streamUseCuda)
            {
                m_streamDeviceKeep.reset();
                m_streamDeviceSecond.reset();
                m_streamDeviceFirst.reset();
                m_streamStaging.reset();
                m_streamPointIdsDevice.reset();
                m_streamPointIdsHost.reset();
                m_streamPackedDevice.reset();
                m_streamPackedHost.reset();
                m_streamDeviceMemory.reset();
                m_streamPinnedMemory.reset();
                m_streamPointStride = 0;
                m_streamUsePackedCuda = false;
            }
        }
#else
        (void)requestCuda;
#endif
        if (requireCuda && !m_streamUseCuda)
            throwError(
                "required exact CUDA hybrid point-program path was not used");
        if (requireAutomatic && (!m_autoCuda || !m_streamUseCuda))
            throwError(
                "required automatic exact CUDA label/NNDistance hybrid path "
                "was not used");
        if (!m_streamUseCuda)
            prepareHostStream(capacity);
        m_streamReady = true;
    }

    void done(PointTableRef table) override
    {
        (void)table;
        resetStreamContexts();
    }

    void processBatch(StreamPointTable& table,
                      point_count_t pointLimit) override
    {
        if (!m_streamReady)
            throwError("streaming point-program context is not ready");
        bool sequentialPoints = true;
        for (PointId point = 0; point < pointLimit; ++point)
            if (table.skip(point))
            {
                sequentialPoints = false;
                break;
            }

        std::vector<PointId> points;
        if (!sequentialPoints)
        {
            points.reserve(static_cast<std::size_t>(pointLimit));
            for (PointId point = 0; point < pointLimit; ++point)
                if (!table.skip(point))
                    points.push_back(point);
            if (points.empty())
                return;
        }

#if PDG_HAS_CUDA
        if (m_streamUseCuda)
        {
            if (m_streamUsePackedCuda)
            {
                auto* fixedTable = dynamic_cast<FixedPointTable*>(&table);
                if (!fixedTable)
                    throwError("packed CUDA stream table type changed");
                executeCudaPackedStreamBatch(*fixedTable, pointLimit, points,
                                             sequentialPoints);
                return;
            }
            if (sequentialPoints)
            {
                points.reserve(static_cast<std::size_t>(pointLimit));
                for (PointId point = 0; point < pointLimit; ++point)
                    points.push_back(point);
            }
            executeCudaStreamBatch(table, pointLimit, points);
            return;
        }
#endif
        if (sequentialPoints)
        {
            points.reserve(static_cast<std::size_t>(pointLimit));
            for (PointId point = 0; point < pointLimit; ++point)
                points.push_back(point);
        }
        gather(table, points, *m_streamHostFirst);
        initializeIndexes(*m_streamHostFirst, points);
        pdg::PointBatch* active = executeHostOperations(
            *m_streamHostFirst, m_streamHostSecond.get(),
            m_streamHostKeep.empty() ? nullptr : m_streamHostKeep.data(),
            m_streamOrdinalStates);
        scatter(table, pointLimit, points, *active);
    }

    PointViewSet run(PointViewPtr view) override
    {
        if (std::getenv("PDG_REQUIRE_STREAMING_HYBRID"))
            throwError("required streaming hybrid path was not used");
        PointViewSet result;
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_HYBRID");
        const bool requireAutomatic =
            std::getenv("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID");
        const bool requireSelectedCuda = requireCuda || m_planCuda;
        if (view->empty())
        {
#if PDG_HAS_CUDA
            if (m_residentContext && m_neighborhoodRegion)
            {
                if (m_hasPredicate || m_hasTransformation ||
                    m_writesCoordinates)
                    throwError("planner-owned resident neighborhood bridge is "
                               "outside the exact envelope");
                pdg_detail::CudaNeighborhoodRegion neighborhoodRegion;
                neighborhoodRegion.id = m_neighborhoodRegion;
                neighborhoodRegion.last = m_neighborhoodRegionLast;
                if (!pdg_detail::tryCudaResidentAssignments(
                        *view, neighborhoodRegion, m_program,
                        /*requireCuda=*/true))
                    throwError("planner-selected empty resident shared-index "
                               "bridge path was not used");
                if (m_neighborhoodRegionLast)
                    pdg_detail::requireResidentExecutionContext()
                        .endDelegatedRegion(
                            *view, static_cast<std::size_t>(m_executionRegion));
                result.insert(view);
                return result;
            }
#endif
            if (requireSelectedCuda)
                throwError("required exact CUDA hybrid point-program path "
                           "received an empty view");
            if (!m_operations.empty())
            {
                static_cast<void>(makeOrdinalStates(pdg::OrdinalMode::Standard,
                                                    view->size()));
                emitTailWarnings(view->size());
            }
            if (!m_hasPredicate)
                result.insert(view);
            return result;
        }
        if (m_operations.empty())
        {
            if (m_neighborhoodRegion && m_neighborhoodRegionLast)
                pdg_detail::clearCudaNeighborhood(*view);
            result.insert(view);
            return result;
        }

        OperationStates ordinalStates =
            makeOrdinalStates(pdg::OrdinalMode::Standard, view->size());
        emitTailWarnings(view->size());

        const bool forceCuda = requireSelectedCuda || m_autoCuda ||
                               std::getenv("PDG_EXPERIMENTAL_CUDA_HYBRID");
        const bool automaticCuda =
            !m_hasPredicate && !m_hasTransformation &&
            !std::getenv("PDG_DISABLE_CUDA_HYBRID") &&
            (m_autoCuda || preferCuda(view->size(), m_program));
        PointViewPtr output = m_hasPredicate ? view->makeNew() : view;
        bool usedCuda = false;
#if PDG_HAS_CUDA
        pdg_detail::CudaNeighborhoodRegion neighborhoodRegion;
        neighborhoodRegion.id = m_neighborhoodRegion;
        neighborhoodRegion.last = m_neighborhoodRegionLast;
        if (m_residentContext && m_neighborhoodRegion)
        {
            // Planner-selected shared-index bridge: consume resident
            // neighborhood columns in place and close the delegated region at
            // its final stage. The exact CUDA path is mandatory.
            if (m_hasPredicate || m_hasTransformation || m_writesCoordinates)
                throwError("planner-owned resident neighborhood bridge is "
                           "outside the exact envelope");
            usedCuda = pdg_detail::tryCudaResidentAssignments(
                *view, neighborhoodRegion, m_program, /*requireCuda=*/true);
            if (!usedCuda)
                throwError("planner-selected resident shared-index bridge "
                           "path was not used");
            if (m_neighborhoodRegionLast)
                pdg_detail::requireResidentExecutionContext()
                    .endDelegatedRegion(
                        *view, static_cast<std::size_t>(m_executionRegion));
        }
        else if (m_residentContext)
        {
            try
            {
                executeResident(*view, m_hasPredicate ? output.get() : nullptr,
                                ordinalStates);
                usedCuda = true;
            }
            catch (const UnsupportedDeviceProgram&)
            {
                throwError("planner-owned resident point program is outside "
                           "the exact resident CUDA envelope");
            }
        }
        if (!usedCuda && m_neighborhoodRegion && !m_hasPredicate &&
            !m_hasTransformation)
            usedCuda = pdg_detail::tryCudaResidentAssignments(
                *view, neighborhoodRegion, m_program, requireCuda);
        bool deviceAvailable = false;
        if (!usedCuda && (forceCuda || automaticCuda))
        {
            try
            {
                deviceAvailable = !pdg::cudaDevices().empty();
            }
            catch (const pdg::CudaError&)
            {
                if (requireSelectedCuda)
                    throw;
            }
        }
        if (!usedCuda && (forceCuda || automaticCuda) && deviceAvailable)
        {
            try
            {
                executeCuda(*view, m_hasPredicate ? output.get() : nullptr,
                            ordinalStates);
                usedCuda = true;
            }
            catch (const UnsupportedDeviceProgram&)
            {
                if (requireSelectedCuda)
                    throwError("required point program is outside the exact "
                               "CUDA expression envelope");
            }
        }
#endif
        if (m_neighborhoodRegion && !usedCuda)
            pdg_detail::clearCudaNeighborhood(*view);
        if (m_planCuda && !usedCuda)
            throwError(
                "planner-selected exact CUDA hybrid point-program path was "
                "not used");
        if (requireCuda && !usedCuda)
            throwError(
                "required exact CUDA hybrid point-program path was not used");
        if (requireAutomatic && (!m_autoCuda || !usedCuda))
            throwError(
                "required automatic exact CUDA label/NNDistance hybrid path "
                "was not used");
        if (!usedCuda)
        {
            executeHost(*view, m_hasPredicate ? output.get() : nullptr,
                        ordinalStates);
        }
        if (m_writesCoordinates)
            view->invalidateProducts();
        result.insert(output);
        return result;
    }

    std::string m_programText;
    std::uint64_t m_neighborhoodRegion = 0;
    bool m_neighborhoodRegionLast = true;
    bool m_planCuda = false;
    bool m_autoCuda = false;
    bool m_residentContext = false;
    std::uint64_t m_executionRegion = 0;
    Json m_stageNodes;
    std::unique_ptr<pdg::DimensionRegistry> m_dimensions;
    pdg::AssignProgram m_program;
    std::vector<Operation> m_operations;
    std::vector<std::variant<pdg::AssignProgram, pdg::PredicateProgram>>
        m_residentOperations;
    std::vector<Binding> m_bindings;
    std::vector<pdg::DimensionId> m_compactionColumns;
    std::vector<pdg::PackedPointColumn> m_packedColumns;
    pdg::DimensionId m_indexId;
    std::unique_ptr<pdg::HostMemoryResource> m_streamHostMemory;
    std::unique_ptr<pdg::PointBatch> m_streamHostFirst;
    std::unique_ptr<pdg::PointBatch> m_streamHostSecond;
    std::vector<std::uint8_t> m_streamHostKeep;
    OperationStates m_streamOrdinalStates;
    std::unique_ptr<pdg::MemoryResource> m_streamPinnedMemory;
    std::unique_ptr<pdg::MemoryResource> m_streamDeviceMemory;
    std::unique_ptr<pdg::PointBatch> m_streamStaging;
    std::unique_ptr<pdg::PointBatch> m_streamDeviceFirst;
    std::unique_ptr<pdg::PointBatch> m_streamDeviceSecond;
    std::unique_ptr<pdg::Allocation> m_streamDeviceKeep;
    std::unique_ptr<pdg::Allocation> m_streamPackedHost;
    std::unique_ptr<pdg::Allocation> m_streamPackedDevice;
    std::unique_ptr<pdg::Allocation> m_streamPointIdsHost;
    std::unique_ptr<pdg::Allocation> m_streamPointIdsDevice;
    std::size_t m_streamPointStride = 0;
    bool m_hasPredicate = false;
    bool m_hasOrdinal = false;
    bool m_hasTransformation = false;
    bool m_hasWrites = false;
    bool m_writesCoordinates = false;
    bool m_streamReady = false;
    bool m_streamUseCuda = false;
    bool m_streamUsePackedCuda = false;
};

static StaticPluginInfo const s_info{std::string(pdg::HybridPointProgramStage),
                                     "Internal fused PDG point-program region",
                                     ""};

CREATE_STATIC_STAGE(PdgPointProgramFilter, s_info)

} // namespace pdal
