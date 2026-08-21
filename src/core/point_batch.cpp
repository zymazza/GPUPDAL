#include <pdg/PointBatch.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace pdg
{

namespace
{
std::size_t checkedBytes(std::size_t count, DimensionType type)
{
    const std::size_t elementSize = dimensionTypeSize(type);
    if (!elementSize)
        throw std::invalid_argument(
            "point columns require a concrete physical type");
    if (count > std::numeric_limits<std::size_t>::max() / elementSize)
        throw std::overflow_error("point column allocation size overflow");
    return count * elementSize;
}
} // unnamed namespace

PointBatch::PointBatch(std::size_t capacity, CoordinateEncoding coordinates,
                       DimensionRegistry& dimensions, MemoryResource& memory)
    : m_capacity(capacity), m_coordinates(std::move(coordinates)),
      m_dimensions(&dimensions), m_memory(&memory)
{
}

PointBatch::~PointBatch() = default;
PointBatch::PointBatch(PointBatch&&) noexcept = default;
PointBatch& PointBatch::operator=(PointBatch&&) noexcept = default;

std::size_t PointBatch::capacity() const noexcept
{
    return m_capacity;
}

std::size_t PointBatch::size() const noexcept
{
    return m_size;
}

void PointBatch::setSize(std::size_t size)
{
    if (size > m_capacity)
        throw std::out_of_range("point batch size exceeds capacity");
    m_size = size;
}

const CoordinateEncoding& PointBatch::coordinateEncoding() const noexcept
{
    return m_coordinates;
}

MemoryKind PointBatch::memoryKind() const noexcept
{
    return m_memory->kind();
}

void* PointBatch::nativeStreamHandle() const noexcept
{
    return m_memory->nativeStreamHandle();
}

MemoryResource& PointBatch::memoryResource() const noexcept
{
    return *m_memory;
}

bool PointBatch::has(DimensionId id) const noexcept
{
    return m_columns.contains(id.value());
}

bool PointBatch::has(std::string_view name) const
{
    const DimensionDefinition* definition = m_dimensions->find(name);
    return definition && has(definition->id);
}

const ColumnInfo& PointBatch::materialize(DimensionId id)
{
    const DimensionDefinition& definition = m_dimensions->require(id);
    return materialize(id, defaultPhysicalType(definition));
}

const ColumnInfo& PointBatch::materialize(DimensionId id,
                                          DimensionType physicalType)
{
    if (auto position = m_columns.find(id.value()); position != m_columns.end())
    {
        if (position->second.info.physicalType != physicalType)
            throw std::invalid_argument(
                "point column already has a different physical type");
        return position->second.info;
    }

    const DimensionDefinition& definition = m_dimensions->require(id);
    const std::size_t bytes = checkedBytes(m_capacity, physicalType);
    Column column{
        {id, definition.type, physicalType, m_memory->kind(), m_capacity},
        m_memory->allocate(bytes, dimensionTypeSize(physicalType))};
    auto [position, inserted] =
        m_columns.emplace(id.value(), std::move(column));
    if (!inserted)
        throw std::logic_error("failed to materialize point column");
    return position->second.info;
}

void PointBatch::erase(DimensionId id)
{
    m_columns.erase(id.value());
}

const ColumnInfo& PointBatch::columnInfo(DimensionId id) const
{
    return requireColumn(id).info;
}

void* PointBatch::rawData(DimensionId id)
{
    return requireColumn(id).allocation->data();
}

const void* PointBatch::rawData(DimensionId id) const
{
    return requireColumn(id).allocation->data();
}

void PointBatch::materializeGhostMask()
{
    if (!m_ghost)
        m_ghost = m_memory->allocate(m_capacity, alignof(std::uint8_t));
}

bool PointBatch::hasGhostMask() const noexcept
{
    return static_cast<bool>(m_ghost);
}

std::uint8_t* PointBatch::ghostData()
{
    if (!m_ghost)
        throw std::logic_error("ghost mask is not materialized");
    return static_cast<std::uint8_t*>(m_ghost->data());
}

const std::uint8_t* PointBatch::ghostData() const
{
    if (!m_ghost)
        throw std::logic_error("ghost mask is not materialized");
    return static_cast<const std::uint8_t*>(m_ghost->data());
}

std::size_t PointBatch::allocatedBytes() const noexcept
{
    std::size_t bytes = m_ghost ? m_ghost->size() : 0;
    for (const auto& [id, column] : m_columns)
    {
        (void)id;
        bytes += column.allocation->size();
    }
    return bytes;
}

DimensionType
PointBatch::defaultPhysicalType(const DimensionDefinition& definition) noexcept
{
    const DimensionId x(StandardDimension::X);
    const DimensionId y(StandardDimension::Y);
    const DimensionId z(StandardDimension::Z);
    if (definition.id == x || definition.id == y || definition.id == z)
        return DimensionType::Signed32;
    return definition.type;
}

PointBatch::Column& PointBatch::requireColumn(DimensionId id)
{
    const auto position = m_columns.find(id.value());
    if (position == m_columns.end())
        throw std::invalid_argument("point column is not materialized");
    return position->second;
}

const PointBatch::Column& PointBatch::requireColumn(DimensionId id) const
{
    const auto position = m_columns.find(id.value());
    if (position == m_columns.end())
        throw std::invalid_argument("point column is not materialized");
    return position->second;
}

} // namespace pdg
