#pragma once

#include <pdg/Coordinate.hpp>
#include <pdg/Dimension.hpp>
#include <pdg/Memory.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace pdg
{

template <typename T>
inline constexpr DimensionType DimensionTypeFor = DimensionType::None;
template <>
inline constexpr DimensionType DimensionTypeFor<std::int8_t> =
    DimensionType::Signed8;
template <>
inline constexpr DimensionType DimensionTypeFor<std::int16_t> =
    DimensionType::Signed16;
template <>
inline constexpr DimensionType DimensionTypeFor<std::int32_t> =
    DimensionType::Signed32;
template <>
inline constexpr DimensionType DimensionTypeFor<std::int64_t> =
    DimensionType::Signed64;
template <>
inline constexpr DimensionType DimensionTypeFor<std::uint8_t> =
    DimensionType::Unsigned8;
template <>
inline constexpr DimensionType DimensionTypeFor<std::uint16_t> =
    DimensionType::Unsigned16;
template <>
inline constexpr DimensionType DimensionTypeFor<std::uint32_t> =
    DimensionType::Unsigned32;
template <>
inline constexpr DimensionType DimensionTypeFor<std::uint64_t> =
    DimensionType::Unsigned64;
template <>
inline constexpr DimensionType DimensionTypeFor<float> = DimensionType::Float;
template <>
inline constexpr DimensionType DimensionTypeFor<double> = DimensionType::Double;

struct ColumnInfo
{
    DimensionId id;
    DimensionType logicalType = DimensionType::None;
    DimensionType physicalType = DimensionType::None;
    MemoryKind memoryKind = MemoryKind::Host;
    std::size_t capacity = 0;
};

class PointBatch
{
public:
    PointBatch(std::size_t capacity, CoordinateEncoding coordinates,
               DimensionRegistry& dimensions, MemoryResource& memory);
    ~PointBatch();

    PointBatch(const PointBatch&) = delete;
    PointBatch& operator=(const PointBatch&) = delete;
    PointBatch(PointBatch&&) noexcept;
    PointBatch& operator=(PointBatch&&) noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    void setSize(std::size_t size);

    [[nodiscard]] const CoordinateEncoding& coordinateEncoding() const noexcept;
    [[nodiscard]] MemoryKind memoryKind() const noexcept;
    [[nodiscard]] void* nativeStreamHandle() const noexcept;
    [[nodiscard]] MemoryResource& memoryResource() const noexcept;
    [[nodiscard]] bool has(DimensionId id) const noexcept;
    [[nodiscard]] bool has(std::string_view name) const;

    const ColumnInfo& materialize(DimensionId id);
    const ColumnInfo& materialize(DimensionId id, DimensionType physicalType);
    void erase(DimensionId id);

    [[nodiscard]] const ColumnInfo& columnInfo(DimensionId id) const;
    [[nodiscard]] void* rawData(DimensionId id);
    [[nodiscard]] const void* rawData(DimensionId id) const;

    template <typename T> [[nodiscard]] T* data(DimensionId id)
    {
        checkType<T>(id);
        return static_cast<T*>(rawData(id));
    }

    template <typename T> [[nodiscard]] const T* data(DimensionId id) const
    {
        checkType<T>(id);
        return static_cast<const T*>(rawData(id));
    }

    template <typename T> [[nodiscard]] std::span<T> hostSpan(DimensionId id)
    {
        if (memoryKind() == MemoryKind::Device)
            throw std::logic_error(
                "device columns cannot be exposed as host spans");
        return {data<T>(id), size()};
    }

    void materializeGhostMask();
    [[nodiscard]] bool hasGhostMask() const noexcept;
    [[nodiscard]] std::uint8_t* ghostData();
    [[nodiscard]] const std::uint8_t* ghostData() const;
    [[nodiscard]] std::size_t allocatedBytes() const noexcept;

private:
    struct Column
    {
        ColumnInfo info;
        std::unique_ptr<Allocation> allocation;
    };

    [[nodiscard]] static DimensionType
    defaultPhysicalType(const DimensionDefinition& definition) noexcept;
    [[nodiscard]] Column& requireColumn(DimensionId id);
    [[nodiscard]] const Column& requireColumn(DimensionId id) const;

    template <typename T> void checkType(DimensionId id) const
    {
        static_assert(DimensionTypeFor<std::remove_cv_t<T>> !=
                          DimensionType::None,
                      "unsupported point column type");
        if (columnInfo(id).physicalType !=
            DimensionTypeFor<std::remove_cv_t<T>>)
            throw std::invalid_argument(
                "point column requested with wrong physical type");
    }

    std::size_t m_capacity;
    std::size_t m_size = 0;
    CoordinateEncoding m_coordinates;
    DimensionRegistry* m_dimensions;
    MemoryResource* m_memory;
    std::unordered_map<std::uint32_t, Column> m_columns;
    std::unique_ptr<Allocation> m_ghost;
};

} // namespace pdg
