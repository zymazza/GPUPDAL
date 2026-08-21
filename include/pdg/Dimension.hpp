#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>

namespace pdg
{

enum class DimensionType : std::uint16_t
{
    None = 0x000,
    Signed8 = 0x101,
    Signed16 = 0x102,
    Signed32 = 0x104,
    Signed64 = 0x108,
    Unsigned8 = 0x201,
    Unsigned16 = 0x202,
    Unsigned32 = 0x204,
    Unsigned64 = 0x208,
    Float = 0x404,
    Double = 0x408
};

enum class StandardDimension : std::uint16_t;

class DimensionId
{
public:
    constexpr DimensionId() = default;
    explicit constexpr DimensionId(std::uint32_t value) : m_value(value) {}
    constexpr DimensionId(StandardDimension value);

    [[nodiscard]] constexpr std::uint32_t value() const noexcept
    {
        return m_value;
    }

    friend constexpr bool operator==(DimensionId, DimensionId) = default;
    friend constexpr auto operator<=>(DimensionId, DimensionId) = default;

private:
    std::uint32_t m_value = 0;
};

struct DimensionDefinition
{
    DimensionId id;
    std::string name;
    DimensionType type = DimensionType::None;
    bool standard = false;
};

inline constexpr std::uint32_t CustomDimensionBase = 1024;

[[nodiscard]] constexpr std::size_t dimensionTypeSize(DimensionType type)
{
    return static_cast<std::uint16_t>(type) & 0xffU;
}

[[nodiscard]] constexpr bool isFloating(DimensionType type)
{
    return (static_cast<std::uint16_t>(type) & 0xf00U) == 0x400U;
}

class DimensionRegistry
{
public:
    DimensionRegistry();
    DimensionRegistry(const DimensionRegistry&) = delete;
    DimensionRegistry& operator=(const DimensionRegistry&) = delete;
    DimensionRegistry(DimensionRegistry&&) = delete;
    DimensionRegistry& operator=(DimensionRegistry&&) = delete;

    [[nodiscard]] const DimensionDefinition* find(std::string_view name) const;
    [[nodiscard]] const DimensionDefinition*
    find(DimensionId id) const noexcept;
    [[nodiscard]] const DimensionDefinition&
    require(std::string_view name) const;
    [[nodiscard]] const DimensionDefinition& require(DimensionId id) const;

    const DimensionDefinition& registerCustom(std::string name,
                                              DimensionType type);
    // Replaces the type of an existing definition (standard or custom) in
    // this registry only, keeping its id and name. A resident execution uses
    // it to follow a file whose extra-bytes VLR declares a standard dimension
    // name with another concrete type, exactly as stock PDAL does (D0278).
    // Throws for an unknown id or a non-concrete type.
    const DimensionDefinition& redefineType(DimensionId id, DimensionType type);

    [[nodiscard]] std::size_t standardCount() const noexcept;
    [[nodiscard]] std::size_t customCount() const noexcept;

private:
    static std::string lookupKey(std::string_view name);
    static bool validName(std::string_view name) noexcept;

    std::deque<DimensionDefinition> m_standard;
    std::deque<DimensionDefinition> m_custom;
    std::unordered_map<std::string, const DimensionDefinition*> m_names;
    std::unordered_map<std::uint32_t, const DimensionDefinition*> m_ids;
    std::uint32_t m_nextCustomId = CustomDimensionBase;
};

} // namespace pdg

#include <pdg/generated/Dimensions.hpp>

namespace pdg
{
constexpr DimensionId::DimensionId(StandardDimension value)
    : m_value(static_cast<std::uint16_t>(value))
{
}
} // namespace pdg
