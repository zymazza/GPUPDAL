#include <pdg/Dimension.hpp>

#include <stdexcept>
#include <utility>

namespace pdg
{

DimensionRegistry::DimensionRegistry()
{
#define PDG_DIMENSION(symbol, value, text, typeName)                           \
    m_standard.push_back(                                                      \
        {DimensionId(value), text, DimensionType::typeName, true});
#define PDG_ALIAS(alias, target)
#include "generated/Dimensions.inc"
#undef PDG_ALIAS
#undef PDG_DIMENSION

    for (const DimensionDefinition& definition : m_standard)
    {
        m_names.emplace(lookupKey(definition.name), &definition);
        m_ids.emplace(definition.id.value(), &definition);
    }

#define PDG_DIMENSION(symbol, value, text, typeName)
#define PDG_ALIAS(alias, target)                                               \
    m_names.emplace(lookupKey(alias),                                          \
                    find(DimensionId(StandardDimension::target)));
#include "generated/Dimensions.inc"
#undef PDG_ALIAS
#undef PDG_DIMENSION
}

const DimensionDefinition* DimensionRegistry::find(std::string_view name) const
{
    const auto position = m_names.find(lookupKey(name));
    return position == m_names.end() ? nullptr : position->second;
}

const DimensionDefinition*
DimensionRegistry::find(DimensionId id) const noexcept
{
    const auto position = m_ids.find(id.value());
    return position == m_ids.end() ? nullptr : position->second;
}

const DimensionDefinition&
DimensionRegistry::require(std::string_view name) const
{
    const DimensionDefinition* definition = find(name);
    if (!definition)
        throw std::invalid_argument("unknown point dimension: " +
                                    std::string(name));
    return *definition;
}

const DimensionDefinition& DimensionRegistry::require(DimensionId id) const
{
    const DimensionDefinition* definition = find(id);
    if (!definition)
        throw std::invalid_argument("unknown point dimension id: " +
                                    std::to_string(id.value()));
    return *definition;
}

const DimensionDefinition& DimensionRegistry::registerCustom(std::string name,
                                                             DimensionType type)
{
    if (!validName(name))
        throw std::invalid_argument("invalid custom dimension name: " + name);
    if (type == DimensionType::None || dimensionTypeSize(type) == 0)
        throw std::invalid_argument(
            "custom dimension requires a concrete type");
    if (find(name))
        throw std::invalid_argument("duplicate point dimension name: " + name);

    const DimensionId id(m_nextCustomId++);
    m_custom.push_back({id, std::move(name), type, false});
    const DimensionDefinition& definition = m_custom.back();
    m_names.emplace(lookupKey(definition.name), &definition);
    m_ids.emplace(id.value(), &definition);
    return definition;
}

const DimensionDefinition& DimensionRegistry::redefineType(DimensionId id,
                                                          DimensionType type)
{
    if (type == DimensionType::None || dimensionTypeSize(type) == 0)
        throw std::invalid_argument(
            "dimension redefinition requires a concrete type");
    const auto position = m_ids.find(id.value());
    if (position == m_ids.end())
        throw std::invalid_argument("dimension redefinition of an unknown id");
    // Definitions live in the deques; the maps hold pointers to them, so an
    // in-place mutation is visible through every lookup.
    auto* definition = const_cast<DimensionDefinition*>(position->second);
    definition->type = type;
    return *definition;
}

std::size_t DimensionRegistry::standardCount() const noexcept
{
    return m_standard.size();
}

std::size_t DimensionRegistry::customCount() const noexcept
{
    return m_custom.size();
}

std::string DimensionRegistry::lookupKey(std::string_view name)
{
    std::string key;
    key.reserve(name.size());
    for (const char character : name)
    {
        if (character >= 'a' && character <= 'z')
            key.push_back(static_cast<char>(character - ('a' - 'A')));
        else
            key.push_back(character);
    }
    return key;
}

bool DimensionRegistry::validName(std::string_view name) noexcept
{
    if (name.empty() || !((name.front() >= 'A' && name.front() <= 'Z') ||
                          (name.front() >= 'a' && name.front() <= 'z')))
        return false;
    for (const char character : name.substr(1))
    {
        const bool letter = (character >= 'A' && character <= 'Z') ||
                            (character >= 'a' && character <= 'z');
        const bool digit = character >= '0' && character <= '9';
        if (!letter && !digit && character != '_')
            return false;
    }
    return true;
}

} // namespace pdg
