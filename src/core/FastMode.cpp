#include <pdg/FastMode.hpp>
#include <pdg/index/SpatialIndex.hpp>

#include <cstdlib>
#include <string_view>

namespace pdg
{

bool fastModeEnabled() noexcept
{
    const char* marker = std::getenv("PDG_INTERNAL_FAST_MODE");
    return marker && std::string_view(marker) == "1";
}

bool relaxedTieOrder() noexcept
{
    return fastModeEnabled();
}

std::uint8_t knnStatusMask() noexcept
{
    return relaxedTieOrder()
               ? static_cast<std::uint8_t>(0xFFU & ~KnnDistanceTie)
               : static_cast<std::uint8_t>(0xFFU);
}

} // namespace pdg
