#include <pdg/PointBatch.hpp>
#include <pdg/stages/Elm.hpp>

#include <limits>
#include <stdexcept>

namespace pdg
{

ElmResult classifyElmDevice(PointBatch&, const ElmProgram&)
{
    throw std::runtime_error("CUDA ELM support is not available");
}

std::size_t elmExactDeviceScratchBytes(std::size_t pointCount)
{
    if (!pointCount)
        return 0U;
    constexpr std::size_t ConservativeBytesPerPoint = 64U;
    if (pointCount > ((std::numeric_limits<std::size_t>::max)() -
                      ElmExactDeviceAllocatorSlackBytes) /
                         ConservativeBytesPerPoint)
        throw std::overflow_error("ELM scratch estimate overflows");
    return pointCount * ConservativeBytesPerPoint +
           ElmExactDeviceAllocatorSlackBytes;
}

} // namespace pdg
