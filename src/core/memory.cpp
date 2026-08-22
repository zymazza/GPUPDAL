#include <pdg/Memory.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace pdg
{

namespace
{
bool isPowerOfTwo(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

bool isDriverVersionTokenCharacter(char value)
{
    return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
           (value >= 'a' && value <= 'z') || value == '.' || value == '_';
}

bool parseUnsignedComponent(std::string_view text, std::size_t& cursor)
{
    const std::size_t begin = cursor;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9')
        ++cursor;
    if (begin == cursor)
        return false;

    unsigned int unused = 0;
    const auto [end, error] =
        std::from_chars(text.data() + begin, text.data() + cursor, unused);
    return error == std::errc{} && end == text.data() + cursor;
}

std::string parseDottedDriverVersion(std::string_view text, std::size_t cursor)
{
    if (cursor != 0U && isDriverVersionTokenCharacter(text[cursor - 1U]))
        return {};

    const std::size_t begin = cursor;
    for (int component = 0; component < 3; ++component)
    {
        if (!parseUnsignedComponent(text, cursor))
            return {};
        if (component != 2)
        {
            if (cursor == text.size() || text[cursor] != '.')
                return {};
            ++cursor;
        }
    }
    if (cursor < text.size() && isDriverVersionTokenCharacter(text[cursor]))
        return {};
    return std::string(text.substr(begin, cursor - begin));
}

class HostAllocation final : public Allocation
{
public:
    HostAllocation(std::size_t bytes, std::size_t alignment)
        : m_size(bytes), m_alignment(alignment)
    {
        if (bytes)
            m_data = ::operator new(bytes, std::align_val_t(alignment));
    }

    ~HostAllocation() override
    {
        if (m_data)
            ::operator delete(m_data, std::align_val_t(m_alignment));
    }

    void* data() noexcept override
    {
        return m_data;
    }

    const void* data() const noexcept override
    {
        return m_data;
    }

    std::size_t size() const noexcept override
    {
        return m_size;
    }

    MemoryKind kind() const noexcept override
    {
        return MemoryKind::Host;
    }

private:
    void* m_data = nullptr;
    std::size_t m_size;
    std::size_t m_alignment;
};
} // unnamed namespace

std::unique_ptr<Allocation> HostMemoryResource::allocate(std::size_t bytes,
                                                         std::size_t alignment)
{
    alignment = std::max(alignment, alignof(std::max_align_t));
    if (!isPowerOfTwo(alignment))
        throw std::invalid_argument(
            "allocation alignment must be a power of two");
    return std::make_unique<HostAllocation>(bytes, alignment);
}

MemoryKind HostMemoryResource::kind() const noexcept
{
    return MemoryKind::Host;
}

void* HostMemoryResource::nativeStreamHandle() const noexcept
{
    return nullptr;
}

std::string formatCudaToolkitVersion(int version)
{
    if (version <= 0)
        return {};
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    const int patch = version % 10;
    if (major <= 0 || minor < 0 || patch < 0)
        return {};

    std::string result = std::to_string(major) + '.' + std::to_string(minor);
    if (patch != 0)
        result += '.' + std::to_string(patch);
    return result;
}

std::string parseNvidiaKernelDriverVersion(std::string_view text)
{
    constexpr std::string_view Marker = "NVRM version:";
    const std::size_t marker = text.find(Marker);
    if (marker == std::string_view::npos)
        return {};

    std::size_t cursor = marker + Marker.size();
    while (cursor < text.size())
    {
        if (text[cursor] >= '0' && text[cursor] <= '9')
        {
            const std::string version = parseDottedDriverVersion(text, cursor);
            if (!version.empty())
                return version;
        }
        ++cursor;
    }
    return {};
}

std::string nvidiaKernelDriverVersion()
{
#ifdef _WIN32
    const HMODULE library = ::LoadLibraryW(L"nvml.dll");
    if (!library)
        return {};
    const auto close = [&] { ::FreeLibrary(library); };
    using Init = int(WINAPI*)();
    using DriverVersion = int(WINAPI*)(char*, unsigned int);
    using Shutdown = int(WINAPI*)();
    const auto init = reinterpret_cast<Init>(
        ::GetProcAddress(library, "nvmlInit_v2"));
    const auto driverVersion = reinterpret_cast<DriverVersion>(
        ::GetProcAddress(library, "nvmlSystemGetDriverVersion"));
    const auto shutdown = reinterpret_cast<Shutdown>(
        ::GetProcAddress(library, "nvmlShutdown"));
    if (!init || !driverVersion || !shutdown || init() != 0)
    {
        close();
        return {};
    }
    char version[96]{};
    const int result = driverVersion(version, sizeof version);
    shutdown();
    close();
    return result == 0 ? std::string(version) : std::string{};
#else
    std::ifstream input("/proc/driver/nvidia/version");
    if (!input)
        return {};
    const std::string content((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
    return parseNvidiaKernelDriverVersion(content);
#endif
}

} // namespace pdg
