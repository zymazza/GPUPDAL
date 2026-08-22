#pragma once

#include <cerrno>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pdg::cli
{

// Read-only regular-file mapping shared by the native LAS executors. Keeping
// this in one RAII type also keeps the direct resident source under the same
// no-follow and address-space checks as the established translation path.
class MappedInput
{
public:
    explicit MappedInput(const std::filesystem::path& path)
    {
#ifdef _WIN32
        m_file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL |
                                   FILE_FLAG_OPEN_REPARSE_POINT,
                               nullptr);
        if (m_file == INVALID_HANDLE_VALUE)
            throwWindowsError("unable to open native LAS input");
        try
        {
            FILE_ATTRIBUTE_TAG_INFO attributes{};
            if (!::GetFileInformationByHandleEx(
                    m_file, FileAttributeTagInfo, &attributes,
                    static_cast<DWORD>(sizeof attributes)))
                throwWindowsError("unable to stat native LAS input");
            if (::GetFileType(m_file) != FILE_TYPE_DISK ||
                (attributes.FileAttributes &
                 (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)))
                throw std::runtime_error(
                    "native LAS input is not a regular file");
            LARGE_INTEGER size{};
            if (!::GetFileSizeEx(m_file, &size))
                throwWindowsError("unable to stat native LAS input");
            if (size.QuadPart < 0 ||
                static_cast<unsigned long long>(size.QuadPart) >
                    std::numeric_limits<std::size_t>::max())
                throw std::overflow_error(
                    "native LAS input exceeds the host address space");
            m_size = static_cast<std::size_t>(size.QuadPart);
            if (m_size)
            {
                m_mapping = ::CreateFileMappingW(
                    m_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (!m_mapping)
                    throwWindowsError("unable to map native LAS input");
                m_data = ::MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
                if (!m_data)
                    throwWindowsError("unable to map native LAS input");
            }
        }
        catch (...)
        {
            reset();
            throw;
        }
#else
        m_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (m_descriptor < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to open native LAS input");
        try
        {
            struct stat status = {};
            if (::fstat(m_descriptor, &status) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "unable to stat native LAS input");
            if (!S_ISREG(status.st_mode) || status.st_size < 0)
                throw std::runtime_error(
                    "native LAS input is not a regular file");
            const std::uintmax_t fileBytes =
                static_cast<std::uintmax_t>(status.st_size);
            if (fileBytes > std::numeric_limits<std::size_t>::max())
                throw std::overflow_error(
                    "native LAS input exceeds the host address space");
            m_size = static_cast<std::size_t>(fileBytes);
            if (m_size)
            {
                m_data = ::mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE,
                                m_descriptor, 0);
                if (m_data == MAP_FAILED)
                {
                    m_data = nullptr;
                    throw std::system_error(errno, std::generic_category(),
                                            "unable to map native LAS input");
                }
            }
        }
        catch (...)
        {
            reset();
            throw;
        }
#endif
    }

    MappedInput(const MappedInput&) = delete;
    MappedInput& operator=(const MappedInput&) = delete;

    ~MappedInput()
    {
        reset();
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return {static_cast<const std::byte*>(m_data), m_size};
    }

private:
    void reset() noexcept
    {
#ifdef _WIN32
        if (m_data)
        {
            ::UnmapViewOfFile(m_data);
            m_data = nullptr;
        }
        if (m_mapping)
        {
            ::CloseHandle(m_mapping);
            m_mapping = nullptr;
        }
        if (m_file != INVALID_HANDLE_VALUE)
        {
            ::CloseHandle(m_file);
            m_file = INVALID_HANDLE_VALUE;
        }
#else
        if (m_data)
        {
            ::munmap(m_data, m_size);
            m_data = nullptr;
        }
        if (m_descriptor >= 0)
        {
            ::close(m_descriptor);
            m_descriptor = -1;
        }
#endif
    }

#ifdef _WIN32
    [[noreturn]] static void throwWindowsError(const char* message)
    {
        throw std::system_error(static_cast<int>(::GetLastError()),
                                std::system_category(), message);
    }

    HANDLE m_file = INVALID_HANDLE_VALUE;
    HANDLE m_mapping = nullptr;
#else
    int m_descriptor = -1;
#endif
    void* m_data = nullptr;
    std::size_t m_size = 0;
};

} // namespace pdg::cli
