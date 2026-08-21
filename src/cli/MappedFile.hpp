#pragma once

#include <cerrno>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
    }

    int m_descriptor = -1;
    void* m_data = nullptr;
    std::size_t m_size = 0;
};

} // namespace pdg::cli
