#include "../pdal/PdgNeighborhood.hpp"
#include "DirectResidentLas.hpp"
#include "Calibrate.hpp"
#include "HybridPipeline.hpp"
#include "MappedFile.hpp"
#include "ResidentPipeline.hpp"
#include <pdg/ExecutionStats.hpp>
#include <pdg/LocalProfile.hpp>
#include <pdg/Memory.hpp>
#include <pdg/Placement.hpp>
#include <pdg/Plan.hpp>
#include <pdg/Scheduler.hpp>
#include <pdg/Version.hpp>
#include <pdg/io/LasFerry.hpp>
#include <pdg/io/LasPointProgram.hpp>
#include <pdg/io/LasTranslate.hpp>
#if PDG_HAS_CUDA
#include <pdg/io/LasTranslateCuda.hpp>
#endif

#include <pdal/Dimension.hpp>
#include <pdal/PointLayout.hpp>
#include <pdal/PointView.hpp>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
using pdg::cli::MappedInput;

void version()
{
    std::cout << "gpupdal " << pdg::Version << '\n'
              << "oracle-pdal " << pdg::OracleCommit << '\n'
              << "cuda "
              << (pdg::cudaBackendCompiled() ? "enabled" : "disabled") << '\n';
}

int doctor()
{
    version();
    if (!pdg::cudaBackendCompiled())
        return 0;
    try
    {
        const auto devices = pdg::cudaDevices();
        const auto printCudaVersion = [](std::string_view label, int version)
        {
            std::cout << label << ' ' << version / 1000 << '.'
                      << (version % 1000) / 10 << '\n';
        };
        printCudaVersion("cuda_toolkit", pdg::cudaCompiledToolkitVersion());
        printCudaVersion("cuda_runtime", pdg::cudaRuntimeVersion());
        printCudaVersion("cuda_driver", pdg::cudaDriverVersion());
        std::cout << "devices " << devices.size() << '\n';
        for (const auto& device : devices)
            std::cout << device.ordinal << ' ' << device.name << " sm_"
                      << device.computeMajor << device.computeMinor << ' '
                      << device.totalMemory << " memory_pools "
                      << (device.memoryPoolsSupported ? "supported"
                                                      : "unsupported")
                      << '\n';
        if (!devices.empty())
        {
            // D0277: which placement profile automatic selection would use.
            const pdg::CudaDeviceSummary& device = devices.front();
            const std::string capability =
                std::to_string(device.computeMajor) + "." +
                std::to_string(device.computeMinor);
            const std::string driver = pdg::nvidiaKernelDriverVersion();
            const std::string toolkit = pdg::formatCudaToolkitVersion(
                pdg::cudaCompiledToolkitVersion());
            const pdg::PlacementCalibrationProfile* profile =
                pdg::placementCalibrationFor(
                    {.name = device.name,
                     .computeCapability = capability,
                     .driverVersion = driver,
                     .cudaToolkitVersion = toolkit});
            const pdg::LocalProfileLookup& local = pdg::loadedLocalProfile();
            std::cout << "placement_profile "
                      << (profile ? std::string(profile->id) + " (" +
                                        std::string(pdg::placementProfileTier(
                                            profile)) +
                                        ")"
                                  : std::string("none (host path; run `gpupdal "
                                                "calibrate`)"))
                      << '\n'
                      << "local_profile "
                      << pdg::localProfileStatusName(local.status) << ' '
                      << local.path.string() << '\n';
        }
        return devices.empty() ? 2 : 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "cuda unavailable: " << error.what() << '\n';
        return 2;
    }
}

bool hasLasExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value)
                   { return static_cast<char>(std::tolower(value)); });
    return extension == ".las";
}

std::size_t nativeWorkerLimit()
{
    const char* configured = std::getenv("PDG_NATIVE_WORKERS");
    if (!configured || !*configured)
        return 0;
    const std::string_view text(configured);
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(
            "PDG_NATIVE_WORKERS must be a positive integer");
    return value;
}

std::size_t cudaChunkPoints(std::size_t defaultValue)
{
    const char* configured = std::getenv("PDG_CUDA_CHUNK_POINTS");
    if (!configured || !*configured)
        return defaultValue;
    const std::string_view text(configured);
    std::size_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(
            "PDG_CUDA_CHUNK_POINTS must be a positive integer");
    return value;
}

std::size_t cudaSchedulerLanes()
{
    const char* configured = std::getenv("PDG_CUDA_SCHEDULER_LANES");
    if (!configured || !*configured)
        return 0U;
    try
    {
        return pdg::parseSchedulerLaneCount(configured);
    }
    catch (const std::invalid_argument&)
    {
        throw std::invalid_argument(
            "PDG_CUDA_SCHEDULER_LANES must be an integer in [2, 6]");
    }
}

pdg::las::DefaultTranslationMetadata translationMetadata()
{
    std::time_t now;
    std::time(&now);
    std::uint16_t year = 1900;
    std::uint16_t dayOfYear = 1;
    if (const std::tm* utc = std::gmtime(&now))
    {
        year = static_cast<std::uint16_t>(year + utc->tm_year);
        dayOfYear = static_cast<std::uint16_t>(dayOfYear + utc->tm_yday);
    }
    return {dayOfYear, year, pdg::las::oracleSoftwareId()};
}

class MappedOutput
{
public:
    MappedOutput(const std::filesystem::path& path, std::size_t size,
                 bool& created)
        : m_size(size)
    {
        created = false;
        if (size > static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
            throw std::overflow_error(
                "native LAS output exceeds the file offset domain");
        m_descriptor =
            ::open(path.c_str(),
                   O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0666);
        if (m_descriptor < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to create native LAS output");
        created = true;
        try
        {
            if (::ftruncate(m_descriptor, static_cast<off_t>(size)) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "unable to size native LAS output");
            m_data = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                            m_descriptor, 0);
            if (m_data == MAP_FAILED)
            {
                m_data = nullptr;
                throw std::system_error(errno, std::generic_category(),
                                        "unable to map native LAS output");
            }
        }
        catch (...)
        {
            reset();
            throw;
        }
    }

    MappedOutput(const MappedOutput&) = delete;
    MappedOutput& operator=(const MappedOutput&) = delete;

    ~MappedOutput()
    {
        reset();
    }

    [[nodiscard]] std::span<std::byte> bytes() noexcept
    {
        return {static_cast<std::byte*>(m_data), m_size};
    }

    void close()
    {
        int failure = 0;
        if (m_data)
        {
            if (::munmap(m_data, m_size) != 0)
                failure = errno;
            m_data = nullptr;
        }
        if (m_descriptor >= 0)
        {
            if (::close(m_descriptor) != 0 && !failure)
                failure = errno;
            m_descriptor = -1;
        }
        if (failure)
            throw std::system_error(failure, std::generic_category(),
                                    "unable to close native LAS output");
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
    std::size_t m_size;
};

class PositionedOutput
{
public:
    PositionedOutput(const std::filesystem::path& path, std::size_t size,
                     bool& created)
        : m_size(size), m_freeBuffers(QueueDepth)
    {
        created = false;
        if (size > static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
            throw std::overflow_error(
                "native LAS output exceeds the file offset domain");
        m_descriptor =
            ::open(path.c_str(),
                   O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0666);
        if (m_descriptor < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to create native LAS output");
        created = true;
        try
        {
            if (::ftruncate(m_descriptor, static_cast<off_t>(size)) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "unable to size native LAS output");
            m_worker = std::thread([this] { run(); });
        }
        catch (...)
        {
            reset();
            throw;
        }
    }

    PositionedOutput(const PositionedOutput&) = delete;
    PositionedOutput& operator=(const PositionedOutput&) = delete;

    ~PositionedOutput()
    {
        reset();
    }

    void writeAt(std::size_t offset, std::span<const std::byte> bytes)
    {
        if (offset > m_size || bytes.size() > m_size - offset)
            throw std::out_of_range(
                "native LAS positioned write exceeds output size");

        std::vector<std::byte> buffer;
        {
            std::unique_lock lock(m_mutex);
            m_bufferAvailable.wait(
                lock, [this]
                { return m_failure || m_stopping || !m_freeBuffers.empty(); });
            rethrowFailure();
            if (m_stopping)
                throw std::logic_error("native LAS output is closed");
            buffer = std::move(m_freeBuffers.back());
            m_freeBuffers.pop_back();
        }
        try
        {
            buffer.resize(bytes.size());
            std::memcpy(buffer.data(), bytes.data(), bytes.size());
        }
        catch (...)
        {
            recycle(std::move(buffer));
            throw;
        }

        {
            std::lock_guard lock(m_mutex);
            if (m_failure)
            {
                m_freeBuffers.push_back(std::move(buffer));
                rethrowFailure();
            }
            m_pending.push_back({offset, std::move(buffer)});
        }
        m_writeReady.notify_one();
    }

    void close()
    {
        close(m_size);
    }

    void close(std::size_t finalSize)
    {
        if (finalSize > m_size ||
            finalSize >
                static_cast<std::size_t>(std::numeric_limits<off_t>::max()))
            throw std::out_of_range("native LAS final output size is invalid");
        stopWorker();
        std::exception_ptr failure;
        {
            std::lock_guard lock(m_mutex);
            failure = m_failure;
        }
        int closeFailure = 0;
        if (!failure && m_descriptor >= 0 &&
            ::ftruncate(m_descriptor, static_cast<off_t>(finalSize)) != 0)
            closeFailure = errno;
        if (m_descriptor >= 0 && ::close(m_descriptor) != 0)
            if (!closeFailure)
                closeFailure = errno;
        m_descriptor = -1;
        if (failure)
            std::rethrow_exception(failure);
        if (closeFailure)
        {
            throw std::system_error(closeFailure, std::generic_category(),
                                    "unable to close native LAS output");
        }
    }

private:
    static constexpr std::size_t QueueDepth = 4;

    struct Request
    {
        std::size_t offset = 0;
        std::vector<std::byte> bytes;
    };

    void writeDirect(const Request& request)
    {
        std::size_t written = 0;
        while (written < request.bytes.size())
        {
            const std::size_t remaining = request.bytes.size() - written;
            const std::size_t chunk = std::min(
                remaining,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t result =
                ::pwrite(m_descriptor, request.bytes.data() + written, chunk,
                         static_cast<off_t>(request.offset + written));
            if (result < 0 && errno == EINTR)
                continue;
            if (result <= 0)
                throw std::system_error(result < 0 ? errno : EIO,
                                        std::generic_category(),
                                        "unable to write native LAS output");
            written += static_cast<std::size_t>(result);
        }
    }

    void run() noexcept
    {
        for (;;)
        {
            Request request;
            {
                std::unique_lock lock(m_mutex);
                m_writeReady.wait(lock, [this]
                                  { return m_stopping || !m_pending.empty(); });
                if (m_pending.empty())
                    return;
                request = std::move(m_pending.front());
                m_pending.pop_front();
            }
            try
            {
                writeDirect(request);
            }
            catch (...)
            {
                std::lock_guard lock(m_mutex);
                m_failure = std::current_exception();
                m_stopping = true;
                m_bufferAvailable.notify_all();
                return;
            }
            recycle(std::move(request.bytes));
        }
    }

    void recycle(std::vector<std::byte> buffer) noexcept
    {
        {
            std::lock_guard lock(m_mutex);
            m_freeBuffers.push_back(std::move(buffer));
        }
        m_bufferAvailable.notify_one();
    }

    void rethrowFailure() const
    {
        if (m_failure)
            std::rethrow_exception(m_failure);
    }

    void stopWorker() noexcept
    {
        {
            std::lock_guard lock(m_mutex);
            m_stopping = true;
        }
        m_writeReady.notify_one();
        m_bufferAvailable.notify_all();
        if (m_worker.joinable())
            m_worker.join();
    }

    void reset() noexcept
    {
        stopWorker();
        if (m_descriptor >= 0)
        {
            ::close(m_descriptor);
            m_descriptor = -1;
        }
    }

    int m_descriptor = -1;
    std::size_t m_size;
    std::mutex m_mutex;
    std::condition_variable m_writeReady;
    std::condition_variable m_bufferAvailable;
    std::deque<Request> m_pending;
    std::vector<std::vector<std::byte>> m_freeBuffers;
    std::thread m_worker;
    std::exception_ptr m_failure;
    bool m_stopping = false;
};

bool writeFileExclusive(const std::filesystem::path& path,
                        std::span<const std::byte> bytes, bool& created)
{
    created = false;
    int descriptor =
        ::open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0666);
    if (descriptor < 0)
        return false;
    created = true;
    std::size_t written = 0;
    try
    {
        while (written < bytes.size())
        {
            const std::size_t remaining = bytes.size() - written;
            const std::size_t chunk = std::min(
                remaining,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t result =
                ::write(descriptor, bytes.data() + written, chunk);
            if (result < 0 && errno == EINTR)
                continue;
            if (result <= 0)
                throw std::system_error(result < 0 ? errno : EIO,
                                        std::generic_category(),
                                        "unable to write native LAS output");
            written += static_cast<std::size_t>(result);
        }
        const int closeResult = ::close(descriptor);
        descriptor = -1;
        if (closeResult != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to close native LAS output");
    }
    catch (...)
    {
        if (descriptor >= 0)
            ::close(descriptor);
        throw;
    }
    return true;
}

bool declaresWholeProgramFusion(const pdg::Plan& plan)
{
    const std::vector<pdg::PlannedStage>& stages = plan.stages();
    if (stages.size() < 3U)
        return false;
    std::vector<std::size_t> middleStages;
    middleStages.reserve(stages.size() - 2U);
    for (std::size_t index = 1U; index + 1U < stages.size(); ++index)
        middleStages.push_back(index);
    return std::any_of(
        plan.summary().fusionCandidates.begin(),
        plan.summary().fusionCandidates.end(),
        [&](const pdg::FusionCandidate& candidate)
        {
            const bool surroundingAnchor =
                (candidate.anchorStage == 0U &&
                 candidate.placement ==
                     pdg::FusionPlacement::ProducerEpilogue) ||
                (candidate.anchorStage + 1U == stages.size() &&
                 candidate.placement == pdg::FusionPlacement::ConsumerPrologue);
            return surroundingAnchor && candidate.pointStages == middleStages;
        });
}

bool tryNativeTranslate(int argc, char** argv)
{
    if (argc != 4 || std::string_view(argv[1]) != "translate" ||
        std::getenv("PDG_DISABLE_NATIVE"))
        return false;
#if !PDG_HAS_CUDA
    if (std::getenv("PDG_REQUIRE_CUDA_TRANSLATE"))
        return false;
#endif

    const std::filesystem::path inputPath(argv[2]);
    const std::filesystem::path outputPath(argv[3]);
    if (!hasLasExtension(inputPath) || !hasLasExtension(outputPath))
        return false;
    std::error_code error;
    if (std::filesystem::exists(outputPath, error) || error)
        return false;

    std::filesystem::path temporaryPath;
    bool temporaryOwned = false;
    try
    {
        const MappedInput mappedInput(inputPath);
        const pdg::las::FileView input(mappedInput.bytes());
        if (!pdg::las::supportsDefaultTranslation(input))
            return false;

        const pdg::las::DefaultTranslationMetadata metadata =
            translationMetadata();
        std::vector<std::byte> output;
#if PDG_HAS_CUDA
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_TRANSLATE");
        if ((std::getenv("PDG_EXPERIMENTAL_CUDA_TRANSLATE") || requireCuda) &&
            pdg::las::supportsDefaultCudaTranslation(input))
        {
            try
            {
                output = pdg::las::translateDefaultCuda(
                    input, metadata, cudaChunkPoints(1U << 20U),
                    cudaSchedulerLanes());
            }
            catch (const pdg::CudaError&)
            {
                if (requireCuda)
                    throw;
                // Until the GPU performance gate is decided, an unavailable
                // device or runtime falls through to the exact native host
                // path.
            }
        }
        if (requireCuda && output.empty())
            throw pdg::las::Error(
                "requested input is outside the exact CUDA envelope");
#endif
        temporaryPath = outputPath;
        temporaryPath += ".pdg-native-" + std::to_string(::getpid());
        if (std::filesystem::exists(temporaryPath, error) || error)
            return false;
        if (output.empty())
        {
            MappedOutput mappedOutput(temporaryPath,
                                      pdg::las::defaultTranslationSize(input),
                                      temporaryOwned);
            pdg::las::translateDefaultInto(
                input, metadata, mappedOutput.bytes(), nativeWorkerLimit());
            mappedOutput.close();
        }
        else if (!writeFileExclusive(temporaryPath, output, temporaryOwned))
            return false;
        if (::link(temporaryPath.c_str(), outputPath.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to publish native LAS output");
        if (::unlink(temporaryPath.c_str()) == 0)
            temporaryOwned = false;
        return true;
    }
    catch (...)
    {
        if (temporaryOwned)
        {
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
        return false;
    }
}

bool tryNativePipeline(int argc, char** argv)
{
    if (argc != 3 || std::string_view(argv[1]) != "pipeline" ||
        std::getenv("PDG_DISABLE_NATIVE"))
        return false;
#if !PDG_HAS_CUDA
    if (std::getenv("PDG_REQUIRE_CUDA_POINT_PROGRAM") ||
        std::getenv("PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM"))
        return false;
#endif

    const std::filesystem::path pipelinePath(argv[2]);
    std::error_code error;
    constexpr std::uintmax_t MaximumPipelineBytes = 16U * 1024U * 1024U;
    if (!std::filesystem::is_regular_file(pipelinePath, error) || error ||
        std::filesystem::file_size(pipelinePath, error) >
            MaximumPipelineBytes ||
        error)
        return false;
    std::ifstream pipelineInput(pipelinePath, std::ios::binary);
    if (!pipelineInput)
        return false;
    const std::string pipelineText(
        (std::istreambuf_iterator<char>(pipelineInput)),
        std::istreambuf_iterator<char>());
    if (pipelineInput.bad())
        return false;

    std::filesystem::path temporaryPath;
    bool temporaryOwned = false;
    try
    {
        pdg::DimensionRegistry dimensions;
        const pdg::Plan plan = pdg::compilePipeline(pipelineText, dimensions);
        const auto& stages = plan.stages();
        if (!plan.summary().allStagesNative || stages.size() < 3 ||
            stages[0].role != pdg::StageRole::Reader ||
            stages[0].descriptor.type != "readers.las" ||
            !stages[0].inputs.empty() ||
            stages.back().role != pdg::StageRole::Writer ||
            stages.back().descriptor.type != "writers.las")
            return false;
        for (std::size_t index = 1; index < stages.size(); ++index)
        {
            if (stages[index].inputs != std::vector<std::size_t>{index - 1U})
                return false;
            if (index + 1U < stages.size() &&
                (stages[index].role != pdg::StageRole::Filter ||
                 (stages[index].descriptor.type != "filters.ferry" &&
                  stages[index].descriptor.type != "filters.assign" &&
                  stages[index].descriptor.type != "filters.expression" &&
                  stages[index].descriptor.type != "filters.range" &&
                  stages[index].descriptor.type != "filters.crop" &&
                  stages[index].descriptor.type != "filters.decimation" &&
                  stages[index].descriptor.type != "filters.head" &&
                  stages[index].descriptor.type != "filters.tail")))
                return false;
        }
        const auto* reader =
            std::get_if<pdg::FileStagePlan>(&stages[0].payload);
        const auto* writer =
            std::get_if<pdg::FileStagePlan>(&stages.back().payload);
        if (!reader || !writer)
            return false;

        const bool declaredWholeProgramFusion =
            declaresWholeProgramFusion(plan);

        const std::filesystem::path inputPath(reader->filename);
        const std::filesystem::path outputPath(writer->filename);
        if (!hasLasExtension(inputPath) || !hasLasExtension(outputPath) ||
            std::filesystem::exists(outputPath, error) || error)
            return false;

        const MappedInput mappedInput(inputPath);
        const pdg::las::FileView input(mappedInput.bytes());
        pdg::AssignProgram fusedProgram;
        pdg::AssignProgram pendingAssignments;
        pdg::las::OrderedPointProgram orderedProgram;
        bool hasOrdinal = false;
        bool hasTail = false;
        const auto flushAssignments = [&]
        {
            if (!pendingAssignments.assignments.empty())
            {
                orderedProgram.operations.emplace_back(
                    std::move(pendingAssignments));
                pendingAssignments = {};
            }
        };
        for (std::size_t index = 1; index + 1U < stages.size(); ++index)
        {
            if (const auto* ferry =
                    std::get_if<pdg::FerryProgram>(&stages[index].payload))
            {
                if (!pdg::las::supportsDefaultFerry(input, *ferry, dimensions))
                    return false;
                pdg::appendFerry(pendingAssignments, *ferry);
                pdg::appendFerry(fusedProgram, *ferry);
            }
            else if (const auto* assignment = std::get_if<pdg::AssignProgram>(
                         &stages[index].payload))
            {
                pdg::appendAssignments(pendingAssignments, *assignment);
                pdg::appendAssignments(fusedProgram, *assignment);
            }
            else if (const auto* predicate = std::get_if<pdg::PredicateProgram>(
                         &stages[index].payload))
            {
                flushAssignments();
                orderedProgram.operations.emplace_back(*predicate);
                orderedProgram.filtersPoints = true;
            }
            else if (const auto* ordinal = std::get_if<pdg::OrdinalProgram>(
                         &stages[index].payload))
            {
                flushAssignments();
                orderedProgram.operations.emplace_back(*ordinal);
                orderedProgram.filtersPoints = true;
                hasOrdinal = true;
                hasTail = hasTail || ordinal->kind == pdg::OrdinalKind::Tail;
            }
            else
                return false;
        }
        flushAssignments();
        orderedProgram.ordinalMode =
            hasTail ? pdg::OrdinalMode::Standard : pdg::OrdinalMode::Streaming;

        std::uint64_t ordinalInputCount = input.header().pointCount;
        bool dataDependentCount = false;
        for (const pdg::las::PointOperation& operation :
             orderedProgram.operations)
        {
            if (std::holds_alternative<pdg::PredicateProgram>(operation))
            {
                dataDependentCount = true;
                continue;
            }
            const auto* ordinal = std::get_if<pdg::OrdinalProgram>(&operation);
            if (!ordinal)
                continue;
            if (orderedProgram.ordinalMode == pdg::OrdinalMode::Standard)
            {
                if (dataDependentCount ||
                    (ordinal->kind == pdg::OrdinalKind::Tail &&
                     ordinal->count > ordinalInputCount))
                    return false;
                static_cast<void>(pdg::makeOrdinalState(
                    *ordinal, orderedProgram.ordinalMode, ordinalInputCount));
                ordinalInputCount = pdg::ordinalStandardOutputCount(
                    *ordinal, ordinalInputCount);
            }
            else if (!pdg::ordinalSupportsMode(*ordinal,
                                               orderedProgram.ordinalMode))
                return false;
        }
        const auto appendUniqueDimension =
            [](std::vector<pdg::DimensionId>& ids, pdg::DimensionId id)
        {
            if (std::find(ids.begin(), ids.end(), id) == ids.end())
                ids.push_back(id);
        };
        for (const pdg::las::PointOperation& operation :
             orderedProgram.operations)
        {
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
            {
                for (pdg::DimensionId id : assignments->reads)
                    appendUniqueDimension(orderedProgram.reads, id);
                for (pdg::DimensionId id : assignments->writes)
                    appendUniqueDimension(orderedProgram.writes, id);
            }
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
                for (pdg::DimensionId id : predicate->reads)
                    appendUniqueDimension(orderedProgram.reads, id);
        }
        if (!pdg::las::supportsDefaultPointProgram(input, orderedProgram,
                                                   dimensions))
            return false;

        temporaryPath = outputPath;
        temporaryPath += ".pdg-native-" + std::to_string(::getpid());
        if (std::filesystem::exists(temporaryPath, error) || error)
            return false;
        const pdg::las::DefaultTranslationMetadata metadata =
            translationMetadata();
        bool cudaTranslated = false;
#if PDG_HAS_CUDA
        const bool requireCuda = std::getenv("PDG_REQUIRE_CUDA_POINT_PROGRAM");
        const bool requireFusedCuda =
            std::getenv("PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM");
        const bool forceCuda =
            std::getenv("PDG_EXPERIMENTAL_CUDA_POINT_PROGRAM") || requireCuda ||
            requireFusedCuda;
        const bool automaticCuda =
            !hasOrdinal && !std::getenv("PDG_DISABLE_CUDA_POINT_PROGRAM") &&
            pdg::las::preferDefaultCudaPointProgram(input.header().pointCount,
                                                    fusedProgram);
        const bool fusedCudaSupported =
            !orderedProgram.filtersPoints && declaredWholeProgramFusion &&
            pdg::las::supportsDefaultFusedCudaPointProgram(input, fusedProgram,
                                                           dimensions);
        const bool cudaPlanSupported =
            orderedProgram.filtersPoints || declaredWholeProgramFusion;
        if ((!requireFusedCuda || fusedCudaSupported) && cudaPlanSupported &&
            (forceCuda || automaticCuda) &&
            pdg::las::supportsDefaultCudaPointProgram(input, orderedProgram,
                                                      dimensions))
        {
            try
            {
                const std::size_t outputBytes =
                    pdg::las::defaultTranslationSize(input);
                PositionedOutput cudaOutput(temporaryPath, outputBytes,
                                            temporaryOwned);
                if (orderedProgram.filtersPoints)
                {
                    const std::uint64_t survivors =
                        pdg::las::translateDefaultOrderedPointProgramCudaToSink(
                            input, metadata, orderedProgram, dimensions,
                            [&](std::size_t offset,
                                std::span<const std::byte> bytes)
                            { cudaOutput.writeAt(offset, bytes); },
                            cudaChunkPoints(1U << 17U), cudaSchedulerLanes());
                    if (survivors >
                        (std::numeric_limits<std::size_t>::max() - 375U) / 36U)
                        throw pdg::las::Error(
                            "filtered LAS output size overflows size_t");
                    cudaOutput.close(375U +
                                     static_cast<std::size_t>(survivors) * 36U);
                }
                else
                {
                    pdg::las::translateDefaultPointProgramCudaToSink(
                        input, metadata, fusedProgram, dimensions,
                        [&](std::size_t offset,
                            std::span<const std::byte> bytes)
                        { cudaOutput.writeAt(offset, bytes); },
                        cudaChunkPoints(1U << 17U), cudaSchedulerLanes());
                    cudaOutput.close();
                }
                cudaTranslated = true;
            }
            catch (const pdg::CudaError&)
            {
                if (temporaryOwned)
                {
                    std::error_code cleanupError;
                    if (!std::filesystem::remove(temporaryPath, cleanupError) ||
                        cleanupError)
                        throw;
                    temporaryOwned = false;
                }
                if (requireCuda || requireFusedCuda)
                    throw;
            }
        }
        if ((requireCuda || requireFusedCuda) && !cudaTranslated)
            throw pdg::las::Error(
                "requested point program is outside the exact CUDA envelope");
#endif
        if (!cudaTranslated)
        {
            if (orderedProgram.filtersPoints)
                return false;
            MappedOutput mappedOutput(temporaryPath,
                                      pdg::las::defaultTranslationSize(input),
                                      temporaryOwned);
            pdg::las::translateDefaultInto(
                input, metadata, mappedOutput.bytes(), nativeWorkerLimit());
            pdg::las::applyDefaultPointProgram(mappedOutput.bytes(), input,
                                               fusedProgram, dimensions,
                                               nativeWorkerLimit());
            mappedOutput.close();
        }
        if (::link(temporaryPath.c_str(), outputPath.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to publish native LAS output");
        if (::unlink(temporaryPath.c_str()) == 0)
            temporaryOwned = false;
        return true;
    }
    catch (...)
    {
        if (temporaryOwned)
        {
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
        return false;
    }
}

std::optional<pdg::cli::DirectResidentLasResult>
tryDirectResidentLasImpl(const pdg::Plan& plan,
                         pdg::DimensionRegistry& dimensions,
                         const pdg::PlanPlacementEstimate& placement,
                         std::size_t deviceMemoryBudgetBytes)
{
#if !PDG_HAS_CUDA
    static_cast<void>(plan);
    static_cast<void>(dimensions);
    static_cast<void>(placement);
    static_cast<void>(deviceMemoryBudgetBytes);
    return std::nullopt;
#else
    const std::vector<pdg::PlannedStage>& stages = plan.stages();
    if (placement.choice != pdg::PlacementChoice::Device ||
        placement.selectedRegionCount != 1U ||
        !plan.summary().allStagesNative || stages.size() < 3U ||
        stages.front().role != pdg::StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().inputs.empty() ||
        stages.back().role != pdg::StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" ||
        !declaresWholeProgramFusion(plan) || !deviceMemoryBudgetBytes)
        return std::nullopt;

    const pdg::PlacementRegionEstimate* selected = nullptr;
    for (const pdg::PlacementRegionEstimate& region : placement.regions)
    {
        if (!region.selected)
            continue;
        if (selected)
            return std::nullopt;
        selected = &region;
    }
    if (!selected)
        return std::nullopt;

    std::vector<std::size_t> middleStages;
    middleStages.reserve(stages.size() - 2U);
    for (std::size_t index = 1U; index + 1U < stages.size(); ++index)
    {
        if (stages[index].role != pdg::StageRole::Filter ||
            stages[index].inputs != std::vector<std::size_t>{index - 1U} ||
            stages[index].residentRegion != selected->residentRegion)
            return std::nullopt;
        middleStages.push_back(index);
    }
    if (stages.back().inputs != std::vector<std::size_t>{stages.size() - 2U} ||
        selected->stageIds != middleStages)
        return std::nullopt;

    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&stages.front().payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&stages.back().payload);
    if (!reader || !writer)
        return std::nullopt;
    const std::filesystem::path inputPath(reader->filename);
    const std::filesystem::path outputPath(writer->filename);
    if (!hasLasExtension(inputPath) || !hasLasExtension(outputPath))
        return std::nullopt;

    pdg::cli::DirectResidentLasResult result;
    result.residentRegion = selected->residentRegion;
    result.stageIds = selected->stageIds;
    std::filesystem::path temporaryPath;
    bool temporaryOwned = false;
    bool directEligible = false;
    try
    {
        const MappedInput mappedInput(inputPath);
        const pdg::las::FileView input(mappedInput.bytes());
        pdg::AssignProgram program;
        pdg::AssignProgram pendingAssignments;
        pdg::las::OrderedPointProgram orderedProgram;
        const auto flushAssignments = [&]
        {
            if (!pendingAssignments.assignments.empty())
            {
                orderedProgram.operations.emplace_back(
                    std::move(pendingAssignments));
                pendingAssignments = {};
            }
        };
        for (std::size_t index : middleStages)
        {
            if (const auto* ferry =
                    std::get_if<pdg::FerryProgram>(&stages[index].payload))
            {
                if (!pdg::las::supportsDefaultFerry(input, *ferry, dimensions))
                    return std::nullopt;
                pdg::appendFerry(program, *ferry);
                pdg::appendFerry(pendingAssignments, *ferry);
            }
            else if (const auto* assignments = std::get_if<pdg::AssignProgram>(
                         &stages[index].payload))
            {
                pdg::appendAssignments(program, *assignments);
                pdg::appendAssignments(pendingAssignments, *assignments);
            }
            else if (const auto* predicate = std::get_if<pdg::PredicateProgram>(
                         &stages[index].payload))
            {
                flushAssignments();
                orderedProgram.operations.emplace_back(*predicate);
                orderedProgram.filtersPoints = true;
            }
            else
                return std::nullopt;
        }
        flushAssignments();
        orderedProgram.ordinalMode = pdg::OrdinalMode::Streaming;
        for (const pdg::las::PointOperation& operation :
             orderedProgram.operations)
        {
            const auto appendUniqueDimension =
                [](std::vector<pdg::DimensionId>& ids, pdg::DimensionId id)
            {
                if (std::find(ids.begin(), ids.end(), id) == ids.end())
                    ids.push_back(id);
            };
            if (const auto* assignments =
                    std::get_if<pdg::AssignProgram>(&operation))
            {
                for (pdg::DimensionId id : assignments->reads)
                    appendUniqueDimension(orderedProgram.reads, id);
                for (pdg::DimensionId id : assignments->writes)
                    appendUniqueDimension(orderedProgram.writes, id);
            }
            else if (const auto* predicate =
                         std::get_if<pdg::PredicateProgram>(&operation))
                for (pdg::DimensionId id : predicate->reads)
                    appendUniqueDimension(orderedProgram.reads, id);
        }
        if (program.assignments.empty())
            return std::nullopt;
        if (orderedProgram.filtersPoints)
        {
            // The declared compacting chain runs through the validated
            // ordered decode/predicate/pack sink; the writer-prologue fusion
            // candidate covering the whole chain is the declared proof.
            if (!pdg::las::supportsDefaultPointProgram(input, orderedProgram,
                                                       dimensions) ||
                !pdg::las::supportsDefaultCudaPointProgram(
                    input, orderedProgram, dimensions))
                return std::nullopt;
        }
        else if (!pdg::las::supportsDefaultPointProgram(input, program,
                                                        dimensions) ||
                 !pdg::las::supportsDefaultFusedCudaPointProgram(input, program,
                                                                 dimensions))
            return std::nullopt;
        directEligible = true;

        const std::size_t chunkPoints = cudaChunkPoints(1U << 17U);
        const std::size_t schedulerLanes = cudaSchedulerLanes();
        std::error_code error;
        if (std::filesystem::exists(outputPath, error))
            throw std::system_error(
                std::make_error_code(std::errc::file_exists),
                "resident LAS output already exists");
        if (error)
            throw std::system_error(error,
                                    "unable to inspect resident LAS output");
        temporaryPath = outputPath;
        temporaryPath += ".pdg-native-" + std::to_string(::getpid());
        if (std::filesystem::exists(temporaryPath, error))
            throw std::system_error(
                std::make_error_code(std::errc::file_exists),
                "resident LAS temporary output already exists");
        if (error)
            throw std::system_error(
                error, "unable to inspect resident LAS temporary output");

        const std::size_t outputBytes = pdg::las::defaultTranslationSize(input);
        PositionedOutput output(temporaryPath, outputBytes, temporaryOwned);
        pdg::las::CudaTranslationMetrics metrics;
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionBegin,
            selected->residentRegion);
        if (orderedProgram.filtersPoints)
        {
            result.orderedExecutor = true;
            const std::uint64_t survivors =
                pdg::las::translateDefaultOrderedPointProgramCudaToSink(
                    input, translationMetadata(), orderedProgram, dimensions,
                    [&](std::size_t offset, std::span<const std::byte> bytes)
                    { output.writeAt(offset, bytes); }, chunkPoints,
                    schedulerLanes, deviceMemoryBudgetBytes, &metrics,
                    &result.schedule);
            if (survivors >
                (std::numeric_limits<std::size_t>::max() - 375U) / 36U)
                throw pdg::las::Error(
                    "filtered LAS output size overflows size_t");
            result.outputPointCount = static_cast<std::size_t>(survivors);
        }
        else
        {
            result.schedule = pdg::las::translateDefaultPointProgramCudaToSink(
                input, translationMetadata(), program, dimensions,
                [&](std::size_t offset, std::span<const std::byte> bytes)
                { output.writeAt(offset, bytes); }, chunkPoints, schedulerLanes,
                deviceMemoryBudgetBytes, &metrics);
            result.outputPointCount =
                static_cast<std::size_t>(input.header().pointCount);
        }
        result.hostToDeviceBytes = metrics.hostToDeviceBytes;
        result.deviceToHostBytes = metrics.deviceToHostBytes;
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::HostToDevice, selected->residentRegion,
            result.hostToDeviceBytes);
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceToHost, selected->residentRegion,
            result.deviceToHostBytes);
        pdg::ExecutionObservationScope::record(
            pdg::ExecutionEventKind::DeviceRegionEnd, selected->residentRegion);
        if (orderedProgram.filtersPoints)
            output.close(375U + result.outputPointCount * 36U);
        else
            output.close();

        if (::link(temporaryPath.c_str(), outputPath.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to publish resident LAS output");
        if (::unlink(temporaryPath.c_str()) == 0)
            temporaryOwned = false;
        return result;
    }
    catch (...)
    {
        if (!temporaryOwned && !directEligible)
            return std::nullopt;
        const std::exception_ptr failure = std::current_exception();
        if (temporaryOwned)
        {
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
        std::rethrow_exception(failure);
    }
#endif
}

bool canonicalLasDimension(pdg::DimensionId id) noexcept
{
    using pdg::StandardDimension;
    constexpr StandardDimension Dimensions[]{
        StandardDimension::X,
        StandardDimension::Y,
        StandardDimension::Z,
        StandardDimension::Intensity,
        StandardDimension::ReturnNumber,
        StandardDimension::NumberOfReturns,
        StandardDimension::ScanDirectionFlag,
        StandardDimension::EdgeOfFlightLine,
        StandardDimension::Classification,
        StandardDimension::Synthetic,
        StandardDimension::KeyPoint,
        StandardDimension::Withheld,
        StandardDimension::Overlap,
        StandardDimension::ScanAngleRank,
        StandardDimension::UserData,
        StandardDimension::PointSourceId,
        StandardDimension::GpsTime,
        StandardDimension::ScanChannel,
        StandardDimension::ClassFlags,
        StandardDimension::Red,
        StandardDimension::Green,
        StandardDimension::Blue};
    return std::any_of(std::begin(Dimensions), std::end(Dimensions),
                       [&](StandardDimension candidate)
                       { return id == pdg::DimensionId(candidate); });
}

struct DirectResidentLasOutputPlan
{
    bool extraDouble = false;
    bool permutedClassification = false;
    bool permutedSort = false;
    pdg::DimensionId dimension;
    pdal::Dimension::Id pdalDimension = pdal::Dimension::Id::Unknown;
    std::string name;
    std::string description;
};

std::optional<DirectResidentLasOutputPlan> directResidentLasOutputPlan(
    const pdg::Plan& plan, bool allowSingleStageCanonicalFilter,
    bool allowPermutedClassification, bool allowPermutedSort)
{
    const std::vector<pdg::PlannedStage>& stages = plan.stages();
    if (plan.summary().residentRegions != 1U || stages.size() < 3U ||
        stages.front().role != pdg::StageRole::Reader ||
        stages.front().descriptor.type != "readers.las" ||
        !stages.front().native || !stages.front().inputs.empty() ||
        stages.back().role != pdg::StageRole::Writer ||
        stages.back().descriptor.type != "writers.las" ||
        stages.back().inputs != std::vector<std::size_t>{stages.size() - 2U})
        return std::nullopt;

    const auto* reader =
        std::get_if<pdg::FileStagePlan>(&stages.front().payload);
    const auto* writer =
        std::get_if<pdg::FileStagePlan>(&stages.back().payload);
    if (!reader || !writer || !hasLasExtension(reader->filename) ||
        !hasLasExtension(writer->filename))
        return std::nullopt;
    std::error_code pathError;
    const std::filesystem::path inputPath =
        std::filesystem::absolute(reader->filename, pathError)
            .lexically_normal();
    if (pathError)
        return std::nullopt;
    const std::filesystem::path outputPath =
        std::filesystem::absolute(writer->filename, pathError)
            .lexically_normal();
    if (pathError || inputPath == outputPath)
        return std::nullopt;
    if (std::filesystem::exists(outputPath, pathError) || pathError)
        return std::nullopt;

    bool writesCanonicalOutput = false;
    bool permutedClassification = false;
    bool permutedSort = false;
    std::vector<pdg::DimensionId> extraWrites;
    std::optional<std::size_t> residentRegion;
    for (std::size_t index = 1U; index + 1U < stages.size(); ++index)
    {
        const pdg::PlannedStage& stage = stages[index];
        const auto* skewness =
            std::get_if<pdg::SkewnessProgram>(&stage.payload);
        const bool exactSkewnessPermutation =
            allowPermutedClassification && stages.size() == 3U && index == 1U &&
            writer->extraDimensionsAll && skewness &&
            skewness->groundClass == 2U && skewness->otherClass == 1U &&
            !skewness->onlyGround &&
            stage.descriptor.kind == pdg::StageKind::Global &&
            stage.descriptor.index.kind == pdg::IndexKind::None &&
            stage.descriptor.writes ==
                std::vector<pdg::DimensionId>{
                    pdg::DimensionId(pdg::StandardDimension::Classification)};
        const auto* ordering =
            std::get_if<pdg::OrderingProgram>(&stage.payload);
        const bool exactSortPermutation =
            allowPermutedSort && stages.size() == 3U && index == 1U &&
            writer->extraDimensionsAll && ordering &&
            ordering->dimensions ==
                std::vector<pdg::DimensionId>{
                    pdg::DimensionId(pdg::StandardDimension::Z)} &&
            ordering->direction == pdg::OrderingDirection::Ascending &&
            ordering->algorithm == pdg::OrderingAlgorithm::Normal &&
            stage.descriptor.kind == pdg::StageKind::Global &&
            stage.descriptor.index.kind == pdg::IndexKind::None &&
            stage.descriptor.reads ==
                std::vector<pdg::DimensionId>{
                    pdg::DimensionId(pdg::StandardDimension::Z)} &&
            stage.descriptor.writes.empty() && stage.descriptor.fusion.pure &&
            stage.descriptor.fusion.cardinalityPreserving &&
            stage.descriptor.fusion.deterministicSafe &&
            !stage.descriptor.fusion.hasWhere &&
            stage.descriptor.fusion.whereMerge ==
                pdg::WhereMergeMode::NotApplicable &&
            !stage.descriptor.mutatesCoordinates &&
            !stage.descriptor.preservesOrder;
        if (stage.role != pdg::StageRole::Filter || !stage.native ||
            stage.inputs != std::vector<std::size_t>{index - 1U} ||
            (!stage.descriptor.preservesOrder && !exactSkewnessPermutation &&
             !exactSortPermutation) ||
            !stage.descriptor.fusion.cardinalityPreserving ||
            stage.residentRegion == pdg::NoResidentRegion)
            return std::nullopt;
        permutedClassification =
            permutedClassification || exactSkewnessPermutation;
        permutedSort = permutedSort || exactSortPermutation;
        if (!residentRegion)
            residentRegion = stage.residentRegion;
        else if (*residentRegion != stage.residentRegion)
            return std::nullopt;
        for (pdg::DimensionId write : stage.descriptor.writes)
        {
            if (!canonicalLasDimension(write))
            {
                extraWrites.push_back(write);
                continue;
            }
            if (writer->extraDimensionsAll && !exactSkewnessPermutation &&
                !exactSortPermutation)
                return std::nullopt;
            if (write != pdg::DimensionId(pdg::StandardDimension::UserData) &&
                write !=
                    pdg::DimensionId(pdg::StandardDimension::Classification))
                return std::nullopt;
            writesCanonicalOutput = true;
        }
    }

    const MappedInput mappedInput(inputPath);
    const pdg::las::FileView input(mappedInput.bytes());
    if (permutedClassification)
    {
        const pdg::las::Header& header = input.header();
        if (!writer->extraDimensionsAll || !writesCanonicalOutput ||
            !extraWrites.empty() || header.versionMajor != 1U ||
            header.versionMinor != 4U || header.pointFormat != 7U ||
            header.pointRecordLength != 36U ||
            !pdg::las::supportsDefaultTranslation(input))
            return std::nullopt;
        return DirectResidentLasOutputPlan{.permutedClassification = true};
    }
    if (permutedSort)
    {
        const pdg::las::Header& header = input.header();
        if (!writer->extraDimensionsAll || writesCanonicalOutput ||
            !extraWrites.empty() || header.versionMajor != 1U ||
            header.versionMinor != 4U || header.pointFormat != 7U ||
            header.pointRecordLength != 36U ||
            !pdg::las::supportsDefaultTranslation(input))
            return std::nullopt;
        return DirectResidentLasOutputPlan{.permutedSort = true};
    }
    if (!writer->extraDimensionsAll)
    {
        if (!plan.summary().allStagesNative || !writesCanonicalOutput ||
            (stages.size() == 3U &&
             (!allowSingleStageCanonicalFilter ||
              (stages[1U].descriptor.type != "filters.radiusassign" &&
               stages[1U].descriptor.type != "filters.neighborclassifier" &&
               stages[1U].descriptor.type != "filters.outlier"))) ||
            !pdg::las::supportsDefaultTranslation(input))
            return std::nullopt;
        return DirectResidentLasOutputPlan{};
    }

    if (writesCanonicalOutput || extraWrites.size() != 1U ||
        !pdg::las::supportsExtraDoubleTranslation(input) ||
        extraWrites.front().value() == 0U ||
        extraWrites.front().value() >=
            static_cast<std::uint32_t>(pdg::StandardDimension::Count))
        return std::nullopt;
    const auto pdalDimension =
        static_cast<pdal::Dimension::Id>(extraWrites.front().value());
    if (pdal::Dimension::defaultType(pdalDimension) !=
        pdal::Dimension::Type::Double)
        return std::nullopt;
    return DirectResidentLasOutputPlan{
        .extraDouble = true,
        .dimension = extraWrites.front(),
        .pdalDimension = pdalDimension,
        .name = pdal::Dimension::name(pdalDimension),
        .description =
            pdal::Dimension::description(pdalDimension).substr(0U, 32U)};
}

void publishDirectResidentLasOutputImpl(const pdg::Plan& plan,
                                        const pdal::PointView& view,
                                        bool allowSingleStageCanonicalFilter,
                                        bool allowPermutedClassification,
                                        bool allowPermutedSort)
{
    const std::optional<DirectResidentLasOutputPlan> outputPlan =
        directResidentLasOutputPlan(plan, allowSingleStageCanonicalFilter,
                                    allowPermutedClassification,
                                    allowPermutedSort);
    if (!outputPlan)
        throw pdg::las::Error(
            "resident LAS output is outside the exact direct envelope");
    const std::vector<pdg::PlannedStage>& stages = plan.stages();
    const auto& reader = std::get<pdg::FileStagePlan>(stages.front().payload);
    const auto& writer = std::get<pdg::FileStagePlan>(stages.back().payload);
    const std::filesystem::path inputPath(reader.filename);
    const std::filesystem::path outputPath(writer.filename);
    const MappedInput mappedInput(inputPath);
    const pdg::las::FileView input(mappedInput.bytes());
    const auto planWrites = [&](pdg::StandardDimension dimension)
    {
        const pdg::DimensionId id(dimension);
        return std::any_of(stages.begin() + 1, stages.end() - 1,
                           [&](const pdg::PlannedStage& stage)
                           {
                               return std::find(stage.descriptor.writes.begin(),
                                                stage.descriptor.writes.end(),
                                                id) !=
                                      stage.descriptor.writes.end();
                           });
    };
    const bool writesUserData = planWrites(pdg::StandardDimension::UserData);
    const bool writesClassification =
        planWrites(pdg::StandardDimension::Classification);
    if (input.header().pointCount != view.size() ||
        (writesUserData &&
         !view.layout()->hasDim(pdal::Dimension::Id::UserData)) ||
        (writesClassification &&
         !view.layout()->hasDim(pdal::Dimension::Id::Classification)) ||
        (outputPlan->extraDouble &&
         !view.layout()->hasDim(outputPlan->pdalDimension)))
        throw pdg::las::Error(
            "resident LAS output view does not match the direct input");

    std::error_code error;
    if (std::filesystem::exists(outputPath, error))
        throw std::system_error(std::make_error_code(std::errc::file_exists),
                                "resident LAS output already exists");
    if (error)
        throw std::system_error(error, "unable to inspect resident LAS output");
    std::filesystem::path temporaryPath = outputPath;
    temporaryPath += ".pdg-resident-output-" + std::to_string(::getpid());
    if (std::filesystem::exists(temporaryPath, error))
        throw std::system_error(std::make_error_code(std::errc::file_exists),
                                "resident LAS temporary output already exists");
    if (error)
        throw std::system_error(
            error, "unable to inspect resident LAS temporary output");

    bool temporaryOwned = false;
    try
    {
        const std::size_t outputBytes =
            outputPlan->extraDouble
                ? pdg::las::extraDoubleTranslationSize(input)
                : pdg::las::defaultTranslationSize(input);
        MappedOutput output(temporaryPath, outputBytes, temporaryOwned);
        if (outputPlan->permutedClassification || outputPlan->permutedSort)
        {
            std::vector<std::uint64_t> sourceOrder(
                static_cast<std::size_t>(view.size()));
            for (pdal::PointId point = 0; point < view.size(); ++point)
            {
                const std::size_t index = static_cast<std::size_t>(point);
                sourceOrder[index] =
                    pdal::pdg_detail::ResidentPointViewAccess::tableId(view,
                                                                       point);
            }
            if (outputPlan->permutedClassification)
            {
                std::vector<std::uint8_t> classification(sourceOrder.size());
                for (pdal::PointId point = 0; point < view.size(); ++point)
                    classification[static_cast<std::size_t>(point)] =
                        view.getFieldAs<std::uint8_t>(
                            pdal::Dimension::Id::Classification, point);
                pdg::las::translateDefaultPermutedClassificationInto(
                    input, translationMetadata(), sourceOrder, classification,
                    output.bytes(), nativeWorkerLimit());
            }
            else
                pdg::las::translateDefaultPermutedInto(
                    input, translationMetadata(), sourceOrder, output.bytes(),
                    nativeWorkerLimit());
        }
        else if (outputPlan->extraDouble)
        {
            std::vector<std::uint64_t> valueBits(
                static_cast<std::size_t>(view.size()));
            for (pdal::PointId point = 0; point < view.size(); ++point)
                valueBits[static_cast<std::size_t>(point)] =
                    std::bit_cast<std::uint64_t>(view.getFieldAs<double>(
                        outputPlan->pdalDimension, point));
            pdg::las::translateExtraDoubleInto(
                input, translationMetadata(),
                {.name = outputPlan->name,
                 .description = outputPlan->description},
                valueBits, output.bytes(), nativeWorkerLimit());
        }
        else
        {
            pdg::las::translateDefaultInto(input, translationMetadata(),
                                           output.bytes(), nativeWorkerLimit());
            if (writesUserData)
            {
                std::vector<std::uint8_t> userData(
                    static_cast<std::size_t>(view.size()));
                for (pdal::PointId point = 0; point < view.size(); ++point)
                    userData[static_cast<std::size_t>(point)] =
                        view.getFieldAs<std::uint8_t>(
                            pdal::Dimension::Id::UserData, point);
                pdg::las::overlayDefaultUserData(output.bytes(), userData);
            }
            if (writesClassification)
            {
                std::vector<std::uint8_t> classification(
                    static_cast<std::size_t>(view.size()));
                for (pdal::PointId point = 0; point < view.size(); ++point)
                    classification[static_cast<std::size_t>(point)] =
                        view.getFieldAs<std::uint8_t>(
                            pdal::Dimension::Id::Classification, point);
                pdg::las::overlayDefaultClassification(output.bytes(),
                                                       classification);
            }
        }
        output.close();
        if (::link(temporaryPath.c_str(), outputPath.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "unable to publish resident LAS output");
        if (::unlink(temporaryPath.c_str()) == 0)
            temporaryOwned = false;
    }
    catch (...)
    {
        if (temporaryOwned)
        {
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
        throw;
    }
}

std::filesystem::path oraclePath(const char* executable)
{
    if (const char* configured = std::getenv("PDG_ORACLE_PDAL");
        configured && *configured)
        return configured;

    std::error_code error;
    std::filesystem::path self =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error)
        self = std::filesystem::absolute(executable, error);
    if (error || self.empty())
        return "pdal";
    return self.parent_path() / "pdal";
}

int runOracle(int argc, char** argv)
{
    const std::filesystem::path oracle = oraclePath(argv[0]);
    const std::string program = oracle.string();
    std::vector<char*> arguments;
    arguments.reserve(static_cast<std::size_t>(argc) + 1U);
    arguments.push_back(const_cast<char*>(program.c_str()));
    for (int index = 1; index < argc; ++index)
        arguments.push_back(argv[index]);
    arguments.push_back(nullptr);

    if (oracle.has_parent_path())
        ::execv(program.c_str(), arguments.data());
    else
        ::execvp(program.c_str(), arguments.data());

    const int error = errno;
    std::cerr << "gpupdal: unable to execute pinned PDAL fallback " << program
              << ": " << std::strerror(error) << '\n';
    return error == ENOENT ? 127 : 126;
}
} // unnamed namespace

namespace pdg::cli
{

std::optional<DirectResidentLasResult>
tryDirectResidentLas(const Plan& plan, DimensionRegistry& dimensions,
                     const PlanPlacementEstimate& placement,
                     std::size_t deviceMemoryBudgetBytes)
{
    return tryDirectResidentLasImpl(plan, dimensions, placement,
                                    deviceMemoryBudgetBytes);
}

bool supportsDirectResidentLasOutput(const Plan& plan,
                                     bool allowSingleStageCanonicalFilter,
                                     bool allowPermutedClassification,
                                     bool allowPermutedSort)
{
    try
    {
        return directResidentLasOutputPlan(
                   plan, allowSingleStageCanonicalFilter,
                   allowPermutedClassification, allowPermutedSort)
            .has_value();
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool supportsDirectResidentExtraDoubleOutput(const Plan& plan)
{
    try
    {
        const std::optional<DirectResidentLasOutputPlan> output =
            directResidentLasOutputPlan(plan, false, false, false);
        return output && output->extraDouble;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void publishDirectResidentLasOutput(const Plan& plan,
                                    const pdal::PointView& view,
                                    bool allowSingleStageCanonicalFilter,
                                    bool allowPermutedClassification,
                                    bool allowPermutedSort)
{
    publishDirectResidentLasOutputImpl(
        plan, view, allowSingleStageCanonicalFilter,
        allowPermutedClassification, allowPermutedSort);
}

} // namespace pdg::cli

int main(int argc, char** argv)
{
    const std::string_view command = argc > 1 ? argv[1] : "";
    if (command == "version")
    {
        version();
        return 0;
    }
    if (command == "doctor")
        return doctor();
    if (command == "calibrate")
        return pdg::cli::runCalibrate(argc, argv);
    if (command == "resident")
        return pdg::cli::runResidentPipeline(argc, argv);
    if (command == "pipeline")
    {
        if (const std::optional<int> automatic =
                pdg::cli::tryAutomaticResidentLasPipeline(argc, argv))
            return *automatic;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"))
    {
        std::cerr << "gpupdal: required automatic resident LAS output path "
                     "was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic eigen-family resident path "
                     "was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic normal/covariance resident "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic rank/optimal resident path "
                     "was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic outlier/NNDistance resident "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv(
            "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic radius-outlier/radial-density "
                     "resident path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic radiusassign resident path "
                     "was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic neighborclassifier resident "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic HAG-NN count-one resident "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT"))
    {
        std::cerr
            << "gpupdal: required automatic HAG-Delaunay count-three resident "
               "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT"))
    {
        std::cerr
            << "gpupdal: required automatic skewness resident path was not "
               "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic sort resident path was not "
                     "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT"))
    {
        std::cerr << "gpupdal: required automatic approximate-coplanar resident "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION"))
    {
        std::cerr
            << "gpupdal: required direct skewness composition path was not "
               "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_DIRECT_SORT_COMPOSITION"))
    {
        std::cerr << "gpupdal: required direct sort composition path was not "
                     "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"))
    {
        std::cerr << "gpupdal: required direct LAS resident source path "
                     "was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY") ||
        std::getenv("PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND"))
    {
        std::cerr << "gpupdal: required direct LAS record summary path was not "
                     "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"))
    {
        std::cerr
            << "gpupdal: required direct LAS no-host-XYZ path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF"))
    {
        std::cerr << "gpupdal: required NNDistance device-only assignment "
                     "handoff was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NND_HOST_RESTORE"))
    {
        std::cerr
            << "gpupdal: required NNDistance host restoration path was not "
               "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NND_PARALLEL_REPAIR"))
    {
        std::cerr << "gpupdal: required parallel selective NNDistance device "
                     "repair path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NND_DEVICE_REPAIR"))
    {
        std::cerr << "gpupdal: required selective NNDistance device repair "
                     "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_OUTLIER_DEVICE_REPAIR"))
    {
        std::cerr << "gpupdal: required selective statistical-outlier device "
                     "repair path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR"))
    {
        std::cerr << "gpupdal: required parallel statistical-outlier device "
                     "repair path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_LOF_PARALLEL_REPAIR"))
    {
        std::cerr << "gpupdal: required parallel LOF repair path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE"))
    {
        std::cerr << "gpupdal: required LOF KD3 coordinate cache path was not "
                     "used\n";
        return 124;
    }

    if (tryNativeTranslate(argc, argv))
        return 0;
    if (tryNativePipeline(argc, argv))
        return 0;
    if (const std::optional<int> hybrid =
            pdg::cli::tryHybridPipeline(argc, argv))
        return *hybrid;
    if (std::getenv("PDG_REQUIRE_CUDA_TRANSLATE"))
    {
        std::cerr
            << "gpupdal: required exact CUDA translation path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_CUDA_POINT_PROGRAM"))
    {
        std::cerr << "gpupdal: required exact CUDA point-program path was not "
                     "used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM"))
    {
        std::cerr
            << "gpupdal: required descriptor-planned fused CUDA point-program "
               "path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_HYBRID") ||
        std::getenv("PDG_REQUIRE_STREAMING_HYBRID") ||
        std::getenv("PDG_REQUIRE_CUDA_HYBRID"))
    {
        std::cerr << "gpupdal: required hybrid pipeline path was not used\n";
        return 124;
    }
    if (std::getenv("PDG_REQUIRE_NATIVE"))
    {
        std::cerr << "gpupdal: requested command is outside the proven native "
                     "compatibility envelope\n";
        return 125;
    }

    return runOracle(argc, argv);
}
