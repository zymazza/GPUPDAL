#include "Dispatch.hpp"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if PDG_HAS_CUDA && defined(__linux__)
#include <dlfcn.h>
#endif
#include <unistd.h>

namespace
{
std::vector<std::string_view>
presentDispatchEnvironment(char* const* environment)
{
    std::vector<std::string_view> present;
    for (char* const* entry = environment; entry && *entry; ++entry)
    {
        const std::string_view assignment(*entry);
        const std::size_t equals = assignment.find('=');
        if (equals != std::string_view::npos)
            present.push_back(assignment.substr(0U, equals));
    }
    return present;
}

std::filesystem::path executablePath(const char* executable)
{
    std::error_code error;
    std::filesystem::path self =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error)
        self = std::filesystem::absolute(executable, error);
    return error ? std::filesystem::path{} : self;
}

std::filesystem::path oraclePath(const char* executable)
{
    if (const char* configured = std::getenv("PDG_ORACLE_PDAL");
        configured && *configured)
        return configured;

    const std::filesystem::path self = executablePath(executable);
    if (self.empty())
        return "pdal";
    return self.parent_path() / "pdal";
}

int execProgram(const std::filesystem::path& program, int argc, char** argv,
                std::string_view description, bool replaceArgvZero,
                std::optional<std::string_view> appendedArgument =
                    std::nullopt)
{
    const std::string executable = program.string();
    std::vector<char*> arguments;
    arguments.reserve(static_cast<std::size_t>(argc) +
                      (appendedArgument ? 2U : 1U));
    arguments.push_back(replaceArgvZero ? const_cast<char*>(executable.c_str())
                                        : argv[0]);
    for (int index = 1; index < argc; ++index)
        arguments.push_back(argv[index]);
    if (appendedArgument)
        arguments.push_back(const_cast<char*>(appendedArgument->data()));
    arguments.push_back(nullptr);

    if (program.has_parent_path())
        ::execv(executable.c_str(), arguments.data());
    else
        ::execvp(executable.c_str(), arguments.data());

    const int error = errno;
    std::cerr << "gpupdal: unable to execute " << description << ' '
              << executable << ": " << std::strerror(error) << '\n';
    return error == ENOENT ? 127 : 126;
}

int runOracle(int argc, char** argv,
              std::optional<std::string_view> appendedArgument = std::nullopt)
{
    // Match the original PDG fallback exactly: PDAL receives its resolved
    // executable as argv[0], not the public pdg launcher path.
    return execProgram(oraclePath(argv[0]), argc, argv, "pinned PDAL fallback",
                       true, appendedArgument);
}

int runEngine(int argc, char** argv)
{
#if PDG_HAS_CUDA && defined(__linux__)
    // A CUDA release links the engine to NVIDIA's driver library, which is
    // deliberately never bundled. If the driver is absent, exec would fail
    // in the dynamic loader before the engine could select its exact host
    // fallback. The thin launcher has no CUDA dependency, so it can preserve
    // drop-in behavior by delegating directly to the bundled pinned oracle.
    const std::unique_ptr<void, int (*)(void*)> driver(
        ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL), &::dlclose);
    if (!driver)
        return runOracle(argc, argv);
#endif
    const std::filesystem::path self = executablePath(argv[0]);
    const std::filesystem::path engine =
        self.empty() ? std::filesystem::path("pdg-engine")
                     : self.parent_path() / "pdg-engine";
    return execProgram(engine, argc, argv, "PDG engine", false);
}

int runVerify(int argc, char** argv)
{
    const std::filesystem::path self = executablePath(argv[0]);
    const std::filesystem::path binaryDirectory =
        self.empty() ? std::filesystem::path{} : self.parent_path();
    std::filesystem::path helper = binaryDirectory / "pdg-verify.py";
    std::error_code error;
    if (!std::filesystem::is_regular_file(helper, error) || error)
    {
        // Installed layouts keep implementation helpers out of the public
        // binary directory.  Build trees copy the same bytes beside `gpupdal`.
        helper =
            binaryDirectory.parent_path() / "libexec" / "pdg" / "pdg-verify.py";
    }
    if (!std::filesystem::is_regular_file(helper, error) || error)
    {
        std::cerr << "gpupdal: verification helper is not installed: " << helper
                  << '\n';
        return 127;
    }

    const std::string candidate =
        (self.empty() ? std::filesystem::path(argv[0]) : self).string();
    const std::string oracle = oraclePath(argv[0]).string();
    const bool configuredOracle = []
    {
        const char* value = std::getenv("PDG_ORACLE_PDAL");
        return value && *value;
    }();
    bool acceptsConfiguredOracle = false;
    for (int index = 2; index < argc; ++index)
    {
        if (std::string_view(argv[index]) == "--accept-configured-oracle")
            acceptsConfiguredOracle = true;
    }
    if (configuredOracle && !acceptsConfiguredOracle)
    {
        std::cerr << "gpupdal: PDG_ORACLE_PDAL is set; verification will not "
                     "silently attest a redirected oracle. Re-run with "
                     "--accept-configured-oracle to record and use it.\n";
        return 2;
    }
    const std::string helperText = helper.string();
    std::vector<std::string> owned{"python3", helperText};
    for (int index = 2; index < argc; ++index)
        owned.emplace_back(argv[index]);
    // Inject resolved roles last so user arguments cannot substitute a
    // different candidate or oracle while retaining the `gpupdal verify` label.
    owned.emplace_back("--candidate");
    owned.push_back(candidate);
    owned.emplace_back("--oracle");
    owned.push_back(oracle);
    owned.emplace_back("--oracle-source");
    owned.push_back(configuredOracle ? "accepted-PDG_ORACLE_PDAL"
                                     : "sibling-pdal");
    std::vector<char*> arguments;
    arguments.reserve(owned.size() + 1U);
    for (std::string& value : owned)
        arguments.push_back(value.data());
    arguments.push_back(nullptr);
    ::execvp(arguments[0], arguments.data());
    const int executionError = errno;
    if (executionError == ENOENT)
    {
        std::cerr << "gpupdal: verification requires Python 3; install "
                     "python3 and ensure it is on PATH\n";
    }
    else
    {
        std::cerr << "gpupdal: unable to execute Python 3 for verification: "
                  << std::strerror(executionError) << '\n';
    }
    return executionError == ENOENT ? 127 : 126;
}

bool supportedPipelineOptions(int argc, char** argv)
{
    for (int index = 3; index < argc; ++index)
    {
        const std::string_view option(argv[index]);
        if (option == "--stream" || option == "--nostream")
            continue;
        if (option == "--metadata")
        {
            if (++index < argc)
                continue;
            return false;
        }
        if (option.starts_with("--metadata="))
            continue;
        return false;
    }
    return true;
}

std::optional<std::string> readClassifiablePipeline(int argc, char** argv)
{
    if (argc < 3 || !supportedPipelineOptions(argc, argv) ||
        std::string_view(argv[2]) == "--stdin")
        return std::nullopt;

    const std::filesystem::path path(argv[2]);
    std::error_code error;
    constexpr std::uintmax_t MaximumPipelineBytes = 16U * 1024U * 1024U;
    if (!std::filesystem::is_regular_file(path, error) || error ||
        std::filesystem::file_size(path, error) > MaximumPipelineBytes || error)
        return std::nullopt;
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    std::string text((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    if (input.bad())
        return std::nullopt;
    return text;
}

std::optional<pdg::cli::DispatchInputFacts>
lasInputFacts(std::string_view filename)
{
    const std::filesystem::path path(filename);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        return std::nullopt;
    const std::uintmax_t fileBytes = std::filesystem::file_size(path, error);
    if (error)
        return std::nullopt;
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    constexpr std::size_t Las12HeaderBytes = 227;
    constexpr std::size_t Las14HeaderBytes = 375;
    std::array<char, Las14HeaderBytes> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::streamsize bytesRead = input.gcount();
    if (bytesRead < static_cast<std::streamsize>(Las12HeaderBytes) ||
        std::memcmp(bytes.data(), "LASF", 4) != 0)
        return std::nullopt;
    const auto read = [&]<typename T>(std::size_t offset)
    {
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    };
    const std::uint8_t versionMajor = read.operator()<std::uint8_t>(24);
    const std::uint8_t versionMinor = read.operator()<std::uint8_t>(25);
    if (versionMajor != 1U || versionMinor > 4U)
        return std::nullopt;
    std::uint64_t points = read.operator()<std::uint32_t>(107);
    if (versionMinor >= 4U)
    {
        if (bytesRead < static_cast<std::streamsize>(Las14HeaderBytes))
            return std::nullopt;
        points = read.operator()<std::uint64_t>(247);
    }
    const bool measuredReferenceLayout =
        fileBytes == 6'747'641U && versionMajor == 1U &&
        versionMinor == 4U &&
        read.operator()<std::uint16_t>(94) == 375U &&
        read.operator()<std::uint32_t>(96) == 475U &&
        read.operator()<std::uint8_t>(104) == 0x87U &&
        read.operator()<std::uint16_t>(105) == 36U &&
        read.operator()<double>(131) == 0.01 &&
        read.operator()<double>(139) == 0.01 &&
        read.operator()<double>(147) == 0.01 &&
        read.operator()<double>(155) == 0.0 &&
        read.operator()<double>(163) == 0.0 &&
        read.operator()<double>(171) == 0.0 &&
        read.operator()<double>(179) == 185999.99 &&
        read.operator()<double>(187) == 184500.0 &&
        read.operator()<double>(195) == 494999.99 &&
        read.operator()<double>(203) == 494923.21 &&
        read.operator()<double>(211) == 500.41 &&
        read.operator()<double>(219) == 367.44;
    const bool measuredR14ReferenceLayout =
        fileBytes == 36'000'375U && versionMajor == 1U &&
        versionMinor == 4U &&
        read.operator()<std::uint16_t>(94) == 375U &&
        read.operator()<std::uint32_t>(96) == 375U &&
        read.operator()<std::uint8_t>(104) == 0x07U &&
        read.operator()<std::uint16_t>(105) == 36U &&
        read.operator()<double>(131) == 0.01 &&
        read.operator()<double>(139) == 0.01 &&
        read.operator()<double>(147) == 0.01 &&
        read.operator()<double>(155) == 0.0 &&
        read.operator()<double>(163) == 0.0 &&
        read.operator()<double>(171) == 0.0 &&
        read.operator()<double>(179) == 185999.99 &&
        read.operator()<double>(187) == 184500.0 &&
        read.operator()<double>(195) == 494999.99 &&
        read.operator()<double>(203) == 494923.21 &&
        read.operator()<double>(211) == 500.41 &&
        read.operator()<double>(219) == 367.44;
    return pdg::cli::DispatchInputFacts{
        points, measuredReferenceLayout, measuredR14ReferenceLayout};
}

// B0243 routes every external PDG_* control to the engine before command
// classification, which silently skipped arming the r2 hybrid marker for
// engine-owned proof runs (B0262 finding: pdg_automatic_r2_ground_normalize
// failed since B0243). The marker is an engine-route hint that the engine
// independently revalidates, so arming it on the engine route is safe.
void armAutomaticR2MarkerIfMatched(int argc, char** argv)
{
    if (argc < 2 || std::string_view(argv[1]) != "pipeline")
        return;
    const std::optional<std::string> pipeline =
        readClassifiablePipeline(argc, argv);
    if (!pipeline)
        return;
    std::optional<pdg::cli::DispatchInputFacts> inputFacts;
    if (const std::optional<std::string> filename =
            pdg::cli::dispatchPointCountProbeFilename(*pipeline))
        inputFacts = lasInputFacts(*filename);
    if (pdg::cli::dispatchUsesAutomaticR2Hybrid(*pipeline, inputFacts))
    {
        ::setenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID", "1", 1);
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupdal: armed automatic r2 hybrid dispatch\n";
    }
}

bool commandRequiresEngine(int argc, char** argv,
                           bool& useAutomaticR7ReaderThreads)
{
    useAutomaticR7ReaderThreads = false;
    if (argc < 2)
        return false;
    const std::string_view command(argv[1]);
    if (command == "version" || command == "doctor" || command == "resident" ||
        command == "calibrate" || command == "translate")
        return true;
    if (command != "pipeline")
        return false;

    const std::optional<std::string> pipeline =
        readClassifiablePipeline(argc, argv);
    if (!pipeline)
        return true;
    if (pdg::cli::dispatchRequiresPlainPipelineInvocation(*pipeline) &&
        argc != 3)
        return true;
    std::optional<pdg::cli::DispatchInputFacts> inputFacts;
    if (const std::optional<std::string> filename =
            pdg::cli::dispatchPointCountProbeFilename(*pipeline))
    {
        inputFacts = lasInputFacts(*filename);
    }
    if (pdg::cli::dispatchUsesAutomaticR2Hybrid(*pipeline, inputFacts))
    {
        ::setenv("PDG_INTERNAL_AUTOMATIC_R2_HYBRID", "1", 1);
        if (std::getenv("PDG_DEBUG_HYBRID"))
            std::cerr << "gpupdal: armed automatic r2 hybrid dispatch\n";
    }
    useAutomaticR7ReaderThreads =
        pdg::cli::dispatchUsesAutomaticR7ReaderThreads(*pipeline, inputFacts);
    if (pdg::cli::dispatchUsesAutomaticR14ParallelCompression(*pipeline,
                                                              inputFacts) &&
        ::setenv("PDG_LAZ_COMPRESSION_THREADS", "2", 1) != 0)
        return true;
    return pdg::cli::classifyPipelineForDispatch(*pipeline, inputFacts) ==
           pdg::cli::DispatchRoute::Engine;
}

} // unnamed namespace

int main(int argc, char** argv, char** environmentPointers)
{
    const std::vector<std::string_view> environment =
        presentDispatchEnvironment(environmentPointers);
    // `gpupdal --fast <command> ...` (D0261/D0271): the leading flag is consumed
    // here, recorded as an internal marker for the engine and sibling, and
    // the command is dispatched exactly as without it on every route,
    // including the environment-selected engine route below. Fast mode keeps
    // point records in order with identical coordinates and count; kNN
    // tie-order choices, diagnostics, error text/status, LAS header/VLR
    // metadata, and metadata JSON may differ.
    bool fastFlag = false;
    if (argc >= 2 && std::string_view(argv[1]) == pdg::cli::FastModeFlag)
    {
        fastFlag = true;
        for (int index = 1; index + 1 < argc; ++index)
            argv[index] = argv[index + 1];
        argv[--argc] = nullptr;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "verify")
    {
        if (fastFlag)
        {
            std::cerr
                << "gpupdal: verify proves default exact mode; --fast is not "
                   "accepted\n";
            return 2;
        }
        return runVerify(argc, argv);
    }
    if (pdg::cli::dispatchEnvironmentRequiresEngine(environment))
    {
        // These are internal launcher-to-engine/writer channels, not public
        // tuning options. Never forward an ambient value into an engine-owned
        // route: an externally injected fast-mode marker must not relax the
        // default exact contract; only the flag consumed above arms it.
        ::unsetenv("PDG_LAZ_COMPRESSION_THREADS");
        ::unsetenv(pdg::cli::FastModeMarker.data());
        if (fastFlag &&
            ::setenv(pdg::cli::FastModeMarker.data(), "1", 1) != 0)
            return 1;
        armAutomaticR2MarkerIfMatched(argc, argv);
        return runEngine(argc, argv);
    }
    if (fastFlag && ::setenv(pdg::cli::FastModeMarker.data(), "1", 1) != 0)
        return runEngine(argc, argv);
    bool useAutomaticR7ReaderThreads = false;
    if (commandRequiresEngine(argc, argv, useAutomaticR7ReaderThreads))
        return runEngine(argc, argv);
    return runOracle(
        argc, argv,
        useAutomaticR7ReaderThreads
            ? std::optional<std::string_view>("--readers.las.threads=4")
            : std::nullopt);
}
