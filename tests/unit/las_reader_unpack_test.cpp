// D0269: readers.las unpacks decoded records on fixed-slot workers in both
// execution modes. The pinned-oracle process matrix proves the published
// binary; this unit proves the one shape that matrix cannot reach with the
// CLI's fixed 10,000-row stream table: a stream batch whose rows come from
// two tiles (a capacity that does not divide the 50,000-record tile), so a
// worker run carries more than one segment. Parallel and serial (control)
// executions must produce byte-identical files, and streaming must equal
// standard mode.

#include <gtest/gtest.h>

#include <pdal/PipelineManager.hpp>
#include <pdal/util/FileUtils.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <unistd.h>
#include <optional>
#include <string>
#include <vector>

namespace
{

class ScopedEnvironment
{
public:
    ScopedEnvironment(const char* name, const char* value) : m_name(name)
    {
        if (const char* prior = std::getenv(name))
            m_prior = prior;
        if (value)
            ::setenv(name, value, 1);
        else
            ::unsetenv(name);
    }
    ~ScopedEnvironment()
    {
        if (m_prior)
            ::setenv(m_name.c_str(), m_prior->c_str(), 1);
        else
            ::unsetenv(m_name.c_str());
    }

private:
    std::string m_name;
    std::optional<std::string> m_prior;
};

std::vector<char> bytes(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

void runPipeline(const std::string& json, pdal::point_count_t streamLimit,
    bool stream)
{
    pdal::PipelineManager manager(streamLimit);
    std::stringstream in(json);
    manager.readPipeline(in);
    if (stream)
    {
        const pdal::PipelineManager::ExecResult result =
            manager.execute(pdal::ExecMode::Stream);
        ASSERT_EQ(pdal::ExecMode::Stream, result.m_mode);
    }
    else
        manager.execute(pdal::ExecMode::Standard);
}

} // unnamed namespace

TEST(LasReaderUnpack, MultiSegmentStreamBatchesMatchSerialAndStandard)
{
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        ("pdg-las-reader-unpack-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const std::string source = (dir / "source.las").string();
    // 123,457 synthetic points: three tiles, the last one partial.
    runPipeline(R"json({"pipeline":[{"type":"readers.faux","count":123457,
        "mode":"ramp","bounds":"([0,1000],[0,1000],[0,100])"},
        {"type":"writers.las","filename":")json" + source + R"json(",
        "minor_version":4,"dataformat_id":6}]})json", 10000, false);

    auto convert = [&](const std::string& out, pdal::point_count_t limit,
        bool stream)
    {
        runPipeline(R"json({"pipeline":[{"type":"readers.las","filename":")json" +
            source + R"json("},{"type":"writers.las","filename":")json" + out +
            R"json(","minor_version":4,"dataformat_id":6}]})json", limit, stream);
        return bytes(out);
    };

    std::vector<char> parallel;
    std::vector<char> standard;
    {
        ScopedEnvironment forced("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
            "3");
        ScopedEnvironment enabled("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
            nullptr);
        // 15,000-row batches: rows 45,000..59,999 span tiles 0 and 1.
        parallel = convert((dir / "parallel.las").string(), 15000, true);
        standard = convert((dir / "standard.las").string(), 15000, false);
    }
    std::vector<char> serial;
    {
        ScopedEnvironment forced("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
            nullptr);
        ScopedEnvironment disabled("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
            "1");
        serial = convert((dir / "serial.las").string(), 15000, true);
    }
    ASSERT_GT(serial.size(), 123457U * 30U);
    EXPECT_EQ(serial, parallel);
    EXPECT_EQ(serial, standard);
    std::filesystem::remove_all(dir);
}
