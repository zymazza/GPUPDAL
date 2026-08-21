// r11 neighborhood attribution harness.
//
// Runs the checked-in r11 prefix (LAS/LAZ reader -> filters.smrf ->
// statistical filters.outlier) through the fork's PDAL library exactly as the
// public pipeline does, then measures the two exact neighborhood consumers on
// the resulting PointView in isolation:
//
//   * the statistical outlier kNN pass (mean_k + 1 = 9 neighbors, per-point
//     mean-distance recurrence, distances kept for a later serial reduction);
//   * the neighbor classifier kNN pass over the `Classification[1:1]` domain
//     (`k = 7`) with upstream's `neighbors()` allocation profile;
//   * the classifier's vote tally alone, isolated from the search, comparing
//     upstream's ordered `std::map` tally with an allocation-free ascending-key
//     array tally that must select the identical winner; and
//   * fixed-chunk multi-worker executions of both kNN passes over the same
//     read-only PointView-owned nanoflann index, whose per-point results must
//     be bit-identical to the serial passes.
//
// The harness changes no product code and publishes nothing. It exists so the
// D0204 rule "use a cheap controlled prototype to reject a weak direction
// before a certified lane is spent on it" can be applied to r11's host limiter
// with a reproducible artifact rather than a sampled profile. Timings are
// harness-local attribution, not complete-process claims.

#include <pdal/KDIndex.hpp>
#include <pdal/Options.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PointRef.hpp>
#include <pdal/PointView.hpp>
#include <pdal/Stage.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace
{
using Clock = std::chrono::steady_clock;

// Process-wide allocation counter. It only counts; allocation behavior is
// unchanged. Available on Linux/glibc; other platforms report null counts.
std::atomic<std::uint64_t> g_allocations{0};
} // namespace

#if defined(__linux__)
extern "C" void* malloc(std::size_t size)
{
    using MallocFn = void* (*)(std::size_t);
    static MallocFn real = nullptr;
    if (!real)
        real = reinterpret_cast<MallocFn>(dlsym(RTLD_NEXT, "malloc"));
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    return real(size);
}
#endif

namespace
{
struct Options
{
    std::string input;
    std::string output;
    std::vector<unsigned> workers{4U, 8U, 11U, 12U};
    std::size_t repeats = 3U;
    int classifierK = 7;
    int meanK = 8;
};

double seconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double>(end - begin).count();
}

std::uint64_t allocations()
{
    return g_allocations.load(std::memory_order_relaxed);
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view option = argv[index];
        if (option == "--help")
        {
            std::cout << "Usage: pdg_r11_neighborhood_attribution --input "
                         "FILE [--output REPORT.json]\n"
                      << "  [--workers N,N,...] [--repeats N] [--k N] "
                         "[--mean-k N]\n";
            std::exit(0);
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " +
                                        std::string(option));
        const std::string value = argv[++index];
        if (option == "--input")
            options.input = value;
        else if (option == "--output")
            options.output = value;
        else if (option == "--repeats")
            options.repeats = static_cast<std::size_t>(std::stoul(value));
        else if (option == "--k")
            options.classifierK = std::stoi(value);
        else if (option == "--mean-k")
            options.meanK = std::stoi(value);
        else if (option == "--workers")
        {
            options.workers.clear();
            std::size_t start = 0;
            while (start <= value.size())
            {
                const std::size_t comma = value.find(',', start);
                const std::string item = value.substr(start,
                    comma == std::string::npos ? std::string::npos
                                               : comma - start);
                if (!item.empty())
                    options.workers.push_back(
                        static_cast<unsigned>(std::stoul(item)));
                if (comma == std::string::npos)
                    break;
                start = comma + 1;
            }
        }
        else
            throw std::invalid_argument("unknown option " +
                                        std::string(option));
    }
    if (options.input.empty())
        throw std::invalid_argument("--input is required");
    if (options.repeats == 0 || options.classifierK < 1 || options.meanK < 1)
        throw std::invalid_argument("counts must be positive");
    for (unsigned count : options.workers)
        if (count < 2U)
            throw std::invalid_argument("--workers entries must be >= 2");
    return options;
}

bool bitIdentical(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
        return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// Upstream statistical-outlier per-point recurrence, unchanged.
void outlierRow(const pdal::KD3Index& index, pdal::PointId id,
    pdal::point_count_t count, pdal::PointIdList& indices,
    std::vector<double>& sqrDists, double& distance)
{
    index.knnSearch(id, count, &indices, &sqrDists);
    for (std::size_t j = 1; j < count; ++j)
    {
        const double delta = std::sqrt(sqrDists[j]) - distance;
        distance += (delta / static_cast<double>(j));
    }
    indices.clear();
    indices.resize(count);
    sqrDists.clear();
    sqrDists.resize(count);
}

// Upstream neighbor-classifier vote: ordered map, first maximum in ascending
// key order, strict majority of the returned neighbor count.
bool mapVote(const pdal::PointView& view, const pdal::PointId* neighbors,
    std::size_t count, int oldClass, int& newClass)
{
    using CountMap = std::map<int, unsigned int>;
    CountMap counts;
    for (std::size_t j = 0; j < count; ++j)
        counts[view.getFieldAs<int>(pdal::Dimension::Id::Classification,
            neighbors[j])]++;
    auto pr = *std::max_element(counts.begin(), counts.end(),
        [](CountMap::const_reference p1, CountMap::const_reference p2)
        { return p1.second < p2.second; });
    const double thresh = static_cast<double>(count) / 2.0;
    newClass = pr.first;
    return pr.second > thresh && oldClass != newClass;
}

// Allocation-free ascending-key tally with the same first-maximum rule.
bool arrayVote(const pdal::PointView& view, const pdal::PointId* neighbors,
    std::size_t count, int oldClass, int& newClass)
{
    constexpr std::size_t MaxKeys = 64;
    if (count > MaxKeys)
        throw std::logic_error("array tally bounded to 64 neighbors");
    int keys[MaxKeys];
    unsigned int votes[MaxKeys];
    std::size_t distinct = 0;
    for (std::size_t j = 0; j < count; ++j)
    {
        const int cls = view.getFieldAs<int>(
            pdal::Dimension::Id::Classification, neighbors[j]);
        std::size_t pos = 0;
        while (pos < distinct && keys[pos] < cls)
            ++pos;
        if (pos < distinct && keys[pos] == cls)
        {
            ++votes[pos];
            continue;
        }
        for (std::size_t q = distinct; q > pos; --q)
        {
            keys[q] = keys[q - 1];
            votes[q] = votes[q - 1];
        }
        keys[pos] = cls;
        votes[pos] = 1U;
        ++distinct;
    }
    std::size_t best = 0;
    for (std::size_t q = 1; q < distinct; ++q)
        if (votes[q] > votes[best])
            best = q;
    const double thresh = static_cast<double>(count) / 2.0;
    newClass = keys[best];
    return votes[best] > thresh && oldClass != newClass;
}

int run(const Options& options)
{
    nlohmann::json report;
    report["schema"] = "pdg-r11-neighborhood-attribution/1";
    report["input"] = options.input;
    report["hardware_concurrency"] = std::thread::hardware_concurrency();
    report["classifier_k"] = options.classifierK;
    report["outlier_mean_k"] = options.meanK;
#if defined(__linux__)
    report["allocation_counting"] = "glibc malloc interposition";
#else
    report["allocation_counting"] = nullptr;
#endif

    pdal::PipelineManager manager;
    pdal::Stage& reader = manager.makeReader(options.input, "readers.las");
    pdal::Options smrfOptions;
    pdal::Stage& smrf = manager.makeFilter("filters.smrf", reader,
        smrfOptions);
    pdal::Options outlierOptions;
    outlierOptions.add("method", "statistical");
    outlierOptions.add("mean_k", options.meanK);
    outlierOptions.add("multiplier", 2.0);
    manager.makeFilter("filters.outlier", smrf, outlierOptions);

    const auto prefixBegin = Clock::now();
    manager.execute();
    const auto prefixEnd = Clock::now();
    pdal::PointViewSet views = manager.views();
    if (views.size() != 1)
        throw std::runtime_error("expected exactly one prefix view");
    pdal::PointViewPtr view = *views.begin();
    const pdal::point_count_t np = view->size();
    report["prefix"] = {
        {"stages", "readers.las -> filters.smrf -> filters.outlier"
                   "(statistical)"},
        {"seconds", seconds(prefixBegin, prefixEnd)},
        {"points", np},
    };

    // Fresh exact tree, as the statistical outlier builds it.
    const auto buildBegin = Clock::now();
    view->invalidateProducts();
    pdal::KD3Index& index = view->build3dIndex();
    const auto buildEnd = Clock::now();
    report["kd3_fresh_build_seconds"] = seconds(buildBegin, buildEnd);

    // ---- Statistical outlier kNN pass -------------------------------------
    const auto count = static_cast<pdal::point_count_t>(options.meanK + 1);
    std::vector<double> serialDistances(np, 0.0);
    {
        pdal::PointIdList indices(count);
        std::vector<double> sqrDists(count);
        const std::uint64_t allocBegin = allocations();
        const auto begin = Clock::now();
        for (pdal::PointId i = 0; i < np; ++i)
            outlierRow(index, i, count, indices, sqrDists, serialDistances[i]);
        const auto end = Clock::now();
        report["outlier_knn_pass"]["serial"] = {
            {"seconds", seconds(begin, end)},
            {"allocations", allocations() - allocBegin},
            {"neighbors", count},
        };
    }
    for (unsigned workers : options.workers)
    {
        std::vector<double> distances(np, 0.0);
        std::vector<std::exception_ptr> failures(workers);
        const std::size_t chunk = (np + workers - 1) / workers;
        const auto begin = Clock::now();
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < workers; ++w)
        {
            threads.emplace_back([&, w]()
            {
                try
                {
                    const std::size_t first = w * chunk;
                    const std::size_t last = (std::min)(
                        static_cast<std::size_t>(np), first + chunk);
                    pdal::PointIdList indices(count);
                    std::vector<double> sqrDists(count);
                    for (std::size_t i = first; i < last; ++i)
                        outlierRow(index, i, count, indices, sqrDists,
                            distances[i]);
                }
                catch (...)
                {
                    failures[w] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads)
            thread.join();
        const auto end = Clock::now();
        for (auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        report["outlier_knn_pass"]["parallel"].push_back({
            {"workers", workers},
            {"seconds", seconds(begin, end)},
            {"bit_identical", bitIdentical(serialDistances, distances)},
        });
    }

    // ---- Neighbor classifier domain and serial search ---------------------
    const auto k = static_cast<pdal::point_count_t>(options.classifierK);
    std::vector<pdal::PointId> domain;
    for (pdal::PointId i = 0; i < np; ++i)
    {
        // Upstream DimRange semantics for `Classification[1:1]`.
        const double v = view->getFieldAs<double>(
            pdal::Dimension::Id::Classification, i);
        if (!(std::isnan(v) || v < 1.0 || v > 1.0))
            domain.push_back(i);
    }
    const std::size_t domainCount = domain.size();
    report["classifier"]["domain_points"] = domainCount;

    std::vector<pdal::PointId> serialNeighbors(domainCount * k);
    {
        pdal::PointRef point(*view, 0);
        const std::uint64_t allocBegin = allocations();
        const auto begin = Clock::now();
        for (std::size_t j = 0; j < domainCount; ++j)
        {
            point.setPointId(domain[j]);
            const pdal::PointIdList found = index.neighbors(point, k);
            if (found.size() != k)
                throw std::runtime_error("unexpected neighbor count");
            std::copy(found.begin(), found.end(),
                serialNeighbors.begin() + static_cast<std::ptrdiff_t>(j * k));
        }
        const auto end = Clock::now();
        report["classifier"]["serial_neighbors"] = {
            {"seconds", seconds(begin, end)},
            {"allocations", allocations() - allocBegin},
        };
    }
    {
        // Same query through the buffer-reusing knnSearch entry point.
        pdal::PointRef point(*view, 0);
        pdal::PointIdList indices(k);
        std::vector<double> sqrDists(k);
        std::vector<pdal::PointId> neighbors(domainCount * k);
        const std::uint64_t allocBegin = allocations();
        const auto begin = Clock::now();
        for (std::size_t j = 0; j < domainCount; ++j)
        {
            point.setPointId(domain[j]);
            index.knnSearch(point, k, &indices, &sqrDists);
            std::copy(indices.begin(), indices.end(),
                neighbors.begin() + static_cast<std::ptrdiff_t>(j * k));
        }
        const auto end = Clock::now();
        report["classifier"]["serial_knnsearch"] = {
            {"seconds", seconds(begin, end)},
            {"allocations", allocations() - allocBegin},
            {"identical", neighbors == serialNeighbors},
        };
    }

    // ---- Vote tally isolated from the search ------------------------------
    for (std::size_t repeat = 0; repeat < options.repeats; ++repeat)
    {
        std::unordered_map<pdal::PointId, int> mapResult;
        std::unordered_map<pdal::PointId, int> arrayResult;
        const std::uint64_t mapAllocBegin = allocations();
        const auto mapBegin = Clock::now();
        for (std::size_t j = 0; j < domainCount; ++j)
        {
            const int oldClass = view->getFieldAs<int>(
                pdal::Dimension::Id::Classification, domain[j]);
            int newClass = 0;
            if (mapVote(*view, serialNeighbors.data() + j * k, k, oldClass,
                    newClass))
                mapResult[domain[j]] = newClass;
        }
        const auto mapEnd = Clock::now();
        const std::uint64_t mapAllocEnd = allocations();
        const auto arrayBegin = Clock::now();
        for (std::size_t j = 0; j < domainCount; ++j)
        {
            const int oldClass = view->getFieldAs<int>(
                pdal::Dimension::Id::Classification, domain[j]);
            int newClass = 0;
            if (arrayVote(*view, serialNeighbors.data() + j * k, k, oldClass,
                    newClass))
                arrayResult[domain[j]] = newClass;
        }
        const auto arrayEnd = Clock::now();
        const std::uint64_t arrayAllocEnd = allocations();
        report["classifier"]["tally"].push_back({
            {"repeat", repeat},
            {"map_seconds", seconds(mapBegin, mapEnd)},
            {"map_allocations", mapAllocEnd - mapAllocBegin},
            {"array_seconds", seconds(arrayBegin, arrayEnd)},
            {"array_allocations", arrayAllocEnd - mapAllocEnd},
            {"reassignments", mapResult.size()},
            {"identical", mapResult == arrayResult},
        });
    }

    // ---- Parallel exact classifier search ---------------------------------
    for (unsigned workers : options.workers)
    {
        std::vector<pdal::PointId> neighbors(domainCount * k);
        std::vector<std::exception_ptr> failures(workers);
        const std::size_t chunk = (domainCount + workers - 1) / workers;
        const auto begin = Clock::now();
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < workers; ++w)
        {
            threads.emplace_back([&, w]()
            {
                try
                {
                    const std::size_t first = w * chunk;
                    const std::size_t last = (std::min)(domainCount,
                        first + chunk);
                    pdal::PointRef point(*view, 0);
                    pdal::PointIdList indices(k);
                    std::vector<double> sqrDists(k);
                    for (std::size_t j = first; j < last; ++j)
                    {
                        point.setPointId(domain[j]);
                        index.knnSearch(point, k, &indices, &sqrDists);
                        std::copy(indices.begin(), indices.end(),
                            neighbors.begin() +
                                static_cast<std::ptrdiff_t>(j * k));
                    }
                }
                catch (...)
                {
                    failures[w] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads)
            thread.join();
        const auto end = Clock::now();
        for (auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        report["classifier"]["parallel_knnsearch"].push_back({
            {"workers", workers},
            {"seconds", seconds(begin, end)},
            {"identical", neighbors == serialNeighbors},
        });
    }

    std::cout << report.dump(2) << '\n';
    if (!options.output.empty())
    {
        std::ofstream out(options.output);
        if (!out)
            throw std::runtime_error("cannot write " + options.output);
        out << report.dump(2) << '\n';
    }
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        return run(parseOptions(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << "pdg_r11_neighborhood_attribution: " << error.what()
                  << '\n';
        return 1;
    }
}
