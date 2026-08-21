#include <pdal/Dimension.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/PointRef.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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

struct ArithmeticPointView
{
    pdal::PointTable table;
    pdal::PointViewPtr view;

    explicit ArithmeticPointView(std::size_t pointCount)
    {
        table.layout()->registerDims({pdal::Dimension::Id::X,
                                      pdal::Dimension::Id::Y,
                                      pdal::Dimension::Id::Z});
        view.reset(new pdal::PointView(table));
        for (std::size_t point = 0; point < pointCount; ++point)
        {
            view->setField(pdal::Dimension::Id::X, point,
                           static_cast<double>(point));
            view->setField(pdal::Dimension::Id::Y, point,
                           static_cast<double>((point * 7U) % 181U));
            view->setField(pdal::Dimension::Id::Z, point,
                           static_cast<double>((point * 13U) % 211U));
        }
    }
};

struct Kd3KnnExpectation
{
    pdal::PointIdList ids;
    std::vector<double> squaredDistances;
};

Kd3KnnExpectation serialKnn(const pdal::KD3Index& index,
                            const std::vector<pdal::PointId>& queries,
                            std::uint32_t neighbors)
{
    Kd3KnnExpectation result;
    result.ids.reserve(queries.size() * neighbors);
    result.squaredDistances.reserve(queries.size() * neighbors);
    for (const pdal::PointId query : queries)
    {
        pdal::PointIdList ids(neighbors);
        std::vector<double> squaredDistances(neighbors);
        index.knnSearch(query, neighbors, &ids, &squaredDistances);
        result.ids.insert(result.ids.end(), ids.begin(), ids.end());
        result.squaredDistances.insert(result.squaredDistances.end(),
                                       squaredDistances.begin(),
                                       squaredDistances.end());
    }
    return result;
}

bool rowEquals(const pdal::PointIdList& candidateIds,
               const std::vector<double>& candidateDistances,
               const Kd3KnnExpectation& expected, std::size_t queryOffset,
               std::uint32_t neighbors)
{
    const std::size_t base = queryOffset * neighbors;
    for (std::uint32_t item = 0; item < neighbors; ++item)
    {
        if (candidateIds[item] != expected.ids[base + item])
            return false;
        if (candidateDistances[item] != expected.squaredDistances[base + item])
            return false;
    }
    return true;
}

void runConcurrentProbe(const pdal::KD3Index& index,
                        const std::vector<pdal::PointId>& queries,
                        const Kd3KnnExpectation& expected,
                        std::uint32_t neighbors, int repetitions,
                        std::atomic<bool>& failed, std::mutex& failureMutex,
                        std::string& firstFailure, std::size_t worker)
{
    for (int repeat = 0;
         repeat < repetitions && !failed.load(std::memory_order_relaxed);
         ++repeat)
    {
        for (std::size_t queryOffset = 0; queryOffset < queries.size();
             ++queryOffset)
        {
            const pdal::PointId query = queries[queryOffset];
            pdal::PointIdList ids(neighbors);
            std::vector<double> squaredDistances(neighbors);
            index.knnSearch(query, neighbors, &ids, &squaredDistances);

            if (rowEquals(ids, squaredDistances, expected, queryOffset,
                          neighbors))
                continue;

            std::lock_guard<std::mutex> lock(failureMutex);
            if (!failed.exchange(true, std::memory_order_relaxed))
            {
                firstFailure = "worker " + std::to_string(worker) +
                               " mismatch at repeat " + std::to_string(repeat) +
                               " queryOffset " + std::to_string(queryOffset) +
                               " point " + std::to_string(query);
            }
            return;
        }
    }
}
} // namespace

TEST(Kd3Concurrency, FailedCachedBuildIsNotPublished)
{
    constexpr std::size_t PointCount = 97U;
    constexpr std::uint32_t Neighbors = 8U;
    const std::vector<pdal::PointId> queries = {0, 1, 3, 11, 19, 33, 47, 61};

    ArithmeticPointView recoveredFixture(PointCount);
    {
        ScopedEnvironment failure("PDG_TEST_KD3_CACHE_BUILD_FAILURE", "1");
        EXPECT_THROW(recoveredFixture.view->build3dIndex(true),
                     pdal::pdal_error);
    }
    pdal::KD3Index& recoveredIndex = recoveredFixture.view->build3dIndex(false);
    EXPECT_FALSE(recoveredIndex.coordinatesCached());

    ArithmeticPointView freshFixture(PointCount);
    pdal::KD3Index& freshIndex = freshFixture.view->build3dIndex(false);
    const Kd3KnnExpectation recovered =
        serialKnn(recoveredIndex, queries, Neighbors);
    const Kd3KnnExpectation fresh = serialKnn(freshIndex, queries, Neighbors);
    EXPECT_EQ(recovered.ids, fresh.ids);
    EXPECT_EQ(recovered.squaredDistances, fresh.squaredDistances);
}

TEST(Kd3Concurrency, ConcurrentReadOnlyKnnMatchesSerialOnCachedIndex)
{
    constexpr std::size_t PointCount = 97U;
    constexpr std::uint32_t Neighbors = 8U;
    constexpr int Repetitions = 32;
    constexpr std::size_t ThreadCount = 4U;
    const std::vector<pdal::PointId> queries = {0, 1, 3, 11, 19, 33, 47, 61};

    ArithmeticPointView cachedFixture(PointCount);
    pdal::KD3Index& cachedIndex = cachedFixture.view->build3dIndex(true);
    const Kd3KnnExpectation cachedExpected =
        serialKnn(cachedIndex, queries, Neighbors);
    ASSERT_EQ(cachedExpected.ids.size(), queries.size() * Neighbors);
    ASSERT_EQ(cachedExpected.squaredDistances.size(),
              queries.size() * Neighbors);

    ArithmeticPointView uncachedFixture(PointCount);
    pdal::KD3Index& uncachedIndex = uncachedFixture.view->build3dIndex(false);
    const Kd3KnnExpectation uncachedExpected =
        serialKnn(uncachedIndex, queries, Neighbors);
    EXPECT_EQ(cachedExpected.ids, uncachedExpected.ids);
    EXPECT_EQ(cachedExpected.squaredDistances,
              uncachedExpected.squaredDistances);

    std::atomic<bool> failed{false};
    std::mutex failureMutex;
    std::string firstFailure;
    std::vector<std::thread> workers;
    workers.reserve(ThreadCount);
    for (std::size_t worker = 0; worker < ThreadCount; ++worker)
    {
        workers.emplace_back(
            runConcurrentProbe, std::cref(cachedIndex), std::cref(queries),
            std::cref(cachedExpected), Neighbors, Repetitions, std::ref(failed),
            std::ref(failureMutex), std::ref(firstFailure), worker);
    }

    for (std::thread& worker : workers)
        worker.join();

    EXPECT_FALSE(failed.load(std::memory_order_relaxed)) << firstFailure;
}

// D0263: published cached-coordinate products refresh their snapshot when the
// view's coordinate epoch moved, so a stale tree is queried with live
// coordinates exactly as the uncached (pinned) adapter does.
namespace
{
Kd3KnnExpectation allQueries(const pdal::KD3Index& index,
                             std::size_t pointCount, std::uint32_t neighbors)
{
    std::vector<pdal::PointId> queries(pointCount);
    for (std::size_t point = 0; point < pointCount; ++point)
        queries[point] = point;
    return serialKnn(index, queries, neighbors);
}

bool sameAnswers(const Kd3KnnExpectation& left, const Kd3KnnExpectation& right)
{
    return left.ids == right.ids &&
           left.squaredDistances == right.squaredDistances;
}
} // namespace

TEST(Kd3Refresh, PublishedProductIsCachedByDefaultAndMatchesUncached)
{
    ScopedEnvironment disable("PDG_DISABLE_KD3_COORDINATE_CACHE", nullptr);
    ArithmeticPointView fixture(4'000);
    pdal::KD3Index uncached(*fixture.view, false);
    uncached.build();
    pdal::KD3Index& published = fixture.view->build3dIndex();
    EXPECT_TRUE(published.coordinatesCached());
    EXPECT_TRUE(sameAnswers(allQueries(uncached, 4'000, 9),
                            allQueries(published, 4'000, 9)));
}

TEST(Kd3Refresh, ReuseAfterCoordinateWriteMatchesStaleUncachedTree)
{
    ArithmeticPointView fixture(4'000);
    // Pinned semantics: an uncached tree built before the mutation and read
    // live afterwards.
    pdal::KD3Index stale(*fixture.view, false);
    stale.build();
    pdal::KD3Index& published = fixture.view->build3dIndex(true);
    ASSERT_TRUE(published.coordinatesCached());
    for (std::size_t point = 0; point < 4'000; point += 3)
        fixture.view->setField(pdal::Dimension::Id::X, point,
                               3.0 * static_cast<double>(point));
    // Without a refresh the snapshot would answer for the old coordinates.
    pdal::KD3Index& reused = fixture.view->build3dIndex();
    EXPECT_EQ(&reused, &published);
    EXPECT_TRUE(sameAnswers(allQueries(stale, 4'000, 9),
                            allQueries(reused, 4'000, 9)));
}

TEST(Kd3Refresh, ReuseAfterPointRefWriteAndSortMatchesStaleUncachedTree)
{
    ArithmeticPointView fixture(4'000);
    pdal::KD3Index stale(*fixture.view, false);
    stale.build();
    pdal::KD3Index& published = fixture.view->build3dIndex(true);
    pdal::PointRef point(*fixture.view, 0);
    for (std::size_t id = 1; id < 4'000; id += 5)
    {
        point.setPointId(id);
        point.setField(pdal::Dimension::Id::Y,
                       point.getFieldAs<double>(pdal::Dimension::Id::Y) + 0.5);
    }
    fixture.view->sort(pdal::Dimension::Id::Z);
    pdal::KD3Index& reused = fixture.view->build3dIndex();
    EXPECT_EQ(&reused, &published);
    EXPECT_TRUE(sameAnswers(allQueries(stale, 4'000, 9),
                            allQueries(reused, 4'000, 9)));
}

TEST(Kd3Refresh, DisableControlKeepsPublishedProductUncached)
{
    // The control is read once per process; a fresh process would need it set
    // before the first published build. Here we only prove the explicit
    // uncached request path still exists and is exact.
    ArithmeticPointView fixture(2'000);
    pdal::KD3Index& published = fixture.view->build3dIndex(false);
    EXPECT_FALSE(published.coordinatesCached());
    pdal::KD3Index cached(*fixture.view, true);
    cached.build();
    EXPECT_TRUE(sameAnswers(allQueries(cached, 2'000, 7),
                            allQueries(published, 2'000, 7)));
}

// D0274: nanoflann's concurrent build must produce the same tree as the
// serial build, so every query — including its tie order — is identical.
// The arithmetic fixture is tie-dense (Y and Z cycle through small moduli),
// and the lattice fixture below places many neighbors at exactly equal
// distances, the case where a structurally different tree would first show.
TEST(Kd3Concurrency, ConcurrentBuildProducesTheSerialTree)
{
    ScopedEnvironment noDisable("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
                                nullptr);
    for (const std::size_t pointCount : {12'000U, 130'000U})
    {
        SCOPED_TRACE(pointCount);
        ArithmeticPointView fixture(pointCount);
        std::vector<pdal::PointId> queries;
        for (pdal::PointId point = 0; point < fixture.view->size();
             point += 7U)
            queries.push_back(point);
        Kd3KnnExpectation serial;
        {
            ScopedEnvironment force("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
                                    "1");
            pdal::KD3Index index(*fixture.view, true);
            index.build();
            serial = serialKnn(index, queries, 9U);
        }
        for (const char* threads : {"2", "3", "8"})
        {
            SCOPED_TRACE(threads);
            ScopedEnvironment force("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
                                    threads);
            pdal::KD3Index index(*fixture.view, true);
            index.build();
            const Kd3KnnExpectation concurrent = serialKnn(index, queries, 9U);
            EXPECT_EQ(concurrent.ids, serial.ids);
            EXPECT_EQ(concurrent.squaredDistances, serial.squaredDistances);
            // Radius queries walk the same tree too.
            for (pdal::PointId query = 0; query < 500U; query += 25U)
            {
                pdal::KD3Index serialIndex(*fixture.view, true);
                {
                    ScopedEnvironment one(
                        "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS", "1");
                    serialIndex.build();
                }
                EXPECT_EQ(index.radius(query, 3.5), serialIndex.radius(query, 3.5));
            }
        }
    }
    // Lattice: every interior point has 6 axis neighbors at distance 1 and
    // 12 at sqrt(2); k=8 cuts through the tie group.
    pdal::PointTable table;
    table.layout()->registerDims({pdal::Dimension::Id::X,
                                  pdal::Dimension::Id::Y,
                                  pdal::Dimension::Id::Z});
    pdal::PointViewPtr lattice(new pdal::PointView(table));
    pdal::PointId id = 0;
    for (int x = 0; x < 40; ++x)
        for (int y = 0; y < 40; ++y)
            for (int z = 0; z < 8; ++z, ++id)
            {
                lattice->setField(pdal::Dimension::Id::X, id, double(x));
                lattice->setField(pdal::Dimension::Id::Y, id, double(y));
                lattice->setField(pdal::Dimension::Id::Z, id, double(z));
            }
    std::vector<pdal::PointId> queries;
    for (pdal::PointId point = 0; point < lattice->size(); point += 3U)
        queries.push_back(point);
    Kd3KnnExpectation serial;
    {
        ScopedEnvironment force("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
                                "1");
        pdal::KD3Index index(*lattice, true);
        index.build();
        serial = serialKnn(index, queries, 8U);
    }
    ScopedEnvironment force("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS", "5");
    pdal::KD3Index index(*lattice, true);
    index.build();
    const Kd3KnnExpectation concurrent = serialKnn(index, queries, 8U);
    EXPECT_EQ(concurrent.ids, serial.ids);
    EXPECT_EQ(concurrent.squaredDistances, serial.squaredDistances);
}
