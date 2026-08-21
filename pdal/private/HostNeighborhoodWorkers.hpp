/******************************************************************************
* Copyright (c) 2026, PDAL-GPU contributors
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#pragma once

// Exact host worker policy for per-point neighborhood passes (D0256/B0258).
//
// A pass qualifies only when every row's result depends on the read-only
// PointView-owned KD index and the read-only input columns of that row's
// query alone, so a fixed contiguous chunk assignment yields bit-identical
// per-row results at any worker count. Anything order-dependent (serial
// recurrences across rows, output application, diagnostics) stays serial in
// the caller. This mirrors the D0237 LOF repair pattern; the policy is the
// same: `PDG_NATIVE_WORKERS` is a positive cap, never a request to
// oversubscribe useful work, and `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS`
// restores the single-threaded pass as the same-final-binary control.

#include <pdal/pdal_types.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace pdal
{
namespace hostworkers
{

// Below this many rows per worker the thread lifecycle costs more than the
// bounded exact kNN work it would parallelize (same constant as LOF repair).
constexpr std::size_t MinimumRowsPerWorker = 4096U;

inline std::optional<std::size_t> positiveEnvironmentCount(const char* name)
{
    const char* configured = std::getenv(name);
    if (!configured || !*configured)
        return std::nullopt;
    const std::string_view text(configured);
    std::size_t value = 0U;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !value)
        throw std::invalid_argument(std::string(name) +
            " must be a positive integer");
    return value;
}

// Number of workers for `rows` independent per-row queries. Returns 1 for
// small inputs, when disabled, or when the machine reports one hardware
// thread. Never exceeds ceil(rows / MinimumRowsPerWorker), the hardware
// concurrency, or the PDG_NATIVE_WORKERS cap.
inline std::size_t neighborhoodWorkerCount(std::size_t rows)
{
    std::size_t workers = 1U;
    if (rows && !std::getenv("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS"))
    {
        const std::size_t usefulWorkers =
            rows / MinimumRowsPerWorker +
            static_cast<std::size_t>(rows % MinimumRowsPerWorker != 0U);
        const std::optional<std::size_t> configured =
            positiveEnvironmentCount("PDG_NATIVE_WORKERS");
        const std::size_t availableWorkers = configured.value_or(
            (std::max<std::size_t>)(1U, std::thread::hardware_concurrency()));
        workers = (std::max<std::size_t>)(1U,
            (std::min)(usefulWorkers, availableWorkers));
    }
#if defined(PDG_ENABLE_TEST_HOOKS)
    // Test-only: force a worker count on tiny fixtures so the chunked path
    // itself is differentially proven, and/or require the observed count.
    if (const std::optional<std::size_t> forced =
            positiveEnvironmentCount("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS"))
        workers = rows ? (std::min)(*forced, rows) : 1U;
    if (const std::optional<std::size_t> required =
            positiveEnvironmentCount(
                "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS"))
        if (*required != workers)
            throw std::runtime_error(
                "Required host neighborhood worker count " +
                std::to_string(*required) + ", observed " +
                std::to_string(workers) + ".");
#endif
    return workers;
}

// The same policy for any exact fixed-chunk row/cell pass (SMRF grid
// morphology, B0264): the name of the disable control is historical and
// covers every host worker pass governed by this header.
inline std::size_t rowWorkerCount(std::size_t rows)
{
    return neighborhoodWorkerCount(rows);
}

// Runs `function(worker, begin, end)` over fixed contiguous [begin, end) row
// ranges, ranked by worker. With one worker the caller's thread runs the
// whole range as worker 0. Every worker's
// first exception is captured; after all workers join, the lowest-ranked
// worker's exception is rethrown, which is the first offending row in
// original order because ranges ascend by worker rank.
template <typename Function>
void forEachRowChunk(std::size_t rows, std::size_t workerCount,
    Function&& function)
{
    if (workerCount <= 1U || rows == 0U)
    {
        function(static_cast<std::size_t>(0U), static_cast<std::size_t>(0U),
            rows);
        return;
    }
    workerCount = (std::min)(workerCount, rows);

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(workerCount);
    workers.reserve(workerCount);
    const std::size_t rowsPerWorker = rows / workerCount;
    const std::size_t remainder = rows % workerCount;
    try
    {
        for (std::size_t worker = 0U; worker < workerCount; ++worker)
        {
            const std::size_t begin =
                worker * rowsPerWorker + (std::min)(worker, remainder);
            const std::size_t end = begin + rowsPerWorker +
                static_cast<std::size_t>(worker < remainder);
            workers.emplace_back([&, worker, begin, end]
            {
                try
                {
                    function(worker, begin, end);
                }
                catch (...)
                {
                    errors[worker] = std::current_exception();
                }
            });
        }
    }
    catch (...)
    {
        for (std::thread& worker : workers)
            worker.join();
        throw;
    }
    for (std::thread& worker : workers)
        worker.join();
    for (const std::exception_ptr& error : errors)
        if (error)
            std::rethrow_exception(error);
}

} // namespace hostworkers
} // namespace pdal
