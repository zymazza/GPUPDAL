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

// A small persistent pool with fixed worker slots (D0266/B0266). Unlike
// pdal::ThreadPool, every run() invokes the task once per slot on the same
// OS thread that owns that slot, so per-slot state that must stay bound to
// one thread (a cloned GDAL/PROJ coordinate transformation) can be created
// lazily inside the slot and reused across batches. Exceptions thrown by a
// slot are captured and the lowest slot's is rethrown after every slot has
// finished, so the first failing row in original order surfaces first when
// slots own ascending contiguous ranges.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pdal
{
namespace hostworkers
{

class HostSlotPool
{
public:
    explicit HostSlotPool(std::size_t slots) : m_slots(slots ? slots : 1U)
    {
        m_errors.resize(m_slots);
        m_threads.reserve(m_slots);
        for (std::size_t slot = 0U; slot < m_slots; ++slot)
            m_threads.emplace_back([this, slot] { loop(slot); });
    }

    ~HostSlotPool()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
            ++m_generation;
        }
        m_wake.notify_all();
        for (std::thread& thread : m_threads)
            thread.join();
    }

    HostSlotPool(const HostSlotPool&) = delete;
    HostSlotPool& operator=(const HostSlotPool&) = delete;

    std::size_t slots() const noexcept
    {
        return m_slots;
    }

    // Runs task(slot) on every slot and returns after all have finished,
    // rethrowing the lowest slot's exception if any slot threw.
    void run(const std::function<void(std::size_t)>& task)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_task = &task;
            m_pending = m_slots;
            for (std::exception_ptr& error : m_errors)
                error = nullptr;
            ++m_generation;
        }
        m_wake.notify_all();
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_done.wait(lock, [this] { return m_pending == 0U; });
            m_task = nullptr;
        }
        for (const std::exception_ptr& error : m_errors)
            if (error)
                std::rethrow_exception(error);
    }

private:
    void loop(std::size_t slot)
    {
        std::uint64_t seen = 0U;
        for (;;)
        {
            const std::function<void(std::size_t)>* task = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, [&] { return m_generation != seen; });
                seen = m_generation;
                if (m_stop)
                    return;
                task = m_task;
            }
            if (task)
            {
                try
                {
                    (*task)(slot);
                }
                catch (...)
                {
                    m_errors[slot] = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (--m_pending == 0U)
                    m_done.notify_all();
            }
        }
    }

    std::size_t m_slots;
    std::vector<std::thread> m_threads;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    const std::function<void(std::size_t)>* m_task = nullptr;
    std::vector<std::exception_ptr> m_errors;
    std::uint64_t m_generation = 0U;
    std::size_t m_pending = 0U;
    bool m_stop = false;
};

// Contiguous [begin, end) of `rows` for `slot` of `slots`, ascending by slot.
inline void slotRange(std::size_t rows, std::size_t slots, std::size_t slot,
    std::size_t& begin, std::size_t& end)
{
    const std::size_t rowsPerSlot = rows / slots;
    const std::size_t remainder = rows % slots;
    begin = slot * rowsPerSlot + (slot < remainder ? slot : remainder);
    end = begin + rowsPerSlot + (slot < remainder ? 1U : 0U);
}

} // namespace hostworkers
} // namespace pdal
