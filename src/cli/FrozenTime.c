#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <dlfcn.h>
#endif

static int frozen_epoch(time_t* value)
{
    const char* text = getenv("PDAL_TEST_FROZEN_EPOCH");
    /* Historical manual replays used the product-namespace spelling. */
    if (!text || !*text)
        text = getenv("PDG_FROZEN_EPOCH");
    if (!text || !*text)
        return 0;
    char* end = NULL;
    const long long parsed = strtoll(text, &end, 10);
    if (!end || *end)
        return 0;
    *value = (time_t)parsed;
    return 1;
}

time_t time(time_t* result)
{
    time_t value;
    if (frozen_epoch(&value))
    {
        if (result)
            *result = value;
        return value;
    }
#ifdef __linux__
    struct timespec now;
    if (syscall(SYS_clock_gettime, CLOCK_REALTIME, &now) != 0)
        return (time_t)-1;
    if (result)
        *result = now.tv_sec;
    return now.tv_sec;
#else
    time_t (*real_time)(time_t*) =
        (time_t (*)(time_t*))dlsym(RTLD_NEXT, "time");
    return real_time(result);
#endif
}

int clock_gettime(clockid_t clock, struct timespec* result)
{
    time_t value;
    if (clock == CLOCK_REALTIME && frozen_epoch(&value))
    {
        result->tv_sec = value;
        result->tv_nsec = 0;
        return 0;
    }
#ifdef __linux__
    return (int)syscall(SYS_clock_gettime, clock, result);
#else
    int (*real_clock_gettime)(clockid_t, struct timespec*) =
        (int (*)(clockid_t, struct timespec*))dlsym(RTLD_NEXT, "clock_gettime");
    return real_clock_gettime(clock, result);
#endif
}

int gettimeofday(struct timeval* result, void* timezone)
{
    time_t value;
    if (frozen_epoch(&value))
    {
        result->tv_sec = value;
        result->tv_usec = 0;
        return 0;
    }
#ifdef __linux__
    return (int)syscall(SYS_gettimeofday, result, timezone);
#else
    int (*real_gettimeofday)(struct timeval*, void*) =
        (int (*)(struct timeval*, void*))dlsym(RTLD_NEXT, "gettimeofday");
    return real_gettimeofday(result, timezone);
#endif
}
