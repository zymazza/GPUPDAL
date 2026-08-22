#pragma once

#ifdef _WIN32

#include <cstdlib>
#include <process.h>

inline int setenv(const char* name, const char* value, int overwrite)
{
    if (!overwrite && std::getenv(name))
        return 0;
    return ::_putenv_s(name, value);
}

inline int unsetenv(const char* name)
{
    return ::_putenv_s(name, "");
}

#endif
