// include OS specific timing library
#ifdef _MSC_VER
    // Windows
    #include <Windows.h>
#else
    // Linux
    #include <time.h>
#endif


double get_ts_ns()
{
#ifdef _MSC_VER
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0)
    QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double(now.QuadPart)*1000000000.0) / double(frequency.QuadPart);
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return double(now.tv_sec)*1000000000.0 + double(now.tv_nsec);
#endif
}
