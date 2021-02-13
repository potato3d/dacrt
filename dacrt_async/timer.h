#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

class timer
{
public:
	timer() { restart(); }
	void restart()
	{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		LARGE_INTEGER li;

		if (!QueryPerformanceFrequency(&li))
		{
			printf("QueryPerformanceFrequency failed!\n");
		}

		PCFreq = (double)li.QuadPart/1000.0;
		QueryPerformanceCounter(&li);
		_start = li.QuadPart;
#else
    	gettimeofday(&_start, 0);
#endif
	}
	double msec()
	{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-_start)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, 0);
    timersub(&timerStop, &_start, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
	}
	double sec() { return msec() * 1e-3; }
	double usec() { return msec() * 1e3; }
	double nsec() { return msec() * 1e6; }

private:
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	typedef __int64 time_point;
#else
	typedef struct timeval time_point;
#endif
	time_point _start;
};
