#pragma once
#include <mm_malloc.h>

template<typename T>
inline T* amalloc(int count)
{
	return (T*)_mm_malloc(count*sizeof(T), 16);
}

inline void afree(void* p)
{
	_mm_free(p);
}
