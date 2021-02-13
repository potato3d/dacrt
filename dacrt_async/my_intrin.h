#pragma once
#include <x86intrin.h>

inline __m128 _mm_hmin_ps(__m128 a)
{
	__m128 t = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,3,0,1));
	a = _mm_min_ps(a, t);
	t = _mm_movehl_ps(a, a);
	return _mm_min_ps(a, t);
}

inline __m128 _mm_hmax_ps(__m128 a)
{
	__m128 t = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,3,0,1));
	a = _mm_max_ps(a, t);
	t = _mm_movehl_ps(a, a);
	return _mm_max_ps(a, t);
}
