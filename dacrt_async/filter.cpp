#include "filter.h"
#include <x86intrin.h>
#include "timer.h"
#include <iostream>

inline __m128 _mm_sel_ps(const __m128& a, const __m128& b, const __m128& mask)
{
    // (((b ^ a) & mask)^a)
    return _mm_xor_ps( a, _mm_and_ps( mask, _mm_xor_ps( b, a ) ) );
}

#define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
do { \
	__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
	__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
	__t0 = _mm256_unpacklo_ps(row0, row1); \
	__t1 = _mm256_unpackhi_ps(row0, row1); \
	__t2 = _mm256_unpacklo_ps(row2, row3); \
	__t3 = _mm256_unpackhi_ps(row2, row3); \
	__t4 = _mm256_unpacklo_ps(row4, row5); \
	__t5 = _mm256_unpackhi_ps(row4, row5); \
	__t6 = _mm256_unpacklo_ps(row6, row7); \
	__t7 = _mm256_unpackhi_ps(row6, row7); \
	__tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0)); \
	__tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2)); \
	__tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0)); \
	__tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2)); \
	__tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0)); \
	__tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2)); \
	__tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0)); \
	__tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2)); \
	row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
	row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
	row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
	row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
	row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
	row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
	row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
	row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31); \
}while(0);

inline void print128(__m128 v)
{
	for(int i = 0; i < 3; ++i)
	{
		std::cout << ((float*)&v)[i] << ", ";
	}
	std::cout << ((float*)&v)[3] << std::endl;
}

inline void print256(__m256 v)
{
	for(int i = 0; i < 7; ++i)
	{
		std::cout << ((float*)&v)[i] << ", ";
	}
	std::cout << ((float*)&v)[7] << std::endl;
}

inline void filterRaysSeq(const Box& box, Ray* rays, int begin, int end, int& rendNew)
{
	for(int i = begin; i < end; ++i)
	{
		const Ray& ray = rays[i];
		const vec3 invDir = reciprocal(ray.d);

		const float tx1 = (box.min.x - ray.o.x) * invDir.x;
		const float tx2 = (box.max.x - ray.o.x) * invDir.x;

		float tmin = std::min(tx1, tx2);
		float tmax = std::max(tx1, tx2);

		const float ty1 = (box.min.y - ray.o.y) * invDir.y;
		const float ty2 = (box.max.y - ray.o.y) * invDir.y;

		tmin = std::max(tmin, std::min(ty1, ty2));
		tmax = std::min(tmax, std::max(ty1, ty2));

		const float tz1 = (box.min.z - ray.o.z) * invDir.z;
		const float tz2 = (box.max.z - ray.o.z) * invDir.z;

		tmin = std::max(tmin, std::min(tz1, tz2));
		tmax = std::min(tmax, std::max(tz1, tz2));

		if(tmin > tmax) continue;

		std::swap(rays[i], rays[rendNew]);
		++rendNew;
	}
}

void filterRays(const Box& box, Ray* rays, int& rend)
{
	int rendNew = 0;
	filterRaysSeq(box, rays, 0, rend, rendNew);
	rend = rendNew;
}

void filterRaysSSE(const Box& box, Ray* rays, int& rend)
{
//	timer t;

	int rendNew = 0;
	int simdLimit = rend/4 * 4;
	int seqLimit = simdLimit + rend % 4;

	__m128 bmin = _mm_load_ps(box.min.ptr());
	__m128 bmax = _mm_load_ps(box.max.ptr());

	__m128 bminx = _mm_shuffle_ps(bmin, bmin, _MM_SHUFFLE(0,0,0,0));
	__m128 bminy = _mm_shuffle_ps(bmin, bmin, _MM_SHUFFLE(1,1,1,1));
	__m128 bminz = _mm_shuffle_ps(bmin, bmin, _MM_SHUFFLE(2,2,2,2));

	__m128 bmaxx = _mm_shuffle_ps(bmax, bmax, _MM_SHUFFLE(0,0,0,0));
	__m128 bmaxy = _mm_shuffle_ps(bmax, bmax, _MM_SHUFFLE(1,1,1,1));
	__m128 bmaxz = _mm_shuffle_ps(bmax, bmax, _MM_SHUFFLE(2,2,2,2));

	for(int i = 0; i < simdLimit; i+=4)
	{
		__m128 rox = _mm_load_ps(&rays[i+0].o.x);
		__m128 rdx = _mm_load_ps(&rays[i+0].d.x);
		__m128 roy = _mm_load_ps(&rays[i+1].o.x);
		__m128 rdy = _mm_load_ps(&rays[i+1].d.x);
		__m128 roz = _mm_load_ps(&rays[i+2].o.x);
		__m128 rdz = _mm_load_ps(&rays[i+2].d.x);
		__m128 row = _mm_load_ps(&rays[i+3].o.x);
		__m128 rdw = _mm_load_ps(&rays[i+3].d.x);

		_MM_TRANSPOSE4_PS(rox, roy, roz, row);
		_MM_TRANSPOSE4_PS(rdx, rdy, rdz, rdw);

		__m128 ridx = _mm_rcp_ps(rdx);
		__m128 ridy = _mm_rcp_ps(rdy);
		__m128 ridz = _mm_rcp_ps(rdz);

		__m128 tx1 = _mm_mul_ps(_mm_sub_ps(bminx, rox), ridx);
		__m128 tx2 = _mm_mul_ps(_mm_sub_ps(bmaxx, rox), ridx);

		__m128 tmin = _mm_min_ps(tx1, tx2);
		__m128 tmax = _mm_max_ps(tx1, tx2);

		__m128 ty1 = _mm_mul_ps(_mm_sub_ps(bminy, roy), ridy);
		__m128 ty2 = _mm_mul_ps(_mm_sub_ps(bmaxy, roy), ridy);

		tmin = _mm_max_ps(tmin, _mm_min_ps(ty1, ty2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(ty1, ty2));

		__m128 tz1 = _mm_mul_ps(_mm_sub_ps(bminz, roz), ridz);
		__m128 tz2 = _mm_mul_ps(_mm_sub_ps(bmaxz, roz), ridz);

		tmin = _mm_max_ps(tmin, _mm_min_ps(tz1, tz2));
		tmax = _mm_min_ps(tmax, _mm_max_ps(tz1, tz2));

		__m128 cmp = _mm_cmple_ps(tmin, tmax);

		int mask = _mm_movemask_ps(cmp);

		if(mask & 0x1)
		{
			std::swap(rays[i+0], rays[rendNew]);
			++rendNew;
		}
		if(mask & 0x2)
		{
			std::swap(rays[i+1], rays[rendNew]);
			++rendNew;
		}
		if(mask & 0x4)
		{
			std::swap(rays[i+2], rays[rendNew]);
			++rendNew;
		}
		if(mask & 0x8)
		{
			std::swap(rays[i+3], rays[rendNew]);
			++rendNew;
		}
	}

	filterRaysSeq(box, rays, simdLimit, seqLimit, rendNew);

	rend = rendNew;

//	std::cout << t.msec() << " " << rend << std::endl;
//	exit(1);
}

void filterRaysAVX(const Box& box, Ray* rays, int& rend)
{
//	timer t;

	int rendNew = 0;
	int simdLimit = rend/8 * 8;
	int seqLimit = simdLimit + rend % 8;

	__m256 b = _mm256_load_ps(&box.min.x);

	__m256 bx = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(0,0,0,0));
	__m256 by = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(1,1,1,1));
	__m256 bz = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2,2,2,2));

	__m128 l = _mm256_extractf128_ps(bx, 0);
	__m256 bminx = _mm256_broadcast_ps(&l);
	l = _mm256_extractf128_ps(by, 0);
	__m256 bminy = _mm256_broadcast_ps(&l);
	l = _mm256_extractf128_ps(bz, 0);
	__m256 bminz = _mm256_broadcast_ps(&l);

	l = _mm256_extractf128_ps(bx, 1);
	__m256 bmaxx = _mm256_broadcast_ps(&l);
	l = _mm256_extractf128_ps(by, 1);
	__m256 bmaxy = _mm256_broadcast_ps(&l);
	l = _mm256_extractf128_ps(bz, 1);
	__m256 bmaxz = _mm256_broadcast_ps(&l);

	for(int i = 0; i < simdLimit; i+=8)
	{
		__m256 rox = _mm256_load_ps(&rays[i+0].o.x);
		__m256 roy = _mm256_load_ps(&rays[i+1].o.x);
		__m256 roz = _mm256_load_ps(&rays[i+2].o.x);
		__m256 row = _mm256_load_ps(&rays[i+3].o.x);
		__m256 rdx = _mm256_load_ps(&rays[i+4].o.x);
		__m256 rdy = _mm256_load_ps(&rays[i+5].o.x);
		__m256 rdz = _mm256_load_ps(&rays[i+6].o.x);
		__m256 rdw = _mm256_load_ps(&rays[i+7].o.x);

		_MM_TRANSPOSE8_PS(rox, roy, roz, row, rdx, rdy, rdz, rdw);

		__m256 ridx = _mm256_rcp_ps(rdx);
		__m256 ridy = _mm256_rcp_ps(rdy);
		__m256 ridz = _mm256_rcp_ps(rdz);

		__m256 tx1 = _mm256_mul_ps(_mm256_sub_ps(bminx, rox), ridx);
		__m256 tx2 = _mm256_mul_ps(_mm256_sub_ps(bmaxx, rox), ridx);

		__m256 tmin = _mm256_min_ps(tx1, tx2);
		__m256 tmax = _mm256_max_ps(tx1, tx2);

		__m256 ty1 = _mm256_mul_ps(_mm256_sub_ps(bminy, roy), ridy);
		__m256 ty2 = _mm256_mul_ps(_mm256_sub_ps(bmaxy, roy), ridy);

		tmin = _mm256_max_ps(tmin, _mm256_min_ps(ty1, ty2));
		tmax = _mm256_min_ps(tmax, _mm256_max_ps(ty1, ty2));

		__m256 tz1 = _mm256_mul_ps(_mm256_sub_ps(bminz, roz), ridz);
		__m256 tz2 = _mm256_mul_ps(_mm256_sub_ps(bmaxz, roz), ridz);

		tmin = _mm256_max_ps(tmin, _mm256_min_ps(tz1, tz2));
		tmax = _mm256_min_ps(tmax, _mm256_max_ps(tz1, tz2));

		__m256 cmp = _mm256_cmp_ps(tmin, tmax, _CMP_LE_OQ);

		int mask = _mm256_movemask_ps(cmp);

		if(mask & 1)
		{
			std::swap(rays[i+0], rays[rendNew]);
			++rendNew;
		}
		if(mask & 2)
		{
			std::swap(rays[i+1], rays[rendNew]);
			++rendNew;
		}
		if(mask & 4)
		{
			std::swap(rays[i+2], rays[rendNew]);
			++rendNew;
		}
		if(mask & 8)
		{
			std::swap(rays[i+3], rays[rendNew]);
			++rendNew;
		}
		if(mask & 16)
		{
			std::swap(rays[i+4], rays[rendNew]);
			++rendNew;
		}
		if(mask & 32)
		{
			std::swap(rays[i+5], rays[rendNew]);
			++rendNew;
		}
		if(mask & 64)
		{
			std::swap(rays[i+6], rays[rendNew]);
			++rendNew;
		}
		if(mask & 128)
		{
			std::swap(rays[i+7], rays[rendNew]);
			++rendNew;
		}
	}

	filterRaysSeq(box, rays, simdLimit, seqLimit, rendNew);

	rend = rendNew;

//	std::cout << t.msec() << " " << rend << std::endl;
//	exit(1);
}
