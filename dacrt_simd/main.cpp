#include "rtv.h"
#include "timer.h"
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <vector>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <x86intrin.h>
#include <mm_malloc.h>

#define HIT_EPSILON 1e-6f

#define TRI_LIMIT 8
#define RAY_LIMIT 8

#define _MM_ALIGN(i) __attribute__((__align__(i)))

#define _mm_get_ps_0(v) _mm_cvtss_f32(v)
#define _mm_get_ps_1(v) _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)))
#define _mm_get_ps_2(v) _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)))
#define _mm_get_ps_3(v) _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)))
#define _mm_get_ps(v, i) _mm_get_ps_##i(v)

#define _mm_get_si128_0(v) _mm_cvtsi128_si32(v)
#define _mm_get_si128_1(v, i) _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1)))
#define _mm_get_si128_2(v, i) _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(2, 2, 2, 2)))
#define _mm_get_si128_3(v, i) _mm_cvtsi128_si32(_mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 3, 3)))
#define _mm_get_si128(v, i) _mm_get_si128_##i(v)

inline int nextMultipleOf4(int a)
{
	return (a + 3) & ~0x03;
}

struct vec3
{
	float x,y,z;
	inline vec3() {}
	inline explicit vec3(float a) : x(a), y(a), z(a) {}
	inline explicit vec3(const float* const p) : x(p[0]), y(p[1]), z(p[2]) {}
	inline vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	inline float operator[](int i) const {return (&x)[i];}
	inline vec3 operator-() {return vec3(-x, -y, -z);}
	inline vec3 operator*(float a) const {return vec3(x*a, y*a, z*a);}
	inline vec3 operator+(const vec3& o) const {return vec3(x+o.x, y+o.y, z+o.z);}
	inline vec3 operator-(const vec3& o) const {return vec3(x-o.x, y-o.y, z-o.z);}
	inline void operator+=(const vec3& o) {x+=o.x; y+=o.y; z+=o.z;}
	inline void operator*=(float a) {x*=a; y*=a; z*=a;}
	inline float operator[](int i) {return (&x)[i];}
};
inline float dot(const vec3& a, const vec3& b) {return a.x*b.x + a.y*b.y + a.z*b.z;}
inline float len(const vec3& v) {return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);}
inline vec3 normalize(const vec3& v) {float il = 1.0f / len(v); return vec3(v.x*il, v.y*il, v.z*il);}
inline vec3 cross(const vec3& a, const vec3& b) {return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}
inline vec3 reciprocal(const vec3& v) {return vec3(1.0f/v.x, 1.0f/v.y, 1.0f/v.z);}

struct TriV
{
	vec3 v0; int pad0;
	vec3 v1; int pad1;
	vec3 v2; int pad2;
};

struct TriN
{
	vec3 n0; int pad0;
	vec3 n1; int pad1;
	vec3 n2; int pad2;
};

struct Ray
{
	vec3 o;
	float tmax;
	vec3 d;
	int id;
};

struct simd4
{
	union
	{
		struct {vec3 v; int _;};
		__m128 m;
	};
	simd4(float a) : m(_mm_set1_ps(a)) {}
	simd4(__m128 a) : m(a) {}
};

struct Box
{
	Box() : min(FLT_MAX), max(-FLT_MAX) {}
	simd4 min;
	simd4 max;
	inline void expand(const vec3& v)
	{
		min.v.x = std::min(min.v.x, v.x);
		min.v.y = std::min(min.v.y, v.y);
		min.v.z = std::min(min.v.z, v.z);

		max.v.x = std::max(max.v.x, v.x);
		max.v.y = std::max(max.v.y, v.y);
		max.v.z = std::max(max.v.z, v.z);
	}
	inline void expand(const TriV& t)
	{
		expand(t.v0);
		expand(t.v1);
		expand(t.v2);
	}
	inline void expand(const Box& b)
	{
		min.m = _mm_min_ps(b.min.m, min.m);
		max.m = _mm_max_ps(b.max.m, max.m);
	}
	inline void expand(const simd4& a)
	{
		min.m = _mm_min_ps(a.m, min.m);
		max.m = _mm_max_ps(a.m, max.m);
	}
	inline simd4 computeCenter() const
	{
		return _mm_mul_ps(_mm_add_ps(min.m, max.m), _mm_set1_ps(0.5f));
	}
};

struct TriSet
{
	TriSet() : verts(0), norms(0), boxes(0), count(0) {}
	Box boxTris;
	Box boxCenters;
	int* ids;
	TriV* verts;
	TriN* norms;
	Box* boxes;
	int count;
};

struct RaySet
{
	RaySet() : rays(0), hits(0), count(0) {}
	Ray* rays;
	int* hits;
	int count;
};

struct Canvas
{
	int w;
	int h;
};

struct Camera
{
	vec3 position;
	vec3 lowerLeftDir;
	vec3 du;
	vec3 dv;
	int nu;
	int nv;
};

static Canvas g_canvas;
static Camera g_camera;

static TriSet g_tris;
static RaySet g_rays;

static timer g_timer;
static double g_filterTime = 0.0;
static double g_intersectTime = 0.0;
static double g_partitionTime = 0.0;

inline float xorf(float f, unsigned int mask)
{
	unsigned int r = (unsigned int&)f ^ mask;
	return (float&)r;
}

inline unsigned int float_as_uint(float f)
{
	union {float a; unsigned int b;};
	a = f;
	return b;
}

inline void intersectAfra(int tbegin, int tend, int rend)
{
	for(int t = tbegin; t < tend; ++t)
	{
		const int tid = g_tris.ids[t];
		const TriV& tri = g_tris.verts[tid];

		const vec3 edge1 = tri.v0 - tri.v1;
		const vec3 edge2 = tri.v2 - tri.v0;
		const vec3 normal = cross(edge1, edge2);

		for(int r = 0; r < rend; ++r)
		{
			Ray& ray = g_rays.rays[r];

			const vec3 cvec = tri.v0 - ray.o;
			const vec3 rvec = cross(ray.d, cvec);
			const float det = dot(normal, ray.d);
			const float a = std::abs(det);
			const unsigned int detSign = (unsigned int&)det & 0x80000000;
			const float ta = xorf(dot(normal, cvec), detSign);
			const float ua = xorf(dot(rvec, edge2), detSign);
			const float va = xorf(dot(rvec, edge1), detSign);
			const float wa = a - ua - va;

			const bool mask = (det != 0.0f) && (ta >= 0.0f) && (ta <= a*ray.tmax) && (std::min(std::min(ua, va), wa) >= 0.0f);

			if(!mask) continue;

			ray.tmax = ta / a;
			g_rays.hits[ray.id] = tid;
		}
	}
}

inline void intersect(int tbegin, int tend, int rend)
{
	for(int t = tbegin; t < tend; ++t)
	{
		const int tid = g_tris.ids[t];
		const TriV& tri = g_tris.verts[tid];

		const vec3 edge1 = tri.v1 - tri.v0;
		const vec3 edge2 = tri.v2 - tri.v0;

		for(int r = 0; r < rend; ++r)
		{
			Ray& ray = g_rays.rays[r];

			const vec3 pvec = cross(ray.d, edge2);
			const float det = dot(edge1, pvec);

			if(det > -HIT_EPSILON && det < HIT_EPSILON) continue;

			const float inv_det = 1.0f / det;
			const vec3 tvec = ray.o - tri.v0;
			const float u = dot(tvec, pvec) * inv_det;

			if(u < 0.0f || u > 1.0f) continue;

			const vec3 qvec = cross(tvec, edge1);
			const float v = dot(ray.d, qvec) * inv_det;

			if(v < 0.0f || u + v > 1.0f) continue;

			const float f = dot(edge2, qvec) * inv_det;
			if(f >= ray.tmax || f < -HIT_EPSILON) continue;

			ray.tmax = f;
			g_rays.hits[ray.id] = tid;
		}
	}
}

inline void intersectSSEAfra(int tbegin, int tend, int rend)
{
	const int rend4 = nextMultipleOf4(rend);

	for(int t = tbegin; t < tend; ++t)
	{
		const int tid = g_tris.ids[t];
		const TriV& tri = g_tris.verts[tid];

		const __m128 v0x = _mm_set1_ps(tri.v0.x);
		const __m128 v0y = _mm_set1_ps(tri.v0.y);
		const __m128 v0z = _mm_set1_ps(tri.v0.z);

		const __m128 e1x = _mm_sub_ps(v0x, _mm_set1_ps(tri.v1.x));
		const __m128 e1y = _mm_sub_ps(v0y, _mm_set1_ps(tri.v1.y));
		const __m128 e1z = _mm_sub_ps(v0z, _mm_set1_ps(tri.v1.z));

		const __m128 e2x = _mm_sub_ps(_mm_set1_ps(tri.v2.x), v0x);
		const __m128 e2y = _mm_sub_ps(_mm_set1_ps(tri.v2.y), v0y);
		const __m128 e2z = _mm_sub_ps(_mm_set1_ps(tri.v2.z), v0z);

		const __m128 tnx = _mm_sub_ps(_mm_mul_ps(e1y, e2z), _mm_mul_ps(e1z, e2y));
		const __m128 tny = _mm_sub_ps(_mm_mul_ps(e1z, e2x), _mm_mul_ps(e1x, e2z));
		const __m128 tnz = _mm_sub_ps(_mm_mul_ps(e1x, e2y), _mm_mul_ps(e1y, e2x));

		for(int r = 0; r < rend4; r+=4)
		{
			__m128 rox = _mm_load_ps(&g_rays.rays[r+0].o.x);
			__m128 rdx = _mm_load_ps(&g_rays.rays[r+0].d.x);
			__m128 roy = _mm_load_ps(&g_rays.rays[r+1].o.x);
			__m128 rdy = _mm_load_ps(&g_rays.rays[r+1].d.x);
			__m128 roz = _mm_load_ps(&g_rays.rays[r+2].o.x);
			__m128 rdz = _mm_load_ps(&g_rays.rays[r+2].d.x);
			__m128 tmax = _mm_load_ps(&g_rays.rays[r+3].o.x);
			__m128 id = _mm_load_ps(&g_rays.rays[r+3].d.x);

			_MM_TRANSPOSE4_PS(rox, roy, roz, tmax);
			_MM_TRANSPOSE4_PS(rdx, rdy, rdz, id);

			_mm_prefetch((const char*)&g_rays.rays[r+16], _MM_HINT_T0);
			_mm_prefetch((const char*)&g_rays.rays[r+18], _MM_HINT_T0);

			const __m128 cvx = _mm_sub_ps(v0x, rox);
			const __m128 cvy = _mm_sub_ps(v0y, roy);
			const __m128 cvz = _mm_sub_ps(v0z, roz);

			const __m128 rvx = _mm_sub_ps(_mm_mul_ps(rdy, cvz), _mm_mul_ps(rdz, cvy));
			const __m128 rvy = _mm_sub_ps(_mm_mul_ps(rdz, cvx), _mm_mul_ps(rdx, cvz));
			const __m128 rvz = _mm_sub_ps(_mm_mul_ps(rdx, cvy), _mm_mul_ps(rdy, cvx));

			const __m128 det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(tnx, rdx), _mm_mul_ps(tny, rdy)), _mm_mul_ps(tnz, rdz));

			const __m128 a = _mm_andnot_ps(_mm_set1_ps(-0.0f), det);

			const __m128 detSign = _mm_and_ps(det, _mm_set1_ps(-0.0f));

			const __m128 ta = _mm_xor_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(tnx, cvx), _mm_mul_ps(tny, cvy)), _mm_mul_ps(tnz, cvz)), detSign);
			const __m128 ua = _mm_xor_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rvx, e2x), _mm_mul_ps(rvy, e2y)), _mm_mul_ps(rvz, e2z)), detSign);
			const __m128 va = _mm_xor_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rvx, e1x), _mm_mul_ps(rvy, e1y)), _mm_mul_ps(rvz, e1z)), detSign);
			const __m128 wa = _mm_sub_ps(_mm_sub_ps(a, ua), va);

			const __m128 mask = _mm_and_ps(_mm_cmpneq_ps(det, _mm_setzero_ps()),
					                       _mm_and_ps(_mm_cmpge_ps(ta, _mm_setzero_ps()),
					                    		      _mm_and_ps(_mm_cmple_ps(ta, _mm_mul_ps(a, tmax)),
					                    		    		     _mm_cmpge_ps(_mm_min_ps(_mm_min_ps(ua, va), wa), _mm_setzero_ps()))));

			const int mmask = _mm_movemask_ps(mask);
			if(!mmask) continue;

			const __m128 aRcp = _mm_rcp_ps(a);

			const __m128 f = _mm_mul_ps(ta, aRcp);

			if(mmask & 0x1)
			{
				g_rays.rays[r+0].tmax = _mm_get_ps(f, 0);
				const int rid = g_rays.rays[r+0].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x2)
			{
				g_rays.rays[r+1].tmax = _mm_get_ps(f, 1);
				const int rid = g_rays.rays[r+1].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x4)
			{
				g_rays.rays[r+2].tmax = _mm_get_ps(f, 2);
				const int rid = g_rays.rays[r+2].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x8)
			{
				g_rays.rays[r+3].tmax = _mm_get_ps(f, 3);
				const int rid = g_rays.rays[r+3].id;
				g_rays.hits[rid] = tid;
			}
		}
	}
}

inline void intersectSSE(int tbegin, int tend, int rend)
{
	const int rend4 = nextMultipleOf4(rend);

	for(int t = tbegin; t < tend; ++t)
	{
		const int tid = g_tris.ids[t];
		const TriV& tri = g_tris.verts[tid];

		__m128 v0x = _mm_set1_ps(tri.v0.x);
		__m128 v0y = _mm_set1_ps(tri.v0.y);
		__m128 v0z = _mm_set1_ps(tri.v0.z);

		__m128 v1x = _mm_set1_ps(tri.v1.x);
		__m128 v1y = _mm_set1_ps(tri.v1.y);
		__m128 v1z = _mm_set1_ps(tri.v1.z);

		__m128 v2x = _mm_set1_ps(tri.v2.x);
		__m128 v2y = _mm_set1_ps(tri.v2.y);
		__m128 v2z = _mm_set1_ps(tri.v2.z);

		// const vec3 edge1 = triv.v[1] - triv.v[0];
		__m128 e1x = _mm_sub_ps(v1x, v0x);
		__m128 e1y = _mm_sub_ps(v1y, v0y);
		__m128 e1z = _mm_sub_ps(v1z, v0z);

		// const vec3 edge2 = triv.v[2] - triv.v[0];
		__m128 e2x = _mm_sub_ps(v2x, v0x);
		__m128 e2y = _mm_sub_ps(v2y, v0y);
		__m128 e2z = _mm_sub_ps(v2z, v0z);

		for(int r = 0; r < rend4; r+=4)
		{
			__m128 rox = _mm_load_ps(&g_rays.rays[r+0].o.x);
			__m128 rdx = _mm_load_ps(&g_rays.rays[r+0].d.x);
			__m128 roy = _mm_load_ps(&g_rays.rays[r+1].o.x);
			__m128 rdy = _mm_load_ps(&g_rays.rays[r+1].d.x);
			__m128 roz = _mm_load_ps(&g_rays.rays[r+2].o.x);
			__m128 rdz = _mm_load_ps(&g_rays.rays[r+2].d.x);
			__m128 tmax = _mm_load_ps(&g_rays.rays[r+3].o.x);
			__m128 id = _mm_load_ps(&g_rays.rays[r+3].d.x);

			_MM_TRANSPOSE4_PS(rox, roy, roz, tmax);
			_MM_TRANSPOSE4_PS(rdx, rdy, rdz, id);

			_mm_prefetch((const char*)&g_rays.rays[r+16], _MM_HINT_T0);
			_mm_prefetch((const char*)&g_rays.rays[r+18], _MM_HINT_T0);

			// const vec3 pvec = cross(ray.d, edge2); a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x
			__m128 pvx = _mm_sub_ps(_mm_mul_ps(rdy, e2z), _mm_mul_ps(rdz, e2y));
			__m128 pvy = _mm_sub_ps(_mm_mul_ps(rdz, e2x), _mm_mul_ps(rdx, e2z));
			__m128 pvz = _mm_sub_ps(_mm_mul_ps(rdx, e2y), _mm_mul_ps(rdy, e2x));

			// const float det = dot(edge1, pvec);
			__m128 det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, pvx), _mm_mul_ps(e1y, pvy)), _mm_mul_ps(e1z, pvz));

			// if(det > -HIT_EPSILON && det < HIT_EPSILON)
			__m128 ndet = _mm_cmplt_ps(det, _mm_set1_ps(-HIT_EPSILON));
			__m128 pdet = _mm_cmpgt_ps(det, _mm_set1_ps(HIT_EPSILON));
			__m128 mask = _mm_or_ps(ndet, pdet);
			if(_mm_movemask_ps(mask) == 0x0) continue;

			// const float inv_det = 1.0f / det;
			__m128 idet = _mm_rcp_ps(det);

			// const vec3 tvec = ray.o - tri.v0;
			__m128 tvx = _mm_sub_ps(rox, v0x);
			__m128 tvy = _mm_sub_ps(roy, v0y);
			__m128 tvz = _mm_sub_ps(roz, v0z);

			// const float u = dot(tvec, pvec) * inv_det;
			__m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(tvx, pvx), _mm_mul_ps(tvy, pvy)), _mm_mul_ps(tvz, pvz)), idet);

			// if(u < 0.0f || u > 1.0f)
			__m128 ug0 = _mm_cmpge_ps(u, _mm_setzero_ps());
			__m128 ul1 = _mm_cmple_ps(u, _mm_set1_ps(1.0f));
			mask = _mm_and_ps(mask, _mm_and_ps(ug0, ul1));
			if(_mm_movemask_ps(mask) == 0x0) continue;

			// const vec3 qvec = cross(tvec, edge1);
			__m128 qvx = _mm_sub_ps(_mm_mul_ps(tvy, e1z), _mm_mul_ps(tvz, e1y));
			__m128 qvy = _mm_sub_ps(_mm_mul_ps(tvz, e1x), _mm_mul_ps(tvx, e1z));
			__m128 qvz = _mm_sub_ps(_mm_mul_ps(tvx, e1y), _mm_mul_ps(tvy, e1x));

			// const float v = dot(ray.d, qvec) * inv_det;
			__m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rdx, qvx), _mm_mul_ps(rdy, qvy)), _mm_mul_ps(rdz, qvz)), idet);

			// if(v < 0.0f || u + v > 1.0f)
			__m128 vg0 = _mm_cmpge_ps(v, _mm_setzero_ps());
			__m128 vl1 = _mm_cmple_ps(_mm_add_ps(u, v), _mm_set1_ps(1.0f));
			mask = _mm_and_ps(mask, _mm_and_ps(vg0, vl1));
			if(_mm_movemask_ps(mask) == 0x0) continue;

			// const float f = dot(edge2, qvec) * inv_det;
			__m128 f = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qvx), _mm_mul_ps(e2y, qvy)), _mm_mul_ps(e2z, qvz)), idet);

			// if(f >= ray.tmax || f < -HIT_EPSILON)
			__m128 flt = _mm_cmplt_ps(f, tmax);
			__m128 fge = _mm_cmpge_ps(f, _mm_set1_ps(-HIT_EPSILON));
			mask = _mm_and_ps(mask, _mm_and_ps(flt, fge));
			int mmask = _mm_movemask_ps(mask);
			if(mmask == 0x0) continue;

			// ray.tmax = f;
			// g_rays.hits[r] = t;
			if(mmask & 0x1)
			{
				g_rays.rays[r+0].tmax = _mm_get_ps(f, 0);
				const int rid = g_rays.rays[r+0].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x2)
			{
				g_rays.rays[r+1].tmax = _mm_get_ps(f, 1);
				const int rid = g_rays.rays[r+1].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x4)
			{
				g_rays.rays[r+2].tmax = _mm_get_ps(f, 2);
				const int rid = g_rays.rays[r+2].id;
				g_rays.hits[rid] = tid;
			}
			if(mmask & 0x8)
			{
				g_rays.rays[r+3].tmax = _mm_get_ps(f, 3);
				const int rid = g_rays.rays[r+3].id;
				g_rays.hits[rid] = tid;
			}
		}
	}
}

inline void partitionTrisMiddle(const Box& boxCenters, int tbegin, int tend, Box& leftBoxTris, Box& leftBoxCenters, Box& rightBoxTris, Box& rightBoxCenters, int& tsplit, int& axis)
{
	const simd4 d = simd4(_mm_sub_ps(boxCenters.max.m, boxCenters.min.m));

	axis = (d.v.x > d.v.y && d.v.x > d.v.z)? 0 : (d.v.y > d.v.z)? 1 : 2;
	float pos = (boxCenters.min.v[axis] + boxCenters.max.v[axis]) * 0.5f;

	tsplit = tbegin;

	for(int t = tbegin; t < tend; ++t)
	{
		const Box& tbox = g_tris.boxes[g_tris.ids[t]];
		const simd4 center = tbox.computeCenter();
		if(center.v[axis] <= pos)
		{
			leftBoxTris.expand(tbox);
			leftBoxCenters.expand(center);
			std::swap(g_tris.ids[t], g_tris.ids[tsplit++]);
		}
		else
		{
			rightBoxTris.expand(tbox);
			rightBoxCenters.expand(center);
		}
	}
}

// todo: reduce number of arguments in all functions
inline void partitionTrisSAH(const Box& boxCenters, int tbegin, int tend, Box& leftBoxTris, Box& leftBoxCenters, Box& rightBoxTris, Box& rightBoxCenters, int& tsplit, int& axis)
{
	// todo: SAH with 32 bins
	partitionTrisMiddle(boxCenters, tbegin, tend, leftBoxTris, leftBoxCenters, rightBoxTris, rightBoxCenters, tsplit, axis);
}

inline int filterRays(const Box& boxTris, int rend)
{
	int rendNew = 0;
	for(int i = 0; i < rend; ++i)
	{
		Ray& ray = g_rays.rays[i];
		const vec3 invDir = reciprocal(ray.d);

		const float tx1 = (boxTris.min.v.x - ray.o.x) * invDir.x;
		const float tx2 = (boxTris.max.v.x - ray.o.x) * invDir.x;

		float tmin = std::min(tx1, tx2);
		float tmax = std::max(tx1, tx2);

		const float ty1 = (boxTris.min.v.y - ray.o.y) * invDir.y;
		const float ty2 = (boxTris.max.v.y - ray.o.y) * invDir.y;

		tmin = std::max(tmin, std::min(ty1, ty2));
		tmax = std::min(tmax, std::max(ty1, ty2));

		const float tz1 = (boxTris.min.v.z - ray.o.z) * invDir.z;
		const float tz2 = (boxTris.max.v.z - ray.o.z) * invDir.z;

		tmin = std::max(tmin, std::min(tz1, tz2));
		tmax = std::min(tmax, std::max(tz1, tz2));

		tmax = std::min(ray.tmax, tmax);

		if(tmin > tmax) continue;

		std::swap(ray, g_rays.rays[rendNew++]);
	}
	return rendNew;
}

inline void swapRaysSSE(Ray& a, Ray& b)
{
	__m128 tempA = _mm_load_ps(&a.o.x);
	__m128 tempB = _mm_load_ps(&b.o.x);
	_mm_store_ps(&a.o.x, tempB);
	_mm_store_ps(&b.o.x, tempA);

	tempA = _mm_load_ps(&a.d.x);
	tempB = _mm_load_ps(&b.d.x);
	_mm_store_ps(&a.d.x, tempB);
	_mm_store_ps(&b.d.x, tempA);
}

inline int filterRaysSSE(const Box& boxTris, int rend)
{
	int rendNew = 0;
	const int rend4 = nextMultipleOf4(rend);

	__m128 bminx = _mm_shuffle_ps(boxTris.min.m, boxTris.min.m, _MM_SHUFFLE(0,0,0,0));
	__m128 bminy = _mm_shuffle_ps(boxTris.min.m, boxTris.min.m, _MM_SHUFFLE(1,1,1,1));
	__m128 bminz = _mm_shuffle_ps(boxTris.min.m, boxTris.min.m, _MM_SHUFFLE(2,2,2,2));

	__m128 bmaxx = _mm_shuffle_ps(boxTris.max.m, boxTris.max.m, _MM_SHUFFLE(0,0,0,0));
	__m128 bmaxy = _mm_shuffle_ps(boxTris.max.m, boxTris.max.m, _MM_SHUFFLE(1,1,1,1));
	__m128 bmaxz = _mm_shuffle_ps(boxTris.max.m, boxTris.max.m, _MM_SHUFFLE(2,2,2,2));

	for(int i = 0; i < rend4; i+=4)
	{
		__m128 rox = _mm_load_ps(&g_rays.rays[i+0].o.x);
		__m128 rdx = _mm_load_ps(&g_rays.rays[i+0].d.x);
		__m128 roy = _mm_load_ps(&g_rays.rays[i+1].o.x);
		__m128 rdy = _mm_load_ps(&g_rays.rays[i+1].d.x);
		__m128 roz = _mm_load_ps(&g_rays.rays[i+2].o.x);
		__m128 rdz = _mm_load_ps(&g_rays.rays[i+2].d.x);
		__m128 otmax = _mm_load_ps(&g_rays.rays[i+3].o.x);
		__m128 id = _mm_load_ps(&g_rays.rays[i+3].d.x);

		_MM_TRANSPOSE4_PS(rox, roy, roz, otmax);
		_MM_TRANSPOSE4_PS(rdx, rdy, rdz, id);

		_mm_prefetch((const char*)&g_rays.rays[i+16], _MM_HINT_T0);
		_mm_prefetch((const char*)&g_rays.rays[i+18], _MM_HINT_T0);

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

		tmax = _mm_min_ps(tmax, otmax);

		__m128 cmp = _mm_cmple_ps(tmin, tmax);

		int mask = _mm_movemask_ps(cmp);

		if(mask & 0x1)
		{
			swapRaysSSE(g_rays.rays[i+0], g_rays.rays[rendNew++]);
		}
		if(mask & 0x2)
		{
			swapRaysSSE(g_rays.rays[i+1], g_rays.rays[rendNew++]);
		}
		if(mask & 0x4)
		{
			swapRaysSSE(g_rays.rays[i+2], g_rays.rays[rendNew++]);
		}
		if(mask & 0x8)
		{
			swapRaysSSE(g_rays.rays[i+3], g_rays.rays[rendNew++]);
		}
	}

	return rendNew;
}

// todo: change to iterative algorithm
inline void trace(const Box& boxTris, const Box& boxCenters, int tbegin, int tend, int rend)
{
	g_timer.restart();
//	rend = filterRays(boxTris, rend);
	rend = filterRaysSSE(boxTris, rend);
	g_filterTime += g_timer.msec();

	if((tend-tbegin) < TRI_LIMIT || rend < RAY_LIMIT)
	{
		g_timer.restart();
//		intersect(tbegin, tend, rend);
//		intersectAfra(tbegin, tend, rend);
//		intersectSSE(tbegin, tend, rend);
		intersectSSEAfra(tbegin, tend, rend);
		g_intersectTime += g_timer.msec();
		return;
	}

	Box leftBoxTris;
	Box leftBoxCenters;
	Box rightBoxTris;
	Box rightBoxCenters;
	int tsplit;
	int axis;

	g_timer.restart();
	if(rend > (tend-tbegin) * 2.0f)
	{
		partitionTrisSAH(boxCenters, tbegin, tend, leftBoxTris, leftBoxCenters, rightBoxTris, rightBoxCenters, tsplit, axis);
	}
	else
	{
		partitionTrisMiddle(boxCenters, tbegin, tend, leftBoxTris, leftBoxCenters, rightBoxTris, rightBoxCenters, tsplit, axis);
	}
	g_partitionTime += g_timer.msec();

	if(g_rays.rays[0].d[axis] > 0.0f)
	{
		trace(leftBoxTris, leftBoxCenters, tbegin, tsplit, rend);
		trace(rightBoxTris, rightBoxCenters, tsplit, tend, rend);
	}
	else
	{
		trace(rightBoxTris, rightBoxCenters, tsplit, tend, rend);
		trace(leftBoxTris, leftBoxCenters, tbegin, tsplit, rend);
	}
}

void generatePrimarySSE()
{
	const __m128i hit = _mm_set1_epi32(-1);
	for(int i = 0; i < g_rays.count; i+=4)
	{
		_mm_stream_si128((__m128i*)(g_rays.hits+i), hit); // marginally faster than _mm_store_si128
	}

	int id = 0;

	__m128 dux = _mm_set1_ps(g_camera.du.x);
	__m128 dvx = _mm_set1_ps(g_camera.dv.x);
	__m128 dx = _mm_set1_ps(g_camera.lowerLeftDir.x);

	dx = _mm_add_ps(dx, _mm_mul_ps(dux, _mm_setr_ps(0.0f, 1.0f, 2.0f, 3.0f)));
	dux = _mm_mul_ps(dux, _mm_set1_ps(4.0f));

	Ray ray;
	ray.o = g_camera.position;
	ray.d = g_camera.lowerLeftDir;
	ray.tmax = FLT_MAX;

	for(int v = 0; v < g_camera.nv; ++v)
	{
		for(int u = 0; u < g_camera.nu; u+=4)
		{
			ray.id = id;
			ray.d.x = _mm_cvtss_f32(dx);
			g_rays.rays[id++] = ray;

			ray.id = id;
			ray.d.x = _mm_get_ps(dx, 1);
			g_rays.rays[id++] = ray;

			ray.id = id;
			ray.d.x = _mm_get_ps(dx, 2);
			g_rays.rays[id++] = ray;

			ray.id = id;
			ray.d.x = _mm_get_ps(dx, 3);
			g_rays.rays[id++] = ray;

			dx = _mm_add_ps(dx, dux);
		}
		dx = _mm_add_ps(dx, dvx);
		ray.d.y += g_camera.dv.y;
	}
}

void generatePrimary()
{
	int id = 0;
	vec3 dir = g_camera.lowerLeftDir;
	Ray ray;
	ray.o = g_camera.position;
	ray.tmax = FLT_MAX;

	for(int v = 0; v < g_camera.nv; ++v)
	{
		for(int u = 0; u < g_camera.nu; ++u)
		{
			ray.id = id;
			ray.d = dir;
			g_rays.rays[id++] = ray;
			dir += g_camera.du;
		}
		dir += g_camera.dv;
	}
	std::fill_n(g_rays.hits, g_rays.count, -1);
}

void tracePrimary()
{
	trace(g_tris.boxTris, g_tris.boxCenters, 0, g_tris.count, g_rays.count);
}

void shadePixels(unsigned char* pixels)
{
	for(int i = 0; i < g_rays.count; ++i)
	{
		const Ray& ray = g_rays.rays[i];
		int h = g_rays.hits[ray.id] >= 0? 1 : 0;
		pixels[ray.id*3+0] = h*255;
		pixels[ray.id*3+1] = h*255;
		pixels[ray.id*3+2] = h*255;
	}
}

void reshapeCB(int w, int h)
{
	if((w%4) != 0)
	{
		w = nextMultipleOf4(w);
		rtvReshapeWindow(w, h);
		return;
	}

	g_canvas.w = w;
	g_canvas.h = h;

	const int npixels = g_canvas.w * g_canvas.h;
	g_rays.count = npixels;

	_mm_free(g_rays.rays);
	g_rays.rays = (Ray*)_mm_malloc(g_rays.count*sizeof(Ray), 16);

	_mm_free(g_rays.hits);
	g_rays.hits = (int*)_mm_malloc(g_rays.count*sizeof(int), 16);
}

void cameraCB(float* peye, float* pcenter, float* pup)
{
	const vec3 eye = vec3(peye);
	const vec3 center = vec3(pcenter);
	const vec3 up = vec3(pup);

	// store position
	g_camera.position = eye;

	// pre-computations
	float invHeight = 1.0f / g_canvas.h;
	float invWidth  = 1.0f / g_canvas.w;

	// compute camera basis
	vec3 axisW = normalize(eye - center);
	vec3 axisV = normalize(up);
	vec3 axisU = cross(axisV, axisW);

	// compute half scale factors for each basis vector
	float sw = g_canvas.w * 0.01f; // try to keep directions around zero in floating-point value
	float sv = sw * tanf(0.523598775f); // half 60o in radians
	float su = sv * g_canvas.w * invHeight;

	// scale each vector
	axisW *= sw;
	axisV *= sv;
	axisU *= su;

	// store final direction
	g_camera.lowerLeftDir = - axisU - axisV - axisW;

	// compute full scales
	axisV *= 2.0f;
	axisU *= 2.0f;

	// interpolation deltas
	g_camera.dv = axisV * invHeight - axisU; // also goes back to start of u-axis
	g_camera.du = axisU * invWidth;

	// number of pixels in U and V directions
	g_camera.nu = g_canvas.w;
	g_camera.nv = g_canvas.h;
}

void renderCB(unsigned char* pixels)
{
	timer ttotal;
	timer t;
	double totalTime = 0.0;
	double generateTime = 0.0;
	double traceTime = 0.0;
	double shadeTime = 0.0;
	g_filterTime = 0.0;
	g_intersectTime = 0.0;
	g_partitionTime = 0.0;

	ttotal.restart();

	t.restart();
	generatePrimarySSE();
	generateTime = t.msec();

	t.restart();
	tracePrimary();
	traceTime = t.msec();

	t.restart();
	shadePixels(pixels);
	shadeTime = t.msec();

	totalTime = ttotal.msec();

	printf("----------------------------------------\n");
	printf("total:            %.1f\n", totalTime);
	printf("  generate:       %.1f\n", generateTime);
	printf("  trace:          %.1f\n", traceTime);
	printf("    filter:       %.1f\n", g_filterTime);
	printf("    intersect:    %.1f\n", g_intersectTime);
	printf("    partition:    %.1f\n", g_partitionTime);
	printf("  shade:          %.1f\n", shadeTime);
}

vec3 tovec3(aiVector3D v)
{
	return vec3(v.x, v.y, v.z);
}

void loadScene(const char* path)
{
	// Create an instance of the Importer class
	Assimp::Importer importer;
	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll
	// propably to request more postprocessing than we do in this example.
	const aiScene* scene = importer.ReadFile(path,
		aiProcess_Triangulate            |
		aiProcess_JoinIdenticalVertices  |
		aiProcess_GenSmoothNormals |
		aiProcess_SortByPType);

	if(!scene->HasMeshes())
	{
		printf("no meshes!\n");
		exit(1);
	}

	std::vector<TriV> triangleVerts;
	std::vector<TriN> triangleNorms;

	for(unsigned int m = 0; m < scene->mNumMeshes; ++m)
	{
		aiMesh* mesh = scene->mMeshes[m];
		if(!mesh->HasFaces() || !mesh->HasNormals())
		{
			continue;
		}
		for(unsigned int f = 0; f < mesh->mNumFaces; ++f)
		{
			const aiFace& face = mesh->mFaces[f];
			if(face.mNumIndices != 3)
			{
				continue;
			}
			TriV tri;
			tri.v0 = tovec3(mesh->mVertices[face.mIndices[0]]);
			tri.v1 = tovec3(mesh->mVertices[face.mIndices[1]]);
			tri.v2 = tovec3(mesh->mVertices[face.mIndices[2]]);
			triangleVerts.push_back(tri);

			TriN triN;
			triN.n0 = tovec3(mesh->mNormals[face.mIndices[0]]);
			triN.n1 = tovec3(mesh->mNormals[face.mIndices[1]]);
			triN.n2 = tovec3(mesh->mNormals[face.mIndices[2]]);
			triangleNorms.push_back(triN);
		}
	}

	printf("v: %d\n", triangleVerts.size());
	printf("n: %d\n", triangleNorms.size());

//triangleVerts.clear();
//triangleNorms.clear();
//
//TriV t;
//t.v0 = vec3(-5,-5,0);
//t.v1 = vec3(5,-5,0);
//t.v2 = vec3(0,5,0);
//triangleVerts.push_back(t);
//
//TriN n;
//n.n0 = vec3(0,0,1);
//n.n1 = vec3(0,0,1);
//n.n2 = vec3(0,0,1);
//triangleNorms.push_back(n);

	g_tris.count = triangleVerts.size();
	g_tris.ids = (int*)_mm_malloc(g_tris.count*sizeof(int), 16);
	g_tris.verts = (TriV*)_mm_malloc(g_tris.count*sizeof(TriV), 16);
	g_tris.norms = (TriN*)_mm_malloc(g_tris.count*sizeof(TriN), 16);
	g_tris.boxes = (Box*)_mm_malloc(g_tris.count*sizeof(Box), 16);

	for(int i = 0; i < g_tris.count; ++i)
	{
		const TriV& tv = triangleVerts[i];

		Box box;
		box.expand(tv);

		g_tris.boxTris.expand(tv);
		g_tris.boxCenters.expand((tv.v0 + tv.v1 + tv.v2)*0.3333333333333333333f);

		g_tris.ids[i] = i;
		g_tris.verts[i] = tv;
		g_tris.norms[i] = triangleNorms[i];
		g_tris.boxes[i] = box;
	}

	vec3 center = (g_tris.boxTris.min.v + g_tris.boxTris.max.v) * 0.5f;
	vec3 eye = center + vec3(0,0,0.2f);
	vec3 up(0,1,0);
	rtvSetCamera(&eye.x, &center.x, &up.x);
}

int main()
{
	rtvInit(1024, 1024);
	rtvSetReshapeCallback(reshapeCB);
	rtvSetCameraCallback(cameraCB);
	rtvSetPixelRenderCallback(renderCB);

	loadScene("/home/potato/Downloads/bunny.ply");

	rtvExec();

	return 0;
}
