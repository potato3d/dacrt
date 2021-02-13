#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cfloat>

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cfloat>

__host__ __device__ inline float minf(float a, float b)
{
	return (a < b)? a : b;
}

__host__ __device__ inline float maxf(float a, float b)
{
	return (a > b)? a : b;
}

struct vec3
{
	float x,y,z;
	__host__ __device__ vec3() {}
	__host__ __device__ explicit vec3(float a) : x(a), y(a), z(a) {}
	__host__ __device__ explicit vec3(const float* const p) : x(p[0]), y(p[1]), z(p[2]) {}
	__host__ __device__ vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	__host__ __device__ float* ptr() { return &x; }
	__host__ __device__ const float* ptr() const { return &x; }
	__host__ __device__ float& operator[](int i) { return (&x)[i]; }
	__host__ __device__ float operator[](int i) const { return (&x)[i]; }
	__host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__ vec3 operator+(const vec3& o) const { return vec3(x+o.x, y+o.y, z+o.z); }
	__host__ __device__ vec3 operator-(const vec3& o) const { return vec3(x-o.x, y-o.y, z-o.z); }
	__host__ __device__ vec3 operator*(float a) const { return vec3(x*a, y*a, z*a); }
	__host__ __device__ void operator+=(const vec3& o) { x+=o.x; y+=o.y; z+=o.z; }
	__host__ __device__ void operator*=(float a) { x*=a; y*=a; z*=a; }
	__host__ __device__ void setmin(const vec3& o) { x = minf(x, o.x); y = minf(y, o.y); z = minf(z, o.z); }
	__host__ __device__ void setmax(const vec3& o) { x = maxf(x, o.x); y = maxf(y, o.y); z = maxf(z, o.z); }
};
__host__ __device__ inline float dot(const vec3& a, const vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) { return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
__host__ __device__ inline float len(const vec3& v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
__host__ __device__ inline vec3 normalize(const vec3& v) { const float invlen = 1.0f / len(v); return vec3(v.x*invlen, v.y*invlen, v.z*invlen); }
__host__ __device__ inline vec3 reciprocal(const vec3& v) { return vec3(1.0f/v.x, 1.0f/v.y, 1.0f/v.z); }


struct Ray // 32 bytes
{
	vec3 o;
	int id;
	vec3 d;
	float tmax;
};

struct TriV // 36 bytes
{
	vec3 v[3];int pad[3];
};

struct Box // 32 bytes
{
	vec3 min; int pad;
	vec3 max; int pad2;
	Box() : min(FLT_MAX), max(-FLT_MAX) {}
	void expand(const vec3& v)
	{
		min.setmin(v);
		max.setmax(v);
	}
	void expand(const TriV& tri)
	{
		expand(tri.v[0]);
		expand(tri.v[1]);
		expand(tri.v[2]);
	}
};

void filterRays(const Box& box, Ray* rays, int& rend);
void filterRaysSSE(const Box& box, Ray* rays, int& rend);
void filterRaysAVX(const Box& box, Ray* rays, int& rend);

