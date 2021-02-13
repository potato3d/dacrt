#include "rtv.h"
#include "rply.h"
#include "timer.h"
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

//#define HIT_CPU
#define HIT_GPU

#define HIT_EPSILON 1e-6f
#define BLOCKSIZE 256

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct TriV // 36 bytes
{
	vec3 v[3];
};

struct TriN // 36 bytes
{
	vec3 n[3];
};

struct Ray // 28 bytes
{
	vec3 o;
	vec3 d;
	int id;
};

struct Box // 24 bytes
{
	vec3 min, max;
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

struct TriSet
{
	Box boxTris;
	Box boxCenters;
	int count;
	int* ids;
	Box* boxes;
	vec3* centers;
	TriV* verts;
	TriN* norms;
};

struct RaySet
{
	int count;
	int* ids;
	Ray* rays;
	float* tmaxs;
	int* hits;
};

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static timer s_t;
static double s_intersectTime = 0.0;
static double s_triSendTime = 0.0;
static double s_raySendTime = 0.0;
static double s_kernelTime = 0.0;
static double s_splitTime = 0.0;
static double s_partitionTriTime = 0.0;
static double s_filterRayTime = 0.0;
#define TIMED_CALL(f, t) s_t.restart(); f; t += s_t.msec();
static int s_numIntersectCalls = 0;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static TriSet g_tris;
static RaySet g_rays;

static TriV* d_tris = 0;
static Ray* d_rays = 0;
static float* d_tmaxs = 0;
static int* d_hits = 0;

static TriV* h_tris = 0;
static Ray* h_rays = 0;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void kernel(TriV* tris, int tbegin, int numTiles, int lastTileSize, Ray* rays, int numRays, float* tmaxs, int* hits)
{
	extern __shared__ TriV sh_tris[];

	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	Ray ray;
	float tmax;
	if(idx < numRays)
	{
		ray = rays[idx];
		tmax = tmaxs[ray.id];
	}

	int hit = -1;

	for(int tile = 0; tile < numTiles; ++tile)
	{
		const bool isLastTile = (tile == numTiles-1);

		if(!isLastTile || threadIdx.x < lastTileSize)
		{
			sh_tris[threadIdx.x] = tris[tile*blockDim.x+threadIdx.x];
		}

		__syncthreads();

		if(idx < numRays)
		{
			const int limit = isLastTile ? lastTileSize : blockDim.x;
			for(int t = 0; t < limit; ++t)
			{
				TriV triv = sh_tris[t];

				// find vectors for two edges sharing vert0
				const vec3 edge1 = triv.v[1] - triv.v[0];
				const vec3 edge2 = triv.v[2] - triv.v[0];

				// begin calculating determinant - also used to calculate U parameter
				const vec3 pvec = cross(ray.d, edge2);

				// if determinant is near zero, ray lies in plane of triangle
				const float det = dot(edge1, pvec);

				if(det > -HIT_EPSILON && det < HIT_EPSILON)
					continue;

				const float inv_det = 1.0f / det;

				// calculate distance from vert0 to ray origin
				const vec3 tvec = ray.o - triv.v[0];

				// calculate U parameter and test bounds
				const float u = dot(tvec, pvec) * inv_det;
				if(u < 0.0f || u > 1.0f)
					continue;

				// prepare to test V parameter
				const vec3 qvec = cross(tvec, edge1);

				// calculate V parameter and test bounds
				const float v = dot(ray.d, qvec) * inv_det;
				if(v < 0.0f || u + v > 1.0f)
					continue;

				// calculate t, ray hits triangle
				const float f = dot(edge2, qvec) * inv_det;

				if(f >= tmax || f < -HIT_EPSILON)
					continue;

				// store valid hit
				tmax = f;
				hit = tile*blockDim.x + t;
			}
		}

		__syncthreads();
	}

	if(idx < numRays)
	{
		tmaxs[ray.id] = tmax;
		if(hit >= 0)
			hits[ray.id] = hit + tbegin;
	}
}

void intersectGPU(int tbegin, int tend, int rend)
{
	static cudaStream_t stream;
	static cudaEvent_t copyEvent;
	static cudaEvent_t kernelEvent;
	static bool first = true;
	if(first)
	{
		cudaStreamCreate(&stream);
		cudaEventCreate(&copyEvent);
		cudaEventCreate(&kernelEvent);
		first = false;
	}

	const int tcount = tend - tbegin;
	const int numBlocks = ceilf((float)rend/(float)BLOCKSIZE);
	const int numTiles = ceilf((float)tcount/(float)BLOCKSIZE);
	const int lastTileSize = tcount < BLOCKSIZE? tcount : tcount % BLOCKSIZE;

	cudaEventSynchronize(copyEvent);

	std::copy(g_tris.verts + tbegin, g_tris.verts + tend, h_tris);
	std::copy(g_rays.rays, g_rays.rays + rend, h_rays);

	cudaEventSynchronize(kernelEvent);

	cudaMemcpy(g_rays.tmaxs, d_tmaxs, g_rays.count*sizeof(float), cudaMemcpyDefault);

	TIMED_CALL(cudaMemcpyAsync(d_tris, h_tris, tcount*sizeof(TriV), cudaMemcpyDefault, stream), s_triSendTime);
	TIMED_CALL(cudaMemcpyAsync(d_rays, h_rays, rend*sizeof(Ray), cudaMemcpyDefault, stream), s_raySendTime);

	cudaEventRecord(copyEvent, stream);

	s_t.restart();
	kernel<<<numBlocks, BLOCKSIZE, BLOCKSIZE*sizeof(TriV), stream>>>(d_tris, tbegin, numTiles, lastTileSize, d_rays, rend, d_tmaxs, d_hits);
	s_kernelTime += s_t.msec();

	cudaEventRecord(kernelEvent, stream);

	++s_numIntersectCalls;
}

void intersectCPU(int tbegin, int tend, int rend)
{
	for(int t = tbegin; t < tend; ++t)
	{
		const TriV& triv = g_tris.verts[t];

		for(int r = 0; r < rend; ++r)
		{
			Ray& ray = g_rays.rays[r];

			// find vectors for two edges sharing vert0
			const vec3 edge1 = triv.v[1] - triv.v[0];
			const vec3 edge2 = triv.v[2] - triv.v[0];

			// begin calculating determinant - also used to calculate U parameter
			const vec3 pvec = cross(ray.d, edge2);

			// if determinant is near zero, ray lies in plane of triangle
			const float det = dot(edge1, pvec);

			if(det > -HIT_EPSILON && det < HIT_EPSILON)
				continue;

			const float inv_det = 1.0f / det;

			// calculate distance from vert0 to ray origin
			const vec3 tvec = ray.o - triv.v[0];

			// calculate U parameter and test bounds
			const float u = dot(tvec, pvec) * inv_det;
			if(u < 0.0f || u > 1.0f)
				continue;

			// prepare to test V parameter
			const vec3 qvec = cross(tvec, edge1);

			// calculate V parameter and test bounds
			const float v = dot(ray.d, qvec) * inv_det;
			if(v < 0.0f || u + v > 1.0f)
				continue;

			// calculate t, ray hits triangle
			const float f = dot(edge2, qvec) * inv_det;

			if(f >= g_rays.tmaxs[ray.id] || f < -HIT_EPSILON)
				continue;

			// store valid hit
			g_rays.tmaxs[ray.id] = f;
			g_rays.hits[ray.id] = t;
		}
	}
	++s_numIntersectCalls;
}

// ALTERNATIVES:
// https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.2010&rep=rep1&type=pdf
// http://www.flipcode.com/archives/SSE_RayBox_Intersection_Test.shtml
// SIMD para varios raios ao mesmo tempo (precisa de SoA)
void filterRays(const Box& box, int& rend)
{
	int rendNew = 0;
	for(int r = 0; r < rend; ++r)
	{
		const Ray& ray = g_rays.rays[r];
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
		tmax = std::min(g_rays.tmaxs[ray.id], std::min(tmax, std::max(tz1, tz2)));

		if(tmin > tmax) continue;

		std::swap(g_rays.rays[r], g_rays.rays[rendNew]);
		++rendNew;
	}
	rend = rendNew;
}

void split(const Box& box, int& axis, float& pos)
{
	const float dx = box.max.x - box.min.x;
	const float dy = box.max.y - box.min.y;
	const float dz = box.max.z - box.min.z;

	axis = (dx > dy && dx > dz)? 0 : (dy > dz)? 1 : 2;
	pos = (box.min[axis] + box.max[axis]) * 0.5f;
}

void partitionTris(int tbegin, int tend, int axis, float pos, int& tsplit, Box& boxLeftTris, Box& boxLeftCenters, Box& boxRightTris, Box& boxRightCenters)
{
	tsplit = tbegin;
	for(int t = tbegin; t < tend; ++t)
	{
		const vec3& center = g_tris.centers[t];
		if(center[axis] < pos)
		{
			boxLeftTris.expand(g_tris.verts[t]);
			boxLeftCenters.expand(center);

			std::swap(g_tris.verts[t], g_tris.verts[tsplit]);
			std::swap(g_tris.centers[t], g_tris.centers[tsplit]);
			std::swap(g_tris.ids[t], g_tris.ids[tsplit]);
			++tsplit;
		}
		else
		{
			boxRightTris.expand(g_tris.verts[t]);
			boxRightCenters.expand(center);
		}
	}
}

void trace(const Box& boxTris, const Box& boxCenters, int tbegin, int tend, int rend)
{
	TIMED_CALL(filterRays(boxTris, rend), s_filterRayTime);

#ifdef HIT_CPU
	if((tend-tbegin) < 8 || rend < 8) // todo: define better
	{
		TIMED_CALL(intersectCPU(tbegin, tend, rend), s_intersectTime);
		return;
	}
#elif defined HIT_GPU
	if((tend-tbegin) < 8192 && rend < 32768)  // todo: define better
	{
		static timer __t;
		__t.restart();
		intersectGPU(tbegin, tend, rend);
		s_intersectTime += __t.msec();
		return;
	}
#endif

	int axis;
	float pos;
	TIMED_CALL(split(boxCenters, axis, pos), s_splitTime);

	Box boxLeftTris;
	Box boxLeftCenters;
	Box boxRightTris;
	Box boxRightCenters;
	int tsplit;
	TIMED_CALL(partitionTris(tbegin, tend, axis, pos, tsplit, boxLeftTris, boxLeftCenters, boxRightTris, boxRightCenters), s_partitionTriTime);

	if(g_rays.rays[0].d[axis] >= 0)
	{
		trace(boxLeftTris, boxLeftCenters, tbegin, tsplit, rend);
		trace(boxRightTris, boxRightCenters, tsplit, tend, rend);
	}
	else
	{
		trace(boxRightTris, boxRightCenters, tsplit, tend, rend);
		trace(boxLeftTris, boxLeftCenters, tbegin, tsplit, rend);
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static Canvas g_canvas;
static Camera g_camera;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void generatePrimary()
{
	vec3 dir = g_camera.lowerLeftDir;
	int id = 0;

	for(int v = 0; v < g_camera.nv; ++v)
	{
		for(int u = 0; u < g_camera.nu; ++u)
		{
			Ray& r = g_rays.rays[id];
			r.o = g_camera.position;
			r.d = dir;
			r.id = id;
			dir += g_camera.du;
			++id;
		}
		dir += g_camera.dv;
	}

	for(int i = 0; i < g_rays.count; ++i)
	{
		g_rays.tmaxs[i] = FLT_MAX;
		g_rays.hits[i]  = -1;
	}

	cudaMemcpy(d_tmaxs, g_rays.tmaxs, g_rays.count*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(d_hits, g_rays.hits, g_rays.count*sizeof(int), cudaMemcpyDefault);
}

void shadePixels(unsigned char* pixels)
{
	cudaMemcpy(g_rays.tmaxs, d_tmaxs, g_rays.count*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(g_rays.hits, d_hits, g_rays.count*sizeof(int), cudaMemcpyDefault);

	for(int i = 0; i < g_rays.count; ++i)
	{
		pixels[i*3+0] = 0;
		pixels[i*3+1] = 0;
		pixels[i*3+2] = 0;
	}

	for(int i = 0; i < g_rays.count; ++i)
	{
		const Ray& ray = g_rays.rays[i];
		const float tmax = g_rays.tmaxs[ray.id];
		if(tmax < FLT_MAX)
		{
			// recompute hit position
			const vec3 hitPos = ray.o + ray.d * tmax;

			// recover triangle hit
			const int tid = g_rays.hits[ray.id];

			// recompute barycentric coordinates
			const TriV& tri = g_tris.verts[tid];
			const vec3 e0 = tri.v[1] - tri.v[0];
			const vec3 e1 = tri.v[2] - tri.v[0];
			const vec3 e2 = hitPos - tri.v[0];
			const float d00 = dot(e0, e0);
			const float d01 = dot(e0, e1);
			const float d11 = dot(e1, e1);
			const float d20 = dot(e2, e0);
			const float d21 = dot(e2, e1);
			const float invDenom = 1.0f / (d00 * d11 - d01 * d01);
			const float v = (d11 * d20 - d01 * d21) * invDenom;
			const float w = (d00 * d21 - d01 * d20) * invDenom;
			const float u = 1.0f - v - w;

			// lerp normal at hit position using barycentric coordinates
			const TriN& trin = g_tris.norms[g_tris.ids[tid]];
			const vec3 lerpN = normalize(trin.n[0]*u + trin.n[1]*v + trin.n[2]*w);

			// compute final color
			const unsigned char c = 255 * dot(lerpN, -normalize(ray.d));

			pixels[ray.id*3+0] = c;
			pixels[ray.id*3+1] = c;
			pixels[ray.id*3+2] = c;
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void reshape(int w, int h)
{
	g_canvas.w = w;
	g_canvas.h = h;

	const int npixels = g_canvas.w * g_canvas.h;
	g_rays.count = npixels;

	// CPU --------------------------------------------------------

//	delete[] g_rays.ids;
//	g_rays.ids = new int[g_rays.count];
//	for(int i = 0; i < g_rays.count; ++i) g_rays.ids[i] = i;

	cudaFreeHost(g_rays.rays);
	cudaMallocHost(&g_rays.rays, g_rays.count*sizeof(Ray));

	cudaFreeHost(g_rays.tmaxs);
	cudaMallocHost(&g_rays.tmaxs, g_rays.count*sizeof(float));

	cudaFreeHost(g_rays.hits);
	cudaMallocHost(&g_rays.hits, g_rays.count*sizeof(int));

	// CPU pinned & GPU --------------------------------------------

	cudaFreeHost(h_rays);
	cudaMallocHost(&h_rays, g_rays.count*sizeof(Ray));

	cudaFree(d_rays);
	cudaMalloc(&d_rays, g_rays.count*sizeof(Ray));

	cudaFree(d_tmaxs);
	cudaMalloc(&d_tmaxs, g_rays.count*sizeof(float));

	cudaFree(d_hits);
	cudaMalloc(&d_hits, g_rays.count*sizeof(int));
}

void camera(float* peye, float* pcenter, float* pup)
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
	float sv = sw * std::tan(0.523598775f); // half 60o in radians
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

void render(unsigned char* pixels)
{
	timer t;
	s_intersectTime = 0.0;
	s_triSendTime = 0.0;
	s_raySendTime = 0.0;
	s_kernelTime = 0.0;
	s_splitTime = 0.0;
	s_partitionTriTime = 0.0;
	s_filterRayTime = 0.0;
	s_numIntersectCalls = 0;

	std::cout << "---------------------------------" << std::endl;

	t.restart();
	generatePrimary();
	std::cout << "generate: " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	t.restart();
	trace(g_tris.boxTris, g_tris.boxCenters, 0, g_tris.count, g_rays.count);
	std::cout << "trace:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	std::cout << "  " << "filterRay: " << std::setw(9) << (int)s_filterRayTime << " ms" << std::endl;

	std::cout << "  " << "split:        " << std::setw(9) << (int)s_splitTime << " ms" << std::endl;
	std::cout << "  " << "intersect:    " << std::setw(9) << (int)s_intersectTime << " ms (count: " << s_numIntersectCalls << ", avg: " << s_intersectTime / s_numIntersectCalls << " ms)" << std::endl;

	std::cout << "    " << "triSend:    " << std::setw(9) << s_triSendTime << " ms" << std::endl;
	std::cout << "    " << "raySend:    " << std::setw(9) << s_raySendTime << " ms" << std::endl;
	std::cout << "    " << "kernel:    " << std::setw(9) << s_kernelTime << " ms (avg: " << s_kernelTime / s_numIntersectCalls << " ms)" << std::endl;

	std::cout << "  " << "partitionTri: " << std::setw(9) << (int)s_partitionTriTime << " ms" << std::endl;

	t.restart();
	shadePixels(pixels);
	std::cout << "shade:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------

void loadScene1tri()
{
	TriV triv;
	triv.v[0] = vec3(-1,-1,0);
	triv.v[1] = vec3(1,-1,0);
	triv.v[2] = vec3(0,1,0);
	cudaMallocHost(&g_tris.verts, 1*sizeof(TriV));
	g_tris.verts[0] = triv;

	TriN trin;
	trin.n[0] = normalize(cross(triv.v[1] - triv.v[0], triv.v[2]-triv.v[0]));
	trin.n[1] = trin.n[0];
	trin.n[2] = trin.n[0];
	g_tris.norms = new TriN[1];
	g_tris.norms[0] = trin;

	g_tris.count = 1;
}

float randf(float min, float max)
{
	return min + (float)rand()/(float)RAND_MAX * (max - min);
}

void loadSceneRand()
{
	g_tris.count = 100000;
	cudaMallocHost(&g_tris.verts, g_tris.count*sizeof(TriV));
	g_tris.norms = new TriN[g_tris.count];
	int dst = 0;

	srand(122);

	for(int i = 0; i < g_tris.count; ++i)
	{
		vec3 c(randf(-5.0f, 5.0f), randf(-5.0f, 5.0f), randf(-5.0f, 5.0f));
		TriV triv;
		triv.v[0] = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		triv.v[1] = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		triv.v[2] = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		g_tris.verts[dst] = triv;

		TriN trin;
		trin.n[0] = normalize(cross(triv.v[1] - triv.v[0], triv.v[2]-triv.v[0]));
		trin.n[1] = trin.n[0];
		trin.n[2] = trin.n[0];
		g_tris.norms[dst] = trin;

		++dst;
	}
}

static std::vector<vec3> vertices;
static std::vector<int> elements;

static int vertex_cb(p_ply_argument argument)
{
	long id;
	ply_get_argument_user_data(argument, NULL, &id);
	if(id == 0)
	{
		vertices.resize(vertices.size()+1);
	}
	vertices.back()[id] = ply_get_argument_value(argument) * 50;
	return 1;
}

static int face_cb(p_ply_argument argument)
{
	long length, value_index;
	ply_get_argument_property(argument, NULL, &length, &value_index);
	switch (value_index)
	{
	case 0:
	case 1:
	case 2:
		elements.push_back(ply_get_argument_value(argument));
		break;
	default:
		break;
	}
	return 1;
}

void loadSceneBunny()
{
	p_ply ply = ply_open("/home/environ/Downloads/bunny.ply", NULL, 0, NULL);
	if (!ply) exit(1);
	if (!ply_read_header(ply)) exit(1);
	long nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
	ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 1);
	ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 2);
	long ntriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
	if (!ply_read(ply)) exit(1);
	ply_close(ply);

	g_tris.count = elements.size()/3;
	cudaMallocHost(&g_tris.verts, g_tris.count*sizeof(TriV));
	g_tris.norms = new TriN[g_tris.count];

	std::vector<vec3> normals(vertices.size());

	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		const int e0 = elements[e+0];
		const int e1 = elements[e+1];
		const int e2 = elements[e+2];

		vec3 v0 = vertices[e0];
		vec3 v1 = vertices[e1];
		vec3 v2 = vertices[e2];

		vec3 n = cross(v1-v0, v2-v0);

		normals[e0] += n;
		normals[e1] += n;
		normals[e2] += n;
	}

	int dst = 0;
	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		const int e0 = elements[e+0];
		const int e1 = elements[e+1];
		const int e2 = elements[e+2];

		TriV triv;
		triv.v[0] = vertices[e0];
		triv.v[1] = vertices[e1];
		triv.v[2] = vertices[e2];
		g_tris.verts[dst] = triv;

		TriN trin;
		trin.n[0] = normalize(normals[e0]);
		trin.n[1] = normalize(normals[e1]);
		trin.n[2] = normalize(normals[e2]);
		g_tris.norms[dst] = trin;

		++dst;
	}
}

void endLoadScene()
{
	g_tris.ids = new int[g_tris.count];
	g_tris.boxes = new Box[g_tris.count];
	g_tris.centers = new vec3[g_tris.count];
	for(int i = 0; i < g_tris.count; ++i)
	{
		g_tris.ids[i] = i;
		const TriV& triv = g_tris.verts[i];
		g_tris.boxes[i].expand(triv);
		g_tris.centers[i] = (triv.v[0] + triv.v[1] + triv.v[2]) * 0.333333333333f;
		g_tris.boxTris.expand(triv);
		g_tris.boxCenters.expand(g_tris.centers[i]);
	}

	cudaMallocHost(&h_tris, g_tris.count*sizeof(TriV));
	cudaMalloc(&d_tris, g_tris.count*sizeof(TriV));

	vec3 center = (g_tris.boxTris.min + g_tris.boxTris.max) * 0.5f;
	vec3 eye = center + vec3(0,0,10);
	vec3 up(0,1,0);
	rtvSetCamera(eye.ptr(), center.ptr(), up.ptr());
}

//---------------------------------------------------------------------------------------------------------------------

int main()
{
	rtvInit(1024, 1024);
	rtvSetReshapeCallback(reshape);
	rtvSetCameraCallback(camera);
	rtvSetRenderCallback(render);

//	loadScene1tri();
//	loadSceneRand();
	loadSceneBunny();

	endLoadScene();

	rtvExec();

	return 0;
}
