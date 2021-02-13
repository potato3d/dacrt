#include "rtv.h"
#include "timer.h"
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include "filter.h"
#include "nvToolsExt.h"
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//#define HIT_CPU
#define HIT_GPU

#define TEC

#define HIT_EPSILON 1e-6f
#define BLOCKSIZE 256
#define GPU_QUEUE_SIZE 10

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct TriN // 36 bytes
{
	vec3 n[3];
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
static double s_configTime = 0.0;
static double s_syncTime = 0.0;
static double s_triCopyTime = 0.0;
static double s_rayCopyTime = 0.0;
static double s_triSendTime = 0.0;
static double s_raySendTime = 0.0;
static double s_kernelTime = 0.0;
static double s_tmaxReadTime = 0.0;
static double s_hitReadTime = 0.0;
static double s_addCallbackTime = 0.0;
static double s_recordTime = 0.0;
static double s_finishTime = 0.0;
static double s_updateResultsTime = 0.0;
static double s_splitTime = 0.0;
static double s_partitionTriTime = 0.0;
static double s_filterRayTime = 0.0;
#define TIMED_CALL(f, t) s_t.restart(); f; t += s_t.msec();
static int s_numIntersectCalls = 0;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static TriSet g_tris;
static RaySet g_rays;

static cudaTextureObject_t tex_tris[GPU_QUEUE_SIZE] = {0};

static TriV* d_tris[GPU_QUEUE_SIZE] = {0};
static Ray* d_rays[GPU_QUEUE_SIZE] = {0};
static float* d_tmaxs[GPU_QUEUE_SIZE] = {0};
static int* d_hits[GPU_QUEUE_SIZE] = {0};

static TriV* h_tris[GPU_QUEUE_SIZE] = {0};
static Ray* h_rays[GPU_QUEUE_SIZE] = {0};
static float* h_tmaxs[GPU_QUEUE_SIZE] = {0};
static int* h_hits[GPU_QUEUE_SIZE] = {0};

static int h_rends[GPU_QUEUE_SIZE] = {0};

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void kernel(const TriV* tris, int tbegin, int numTiles, int lastTileSize, const Ray* rays, int numRays, float* tmaxs, int* hits)
{
	extern __shared__ TriV sh_tris[];

	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	Ray ray;
	float tmax = FLT_MAX;
	int hit = -1;
	if(idx < numRays)
	{
		float4* ray4 = (float4*)&ray;
		const float4* rays4 = (const float4*)&rays[idx];
#ifdef TEC
		ray4[0] = rays4[0];
		ray4[1] = rays4[1];
#else
		ray4[0] = __ldg(rays4 + 0);
		ray4[1] = __ldg(rays4 + 1);
#endif

//		tmax = __ldg(tmaxs + idx);
//		hit = __ldg(hits + idx);
	}

	for(int tile = 0; tile < numTiles; ++tile)
	{
		const bool isLastTile = (tile == numTiles-1);

		if(!isLastTile || threadIdx.x < lastTileSize)
		{
			float4* tri4 = (float4*)&sh_tris[threadIdx.x];

			const float4* tris4 = (const float4*)&tris[tile*blockDim.x+threadIdx.x];
#ifdef TEC
			tri4[0] = tris4[0];
			tri4[1] = tris4[1];
			((float*)tri4)[8] = ((float*)tris4)[8];
#else
			tri4[0] = __ldg(tris4 + 0);
			tri4[1] = __ldg(tris4 + 1);
			((float*)tri4)[8] = __ldg((float*)tris4 + 8);
#endif

//			const int tid = (tile*blockDim.x+threadIdx.x)*3;
//			tri4[0] = tex1Dfetch<float4>(tris, tid);
//			tri4[1] = tex1Dfetch<float4>(tris, tid+1);
//			((float*)tri4)[8] = tex1Dfetch<float>(tris, tid+2);
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

//	if(idx < numRays)
	{
		tmaxs[idx] = tmax;
		hits[idx] = hit + tbegin;
	}
}

void updateKernelResults(cudaStream_t stream, cudaError_t status, void* userData)
{
//	nvtxRangePushA("updateKernelResults");
//	s_t.restart();

	int job = *((int*)&userData);
	int rend = h_rends[job];
	float* tmaxs = h_tmaxs[job];
	int* hits = h_hits[job];
	Ray* rays = h_rays[job];

	for(int i = 0; i < rend; ++i)
	{
		const Ray& ray = rays[i];
		if(tmaxs[i] < g_rays.tmaxs[ray.id])
		{
			g_rays.tmaxs[ray.id] = tmaxs[i];
			g_rays.hits[ray.id] = hits[i];
		}
	}

//	s_updateResultsTime += s_t.msec();

//	nvtxRangePop();
}

void intersectGPU(int tbegin, int tend, int rend)
{
	static cudaStream_t stream[GPU_QUEUE_SIZE];
	static cudaEvent_t jobEvent[GPU_QUEUE_SIZE];
	static int job = 0;
	static bool first = true;
	if(first)
	{
		for(int i = 0; i < GPU_QUEUE_SIZE; ++i)
		{
			cudaStreamCreate(stream + i);
			cudaEventCreateWithFlags(&jobEvent[i], cudaEventBlockingSync | cudaEventDisableTiming);
		}
		first = false;
	}

//	s_t.restart();
	const int tcount = tend - tbegin;
	const int numBlocks = ceilf((float)rend/(float)BLOCKSIZE);
	const int numTiles = ceilf((float)tcount/(float)BLOCKSIZE);
	const int lastTileSize = tcount < BLOCKSIZE? tcount : tcount % BLOCKSIZE;
//	s_configTime += s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaEventSynchronize(jobEvent[job]);
//	cudaDeviceSynchronize();
//	s_syncTime = s_t.msec();

	h_rends[job] = rend;

//	nvtxRangePushA("copyStuff");
//	s_t.restart();
	std::copy(g_tris.verts + tbegin, g_tris.verts + tend, h_tris[job]);
//	s_triCopyTime += s_t.msec();

//	s_t.restart();
	std::copy(g_rays.rays, g_rays.rays + rend, h_rays[job]);
//	s_rayCopyTime += s_t.msec();
//	nvtxRangePop();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaMemcpyAsync(d_tris[job], h_tris[job], tcount*sizeof(TriV), cudaMemcpyDefault, stream[job]);
//	cudaDeviceSynchronize();
//	s_triSendTime = s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaMemcpyAsync(d_rays[job], h_rays[job], rend*sizeof(Ray), cudaMemcpyDefault, stream[job]);
//	cudaDeviceSynchronize();
//	s_raySendTime = s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	kernel<<<numBlocks, BLOCKSIZE, BLOCKSIZE*sizeof(TriV), stream[job]>>>(d_tris[job], tbegin, numTiles, lastTileSize, d_rays[job], rend, d_tmaxs[job], d_hits[job]);
//	cudaDeviceSynchronize();
//	s_kernelTime += s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaMemcpyAsync(h_tmaxs[job], d_tmaxs[job], rend*sizeof(float), cudaMemcpyDefault, stream[job]);
//	cudaDeviceSynchronize();
//	s_tmaxReadTime += s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaMemcpyAsync(h_hits[job], d_hits[job], rend*sizeof(int), cudaMemcpyDefault, stream[job]);
//	cudaDeviceSynchronize();
//	s_hitReadTime += s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaStreamAddCallback(stream[job], updateKernelResults, (void*)job, 0);
//	cudaDeviceSynchronize();
//	s_addCallbackTime += s_t.msec();

//	cudaDeviceSynchronize();
//	s_t.restart();
	cudaEventRecord(jobEvent[job], stream[job]);
//	cudaDeviceSynchronize();
//	s_recordTime += s_t.msec();

//	s_t.restart();
	job = (job + 1) % GPU_QUEUE_SIZE;
	++s_numIntersectCalls;
//	s_finishTime += s_t.msec();
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

static int tt = 0;
static int rr = 0;

#define FILTER_RAYS filterRaysSSE

void trace(const Box& boxTris, const Box& boxCenters, int tbegin, int tend, int rend)
{
#ifdef HIT_CPU
	if((tend-tbegin) < 8 || rend < 8) // todo: define better
	{
		TIMED_CALL(intersectCPU(tbegin, tend, rend), s_intersectTime);
		rr += rend;
		tt += tend-tbegin;
		return;
	}
#elif defined HIT_GPU
	if((tend-tbegin) < 32768 && rend < 32768)  // todo: define better
	{
		static timer __t;
//		cudaDeviceSynchronize();
		__t.restart();
		intersectGPU(tbegin, tend, rend);
//		cudaDeviceSynchronize();
		rr += rend;
		tt += tend-tbegin;
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
		int rendUp = rend;
		TIMED_CALL(FILTER_RAYS(boxLeftTris, g_rays.rays, rend), s_filterRayTime);
		trace(boxLeftTris, boxLeftCenters, tbegin, tsplit, rend);
		rend = rendUp;
		TIMED_CALL(FILTER_RAYS(boxRightTris, g_rays.rays, rend), s_filterRayTime);
		trace(boxRightTris, boxRightCenters, tsplit, tend, rend);
	}
	else
	{
		int rendUp = rend;
		TIMED_CALL(FILTER_RAYS(boxRightTris, g_rays.rays, rend), s_filterRayTime);
		trace(boxRightTris, boxRightCenters, tsplit, tend, rend);
		rend = rendUp;
		TIMED_CALL(FILTER_RAYS(boxLeftTris, g_rays.rays, rend), s_filterRayTime);
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

//	cudaMemcpy(d_tmaxs, g_rays.tmaxs, g_rays.count*sizeof(float), cudaMemcpyDefault);
//	cudaMemcpy(d_hits, g_rays.hits, g_rays.count*sizeof(int), cudaMemcpyDefault);
}

void shadePixels(unsigned char* pixels)
{
#ifdef HIT_GPU
//	cudaMemcpy(g_rays.tmaxs, d_tmaxs, g_rays.count*sizeof(float), cudaMemcpyDefault);
//	cudaMemcpy(g_rays.hits, d_hits, g_rays.count*sizeof(int), cudaMemcpyDefault);
	cudaDeviceSynchronize();
#endif

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

void reshape2(int w, int h)
{
	g_canvas.w = w;
	g_canvas.h = h;

	const int npixels = g_canvas.w * g_canvas.h;
	g_rays.count = npixels;

	// CPU --------------------------------------------------------

	cudaFreeHost(g_rays.rays);
	cudaMallocHost(&g_rays.rays, g_rays.count*sizeof(Ray));

	cudaFreeHost(g_rays.tmaxs);
	cudaMallocHost(&g_rays.tmaxs, g_rays.count*sizeof(float));

	cudaFreeHost(g_rays.hits);
	cudaMallocHost(&g_rays.hits, g_rays.count*sizeof(int));

	// CPU pinned & GPU --------------------------------------------

	for(int i = 0; i < GPU_QUEUE_SIZE; ++i)
	{
		cudaFreeHost(h_rays[i]);
		cudaMallocHost(&h_rays[i], g_rays.count*sizeof(Ray));

		cudaFree(d_rays[i]);
		cudaMalloc(&d_rays[i], g_rays.count*sizeof(Ray));

		cudaFreeHost(h_tmaxs[i]);
		cudaMallocHost(&h_tmaxs[i], g_rays.count*sizeof(float));

		cudaFree(d_tmaxs[i]);
		cudaMalloc(&d_tmaxs[i], g_rays.count*sizeof(float));

		cudaFreeHost(h_hits[i]);
		cudaMallocHost(&h_hits[i], g_rays.count*sizeof(int));

		cudaFree(d_hits[i]);
		cudaMalloc(&d_hits[i], g_rays.count*sizeof(int));
	}
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
	s_configTime = 0.0;
	s_syncTime = 0.0;
	s_triCopyTime = 0.0;
	s_rayCopyTime = 0.0;
	s_triSendTime = 0.0;
	s_raySendTime = 0.0;
	s_kernelTime = 0.0;
	s_splitTime = 0.0;
	s_partitionTriTime = 0.0;
	s_filterRayTime = 0.0;
	s_tmaxReadTime = 0.0;
	s_hitReadTime = 0.0;
	s_addCallbackTime = 0.0;
	s_recordTime = 0.0;
	s_finishTime = 0.0;
	s_updateResultsTime = 0.0;
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

	std::cout << "    " << "config:    " << std::setw(9) << s_configTime << " ms" << std::endl;
	std::cout << "    " << "sync:    " << std::setw(9) << s_syncTime << " ms" << std::endl;
	std::cout << "    " << "triCopy:    " << std::setw(9) << s_triCopyTime << " ms" << std::endl;
	std::cout << "    " << "rayCopy:    " << std::setw(9) << s_rayCopyTime << " ms" << std::endl;
	std::cout << "    " << "triSend:    " << std::setw(9) << s_triSendTime << " ms" << std::endl;
	std::cout << "    " << "raySend:    " << std::setw(9) << s_raySendTime << " ms" << std::endl;
	std::cout << "    " << "kernel:    " << std::setw(9) << s_kernelTime << " ms (avg: " << s_kernelTime / s_numIntersectCalls << " ms)" << std::endl;
	std::cout << "    " << "tmaxRead:    " << std::setw(9) << s_tmaxReadTime << " ms" << std::endl;
	std::cout << "    " << "hitRead:    " << std::setw(9) << s_hitReadTime << " ms" << std::endl;
	std::cout << "    " << "addCallback:    " << std::setw(9) << s_addCallbackTime << " ms" << std::endl;
	std::cout << "    " << "record:    " << std::setw(9) << s_recordTime << " ms" << std::endl;
	std::cout << "    " << "finish:    " << std::setw(9) << s_finishTime << " ms" << std::endl;
	std::cout << "    " << "updateResults:    " << std::setw(9) << s_updateResultsTime << " ms" << std::endl;

	std::cout << "  " << "partitionTri: " << std::setw(9) << (int)s_partitionTriTime << " ms" << std::endl;

	t.restart();
	shadePixels(pixels);
	std::cout << "shade:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	std::cout << "RR_AVG: " << (float)rr / (float)s_numIntersectCalls << std::endl;
	std::cout << "TT_AVG: " << (float)tt / (float)s_numIntersectCalls << std::endl;

	rr = 0;
	tt = 0;
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

vec3 tovec3(const aiVector3D& v)
{
	return vec3(v.x, v.y, v.z);
}

void loadSceneBunny()
{
	// Create an instance of the Importer class
	Assimp::Importer importer;
	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll
	// propably to request more postprocessing than we do in this example.
	const aiScene* scene = importer.ReadFile("/home/environ/Downloads/dragon.ply",
		aiProcess_Triangulate            |
		aiProcess_JoinIdenticalVertices  |
		aiProcess_GenSmoothNormals |
		aiProcess_SortByPType);

	if(!scene->HasMeshes())
	{
		std::cout << "no meshes!" << std::endl;
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
			tri.v[0] = tovec3(mesh->mVertices[face.mIndices[0]]) * 50.0f;
			tri.v[1] = tovec3(mesh->mVertices[face.mIndices[1]]) * 50.0f;
			tri.v[2] = tovec3(mesh->mVertices[face.mIndices[2]]) * 50.0f;
			triangleVerts.push_back(tri);

			TriN triN;
			triN.n[0] = tovec3(mesh->mNormals[face.mIndices[0]]);
			triN.n[1] = tovec3(mesh->mNormals[face.mIndices[1]]);
			triN.n[2] = tovec3(mesh->mNormals[face.mIndices[2]]);
			triangleNorms.push_back(triN);
		}
	}

	std::cout << "nv: " << triangleVerts.size() << std::endl;
	std::cout << "nn: " << triangleNorms.size() << std::endl;

	g_tris.count = triangleVerts.size();

	cudaMallocHost(&g_tris.verts, g_tris.count*sizeof(TriV));
	std::copy(triangleVerts.begin(), triangleVerts.end(), g_tris.verts);

	g_tris.norms = new TriN[g_tris.count];
	std::copy(triangleNorms.begin(), triangleNorms.end(), g_tris.norms);
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

	for(int i = 0; i < GPU_QUEUE_SIZE; ++i)
	{
		cudaMallocHost(&h_tris[i], g_tris.count*sizeof(TriV));
		cudaMalloc(&d_tris[i], g_tris.count*sizeof(TriV));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = d_tris[i];
		resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.linear.desc.x = 32; // bits per channel
		resDesc.res.linear.desc.y = 32; // bits per channel
		resDesc.res.linear.desc.z = 32; // bits per channel
		resDesc.res.linear.desc.w = 32; // bits per channel
		resDesc.res.linear.sizeInBytes = g_tris.count*sizeof(TriV);

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		// create texture object: we only have to do this once!
		cudaCreateTextureObject(&tex_tris[i], &resDesc, &texDesc, NULL);
	}

	vec3 center = (g_tris.boxTris.min + g_tris.boxTris.max) * 0.5f;
	vec3 eye = center + vec3(0,0,10);
	vec3 up(0,1,0);
	rtvSetCamera(eye.ptr(), center.ptr(), up.ptr());
}

//---------------------------------------------------------------------------------------------------------------------

int main()
{
	rtvInit(1024, 1024);
	rtvSetReshapeCallback(reshape2);
	rtvSetCameraCallback(camera);
	rtvSetPixelRenderCallback(render);

//	loadScene1tri();
//	loadSceneRand();
	loadSceneBunny();

	endLoadScene();

	rtvExec();

	return 0;
}
