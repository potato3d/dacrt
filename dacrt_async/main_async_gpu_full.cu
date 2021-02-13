#include "rtv.h"
#include "rply.h"
#include "timer.h"
#define CUB_CDP
#include "cub/cub.cuh"
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cuda_gl_interop.h>

#define HIT_EPSILON 1e-6f
#define GPU_QUEUE_SIZE 10

#define INTERSECT_BLOCKSIZE 256
#define FILTER_BLOCKSIZE 256
#define GENERATE_BLOCKSIZE 16
#define SHADE_BLOCKSIZE 256

#define ERROR_DELTA 1e-6f

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__host__ __device__ inline float minf(float a, float b)
{
	return (a < b)? a : b;
}

__host__ __device__ inline float maxf(float a, float b)
{
	return (a > b)? a : b;
}

template<typename T>
__device__ inline void swap_device(T& a, T& b)
{
    T c(a); a=b; b=c;
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

__host__ __device__ bool diff(const vec3& a, const vec3& b)
{
	return  fabs(a.x-b.x) > ERROR_DELTA ||
			fabs(a.y-b.y) > ERROR_DELTA ||
			fabs(a.z-b.z) > ERROR_DELTA;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct TriV // 36 bytes
{
	vec3 v[3];
};

struct TriN // 36 bytes
{
	vec3 n[3];
};

struct Ray // 32 bytes
{
	vec3 o;
	vec3 d;
	float tmax;
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

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

static timer s_t;
static double s_intersectTime = 0.0;
static double s_intersectSyncTime = 0.0;
static double s_triCopyTime = 0.0;
static double s_triSendTime = 0.0;
static double s_kernelTime = 0.0;
static double s_splitTime = 0.0;
static double s_partitionTriTime = 0.0;
static double s_filterRayTime = 0.0;
#define TIMED_CALL(f, t) s_t.restart(); f; t += s_t.msec();
static int s_numIntersectCalls = 0;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct cudaGraphicsResource* g_pixelsResource = 0;

static TriSet g_tris;
static int g_rcount = 0;

static cudaStream_t g_stream;
static cudaEvent_t g_jobEvent[GPU_QUEUE_SIZE];
static int g_nextJob = 0;

static TriV* h_tris[GPU_QUEUE_SIZE] = {0};

static TriV* d_tris = 0;
static int* d_triIDs = 0;
static TriN* d_normals = 0;

static bool* d_flags = 0;
static int* d_hits = 0;

static Ray* d_rays_temp = 0;
static size_t d_rays_temp_size_bytes = 0;

__device__ Ray* d_rays;
__device__ Ray* d_rays_out;
__device__ int* d_rend;
__device__ int d_rendCurr;

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void kernelIntersect(const TriV* tris, int tbegin, int numTiles, int lastTileSize, Ray* rays, int numRays, int* hits)
{
	extern __shared__ TriV sh_tris[];

	const int idx = blockIdx.x*blockDim.x+threadIdx.x;

	Ray ray;
	if(idx < numRays)
	{
		ray = rays[idx];
	}

	int hit = -1;

	for(int tile = 0; tile < numTiles; ++tile)
	{
		const bool isLastTile = (tile == numTiles-1);

		if(!isLastTile || threadIdx.x < lastTileSize)
		{
			float4* shP = (float4*)(sh_tris + threadIdx.x);
			float4* trP = (float4*)(tris + tile*blockDim.x+threadIdx.x);

			float* shPx = (float*)(sh_tris + threadIdx.x);
			float* trPx = (float*)(tris + tile*blockDim.x+threadIdx.x);

			shP[0] = trP[0];
			shP[1] = trP[1];
			shPx[8] = trPx[8];
		}

		__syncthreads();

		if(idx < numRays)
		{
			const int limit = isLastTile ? lastTileSize : blockDim.x;
			for(int t = 0; t < limit; ++t)
			{
				const TriV triv = sh_tris[t];

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

				if(f >= ray.tmax || f < -HIT_EPSILON)
					continue;

				// store valid hit
				ray.tmax = f;
				hit = tile*blockDim.x + t;
			}
		}

		__syncthreads();
	}

	if(idx < numRays && hit >= 0)
	{
		rays[idx].tmax = ray.tmax;
		hits[ray.id] = hit + tbegin;
	}
}

__global__ void intersectLauncher(const TriV* tris, int tbegin, int numTiles, int lastTileSize, int* hits)
{
	const int rend = d_rend[d_rendCurr];
	if(rend == 0)
	{
		return;
	}
	const int numBlocks = ceilf((float)rend/(float)INTERSECT_BLOCKSIZE);
	kernelIntersect<<<numBlocks, INTERSECT_BLOCKSIZE, INTERSECT_BLOCKSIZE*sizeof(TriV)>>>(tris, tbegin, numTiles, lastTileSize, d_rays, rend, hits);
}

static int s_intersectRendAvg = 0;
static int s_intersectTcountAvg = 0;

void intersect(int tbegin, int tend)
{
//	int rendCurr;
//	cudaMemcpyFromSymbol(&rendCurr, d_rendCurr, sizeof(rendCurr));
//
//	int* drend;
//	cudaMemcpyFromSymbol(&drend, d_rend, sizeof(drend));
//
//	int* rend = new int[100];
//	cudaMemcpy(rend, drend, 100*sizeof(int), cudaMemcpyDefault);
//	//std::cout << "intersect rend: " << rend[rendCurr] << std::endl;
//
//	s_intersectRendAvg += rend[rendCurr];
//
//	delete[] rend;
//	s_intersectTcountAvg += tend-tbegin;




	timer tt;

	const int tcount = tend - tbegin;
	const int numTiles = ceilf((float)tcount/(float)INTERSECT_BLOCKSIZE);
	const int lastTileSize = tcount < INTERSECT_BLOCKSIZE? tcount : tcount % INTERSECT_BLOCKSIZE;

	TIMED_CALL(cudaEventSynchronize(g_jobEvent[g_nextJob]), s_intersectSyncTime);

	TIMED_CALL(std::copy(g_tris.verts + tbegin, g_tris.verts + tend, h_tris[g_nextJob]), s_triCopyTime);
	TIMED_CALL(cudaMemcpyAsync(d_tris, h_tris[g_nextJob], tcount*sizeof(TriV), cudaMemcpyDefault, g_stream), s_triSendTime);

	cudaEventRecord(g_jobEvent[g_nextJob], g_stream);

	cudaDeviceSynchronize();
	s_t.restart();
	intersectLauncher<<<1, 1, 0, g_stream>>>(d_tris, tbegin, numTiles, lastTileSize, d_hits);
	cudaDeviceSynchronize();
	s_kernelTime += s_t.msec();

	g_nextJob = (g_nextJob + 1) % GPU_QUEUE_SIZE;

	s_intersectTime += tt.msec();
	++s_numIntersectCalls;
}

__global__ void kernelClassifyRays(Box box, const Ray* rays, bool* rflags, int numRays)
{
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= numRays)
	{
		return;
	}

	// classify rays that intersect box (0: no intersect, 1: intersect)
	const Ray ray = rays[idx];
	const vec3 invDir = reciprocal(ray.d);

	const float tx1 = (box.min.x - ray.o.x) * invDir.x;
	const float tx2 = (box.max.x - ray.o.x) * invDir.x;

	float tmin = minf(tx1, tx2);
	float tmax = maxf(tx1, tx2);

	const float ty1 = (box.min.y - ray.o.y) * invDir.y;
	const float ty2 = (box.max.y - ray.o.y) * invDir.y;

	tmin = maxf(tmin, minf(ty1, ty2));
	tmax = minf(tmax, maxf(ty1, ty2));

	const float tz1 = (box.min.z - ray.o.z) * invDir.z;
	const float tz2 = (box.max.z - ray.o.z) * invDir.z;

	tmin = maxf(tmin, minf(tz1, tz2));
	tmax = minf(ray.tmax, minf(tmax, maxf(tz1, tz2)));

	rflags[idx] = tmin <= tmax;
}

__global__ void filterRaysLauncher(Box boxTris, Ray* rays_temp, size_t rays_temp_size_bytes, bool* rflags)
{
	const int rend = d_rend[d_rendCurr];
	if(rend == 0)
	{
		return;
	}
	const int numBlocks = ceilf((float)rend/(float)FILTER_BLOCKSIZE);
	kernelClassifyRays<<<numBlocks, FILTER_BLOCKSIZE>>>(boxTris, d_rays, rflags, rend);

	// partition rays using flags and find left partition size (refs: CUB, CUdpp, thrust)
	cub::DevicePartition::Flagged(rays_temp, rays_temp_size_bytes, d_rays, rflags, d_rays_out, d_rend + d_rendCurr, rend);

	cudaMemcpyAsync(d_rays, d_rays_out, rend*sizeof(Ray), cudaMemcpyDefault);
}

void filterRays(const Box& boxTris)
{
//	int rendCurr;
//	cudaMemcpyFromSymbol(&rendCurr, d_rendCurr, sizeof(rendCurr));
//
//	int* drend;
//	cudaMemcpyFromSymbol(&drend, d_rend, sizeof(drend));
//
//	int* rend = new int[100];
//	cudaMemcpy(rend, drend, 100*sizeof(int), cudaMemcpyDefault);
//
//	int rendBefore = rend[rendCurr];



	filterRaysLauncher<<<1, 1, 0, g_stream>>>(boxTris, d_rays_temp, d_rays_temp_size_bytes, d_flags);



//	cudaMemcpy(rend, drend, 100*sizeof(int), cudaMemcpyDefault);
//	std::cout << "stack: " << rendCurr << ", rendBefore: " << rendBefore << ", rendAfter: " << rend[rendCurr] << std::endl;
//
//	delete[] rend;
}

void partitionTris(int tbegin, int tend, int axis, float pos, int& tsplit, Box& boxTrisLeft, Box& boxCentersLeft, Box& boxTrisRight, Box& boxCentersRight)
{
	tsplit = tbegin;
	for(int t = tbegin; t < tend; ++t)
	{
		const vec3& center = g_tris.centers[t];
		if(center[axis] < pos)
		{
			boxTrisLeft.expand(g_tris.verts[t]);
			boxCentersLeft.expand(center);

			std::swap(g_tris.verts[t], g_tris.verts[tsplit]);
			std::swap(g_tris.centers[t], g_tris.centers[tsplit]);
			std::swap(g_tris.ids[t], g_tris.ids[tsplit]);
			++tsplit;
		}
		else
		{
			boxTrisRight.expand(g_tris.verts[t]);
			boxCentersRight.expand(center);
		}
	}
}

void split(const Box& box, int& axis, float& pos)
{
	const float dx = box.max.x - box.min.x;
	const float dy = box.max.y - box.min.y;
	const float dz = box.max.z - box.min.z;

	axis = (dx > dy && dx > dz)? 0 : (dy > dz)? 1 : 2;
	pos = (box.min[axis] + box.max[axis]) * 0.5f;
}

__global__ void kernelPushState()
{
	d_rend[d_rendCurr+1] = d_rend[d_rendCurr];
	++d_rendCurr;
}

__global__ void kernelPopState()
{
	--d_rendCurr;
}

void pushStateGPU()
{
	kernelPushState<<<1, 1, 0, g_stream>>>();
}

void popStateGPU()
{
	kernelPopState<<<1, 1, 0, g_stream>>>();
}

void trace(const Box& boxTris, const Box& boxCenters, int tbegin, int tend)
{
//	std::cout << "-------------------------" << std::endl;
//	std::cout << "tbegin: " << tbegin << ", tend: " << tend << ", total: " << tend-tbegin << std::endl;


	pushStateGPU();

	cudaDeviceSynchronize();
	s_t.restart();
	filterRays(boxTris);
	cudaDeviceSynchronize();
	s_filterRayTime += s_t.msec();

	if((tend-tbegin) <= 8192)
	{
//		int rendCurr;
//		cudaMemcpyFromSymbol(&rendCurr, d_rendCurr, sizeof(rendCurr));
//		int* drend;
//		cudaMemcpyFromSymbol(&drend, d_rend, sizeof(drend));
//		int rend;
//		cudaMemcpy(&rend, drend + rendCurr, sizeof(rend), cudaMemcpyDeviceToHost);
//		if(rend <= 32768)
		{
			intersect(tbegin, tend);
			popStateGPU();
			return;
		}
	}

	int axis;
	float pos;
	TIMED_CALL(split(boxCenters, axis, pos), s_splitTime);

	Box boxTrisLeft;
	Box boxCentersLeft;
	Box boxTrisRight;
	Box boxCentersRight;
	int tsplit;
	TIMED_CALL(partitionTris(tbegin, tend, axis, pos, tsplit, boxTrisLeft, boxCentersLeft, boxTrisRight, boxCentersRight), s_partitionTriTime);

	// TODO: how to order traversal? if rays[0].d[axis] > 0 then recurse to the left first

	trace(boxTrisLeft, boxCentersLeft, tbegin, tsplit);
	trace(boxTrisRight, boxCentersRight, tsplit, tend);

	popStateGPU();
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

__global__ void kernelGeneratePrimary(vec3 camPos, vec3 lowerLeftDir, vec3 du, vec3 dv, Ray* rays, int numRaysX, int numRaysY, int* hits)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(tx >= numRaysX || ty >= numRaysY)
	{
		return;
	}

	const int globalIdx = ty * numRaysX + tx;

	Ray ray;
	ray.o = camPos;
	ray.d = lowerLeftDir + du * tx + dv * ty + du * numRaysX * ty;
	ray.tmax = FLT_MAX;
	ray.id = globalIdx;

	rays[globalIdx] = ray;
	hits[globalIdx] = -1;
}

__global__ void generatePrimaryLauncher(dim3 numBlocks, vec3 camPos, vec3 lowerLeftDir, vec3 du, vec3 dv, int numRaysX, int numRaysY, int* hits)
{
	dim3 blockSize;
	blockSize.x = GENERATE_BLOCKSIZE;
	blockSize.y = GENERATE_BLOCKSIZE;
	kernelGeneratePrimary<<<numBlocks, blockSize>>>(camPos, lowerLeftDir, du, dv, d_rays, numRaysX, numRaysY, hits);
	d_rendCurr = 0;
	d_rend[d_rendCurr] = numRaysX * numRaysY;
}

void generatePrimary()
{
	dim3 numBlocks;
	numBlocks.x = ceilf(((float)g_camera.nu/(float)GENERATE_BLOCKSIZE));
	numBlocks.y = ceilf(((float)g_camera.nv/(float)GENERATE_BLOCKSIZE));
	generatePrimaryLauncher<<<1, 1, 0, g_stream>>>(numBlocks, g_camera.position, g_camera.lowerLeftDir, g_camera.du, g_camera.dv, g_camera.nu, g_camera.nv, d_hits);
}

__global__ void kernelShadePixels(unsigned char* pixels, const Ray* rays, int numRays, const TriV* tris, const int* triIDs, const TriN* normals, const int* hits)
{
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if(idx >= numRays)
	{
		return;
	}

	const Ray ray = rays[idx];

	if(ray.tmax == FLT_MAX)
	{
		pixels[ray.id*3+0] = 0;
		pixels[ray.id*3+1] = 0;
		pixels[ray.id*3+2] = 0;
		return;
	}

	// recover triangle hit
	const int tid = hits[ray.id];
	const TriV tri = tris[tid];
	const TriN trin = normals[triIDs[tid]];

	// recompute hit position
	const vec3 hitPos = ray.o + ray.d * ray.tmax;

	// recompute barycentric coordinates
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
	const vec3 lerpN = normalize(trin.n[0]*u + trin.n[1]*v + trin.n[2]*w);

	// compute final color
	const unsigned char color = 255 * dot(lerpN, -normalize(ray.d));

	// store in pixel
	pixels[ray.id*3+0] = color;
	pixels[ray.id*3+1] = color;
	pixels[ray.id*3+2] = color;
}

__global__ void shadePixelsLauncher(int numBlocks, unsigned char* pixels, int numRays, const TriV* tris, const int* triIDs, const TriN* normals, const int* hits)
{
	kernelShadePixels<<<numBlocks, SHADE_BLOCKSIZE>>>(pixels, d_rays, numRays, tris, triIDs, normals, hits);
}

void shadePixels()
{
	cudaMemcpyAsync(d_tris, g_tris.verts, g_tris.count*sizeof(TriV), cudaMemcpyDefault, g_stream);
	cudaMemcpyAsync(d_triIDs, g_tris.ids, g_tris.count*sizeof(int), cudaMemcpyDefault, g_stream);
	cudaMemcpyAsync(d_normals, g_tris.norms, g_tris.count*sizeof(TriN), cudaMemcpyDefault, g_stream);

	unsigned char* pixels = 0;
	cudaGraphicsMapResources(1, &g_pixelsResource, g_stream);
	size_t num_bytes = 0;
	cudaGraphicsResourceGetMappedPointer((void**)&pixels, &num_bytes, g_pixelsResource);

	const int numBlocks = ceilf(((float)(g_rcount)/(float)SHADE_BLOCKSIZE));
	shadePixelsLauncher<<<1, 1, 0, g_stream>>>(numBlocks, pixels, g_rcount, d_tris, d_triIDs, d_normals, d_hits);

	cudaGraphicsUnmapResources(1, &g_pixelsResource, g_stream);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void reshape(int w, int h)
{
	g_canvas.w = w;
	g_canvas.h = h;

	g_rcount = g_canvas.w * g_canvas.h;

	Ray* rays;
	cudaMemcpyFromSymbol(&rays, d_rays, sizeof(rays));
	cudaFree(rays);
	cudaMalloc(&rays, g_rcount*sizeof(Ray));
	cudaMemcpyToSymbol(d_rays, &rays, sizeof(rays));

	Ray* rays_out;
	cudaMemcpyFromSymbol(&rays_out, d_rays_out, sizeof(rays_out));
	cudaFree(rays_out);
	cudaMalloc(&rays_out, g_rcount*sizeof(Ray));
	cudaMemcpyToSymbol(d_rays_out, &rays_out, sizeof(rays_out));

	cudaFree(d_hits);
	cudaMalloc(&d_hits, g_rcount*sizeof(int));

	cudaFree(d_flags);
	cudaMalloc(&d_flags, g_rcount*sizeof(bool));

	cudaFree(d_rays_temp);
	d_rays_temp = 0;
	Ray* in = 0;
	Ray* out = 0;
	int* n = 0;
	cub::DevicePartition::Flagged(d_rays_temp, d_rays_temp_size_bytes, in, d_flags, out, n, g_rcount);
	cudaMalloc(&d_rays_temp, d_rays_temp_size_bytes);

	int* rend;
	cudaMemcpyFromSymbol(&rend, d_rend, sizeof(rend));
	cudaFree(rend);
	cudaMalloc(&rend, 1000*sizeof(int));
	cudaMemcpyToSymbol(d_rend, &rend, sizeof(rend));
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

void render(unsigned int pixelBufferID)
{
//	static int N = 0;
//	if(N++>10)exit(0);


	if(!g_pixelsResource)
	{
		cudaGraphicsGLRegisterBuffer(&g_pixelsResource, pixelBufferID, cudaGraphicsMapFlagsWriteDiscard);
	}

	timer t;
	s_intersectTime = 0.0;
	s_intersectSyncTime = 0.0;
	s_triCopyTime = 0.0;
	s_triSendTime = 0.0;
	s_kernelTime = 0.0;
	s_splitTime = 0.0;
	s_partitionTriTime = 0.0;
	s_filterRayTime = 0.0;
	s_numIntersectCalls = 0;
	s_intersectRendAvg = 0;
	s_intersectTcountAvg = 0;

	std::cout << "---------------------------------" << std::endl;

	t.restart();
	generatePrimary();
	std::cout << "generate: " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	t.restart();
	trace(g_tris.boxTris, g_tris.boxCenters, 0, g_tris.count);
	std::cout << "trace:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	std::cout << "  " << "filterRay:    " << std::setw(9) << (int)s_filterRayTime << " ms" << std::endl;

	std::cout << "  " << "split:        " << std::setw(9) << (int)s_splitTime << " ms" << std::endl;
	std::cout << "  " << "intersect:    " << std::setw(9) << (int)s_intersectTime << " ms (count: " << s_numIntersectCalls << ", avg: " << s_intersectTime / s_numIntersectCalls << " ms)" << std::endl;

	std::cout << "    " << "sync:       " << std::setw(9) << s_intersectSyncTime << " ms" << std::endl;
	std::cout << "    " << "triCopy:    " << std::setw(9) << s_triCopyTime << " ms" << std::endl;
	std::cout << "    " << "triSend:    " << std::setw(9) << s_triSendTime << " ms" << std::endl;
	std::cout << "    " << "kernel:     " << std::setw(9) << s_kernelTime << " ms (avg: " << s_kernelTime / s_numIntersectCalls << " ms)" << std::endl;

	std::cout << "  " << "partitionTri: " << std::setw(9) << (int)s_partitionTriTime << " ms" << std::endl;

	t.restart();
	shadePixels();
	std::cout << "shade:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	std::cout << "intersect rend avg: " << (float)s_intersectRendAvg / s_numIntersectCalls << std::endl;
	std::cout << "intersect tcount avg: " << (float)s_intersectTcountAvg / s_numIntersectCalls << std::endl;
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
	cudaMallocHost(&g_tris.norms, 1*sizeof(TriN));
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
	cudaMallocHost(&g_tris.norms, g_tris.count*sizeof(TriN));
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
	p_ply ply = ply_open("/home/potato/Downloads/bunny.ply", NULL, 0, NULL);
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
	cudaMallocHost(&g_tris.norms, g_tris.count*sizeof(TriN));

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
	cudaMallocHost(&g_tris.ids, g_tris.count*sizeof(int));
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
	}

	cudaMalloc(&d_tris, g_tris.count*sizeof(TriV));
	cudaMalloc(&d_triIDs, g_tris.count*sizeof(int));
	cudaMalloc(&d_normals, g_tris.count*sizeof(TriN));

	vec3 center = (g_tris.boxTris.min + g_tris.boxTris.max) * 0.5f;
	vec3 eye = center + vec3(0,0,10);
	vec3 up(0,1,0);
	rtvSetCamera(eye.ptr(), center.ptr(), up.ptr());
}

//---------------------------------------------------------------------------------------------------------------------

__device__ int dN;

int main()
{
	rtvInit(1024, 1024);
	rtvSetReshapeCallback(reshape);
	rtvSetCameraCallback(camera);
	rtvSetBufferRenderCallback(render);

//	loadScene1tri();
//	loadSceneRand();
	loadSceneBunny();

	endLoadScene();

	cudaStreamCreate(&g_stream);
	for(int i = 0; i < GPU_QUEUE_SIZE; ++i)
	{
		cudaEventCreate(&g_jobEvent[i]);
	}

	rtvExec();

	return 0;
}
