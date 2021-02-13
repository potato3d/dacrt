#include "rtv.h"
#include "rply.h"
#include "timer.h"
#include <cfloat>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

//---------------------------------------------------------------------------------------------------------------------

#define HIT_GPU

#define BLOCKSIZE 512

#ifdef HIT_CPU
#define TRI_LIMIT 8
#define RAY_LIMIT 8
#elif defined(HIT_GPU)
#define TRI_LIMIT 8192
#define RAY_LIMIT 32768
#endif

#define HIT_EPSILON 1e-6f

//---------------------------------------------------------------------------------------------------------------------

struct vec3
{
	float x,y,z;
	__host__ __device__ vec3() : x(0), y(0), z(0) {}
	__host__ __device__ explicit vec3(float a) : x(a), y(a), z(a) {}
	__host__ __device__ explicit vec3(float* p) : x(p[0]), y(p[1]), z(p[2]) {}
	__host__ __device__ vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	__host__ __device__ float* ptr() { return &x; }
	__host__ __device__ bool operator!=(const vec3& o) const { return x != o.x || y != o.y || z != o.z; }
	__host__ __device__ vec3 operator+(const vec3& o) const { return vec3(x+o.x, y+o.y, z+o.z); }
	__host__ __device__ vec3 operator-(const vec3& o) const { return vec3(x-o.x, y-o.y, z-o.z); }
	__host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__ void operator+=(const vec3& o) { x+=o.x; y+=o.y; z+=o.z; }
	__host__ __device__ vec3 operator*(float a) const { return vec3(x*a, y*a, z*a); }
	__host__ __device__ void operator*=(float a) { x*=a; y*=a; z*=a; }
	__host__ __device__ float dot(const vec3& o) const { return x*o.x + y*o.y + z*o.z; }
	__host__ __device__ vec3 cross(const vec3& o) const { return vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x); }
	__host__ __device__ float length() const { return sqrtf(x*x + y*y + z*z); }
	__host__ __device__ vec3 normalized() const
	{
		const float invLen = 1.0f / length();
		return vec3(x*invLen, y*invLen, z*invLen);
	}
	__host__ __device__ vec3 reciprocal() const
	{
		return vec3(1.0f/x, 1.0f/y, 1.0f/z);
	}
	__host__ __device__ float operator[](int i) const { return (&x)[i]; }
	__host__ __device__ float& operator[](int i) { return (&x)[i]; }
};

struct TriV
{
	vec3 v0, v1, v2;
};

struct TriN
{
	vec3 n0, n1, n2;
};

struct Ray
{
	vec3 o;
	vec3 d;
};

struct Box
{
	vec3 min, max;
	Box() : min(FLT_MAX), max(-FLT_MAX) {}
	void expand(const vec3& v)
	{
		min.x = std::min(min.x, v.x);
		min.y = std::min(min.y, v.y);
		min.z = std::min(min.z, v.z);

		max.x = std::max(max.x, v.x);
		max.y = std::max(max.y, v.y);
		max.z = std::max(max.z, v.z);
	}
	void expand(const TriV& tri)
	{
		expand(tri.v0);
		expand(tri.v1);
		expand(tri.v2);
	}
};

struct Canvas
{
	bool operator!=(const Canvas& o) { return w != o.w || h != o.h; }
	int w;
	int h;
};

struct LookAt
{
	bool operator!=(const LookAt& o) { return eye != o.eye || center != o.center || up != o.up; }
	vec3 eye;
	vec3 center;
	vec3 up;
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

struct TriSet
{
	TriSet() : ids(0), tris(0), norms(0), count(0), end(0) {}
	int* ids;
	TriV* tris;
	TriN* norms;
	int count;
	int end;

};

struct RaySet
{
	RaySet() : ids(0), rays(0), tmaxs(0), hits(0), count(0), end(0) {}
	int* ids;
	Ray* rays;
	float* tmaxs;
	int* hits;
	int count;
	int end;
};

//---------------------------------------------------------------------------------------------------------------------

__global__ void kernel(TriV* tris, int* triIDs, int numTiles, int lastTileSize, Ray* rays, int* rayIDs, int numRays, float* tmaxs, int* hits)
{
	// per-block shared memory
	__shared__ char sh_tris_byte[BLOCKSIZE*sizeof(TriV)];
	TriV* sh_tris = (TriV*)sh_tris_byte;

	// compute this thread's global index
	int gThreadIdx = blockIdx.x*blockDim.x + threadIdx.x;

	// get this thread's ray information
	Ray ray;
	float tmax;

	if(gThreadIdx < numRays)
	{
		ray = rays[rayIDs[gThreadIdx]];
		tmax = tmaxs[gThreadIdx];
	}

	// ID of triangle hit, if any
	int hit = -1;

	// for each tile of triangles
	for(int tile = 0; tile < numTiles; ++tile)
	{
		const bool isLastTile = (tile == numTiles-1);

		// load next tile in parallel: each thread loads a separate triangle from global to shared memory
		if(!isLastTile || threadIdx.x < lastTileSize)
		{
			sh_tris[threadIdx.x] = tris[triIDs[tile*blockDim.x + threadIdx.x]];
		}

		// make sure everything is loaded in shared memory
		__syncthreads();
		// ---------------------------------------------------------------------------------------------------

		if(gThreadIdx < numRays)
		{
			// for each triangle in shared memory
			const int limit = isLastTile ? lastTileSize : blockDim.x;
			for(int tid = 0; tid < limit; ++tid)
			{
				const TriV tri = sh_tris[tid];

				/* find vectors for two edges sharing vert0 */
				const vec3 edge1 = tri.v1 - tri.v0;
				const vec3 edge2 = tri.v2 - tri.v0;

				/* begin calculating determinant - also used to calculate U parameter */
				const vec3 pvec = ray.d.cross(edge2);

				/* if determinant is near zero, ray lies in plane of triangle */
				const float det = edge1.dot(pvec);

				if(det > -HIT_EPSILON && det < HIT_EPSILON)
					continue;

				const float inv_det = 1.0f / det;

				/* calculate distance from vert0 to ray origin */
				const vec3 tvec = ray.o - tri.v0;

				/* calculate U parameter and test bounds */
				const float u = tvec.dot(pvec) * inv_det;
				if(u < 0.0f || u > 1.0f)
					continue;

				/* prepare to test V parameter */
				const vec3 qvec = tvec.cross(edge1);

				/* calculate V parameter and test bounds */
				const float v = ray.d.dot(qvec) * inv_det;
				if(v < 0.0f || u + v > 1.0f)
					continue;

				/* calculate t, ray hits triangle */
				const float f = edge2.dot(qvec) * inv_det;

				if((f >= tmax) || (f < -HIT_EPSILON))
					continue;

				// Have a valid hit point here. Store it.
				tmax = f;
				hit = tile*blockDim.x + tid;
			}
		}

		// wait for every thread to compute its intersections before looping back and loading new tile
		__syncthreads();
		// ---------------------------------------------------------------------------------------------------
	}

	// return final results
	if(gThreadIdx < numRays)
	{
		tmaxs[gThreadIdx] = tmax;
		hits[gThreadIdx] = hit;
	}
}

//---------------------------------------------------------------------------------------------------------------------

static timer s_t;
static double s_intersectTime = 0.0;
static double s_triSendTime = 0.0;
static double s_raySendTime = 0.0;
static double s_tmaxCopyTime = 0.0;
static double s_tmaxSendTime = 0.0;
static double s_kernelTime = 0.0;
static double s_tmaxReadTime = 0.0;
static double s_hitReadTime = 0.0;
static double s_hitUpdateTime = 0.0;
static double s_splitTime = 0.0;
static double s_partitionTriTime = 0.0;
static double s_partitionRayTime = 0.0;
#define TIMED_CALL(f, t) s_t.restart(); f; t += s_t.msec();
static int s_numKernelCalls = 0;



// GPU data
int* d_triIDs = 0;
TriV* d_tris = 0;

int* d_rayIDs = 0;
Ray* d_rays = 0;
float* d_tmaxs = 0;
int* d_hits = 0;

// Host data, used to read back results temporarily
float* h_tmaxs = 0;
int* h_hits = 0;

#ifdef HIT_GPU
void intersect(const TriSet& triSet, RaySet& raySet)
{
	// send current IDs to GPU
	TIMED_CALL(cudaMemcpy(d_triIDs, triSet.ids, triSet.end*sizeof(int), cudaMemcpyDefault), s_triSendTime);
	TIMED_CALL(cudaMemcpy(d_rayIDs, raySet.ids, raySet.end*sizeof(int), cudaMemcpyDefault), s_raySendTime);

	// send current tmax's to GPU
	s_t.restart();
	for(int i = 0; i < raySet.end; ++i)
	{
		h_tmaxs[i] = raySet.tmaxs[raySet.ids[i]];
	}
	s_tmaxCopyTime += s_t.msec();
	TIMED_CALL(cudaMemcpy(d_tmaxs, h_tmaxs, raySet.end*sizeof(float), cudaMemcpyDefault), s_tmaxSendTime);

	// determine parameters
	const int numBlocks = ceilf((float)raySet.end / (float)BLOCKSIZE);
	const int numTiles = ceilf((float)triSet.end / (float)BLOCKSIZE);
	const int lastTileSize = triSet.end < BLOCKSIZE? triSet.end : triSet.end % BLOCKSIZE;

	// launch kernel
	s_t.restart();
	kernel<<<numBlocks, BLOCKSIZE>>>(d_tris, d_triIDs, numTiles, lastTileSize, d_rays, d_rayIDs, raySet.end, d_tmaxs, d_hits);
	cudaDeviceSynchronize();
	s_kernelTime += s_t.msec();
	++s_numKernelCalls;

	// retrieve all results from GPU
	TIMED_CALL(cudaMemcpy(h_tmaxs, d_tmaxs, raySet.end*sizeof(float), cudaMemcpyDefault), s_tmaxReadTime);
	TIMED_CALL(cudaMemcpy(h_hits, d_hits, raySet.end*sizeof(int), cudaMemcpyDefault), s_hitReadTime);

	// update data on CPU
	s_t.restart();
	for(int i = 0; i < raySet.end; ++i)
	{
		if(h_hits[i] >= 0)
		{
			const int rayID = raySet.ids[i];
			raySet.tmaxs[rayID] = h_tmaxs[i];
			raySet.hits[rayID] = triSet.ids[h_hits[i]];
		}
	}
	s_hitUpdateTime += s_t.msec();
}
#endif

//---------------------------------------------------------------------------------------------------------------------

#ifdef HIT_CPU
void intersect(const TriSet& triSet, RaySet& raySet)
{
	for(int t = 0; t < triSet.end; ++t)
	{
		const int triID = triSet.ids[t];
		const TriV& tri = triSet.tris[triID];

		for(int r = 0; r < raySet.end; ++r)
		{
			const int rayID = raySet.ids[r];
			Ray& ray = raySet.rays[rayID];

			/* find vectors for two edges sharing vert0 */
			const vec3 edge1 = tri.v1 - tri.v0;
			const vec3 edge2 = tri.v2 - tri.v0;

			/* begin calculating determinant - also used to calculate U parameter */
			const vec3 pvec = ray.d.cross(edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			const float det = edge1.dot(pvec);

			if(det > -HIT_EPSILON && det < HIT_EPSILON)
				continue;

			const float inv_det = 1.0f / det;

			/* calculate distance from vert0 to ray origin */
			const vec3 tvec = ray.o - tri.v0;

			/* calculate U parameter and test bounds */
			const float u = tvec.dot(pvec) * inv_det;
			if(u < 0.0f || u > 1.0f)
				continue;

			/* prepare to test V parameter */
			const vec3 qvec = tvec.cross(edge1);

			/* calculate V parameter and test bounds */
			const float v = ray.d.dot(qvec) * inv_det;
			if(v < 0.0f || u + v > 1.0f)
				continue;

			/* calculate t, ray hits triangle */
			const float f = edge2.dot(qvec) * inv_det;

			if((f >= raySet.tmaxs[rayID]) || (f < -HIT_EPSILON))
				continue;

			// Have a valid hit point here. Store it.
			raySet.tmaxs[rayID] = f;
			raySet.hits[rayID] = triID;
		}
	}
}
#endif

void splitBox(const Box& triBox, const TriSet& triSet, const RaySet& raySet, Box& nearBox, Box& farBox)
{
	const float dx = triBox.max.x - triBox.min.x;
	const float dy = triBox.max.y - triBox.min.y;
	const float dz = triBox.max.z - triBox.min.z;
	const int axis = (dx > dy && dx > dz)? 0 : (dy > dz)? 1 : 2;

	const float pos = (triBox.min[axis] + triBox.max[axis]) * 0.5f;

	Box left = triBox;
	Box right = triBox;
	left.max[axis] = pos;
	right.min[axis] = pos;

	const bool leftNear = raySet.rays[0].d[pos] >= 0.0f;
	nearBox = leftNear? left : right;
	farBox = leftNear? right : left;
}

void partitionTris(const Box& box, TriSet& triSet)
{
	int newEnd = 0;
	for(int i = 0; i < triSet.end; ++i)
	{
		Box triBox;
		triBox.expand(triSet.tris[triSet.ids[i]]);

		if(box.max.x < triBox.min.x) continue;
		if(box.min.x > triBox.max.x) continue;
		if(box.max.y < triBox.min.y) continue;
		if(box.min.y > triBox.max.y) continue;
		if(box.max.z < triBox.min.z) continue;
		if(box.min.z > triBox.max.z) continue;

		std::swap(triSet.ids[i], triSet.ids[newEnd]);
		++newEnd;
	}
	triSet.end = newEnd;
}

void partitionRays(const Box& box, RaySet& raySet)
{
	int newEnd = 0;
	for(int i = 0; i < raySet.end; ++i)
	{
		const Ray& ray = raySet.rays[raySet.ids[i]];
		const vec3 invDir = ray.d.reciprocal();

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
		tmax = std::min(raySet.tmaxs[raySet.ids[i]], std::min(tmax, std::max(tz1, tz2)));

		if(tmin > tmax) continue;

		std::swap(raySet.ids[i], raySet.ids[newEnd]);
		++newEnd;
	}
	raySet.end = newEnd;
}

//---------------------------------------------------------------------------------------------------------------------

void reallocData(const Canvas& canvas, RaySet& raySet)
{
	const int npixels = canvas.w * canvas.h;
	raySet.count = npixels;

	// ---------------- v CPU v ----------------

	cudaFreeHost(raySet.ids);
	cudaMallocHost(&raySet.ids, raySet.count*sizeof(int));
	for(int i = 0; i < raySet.count; ++i)
	{
		raySet.ids[i] = i;
	}

	cudaFreeHost(raySet.rays);
	cudaMallocHost(&raySet.rays, raySet.count*sizeof(Ray));

	cudaFreeHost(raySet.tmaxs);
	cudaMallocHost(&raySet.tmaxs, raySet.count*sizeof(float));

	cudaFreeHost(h_tmaxs);
	cudaMallocHost(&h_tmaxs, raySet.count*sizeof(float));

	cudaFreeHost(raySet.hits);
	cudaMallocHost(&raySet.hits, raySet.count*sizeof(int));

	cudaFreeHost(h_hits);
	cudaMallocHost(&h_hits, raySet.count*sizeof(int));

	// ---------------- v GPU v ----------------

	cudaFree(d_rayIDs);
	cudaMalloc(&d_rayIDs, raySet.count*sizeof(int));

	cudaFree(d_rays);
	cudaMalloc(&d_rays, raySet.count*sizeof(Ray));

	cudaFree(d_tmaxs);
	cudaMalloc(&d_tmaxs, raySet.count*sizeof(float));

	cudaFree(d_hits);
	cudaMalloc(&d_hits, raySet.count*sizeof(int));
}

void updateCamera(const Canvas& canvas, const LookAt& lookAt, Camera& camera)
{
	// store position
	camera.position = lookAt.eye;

	// pre-computations
	float invHeight = 1.0f / canvas.h;
	float invWidth  = 1.0f / canvas.w;

	// compute camera basis
	vec3 axisW = (lookAt.eye - lookAt.center).normalized();
	vec3 axisV = lookAt.up.normalized();
	vec3 axisU = axisV.cross(axisW);

	// compute half scale factors for each basis vector
	float sw = canvas.w * 0.01f; // try to keep directions around zero in floating-point value
	float sv = sw * std::tan(0.523598775f); // half 60o in radians
	float su = sv * canvas.w * invHeight;

	// scale each vector
	axisW *= sw;
	axisV *= sv;
	axisU *= su;

	// store final direction
	camera.lowerLeftDir = - axisU - axisV - axisW;

	// compute full scales
	axisV *= 2.0f;
	axisU *= 2.0f;

	// interpolation deltas
	camera.dv = axisV * invHeight - axisU; // also goes back to start of u-axis
	camera.du = axisU * invWidth;

	// number of pixels in U and V directions
	camera.nu = canvas.w;
	camera.nv = canvas.h;
}

void generatePrimary(const Camera& camera, RaySet& raySet)
{
	vec3 dir = camera.lowerLeftDir;

	for(int v = 0; v < camera.nv; ++v)
	{
		for(int u = 0; u < camera.nu; ++u)
		{
			Ray& r = raySet.rays[v*camera.nu+u];
			r.o = camera.position;
			r.d = dir;
			dir += camera.du;
		}
		dir += camera.dv;
	}

	for(int i = 0; i < raySet.count; ++i)
	{
		raySet.tmaxs[i] = FLT_MAX;
		raySet.hits[i]  = -1;
	}

	cudaMemcpy(d_rays, raySet.rays, raySet.count*sizeof(Ray), cudaMemcpyDefault);
}

void traceRays(const Box& triBox, TriSet& triSet, RaySet& raySet)
{
#ifdef HIT_GPU
	if(triSet.end == 0 || raySet.end == 0)
	{
		return;
	}
	if(triSet.end < TRI_LIMIT && raySet.end < RAY_LIMIT)
#endif
#ifdef HIT_CPU
	if(triSet.end < TRI_LIMIT || raySet.end < RAY_LIMIT)
#endif
	{
		static timer s_int_t;
		s_int_t.restart();
		intersect(triSet, raySet);
		s_intersectTime += s_int_t.msec();
		return;
	}

	Box nearBox;
	Box farBox;
	TIMED_CALL(splitBox(triBox, triSet, raySet, nearBox, farBox), s_splitTime);

	int triEnd = triSet.end;
	int rayEnd = raySet.end;

	TIMED_CALL(partitionTris(nearBox, triSet), s_partitionTriTime);
	TIMED_CALL(partitionRays(nearBox, raySet), s_partitionRayTime);
	traceRays(nearBox, triSet, raySet);

	triSet.end = triEnd;
	raySet.end = rayEnd;

	TIMED_CALL(partitionTris(farBox, triSet), s_partitionTriTime);
	TIMED_CALL(partitionRays(farBox, raySet), s_partitionRayTime);
	traceRays(farBox, triSet, raySet);
}

void doTraceRays(const Box& triBox, TriSet& triSet, RaySet& raySet)
{
	triSet.end = triSet.count;
	raySet.end = raySet.count;
	traceRays(triBox, triSet, raySet);
}

void shadePixels(const TriSet& triSet, const RaySet& raySet, unsigned char* pixels)
{
	for(int i = 0; i < raySet.count; ++i)
	{
		unsigned char c = 0;
		const int triID = raySet.hits[i];

		if(triID >= 0)
		{
			// recompute hit position
			const Ray& ray = raySet.rays[i];
			const vec3 hitPos = ray.o + ray.d * raySet.tmaxs[i];

			// recompute barycentric coordinates
			const TriV& tri = triSet.tris[triID];
			const vec3 e0 = tri.v1 - tri.v0;
			const vec3 e1 = tri.v2 - tri.v0;
			const vec3 e2 = hitPos - tri.v0;
			const float d00 = e0.dot(e0);
			const float d01 = e0.dot(e1);
			const float d11 = e1.dot(e1);
			const float d20 = e2.dot(e0);
			const float d21 = e2.dot(e1);
			const float invDenom = 1.0f / (d00 * d11 - d01 * d01);
			const float v = (d11 * d20 - d01 * d21) * invDenom;
			const float w = (d00 * d21 - d01 * d20) * invDenom;
			const float u = 1.0f - v - w;

			// lerp normal
			const TriN& norm = triSet.norms[triID];
			const vec3 lerpN = (norm.n0*u + norm.n1*v + norm.n2*w).normalized();

			// compute final color
			c = 255 * lerpN.dot(-ray.d.normalized());
		}

		pixels[i*3+0] = c;
		pixels[i*3+1] = c;
		pixels[i*3+2] = c;
	}
}

//---------------------------------------------------------------------------------------------------------------------

static Canvas s_canvas;
static LookAt s_lookAt;
static Camera s_camera;
static TriSet s_triSet;
static Box    s_triBox;
static RaySet s_raySet;

void reshape(int w, int h)
{
	s_canvas.w = w;
	s_canvas.h = h;
	reallocData(s_canvas, s_raySet);
}

void camera(float* eye, float* center, float* up)
{
	s_lookAt.eye = vec3(eye);
	s_lookAt.center = vec3(center);
	s_lookAt.up = vec3(up);
	updateCamera(s_canvas, s_lookAt, s_camera);
}

void render(unsigned char* pixels)
{
	timer t;
	s_splitTime = 0.0;
	s_intersectTime = 0.0;
	s_partitionTriTime = 0.0;
	s_partitionRayTime = 0.0;
	s_triSendTime = 0.0;
	s_raySendTime = 0.0;
	s_tmaxCopyTime = 0.0;
	s_tmaxSendTime = 0.0;
	s_kernelTime = 0.0;
	s_tmaxReadTime = 0.0;
	s_hitReadTime = 0.0;
	s_hitUpdateTime = 0.0;
	s_numKernelCalls = 0;

	std::cout << "---------------------------------" << std::endl;

	t.restart();
	generatePrimary(s_camera, s_raySet);
	std::cout << "generate: " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	t.restart();
	doTraceRays(s_triBox, s_triSet, s_raySet);
	std::cout << "trace:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;

	std::cout << "  " << "split:        " << std::setw(9) << (int)s_splitTime << " ms" << std::endl;
	std::cout << "  " << "intersect:    " << std::setw(9) << (int)s_intersectTime << " ms" << std::endl;

	std::cout << "    " << "triSend:    " << std::setw(9) << s_triSendTime << " ms" << std::endl;
	std::cout << "    " << "raySend:    " << std::setw(9) << s_raySendTime << " ms" << std::endl;
	std::cout << "    " << "tmaxCopy:    " << std::setw(9) << s_tmaxCopyTime << " ms" << std::endl;
	std::cout << "    " << "tmaxSend:    " << std::setw(9) << s_tmaxSendTime << " ms" << std::endl;
	std::cout << "    " << "kernel:    " << std::setw(9) << s_kernelTime << " ms (count: " << s_numKernelCalls << ", avg: " << s_kernelTime / s_numKernelCalls << " ms)" << std::endl;
	std::cout << "    " << "tmaxRead:    " << std::setw(9) << s_tmaxReadTime << " ms" << std::endl;
	std::cout << "    " << "hitRead:    " << std::setw(9) << s_hitReadTime << " ms" << std::endl;
	std::cout << "    " << "hitUpdate:    " << std::setw(9) << s_hitUpdateTime << " ms" << std::endl;

	std::cout << "  " << "partitionTri: " << std::setw(9) << (int)s_partitionTriTime << " ms" << std::endl;
	std::cout << "  " << "partitionRay: " << std::setw(9) << (int)s_partitionRayTime << " ms" << std::endl;

	t.restart();
	shadePixels(s_triSet, s_raySet, pixels);
	std::cout << "shade:    " << std::setw(15) << (int)t.msec() << " ms" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------

void loadScene1tri(TriSet& triSet)
{
	TriV tri;
	tri.v0 = vec3(-1,-1,0);
	tri.v1 = vec3(1,-1,0);
	tri.v2 = vec3(0,1,0);

	triSet.tris = new TriV[1];
	triSet.tris[0] = tri;

	TriN norm;
	norm.n0 = (tri.v1 - tri.v0).cross(tri.v2-tri.v0).normalized();
	norm.n1 = norm.n0;
	norm.n2 = norm.n0;
	triSet.norms = new TriN[1];
	triSet.norms[0] = norm;

	triSet.ids = new int[1];
	triSet.ids[0] = 0;

	triSet.count = 1;
}

float randf(float min, float max)
{
	return min + (float)rand()/(float)RAND_MAX * (max - min);
}

void loadSceneRand(TriSet& triSet)
{
	triSet.count = 100000;
	triSet.tris = new TriV[triSet.count];
	triSet.norms = new TriN[triSet.count];
	int dst = 0;

	srand(122);

	for(int i = 0; i < triSet.count; ++i)
	{
		vec3 c(randf(-5.0f, 5.0f), randf(-5.0f, 5.0f), randf(-5.0f, 5.0f));
		TriV tri;
		tri.v0 = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		tri.v1 = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		tri.v2 = c + vec3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
		triSet.tris[dst] = tri;
		TriN norm;
		norm.n0 = (tri.v1 - tri.v0).cross(tri.v2-tri.v0).normalized();
		norm.n1 = norm.n0;
		norm.n2 = norm.n0;
		triSet.norms[dst] = norm;
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

void loadSceneBunny(TriSet& triSet)
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

	triSet.count = elements.size()/3;
	triSet.tris = new TriV[triSet.count];
	triSet.norms = new TriN[triSet.count];

	std::vector<vec3> normals(vertices.size());

	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		int e0 = elements[e+0];
		int e1 = elements[e+1];
		int e2 = elements[e+2];

		vec3 v0 = vertices[e0];
		vec3 v1 = vertices[e1];
		vec3 v2 = vertices[e2];

		vec3 n = (v1-v0).cross(v2-v0);

		normals[e0] += n;
		normals[e1] += n;
		normals[e2] += n;
	}

	int i = 0;
	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		int e0 = elements[e+0];
		int e1 = elements[e+1];
		int e2 = elements[e+2];

		TriV t;
		t.v0 = vertices[e0];
		t.v1 = vertices[e1];
		t.v2 = vertices[e2];
		triSet.tris[i] = t;

		TriN n;
		n.n0 = normals[e0].normalized();
		n.n1 = normals[e1].normalized();
		n.n2 = normals[e2].normalized();
		triSet.norms[i] = n;

		++i;
	}
}

void commonSetup(TriSet& triSet, Box& triBox)
{
	triSet.ids = new int[triSet.count];
	for(int i = 0; i < triSet.count; ++i)
	{
		triSet.ids[i] = i;
		triBox.expand(triSet.tris[i]);
	}

	cudaMalloc(&d_triIDs, triSet.count*sizeof(int));
	cudaMemcpy(d_triIDs, triSet.ids, triSet.count*sizeof(int), cudaMemcpyDefault);

	cudaMalloc(&d_tris, triSet.count*sizeof(TriV));
	cudaMemcpy(d_tris, triSet.tris, triSet.count*sizeof(TriV), cudaMemcpyDefault);

	vec3 center = (triBox.min + triBox.max) * 0.5f;
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

//	loadScene1tri(s_triSet);
//	loadSceneRand(s_triSet);
	loadSceneBunny(s_triSet);

	commonSetup(s_triSet, s_triBox);

	rtvExec();

	return 0;
}
