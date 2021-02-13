#include <stdio.h>
#include <stdlib.h>
#include <helper_math.h>
#include <algorithm>

const int HIT_EPSILON = 1e-6f;

struct triangle
{
	float3 v0;
	float3 v1;
	float3 v2;
};

__global__ void kernel(triangle* tris, float3 rsrc, float3* rdirs, int* hits, int numTiles)
{
	extern __shared__ triangle sh_tris[];

	float bestDist = 3.40282347E+38F;
	int bestTri = -1;

	int gtid = blockIdx.x*blockDim.x+threadIdx.x;
	float3 rdir = rdirs[gtid];

	for(int tile = 0; tile < numTiles; ++tile)
	{
		// load triangles to shared memory
		sh_tris[threadIdx.x] = tris[tile*blockDim.x+threadIdx.x];

		__syncthreads();

		// compute intersections
		for(unsigned int tid = 0; tid < blockDim.x; ++tid)
		{
			triangle tri = sh_tris[tid];

			/* find vectors for two edges sharing vert0 */
			const float3 edge1 = tri.v1 - tri.v0;
			const float3 edge2 = tri.v2 - tri.v0;

			/* begin calculating determinant - also used to calculate U parameter */
			const float3 pvec = cross(rdir, edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			float det = dot(edge1, pvec);

			if(det > -HIT_EPSILON && det < HIT_EPSILON)
				continue;

			const float inv_det = 1.0f / det;

			/* calculate distance from vert0 to ray origin */
			const float3 tvec = rsrc - tri.v0;

			/* calculate U parameter and test bounds */
			const float u = dot(tvec, pvec) * inv_det;
			if(u < 0.0f || u > 1.0f)
				continue;

			/* prepare to test V parameter */
			const float3 qvec = cross(tvec, edge1);

			/* calculate V parameter and test bounds */
			const float v = dot(rdir, qvec) * inv_det;
			if(v < 0.0f || u + v > 1.0f)
				continue;

			/* calculate t, ray hits triangle */
			const float f = dot(edge2, qvec) * inv_det;

			if((f >= bestDist) || (f < -HIT_EPSILON))
				continue;

			// Have a valid hit point here. Store it.
			bestDist = f;
			bestTri = tid;
		}

		__syncthreads();
	}

	// write final result
	hits[gtid] = bestTri;
}

float randf(float min, float max)
{
	return (float)rand()/(float)RAND_MAX * (max - min);
}

int main()
{
	const int ntri = 4096;
	const int nray = 4096;
    const int repeat = 5;

    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    srand(122);

    triangle* h_tris;
    cudaMallocHost(&h_tris, ntri*sizeof(triangle));
    triangle* d_tris;
    cudaMalloc(&d_tris, ntri*sizeof(triangle));
    for(int i = 0; i < ntri; ++i)
    {
    	float3 c = make_float3(randf(-5, 5),randf(-5, 5), randf(-5, 5));
    	triangle t;
    	t.v0 = c + make_float3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
    	t.v1 = c + make_float3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
    	t.v2 = c + make_float3(randf(0.1f, 0.5f), randf(0.1f, 0.5f), randf(0.1f, 0.5f));
    	h_tris[i] = t;
    }
    cudaMemcpy(d_tris, h_tris, ntri*sizeof(triangle), cudaMemcpyDefault);

    float3 src = make_float3(0,0,10);

    float3* h_dirs;
    cudaMallocHost(&h_dirs, nray*sizeof(float3));
    float3* d_dirs;
    cudaMalloc(&d_dirs, nray*sizeof(float3));
    for(int i = 0; i < nray; ++i)
    {
    	float3 dir = make_float3(randf(-10, 10), randf(-10, 10), 0.0);
    	h_dirs[i] = dir - src;
    }
    cudaMemcpy(d_dirs, h_dirs, nray*sizeof(float3), cudaMemcpyDefault);

    int* h_hits;
    cudaMallocHost(&h_hits, nray*sizeof(int));
    int* d_hits;
    cudaMalloc(&d_hits, nray*sizeof(int));
    std::fill_n(h_hits, nray, -1);
    cudaMemcpy(d_hits, h_hits, nray*sizeof(int), cudaMemcpyDefault);

    int blockSize = 1024; // 1024
    int numBlocks = nray / blockSize;
    int numTiles = ntri / blockSize;
    int sharedMemSize = blockSize * sizeof(triangle);

    for(int i=0; i<repeat; i++)
    {
        cudaEventRecord(start, 0);
        kernel<<<numBlocks, blockSize, sharedMemSize>>>(d_tris, src, d_dirs, d_hits, numTiles);
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cumulative_time = cumulative_time + time;
    }

    printf("Kernel time:  %3.5f ms \n", cumulative_time / repeat);

    cudaMemcpy(h_hits, d_hits, nray*sizeof(int), cudaMemcpyDefault);
    int nhits = 0;
    for(int i = 0; i < nray; ++i)
    {
    	if(h_hits[i] >= 0)
    	{
    		++nhits;
    	}
    }

    printf("nhits: %d\n", nhits);

    return 0;
}
