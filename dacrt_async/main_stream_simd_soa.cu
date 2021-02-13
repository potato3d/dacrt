#include "types.h"
#include "dacrt.h"
#include "rtv.h"
#include "rply.h"
#include "timer.h"
#include "my_malloc.h"
#include <cfloat>
#include <vector>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <fstream>

#define GENERATE_BLOCKSIZE 16
#define SHADE_BLOCKSIZE 256

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

static TriSet s_tris;
static RaySet s_rays;

static float* d_rox = 0;
static float* d_roy = 0;
static float* d_roz = 0;
static float* d_rdx = 0;
static float* d_rdy = 0;
static float* d_rdz = 0;
static int* d_id = 0;
static float* d_tmax = 0;

static int* d_hit = 0;

struct cudaGraphicsResource* s_pixelsResource = 0;

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

struct Canvas
{
	int w;
	int h;
};

struct Camera
{
	float3 position;
	float3 lowerLeftDir;
	float3 du;
	float3 dv;
	int nu;
	int nv;
};

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

static Canvas g_canvas;
static Camera g_camera;

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

__global__ void kernelGeneratePrimary(float3 camPos, float3 lowerLeftDir, float3 du, float3 dv, int numRaysX, int numRaysY,
									  float* rox, float* roy, float* roz, float* rdx, float* rdy, float* rdz, int* id, float* tmax, int* hit)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(tx >= numRaysX || ty >= numRaysY)
	{
		return;
	}

	const int idx = ty * numRaysX + tx;

	rox[idx] = camPos.x;
	roy[idx] = camPos.y;
	roz[idx] = camPos.z;

	float3 dir = lowerLeftDir + du * tx + dv * ty + du * numRaysX * ty;

	rdx[idx] = dir.x;
	rdy[idx] = dir.y;
	rdz[idx] = dir.z;

	tmax[idx] = FLT_MAX;
	id[idx] = idx;

	hit[idx] = -1;
}

void generatePrimary()
{
	dim3 numBlocks;
	numBlocks.x = ceilf(((float)g_camera.nu/(float)GENERATE_BLOCKSIZE));
	numBlocks.y = ceilf(((float)g_camera.nv/(float)GENERATE_BLOCKSIZE));
	dim3 blockSize;
	blockSize.x = GENERATE_BLOCKSIZE;
	blockSize.y = GENERATE_BLOCKSIZE;
	kernelGeneratePrimary<<<numBlocks, blockSize>>>(g_camera.position, g_camera.lowerLeftDir, g_camera.du, g_camera.dv, g_camera.nu, g_camera.nv,
													d_rox, d_roy, d_roz, d_rdx, d_rdy, d_rdz, d_id, d_tmax, d_hit);

	float* h_rdx = new float[s_rays.count];
	float* h_rdy = new float[s_rays.count];
	float* h_rdz = new float[s_rays.count];
	cudaMemcpy(h_rdx, d_rdx, s_rays.count*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(h_rdy, d_rdy, s_rays.count*sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(h_rdy, d_rdy, s_rays.count*sizeof(float), cudaMemcpyDefault);

	std::ofstream f("out.txt");
	for(int i = 0; i < s_rays.count; ++i)
	{
		f << h_rdx[i] << ", " << h_rdy[i] << ", " << h_rdz[i] << std::endl;
	}
	f.close();
	exit(1);
}

__global__ void kernelShadePixels(unsigned char* pixels, int numRays, float* rox, float* roy, float* roz, float* rdx, float* rdy, float* rdz)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= numRays)
	{
		return;
	}

	float3 dir = (make_float3(rdx[idx], rdy[idx], rdz[idx]));

	pixels[idx*3+0] = dir.x * 255;
	pixels[idx*3+1] = dir.y * 255;
	pixels[idx*3+2] = 0;
}

void shadePixels(unsigned int pixelBufferID)
{
	if(!s_pixelsResource)
	{
		cudaGraphicsGLRegisterBuffer(&s_pixelsResource, pixelBufferID, cudaGraphicsMapFlagsWriteDiscard);
	}

	unsigned char* pixels = 0;
	cudaGraphicsMapResources(1, &s_pixelsResource);
	size_t num_bytes = 0;
	cudaGraphicsResourceGetMappedPointer((void**)&pixels, &num_bytes, s_pixelsResource);

	const int numBlocks = ceilf(((float)(s_rays.count)/(float)SHADE_BLOCKSIZE));
	kernelShadePixels<<<numBlocks, SHADE_BLOCKSIZE, 0>>>(pixels, g_camera.nu*g_camera.nv, d_rox, d_roy, d_roz, d_rdx, d_rdy, d_rdz);

	cudaGraphicsUnmapResources(1, &s_pixelsResource);
}

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

int round4(int a)
{
   return (a + 3) & ~3;
}

void reshapeCB(int w, int h)
{
	g_canvas.w = w;
	g_canvas.h = h;

	s_rays.count = round4(g_canvas.w * g_canvas.h);

	raySetInitialize(s_rays);

	cudaFree(d_rox);
	cudaFree(d_roy);
	cudaFree(d_roz);

	cudaFree(d_rdx);
	cudaFree(d_rdy);
	cudaFree(d_rdz);

	cudaFree(d_tmax);

	cudaFree(d_id);

	cudaFree(d_hit);

	cudaMalloc(&d_rox, s_rays.count*sizeof(float));
	cudaMalloc(&d_roy, s_rays.count*sizeof(float));
	cudaMalloc(&d_roz, s_rays.count*sizeof(float));

	cudaMalloc(&d_rdx, s_rays.count*sizeof(float));
	cudaMalloc(&d_rdy, s_rays.count*sizeof(float));
	cudaMalloc(&d_rdz, s_rays.count*sizeof(float));

	cudaMalloc(&d_tmax, s_rays.count*sizeof(float));
	cudaMalloc(&d_id, s_rays.count*sizeof(int));

	cudaMalloc(&d_hit, s_rays.count*sizeof(int));
}

void cameraCB(float* peye, float* pcenter, float* pup)
{
	const float3 eye = make_float3(peye[0], peye[1], peye[2]);
	const float3 center = make_float3(pcenter[0], pcenter[1], pcenter[2]);
	const float3 up = make_float3(pup[0], pup[1], pup[2]);

	// store position
	g_camera.position = eye;

	// pre-computations
	float invHeight = 1.0f / g_canvas.h;
	float invWidth  = 1.0f / g_canvas.w;

	// compute camera basis
	float3 axisW = normalize(eye - center);
	float3 axisV = normalize(up);
	float3 axisU = cross(axisV, axisW);

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

void renderCB(unsigned int pixelBufferID)
{
	generatePrimary();
	trace(s_tris.bt, s_tris.bc, 0, s_tris.count, s_rays.count);
	shadePixels(pixelBufferID);
}

// ----------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------

static std::vector<float3> vertices;
static std::vector<int> elements;

float& at(float3& v, int i)
{
	switch(i)
	{
	case 0: return v.x;
	case 1: return v.y;
	default: return v.z;
	}
}

int vertex_cb(p_ply_argument argument)
{
	long id;
	ply_get_argument_user_data(argument, NULL, &id);
	if(id == 0) vertices.resize(vertices.size()+1);
	at(vertices.back(), id) = ply_get_argument_value(argument) * 50;
	return 1;
}

int face_cb(p_ply_argument argument)
{
	long length, value_index;
	ply_get_argument_property(argument, NULL, &length, &value_index);
	if(value_index >= 0 && value_index <= 2) elements.push_back(ply_get_argument_value(argument));
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

	int count = elements.size()/3;
	s_tris.count = round4(count);

	s_tris.v0x = amalloc<float>(s_tris.count);
	s_tris.v0y = amalloc<float>(s_tris.count);
	s_tris.v0z = amalloc<float>(s_tris.count);
	s_tris.v1x = amalloc<float>(s_tris.count);
	s_tris.v1y = amalloc<float>(s_tris.count);
	s_tris.v1z = amalloc<float>(s_tris.count);
	s_tris.v2x = amalloc<float>(s_tris.count);
	s_tris.v2y = amalloc<float>(s_tris.count);
	s_tris.v2z = amalloc<float>(s_tris.count);

	s_tris.n0x = amalloc<float>(s_tris.count);
	s_tris.n0y = amalloc<float>(s_tris.count);
	s_tris.n0z = amalloc<float>(s_tris.count);
	s_tris.n1x = amalloc<float>(s_tris.count);
	s_tris.n1y = amalloc<float>(s_tris.count);
	s_tris.n1z = amalloc<float>(s_tris.count);
	s_tris.n2x = amalloc<float>(s_tris.count);
	s_tris.n2y = amalloc<float>(s_tris.count);
	s_tris.n2z = amalloc<float>(s_tris.count);

	std::vector<float3> normals(vertices.size());

	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		const int e0 = elements[e+0];
		const int e1 = elements[e+1];
		const int e2 = elements[e+2];

		float3 v0 = vertices[e0];
		float3 v1 = vertices[e1];
		float3 v2 = vertices[e2];

		float3 n = cross(v1-v0, v2-v0);

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

		float3 v0 = vertices[e0];
		float3 v1 = vertices[e1];
		float3 v2 = vertices[e2];

		s_tris.v0x[dst] = v0.x;
		s_tris.v0y[dst] = v0.y;
		s_tris.v0z[dst] = v0.z;
		s_tris.v1x[dst] = v1.x;
		s_tris.v1y[dst] = v1.y;
		s_tris.v1z[dst] = v1.z;
		s_tris.v2x[dst] = v2.x;
		s_tris.v2y[dst] = v2.y;
		s_tris.v2z[dst] = v2.z;

		float3 n0 = normalize(normals[e0]);
		float3 n1 = normalize(normals[e1]);
		float3 n2 = normalize(normals[e2]);

		s_tris.n0x[dst] = n0.x;
		s_tris.n0y[dst] = n0.y;
		s_tris.n0z[dst] = n0.z;
		s_tris.n1x[dst] = n1.x;
		s_tris.n1y[dst] = n1.y;
		s_tris.n1z[dst] = n1.z;
		s_tris.n2x[dst] = n2.x;
		s_tris.n2y[dst] = n2.y;
		s_tris.n2z[dst] = n2.z;

		++dst;
	}

	std::fill_n(s_tris.v0x + s_tris.count, s_tris.count - count, s_tris.v0x[s_tris.count-1]);
	std::fill_n(s_tris.v0y + s_tris.count, s_tris.count - count, s_tris.v0y[s_tris.count-1]);
	std::fill_n(s_tris.v0z + s_tris.count, s_tris.count - count, s_tris.v0z[s_tris.count-1]);
	std::fill_n(s_tris.v1x + s_tris.count, s_tris.count - count, s_tris.v1x[s_tris.count-1]);
	std::fill_n(s_tris.v1y + s_tris.count, s_tris.count - count, s_tris.v1y[s_tris.count-1]);
	std::fill_n(s_tris.v1z + s_tris.count, s_tris.count - count, s_tris.v1z[s_tris.count-1]);
	std::fill_n(s_tris.v2x + s_tris.count, s_tris.count - count, s_tris.v2x[s_tris.count-1]);
	std::fill_n(s_tris.v2y + s_tris.count, s_tris.count - count, s_tris.v2y[s_tris.count-1]);
	std::fill_n(s_tris.v2z + s_tris.count, s_tris.count - count, s_tris.v2z[s_tris.count-1]);
}

void endLoadScene()
{
	triSetInitialize(s_tris);

	float3 btmin = make_float3(s_tris.bt.minx[0], s_tris.bt.miny[0], s_tris.bt.minz[0]);
	float3 btmax = make_float3(s_tris.bt.maxx[0], s_tris.bt.maxy[0], s_tris.bt.maxz[0]);

	float3 center = (btmin + btmax) * 0.5f;
	float3 eye = center + make_float3(0,0,10);
	float3 up = make_float3(0,1,0);
	rtvSetCamera(&eye.x, &center.x, &up.x);
}

int main()
{
	rtvInit(1024, 1024);
	rtvSetReshapeCallback(reshapeCB);
	rtvSetCameraCallback(cameraCB);
	rtvSetBufferRenderCallback(renderCB);

	loadSceneBunny();

	endLoadScene();

	rtvExec();

	return 0;
}
