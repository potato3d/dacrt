#include <bl/bl.h>
#include "rtv.h"
#include "rply.h"

// ---------------------------------------------------------------------------------------------------------------------

struct Triangle
{
	vec3 v0 = vec3::ZERO;
	vec3 v1 = vec3::ZERO;
	vec3 v2 = vec3::ZERO;
};

struct AABB
{
	vec3 min = vec3(math::limit_posf());
	vec3 max = vec3(math::limit_negf());
};

struct Camera
{
//	vec3 position = {0,0,15};
	vec3 position = {-1.18093, 4.68028, 8.13683}; // bunny
	quat orientation = quat(0, {0,1,0});
	float fovY = math::to_radians(60.0f);
	int imgWidth = 800;
	int imgHeight = 600;
};

struct View
{
	vec3 position = vec3::ZERO;
	vec3 lowerLeftDir = vec3::ZERO;
	vec3 du = vec3::ZERO;
	vec3 dv = vec3::ZERO;
	int nu = 0;
	int nv = 0;
};

struct RaySet
{
	vec3  src = vec3::ZERO;
	vec3* dirs = nullptr;
	vec3* invDirs = nullptr;
	int* ids = nullptr;

	float* hitDists = nullptr;
	int* hitTris = nullptr;
	vec3* hitCoords = nullptr;

	int count = 0;
	int terminatedPivot = 0;
};

struct TriSet
{
	Triangle* tris = nullptr;
	int* ids = nullptr;
	int count = 0;
};

// ---------------------------------------------------------------------------------------------------------------------

Camera s_camera;
View s_view;
AABB s_bound;
RaySet s_rays;
TriSet s_tris;

const float HIT_EPSILON = 1e-6f;
const int MIN_TRIS = 8;
const int MIN_RAYS = 8;

// ---------------------------------------------------------------------------------------------------------------------

void updateCamera(Camera& camera)
{
	if(rtvResetCamera())
	{
		s_camera.position = Camera().position;
		s_camera.orientation = Camera().orientation;
	}
	else
	{
		float angle;
		float axisx, axisy, axisz;
		rtvRotateCamera(&angle, &axisx, &axisy, &axisz);
		camera.orientation = camera.orientation.mul(quat(angle, {axisx, axisy, axisz}));

		vec3 dt;
		rtvMoveCamera(&dt.x, &dt.y, &dt.z);
		camera.position += camera.orientation.mul(dt * 2.0f);
	}
}

void updateView(const Camera& camera,
				View& view)
{
	// position ------------------------------------------------------------------------

	view.position = camera.position;

	// lower left direction ------------------------------------------------------------

	// pre-computations
	float invHeight = 1.0f / camera.imgHeight;
	float invWidth  = 1.0f / camera.imgWidth;

	// compute camera basis
	vec3 axisW = camera.orientation.mul(vec3::UNIT_Z);
	vec3 axisV = camera.orientation.mul(vec3::UNIT_Y);
	vec3 axisU = camera.orientation.mul(vec3::UNIT_X);

	// compute half scale factors for each basis vector
	float sw = camera.imgWidth*0.01f; // try to keep directions around zero in floating-point value
	float sv = sw * math::tan(camera.fovY*0.5f);
	float su = sv * camera.imgWidth * invHeight;

	// scale each vector
	axisW *= sw;
	axisV *= sv;
	axisU *= su;

	// store final direction
	view.lowerLeftDir = - axisU - axisV - axisW;

	// deltas along U and V directions --------------------------------------------------

	// compute full scales
	axisV *= 2.0f;
	axisU *= 2.0f;

	// interpolation deltas
	view.dv = axisV * invHeight - axisU; // also goes back to start of u-axis
	view.du = axisU * invWidth;

	// image size -----------------------------------------------------------------------

	// number of pixels in U and V directions
	view.nu = camera.imgWidth;
	view.nv = camera.imgHeight;
}

void generatePrimary(const View& view,
					 RaySet& rays)
{
	rays.src = view.position;
	rays.terminatedPivot = 0;

	vec3 dir = view.lowerLeftDir;
	int i = 0;

	for(int v = 0; v < view.nv; ++v)
	{
		for(int u = 0; u < view.nu; ++u)
		{
			rays.dirs[i] = dir;
			rays.invDirs[i] = dir.reciprocal();
			rays.ids[i] = i;
			rays.hitDists[i] = math::limit_posf();
			++i;

			dir += view.du;
		}
		dir += view.dv;
	}
}

void expandBound(AABB& b, const vec3& v)
{
	b.min.x = math::min(b.min.x, v.x);
	b.min.y = math::min(b.min.y, v.y);
	b.min.z = math::min(b.min.z, v.z);

	b.max.x = math::max(b.max.x, v.x);
	b.max.y = math::max(b.max.y, v.y);
	b.max.z = math::max(b.max.z, v.z);
}

void expandBound(AABB& b, const Triangle& t)
{
	expandBound(b, t.v0);
	expandBound(b, t.v1);
	expandBound(b, t.v2);
}

void splitBound(const AABB& bound, const TriSet& tris, const RaySet& rays,
				AABB& bnear, AABB& bfar)
{
	float dx = bound.max.x - bound.min.x;
	float dy = bound.max.y - bound.min.y;
	float dz = bound.max.z - bound.min.z;
	int axis = (dx > dy && dx > dz)? 0 : (dy > dz)? 1 : 2;

	float pos = (bound.min[axis] + bound.max[axis]) * 0.5f;

	AABB left = bound;
	AABB right = bound;
	left.max[axis] = pos;
	right.min[axis] = pos;

	bool lnear = rays.src[axis] < pos;
	bnear = lnear? left : right;
	bfar = lnear? right : left;
}

bool hitTriBound(const AABB& bound, const TriSet& tris, int id)
{
	AABB tbound;
	expandBound(tbound, tris.tris[id]);
	if (bound.max.x < tbound.min.x) return false;
	if (bound.min.x > tbound.max.x) return false;
	if (bound.max.y < tbound.min.y) return false;
	if (bound.min.y > tbound.max.y) return false;
	if (bound.max.z < tbound.min.z) return false;
	if (bound.min.z > tbound.max.z) return false;
	return true;
}

// src: http://tavianator.com/cgit/dimension.git/tree/libdimension/bvh.c#n191
//		http://tavianator.com/2011/05/fast-branchless-raybounding-box-intersections/
// alternative (slower?): http://www.cs.utah.edu/~awilliam/box/
bool hitRayBound(const AABB& bound, const RaySet& rays, int id)
{
	const vec3& invDir = rays.invDirs[id];

	const float tx1 = (bound.min.x - rays.src.x) * invDir.x;
	const float tx2 = (bound.max.x - rays.src.x) * invDir.x;

	float tmin = math::min(tx1, tx2);
	float tmax = math::max(tx1, tx2);

	const float ty1 = (bound.min.y - rays.src.y) * invDir.y;
	const float ty2 = (bound.max.y - rays.src.y) * invDir.y;

	tmin = math::max(tmin, math::min(ty1, ty2));
	tmax = math::min(tmax, math::max(ty1, ty2));

	const float tz1 = (bound.min.z - rays.src.z) * invDir.z;
	const float tz2 = (bound.max.z - rays.src.z) * invDir.z;

	tmin = math::max(tmin, math::min(tz1, tz2));
	tmax = math::min(tmax, math::max(tz1, tz2));

	return tmax >= tmin;
}

int partitionTris(const AABB& bound,
				   TriSet& tris, int triPivot)
{
	int newPivot = 0;
	for(int i = newPivot; i < triPivot; ++i)
	{
		if(hitTriBound(bound, tris, tris.ids[i]))
		{
			swap(tris.ids[i], tris.ids[newPivot]);
			++newPivot;
		}
	}
	return newPivot;
}

int partitionRays(const AABB& bound,
				   RaySet& rays, int rayPivot)
{
	int newPivot = rays.terminatedPivot;
	for(int i = newPivot; i < rayPivot; ++i)
	{
		if(hitRayBound(bound, rays, rays.ids[i]))
		{
			swap(rays.ids[i], rays.ids[newPivot]);
			++newPivot;
		}
	}
	return newPivot;
}

// Moller Trumbore
void hitRayTri(const TriSet& tris, int tid,
			   RaySet& rays, int rid)
{
	const vec3& v0 = tris.tris[tid].v0;
	const vec3& v1 = tris.tris[tid].v1;
	const vec3& v2 = tris.tris[tid].v2;

	const vec3& rdir = rays.dirs[rid];

	/* find vectors for two edges sharing vert0 */
	const vec3 edge1 = v1 - v0;
	const vec3 edge2 = v2 - v0;

	/* begin calculating determinant - also used to calculate U parameter */
	const vec3 pvec = rdir.cross(edge2);

	/* if determinant is near zero, ray lies in plane of triangle */
	float det = edge1.dot(pvec);

	if(det > -HIT_EPSILON && det < HIT_EPSILON)
		return;

	const float inv_det = 1.0f / det;

	/* calculate distance from vert0 to ray origin */
	const vec3 tvec = rays.src - v0;

	/* calculate U parameter and test bounds */
	const float u = tvec.dot(pvec) * inv_det;
	if(u < 0.0f || u > 1.0f)
		return;

	/* prepare to test V parameter */
	const vec3 qvec = tvec.cross(edge1);

	/* calculate V parameter and test bounds */
	const float v = rdir.dot(qvec) * inv_det;
	if(v < 0.0f || u + v > 1.0f)
		return;

	/* calculate t, ray hits triangle */
	const float f = edge2.dot(qvec) * inv_det;

	if((f >= rays.hitDists[rid]) || (f < -HIT_EPSILON))
		return;

	// Have a valid hit point here. Store it.
	rays.hitDists[rid] = f;
	rays.hitTris[rid] = tid;
	rays.hitCoords[rid] = vec3(1.0f - (u + v), u, v);
}

void intersect(const TriSet& tris, int triPivot,
			   RaySet& rays, int rayPivot)
{
	for(int r = rays.terminatedPivot; r < rayPivot; ++r)
	{
		const int rid = rays.ids[r];
		for(int t = 0; t < triPivot; ++t)
		{
			hitRayTri(tris, tris.ids[t], rays, rid);
		}
		if(rays.hitDists[rid] < math::limit_posf())
		{
			swap(rays.ids[r], rays.ids[rays.terminatedPivot]);
			++rays.terminatedPivot;
		}
	}
}

timer tt;
double dtt = 0.0;

void traceMora(const AABB& bound,
			   TriSet& tris, int triPivot, RaySet& rays, int rayPivot)
{
	if((rayPivot - rays.terminatedPivot) <= MIN_RAYS || triPivot <= MIN_TRIS)
	{
//		tt.restart();
		intersect(tris, triPivot, rays, rayPivot);
//		dtt += tt.msec();
		return;
	}

	AABB bnear;
	AABB bfar;
	splitBound(bound, tris, rays, bnear, bfar);

	int newTriPivot = partitionTris(bnear, tris, triPivot);
	int newRayPivot = partitionRays(bnear, rays, rayPivot);
	traceMora(bnear, tris, newTriPivot, rays, newRayPivot);

	newTriPivot = partitionTris(bfar, tris, triPivot);
	newRayPivot = partitionRays(bfar, rays, rayPivot);
	traceMora(bfar, tris, newTriPivot, rays, newRayPivot);
}

void shadePixels(const TriSet& tris, const RaySet& rays,
				 unsigned char* pixels)
{
	for(int i = 0; i < rays.count; ++i)
	{
		unsigned char c = 255;

		if(rays.hitDists[i] < math::limit_posf())
		{
			const Triangle& t = tris.tris[rays.hitTris[i]];
			c = 255 * (t.v1-t.v0).cross(t.v2-t.v0).normalized().dot(-rays.dirs[i].normalized());
		}

		pixels[i*3+0] = c;
		pixels[i*3+1] = c;
		pixels[i*3+2] = c;
	}
}

// ---------------------------------------------------------------------------------------------------------------------

void reshape(int w, int h)
{
	s_camera.imgWidth = w;
	s_camera.imgHeight = h;

	s_rays.count = w*h;

	delete[] s_rays.dirs;
	s_rays.dirs = new vec3[s_rays.count];

	delete[] s_rays.invDirs;
	s_rays.invDirs = new vec3[s_rays.count];

	delete[] s_rays.ids;
	s_rays.ids = new int[s_rays.count];
	iota(s_rays.ids, s_rays.ids + s_rays.count, 0);

	delete[] s_rays.hitTris;
	s_rays.hitTris = new int[s_rays.count];

	delete[] s_rays.hitDists;
	s_rays.hitDists = new float[s_rays.count];

	delete[] s_rays.hitCoords;
	s_rays.hitCoords = new vec3[s_rays.count];
}

void render(unsigned char* pixels)
{
//	dtt = 0.0;

	updateCamera(s_camera);
	updateView(s_camera, s_view);
	generatePrimary(s_view, s_rays);
	traceMora(s_bound, s_tris, s_tris.count, s_rays, s_rays.count);
	shadePixels(s_tris, s_rays, pixels);

//	io::print(dtt);
}



vector<vec3> vertices;
vector<int> elements;

static int vertex_cb(p_ply_argument argument) {
	long id;
	ply_get_argument_user_data(argument, NULL, &id);
	if(id == 0)
	{
		vertices.resize(vertices.size()+1);
	}
	vertices.back()[id] = ply_get_argument_value(argument) * 50;
	return 1;
}

static int face_cb(p_ply_argument argument) {
	long length, value_index;
		ply_get_argument_property(argument, NULL, &length, &value_index);
		switch (value_index) {
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

void bench_ray_tri_intersection()
{
	struct triangle
	{
		vec3 v0;
		vec3 v1;
		vec3 v2;
	};

	const int nray = 1024;
	const int ntri = 1024;

	auto h_tris = new triangle[ntri];

	auto randc = make_random(-5.0f, 5.0f);
	auto randv = make_random(0.1f, 0.5f);

	for(int i = 0; i < ntri; ++i)
	{
		vec3 c = {randc(),randc(), randc()};
		triangle t;
		t.v0 = c + vec3{randv(), randv(), randv()};
		t.v1 = c + vec3{randv(), randv(), randv()};
		t.v2 = c + vec3{randv(), randv(), randv()};
		h_tris[i] = t;
	}

	vec3 rsrc(0,0,10);

	vec3* h_dirs = new vec3[nray];

	auto randd = make_random(-10.0f, 10.0f);

	for(int i = 0; i < nray; ++i)
	{
		vec3 dir = vec3(randd(), randd(), 0.0);
		h_dirs[i] = dir - rsrc;
	}

	int* h_hits = new int[nray];

	timer rt;

	for(int r = 0; r < nray; ++r)
	{
		float bestDist = math::limit_posf();
		int bestTri = -1;
		const auto& rdir = h_dirs[r];
		for(int t = 0; t < ntri; ++t)
		{
			const triangle& tri = h_tris[t];

			/* find vectors for two edges sharing vert0 */
			const vec3 edge1 = tri.v1 - tri.v0;
			const vec3 edge2 = tri.v2 - tri.v0;

			/* begin calculating determinant - also used to calculate U parameter */
			const vec3 pvec = rdir.cross(edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			float det = edge1.dot(pvec);

			if(det > -HIT_EPSILON && det < HIT_EPSILON)
				continue;

			const float inv_det = 1.0f / det;

			/* calculate distance from vert0 to ray origin */
			const vec3 tvec = rsrc - tri.v0;

			/* calculate U parameter and test bounds */
			const float u = tvec.dot(pvec) * inv_det;
			if(u < 0.0f || u > 1.0f)
				continue;

			/* prepare to test V parameter */
			const vec3 qvec = tvec.cross(edge1);

			/* calculate V parameter and test bounds */
			const float v = rdir.dot(qvec) * inv_det;
			if(v < 0.0f || u + v > 1.0f)
				continue;

			/* calculate t, ray hits triangle */
			const float f = edge2.dot(qvec) * inv_det;

			if((f >= bestDist) || (f < -HIT_EPSILON))
				continue;

			// Have a valid hit point here. Store it.
			bestDist = f;
			bestTri = t;
		}
		h_hits[r] = bestTri;
	}

	io::print(rt.msec(), "ms");

	int nhits = 0;
	for(int i = 0; i < nray; ++i)
	{
		if(h_hits[i] >= 0)
			++nhits;
	}

	io::print("hits:", nhits);
}

int main()
{
	long nvertices, ntriangles;
	p_ply ply = ply_open("/home/potato/Downloads/bunny.ply", NULL, 0, NULL);
	if (!ply) return 1;
	if (!ply_read_header(ply)) return 1;
	nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
	ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 1);
	ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 2);
	ntriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
	if (!ply_read(ply)) return 1;
	ply_close(ply);

	s_tris.count = elements.size()/3;
	s_tris.tris = new Triangle[s_tris.count];

	int i = 0;
	for(unsigned int e = 0; e < elements.size(); e+=3)
	{
		Triangle t;
		t.v0 = vertices[elements[e+0]];
		t.v1 = vertices[elements[e+1]];
		t.v2 = vertices[elements[e+2]];
		s_tris.tris[i++] = t;
	}

// ---------------------------------------------------------------------------

//	s_tris.count = 2;

//	s_tris.tris = new Triangle[s_tris.count];
//	s_tris.tris[0].v0 = {-1.5,0,0};
//	s_tris.tris[0].v1 = {-0.5,0,0};
//	s_tris.tris[0].v2 = {-1,1,0};

//	s_tris.tris[1].v0 = {0.5,0,0};
//	s_tris.tris[1].v1 = {1.5,0,0};
//	s_tris.tris[1].v2 = {1,1,0};

// ---------------------------------------------------------------------------

//	s_tris.count = 70*100*10;
//	s_tris.tris = new Triangle[s_tris.count];
//	int i = 0;

//	float s = 122;
//	auto rc = make_random(-5.0f, 5.0f, s);
//	auto rd = make_random(0.1f, 0.5f, s);

//	for(int z = 0; z < 70; ++z)
//	{
//		for(int y = 0; y < 100; ++y)
//		{
//			for(int x = 0; x < 10; ++x)
//			{
//				vec3 c(rc(), rc(), rc());
//				Triangle t;
//				t.v0 = c + vec3(rd(), rd(), rd());
//				t.v1 = c + vec3(rd(), rd(), rd());
//				t.v2 = c + vec3(rd(), rd(), rd());
//				s_tris.tris[i++] = t;
//			}
//		}
//	}



	s_tris.ids = new int[s_tris.count];
	iota(s_tris.ids, s_tris.ids + s_tris.count, 0);

	for(int i = 0; i < s_tris.count; ++i)
	{
		expandBound(s_bound, s_tris.tris[i]);
	}

	rtvInit(640, 480);
	rtvReshapeCB(reshape);
	rtvRenderCB(render);
	rtvExec();
	return 0;
}
