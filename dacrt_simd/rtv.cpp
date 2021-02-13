#include "rtv.h"

#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <GL/glxew.h>
#endif
#include <GL/freeglut.h>

#include <sstream>
#include <iomanip>

// ---------------------------------------------------------------------------------------------------------------------

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

class timer
{
public:
	timer() { restart(); }

	void restart()
	{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		LARGE_INTEGER li;

		if (!QueryPerformanceFrequency(&li))
		{
			printf("QueryPerformanceFrequency failed!\n");
		}

		PCFreq = (double)li.QuadPart/1000.0;
		QueryPerformanceCounter(&li);
		_start = li.QuadPart;
#else
    	gettimeofday(&_start, NULL);
#endif
	}

	double msec()
	{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-_start)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &_start, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
	}

	double sec()  { return msec() * 1e-3; }
	double usec() { return msec() * 1e3; }
	double nsec() { return msec() * 1e6; }

private:
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	typedef __int64 time_point;
#else
	typedef struct timeval time_point;
#endif
	time_point _start;
};

// ---------------------------------------------------------------------------------------------------------------------

struct vec3
{
	float x, y, z;
	vec3() : x(0), y(0), z(0) {}
	vec3(float a, float b, float c) : x(a), y(b), z(c) {}
	vec3(float* p) : x(p[0]), y(p[1]), z(p[2]) {}
	float* ptr() { return &x; }
};

// ---------------------------------------------------------------------------------------------------------------------

#define BYTE_OFFSET(x) (char*)0 + x

static int s_imgWidth = 800;
static int s_imgHeight = 600;
static int s_imgBytes = 0;

static void(*s_reshapeCB)(int, int) = 0;
static void(*s_cameraCB)(float*, float*, float*) = 0;
static void(*s_pixelRenderCB)(unsigned char*) = 0;
static void(*s_bufferRenderCB)(unsigned int) = 0;

static GLuint s_pboID = 0;
static GLuint s_texID = 0;
static GLuint s_vaoID = 0;

static vec3 s_eye = vec3(0,0,15);
static vec3 s_center = vec3(0,0,0);
static vec3 s_up = vec3(0,1,0);

// ---------------------------------------------------------------------------------------------------------------------

static void updateFPS()
{
	static int nframes = 0;
	static timer fpsTimer;
	++nframes;
	double sec = fpsTimer.sec();
	if(sec > 0.5)
	{
		double fps = nframes / sec;
		double ms = 1000.0 / fps;
		std::stringstream ss;
		ss << "rtview | " << std::fixed << std::setprecision(2) << fps << " fps | " << ms << " ms";
		glutSetWindowTitle(ss.str().data());
		nframes = 0;
		fpsTimer.restart();
	}
}

static void initGL()
{
	glewInit();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glGenBuffers(1, &s_pboID);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &s_texID);
	glBindTexture(GL_TEXTURE_2D, s_texID);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	float vboData[] = {-1,-1,0,0,
						1,-1,1,0,
						1, 1,1,1,
					   -1, 1,0,1};
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vboData), vboData, GL_STATIC_DRAW);

	glGenVertexArrays(1, &s_vaoID);
	glBindVertexArray(s_vaoID);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 4*sizeof(float), BYTE_OFFSET(0));
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 4*sizeof(float), BYTE_OFFSET(2*sizeof(float)));
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

#ifdef _WIN32
	wglSwapIntervalEXT(0);
#else
	glXSwapIntervalSGI(0);
#endif
}

// ---------------------------------------------------------------------------------------------------------------------

static void keypress(unsigned char c, int /*x*/, int /*y*/)
{
	switch(c)
	{
	default:
		break;
	}
}

static void keyrelease(unsigned char c, int /*x*/, int /*y*/)
{
	switch(c)
	{
	case 27:
		glutLeaveMainLoop();
	default:
		break;
	}
}

static void mouseclick(int button, int state, int x, int y)
{

}

static void mousemove(int x, int y)
{

}

static void reshape(int w, int h)
{
	s_imgWidth  = w;
	s_imgHeight = h;
	s_imgBytes  = s_imgWidth * s_imgHeight * 3 * sizeof(unsigned char);

	glViewport(0, 0, w, h);

	glBindTexture(GL_TEXTURE_2D, s_texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, s_imgWidth, s_imgHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s_pboID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, s_imgBytes, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	s_reshapeCB(w, h);
}

static void display()
{
	// todo: only if camera changed
	s_cameraCB(s_eye.ptr(), s_center.ptr(), s_up.ptr());

	if(s_bufferRenderCB)
	{
		s_bufferRenderCB(s_pboID);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s_pboID);

	if(s_pixelRenderCB)
	{
		unsigned char* pixels = (unsigned char*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, s_imgBytes, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
		s_pixelRenderCB(pixels);
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	}

	glBindTexture(GL_TEXTURE_2D, s_texID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s_imgWidth, s_imgHeight, GL_RGB, GL_UNSIGNED_BYTE, BYTE_OFFSET(0));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glBindVertexArray(s_vaoID);
	glDrawArrays(GL_QUADS, 0, 4);

	glutSwapBuffers();

	updateFPS();
}

// ---------------------------------------------------------------------------------------------------------------------

void rtvInit(int w, int h)
{
	s_imgWidth = w;
	s_imgHeight = h;
	int argc = 0;
	glutInit(&argc, NULL);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(s_imgWidth, s_imgHeight);
	glutInitWindowPosition(400, 50);
	glutCreateWindow("rtview");
	glutKeyboardFunc(keypress);
	glutKeyboardUpFunc(keyrelease);
	glutMouseFunc(mouseclick);
	glutMotionFunc(mousemove);
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	initGL();
}

void rtvSetReshapeCallback(void(*callback)(int, int))
{
	s_reshapeCB = callback;
}

void rtvSetCameraCallback(void(*callback)(float*, float*, float*))
{
	s_cameraCB = callback;
}

void rtvSetPixelRenderCallback(void(*callback)(unsigned char*))
{
	s_pixelRenderCB = callback;
	if(s_bufferRenderCB)
	{
		s_bufferRenderCB = 0;
	}
}

void rtvSetBufferRenderCallback(void(*callback)(unsigned int))
{
	s_bufferRenderCB = callback;
	if(s_pixelRenderCB)
	{
		s_pixelRenderCB = 0;
	}
}

void rtvSetCamera(float* eye, float* center, float* up)
{
	s_eye = vec3(eye);
	s_center = vec3(center);
	s_up = vec3(up);
	s_cameraCB(s_eye.ptr(), s_center.ptr(), s_up.ptr());
}

void rtvReshapeWindow(int w, int h)
{
	glutReshapeWindow(w, h);
}

void rtvExec()
{
	glutMainLoop();
}
