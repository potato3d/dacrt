#include "rtv.h"
#include <bl/bl.h>

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

using namespace std;
#define BYTE_OFFSET(x) (char*)0 + x

// ---------------------------------------------------------------------------------------------------------------------

static int s_imgWidth = 800;
static int s_imgHeight = 600;
static int s_imgBytes = 0;

static void(*s_reshapeCB)(int, int);
static void(*s_renderCB)(unsigned char*);

static GLuint s_pboID = 0;
static GLuint s_texID = 0;
static GLuint s_vaoID = 0;

static int s_moveForward = 0;
static int s_moveBackward = 0;
static int s_moveLeft = 0;
static int s_moveRight = 0;
static int s_moveUp = 0;
static int s_moveDown = 0;
static bool s_mouseDrag = false;
static int s_mouseStartX = 0;
static int s_mouseStartY = 0;
static int s_mouseEndX = 0;
static int s_mouseEndY = 0;

static bool s_resetCamera = false;

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
		stringstream ss;
		ss << "rtview | " << fixed << setprecision(2) << fps << " fps | " << ms << " ms";
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
	case 27:
		exit(0);
	case ' ':
		s_resetCamera = true;
		break;
	case 'w':
	case 'W':
		s_moveForward = 1;
		break;
	case 's':
	case 'S':
		s_moveBackward = 1;
		break;
	case 'a':
	case 'A':
		s_moveLeft = 1;
		break;
	case 'd':
	case 'D':
		s_moveRight = 1;
		break;
	case 'r':
	case 'R':
		s_moveUp = 1;
		break;
	case 'f':
	case 'F':
		s_moveDown = 1;
		break;
	default:
		break;
	}
}

static void keyrelease(unsigned char c, int /*x*/, int /*y*/)
{
	switch(c)
	{
	case 27:
		exit(0);
	case 'w':
	case 'W':
		s_moveForward = 0;
		break;
	case 's':
	case 'S':
		s_moveBackward = 0;
		break;
	case 'a':
	case 'A':
		s_moveLeft = 0;
		break;
	case 'd':
	case 'D':
		s_moveRight = 0;
		break;
	case 'r':
	case 'R':
		s_moveUp = 0;
		break;
	case 'f':
	case 'F':
		s_moveDown = 0;
		break;
	default:
		break;
	}
}

static void mouseclick(int button, int state, int x, int y)
{
	if(button == GLUT_LEFT_BUTTON)
	{
		if(state == GLUT_DOWN)
		{
			s_mouseDrag = true;
			s_mouseStartX = x;
			s_mouseStartY = y;
			s_mouseEndX = x;
			s_mouseEndY = y;
		}
		else
		{
			s_mouseDrag = false;
		}
	}
}

static void mousemove(int x, int y)
{
	if(!s_mouseDrag)
	{
		return;
	}
	s_mouseEndX = x;
	s_mouseEndY = y;
}

static void reshape(int w, int h)
{
	s_imgWidth  = w;
	s_imgHeight = h;
	s_imgBytes  = s_imgWidth * s_imgHeight * 3 * sizeof(unsigned char);

	glViewport(0, 0, w, h);

	glBindTexture(GL_TEXTURE_2D, s_texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, s_imgWidth, s_imgHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s_pboID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, s_imgBytes, nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	s_reshapeCB(w, h);
}

static void display()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s_pboID);
	unsigned char* pixels = (unsigned char*)glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, s_imgBytes, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	s_renderCB(pixels);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

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
	glutInit(&argc, nullptr);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(s_imgWidth, s_imgHeight);
	glutInitWindowPosition(10, 10);
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

void rtvReshapeCB(void(*callback)(int, int))
{
	s_reshapeCB = callback;
}

void rtvRenderCB(void(*callback)(unsigned char*))
{
	s_renderCB = callback;
}

bool rtvResetCamera()
{
	bool reset = s_resetCamera;
	s_resetCamera = false;
	return reset;
}

void rtvMoveCamera(float* dx, float* dy, float* dz)
{
	static timer t;
	vec3 dt = vec3(-s_moveLeft+s_moveRight, s_moveUp-s_moveDown, s_moveBackward-s_moveForward).normalized() * t.sec();
	*dx = dt.x;
	*dy = dt.y;
	*dz = dt.z;
	t.restart();
}

void rtvRotateCamera(float* angle, float* axisx, float* axisy, float* axisz)
{
	if(!s_mouseDrag)
	{
		*angle = 0;
		*axisx = 1;
		*axisy = 0;
		*axisz = 0;
		return;
	}

	float rotateAroundX = math::HALF_PI * (s_mouseEndY - s_mouseStartY) / s_imgHeight;
	float rotateAroundY = math::HALF_PI * (s_mouseEndX - s_mouseStartX) / s_imgWidth;

	quat q = quat(-rotateAroundX, vec3::UNIT_X).mul(quat(-rotateAroundY, vec3::UNIT_Y));

	float radians;
	vec3 axis;
	q.get_rotation(radians, axis);
	*angle = radians == 0.0f? 0.0f : math::max(radians * 0.5f, 0.02f);
	*axisx = axis.x;
	*axisy = axis.y;
	*axisz = axis.z;
}

void rtvExec()
{
	glutMainLoop();
}
