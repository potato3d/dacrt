#pragma once

extern "C"
{
	// pixels are RGB, one byte per component

	void rtvInit(int w, int h);
	void rtvSetReshapeCallback(void(*callback)(int w, int h));
	void rtvSetCameraCallback(void(*callback)(float* eye, float* center, float* up));
	void rtvSetPixelRenderCallback(void(*callback)(unsigned char* pixels));
	void rtvSetBufferRenderCallback(void(*callback)(unsigned int pixelBufferID));
	void rtvSetCamera(float* eye, float* center, float* up);
	void rtvReshapeWindow(int w, int h);
	void rtvExec();
}
