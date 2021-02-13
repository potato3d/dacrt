#pragma once

void rtvInit(int w, int h);
void rtvReshapeCB(void(*callback)(int w, int h));
void rtvRenderCB(void(*callback)(unsigned char* pixels));
bool rtvResetCamera();
void rtvMoveCamera(float* dx, float* dy, float* dz);
void rtvRotateCamera(float* angle, float* axisx, float* axisy, float* axisz);
void rtvExec();
