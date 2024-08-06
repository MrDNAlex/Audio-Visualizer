#pragma once

#ifndef VISUALIZER_CUH
#define VISUALIZER_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "MemoryManagement.h"
#include "ImageSavers.h"

struct RectInfo
{
	//Top left Corner
	int xPos;
	int yPos;
	int width;
	int height;
	
	//Color Info
	int red;
	int green;
	int blue;
	int alpha;
};

cudaError_t VisualizeFrame(RectInfo* rects, int numOfRects, int frameIndex);

__global__ void VisualizeFrameGPU(RectInfo* rects, int* numOfRects, int* width, int* height, unsigned char* frame);

#endif // VISUALIZER_CUH