#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "MemoryManagement.h"
#include "lodepng.h"

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

void generateBitmapImage(unsigned char* image, int height, int width, const char* imageFileName);

