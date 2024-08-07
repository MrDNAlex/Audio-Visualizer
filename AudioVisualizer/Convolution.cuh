#pragma once

//#ifndef CONVOLUTION_H
//#define CONVOLUTION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <array>

__global__ void ConvolutionGPU(float* input, float* kernel, float* output, int* arraySize, int* kernelSize, int* step, int* outputSize);

cudaError_t Convolution(float* input, float* kernel, float* output, int arraySize, int kernelSize, int stepSize, int outputSize);

int getConvolutionOutputSize(int width, int kernelSize, int stepSize);

int get2DConvolutionOutputSize(int arraySizeX, int arraySizeY, int kernelSizeX, int kernelSizeY, int stepSizeX, int stepSizeY);

std::pair<int, int> GetConvolutionOutputSize2D(int width, int height, int kernelWidth, int kernelHeight, int stepWidth, int stepHeight);


__global__ void Convolution2DGPU(float* input, float* kernel, float* output, int* inputWidth, int* inputHeight, int* kernelWidth, int* kernelHeight, int* step, int* outputWidth, int* outputHeight);

cudaError_t Convolution2D(float* input, float* kernel, float* output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int stepWidth, int stepHeight, int outputWidth, int outputHeight);

void DrawImage(int index);

//#endif // CONVOLUTION_H