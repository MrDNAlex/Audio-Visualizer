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

int GetConvolutionOutputSize(int arraySize, int kernelSize, int stepSize);

int Get2DConvolutionOutputSize(int arraySizeX, int arraySizeY, int kernelSizeX, int kernelSizeY, int stepSizeX, int stepSizeY);



//__global__ void Convolution2DGPU(float* input, float* kernel, float* output, int* kernel_size, int* signal_size);

//cudaError_t Convolution2D(float* input, float* kernel, float* output, int kernel_size, int signal_size);

//#endif // CONVOLUTION_H