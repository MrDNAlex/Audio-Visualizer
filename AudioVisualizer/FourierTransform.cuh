#pragma once
#ifndef FOURIERTRANSFORM_H
#define FOURIERTRANSFORM_H

// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
//#include <stdio.h>
#include <math.h>
#include <vector>
#include <array>

__global__ void DFTGPU(float* input, float* output_real, float* output_imag, int* fft_size, int* numOfFrames);

cudaError_t FourierTransform(float* input, float* output_real, float* output_imag, int fft_size, int numOfFrames);

#endif // FOURIERTRANSFORM_H