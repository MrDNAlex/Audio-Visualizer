#pragma once
//#ifndef FOURIERTRANSFORM_H
//#define FOURIERTRANSFORM_H

// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <array>

struct FourierData {
	float real;
	float imag;
};

__global__ void NyquistFrequency(float* input, float* output, int* fft_size, int* numOfFrames);

__global__ void DFTMagnitudeGPU(float* input, float* output, int* fft_size, int* numOfFrames);

__global__ void DFTGPU(float* input, FourierData* output, int* fft_size, int* numOfFrames);

cudaError_t NyquistFrequencyMag(float* input, float* output, int fft_size, int numOfFrames);

cudaError_t FourierTransformMagnitude(float* input, float* output, int fft_size, int numOfFrames);

cudaError_t FourierTransform(float* input, FourierData* output, int fft_size, int numOfFrames);

//#endif // FOURIERTRANSFORM_H