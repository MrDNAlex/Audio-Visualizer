#ifndef FOURIERTRANSFORM_H
#define FOURIERTRANSFORM_H

// CUDA Runtime
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <array>

__global__ void DFTGPU(std::vector<float>* fft_input, std::vector<std::array<float, 2>>* fft_output, int* d_N);

cudaError_t FourierTransform(std::vector<float> h_input, std::vector<std::array<float, 2>> h_output, int N);

#endif // FOURIERTRANSFORM_H