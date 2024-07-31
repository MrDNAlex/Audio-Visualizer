#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <MemoryManagement.cpp>
#include <vector>
#include <array>

__global__ void DFTGPU()
{

}


cudaError_t FourierTransform(std::vector<float> h_input, std::vector<std::array<float, 2>> h_output, int N)
{
	std::vector<float>* d_input;
	std::vector<std::array<float, 2>>* d_output;
	int* d_N;

	cudaError_t cudaStatus;

	cudaStatus = AssignVariable((void**)&d_input, &h_input, sizeof(std::vector<float>), h_input.size());
	cudaStatus = AssignVariable((void**)&d_N, &N, sizeof(int));
	cudaStatus = AssignMemory((void**)&d_output, sizeof(std::vector<std::array<float, 2>>), h_output.size());

	int threads = 1024;
	int blocks = (h_input.size() / 1024) + 1;

	DFTGPU << <blocks, threads >> >();

	cudaStatus = GetVariable((void**)&h_output, d_output, sizeof(std::vector<std::array<float, 2>>), h_output.size());

	return cudaStatus;
}










