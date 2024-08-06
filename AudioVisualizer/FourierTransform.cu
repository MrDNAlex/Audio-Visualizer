#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"
#include "FourierTransform.cuh"
#include <iostream>
#include <chrono>

__global__ void DFTMagnitudeGPU(float* input, float* output, int* fft_size, int* numOfFrames)
{
	const float pi = 3.14159265358979323846;
	int totalThreads = blockDim.x * gridDim.x;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int fftSize = *fft_size;
	int frameNums = *numOfFrames;

	int totalOps = fftSize * frameNums;

	if (threadId < totalOps) {
		int frameIndex = threadId / fftSize;  
		int k = threadId % fftSize;

		float real = 0.0f;
		float imag = 0.0f;
		float angleStart = 2 * pi * k / fftSize;

		for (int n = 0; n < fftSize; n++)
		{
			float angle = angleStart * n;
			float inputValue = input[frameIndex * fftSize + n];

			real += inputValue * cosf(angle);
			imag += inputValue * sinf(angle);
		}

		output[frameIndex * fftSize + k] = sqrtf(real * real + imag * imag);
	}
}

cudaError_t FourierTransformMagnitude(float* input, float* output, int fft_size, int numOfFrames)
{
	std::cout << "FourierTransform" << std::endl;

	int signalSize = fft_size * numOfFrames;

	float* kernel_input = 0;
	float* kernel_output = 0;
	int* kernel_fft_size = 0;
	int* kernel_numOfFrames = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, input, sizeof(float), signalSize);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output, sizeof(float), signalSize);

	int threadsPerBlock = 1024;
	int totalOps = fft_size * numOfFrames;
	int blocks = (totalOps + threadsPerBlock - 1) / threadsPerBlock;

	DFTMagnitudeGPU << <blocks, threadsPerBlock >> > (kernel_input, kernel_output, kernel_fft_size, kernel_numOfFrames);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, kernel_output, sizeof(float), signalSize);

	cudaFree(kernel_input);
	cudaFree(kernel_output);
	cudaFree(kernel_fft_size);
	cudaFree(kernel_numOfFrames);

	std::cout << "Finished FourierTransform" << std::endl;

	return cudaStatus;
}	

__global__ void DFTGPU(float* input, FourierData* output, int* fft_size, int* numOfFrames)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int N = *fft_size;
	int outputIndex = N * index;
	const float pi = 3.14159265358979323846;

	if (index >= *numOfFrames * *fft_size) return;

	for (int k = 0; k < N; k++)
	{
		float real = 0.0f;
		float imag = 0.0f;
		float angleStart = 2 * pi * k / N;

		for (int n = 0; n < N; n++)
		{
			float angle = angleStart * n;
			float inputValue = input[outputIndex + n];

			real += inputValue * cosf(angle);
			imag += inputValue * sinf(angle);
		}

		int signalIndex = outputIndex + k;

		output[signalIndex].real = real;
		output[signalIndex].imag = imag;
	}
}

cudaError_t FourierTransform(float* input, FourierData* output, int fft_size, int numOfFrames)
{
	std::cout << "FourierTransform" << std::endl;

	int signalSize = fft_size * numOfFrames;

	float* kernel_input = 0;
	FourierData* kernel_output = 0;
	int* kernel_fft_size = 0;
	int* kernel_numOfFrames = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, input, sizeof(float), signalSize);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output, sizeof(FourierData), signalSize);

	int threads = 1024;
	int blocks = (numOfFrames / threads) + 1;

	DFTGPU << <blocks, threads >> > (kernel_input, kernel_output, kernel_fft_size, kernel_numOfFrames);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, kernel_output, sizeof(float), signalSize);

	cudaFree(kernel_input);
	cudaFree(kernel_output);
	cudaFree(kernel_fft_size);
	cudaFree(kernel_numOfFrames);

	std::cout << "Finished FourierTransform" << std::endl;

	return cudaStatus;
}