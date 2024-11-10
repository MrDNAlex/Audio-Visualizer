#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"
#include "FourierTransform.cuh"
#include <iostream>
#include <chrono>

cudaDeviceProp getDeviveProperties()
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	return prop;
}

__global__ void NyquistFrequency(float* input, float* output, int* fft_size, int* numOfFrames, int* totalThreads)
{
	const float pi = 3.14159265358979323846;

	int fftSize = *fft_size;
	int halfFFTSize = fftSize / 2;
	int frameNums = *numOfFrames;
	int parallelThreads = *totalThreads;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int totalOps = halfFFTSize * frameNums;

	while (index < totalOps)
	{
		int frameIndex = index / (halfFFTSize);
		int k = index % (halfFFTSize);

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

		output[frameIndex * (halfFFTSize) + k] = sqrtf(real * real + imag * imag);

		index += parallelThreads;
	}
}

cudaError_t NyquistFrequencyMag(float* input, float* output, int fft_size, int numOfFrames)
{
	std::cout << "FourierTransform" << std::endl;

	int multiprocessorCount = getDeviveProperties().multiProcessorCount;
	int threads = 1024;
	int totalThreads = multiprocessorCount * threads;
	int count = 0;

	int signalSize = fft_size * numOfFrames;

	float* kernel_input = 0;
	float* kernel_output = 0;
	int* kernel_fft_size = 0;
	int* kernel_numOfFrames = 0;
	int* kernel_totalThreads = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, input, sizeof(float), signalSize);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_totalThreads, &totalThreads, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output, sizeof(float), signalSize / 2);

	int threadsPerBlock = 1024;
	int totalOps = fft_size * numOfFrames;
	int blocks = (totalOps + threadsPerBlock - 1) / threadsPerBlock;

	NyquistFrequency << <multiprocessorCount, threadsPerBlock >> > (kernel_input, kernel_output, kernel_fft_size, kernel_numOfFrames, kernel_totalThreads);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, kernel_output, sizeof(float), signalSize / 2);

	cudaFree(kernel_input);
	cudaFree(kernel_output);
	cudaFree(kernel_fft_size);
	cudaFree(kernel_numOfFrames);
	cudaFree(kernel_totalThreads);

	std::cout << "Finished FourierTransform" << std::endl;

	return cudaStatus;
}

__global__ void DFTMagnitudeGPU(float* input, float* output, int* fft_size, int* numOfFrames, int* totalThreads)
{
	const float pi = 3.14159265358979323846;

	int fftSize = *fft_size;
	int frameNums = *numOfFrames;
	int parallelThreads = *totalThreads;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int totalOps = fftSize * frameNums;

	while (index < totalOps)
	{
		int frameIndex = index / fftSize;
		int k = index % fftSize;

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

		index += parallelThreads;
	}
}

cudaError_t FourierTransformMagnitude(float* input, float* output, int fft_size, int numOfFrames)
{
	std::cout << "FourierTransform" << std::endl;

	int multiprocessorCount = getDeviveProperties().multiProcessorCount;
	int threads = 1024;
	int totalThreads = multiprocessorCount * threads;
	int count = 0;

	int signalSize = fft_size * numOfFrames;

	float* kernel_input = 0;
	float* kernel_output = 0;
	int* kernel_fft_size = 0;
	int* kernel_numOfFrames = 0;
	int* kernel_totalThreads = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, input, sizeof(float), signalSize);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_totalThreads, &totalThreads, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output, sizeof(float), signalSize);

	int threadsPerBlock = 1024;
	int totalOps = fft_size * numOfFrames;
	int blocks = (totalOps + threadsPerBlock - 1) / threadsPerBlock;

	DFTMagnitudeGPU << <multiprocessorCount, threadsPerBlock >> > (kernel_input, kernel_output, kernel_fft_size, kernel_numOfFrames, kernel_totalThreads);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, kernel_output, sizeof(float), signalSize);

	cudaFree(kernel_input);
	cudaFree(kernel_output);
	cudaFree(kernel_fft_size);
	cudaFree(kernel_numOfFrames);
	cudaFree(kernel_totalThreads);

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