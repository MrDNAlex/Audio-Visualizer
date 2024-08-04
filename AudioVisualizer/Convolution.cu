#include "Convolution.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"

int GetConvolutionOutputSize(int arraySize, int kernelSize, int stepSize)
{
	return ((arraySize - kernelSize) / stepSize) + 1;
}

__global__ void ConvolutionGPU(float* input, float* kernel, float* output, int* arraySize, int* kernelSize, int* step, int* outputSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int N = *arraySize;
	int K = *kernelSize;
	int S = *step;

	if (index >= *outputSize) return;

	float sum = 0.0f;

	int startIndex = index * S;

	for (int i = 0; i < K; i++) {
		int inputIndex = startIndex + i;
		int kernelIndex = K - 1 - i;

		// Check bounds for the input array
		if (inputIndex >= N) break;

		sum += input[inputIndex] * kernel[kernelIndex];
	}

	output[index] = sum;
}

cudaError_t Convolution(float* input, float* kernel, float* output, int arraySize, int kernelSize, int stepSize, int outputSize)
{
	float* gpuInput = 0;
	float* gpuKernel = 0;
	float* gpuOutput = 0;
	int* gpuArraySize = 0;
	int* gpuKernelSize = 0;
	int* gpuStep = 0;
	int* gpuOutputSize = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&gpuArraySize, &arraySize, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuKernelSize, &kernelSize, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuStep, &stepSize, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuOutputSize, &outputSize, sizeof(int));

	cudaStatus = AssignVariable((void**)&gpuInput, input, sizeof(float), arraySize);
	cudaStatus = AssignVariable((void**)&gpuKernel, kernel, sizeof(float), kernelSize);

	cudaStatus = AssignMemory((void**)&gpuOutput, sizeof(float), outputSize);

	int threads = 1024;
	int blocks = (outputSize / threads) + 1;

	ConvolutionGPU << <blocks, threads >> > (gpuInput, gpuKernel, gpuOutput, gpuArraySize, gpuKernelSize, gpuStep, gpuOutputSize);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, gpuOutput, sizeof(float), outputSize);

	cudaFree(gpuInput);
	cudaFree(gpuKernel);
	cudaFree(gpuOutput);
	cudaFree(gpuArraySize);
	cudaFree(gpuKernelSize);
	cudaFree(gpuStep);
	cudaFree(gpuOutputSize);

	return cudaStatus;
}