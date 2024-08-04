#include "Convolution.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.h"

int GetConvolutionOutputSize(int width, int kernelSize, int stepSize)
{
	return ((width - kernelSize) / stepSize) + 1;
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

std::pair<int, int> GetConvolutionOutputSize2D(int width, int height, int kernelWidth, int kernelHeight, int stepWidth, int stepHeight) {
	int outputWidth = GetConvolutionOutputSize(width, kernelWidth, stepWidth);
	int outputHeight = GetConvolutionOutputSize(height, kernelHeight, stepHeight);

	return { outputWidth, outputHeight };
}

__global__ void Convolution2DGPU(float* input, float* kernel, float* output, int* inputWidth, int* inputHeight, int* kernelWidth, int* kernelHeight, int* stepWidth, int* stepHeight, int* outputWidth, int* outputHeight)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int KWidth = *kernelWidth;
	int KHeight = *kernelHeight;
	int SWidth = *stepWidth;
	int SHeight = *stepHeight;

	if (col >= *outputWidth || row >= *outputHeight) return;

	float sum = 0.0f;

	// Starting point of the kernel top left corner overlay with input
	int startRow = row * SHeight;
	int startCol = col * SWidth;

	for (int i = 0; i < KHeight; ++i) {
		for (int j = 0; j < KWidth; ++j) {
			int rowIndex = startRow + i;
			int colIndex = startCol + j;

			// Check bounds for the input array
			if (rowIndex >= *inputHeight || colIndex >= *inputWidth) continue;

			// Find index in input and kernel
			int inputIndex = rowIndex * *inputWidth + colIndex;
			int kernelIndex = (KHeight - 1 - i) * KWidth + (KWidth - 1 - j);

			sum += input[inputIndex] * kernel[kernelIndex];
		}
	}

	// Write the result to the output
	int outputIndex = row * *outputWidth + col;
	output[outputIndex] = sum;
}

cudaError_t Convolution2D(float* input, float* kernel, float* output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int stepWidth, int stepHeight, int outputWidth, int outputHeight)
{
	float* gpuInput = 0;
	float* gpuKernel = 0;
	float* gpuOutput = 0;
	int* gpuInputWidth = 0;
	int* gpuInputHeight = 0;
	int* gpuKernelWidth = 0;
	int* gpuKernelHeight = 0;
	int* gpuStepWidth = 0;
	int* gpuStepHeight = 0;
	int* gpuOutputWidth = 0;
	int* gpuOutputHeight = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&gpuInputWidth, &inputWidth, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuInputHeight, &inputHeight, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuKernelWidth, &kernelWidth, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuKernelHeight, &kernelHeight, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuStepWidth, &stepWidth, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuStepHeight, &stepHeight, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuOutputWidth, &outputWidth, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuOutputHeight, &outputHeight, sizeof(int));

	cudaStatus = AssignVariable((void**)&gpuInput, input, sizeof(float), inputWidth * inputHeight);
	cudaStatus = AssignVariable((void**)&gpuKernel, kernel, sizeof(float), kernelWidth * kernelHeight);

	cudaStatus = AssignMemory((void**)&gpuOutput, sizeof(float), outputWidth * outputHeight);

	dim3 threads(16, 16);
	dim3 blocks((outputWidth / threads.x) + 1, (outputHeight / threads.y) + 1);

	Convolution2DGPU << <blocks, threads >> > (gpuInput, gpuKernel, gpuOutput, gpuInputWidth, gpuInputHeight, gpuKernelWidth, gpuKernelHeight, gpuStepWidth, gpuStepHeight, gpuOutputWidth, gpuOutputHeight);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable(output, gpuOutput, sizeof(float), outputWidth * outputHeight);

	cudaFree(gpuInput);
	cudaFree(gpuKernel);
	cudaFree(gpuOutput);
	cudaFree(gpuInputWidth);
	cudaFree(gpuInputHeight);
	cudaFree(gpuKernelWidth);
	cudaFree(gpuKernelHeight);
	cudaFree(gpuStepWidth);
	cudaFree(gpuStepHeight);
	cudaFree(gpuOutputWidth);
	cudaFree(gpuOutputHeight);

	return cudaStatus;
}
