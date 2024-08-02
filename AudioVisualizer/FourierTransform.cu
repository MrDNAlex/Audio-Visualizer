#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.cpp"
#include "FourierTransform.cuh"
#include <iostream>

__global__ void DFTGPU(float* input, float* output_real, float* output_imag, int* fft_size, int* numOfFrames)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int N = *fft_size;
	int outputIndex = N * index;
	const float pi = 3.14159265358979323846;

	if (index >= *numOfFrames) return;

	for (int k = 0; k < N; k++)
	{
		float real = 0.0f;
		float imag = 0.0f;

		for (int n = 0; n < N; n++) {

			float angle = 2 * pi * k * n / N;

			float inputValue = input[outputIndex + n];

			real += inputValue * cosf(angle);
			imag += inputValue * sinf(angle);
		}

		output_real[outputIndex + k] = real;
		output_imag[outputIndex + k] = imag;
	}
}

cudaError_t FourierTransform(float* input, float* output_real, float* output_imag, int fft_size, int numOfFrames)
{
	std::cout << "FourierTransform" << std::endl;

	int signalSize = fft_size * numOfFrames;

	float* kernel_input = 0;
	float* kernel_output_imag = 0;
	float* kernel_output_real = 0;
	int* kernel_fft_size = 0;
	int* kernel_numOfFrames = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, input, sizeof(float), signalSize);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output_real, sizeof(float), signalSize);
	cudaStatus = AssignMemory((void**)&kernel_output_imag, sizeof(float), signalSize);

	int threads = 1024;
	int blocks = (numOfFrames / 1024) + 1;

	DFTGPU << <blocks, threads >> > (kernel_input, kernel_output_real, kernel_output_imag, kernel_fft_size, kernel_numOfFrames);

	cudaStatus = cudaDeviceSynchronize();

	//output_real = new float[signalSize];
	//output_imag = new float[signalSize];

	cudaStatus = GetVariable(output_real, kernel_output_real, signalSize);
	cudaStatus = GetVariable(output_imag, kernel_output_imag, signalSize);

	cudaFree(kernel_input);
	cudaFree(kernel_output_real);
	cudaFree(kernel_output_imag);
	cudaFree(kernel_fft_size);
	cudaFree(kernel_numOfFrames);



	return cudaStatus;

}