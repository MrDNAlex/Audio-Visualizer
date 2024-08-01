#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryManagement.cpp"
//#include "FourierTransform.cuh"


__global__ void DFTGPU(float* input, float* output, int fft_size, int numOfFrames)
{
	//int fftSize = *fft_size;
	//int frames = *numOfFrames;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int N = fft_size;

	int outputIndex = fft_size * index;

	float pi = 3.14159265358979323846;

	for (int k = 0; k < N; k++)
	{
		float real = 0.0f;
		float imag = 0.0f;


		for (int n = 0; n < N; n++) {

			float angle = 2 * 3.141 * k * n / N;

			float inputValue = input[outputIndex + n];

			real += inputValue * cosf(angle);
			imag += inputValue * sinf(angle);
		}

		output[outputIndex + k * 2] = real;
		output[outputIndex + k * 2 + 1] = imag;

		/*float2 sum = make_float2(0.0f, 0.0f);
		for (int n = 0; n < N; n++)
		{
			float2 value = make_float2(cos(2 * pi * n * k / N), -sin(2 * pi * n * k / N));
			float2 input_value = make_float2(input[n + index * N], 0.0f);
			sum.x += input_value.x * value.x - input_value.y * value.y;
			sum.y += input_value.x * value.y + input_value.y * value.x;
		}
		output[outputIndex + k] = sqrt(sum.x * sum.x + sum.y * sum.y);*/
	}
}

cudaError_t FourierTransform(float* input, float* output, int fft_size, int numOfFrames)
{
	float* kernel_input = 0;
	float* kernel_output = 0;
	int kernel_fft_size = 0;
	int kernel_numOfFrames = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&kernel_input, &input, sizeof(float), fft_size * numOfFrames);
	cudaStatus = AssignVariable((void**)&kernel_fft_size, &fft_size, sizeof(int));
	cudaStatus = AssignVariable((void**)&kernel_numOfFrames, &numOfFrames, sizeof(int));

	cudaStatus = AssignMemory((void**)&kernel_output, sizeof(float), fft_size * numOfFrames * 2);

	int threads = 1024;
	int blocks = (numOfFrames / 1024) + 1;

	DFTGPU << <blocks, threads >> > (kernel_input, kernel_output, kernel_fft_size, kernel_numOfFrames);

	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = GetVariable((void**)&output, kernel_output, sizeof(float), fft_size * numOfFrames);

	return cudaStatus;

}