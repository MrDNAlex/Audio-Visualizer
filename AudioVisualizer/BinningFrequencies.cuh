
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <MemoryManagement.h>
#include <vector>


__global__ void BinFrequencie(float* input, float* output, int* fft_size, int* numOfFrames, int* totalThreads);

std::vector<std::vector<float>> BinFrequencies();









