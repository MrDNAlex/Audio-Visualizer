#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <MemoryManagement.h>
#include <vector>


__device__ float* normalizeGaus(float* vector, int size);

__device__ float getMean(float* vector, int size);

__device__ float getStandardDeviation(float* vector, int size);

__device__ int binarySearch(float* list, int size, float item);

float* linspacePtr(float start, float stop, int num);

float* logspacePtr(float start, float stop, int num);

cudaDeviceProp getDeviveProperties();

__global__ void BinFrequencie(int* gpuHalfDFTSize, int* gpuNumOfFrames, int* gpuNumOfBands, int* gpuTotalThreads, float* frequencyBins, float* logFrequencies, float* nyquistData, float* binnedFrequencies);

std::vector<std::vector<float>> BinFrequencies(float* nyquistFrequencies, int halfDFTSize, int numOfFrames, int numOfBands, int sampleRate);

float getMaxPntr(float* vector, int size);

float* normalizePntr(float* vector, int size);

float getStandardDeviationPntr(float* vector, int size);

float* normalizeGausPntr(float* vector, int size, float stdRange);

float getMeanPntr(float* vector, int size);





