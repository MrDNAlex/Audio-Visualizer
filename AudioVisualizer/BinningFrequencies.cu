#include "BinningFrequencies.cuh"



__global__ void BinFrequencie(float* input, float* output, int* fft_size, int* numOfFrames, int* totalThreads)
{

}

//std::vector<std::vector<float>> BinFrequencies(float* nyquistFrequencies, int nyquistLength, int sampleRate, int dftSize, )
//{
//
//	std::vector<float> freqBins(halfDFTSize);
//
//	for (int i = 0; i < halfDFTSize; i++) {
//		freqBins[i] = (float)i * sampleRate / dftSize;
//	}
//
//
//
//
//}