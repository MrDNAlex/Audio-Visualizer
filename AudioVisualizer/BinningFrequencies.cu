#include "BinningFrequencies.cuh"

__device__ float* normalizeGaus(float* vector, int size)
{
	float mean = getMean(vector, size);
	float std = getStandardDeviation(vector, size);

	float interval = mean + 1.7 * std;

	if (std == 0)
		std = 0.0001;

	for (int i = 0; i < size; i++)
	{
		float value = vector[i];

		if (value > interval)
			value = interval;

		vector[i] = (value) / (1.7 * std);
	}

	return vector;
}

__device__ float getMean(float* vector, int size)
{
	float sum = 0;

	for (int i = 0; i < size; i++)
	{
		sum += vector[i];
	}

	return sum / size;
}

__device__ float getStandardDeviation(float* vector, int size)
{
	float mean = getMean(vector, size);
	float sum = 0;

	for (int i = 0; i < size; i++)
	{
		sum += pow(vector[i] - mean, 2);
	}

	return sqrtf(sum / size);
}

__device__ int binarySearch(float* list, int size, float item)
{
	int low = 0;
	int high = size - 1;

	while (low != (high + 1))
	{
		int middle = (low + high) / 2;

		if (list[middle] == item)
		{
			return middle;
		}
		else if (list[middle] < item)
		{
			low = middle + 1;
		}
		else
		{
			high = middle - 1;
		}
	}

	if (0 <= low && low < size)
	{
		return low;
	}
	else
	{
		return -1;
	}
}

__global__ void BinFrequencie(int* gpuHalfDFTSize, int* gpuNumOfFrames, int* gpuNumOfBands, int* gpuTotalThreads, float* frequencyBins, float* logFrequencies, float* nyquistData, float* binnedFrequencies)
{
	int halfDFTSize = *gpuHalfDFTSize;
	int numOfFrames = *gpuNumOfFrames;
	int bands = *gpuNumOfBands;
	int totalThreads = *gpuTotalThreads;

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < numOfFrames)
	{
		float* band = new float[bands];
		for (int j = 0; j < bands; j++)
		{
			int startIndex = binarySearch(frequencyBins, halfDFTSize, logFrequencies[j]);
			int endIndex = binarySearch(frequencyBins, halfDFTSize, logFrequencies[j + 1]);

			int delta = endIndex - startIndex;

			if (delta > 0)
			{
				float sum = 0;
				for (int k = startIndex; k <= endIndex; k++)
				{
					sum += nyquistData[index * halfDFTSize + k];
				}

				band[j] = (sum / (float)delta);
			}
			else
			{
				band[j] = 0;
			}
		}

		band = normalizeGaus(band, bands);

		for (int j = 0; j < bands; j++)
		{
			binnedFrequencies[index * bands + j] = band[j];
		}

		index += totalThreads;
	}
}

float* linspacePtr(float start, float stop, int num) {
	float* result = new float[num];
	float step = (stop - start) / (num - 1);

	for (int i = 0; i < num; ++i) {
		result[i] = start + i * step;
	}

	return result;
}

float* logspacePtr(float start, float stop, int num) {
	float* result = new float[num];
	float* lin_space = linspacePtr(start, stop, num);

	for (int i = 0; i < num; i++) {
		result[i] = pow(10, lin_space[i]);
	}

	delete[] lin_space;

	return result;
}

std::vector<std::vector<float>> BinFrequencies(float* nyquistFrequencies, int halfDFTSize, int numOfFrames, int numOfBands, int sampleRate)
{
	//Get the Frequency Bins
	float* freqBins = new float[halfDFTSize];

	for (int i = 0; i < halfDFTSize; i++) {
		freqBins[i] = (float)i * sampleRate / halfDFTSize;
	}

	float start = log10(5); // Change to 5?
	float stop = log10(freqBins[halfDFTSize - 1]);
	float* logFreqs = logspacePtr(start, stop, numOfBands + 1);

	int nyquistLength = halfDFTSize * numOfFrames;

	//CUDA Stuff
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	int* gpuFFTSize = 0;
	int* gpuNumOfFrames = 0;
	int* gpuNumOfBands = 0;
	int* gpuTotalThreads = 0;
	float* gpuNyquist = 0;
	float* gpuBandData = 0;
	float* gpuFrequencyBins = 0;
	float* gpuLogFrequencies = 0;

	int threadsPerBlock = 1024;
	int blocks = getDeviveProperties().multiProcessorCount;
	int totalThreads = blocks * threadsPerBlock;

	cudaStatus = AssignVariable((void**)&gpuFFTSize, &halfDFTSize, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuNumOfFrames, &numOfFrames, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuNumOfBands, &numOfBands, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuTotalThreads, &totalThreads, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuFrequencyBins, freqBins, sizeof(float), halfDFTSize);
	cudaStatus = AssignVariable((void**)&gpuLogFrequencies, logFreqs, sizeof(float), numOfBands + 1);
	cudaStatus = AssignVariable((void**)&gpuNyquist, nyquistFrequencies, sizeof(float), nyquistLength);

	cudaStatus = AssignMemory((void**)&gpuBandData, sizeof(float), numOfBands * numOfFrames);

	BinFrequencie << <blocks, threadsPerBlock >> > (gpuFFTSize, gpuNumOfFrames, gpuNumOfBands, gpuTotalThreads, gpuFrequencyBins, gpuLogFrequencies, gpuNyquist, gpuBandData);

	cudaStatus = cudaDeviceSynchronize();

	//Receive the Band Data
	float* bandData = new float[numOfBands * numOfFrames];

	cudaStatus = GetVariable(bandData, gpuBandData, sizeof(float), numOfBands * numOfFrames);

	std::vector<std::vector<float>> result;

	for (int i = 0; i < numOfFrames; i++)
	{
		std::vector<float> frame;

		for (int j = 0; j < numOfBands; j++)
		{
			frame.push_back(bandData[i * numOfBands + j]);
		}

		result.push_back(frame);
	}

	//Free Memory
	cudaFree(gpuFFTSize);
	cudaFree(gpuNumOfFrames);
	cudaFree(gpuNumOfBands);
	cudaFree(gpuTotalThreads);
	cudaFree(gpuNyquist);
	cudaFree(gpuFrequencyBins);
	cudaFree(gpuLogFrequencies);
	cudaFree(gpuBandData);

	delete[] freqBins;
	delete[] logFreqs;
	delete[] bandData;

	return result;
}