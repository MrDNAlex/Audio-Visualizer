#pragma once
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <array>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\FourierTransform.cuh"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\Convolution.cuh"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\Visualizer.cuh"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\BinningFrequencies.cuh"


//Monstercat Settings
//68 Bands (64 in reality (Actually it might be 80)
//5-20000 Hz
//2048 * 8 FFT Size
//30 FPS --> Upgrade to 60 FPS

class SongProcessing
{
public:

	//Processing properties
	int fps;

	int dftSize;
	int halfDFTSize;

	int numFrames;

	float* signal;

	std::vector<float> signalVector;

	float* processedSignal;

	int signalLength;

	int barHeight = 400;
	int bands = 65;


	//Audio file properties
	char* songPath;

	int sample_rate;

	int audioFrames;

	int channels;

	std::vector<short> audioBuffer;

	SNDFILE* file;

	SF_INFO info;

	SongProcessing(char* songPath, int fps, int dftSize)
	{
		this->fps = fps;
		this->dftSize = dftSize;
		this->halfDFTSize = dftSize / 2;

		extractAudio(songPath);

		this->numFrames = getNumOfFrames(audioBuffer, fps, dftSize, sample_rate, channels);
		this->signalLength = numFrames * dftSize;

		extractSignal(audioBuffer, fps, dftSize, sample_rate, channels);
	}

	void extractAudio(char* songPath)
	{
		memset(&info, 0, sizeof(info));

		file = sf_open(songPath, SFM_READ, &info);
		if (!file) {
			std::cerr << "Error opening file." << std::endl;
		}

		this->audioBuffer = std::vector<short>(info.frames * info.channels);
		this->sample_rate = info.samplerate;
		this->audioFrames = info.frames;
		this->channels = info.channels;

		sf_count_t num_frames = sf_readf_short(file, audioBuffer.data(), info.frames);

		sf_close(file);
	}

	std::vector<float> linspace(float start, float stop, int num) {
		std::vector<float> result;
		float step = (stop - start) / (num - 1);

		for (int i = 0; i < num; ++i) {
			result.push_back(start + i * step);
		}

		return result;
	}

	float* linspacePtr(float start, float stop, int num) {
		float* result = new float[num];
		float step = (stop - start) / (num - 1);

		for (int i = 0; i < num; ++i) {
			result[i] = start + i * step;
		}

		return result;
	}

	std::vector<float> logspace(float start, float stop, int num) {
		std::vector<float> result;
		std::vector<float> lin_space = linspace(start, stop, num);

		for (double v : lin_space) {
			result.push_back(pow(10, v));
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

	int binarySearch(std::vector<float> list, float item)
	{
		int low = 0;
		int high = list.size() - 1;

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

		if (0 <= low && low < list.size())
		{
			return low;
		}
	}

	float getMax(std::vector<float> vector)
	{
		float max = 0;

		for (int i = 0; i < vector.size(); i++)
		{
			if (vector[i] > max)
				max = vector[i];
		}

		return max;
	}

	float getMaxPntr(float* vector, int size)
	{
		float max = 0;

		for (int i = 0; i < size; i++)
		{
			if (vector[i] > max)
				max = vector[i];
		}

		return max;
	}

	float getMinPntr(float* vector, int size)
	{
		float min = 10000000;

		for (int i = 0; i < size; i++)
		{
			if (vector[i] < min)
				min = vector[i];
		}

		return min;
	}

	std::vector<float> gaussianNormalization(const std::vector<float>& data) {
		float sum = std::accumulate(data.begin(), data.end(), 0.0);
		float mean = sum / data.size();

		float sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
		float stdDev = std::sqrt(sq_sum / data.size() - mean * mean);

		std::vector<float> normalizedData(data.size());
		std::transform(data.begin(), data.end(), normalizedData.begin(),
			[mean, stdDev](float value) { return (value) / stdDev; });

		return normalizedData;
	}

	std::vector<float> normalizeGaus(std::vector<float> vector, float stdRange)
	{
		float mean = getMean(vector);
		float std = getStandardDeviation(vector);

		float interval = mean + stdRange * std;

		if (std == 0)
			std = 0.0001;

		for (int i = 0; i < vector.size(); i++)
		{
			float value = vector[i];

			if (value > interval)
				value = interval;

			vector[i] = (value) / (stdRange * std);
		}

		return vector;
	}

	float getStandardDeviationPntr(float* vector, int size)
	{
		float mean = getMeanPntr(vector, size);
		float sum = 0;

		for (int i = 0; i < size; i++)
		{
			sum += pow(vector[i] - mean, 2);
		}

		return sqrtf(sum / size);
	}

	float* normalizeGausPntr(float* vector, int size, float stdRange)
	{
		float mean = getMeanPntr(vector, size);
		float std = getStandardDeviationPntr(vector, size);

		float interval = mean + stdRange * std;

		if (std == 0)
			std = 0.0001;

		for (int i = 0; i < size; i++)
		{
			float value = vector[i];

			if (value > interval)
				value = interval;

			vector[i] = (value) / (stdRange * std);
		}

		return vector;
	}

	float getMeanPntr(float* vector, int size)
	{
		float sum = 0;

		for (int i = 0; i < size; i++)
		{
			sum += vector[i];
		}

		return sum / size;
	}

	float* normalizePntr(float* vector, int size)
	{
		float* normalized = new float[size];

		float max = getMaxPntr(vector, size);

		if (max == 0)
			max = 0.0001;

		for (int i = 0; i < size; i++)
			normalized[i] = vector[i] / max;

		return normalized;
	}

	std::vector<float> normalize(std::vector<float> vector)
	{
		float max = getMax(vector);

		if (max == 0)
			max = 0.0001;

		for (int i = 0; i < vector.size(); i++)
			vector[i] = vector[i] / max;

		return vector;
	}

	float* extractNyquistFrequencies(float* dft)
	{
		float* nyquist = new float[numFrames * (halfDFTSize)];

		std::cout << "Extracting Nyquist Frequencies" << std::endl;

		for (int i = 0; i < numFrames; i++) {
			for (int j = 0; j < halfDFTSize; j++) {
				int index = i * dftSize + j;
				nyquist[i * halfDFTSize + j] = dft[index];
			}
		}

		std::cout << "Finished Extracting Nyquist Frequency" << std::endl;

		return nyquist;
	}

	float getMean(std::vector<float> vector)
	{
		float sum = 0;

		for (int i = 0; i < vector.size(); i++)
		{
			sum += vector[i];
		}

		return sum / vector.size();
	}

	float getStandardDeviation(std::vector<float> vector)
	{
		float mean = getMean(vector);
		float sum = 0;

		for (int i = 0; i < vector.size(); i++)
		{
			sum += pow(vector[i] - mean, 2);
		}

		return sqrt(sum / vector.size());
	}

	//Make a Helper Function to Vectorize and Pointerize Data

	std::vector<float> aWeight(int size, float maxFreq)
	{
		std::vector<float> aWeighting(size);

		for (int i = 0; i < size; i++)
		{
			float frequencies = (((float)i) / size) * maxFreq;

			float ra = 12194 * 12194 * pow(frequencies, 4);
			float rb = pow(frequencies, 2) + 20.6 * 20.6;
			float rc = pow(frequencies, 2) + 107.7 * 107.7;
			float rd = pow(frequencies, 2) + 737.9 * 737.9;
			float re = pow(frequencies, 2) + 12194 * 12194;

			aWeighting[i] = ra / (rb * sqrt(rc * rd) * re);
		}

		return aWeighting;
	}

	void processSignal()
	{
		float* dftData = new float[dftSize * numFrames];

		FourierTransformMagnitude(signal, dftData, dftSize, numFrames);

		float* nyquist = extractNyquistFrequencies(dftData);

		//

		std::vector<float> aWeighting = aWeight(halfDFTSize, sample_rate);

		for (int i = 0; i < numFrames; i++)
		{
			for (int j = 0; j < halfDFTSize; j++)
			{
				nyquist[i * halfDFTSize + j] *= aWeighting[j];
			}
		}

		nyquist = normalizeGausPntr(nyquist, numFrames * halfDFTSize, 2);

		//nyquist = normalizePntr(nyquist, numFrames * halfDFTSize);

		delete[] dftData;

		std::cout << "Binning Frequencies" << std::endl;

		std::vector<std::vector<float>> bandData = BinFrequencies(nyquist, halfDFTSize, numFrames, bands, sample_rate / 2);

		//for (int i = 0; i < numFrames; i++)
		//{
		//	for (int j = 0; j < bands; j++)
		//	{
		//		bandData[i][j] = fminf(ExpScaling(bandData[i][j], 0.5), 1); //Fminf and Exp at 0.8?
		//	}
		//}

		/*std::vector<float> norm(numFrames * bands);

		for (int i = 0; i < numFrames; i++)
		{
			for (int j = 0; j < bands; j++)
			{
				norm[i * bands + j] = bandData[i][j];
			}
		}

		norm = normalizeGaus(norm, 2);

		for (int i = 0; i < numFrames; i++)
		{
			for (int j = 0; j < bands; j++)
			{
				bandData[i][j] = norm[i * bands + j];
			}
		}*/

		//bandData = ApplyDRC(bandData, 0.8); //0.7?

		//Just Normalize the Entire Binned Frequencies? Then apply the fminf and 0.8 exo scaling

		std::vector<std::vector<float>> normalizedBandData(bands);

		//Lets just try to refind the settings for Spacing 2. That looked good IMO

		for (int i = 0; i < bands; i++)
		{
			std::vector<float> band(numFrames);
			for (int j = 0; j < numFrames; j++)
			{
				band[j] = bandData[j][i];
			}

			//normalizedBandData[i] = normalizeGaus(band, 2); //1.9
			normalizedBandData[i] = band;
		}

		std::vector<std::vector<float>> rotatedNormalizedBandData(numFrames);

		for (int i = 0; i < numFrames; i++)
		{
			std::vector<float> frame(bands);
			for (int j = 0; j < bands; j++)
			{
				//frame[j] = fminf(ExpScaling(normalizedBandData[j][i], 0.5), 1); //Fminf and Exp at 0.8?
				//frame[j] = ExpScaling(normalizedBandData[j][i], 0.8); //Fminf and Exp at 0.8?
				frame[j] = normalizedBandData[j][i]; //Fminf and Exp at 0.8?
			}

			//rotatedNormalizedBandData[i] = normalize(frame);
			//rotatedNormalizedBandData[i] = normalizeGaus(frame);
			rotatedNormalizedBandData[i] = frame;
		}

		//rotatedNormalizedBandData = ApplyDRC(rotatedNormalizedBandData, 1.3);

		delete[] nyquist;

		std::cout << "Finished Binning Frequencies" << std::endl;

		std::vector<std::vector<float>> convolvedBands = smoothData(rotatedNormalizedBandData, bands);

		std::cout << "Generating frames" << std::endl;

		for (int i = 0; i < convolvedBands.size(); i++)
		{
			generateFrame(i, convolvedBands[i]);
		}
	}

	float ExpScaling(float magnitude, float alpha = 0.5)
	{
		return exp(alpha * magnitude) - 1;
	}

	std::vector<std::vector<float>> ApplyDRC(std::vector<std::vector<float>> bandData, float compressionRatio)
	{
		std::vector<std::vector<float>> drcBandData(bandData.size());

		for (int i = 0; i < bandData.size(); i++)
		{
			drcBandData[i] = std::vector<float>(bandData[i].size());

			for (int j = 0; j < bandData[i].size(); j++)
			{
				drcBandData[i][j] = DRC(bandData[i][j], compressionRatio, 1.0);
			}
		}

		return drcBandData;
	}

	float DRC(float magnitude, float compressionRatio, float gain)
	{
		return gain * pow(magnitude, compressionRatio);
	}

	std::vector<std::vector<float>> smoothData(std::vector<std::vector<float>> bandData, int bands)
	{
		std::cout << "Smoothening Data" << std::endl;

		int inputHeight = bandData.size();
		int inputWidth = bands;

		int kernelWidth = 3;
		int kernelHeight = 3;
		int kernelSize = kernelWidth * kernelHeight;

		int stepWidth = 1;
		int stepHeight = 1;

		int outputWidth = getConvolutionOutputSize(inputWidth, kernelWidth, stepWidth);
		int outputHeight = getConvolutionOutputSize(inputHeight, kernelHeight, stepHeight);

		float* output = new float[outputWidth * outputHeight];

		float* kernel = new float[kernelSize];
		for (int i = 0; i < kernelSize; i++)
		{
			kernel[i] = 1.0f / (float)kernelSize;
		}

		float* input = new float[inputWidth * inputHeight];

		for (int i = 0; i < inputHeight; i++)
		{
			for (int j = 0; j < inputWidth; j++)
			{
				input[i * inputWidth + j] = bandData[i][j];
			}
		}

		Convolution2D(input, kernel, output, inputWidth, inputHeight, kernelWidth, kernelHeight, stepWidth, stepHeight, outputWidth, outputHeight);

		std::vector<std::vector<float>> convolvedBands;

		for (int i = 0; i < outputHeight; i++)
		{
			std::vector<float> band;
			for (int j = 0; j < outputWidth; j++)
			{
				band.push_back(output[i * outputWidth + j]);
			}
			convolvedBands.push_back(band);
		}

		delete[] output;
		delete[] kernel;
		delete[] input;

		std::cout << "Finished Smoothing Data" << std::endl;

		return convolvedBands;
	}

	void generateFrame(int index, std::vector<float> bars)
	{
		int imageWidth = 1920;
		int imageHeight = 1080;

		int numBars = bars.size();
		int spacing = 8;

		int barWidth = (imageWidth - (spacing * (numBars))) / numBars;

		RectInfo* rects = new RectInfo[numBars];

		for (int i = 0; i < numBars; i++)
		{
			RectInfo rectInfo = RectInfo();

			rectInfo.width = barWidth;
			rectInfo.height = bars[i] * barHeight + spacing;
			rectInfo.xPos = spacing + (barWidth + spacing) * i;
			rectInfo.yPos = imageHeight - rectInfo.height - spacing;

			rectInfo.alpha = 255;
			rectInfo.red = 255;
			rectInfo.green = 0;
			rectInfo.blue = 0;

			rects[i] = rectInfo;
		}

		unsigned char* frame = VisualizeFrame(rects, numBars, index);

		char filename[100];

		sprintf(filename, "AudioVisualizerCache\\Frames\\frame%d.jpg", index);
		SaveJPEG(filename, frame, imageWidth, imageHeight);

		delete[] rects;
		delete[] frame;
	}

	void extractSignal(std::vector<short> audioSignal, int fps = 24, int fftSize = 2048, int sampleRate = 22050, int channels = 1)
	{
		int frameSize = (sampleRate * channels) / fps;

		int numFrames = getNumOfFrames(audioSignal, fps, fftSize, sampleRate, channels);

		signal = new float[signalLength];

		for (int i = 0; i < numFrames; i++) {
			for (int j = 0; j < fftSize; j++) {
				// Normalizing the short sample to float in the range [-1.0, 1.0]
				signal[i * fftSize + j] = audioSignal[i * frameSize + j] / 32768.0f;
			}
		}
	}

	int getNumOfFrames(std::vector<short> audioSignal, int fps = 24, int fftSize = 2048, int sample_rate = 22050, int channels = 1)
	{
		int frameSize = (sample_rate * channels) / fps; // This calculates how many samples per frame based on fps

		// Calculating the number of frames we can fit into the audioSignal with the given frameSize
		int numFrames = ((audioSignal.size() - fftSize) / frameSize);

		return numFrames;
	}

	void debugInfo()
	{
		std::cout << "Frames: " << info.frames << std::endl;
		std::cout << "Sample Rate: " << info.samplerate << std::endl;
		std::cout << "Channels: " << info.channels << std::endl;
		std::cout << "Duration: " << static_cast<double>(numFrames) / info.samplerate << " seconds." << std::endl;
		std::cout << "Buffer Size: " << audioBuffer.size() << std::endl;
	}
};

