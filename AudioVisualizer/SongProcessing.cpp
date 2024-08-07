#pragma once
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <array>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\FourierTransform.cuh"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\Convolution.cuh"
#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\Visualizer.cuh"


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


	//Audio file properties
	int sample_rate;

	int audioFrames;

	int channels;

	std::vector<short> audioBuffer;

	SNDFILE* file;

	SF_INFO info;

	SongProcessing(char* songPath, int fps, int dft_size)
	{
		this->fps = fps;
		this->dftSize = dft_size;
		this->halfDFTSize = dft_size / 2;

		extractAudio(songPath);

		this->numFrames = getNumOfFrames(audioBuffer, fps, dft_size, sample_rate, channels);
		this->signalLength = numFrames * dft_size;

		extractSignal(audioBuffer, fps, dft_size, sample_rate, channels);
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

	void processSignal()
	{
		int bands = 64;
		float* dftData = new float[dftSize * numFrames];

		std::vector<std::vector<float>> frames;

		for (int i = 0; i < numFrames; i++) {
			std::vector<float> frame;
			for (int j = 0; j < dftSize; j++) {
				frame.push_back(signal[i * dftSize + j]);
			}
			frames.push_back(frame);
		}

		FourierTransformMagnitude(signal, dftData, dftSize, numFrames);

		float* nyquist = extractNyquistFrequencies(dftData);

		delete[] dftData;

		std::cout << "Binning Frequencies" << std::endl;

		std::vector<float> freqBins(halfDFTSize);

		for (int i = 0; i < halfDFTSize; i++) {
			freqBins[i] = (float)i * sample_rate / dftSize;
			printf("Freq (%d): %f\n", i, freqBins[i]);
		}

		float start = 0;
		float stop = log10(freqBins[halfDFTSize - 1]);
		//float stop = freqBins[halfDFTSize - 1];

		float* logFreqs = logspacePtr(start, stop, bands + 1);

		for (int i = 0; i < bands; i++)
		{
			printf("LogFreq (%d): %f\n", i, logFreqs[i]);
		}

		std::vector<std::vector<float>> bandData(numFrames);

		for (int i = 0; i < numFrames; i++) {

			std::vector<float> band(bands);
			for (int j = 0; j < bands; j++)
			{
				int startIndex = binarySearch(freqBins, logFreqs[j]);
				int endIndex = binarySearch(freqBins, logFreqs[j + 1]);

				int delta = endIndex - startIndex;

				if (delta > 0)
				{
					float sum = 0;
					for (int k = startIndex; k <= endIndex; k++)
					{
						sum += nyquist[i * halfDFTSize + k];
					}

					band[j] = (sum / (float)delta);
				}
				else
				{
					band[j] = 0;
				}
			}

			std::vector<float> vec = normalize(band);

			bandData[i] = vec;
		}

		delete[] logFreqs;

		std::cout << "Finished Binning Frequencies" << std::endl;

		std::vector<std::vector<float>> convolvedBands = smoothData(bandData, bands);

		std::cout << "Generating frames" << std::endl;

		delete[] nyquist;

		for (int i = 0; i < convolvedBands.size(); i++)
		{
			generateFrame(i, convolvedBands[i]);
		}
	}

	std::vector<std::vector<float>> smoothData(std::vector<std::vector<float>> bandData, int bands)
	{
		std::cout << "Smoothening Data" << std::endl;

		int inputHeight = bandData.size();
		int inputWidth = bands;

		int kernelWidth = 5;
		int kernelHeight = 5;
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
		int spacing = 10;

		int barWidth = (imageWidth - (spacing * (numBars + 1))) / numBars;

		RectInfo* rects = new RectInfo[numBars];

		for (int i = 0; i < numBars; i++)
		{
			RectInfo rectInfo = RectInfo();

			rectInfo.width = barWidth;
			rectInfo.height = bars[i] * 800 + spacing;
			rectInfo.xPos = spacing + (barWidth + spacing) * i;
			rectInfo.yPos = imageHeight - rectInfo.height - spacing;

			rectInfo.alpha = 255;
			rectInfo.red = 255;
			rectInfo.green = 0;
			rectInfo.blue = 0;

			rects[i] = rectInfo;
		}

		VisualizeFrame(rects, numBars, index);

		delete[] rects;
	}

	void extractSignal(std::vector<short> audioSignal, int fps = 24, int fftSize = 2048, int sampleRate = 22050, int channels = 1)
	{
		int frameSize = (sampleRate * channels) / fps; // This calculates how many samples per frame based on fps

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

