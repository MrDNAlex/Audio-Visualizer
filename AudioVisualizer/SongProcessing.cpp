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


class SongProcessing
{

public:

	//Processing properties
	int fps;

	int fft_size;

	int numFrames;

	float* signal;

	float* processedSignal;

	int signalLength;


	//Audio file properties

	int sample_rate;

	int audioFrames;

	int channels;

	std::vector<short> audioBuffer;

	SNDFILE* file;

	SF_INFO info;

	SongProcessing(char* songPath, int fps, int fft_size)
	{

		this->fps = fps;
		this->fft_size = fft_size;

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

		numFrames = GetNumOfFrames(audioBuffer, fps, fft_size, sample_rate, channels);

		signalLength = numFrames * fft_size;

		ExtractSignal(audioBuffer, fps, fft_size, sample_rate, channels);

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

	std::vector<float> logspace(float start, float stop, int num) {
		std::vector<float> result;
		std::vector<float> lin_space = linspace(start, stop, num);

		for (double v : lin_space) {
			result.push_back(pow(10, v));
		}

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

	float GetMax(std::vector<float> vector)
	{
		float max = 0;

		for (int i = 0; i < vector.size(); i++)
		{
			if (vector[i] > max)
				max = vector[i];
		}

		return max;
	}

	std::vector<float> Normalize(std::vector<float> vector)
	{
		float max = GetMax(vector);
		//std::vector<float> vec = *vector;

		if (max == 0)
			max = 0.0001;

		for (int i = 0; i < vector.size(); i++)
			vector[i] = vector[i] / max;

		return vector;
	}

	void ProcessSignal()
	{
		int bands = 60;
		float* data = new float[fft_size * numFrames];

		FourierTransformMagnitude(signal, data, fft_size, numFrames);

		std::vector<std::vector<float>> nyquistMag;

		for (int i = 0; i < numFrames; i++) {
			std::vector<float> frame;

			for (int j = 0; j < fft_size / 2; j++) {
				int index = i * fft_size + j;
				frame.push_back(data[index]);
			}
			nyquistMag.push_back(frame);
		}

		std::vector<float> freqBins(nyquistMag[0].size());

		for (int i = 0; i < fft_size / 2; i++) {
			freqBins[i] = (float)i * sample_rate / fft_size;
		}

		float start = 0;
		float stop = log10(freqBins[(fft_size / 2) - 1]);

		std::vector<float> logFreqs = logspace(start, stop, bands + 1);

		std::vector<std::vector<float>> bandData(nyquistMag.size());

		for (int i = 0; i < nyquistMag.size(); i++) {

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
						sum += nyquistMag[i][k];
					}

					band[j] = (sum / (float)delta);
				}
				else
				{
					band[j] = 0;
				}
			}

			std::vector<float> vec = Normalize(band);

			bandData[i] = vec;
		}


		int inputSize = 10;
		int kernelSize = 3;
		int stepSize = 1;
		int outputSize = GetConvolutionOutputSize(inputSize, kernelSize, stepSize);


		float* testSignal = new float[inputSize] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		float* testKernel = new float[kernelSize] {1, 0, -1};

		float* output = new float[outputSize];


		Convolution(testSignal, testKernel, output, inputSize, kernelSize, stepSize, outputSize);

		for (int i = 0; i < outputSize; i++)
		{
			std::cout << output[i] << std::endl;
		}


		int hello = 0;



		//std::vector<std::vector<std::array<float, 2>>> classic = oldMethod();

		//std::cout << "Comparing Nyquist and Classic DFT..." << std::endl;

		//for (int i = 0; i < numFrames; i++)
		//{
		//	for (int j = 0; j < nyquistMag[i].size(); j++)
		//	{
		//		float mag = sqrtf(classic[i][j][0] * classic[i][j][0] + classic[i][j][1] * classic[i][j][1]);

		//		std::cout << "Mags: " << nyquistMag[i][j] - mag << std::endl;


		//		//std::cout << "Real: " << nyquistMag[i][j] - classic[i][j][0] << " Imag: " << nyquist[i][j][1] - classic[i][j][1] << std::endl;
		//	}
		//}
	}

	std::vector<std::vector<std::array<float, 2>>> oldMethod()
	{
		std::vector<std::vector<float>> frames = FrameAudio(audioBuffer, fps, fft_size, sample_rate, info.channels);
		std::vector<std::vector<std::array<float, 2>>> dft_frames = DFTFrames(frames);
		std::vector<std::vector<std::array<float, 2>>> dft_frames_nyquist = ExtractNyquist(dft_frames);

		std::cout << "Classic DFT Calculated." << std::endl;

		return dft_frames_nyquist;
	}

	std::vector<std::vector<std::array<float, 2>>> ExtractNyquist(std::vector<std::vector<std::array<float, 2>>> dft_frames)
	{
		std::vector<std::vector<std::array<float, 2>>> dft_frames_nyquist;

		std::cout << "Extracting Nyquist DFT..." << std::endl;
		for (int i = 0; i < dft_frames.size(); i++) {
			std::vector<std::array<float, 2>> dft_frame_nyquist;
			for (int j = 0; j < (dft_frames[i].size() / 2) + 1; j++) {
				dft_frame_nyquist.push_back(dft_frames[i][j]);
			}
			dft_frames_nyquist.push_back(dft_frame_nyquist);
		}
		std::cout << "Exracted Nyquist DFT" << std::endl;

		return dft_frames_nyquist;
	}

	void ExtractSignal(std::vector<short> audioSignal, int fps = 24, int fft_size = 2048, int sample_rate = 22050, int channels = 1)
	{
		int frameSize = (sample_rate * channels) / fps; // This calculates how many samples per frame based on fps

		int numFrames = GetNumOfFrames(audioSignal, fps, fft_size, sample_rate, channels);

		signal = new float[signalLength];

		for (int i = 0; i < numFrames; i++) {
			for (int j = 0; j < fft_size; j++) {
				// Normalizing the short sample to float in the range [-1.0, 1.0]
				signal[i * fft_size + j] = audioSignal[i * frameSize + j] / 32768.0f;
			}
		}
	}

	int GetNumOfFrames(std::vector<short> audioSignal, int fps = 24, int fft_size = 2048, int sample_rate = 22050, int channels = 1)
	{
		int frameSize = (sample_rate * channels) / fps; // This calculates how many samples per frame based on fps

		// Calculating the number of frames we can fit into the audioSignal with the given frameSize
		int numFrames = ((audioSignal.size() - fft_size) / frameSize);

		return numFrames;
	}

	void DebugInfo()
	{
		std::cout << "Frames: " << info.frames << std::endl;
		std::cout << "Sample Rate: " << info.samplerate << std::endl;
		std::cout << "Channels: " << info.channels << std::endl;
		std::cout << "Duration: " << static_cast<double>(numFrames) / info.samplerate << " seconds." << std::endl;
		std::cout << "Buffer Size: " << audioBuffer.size() << std::endl;
	}

	std::vector<std::vector<float>> FrameAudio(std::vector<short> audioSignal, int fps = 24, int fft_size = 2048, int sample_rate = 22050, int channels = 1)
	{
		std::cout << "Framing Audio..." << std::endl;

		std::vector<std::vector<float>> frames;

		int frameSize = (sample_rate * channels) / fps; // This calculates how many samples per frame based on fps

		// Calculating the number of frames we can fit into the audioSignal with the given frameSize
		int numFrames = ((audioSignal.size() - fft_size) / frameSize);

		for (int i = 0; i < numFrames; i++) {
			std::vector<float> frame(fft_size);
			for (int j = 0; j < fft_size; j++) {
				// Normalizing the short sample to float in the range [-1.0, 1.0]
				frame[j] = audioSignal[i * frameSize + j] / 32768.0f;
			}
			frames.push_back(frame);
		}

		std::cout << "Audio Framed." << std::endl;

		return frames;
	}

	std::vector<std::array<float, 2>> DFT(std::vector<float> frame)
	{
		int N = frame.size();
		std::vector<std::array<float, 2>> X(N);
		for (int k = 0; k < N; k++) {
			float real = 0.0f;
			float imag = 0.0f;
			for (int n = 0; n < N; n++) {

				float angle = 2 * 3.141 * k * n / N;

				real += frame[n] * cosf(angle);
				imag += frame[n] * sinf(angle);
			}
			X[k][0] = real;
			X[k][1] = imag;
		}
		return X;
	}

	std::vector<std::array<float, 2>> FFT(std::vector<float> frame)
	{

		const float pi = 3.14159265358979323846;

		int N = frame.size();

		if (N <= 1) {
			return std::vector<std::array<float, 2>>(1, { frame[0], 0.0f });
		}

		//Compute Even signals
		std::vector<float> frame_even;
		std::vector<float> frame_odd;

		for (int i = 0; i < N; i += 2) {
			frame_even.push_back(frame[i]);
			frame_odd.push_back(frame[i + 1]);
		}

		std::vector<std::array<float, 2>> even = FFT(frame_even);
		std::vector<std::array<float, 2>> odd = FFT(frame_odd);

		std::vector<float> real(N);
		std::vector<float> imag(N);

		for (int k = 0; k < N / 2; k++) {
			float angle = -2 * pi * k / N;
			float real_odd = cosf(angle) * odd[k][0] + sinf(angle) * odd[k][1];
			float imag_odd = -sinf(angle) * odd[k][0] + cosf(angle) * odd[k][1];

			real[k] = even[k][0] + real_odd;
			imag[k] = even[k][1] + imag_odd;

			real[k + N / 2] = even[k][0] - real_odd;
			imag[k + N / 2] = even[k][1] - imag_odd;
		}

		std::vector<std::array<float, 2>> X(N);

		for (int i = 0; i < real.size(); i++) {
			X[i][0] = real[i];
			X[i][1] = imag[i];
		}

		return X;
	}

	std::vector<std::vector<std::array<float, 2>>> DFTFrames(std::vector<std::vector<float>> frames)
	{
		std::vector<std::vector<std::array<float, 2>>> dft_frames;

		std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();

		std::cout << "Calculating DFT..." << std::endl;

		for (int i = 0; i < frames.size(); i++) {
			if (i % 100 == 0) {
				std::cout << ((float)i) / ((float)frames.size()) << "% Complete" << std::endl;
			}
			dft_frames.push_back(FFT(frames[i]));
		}

		std::cout << "DFT Calculated." << std::endl;

		std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();

		std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;

		return dft_frames;
	}

};

