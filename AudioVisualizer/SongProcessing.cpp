#pragma once
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <array>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "D:\NanoDNA Studios\Programming\Audio-Visualizer\AudioVisualizer\kernel.cu"


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

		sf_count_t num_frames = sf_readf_short(file, audioBuffer.data(), info.frames);

		this->sample_rate = info.samplerate;
		this->audioFrames = info.frames;
		this->channels = info.channels;
		this->audioBuffer = std::vector<short>(info.frames * info.channels);

		numFrames = GetNumOfFrames(audioBuffer, fps, fft_size, sample_rate, channels);

		signalLength = numFrames * fft_size;

		ExtractSignal(audioBuffer, fps, fft_size, sample_rate, channels);

		sf_close(file);
	}

	void ProcessSignal()
	{
		//FourierTransform(signal, processedSignal, fft_size, numFrames);

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
			float angle = -2 * 3.141 * k / N;
			float real_odd = cosf(angle) * odd[k][0] - sinf(angle) * odd[k][1];
			float imag_odd = sinf(angle) * odd[k][0] + cosf(angle) * odd[k][1];

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
			if (i % 20 == 0) {
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

