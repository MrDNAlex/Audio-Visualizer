// AudioVisualizer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <sndfile.h>
#include <vector>
#include <math.h>
#include <array>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FourierTransform.cu"

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
			std::cout << ((float)i)/((float)frames.size()) <<  "% Complete"  << std::endl;
		}

		dft_frames.push_back(FFT(frames[i]));
	}

	std::cout << "DFT Calculated." << std::endl;

	std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;

	return dft_frames;
}

int main()
{

	SNDFILE* file;
	SF_INFO info;
	memset(&info, 0, sizeof(info));

	file = sf_open("Overkill.wav", SFM_READ, &info);
	if (!file) {
		std::cerr << "Error opening file." << std::endl;
		return 1;
	}

	std::vector<short> buffer(info.frames * info.channels);
	sf_count_t num_frames = sf_readf_short(file, buffer.data(), info.frames);


	std::cout << "Frames: " << info.frames << std::endl;
	std::cout << "Sample Rate: " << info.samplerate << std::endl;
	std::cout << "Channels: " << info.channels << std::endl;
	std::cout << "Duration: " << static_cast<double>(num_frames) / info.samplerate << " seconds." << std::endl;

	int sampleRate = info.samplerate;
	int fps = 30;
	int fft_size = 2048 * 1;

	std::cout << "Buffer Size: " << buffer.size() << std::endl;

	//for (int i = 0; i < buffer.size(); i++) {
	//    // Process each sample
	//    // Example: Print the sample value
	//    std::cout << "Sample " << i << ": " << buffer[i] << std::endl;
	//}

	
	std::cout << "Framing Audio..." << std::endl;

	std::vector<std::vector<float>> frames = FrameAudio(buffer, fps, fft_size, sampleRate, info.channels);

	std::cout << "Audio Framed." << std::endl;

	//std::cout << "Frames: " << frames.size() << std::endl;
	//std::cout << "Frame Size: " << frames[0].size() << std::endl;

	std::vector<std::vector<std::array<float, 2>>> dft_frames = DFTFrames(frames);

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

	sf_close(file);
}





// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
