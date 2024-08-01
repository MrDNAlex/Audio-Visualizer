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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "FourierTransform.cuh"
#include "SongProcessing.cpp"

int main()
{
	int fps = 30;
	int fft_size = 2048 * 1;

	SongProcessing song = SongProcessing("Overkill.wav", fps, fft_size);

	song.ProcessSignal();

	std::cout << "Framing Audio..." << std::endl;

	song.DebugInfo();

	//song.ProcessSignal();

	////std::vector<std::vector<float>> frames = FrameAudio(buffer, fps, fft_size, sampleRate, info.channels);

	//std::cout << "Audio Framed." << std::endl;


	//std::vector<std::vector<std::array<float, 2>>> dft_frames = DFTFrames(frames);

	//std::vector<std::vector<std::array<float, 2>>> dft_frames_nyquist;

	//std::cout << "Extracting Nyquist DFT..." << std::endl;

	//for (int i = 0; i < dft_frames.size(); i++) {
	//	std::vector<std::array<float, 2>> dft_frame_nyquist;
	//	for (int j = 0; j < (dft_frames[i].size() / 2) + 1; j++) {
	//		dft_frame_nyquist.push_back(dft_frames[i][j]);
	//	}
	//	dft_frames_nyquist.push_back(dft_frame_nyquist);
	//}

	//std::cout << "Exracted Nyquist DFT" << std::endl;



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
