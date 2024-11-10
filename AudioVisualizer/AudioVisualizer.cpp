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
#include "SongProcessing.cpp"


char audioPath[1000];

void clearScreen()
{
#ifdef _WIN32
	system("cls");
	system("clear");
#else
	system("clear");
#endif
}

void debugCLI(int argc, char* argv[])
{
	for (int i = 0; i < argc; i++)
	{
		std::cout << argv[i] << std::endl;
	}
}

void createCache()
{
	system("mkdir AudioVisualizerCache");
	system("mkdir AudioVisualizerCache\\Audio");
	system("mkdir AudioVisualizerCache\\Frames");
}

void convertAudio(char* audioPath[])
{
	char conversionCommand[1000];

	sprintf(conversionCommand, "ffmpeg -i \"%s\" \"AudioVisualizerCache\\Audio\\audio.wav\"", *audioPath);

	system(conversionCommand);

	clearScreen();
}

//Change to extracting the name of the file
// Make a folder named after the song in the Cache
// Make a folder named Frames and Audio in the Cache
// Create the Visualizer, name the output file after the song


int main(int argc, char* argv[])
{
	auto start = std::chrono::high_resolution_clock::now();

	debugCLI(argc, argv);

	sprintf(audioPath, argv[1]);

	createCache();

	char conversionCommand[1000];

	sprintf(conversionCommand, "ffmpeg -i \"%s\" \"AudioVisualizerCache\\Audio\\audio.wav\"", audioPath);

	system(conversionCommand);

	clearScreen();

	sprintf(audioPath, "AudioVisualizerCache\\Audio\\audio.wav");

	int fps = 30;
	int fft_size = 2048 * 8; // 8

	SongProcessing song = SongProcessing(audioPath, fps, fft_size);

	song.debugInfo();

	song.processSignal();

	char ffmpegCommand[1000];

	char framesPath[1000];

	char outputVideoPath[1000];

	sprintf(outputVideoPath, "AudioVisualizerCache\\output.mp4");

	sprintf(framesPath, "AudioVisualizerCache\\Frames\\frame%%d.jpg");

	sprintf(ffmpegCommand, "ffmpeg -framerate %d -i %s -i %s -c:v libx264 -r %d -pix_fmt yuv420p %s", fps, framesPath, audioPath, fps, outputVideoPath);

	printf("Making Video\n");

	system(ffmpegCommand);

	clearScreen();

	system("rm -r AudioVisualizerCache\\Frames");

	auto end = std::chrono::high_resolution_clock::now();

	// Calculate duration in microseconds
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

	std::cout << "Time taken by function: " << duration << " seconds" << std::endl;
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
