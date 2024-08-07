//std::vector<std::vector<std::array<float, 2>>> oldMethod()
//{
//	std::vector<std::vector<float>> frames = FrameAudio(audioBuffer, fps, dftSize, sample_rate, info.channels);
//	std::vector<std::vector<std::array<float, 2>>> dft_frames = DFTFrames(frames);
//	std::vector<std::vector<std::array<float, 2>>> dft_frames_nyquist = ExtractNyquist(dft_frames);
//
//	std::cout << "Classic DFT Calculated." << std::endl;
//
//	return dft_frames_nyquist;
//}
//
//std::vector<std::vector<std::array<float, 2>>> ExtractNyquist(std::vector<std::vector<std::array<float, 2>>> dft_frames)
//{
//	std::vector<std::vector<std::array<float, 2>>> dft_frames_nyquist;
//
//	std::cout << "Extracting Nyquist DFT..." << std::endl;
//	for (int i = 0; i < dft_frames.size(); i++) {
//		std::vector<std::array<float, 2>> dft_frame_nyquist;
//		for (int j = 0; j < (dft_frames[i].size() / 2) + 1; j++) {
//			dft_frame_nyquist.push_back(dft_frames[i][j]);
//		}
//		dft_frames_nyquist.push_back(dft_frame_nyquist);
//	}
//	std::cout << "Exracted Nyquist DFT" << std::endl;
//
//	return dft_frames_nyquist;
//}
//
//
//std::vector<std::vector<float>> FrameAudio(std::vector<short> audioSignal, int fps = 24, int fft_size = 2048, int sample_rate = 22050, int channels = 1)
//{
//	std::cout << "Framing Audio..." << std::endl;
//
//	std::vector<std::vector<float>> frames;
//
//	int frameSize = (sample_rate * channels) / fps; // This calculates how many samples per frame based on fps
//
//	// Calculating the number of frames we can fit into the audioSignal with the given frameSize
//	int numFrames = ((audioSignal.size() - fft_size) / frameSize);
//
//	for (int i = 0; i < numFrames; i++) {
//		std::vector<float> frame(fft_size);
//		for (int j = 0; j < fft_size; j++) {
//			// Normalizing the short sample to float in the range [-1.0, 1.0]
//			frame[j] = audioSignal[i * frameSize + j] / 32768.0f;
//		}
//		frames.push_back(frame);
//	}
//
//	std::cout << "Audio Framed." << std::endl;
//
//	return frames;
//}
//
//
//std::vector<std::array<float, 2>> DFT(std::vector<float> frame)
//{
//	int N = frame.size();
//	std::vector<std::array<float, 2>> X(N);
//	for (int k = 0; k < N; k++) {
//		float real = 0.0f;
//		float imag = 0.0f;
//		for (int n = 0; n < N; n++) {
//
//			float angle = 2 * 3.141 * k * n / N;
//
//			real += frame[n] * cosf(angle);
//			imag += frame[n] * sinf(angle);
//		}
//		X[k][0] = real;
//		X[k][1] = imag;
//	}
//	return X;
//}
//
//std::vector<std::array<float, 2>> FFT(std::vector<float> frame)
//{
//
//	const float pi = 3.14159265358979323846;
//
//	int N = frame.size();
//
//	if (N <= 1) {
//		return std::vector<std::array<float, 2>>(1, { frame[0], 0.0f });
//	}
//
//	//Compute Even signals
//	std::vector<float> frame_even;
//	std::vector<float> frame_odd;
//
//	for (int i = 0; i < N; i += 2) {
//		frame_even.push_back(frame[i]);
//		frame_odd.push_back(frame[i + 1]);
//	}
//
//	std::vector<std::array<float, 2>> even = FFT(frame_even);
//	std::vector<std::array<float, 2>> odd = FFT(frame_odd);
//
//	std::vector<float> real(N);
//	std::vector<float> imag(N);
//
//	for (int k = 0; k < N / 2; k++) {
//		float angle = -2 * pi * k / N;
//		float real_odd = cosf(angle) * odd[k][0] + sinf(angle) * odd[k][1];
//		float imag_odd = -sinf(angle) * odd[k][0] + cosf(angle) * odd[k][1];
//
//		real[k] = even[k][0] + real_odd;
//		imag[k] = even[k][1] + imag_odd;
//
//		real[k + N / 2] = even[k][0] - real_odd;
//		imag[k + N / 2] = even[k][1] - imag_odd;
//	}
//
//	std::vector<std::array<float, 2>> X(N);
//
//	for (int i = 0; i < real.size(); i++) {
//		X[i][0] = real[i];
//		X[i][1] = imag[i];
//	}
//
//	return X;
//}
//
//
//std::vector<std::vector<std::array<float, 2>>> DFTFrames(std::vector<std::vector<float>> frames)
//{
//	std::vector<std::vector<std::array<float, 2>>> dft_frames;
//
//	std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//	std::cout << "Calculating DFT..." << std::endl;
//
//	for (int i = 0; i < frames.size(); i++) {
//		if (i % 100 == 0) {
//			std::cout << ((float)i) / ((float)frames.size()) << "% Complete" << std::endl;
//		}
//		dft_frames.push_back(FFT(frames[i]));
//	}
//
//	std::cout << "DFT Calculated." << std::endl;
//
//	std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
//
//	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << std::endl;
//
//	return dft_frames;
//}