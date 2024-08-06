//__global__ void FFTGPU(float* input, int* size, int* index, FourierData* output)
//{
//	int GPUIndex = *index;
//	int N = *size;
//	int outputIndex = N * GPUIndex;
//	const float pi = 3.14159265358979323846;
//	
//	if (N <= 1)
//	{
//		output = new FourierData[1]{ {input[outputIndex], 0.0f} };
//		return;
//	}
//		
//	float* even = new float[N / 2];
//	float* odd = new float[N / 2];
//
//	for (int i = 0; i < N; i += 2)
//	{
//		even[i] = input[outputIndex + i];
//		odd[i] = input[outputIndex + i + 1];
//	}
//
//	int halfSize = N / 2;
//
//	FourierData* evenResult = new FourierData[halfSize];
//	FourierData* oddResult = new FourierData[halfSize];
//
//	FFTGPU(even, size, index, evenResult);
//	FFTGPU(odd, size, index, oddResult);
//
//	FourierData* signal = new FourierData[N];
//
//	for (int k = 0; k < N / 2; k++)
//	{
//		float angle = -2 * pi * k / N;
//
//		float real_odd = cosf(angle) * oddResult[k].real + sinf(angle) * oddResult[k].imag;
//		float imag_odd = -sinf(angle) * oddResult[k].real + cosf(angle) * oddResult[k].imag;
//
//
//		signal[k].real = evenResult[k].real + real_odd;
//		signal[k].imag = evenResult[k].imag + imag_odd;
//
//		signal[k + N / 2].real = evenResult[k].real - real_odd;
//		signal[k + N / 2].imag = evenResult[k].imag - imag_odd;
//	}
//
//	delete evenResult;
//	delete oddResult;
//}

//__global__ void FourierTransformGPU(float* input, FourierData* output, int* fft_size, int* numOfFrames)
//{
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int N = *fft_size;
//	int outputIndex = N * index;
//	const float pi = 3.14159265358979323846;
//
//	FourierData* signal = new FourierData[N];
//
//	FFTGPU(input, fft_size, &index, signal);
//
//}

//#include <nvjpeg.h>

//int SaveJPEG(const char* outputFilename, int width, int height, unsigned char* hostImageData) {
//
//	nvjpegHandle_t handle;
//	nvjpegEncoderState_t state;
//	nvjpegEncoderParams_t params;
//	nvjpegStatus_t status;
//
//	// Initialize nvJPEG components
//	status = nvjpegCreateSimple(&handle);
//	if (status != NVJPEG_STATUS_SUCCESS) {
//		std::cerr << "Failed to create nvJPEG handle" << std::endl;
//		return -1;
//	}
//
//	status = nvjpegEncoderStateCreate(handle, &state, NULL);
//	if (status != NVJPEG_STATUS_SUCCESS) {
//		std::cerr << "Failed to create encoder state" << std::endl;
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	status = nvjpegEncoderParamsCreate(handle, &params, NULL);
//	if (status != NVJPEG_STATUS_SUCCESS) {
//		std::cerr << "Failed to create encoder parameters" << std::endl;
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	// Use provided image data instead of allocating new memory
//	if (!hostImageData) {
//		std::cerr << "Invalid image data provided" << std::endl;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	nvjpegImage_t src;
//	src.channel[0] = hostImageData;       // R channel
//	src.channel[1] = hostImageData + 1;   // G channel
//	src.channel[2] = hostImageData + 2;   // B channel
//	src.pitch[0] = 3 * width;             // Pitch for R channel
//	src.pitch[1] = 3 * width;             // Pitch for G channel
//	src.pitch[2] = 3 * width;             // Pitch for B channel
//
//	// Encoding the image
//	status = nvjpegEncodeImage(handle, state, params, &src, NVJPEG_INPUT_RGB, width, height, NULL);
//	if (status != NVJPEG_STATUS_SUCCESS) {
//		std::cerr << "Failed to encode image" << std::endl;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	size_t length;
//	unsigned char* jpegStream = NULL;
//	status = nvjpegEncodeRetrieveBitstream(handle, state, NULL, &length, NULL);
//	if (status != NVJPEG_STATUS_SUCCESS || length == 0) {
//		std::cerr << "Failed to retrieve bitstream length or length is zero" << std::endl;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	jpegStream = new unsigned char[length];
//	if (!jpegStream) {
//		std::cerr << "Failed to allocate memory for JPEG stream" << std::endl;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	status = nvjpegEncodeRetrieveBitstream(handle, state, jpegStream, &length, NULL);
//	if (status != NVJPEG_STATUS_SUCCESS) {
//		std::cerr << "Failed to retrieve JPEG stream" << std::endl;
//		delete[] jpegStream;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	// Writing to file
//	std::ofstream outFile(outputFilename, std::ios::out | std::ios::binary);
//	if (!outFile) {
//		std::cerr << "Failed to open file for writing" << std::endl;
//		delete[] jpegStream;
//		nvjpegEncoderParamsDestroy(params);
//		nvjpegEncoderStateDestroy(state);
//		nvjpegDestroy(handle);
//		return -1;
//	}
//
//	outFile.write(reinterpret_cast<char*>(jpegStream), length);
//	outFile.close();
//
//	// Cleanup
//	delete[] jpegStream;
//	nvjpegEncoderParamsDestroy(params);
//	nvjpegEncoderStateDestroy(state);
//	nvjpegDestroy(handle);
//
//	return 0; // Success
//}


	// Save to PNG using lodepng
	//unsigned error = lodepng_encode32_file(filename, frame, width, height);



//BMP = blue, green, red
			//frame[index] = rect.blue; //Color the pixel
			//frame[index + 1] = rect.green; //Color the pixel
			//frame[index + 2] = rect.red; //Color the pixel
			//frame[index + 3] = rect.alpha;


			//PNG = red, green, blue
//frame[index] = rect.red; //Color the pixel
//frame[index + 1] = rect.green; //Color the pixel
//frame[index + 2] = rect.blue; //Color the pixel
//frame[index + 3] = rect.alpha;

//JPG = red, green, blue
//frame[index] = rect.red; //Color the pixel
//frame[index + 1] = rect.green; //Color the pixel
//frame[index + 2] = rect.blue; //Color the pixel
//frame[index + 3] = rect.alpha;
