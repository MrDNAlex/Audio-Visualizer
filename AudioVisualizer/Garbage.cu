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