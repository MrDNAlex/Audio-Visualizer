#include "Visualizer.cuh"

__global__ void VisualizeFrameGPU(RectInfo* rects, int* numOfRects, int* width, int* height, unsigned char* frame)
{
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int N = *numOfRects;
	int imgWidth = *width;
	int imgHeight = *height;

	if (xIndex >= imgWidth || yIndex >= imgHeight) return;

	bool drawn = false;

	for (int i = 0; i < N; i++)
	{
		RectInfo rect = rects[i];

		if (xIndex >= rect.xPos && xIndex < rect.xPos + rect.width && yIndex >= rect.yPos && yIndex < rect.yPos + rect.height)
		{
			int index = (yIndex * imgWidth + xIndex) * 3;

			//JPG = red, green, blue
			frame[index] = rect.red; //Color the pixel
			frame[index + 1] = rect.green; //Color the pixel
			frame[index + 2] = rect.blue; //Color the pixel

			drawn = true;
		}
	}

	if (!drawn)
	{
		int index = (yIndex * imgWidth + xIndex) * 3;

		//JPG = red, green, blue
		frame[index] = 0; //Color the pixel
		frame[index + 1] = 0; //Color the pixel
		frame[index + 2] = 0; //Color the pixel
	}
}

unsigned char* VisualizeFrame(RectInfo* rects, int numOfRects, int frameIndex)
{
	int width = 1920;
	int height = 1080;
	int frameSize = width * height * 3;

	RectInfo* gpuRects = 0;
	int* gpuNumOfRects = 0;
	int* gpuWidth = 0;
	int* gpuHeight = 0;
	unsigned char* gpuFrame = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	cudaStatus = AssignVariable((void**)&gpuNumOfRects, &numOfRects, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuWidth, &width, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuHeight, &height, sizeof(int));

	cudaStatus = AssignVariable((void**)&gpuRects, rects, sizeof(RectInfo), numOfRects);
	cudaStatus = AssignMemory((void**)&gpuFrame, sizeof(unsigned char), frameSize);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	VisualizeFrameGPU << <numBlocks, threadsPerBlock >> > (gpuRects, gpuNumOfRects, gpuWidth, gpuHeight, gpuFrame);

	cudaStatus = cudaDeviceSynchronize();

	unsigned char* frame = new unsigned char[frameSize];

	cudaStatus = GetVariable(frame, gpuFrame, sizeof(unsigned char), frameSize);

	cudaFree(gpuRects);
	cudaFree(gpuNumOfRects);
	cudaFree(gpuWidth);
	cudaFree(gpuHeight);
	cudaFree(gpuFrame);

	return frame;
}


