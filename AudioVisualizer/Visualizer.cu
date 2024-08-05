#include "Visualizer.cuh"

#include <fstream>
#include <vector>
#include <cstdint>

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

			//BMP = blue, green, red

			frame[index] = rect.blue; //Color the pixel
			frame[index + 1] = rect.green; //Color the pixel
			frame[index + 2] = rect.red; //Color the pixel
			//frame[index + 3] = rect.alpha;

			drawn = true;
		}
	}

	if (!drawn)
	{
		int index = (yIndex * imgWidth + xIndex) * 3;

		frame[index] = 0; //Color the pixel
		frame[index + 1] = 0; //Color the pixel
		frame[index + 2] = 0; //Color the pixel
		//frame[index + 3] = 0;
	}
}

cudaError_t VisualizeFrame(RectInfo* rects, int numOfRects, int frameIndex)
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

	char filename[100];

	// Format the filename with the index
	sprintf(filename, "C:\\Users\\MrDNA\\Downloads\\Frames\\frame%d.bmp", frameIndex);

	// Save to PNG using lodepng
	//unsigned error = lodepng_encode32_file(filename, frame, width, height);

	generateBitmapImage(frame, height, width, filename);


	cudaFree(gpuRects);
	cudaFree(gpuNumOfRects);
	cudaFree(gpuWidth);
	cudaFree(gpuHeight);
	cudaFree(gpuFrame);

	delete[] frame;

	return cudaStatus;
}

struct BMPHeader {
	uint16_t file_type{ 0x4D42 };          // File type always BM which is 0x4D42
	uint32_t file_size{ 0 };               // Size of the file (in bytes)
	uint16_t reserved1{ 0 };               // Reserved, always 0
	uint16_t reserved2{ 0 };               // Reserved, always 0
	uint32_t offset_data{ 0 };             // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
	uint32_t size{ 0 };                    // Size of this header (in bytes)
	int32_t width{ 0 };                    // width of bitmap in pixels
	int32_t height{ 0 };                   // width of bitmap in pixels
	uint16_t planes{ 1 };                  // No. of planes for the target device, this is always 1
	uint16_t bit_count{ 0 };               // No. of bits per pixel
	uint32_t compression{ 0 };             // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
	uint32_t size_image{ 0 };              // 0 - for uncompressed images
	int32_t x_pixels_per_meter{ 0 };
	int32_t y_pixels_per_meter{ 0 };
	uint32_t colors_used{ 0 };             // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
	uint32_t colors_important{ 0 };        // No. of colors used for displaying the bitmap. If 0 all colors are required
};

void generateBitmapImage(unsigned char* image, int height, int width, const char* imageFileName) {
	BMPHeader bmpHeader;
	BMPInfoHeader bmpInfoHeader;
	int bytesPerLine = (width * 3 + 3) & ~3;
	int fileSize = 54 + bytesPerLine * height;

	bmpHeader.file_size = fileSize;
	bmpHeader.offset_data = 54;

	bmpInfoHeader.size = 40;
	bmpInfoHeader.width = width;
	bmpInfoHeader.height = height;
	bmpInfoHeader.bit_count = 24;
	bmpInfoHeader.compression = 0;
	bmpInfoHeader.size_image = bytesPerLine * height;

	std::ofstream file(imageFileName, std::ios::out | std::ios::binary);
	if (!file) {
		throw std::runtime_error("Unable to open the file");
	}

	file.write(reinterpret_cast<const char*>(&bmpHeader.file_type), sizeof(bmpHeader.file_type));
	file.write(reinterpret_cast<const char*>(&bmpHeader.file_size), sizeof(bmpHeader.file_size));
	file.write(reinterpret_cast<const char*>(&bmpHeader.reserved1), sizeof(bmpHeader.reserved1));
	file.write(reinterpret_cast<const char*>(&bmpHeader.reserved2), sizeof(bmpHeader.reserved2));
	file.write(reinterpret_cast<const char*>(&bmpHeader.offset_data), sizeof(bmpHeader.offset_data));

	file.write(reinterpret_cast<const char*>(&bmpInfoHeader), sizeof(bmpInfoHeader));

	for (int i = 0; i < height; ++i) {
		int index = (height - i - 1) * width * 3;
		file.write(reinterpret_cast<const char*>(&image[index]), width * 3);
		// Padding to align the rows on a 4-byte boundary
		unsigned char padding[3] = { 0, 0, 0 };
		file.write(reinterpret_cast<const char*>(padding), (bytesPerLine - width * 3));
	}

	file.close();
}
