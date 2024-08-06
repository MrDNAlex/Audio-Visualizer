#pragma once
#include <iostream>
#include <fstream>

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

void GenerateBitmapImage(unsigned char* image, int height, int width, const char* imageFileName) {
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
