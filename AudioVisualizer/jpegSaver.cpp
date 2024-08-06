//#include "jpegSaver.h"
//
//void SaveJPEG(char* filename, unsigned char* image_buffer, int image_width, int image_height) {
//	struct jpeg_compress_struct cinfo;
//	struct jpeg_error_mgr jerr;
//
//	FILE* outfile;
//	JSAMPROW row_pointer[1];
//	int row_stride;
//
//	cinfo.err = jpeg_std_error(&jerr);
//	jpeg_create_compress(&cinfo);
//	if ((outfile = fopen(filename, "wb")) == NULL) {
//		fprintf(stderr, "can't open %s\n", filename);
//		exit(1);
//	}
//	jpeg_stdio_dest(&cinfo, outfile);
//
//	cinfo.image_width = image_width;
//	cinfo.image_height = image_height;
//	cinfo.input_components = 3; // 3 for RGB
//	cinfo.in_color_space = JCS_RGB;
//
//	jpeg_set_defaults(&cinfo);
//	jpeg_set_quality(&cinfo, 100, TRUE); // 75 is a good default quality
//	jpeg_start_compress(&cinfo, TRUE);
//
//	row_stride = image_width * 3; // assuming each pixel is RGB
//
//	while (cinfo.next_scanline < cinfo.image_height) {
//		row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
//		(void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
//	}
//
//	jpeg_finish_compress(&cinfo);
//	fclose(outfile);
//	jpeg_destroy_compress(&cinfo);
//}