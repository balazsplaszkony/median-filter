#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include <chrono>
#include <ctime>

#include <math.h>

#include <omp.h>

#include <IL/ilut.h>
#include <IL/ilu.h>

#include "timestamp.h"


#ifdef _MSC_VER
	#define memalign(a, s) _aligned_malloc((s), (a))
	#define memfree(a) _aligned_free((a))
#else
	#define memfree(a) free((a))
#endif



#define FILTER_W 5
#define FILTER_H 5

 void median_img_scalar(int imgHeight, int imgWidth, int imgWidthF,
 			   uint8_t *imgSrcExt, uint8_t *imgDst);

void median_img_avx(int imgHeight, int imgWidth, int imgWidthF,
			   uint8_t *imgSrcExt, uint8_t *imgDst);

void median_img_avx_optimized(int imgHeight, int imgWidth, int imgWidthF,
			   uint8_t *imgSrcExt, uint8_t *imgDst);

// Function to compare two images byte-by-byte
int compare_with_ref(const char* generated_file, const char* reference_file) {
	 ILuint imgGen, imgRef;

	     // Load generated image
	     ilGenImages(1, &imgGen);
	     ilBindImage(imgGen);
	     if (!ilLoadImage(generated_file)) {
	         printf("Error loading generated image: %s\n", generated_file);
	         return -1;
	     }

	     // Get data from the generated image
	     int widthGen = ilGetInteger(IL_IMAGE_WIDTH);
	     int heightGen = ilGetInteger(IL_IMAGE_HEIGHT);
	     int formatGen = ilGetInteger(IL_IMAGE_FORMAT);
	     ILubyte* dataGen = ilGetData();  // Get data for generated image before switching

	     // Load reference image
	     ilGenImages(1, &imgRef);
	     ilBindImage(imgRef);  // Switch to the reference image
	     if (!ilLoadImage(reference_file)) {
	         printf("Error loading reference image: %s\n", reference_file);
	         return -1;
	     }

	     // Get data from the reference image
	     int widthRef = ilGetInteger(IL_IMAGE_WIDTH);
	     int heightRef = ilGetInteger(IL_IMAGE_HEIGHT);
	     int formatRef = ilGetInteger(IL_IMAGE_FORMAT);
	     ILubyte* dataRef = ilGetData();  // Get data for reference image

	     // Ensure both images have the same dimensions and format
	     if (widthGen != widthRef || heightGen != heightRef) {
	         printf("Images have different dimensions (Generated: %dx%d, Reference: %dx%d).\n", widthGen, heightGen, widthRef, heightRef);
	         return -1;
	     }

	     if (formatGen != formatRef) {
	         printf("Images have different formats (Generated: %d, Reference: %d).\n", formatGen, formatRef);
	         return -1;
	     }
	     int ret = 0;
	    int size = widthGen * heightGen * ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);  // Ensure correct byte size
	    for (int i = 0; i < size; i++) {
	        if (dataGen[i] != dataRef[i]) {
	            printf("Images differ at byte %d (Generated: 0x%02x, Reference: 0x%02x).\n", i, dataGen[i], dataRef[i]);
	            ret = -1; break;
	        }
	        if (i % 1000000 == 0)
	        {
	        	//printf("generated %d reference: %d).\n", dataGen[i], dataRef[i]);
	        }
	        if (i < 200 || (i >= 4922*3 && i < 200 + 3*(4922)))
	        {
	            //printf("byte %d (Generated: 0x%02x, Reference: 0x%02x).\n", i, dataGen[i], dataRef[i]);
	        }

	    }
	    if (ret == 0)
	    	printf("Images are identical.\n");

	    ilDeleteImages(1, &imgGen);
	    ilDeleteImages(1, &imgRef);

	    return 0;
	}

int main(int argc, char *argv[])
{

	char *dst_fname;
	char *dst_ext;
	int runs[2];
	if (argc != 4) {
		printf("Usage:\nfir-img 'input filename' 'output filename' 'vector runs'\nExiting.\n");
		return -1;
	}
	else {
		int last_dot_pos;
		last_dot_pos = strrchr(argv[2], '.') - argv[2];

		dst_ext = (char*)malloc(strlen(argv[2])-last_dot_pos+1);
		dst_fname = (char*)malloc(strlen(argv[2])-last_dot_pos+1);

		strcpy(dst_ext, argv[2] + last_dot_pos + 1);
		*(argv[2]+last_dot_pos) = '\0';
		strcpy(dst_fname, argv[2]);

		printf("Input file: %s\n", argv[1]);
		printf("Output file: %s_[].%s\n", dst_fname, dst_ext);

		runs[0] = strtol(argv[3], NULL, 10);
	}

	///////////////////////////////////////////
	// Load input image
	ilInit(); iluInit();
	ILboolean ret;
	ILuint ilImg = 0;
	ilGenImages(1, &ilImg);
	ilBindImage(ilImg);
	ret = ilLoadImage((ILconst_string)(argv[1]));
	if (!ret) {
		printf("Error opening input image, exiting.\n");
		return -1;
	}
	ILubyte* imgData = ilGetData();

	int imgWidth = ilGetInteger(IL_IMAGE_WIDTH);
	int imgHeight = ilGetInteger(IL_IMAGE_HEIGHT);
	ILint imgOrigin = ilGetInteger(IL_ORIGIN_MODE);

	printf("Input resolution: %dx%d\n", imgWidth, imgHeight);

	///////////////////////////////////////////
	// Extend input image with zeros
	uint8_t* imgSrcExt;
	int imgWidthF = imgWidth + FILTER_W - 1;
	int imgHeightF = imgHeight + FILTER_H - 1;
	int imgFOfssetW = (FILTER_W - 1) / 2;
	int imgFOfssetH = (FILTER_H - 1) / 2;

	int buff_size_src = 3 * imgWidthF * imgHeightF * sizeof(uint8_t);
	int buff_size_dst = 3 * imgWidth * imgHeight * sizeof(uint8_t);

	imgSrcExt = (uint8_t*)(memalign(4096, buff_size_src));
	int row, col;
	for (row = 0; row < imgHeightF; row++)
	{
		for (col = 0; col < imgWidthF; col++)
		{
			int pixel = (row * imgWidthF + col) * 3;
			*(imgSrcExt + pixel + 0) = 0;
			*(imgSrcExt + pixel + 1) = 0;
			*(imgSrcExt + pixel + 2) = 0;
		}
	}

	for (row = 0; row < imgHeight; row++)
	{
		for (col = 0; col < imgWidth; col++)
		{
			int pixel_dst = ((row + imgFOfssetH) * imgWidthF + (col + imgFOfssetW)) * 3;
			int pixel_src = (row * imgWidth + col) * 3;
			*(imgSrcExt + pixel_dst + 0) = (uint8_t)(*(imgData + pixel_src + 0));
			*(imgSrcExt + pixel_dst + 1) = (uint8_t)(*(imgData + pixel_src + 1));
			*(imgSrcExt + pixel_dst + 2) = (uint8_t)(*(imgData + pixel_src + 2));
		}
	}
	printf("imgSrcExt generated\n");

	uint8_t* imgRes[3];
	imgRes[0] = (uint8_t*)(memalign(4096, buff_size_dst));
	imgRes[1] = (uint8_t*)(memalign(4096, buff_size_dst));
	imgRes[2] = (uint8_t*)(memalign(4096, buff_size_dst));


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	double ts_start, ts_end, elapsed, perf;

#if 1
	ts_start = get_ts_ns();

	median_img_scalar(imgHeight,
	          imgWidth,
		      imgWidthF,
			  imgSrcExt,
			  imgRes[0]);
		ts_end = get_ts_ns();
		elapsed = (ts_end - ts_start);
		perf = double(imgHeight*imgWidth)*1000.0/elapsed;

	printf("CPU scalar time: %f ms, Mpixel/s: %f\n", (elapsed/1000000.0), perf);
#endif

#if 1
	ts_start = get_ts_ns();
	if (imgHeight % 2 == 0 && (imgWidth * 3) % 32 == 0)
	{
		for (int run=0; run<runs[0]; run++)
		{
			median_img_avx_optimized(imgHeight,
					imgWidth,
					imgWidthF,
					imgSrcExt,
					imgRes[1]);
		}
	}
	else
	{
		for (int run=0; run<runs[0]; run++)
		{
			median_img_avx(imgHeight,
					imgWidth,
					imgWidthF,
					imgSrcExt,
					imgRes[1]);
		}
	}

	ts_end = get_ts_ns();
	elapsed = (ts_end - ts_start)/double(runs[0]);

	perf = double(imgHeight*imgWidth)*1000.0/elapsed;
	printf("CPU AVX time: %f ms, Mpixel/s: %f\n", (elapsed/1000000.0), perf);
#endif

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Write output images
	for (int i = 0; i < 2; i++)
	{
		for (row = 0; row < imgHeight; row++)
		{
			for (col = 0; col < imgWidth; col++)
			{
				int pixel_src = (row * imgWidth + col) * 3;
				int pixel_dst = (row * imgWidth + col) * 3;
				*(imgData + pixel_dst + 0) = (ILubyte)(*(imgRes[i] + pixel_src + 0));
				*(imgData + pixel_dst + 1) = (ILubyte)(*(imgRes[i] + pixel_src + 1));
				*(imgData + pixel_dst + 2) = (ILubyte)(*(imgRes[i] + pixel_src + 2));
			}
		}

		ret = ilSetData(imgData);
		ilEnable(IL_FILE_OVERWRITE);
		char dst_file[100];
		sprintf(dst_file, "%s_%d.%s", dst_fname, i, dst_ext);
		ilSaveImage((ILconst_string)(dst_file));
	}
	ilDeleteImages(1, &ilImg);

	memfree(imgSrcExt);
	memfree(imgRes[0]);
	memfree(imgRes[1]);
	memfree(imgRes[2]);


	// Compare the generated output image with a reference image
	const char* reference_file = "output_ref.bmp";  // Set the path of the reference image here
	char generated_file[100];
	sprintf(generated_file, "%s_1.%s", dst_fname, dst_ext);  // The generated output file for comparison

	// Call the comparison function
	int result = compare_with_ref(generated_file, reference_file);
	if (result == 0) {
		printf("Generated image matches the reference image.\n");
	} else {
		printf("Generated image does not match the reference image.\n");
	}


	free(dst_ext);
	free(dst_fname);
	printf("fir-img done.\n");
	return 0;

}
