#include "stdint.h"

#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"
#include "stdio.h"

#define FILTER_H 5
#define FILTER_W 5

#define CALCULATE_2_PIXELS_AT_ONCE

__m256i tmp;
// Swap macro using AVX2 intrinsics for sorting 8-bit data
#define COMP_SWAP_AVX(a, b)       \
{                                      \
    tmp = a;                  \
    a = _mm256_min_epu8(a, b);       \
    b = _mm256_max_epu8(tmp, b);      \
}

#pragma GCC optimize ("unroll-loops")

// Optimized function that calculates components from 2 rows at the same time.
// This function produces correct output only if the number of pixels in 1 row is the multiple of 32, and the number of rows is even.
__attribute__((hot))
void median_img_avx_optimized(int imgHeight, int imgWidth, int imgWidthF,
			   uint8_t *imgSrcExt, uint8_t *imgDst)
{

	#pragma omp parallel for
	    for (int row = 0; row < imgHeight; row+=2)
	    {
	        __m256i  common_pixels[20];
			// first 5 vectors contain first pixels unique col,
			// after that 6 vectors for sorted common middle values
			// after that, second pixels unique col
			// and eventually sorted common again
	        __m256i  median_merger[22];

	        // Updating read and write indexes for the current rows
			int wr_base = row*imgWidth*3;
			int rd_base = row*imgWidthF*3;

	        for (int col = 0; col < (imgWidth*3); col += 32)
	        {

	        	// Loading components for the top row
				median_merger[0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 0));
				median_merger[1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 3));
				median_merger[2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 6));
				median_merger[3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 9));
				median_merger[4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 12));

				// Loading common components
				common_pixels[0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 0));
				common_pixels[1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 3));
				common_pixels[2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 6));
				common_pixels[3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 9));
				common_pixels[4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 12));

				common_pixels[5] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 0));
				common_pixels[6] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 3));
				common_pixels[7] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 6));
				common_pixels[8] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 9));
				common_pixels[9] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 12));

				common_pixels[10] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 0));
				common_pixels[11] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 3));
				common_pixels[12] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 6));
				common_pixels[13] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 9));
				common_pixels[14] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 12));

				common_pixels[15] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 0));
				common_pixels[16] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 3));
				common_pixels[17] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 6));
				common_pixels[18] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 9));
				common_pixels[19] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 12));

				// Sorting common components with Batcher's odd-even mergesort
#include "comp_swap_avx_20.txt"

				// Copying middle components from the sorted common array.
				median_merger[5]  = common_pixels[7];
				median_merger[6]  = common_pixels[8];
				median_merger[7]  = common_pixels[9];
				median_merger[8]  = common_pixels[10];
				median_merger[9]  = common_pixels[11];
				median_merger[10] = common_pixels[12];

				// Sorting 5 unique and 6 middle common components for top row
#include "comp_swap_avx_top.txt"

				// Storing output for the top row
				_mm256_storeu_si256((__m256i*)(imgDst + wr_base), median_merger[5]);

				median_merger[11] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 5*imgWidthF*3 + 0));
				median_merger[12] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 5*imgWidthF*3 + 3));
				median_merger[13] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 5*imgWidthF*3 + 6));
				median_merger[14] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 5*imgWidthF*3 + 9));
				median_merger[15] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 5*imgWidthF*3 + 12));

				// Copying middle components from the sorted common array.
				median_merger[16] = common_pixels[7];
				median_merger[17] = common_pixels[8];
				median_merger[18] = common_pixels[9];
				median_merger[19] = common_pixels[10];
				median_merger[20] = common_pixels[11];
				median_merger[21] = common_pixels[12];

				// Sorting 5 unique and 6 middle common components for bottom row
#include "com_swap_avx_bottom.txt"

				// Storing output for the bottom row
				_mm256_storeu_si256((__m256i*)(imgDst + wr_base + imgWidth*3), median_merger[16]);

				// Incrementing read and write indexes
				wr_base = wr_base + 32;
				rd_base = rd_base + 32;
			}
	    }
}


__attribute__((hot))
void median_img_avx(int imgHeight, int imgWidth, int imgWidthF,
			   uint8_t *imgSrcExt, uint8_t *imgDst)
{
#pragma omp parallel for
    for (int row = 0; row < imgHeight; row++)
    {
        __m256i  window[25];

        // Updating read and write indexes for the current rows
		int wr_base = row*imgWidth*3;
		int rd_base = row*imgWidthF*3;

        for (int col = 0; col < (imgWidth*3); col += 32)
        {
    		// Unrolled loading of components.

			window[0 * FILTER_W  + 0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 0));
			window[0 * FILTER_W  + 1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 3));
			window[0 * FILTER_W  + 2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 6));
			window[0 * FILTER_W  + 3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 9));
			window[0 * FILTER_W  + 4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base  + 12));

			window[1 * FILTER_W  + 0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 0));
			window[1 * FILTER_W  + 1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 3));
			window[1 * FILTER_W  + 2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 6));
			window[1 * FILTER_W  + 3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 9));
			window[1 * FILTER_W  + 4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + imgWidthF*3 + 12));

			window[2 * FILTER_W  + 0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 0));
			window[2 * FILTER_W  + 1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 3));
			window[2 * FILTER_W  + 2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 6));
			window[2 * FILTER_W  + 3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 9));
			window[2 * FILTER_W  + 4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 2*imgWidthF*3 + 12));

			window[3 * FILTER_W  + 0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 0));
			window[3 * FILTER_W  + 1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 3));
			window[3 * FILTER_W  + 2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 6));
			window[3 * FILTER_W  + 3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 9));
			window[3 * FILTER_W  + 4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 3*imgWidthF*3 + 12));

			window[4 * FILTER_W  + 0] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 0));
			window[4 * FILTER_W  + 1] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 3));
			window[4 * FILTER_W  + 2] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 6));
			window[4 * FILTER_W  + 3] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 9));
			window[4 * FILTER_W  + 4] = _mm256_loadu_si256((__m256i*)(imgSrcExt + rd_base + 4*imgWidthF*3 + 12));

			// Sorting components with Batcher's odd-even mergesort
#include "comp_swap_avx_25.txt"

			// Storing output
			_mm256_store_si256((__m256i*)(imgDst + wr_base), window[12]);

			// Incrementing read and write indexes
			wr_base = wr_base + 32;
			rd_base = rd_base + 32;
        }
    }
}
