#include <stdio.h>
#include <stdlib.h>
#include "stdint.h"
#include <omp.h>
#include "timestamp.h"

#define FILTER_H 5
#define FILTER_W 5


uint8_t temp;

// Macro for comparison and swap of 2 values.
// Implemented with 2 ternary operations, to help the compiler generate branchless code with using cmov.
// I also experimented with using bit manipulating, but ultimately it turned out to be somewhat slower.
#define COMP_SWAP(a, b) \
    temp = a; \
    a = (a > b) ? b : a; \
    b = (a == b) ? temp : b;


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


__attribute__((optimize("no-tree-vectorize"), hot))
uint8_t medianOfTwo(uint8_t arr1[5], uint8_t arr2[6]) {
    int low = 0, high = 5;

    while (low <= high)
    {
        int mid1 = (low + high) / 2;
        int mid2 = (5 + 6 + 1) / 2 - mid1;

        // Elements to the left and right of the partition in arr1.
        uint8_t l1 = (mid1 == 0) ? 0x00 : arr1[mid1 - 1];
        uint8_t r1 = (mid1 == 5) ? 0xff : arr1[mid1];

        // Elements to the left and right of the partition in arr2.
        uint8_t l2 = (mid2 == 0) ? 0x00 : arr2[mid2 - 1];
        uint8_t r2 = (mid2 == 6) ? 0xff : arr2[mid2];

        // Check if it's a valid partition.
        if (l1 <= r2 && l2 <= r1)
        {
        	return MAX(l1, l2);
        }

        // Adjust the binary search bounds
        if (l1 > r2)
        {
            high = mid1 - 1;
        }
        else
        {
            low = mid1 + 1;
        }
    }

    return 0;  // This line should never be reached.
}

__attribute__((optimize("no-tree-vectorize"), hot))
void median_img_scalar(int imgHeight, int imgWidth, int imgWidthF, uint8_t *imgSrcExt, uint8_t *imgDst)
{
	// Every thread computes a different row of the picture, so each is independent,
	//no need for further synchronization.
#pragma omp parallel for
	for (int row = 0; row < imgHeight; row++)
    {
        uint8_t window[3][25];
        uint8_t common_pixels[3][20];
        uint8_t possible_common_median_values[3][6];
        uint8_t first_pixels_unique_col[3][5];
        uint8_t second_pixels_unique_col[3][5];

        for (int col = 0; col < imgWidth; col ++)
		{
            for (int fy = 0; fy < FILTER_H; fy++)
            {
				// Read the pixels needed for the calcualtion of 2 pixels' median values
				// at the same time, if it is not the end of the line.
                if (col < (imgWidth -1))
                {
                    for (int fx = 0; fx < FILTER_W + 1; fx++)
                    {
                        // Unique column of the first pixel
                        if (fx == 0)
                        {
                            first_pixels_unique_col[0][fy] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3];
                            first_pixels_unique_col[1][fy] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 1];
                            first_pixels_unique_col[2][fy] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 2];
                        }

                        // Common columns
                        else if (fx != FILTER_W)
                        {
                            common_pixels[0][fy*4 + fx - 1] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 ];
                            common_pixels[1][fy*4 + fx - 1] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 1];
                            common_pixels[2][fy*4 + fx - 1] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 2];
                        }

                        // Unique column of the second pixel
                        else if (fx == FILTER_W)
                        {
                            second_pixels_unique_col[0][(fy)] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3];
                            second_pixels_unique_col[1][(fy)] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 1];
                            second_pixels_unique_col[2][(fy)] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 2];
                        }

                    }
                }
				// Read a 5x5 window if it is the end of the line.
                else
                {
                    for (int fx = 0; fx < FILTER_W ; fx++)
                    {
                        window[0][fy * FILTER_W + fx] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 ];
                        window[1][fy * FILTER_W + fx] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 1];
                        window[2][fy * FILTER_W + fx] = imgSrcExt[((row + fy) * imgWidthF + col + fx) * 3 + 2];
                    }
                }

            }

			// Calculate the median for 2 values at the same time, if it is not the end of the line
            if (col < (imgWidth -1))
            {
            	// Batcher's Odd-Even Mergesort for the 20 common pixels
				// Program used to generate the comparisons for n = 20 array
//				int n = 20;
//				int p;
//				int logp;
//				for (p = 1, logp = 0; p < n; p <<= 1, logp++) {
//					for (int k = p; k >= 1; k >>= 1) {
//						for (int j = k % p; j <= (n - 1 - k); j += 2 * k) {
//							for (int i = 0; i <= (k - 1 < n - j - k - 1 ? k - 1 : n - j - k - 1); i++) {
//								if (((i + j) >> (logp + 1)) == ((i + j + k) >> (logp + 1)))
//								{
//									for(int idx = 0; idx < 3; idx++)
//									{
//										// Print the comparisons for all colors
//										printf("COMP_SWAP(common_pixels[%d][%d], common_pixels[%d][%d]);\n", idx, (i+j), idx, (i+j+k) );
//									}
//								}
//							}
//						}
//					}
//				}
//				if(col < 20 && row == 0)
//				{
//					printf("byte %d: value: %d\n", (col*3)/2 + 0, common_pixels[0][9]);
//					printf("byte %d: value: %d\n", (col*3)/2 + 1, common_pixels[1][9]);
//					printf("byte %d: value: %d\n", (col*3)/2 + 2, common_pixels[2][9]);
//
//				}
				#include "comp_swap_20.txt"

				// Copy middle elements of the sorted common pixel arrays, to a new array.
                possible_common_median_values[0][0] = common_pixels[0][7];
                possible_common_median_values[1][0] = common_pixels[1][7];
                possible_common_median_values[2][0] = common_pixels[2][7];
                possible_common_median_values[0][1] = common_pixels[0][7 + 1];
                possible_common_median_values[1][1] = common_pixels[1][7 + 1];
                possible_common_median_values[2][1] = common_pixels[2][7 + 1];
                possible_common_median_values[0][2] = common_pixels[0][7 + 2];
                possible_common_median_values[1][2] = common_pixels[1][7 + 2];
                possible_common_median_values[2][2] = common_pixels[2][7 + 2];
                possible_common_median_values[0][3] = common_pixels[0][7 + 3];
                possible_common_median_values[1][3] = common_pixels[1][7 + 3];
                possible_common_median_values[2][3] = common_pixels[2][7 + 3];
                possible_common_median_values[0][4] = common_pixels[0][7 + 4];
                possible_common_median_values[1][4] = common_pixels[1][7 + 4];
                possible_common_median_values[2][4] = common_pixels[2][7 + 4];
                possible_common_median_values[0][5] = common_pixels[0][7 + 5];
                possible_common_median_values[1][5] = common_pixels[1][7 + 5];
                possible_common_median_values[2][5] = common_pixels[2][7 + 5];




            	// Batcher's Odd-Even Mergesort for the 5 common pixels

				#include "comp_swap_5.txt"

                uint8_t median[3][2];
				// Finding the median of all 6 values.
				// For each value, the median is the median of the middle values
				// in the sorted array of thecommon pixels, and the unique column.
                median[0][0] = medianOfTwo(first_pixels_unique_col[0], possible_common_median_values[0]);
                median[1][0] = medianOfTwo(first_pixels_unique_col[1], possible_common_median_values[1]);
                median[2][0] = medianOfTwo(first_pixels_unique_col[2], possible_common_median_values[2]);
                median[0][1] = medianOfTwo(second_pixels_unique_col[0], possible_common_median_values[0]);
                median[1][1] = medianOfTwo(second_pixels_unique_col[1], possible_common_median_values[1]);
                median[2][1] = medianOfTwo(second_pixels_unique_col[2], possible_common_median_values[2]);

				// Writing first pixel
                imgDst[(row * imgWidth + col)*3    ] = median[0][0];
                imgDst[(row * imgWidth + col)*3 + 1] = median[1][0];
                imgDst[(row * imgWidth + col)*3 + 2] = median[2][0];

				// Writing second pixel
                imgDst[(row * imgWidth + col + 1)*3 + 0] = median[0][1];
                imgDst[(row * imgWidth + col + 1)*3 + 1] = median[1][1];
                imgDst[(row * imgWidth + col + 1)*3 + 2] = median[2][1];

				// Increment col counter, if it is safe (there are at least 2 more pixels in the row)
                if (col < (imgWidth - 2))
                	col++;
            }
            else
            {

                // Batcher's Odd-Even Mergesort for 25 pixels, not unrolled  demonstration
				// This branch is only executed maximum once for every row, so unrolling it would not cause a noticeable speed up,
				// but would increase the size of the compiled binary significantly.

                int p, logp; // keeping record of logp to avoid division
                for (p = 1, logp = 0; p < 25; p <<= 1, logp++)
                {
                    for (int k = p; k >= 1; k >>= 1)
                    {
                        for (int j = k % p; j <= (25 - 1 - k); j += (k << 1))
                        {
                            for (int i = 0; i <= (k - 1 < 25 - j - k - 1 ? k - 1 : 25 - j - k - 1); i++)
                            {
								// bitshifting with logp+1 instead of division with 2*p
								// using the fact, that p is a power of 2
								// working with integers, so floor() is automatically implied
                                if (((i + j) >> (logp + 1)) == ((i + j + k) >> (logp + 1)))
                                {
                                	COMP_SWAP(window[0][i + j], window[0][i + j + k]);
                            		COMP_SWAP(window[1][i + j], window[1][i + j + k]);
                            		COMP_SWAP(window[2][i + j], window[2][i + j + k]);
                                }
                            }
                        }
                    }
                }

                imgDst[(row * imgWidth + col)*3    ] = window[0][12];
                imgDst[(row * imgWidth + col)*3 + 1] = window[1][12];
                imgDst[(row * imgWidth + col)*3 + 2] = window[2][12];
            }
		}
	}
}
