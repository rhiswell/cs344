/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative histogram of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative histogram by following these
  steps.

*/

#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <assert.h>

#define DEBUG_INFO(fmt, args...) { printf("DEBUG: " fmt "\n", ##args); }

inline int NextPow2(int n)
{
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

__device__ __constant__ float kVal;

__global__ void kernelScatter(float *d_buf, int start, int end)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_1D_pos < end)
    d_buf[thread_1D_pos + start] = kVal;
}

__global__ void kernelFindMin(float *d_buf, int stride, int end)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int new_idx = thread_1D_pos * (2 * stride);

  if (new_idx < end) {
    int first = d_buf[new_idx], second = d_buf[new_idx + stride];
    d_buf[new_idx] = min(first, second);
  }
}

__global__ void kernelFindMax(float *d_buf, int stride, int end)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int new_idx = thread_1D_pos * (2 * stride);

  if (new_idx < end) {
    int first = d_buf[new_idx], second = d_buf[new_idx + stride];
    d_buf[new_idx] = max(first, second);
  }
}

__global__ void kernelHistogram(const float * const d_logLuminance,
                                unsigned int *d_histogram,
                                float min_logLum,
                                float lumRange,
                                const size_t numBins,
                                int len)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_1D_pos < len) {
    int bin = (d_logLuminance[thread_1D_pos] - min_logLum) / lumRange * numBins;
    atomicAdd(&d_histogram[bin], 1);
  }
}

__global__ void kernelScanUpsweep(unsigned int *d_hist, int len, int twod)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int twod1 = twod * 2;
  int new_idx = thread_1D_pos * twod1;

  if (new_idx < len)
    d_hist[new_idx + twod1 - 1] += d_hist[new_idx + twod - 1];
}

__global__ void kernelScanDownsweep(unsigned int *d_hist, int len, int twod)
{
  int thread_1D_pos = blockIdx.x * blockDim.x + threadIdx.x;
  int twod1 = twod * 2;
  int new_idx = thread_1D_pos * twod1;

  if (new_idx < len) {
    int buf = d_hist[new_idx + twod - 1];
    d_hist[new_idx + twod - 1] = d_hist[new_idx + twod1 - 1];
    d_hist[new_idx + twod1 - 1] += buf;
  }
}

__global__ void kernelSetLastZero(unsigned int *d_buf, int len)
{
  d_buf[len-1] = 0;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
    store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
    the cumulative histogram of luminance values (this should go in the
    incoming d_cdf pointer which already has been allocated for you)       */
  int numPixels = numRows * numCols;
  int roundedLen = NextPow2(numPixels);
  int deltaLen = (roundedLen - numPixels);
  float *d_buf0, *d_buf1;
  dim3 blockSize(256);
  dim3 gridSize;

  checkCudaErrors(cudaMalloc(&d_buf0, roundedLen * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_buf1, roundedLen * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_buf0, d_logLuminance, numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_buf1, d_logLuminance, numPixels * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(kVal, d_logLuminance, sizeof(float)));

  // Prepare d_buf
  gridSize.x = (deltaLen + blockSize.x - 1) / blockSize.x;
  kernelScatter<<<gridSize, blockSize>>>(d_buf0, numPixels, roundedLen);
  kernelScatter<<<gridSize, blockSize>>>(d_buf1, numPixels, roundedLen);
  checkCudaErrors(cudaDeviceSynchronize());

  // Reduce d_buf to find min / max
  int steps = log2((double) roundedLen);
  for (int step = 0; step < steps; step++) {
    int stride = (1 << step);
    gridSize.x = (roundedLen + stride - 1) / stride;
    kernelFindMin<<<gridSize, blockSize>>>(d_buf0, stride, roundedLen);
    kernelFindMax<<<gridSize, blockSize>>>(d_buf1, stride, roundedLen);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Copy min / max from device to host
  checkCudaErrors(cudaMemcpy(&min_logLum, d_buf0, sizeof(float), cudaMemcpyDeviceToHost)); 
  checkCudaErrors(cudaMemcpy(&max_logLum, d_buf1, sizeof(float), cudaMemcpyDeviceToHost));
  DEBUG_INFO("min = %f, max = %f, bins = %d", min_logLum, max_logLum, numBins);

  // Figure out histogram of luminance with atomicAdd
  gridSize.x = (numPixels + blockSize.x - 1) / blockSize.x;
  int lumRange = max_logLum - min_logLum;
  kernelHistogram<<<gridSize, blockSize>>>(d_logLuminance, 
                                           d_cdf, 
                                           min_logLum, 
                                           lumRange, 
                                           numBins, 
                                           numPixels);
  checkCudaErrors(cudaDeviceSynchronize());

  // Scan d_histogram to figure out CDF. And numBins should be power of 2.
  for (int twod = 1; twod < numBins; twod *= 2) {
    int twod1 = twod * 2;
    gridSize.x = (numBins / twod1 + blockSize.x - 1) / blockSize.x;
    kernelScanUpsweep<<<gridSize, blockSize>>>(d_cdf, numBins, twod);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  kernelSetLastZero<<<1, 1>>>(d_cdf, numBins);
  checkCudaErrors(cudaDeviceSynchronize());

  for (int twod = numBins / 2; twod >= 1; twod /= 2) {
    int twod1 = twod * 2;
    gridSize.x = (numBins / twod1 + blockSize.x - 1) / blockSize.x;
    kernelScanDownsweep<<<gridSize, blockSize>>>(d_cdf, numBins, twod);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  cudaFree(d_buf0);
  cudaFree(d_buf1);
}
