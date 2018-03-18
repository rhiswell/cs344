//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <assert.h>
#include <algorithm>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


#define DEBUG_INFO(fmt, args...)    { printf("DEBUG: " fmt "\n", ##args); }

#define BLOCKSIZE 256

__global__ void kernelHistgramByGroup(unsigned int * const d_in, 
                                      unsigned int *d_hist0_by_group, 
                                      unsigned int *d_hist1_by_group,
                                      int len,
                                      int bitshift)
{
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int local_hist[2];

    local_hist[0] = 0;
    local_hist[1] = 0;
    __syncthreads();

    if (idx < len) {
        int bitval = (d_in[idx] >> bitshift) & 0x01;
        atomicAdd(&local_hist[bitval], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_hist0_by_group[blockIdx.x] = local_hist[0];
        d_hist1_by_group[blockIdx.x] = local_hist[1];
    }
    __syncthreads();
}

__device__ __inline__ void exclusiveScan(int *in, int len)
{
    int i = threadIdx.x;

    // Up-sweep
    for (int twod = 1; twod < len; twod *= 2) {
        int twod1 = twod * 2;
        if (i % twod1 == 0) {
            in[i + twod1 - 1] += in[i + twod - 1];
        }
        __syncthreads();
    }

    // Down-sweep
    in[len - 1] = 0;
    __syncthreads();

    for (int twod = len / 2; twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        if (i % twod1 == 0) {
            int t = in[i + twod - 1];
            in[i + twod - 1] = in[i + twod1 - 1];
            in[i + twod1 - 1] += t;
        }
        __syncthreads();
    }
}

__global__ void kernelCalNewIdxAndPutEle(unsigned int *d_val_in, 
                                         unsigned int *d_val_out, 
                                         unsigned int *d_pos_in,
                                         unsigned int *d_pos_out,
                                         unsigned int *d_cdf0_by_group, 
                                         unsigned int *d_cdf1_by_group, 
                                         int len,
                                         unsigned int start_pos0, 
                                         unsigned int start_pos1, 
                                         int bitshift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bitval;
    
    __shared__ int predicate[BLOCKSIZE];

    // \begin to handle 0
    // Figure out internal offset using compact
    predicate[threadIdx.x] = 1;
    if (idx < len) {
        bitval = (d_val_in[idx] >> bitshift) & 0x01;
        predicate[threadIdx.x] = ((bitval == 0) ? 1 : 0);
    }
    __syncthreads();

    exclusiveScan(predicate, BLOCKSIZE);

    if (idx < len && bitval == 0) {
        int new_idx = start_pos0 + d_cdf0_by_group[blockIdx.x] + predicate[threadIdx.x];
        d_val_out[new_idx] = d_val_in[idx];
        d_pos_out[new_idx] = d_pos_in[idx];
    }
    __syncthreads();
    // \end

    // \begin to handle 1
    predicate[threadIdx.x] = 1;
    if (idx < len) {
        predicate[threadIdx.x] = ((bitval == 1) ? 1 : 0);
    }
    __syncthreads();

    exclusiveScan(predicate, BLOCKSIZE);

    if (idx < len && bitval == 1) {
        int new_idx = start_pos1 + d_cdf1_by_group[blockIdx.x] + predicate[threadIdx.x];
        d_val_out[new_idx] = d_val_in[idx];
        d_pos_out[new_idx] = d_pos_in[idx];
    }
    __syncthreads();
    // \end
}

// Refer: http://blog.csdn.net/u010445006/article/details/74852690
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    size_t nr_elems = numElems;

    dim3 blocks(BLOCKSIZE);
    dim3 grids((nr_elems + blocks.x -1) / blocks.x);

    unsigned int *d_cdf0_by_group;
    unsigned int *d_cdf1_by_group;

    checkCudaErrors(cudaMalloc(&d_cdf0_by_group, grids.x * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_cdf1_by_group, grids.x * sizeof(unsigned int)));

    unsigned int *d_val_buf0 = d_inputVals;
    unsigned int *d_val_buf1 = d_outputVals;
    unsigned int *d_pos_buf0 = d_inputPos;
    unsigned int *d_pos_buf1 = d_outputPos;
    unsigned int *tmp_ptr;

    for (int bitshift = 0; bitshift < 33; bitshift++) {

        // Group inputVals in block
        // Step #0: calculate cdf of {0, 1} by group
        kernelHistgramByGroup<<<grids, blocks>>>(d_val_buf0, 
                                                 d_cdf0_by_group, 
                                                 d_cdf1_by_group, 
                                                 nr_elems, 
                                                 bitshift);
        checkCudaErrors(cudaDeviceSynchronize());

        unsigned int last_group_hist0;
        checkCudaErrors(cudaMemcpy(&last_group_hist0, 
                                   d_cdf0_by_group + grids.x - 1, 
                                   sizeof(unsigned int), 
                                   cudaMemcpyDeviceToHost));

        thrust::exclusive_scan(thrust::device, d_cdf0_by_group, d_cdf0_by_group + grids.x, d_cdf0_by_group);
        thrust::exclusive_scan(thrust::device, d_cdf1_by_group, d_cdf1_by_group + grids.x, d_cdf1_by_group);
        checkCudaErrors(cudaDeviceSynchronize());

        // Use result above to figure out prefix-sum of {0, 1}
        unsigned int start_pos0 = 0;
        unsigned int start_pos1 = 0;
        checkCudaErrors(cudaMemcpy(&start_pos1,
                                   d_cdf0_by_group + grids.x - 1,
                                   sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));
        start_pos1 += last_group_hist0;

        // Step #1: calculate new index and put the element into right slot
        kernelCalNewIdxAndPutEle<<<grids, blocks>>>(d_val_buf0,
                                                    d_val_buf1,
                                                    d_pos_buf0,
                                                    d_pos_buf1,
                                                    d_cdf0_by_group,
                                                    d_cdf1_by_group,
                                                    nr_elems,
                                                    start_pos0,
                                                    start_pos1,
                                                    bitshift);
        checkCudaErrors(cudaDeviceSynchronize());

        // Step #2: Swap buffer
        tmp_ptr = d_val_buf0;
        d_val_buf0 = d_val_buf1;
        d_val_buf1 = tmp_ptr;

        tmp_ptr = d_pos_buf0;
        d_pos_buf0 = d_pos_buf1;
        d_pos_buf1 = tmp_ptr;
    }

    checkCudaErrors(cudaFree(d_cdf0_by_group));
    checkCudaErrors(cudaFree(d_cdf1_by_group));
}
