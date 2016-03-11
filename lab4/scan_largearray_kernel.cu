#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>
#include <cstdio>
#include <cuda.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// #define smem_offset(a) (a)

// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024
#define BLOCK_MAX_INDEX 2047
// Lab4: Host Helper Functions (allocate your own data structure...)

// Lab4: Device Functions


// Lab4: Kernel Functions
__device__ int smem_offset(int a)
{
    // a + a / NUM_BANKS
    return a + (a >> LOG_NUM_BANKS);
    // return a;
}
__global__ void prefix_sum_block_kernel(float *out, float *in, int n, float *block_sum, bool is_last=false, int block_idx=0)
{
    //shared memory size is index of 2047 + 1
    __shared__ float smem[BLOCK_MAX_INDEX + (BLOCK_MAX_INDEX >> LOG_NUM_BANKS) + 1];

    const int x = threadIdx.x;
    int base;
    if (is_last)
        base = block_idx * BLOCK_SIZE * 2;
    else
        base = blockIdx.x * blockDim.x * 2;

    smem[smem_offset(x)] = in[base + x];
    smem[smem_offset(x + blockDim.x)] = (base + x + blockDim.x < n) ? in[base + x + blockDim.x] : 0;
    __syncthreads();

    int offset = 1;

    for (int d = blockDim.x; d > 0; d >>= 1, offset <<= 1)
    {
        if(x < d)
        {
            int idx1 = offset * (2 * x + 1) - 1;
            int idx2 = idx1 + offset;
            smem[smem_offset(idx2)] += smem[smem_offset(idx1)];
        }
        __syncthreads();
    }

    if (x == 0) 
    {
        int idx = (blockDim.x << 1) - 1;
        idx = smem_offset(idx);

        //block_sum is NULL if not need to write
        if(block_sum)
        {
            if (is_last)
                block_sum[block_idx] = smem[idx];
            else
                block_sum[blockIdx.x] = smem[idx];
        }

        smem[idx] = 0;
    }

    offset >>= 1;
    __syncthreads();

    for (int d = 1; d < (blockDim.x << 1); d <<= 1, offset >>= 1)
    {
        if (x < d)
        {
            int idx1 = offset * (2 * x + 1) - 1;
            int idx2 = idx1 + offset;
            idx1 = smem_offset(idx1);
            idx2 = smem_offset(idx2);

            float tmp = smem[idx1];
            smem[idx1] = smem[idx2];
            smem[idx2] += tmp;
        }
        __syncthreads();
    }

    out[base + x] = smem[smem_offset(x)];
    if (base + x + blockDim.x < n)
        out[base + x + blockDim.x] = smem[smem_offset(x + blockDim.x)];

}

__global__ void add_remaining(float *pre_sum, float *deltas, int n)
{
    __shared__ float delta;
    if (threadIdx.x == 0)
        delta = deltas[blockIdx.x];

    __syncthreads();
    int x = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2;

    if (base + x < n)
        pre_sum[base + x] += delta;
    if (base + x + blockDim.x < n)
        pre_sum[base + x + blockDim.x] += delta;
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
float** allocate(int n, int &level)
{
    float **pre_sums;
    int grid_size, m = n;
    level = 0;
    do
    {
        grid_size = (m + (BLOCK_SIZE << 1) - 1) / (BLOCK_SIZE << 1);
        m = grid_size;
        if (grid_size > 1) level++;
    } while (grid_size > 1);

    pre_sums = new float*[level];

    m = n;
    level = 0;

    do
    {
        grid_size = (m + (BLOCK_SIZE << 1) - 1) / (BLOCK_SIZE << 1);
        if (grid_size > 1)
            CUDA_SAFE_CALL( cudaMalloc(&pre_sums[level++], grid_size * sizeof(float)) );
        m = grid_size;
    } while (grid_size > 1);

    return pre_sums;
}

void destroy(float **pre_sums, int level)
{
    for (int i = 0; i < level; i++)
        CUDA_SAFE_CALL(cudaFree(pre_sums[i]));

    delete[] pre_sums;
}

int floorpow2(int a)
{
    int ret = 2;
    while(ret < a)
        ret <<= 1;
    return ret;
}

void out_device_array(float *d_arr, int n)
{
    float *h_arr = new float[n];

    CUDA_SAFE_CALL( cudaMemcpy(h_arr, d_arr, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n-----------------\n");
    for(int i = 0; i < n; i++)
        printf("%f ", h_arr[i]);
    printf("\n-----------------\n");
    delete[] h_arr;
}

void prescan_recurse(float *d_out, float *d_in, int n, float** pre_sums, int level)
{
    int grid_size = (n + (BLOCK_SIZE << 1) - 1) / (BLOCK_SIZE << 1) - 1;
    int num_except_last = grid_size * (BLOCK_SIZE << 1);
    int block_size_last = floorpow2(n - num_except_last) / 2;


    if(grid_size)
    {

        prefix_sum_block_kernel<<<grid_size, BLOCK_SIZE>>>(d_out, d_in, num_except_last, pre_sums[level]);
        prefix_sum_block_kernel<<<1, block_size_last>>>(d_out, d_in, n, pre_sums[level], true, grid_size);
        prescan_recurse(pre_sums[level], pre_sums[level], grid_size + 1, pre_sums, level + 1);

        add_remaining<<<grid_size + 1, BLOCK_SIZE>>>(d_out, pre_sums[level], n);
    }
    else
    {
        prefix_sum_block_kernel<<<1, block_size_last>>>(d_out, d_in, n, NULL);
    }
}

void prescanArray(float *d_out, float *d_in, int numElements, float** pre_sums)
{
    prescan_recurse(d_out, d_in, numElements, pre_sums, 0);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
