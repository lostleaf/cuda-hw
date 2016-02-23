#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#include "helper_cuda.h"

const int BIN_SIZE = HISTO_WIDTH * HISTO_HEIGHT;
const int INPUT_WIDTH_PAD = (INPUT_WIDTH+ 128) & 0xFFFFFF80;
const int N = 64;

__global__ void hist_kernel(int *d_bins_32, uint32_t *d_input)
{
    __shared__ int d_bins_32_smem[HISTO_WIDTH * HISTO_HEIGHT];
    int x = blockIdx.x * (N * blockDim.x) + threadIdx.x;
    d_bins_32_smem[threadIdx.x] = 0;
    __syncthreads();
    for (int j = 0; j < N; j++, x += blockDim.x)
    {
        int i = x % INPUT_WIDTH + x / INPUT_WIDTH * INPUT_WIDTH_PAD;
        uint32_t val = d_input[i];
        atomicAdd(d_bins_32_smem + val, 1);
    }
    __syncthreads();
    atomicAdd(d_bins_32 + threadIdx.x, d_bins_32_smem[threadIdx.x]);
}

__global__ void hist32to8_kernel(int *d_bins_32, uint8_t *d_bins)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < BIN_SIZE)
    {
        int val = d_bins_32[x];
        if (val > UINT8_MAX)
            val = UINT8_MAX;
        d_bins[x] = (uint8_t) val;
    }
}

void opt_2dhisto(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    int block_size = 1024;
    int grid_size = (INPUT_HEIGHT * INPUT_WIDTH - 1) / (N * block_size) + 1;

    cudaMemset(d_bins_32, 0, sizeof(uint32_t) * BIN_SIZE);

    hist_kernel<<<grid_size, block_size>>>(d_bins_32, d_input);

    grid_size = (BIN_SIZE - 1) / block_size + 1;
    hist32to8_kernel<<<grid_size, block_size>>>(d_bins_32, d_bins);

    checkCudaErrors(cudaGetLastError());
}

/* Include below the implementation of any other functions you need */

void init(uint8_t **d_bins, uint32_t **d_input, int **d_bins_32, uint32_t *h_input)
{
    cudaMalloc(d_bins, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t));
    cudaMalloc(d_bins_32, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
    cudaMalloc(d_input, INPUT_HEIGHT * INPUT_WIDTH_PAD * sizeof(int));

    cudaMemcpy(*d_input, h_input, INPUT_HEIGHT * INPUT_WIDTH_PAD * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void final(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32, uint8_t *h_bins)
{
    cudaMemcpy(h_bins, d_bins, BIN_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_bins);
    cudaFree(d_bins_32);
    cudaFree(d_input);
}
