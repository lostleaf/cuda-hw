#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#include "helper_cuda.h"

__global__ void hist_kernel(int *d_bins_32, uint32_t *d_input, int input_width, int input_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x%INPUT_HEIGHT < INPUT_WIDTH)
    {
        uint32_t val = d_input[x];
        if (val == 0)
            printf("%d %d\n", x, val);
        atomicAdd(d_bins_32 + val, 1);
    }
}

__global__ void hist32to8_kernel(int *d_bins_32, uint8_t *d_bins, int bins_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < bins_size)
    {
        int val = d_bins_32[x];
        if (val > 255)
            val = 255;
        d_bins[x] = (uint8_t) val;
    }
}

void opt_2dhisto(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32, int input_width, int input_height, int bins_size)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    int block_size = 1024;
    int grid_size = (INPUT_HEIGHT * ((INPUT_WIDTH+ 128) & 0xFFFFFF80) - 1) / block_size + 1;
    cudaMemset(d_bins_32, 0, sizeof(uint32_t) * bins_size);
    hist_kernel<<<grid_size, block_size>>>(d_bins_32, d_input, input_width, input_height);
    grid_size = (bins_size - 1) / block_size + 1;
    hist32to8_kernel<<<grid_size, block_size>>>(d_bins_32, d_bins, bins_size);
    checkCudaErrors(cudaGetLastError());
}

/* Include below the implementation of any other functions you need */

void init(uint8_t **d_bins, uint32_t **d_input, int **d_bins_32, int hist_width, int hist_height, 
    int input_width, int input_height, uint32_t *h_input)
{
    cudaMalloc(d_bins, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t));
    cudaMalloc(d_bins_32, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint32_t));
    cudaMalloc(d_input, INPUT_HEIGHT * ((INPUT_WIDTH+ 128) & 0xFFFFFF80) * sizeof(int));

    cudaMemcpy(*d_input, h_input, INPUT_HEIGHT * ((INPUT_WIDTH+ 128) & 0xFFFFFF80) * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void final(uint8_t *d_bins, uint32_t *d_input, int *d_bins_32, uint8_t *h_bins, int bins_size)
{
    cudaMemcpy(h_bins, d_bins, bins_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_bins);
    cudaFree(d_bins_32);
    cudaFree(d_input);
}