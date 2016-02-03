/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#define TILE_WIDTH 32
#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = threadIdx.x, tile_row = threadIdx.y;
    float ret = 0;

    /* printf("%d %d\n", threadIdx.x, threadIdx.y); */
    __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];
    
    int ncol_tiles = M.width / TILE_WIDTH + (M.width % TILE_WIDTH ? 1 : 0);
    for (int i = 0; i < ncol_tiles; i++)
    {
/*
For each warp, threadIdx.y is the same, thus row and tile_row are the same and tile_col and col are consecutive.
So the memory access here is coalesced.
*/
        Ms[threadIdx.y][threadIdx.x] = (row < M.height && tile_col < M.width) ? M.elements[row * M.width + tile_col] : 0;
        Ns[threadIdx.y][threadIdx.x] = (tile_row < N.height && col < N.width) ? N.elements[tile_row * N.width + col] : 0;
        //if (i == 0) printf("%d %d %d %d %d %d\n", row, col, tile_row, tile_col, row * M.width + tile_col, tile_row * N.width + col);
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
            ret += Ms[threadIdx.y][j] * Ns[j][threadIdx.x];

        __syncthreads();

        tile_col += TILE_WIDTH;
        tile_row += TILE_WIDTH;
    }

    if (row < P.height && col < P.width)
        P.elements[row * P.width + col] = ret;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
