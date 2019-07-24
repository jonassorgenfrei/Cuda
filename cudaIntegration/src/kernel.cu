////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

 // System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>


__global__
void add(int n, float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];

}

////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
//! @param argc  command line argument count
//! @param argv  command line arguments
//! @param data  data to process on the device
//! @param len   len of \a data
////////////////////////////////////////////////////////////////////////////////
extern "C" void
runCuda(const int argc, const char **argv, float *dataX, float *dataY, unsigned int N)
{
	const unsigned int blockSize = 256;
	const unsigned int numBlocks = (N + blockSize - 1) / blockSize;

	const unsigned int mem_size = sizeof(float) * N;
	// allocate device memory
	float *d_dataX;
	cudaMalloc((void **)&d_dataX, mem_size);
	cudaMemcpy(d_dataX, dataX, mem_size, cudaMemcpyHostToDevice);

	float *d_dataY;
	cudaMalloc((void **)&d_dataY, mem_size);
	cudaMemcpy(d_dataY, dataY, mem_size, cudaMemcpyHostToDevice);

	add <<< numBlocks, blockSize >>> (N, d_dataX, d_dataY); // has to be multiple of 32 in size

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	cudaMemcpy(dataX, d_dataX, mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dataY, d_dataY, mem_size, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(d_dataX);
	cudaFree(d_dataY); 
}
