#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <iostream>

/**
 * 0 - No Cuda Code
 * 1 - Single Thread Kernel (with race condition)
 * 2 - Multi Threading Kernel 
 * 3 - Grid: Multi Threading Blocks Kernel
 */
#define Version 3

#if (Version == 0)
 // function to add the elements of two arrays
void add(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}
#else

 // CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float *x, float *y)
{

#if (Version == 1)
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
#else
	#if (Version == 2)
	/* V2 */
	int index = threadIdx.x;
	int stride = blockDim.x;
	// index of the current thread within its block
	// number of threads in the block
	#elif (Version == 3)
	/* V3 */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	// gridDim: number of thread blocks
	// blockDim: number of threads in each block
	// bkockIdx: index the current block within the grid
	// threadIdx: index of the current thread within the block
	#endif

	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
#endif
}
#endif

int main(void)
{
	int N = 1 << 20;
	
#if (Version == 0)
	float *x = new float[N];
	float *y = new float[N];

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	// Run kernel on 1M elements on the CPU
	add(N, x, y);

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;
	// Free memory
	delete[] x;
	delete[] y;
#else
	// Allocate Unified Memory – accessible from CPU or GPU
	float *x, *y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

#if (Version == 1)
	// Run kernel on 1M elements on the GPU
	add <<<1, 1>> > (N, x, y);
#elif (Version == 2)
	int blockSize = 256;
	int numBlocks = 1;
	add <<< numBlocks, blockSize >>> (N, x, y); // has to be multiple of 32 in size
#elif (Version == 3)
	// Run kernel on 1M elements on the GPU
	// Execution configuration
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add <<< numBlocks, blockSize >>> (N, x, y); // has to be multiple of 32 in size
#endif
	
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);
#endif
	return 0;
}