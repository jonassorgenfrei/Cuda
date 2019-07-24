/* TODO MAKE SIMPLE WITHOUT HELPERS !!! */

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void runCuda(const int argc, const char **argv,
	float *dataX, float *dataY, unsigned int N);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	const unsigned int N = 1 << 20;

	float * x;
	x = (float*)malloc(N*sizeof(float));
	float * y;
	y = (float*)malloc(N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// run the device part of the program
	runCuda(argc, (const char **)argv, x, y, N);
	
	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

	free(x);
	free(y);

	exit(EXIT_SUCCESS);
}