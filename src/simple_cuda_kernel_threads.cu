#include <iostream>
#include <math.h>

// CUDA Kernel function to add the elements of two arrays on the GPU

__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  // <<< (gridsize), (blocksize) >>>
  // <<<1,1>>> means 1 block with 1 thread
  // "CUDA GPUs run kernels using blocks of threads that are a multiple of 
  // 32 in size, so 256 threads is a reasonable size to choose.""
  add<<<1,256>>>(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}