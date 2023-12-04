# cuda-on-colab
Showing how to use CUDA on google colab (colab.research.google.com)

[ ] run on colab button
[ ] basic c++ example
    [x] copy
    [ ] commit
    [ ] build
    [ ] run
    [ ] profile

## src/simple.cpp

This is copied from [1] and is a simple example of using CUDA to add the elements of two arrays with a million elements each.

```
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
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

  // Run kernel on 1M elements on the CPU
  add(N, x, y);

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

```

## __global__

In order to run on the GPU, we need to change the program suffix from .cpp to .cu so that the nvcc compiler will be used.  We also need to add the __global__ signifier to the add function.  

```

# Sources

[1] Nvida tutorial (https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
[2] Compilation details (https://stackoverflow.com/questions/34527420/a-simple-c-helloworld-with-cuda)