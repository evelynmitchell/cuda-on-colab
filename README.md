# cuda-on-colab
Showing how to use CUDA on google colab (colab.research.google.com)

Run example in Collab: <a target="_blank" href="https://colab.research.google.com/github/evelynmitchell/cuda-on-colab/blob/master/CudaColab.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```
[x] run on colab button
[/] basic c++ example
    [x] copy
    [x] commit
    [x] build
    [x] run
    [x] profile
    [ ] multi-threaded
    [ ] multi-gpu
```

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

In order to run on the GPU, we need to change the program suffix from .cpp to .cu so that the nvcc compiler will be used.[2]  We also need to add the __global__ signifier to the add function.  

In addition, just adding __global__ is not enough.  We also need to add the <<<1,1>>> syntax to the call to the add function.  This is the kernel launch configuration.  The first number is the number of blocks and the second number is the number of threads per block.  The <<<1,1>>> syntax is required even if we only have one block and one thread per block.  [3]

```
/content/cuda-on-colab/src/simple_cuda.cu(26): error: a __global__ function call must be configured

1 error detected in the compilation of "/content/cuda-on-colab/src/simple_cuda.cu".


```

To fix this, we need to change the function invocation to be:

```
  // Run kernel on 1M elements on the GPU
  // <<< (gridsize), (blocksize) >>>
  // <<<1,1>>> means 1 block with 1 thread
  add<<<1,1>>>(N, x, y);
```

## Profiling

nvprof is a command line profiler that comes with the CUDA toolkit.  It can be used to profile the execution of the program.  To use it, we need to add the following to run the program:

```
nvprof ./simple_cuda_kernel
``` 
This produces an output similar to:
```
==764== NVPROF is profiling process 764, command: ./simple_cuda_kernal_launch
Max error: 1
==764== Profiling application: ./simple_cuda_kernal_launch
==764== Warning: 1 records have invalid timestamps due to insufficient device buffer space. You can configure the buffer space using the option --device-buffer-size.
==764== Profiling result:
No kernels were profiled.
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
      API calls:   98.68%  102.66ms         1  102.66ms  102.66ms  102.66ms  cudaLaunchKernel
                    1.18%  1.2271ms         1  1.2271ms  1.2271ms  1.2271ms  cuDeviceGetPCIBusId
                    0.11%  112.96us       101  1.1180us     149ns  48.383us  cuDeviceGetAttribute
                    0.02%  25.821us         1  25.821us  25.821us  25.821us  cuDeviceGetName
                    0.00%  1.7480us         3     582ns     207ns  1.2670us  cuDeviceGetCount
                    0.00%     661ns         2     330ns     176ns     485ns  cuDeviceGet
                    0.00%     347ns         1     347ns     347ns     347ns  cuModuleGetLoadingMode
                    0.00%     344ns         1     344ns     344ns     344ns  cuDeviceTotalMem
                    0.00%     255ns         1     255ns     255ns     255ns  cuDeviceGetUuid
```

## Memory allocation

CUDA programs have a global shared memory alloation between the CPU and GPU. "Unified Memory lowers the bar of entry to parallel programming on the CUDA platform, by making device memory management an optimization, rather than a requirement."[4]

The memory is allocated using cudaMallocManaged.  This is a CUDA API function that allocates memory that is accessible from the host and the device.  The memory is allocated on the device and the host can access it.  The device can access it without copying it from the host.  The memory is freed using cudaFree.

```
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

...
  cudaFree(x);
  cudaFree(y);

```

The example program is simple_cuda_memory_alloc.cu

The compliation with nvcc and profiling with nvprof is shown in the notebook.

```
==2424== Unified Memory profiling result:
Device "Tesla V100-SXM2-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  843.8660us  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  357.2770us  Device To Host
      12         -         -         -           -  4.243755ms  Gpu page fault groups
Total CPU Page faults: 36
```


# Sources

[1] Nvida tutorial (https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
[2] Compilation details (https://stackoverflow.com/questions/34527420/a-simple-c-helloworld-with-cuda)
[3] More nvcc details (https://stackoverflow.com/questions/67177794/error-a-global-function-call-must-be-configured)
[4] (https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)