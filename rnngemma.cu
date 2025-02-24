# This is a port of the Pallad kernel from Recurrentgemma to CUDS

# https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma%2Fjax%2Fpallas.py

# There are some language differences between Pallas an CUDA , but a fairly straightforward mappinf of concepts 

# Pallas         CUDA
# kernel         kernel      - the code that runs on the specialzed hw
# task                        - the group of operations that can be done in parallel 
# data                        - the chunk of data to be worked on
# communication               - how the CPU and TPU share code and data
#                kernel       - the code that is excecuted on the GPU
#                __global__   - the keywod starting a kernel code block
#                threads
#                blocks
#                grids
#                memory 
#                 global       - cuda memory shared with cpu
#                 HMM          - on gPU memory shared among All SMs
#                 SM           - per processor memory  4k bytes or so, vert fast
#               __syncthreads__ - the checkpoint which synchronized per-thread execution globally 

# tasks are translated to kernels

# 
import functools
import math
from typing import NamedTuple

import torch
# import shard_map # i don't know what this is
import numpy as np
# array typing as at

class CudaShardingSoec(NamedTuple):
  """ the sharding spec for running a cuda kernel with shared values. """
  #mesh.jax.sharding