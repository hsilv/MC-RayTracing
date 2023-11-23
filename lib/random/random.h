#ifndef RANDOM_H
#define RANDOM_H

#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void random_init(unsigned long long seed);
__device__ float randomC();

#endif