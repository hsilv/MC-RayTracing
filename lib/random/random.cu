#include "random.h"
#include <cuda_runtime.h>

__device__ curandState_t state;


__global__ void random_init(unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state);
}

__device__ float randomC() {
    return curand_uniform(&state);
}