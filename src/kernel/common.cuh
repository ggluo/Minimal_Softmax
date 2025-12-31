#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32

// Warp 级 Max 归约
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp 级 Sum 归约
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ----------------------------------------------------------------------
// 辅助函数：Warp 内全归约 (Butterfly Reduction)
// 相比 down_sync，xor_sync 能让所有线程同时拿到结果，更加稳健
// ----------------------------------------------------------------------
__inline__ __device__ float warpAllReduceMax(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}