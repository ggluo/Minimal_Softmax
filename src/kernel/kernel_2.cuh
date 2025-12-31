#define WARP_SIZE 32

__global__ void softmax_warp_primitives(const float* __restrict__ input, 
                                        float* __restrict__ output, 
                                        int batch_size, int dim) {
    // 这里的 Block 是一维的，size=32
    int row_idx = blockIdx.x; 
    int tid = threadIdx.x; // 0-31
    int lane_id = tid;     // warp 内的 id

    if (row_idx >= batch_size) return;

    const float* row_input = input + row_idx * dim;
    float* row_output = output + row_idx * dim;

    // 因为 dim=128, warpSize=32，每个线程需要处理 128/32 = 4 个元素
    // 这里的实现假设 dim 是 32 的倍数
    float thread_max = -__FLT_MAX__;
    float thread_sum = 0.0f;

    // 1. 本地计算 Max
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // 2. Warp 内归约 Max
    float warp_max = warpReduceMax(thread_max);
    // 广播 Max 到 Warp 内所有线程
    warp_max = __shfl_sync(0xffffffff, warp_max, 0);

    // 3. 本地计算 Exp Sum
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        thread_sum += __expf(row_input[i] - warp_max);
    }

    // 4. Warp 内归约 Sum
    float warp_sum = warpReduceSum(thread_sum);
    // 广播 Sum 到 Warp 内所有线程
    warp_sum = __shfl_sync(0xffffffff, warp_sum, 0);

    // 5. 归一化并写回
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        row_output[i] = __expf(row_input[i] - warp_max) / warp_sum;
    }
}