__global__ void softmax_warp_primitives_2d(const float* __restrict__ input, 
                                           float* __restrict__ output, 
                                           int batch_size, int dim) {
    // 2D Block 索引: 
    // threadIdx.x (0-31) 是 Warp 内的 Lane ID
    // threadIdx.y (0-3)  是 Block 内的 Warp ID
    
    // 计算当前 Warp 负责的行号
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane_id = threadIdx.x;

    if (row_idx >= batch_size) return;

    // 后续逻辑与 Kernel 3 完全一致
    const float* row_input = input + row_idx * dim;
    float* row_output = output + row_idx * dim;

    float thread_max = - __FLT_MAX__;
    float thread_sum = 0.0f;

    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }

    float warp_max = warpReduceMax(thread_max);
    warp_max = __shfl_sync(0xffffffff, warp_max, 0);

    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        thread_sum += __expf(row_input[i] - warp_max);
    }

    float warp_sum = warpReduceSum(thread_sum);
    warp_sum = __shfl_sync(0xffffffff, warp_sum, 0);

    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        row_output[i] = __expf(row_input[i] - warp_max) / warp_sum;
    }
}