__global__ void softmax_optimized(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int batch_size, int dim) {
    int row_idx = blockIdx.x; // 一个 Block 处理一行
    int tid = threadIdx.x;

    if (row_idx >= batch_size) return;

    // 动态共享内存，大小需要在 Kernel 启动时指定
    extern __shared__ float shared_mem[];

    const float* row_input = input + row_idx * dim;
    float* row_output = output + row_idx * dim;

    // ----------------------------------------------------
    // 阶段 1: 查找最大值
    // ----------------------------------------------------
    float my_val = (tid < dim) ? row_input[tid] : -__FLT_MAX__;
    shared_mem[tid] = my_val;
    __syncthreads();

    // 树形归约 Max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_mem[0];
    __syncthreads(); // 必须同步，防止 shared_mem 被 Phase 2 覆盖

    // ----------------------------------------------------
    // 阶段 2: 计算指数和
    // ----------------------------------------------------
    // 重新加载数据或者直接利用寄存器（如果 dim 小），这里为了简单再次读取 shared_mem 会有 bank conflict
    // 更高效的做法是利用寄存器缓存 my_val
    float my_exp = 0.0f;
    if (tid < dim) {
         my_exp = __expf(my_val - max_val);
    }
    shared_mem[tid] = my_exp;
    __syncthreads();

    // 树形归约 Sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = shared_mem[0];
    // 这里不需要 __syncthreads()，因为之后只是读取 sum_exp

    // ----------------------------------------------------
    // 阶段 3: 归一化
    // ----------------------------------------------------
    if (tid < dim) {
        row_output[tid] = my_exp / sum_exp;
    }
}