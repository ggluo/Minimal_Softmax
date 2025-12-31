__global__ void softmax_basic(const float* __restrict__ input, 
                              float* __restrict__ output, 
                              int batch_size, int dim) {
    // 计算当前线程负责的行号
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx >= batch_size) return;

    // 指针偏移到当前行的起始位置
    const float* row_input = input + row_idx * dim;
    float* row_output = output + row_idx * dim;

    // 阶段 1: 查找最大值
    float max_val = -__FLT_MAX__;
    for (int i = 0; i < dim; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // 阶段 2: 计算指数和 (Sum Exp)
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_exp += __expf(row_input[i] - max_val);
    }

    // 阶段 3: 归一化并写入
    for (int i = 0; i < dim; i++) {
        row_output[i] = __expf(row_input[i] - max_val) / sum_exp;
    }
}