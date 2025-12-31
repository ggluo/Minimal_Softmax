// 假设: dim=128, warpSize=32 -> 每个线程负责 4 个元素
// ROWS_PER_WARP: 一个 warp 一次迭代处理多少行 (文章中是 4)
template<int ROWS_PER_WARP, int COLS_PER_THREAD>
__global__ void softmax_warp_multi_row(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int batch_size, int dim) {
    // Block Dim 依然是 dim3(32, 4)，即每个 Block 有 4 个 Warp
    // 每个 Warp 的全局 ID
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x;

    // 这个 Warp 负责的起始行
    // 整个 Grid 的总 Warp 数 = gridDim.x * blockDim.y
    // 步长 stride = 总 Warp 数 * 每次处理的行数
    int global_warp_stride = (gridDim.x * blockDim.y) * ROWS_PER_WARP;
    
    // 寄存器缓存，用于存储加载的数据和中间结果
    // buf[row][col]: 存储 ROWS_PER_WARP 行，每行 COLS_PER_THREAD 个元素
    float buf[ROWS_PER_WARP][COLS_PER_THREAD];
    
    // 用于存储每行的局部 max 和 sum
    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];

    // 外层循环：Warp 沿着 Batch 维度跨步处理
    for (int row_start = warp_id * ROWS_PER_WARP; row_start < batch_size; row_start += global_warp_stride) {
        
        // 1. 加载数据并计算局部 Max
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int current_row = row_start + r;
            row_max[r] = -FLT_MAX;
            
            if (current_row < batch_size) {
                const float* row_input = input + current_row * dim;
                
                #pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; ++c) {
                    // 连续访存优化：lane_id + c * 32
                    int element_idx = lane_id + c * WARP_SIZE;
                    if (element_idx < dim) {
                        float val = row_input[element_idx];
                        buf[r][c] = val; // 存入寄存器
                        row_max[r] = fmaxf(row_max[r], val);
                    } else {
                        buf[r][c] = -FLT_MAX; // Padding
                    }
                }
            }
        }

        // 2. Warp 内归约 Max (4行同时进行)
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            row_max[r] = warpReduceMax(row_max[r]);
            row_max[r] = __shfl_sync(0xffffffff, row_max[r], 0);
        }

        // 3. 计算 Exp Sum
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            row_sum[r] = 0.0f;
            int current_row = row_start + r;
            
            if (current_row < batch_size) {
                #pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; ++c) {
                    // 使用之前缓存的 buf，避免重复访存
                    float val = __expf(buf[r][c] - row_max[r]);
                    buf[r][c] = val; // 更新寄存器为 exp 值
                    row_sum[r] += val;
                }
            }
        }

        // 4. Warp 内归约 Sum (4行同时进行)
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            row_sum[r] = warpReduceSum(row_sum[r]);
            row_sum[r] = __shfl_sync(0xffffffff, row_sum[r], 0);
        }

        // 5. 归一化并写回
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int current_row = row_start + r;
            if (current_row < batch_size) {
                float* row_output = output + current_row * dim;
                float inv_sum = 1.0f / row_sum[r]; // 预计算倒数
                
                #pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; ++c) {
                    int element_idx = lane_id + c * WARP_SIZE;
                    if (element_idx < dim) {
                        row_output[element_idx] = buf[r][c] * inv_sum;
                    }
                }
            }
        }
    }
}