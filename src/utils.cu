#include <stdio.h>
#include "utils.cuh"
#include "kernel.cuh"

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) {
    return 1.0e-6 * (end - beg);
}

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void randomize_matrix(float *mat, int N) {
    // NOTICE: 使用gettimeofdays替代srand((unsigned)time(NULL));time精度过低，产生相同随机数
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float) (rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N) {
    int i;
    printf("[");
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            printf("%5.2f ", A[i]);
        else
            printf("%5.2f, ", A[i]);
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
                printf(";\n");
        }
    }
    printf("]\n");
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void test_softmax_basic(const float *__restrict input,
                       float *__restrict output, int batch_size, int dim,
                       bool dummy) {
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x, 1, 1);
    
    if (dummy) {
        printf("Launching dummy softmax_basic Kernel to warmup: Grid[%d], Block[%d]\n", 
               grid_dim.x, block_dim.x);
    }
    
    softmax_basic<<<grid_dim, block_dim>>>(input, output, batch_size, dim);
}

void test_softmax_optimized(const float *__restrict input,
                           float *__restrict output, int batch_size, int dim,
                           bool dummy) {
    // kernel_1 使用动态共享内存，每个 block 处理一行
    
    if (dim <= 128) {
        dim3 block_dim(128, 1, 1);  // 假设 dim <= 128
        dim3 grid_dim(batch_size, 1, 1);
        size_t shared_mem_size = block_dim.x * sizeof(float);
        if (dummy) {
        printf("Launching dummy softmax_optimized Kernel to warmup: Grid[%d], Block[%d], SharedMem[%zu]\n", 
               grid_dim.x, block_dim.x, shared_mem_size);
        }
    
        softmax_optimized<<<grid_dim, block_dim, shared_mem_size>>>(input, output, batch_size, dim);
    }
    else if (dim <= 256) {
        dim3 block_dim(256, 1, 1);  // 假设 dim <= 256
        dim3 grid_dim(batch_size, 1, 1);
        size_t shared_mem_size = block_dim.x * sizeof(float);
        if (dummy) {
        printf("Launching dummy softmax_optimized Kernel to warmup: Grid[%d], Block[%d], SharedMem[%zu]\n", 
               grid_dim.x, block_dim.x, shared_mem_size);    }
    
        softmax_optimized<<<grid_dim, block_dim, shared_mem_size>>>(input, output, batch_size, dim);
    }
    else if (dim <= 512)
    {
        dim3 block_dim(512, 1, 1);  // 假设 dim <= 512
        dim3 grid_dim(batch_size, 1, 1);
        size_t shared_mem_size = block_dim.x * sizeof(float);
        if (dummy) {
        printf("Launching dummy softmax_optimized Kernel to warmup: Grid[%d], Block[%d], SharedMem[%zu]\n", 
               grid_dim.x, block_dim.x, shared_mem_size);
    }
        softmax_optimized<<<grid_dim, block_dim, shared_mem_size>>>(input, output, batch_size, dim);
    }
    else {
        printf("Error: Unsupported dim %d (Need to add more template instances)\n", dim);
    }
}

void test_softmax_warp_primitives(const float *__restrict input,
                                 float *__restrict output, int batch_size, int dim,
                                 bool dummy) {
    // kernel_2 使用 warp 原语，每个 block 有 32 个线程，每个 block 处理一行
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim(batch_size, 1, 1);
    
    if (dummy) {
        printf("Launching dummy softmax_warp_primitives Kernel to warmup: Grid[%d], Block[%d]\n", 
               grid_dim.x, block_dim.x);
    }
    
    softmax_warp_primitives<<<grid_dim, block_dim>>>(input, output, batch_size, dim);
}

void test_softmax_warp_primitives_2d(const float *__restrict input,
                                    float *__restrict output, int batch_size, int dim,
                                    bool dummy) {
    // kernel_3 使用 2D block，每个 block 有 4 个 warp，每个 warp 处理一行
    dim3 block_dim(32, 4, 1);  // 32 threads per warp, 4 warps per block
    dim3 grid_dim((batch_size + block_dim.y - 1) / block_dim.y, 1, 1);
    
    if (dummy) {
        printf("Launching dummy softmax_warp_primitives_2d Kernel to warmup: Grid[%d], Block[%d,%d]\n", 
               grid_dim.x, block_dim.x, block_dim.y);
    }
    
    softmax_warp_primitives_2d<<<grid_dim, block_dim>>>(input, output, batch_size, dim);
}

void test_softmax_warp_multi_row(const float *__restrict input,
                                float *__restrict output, int batch_size, int dim, 
                                bool dummy) {
    // ----------------------------------------------------------------
    // 关键修复：根据 dim 动态选择模板参数
    // ----------------------------------------------------------------
    
    // 场景 1: Dim <= 128 (每个线程处理 4 个元素)
    // 32 threads * 4 cols = 128 capacity
    if (dim <= 128) {
        const int ROWS = 4;
        const int COLS = 4; 
        
        dim3 block(32, 4);
        int total_warps = (batch_size + ROWS - 1) / ROWS;
        dim3 grid((total_warps + block.y - 1) / block.y);
        
        // 调用 <4, 4> 版本
        softmax_warp_multi_row<ROWS, COLS><<<grid, block>>>(input, output, batch_size, dim);
    } 
    // 场景 2: Dim <= 256 (每个线程处理 8 个元素)
    // 32 threads * 8 cols = 256 capacity
    else if (dim <= 256) {
        const int ROWS = 4;
        const int COLS = 8; // <--- 这里改为 8 !!!
        
        dim3 block(32, 4);
        int total_warps = (batch_size + ROWS - 1) / ROWS;
        dim3 grid((total_warps + block.y - 1) / block.y);
        
        // 调用 <4, 8> 版本
        softmax_warp_multi_row<ROWS, COLS><<<grid, block>>>(input, output, batch_size, dim);
    }
    // 场景 3: Dim <= 512 (每个线程处理 16 个元素)
    else if (dim <= 512) {
        const int ROWS = 4;
        const int COLS = 16; // 32 * 16 = 512
        
        dim3 block(32, 4);
        int total_warps = (batch_size + ROWS - 1) / ROWS;
        dim3 grid((total_warps + block.y - 1) / block.y);
        
        softmax_warp_multi_row<ROWS, COLS><<<grid, block>>>(input, output, batch_size, dim);
    }
    else {
        printf("Error: Unsupported dim %d (Need to add more template instances)\n", dim);
    }
}

void test_softmax_kernel(const float *__restrict input,
                        float *__restrict output, int batch_size, int dim,
                        int kernel_num, bool dummy) {
    switch (kernel_num) {
        case 0:
            test_softmax_basic(input, output, batch_size, dim, dummy);
            break;
        case 1:
            test_softmax_optimized(input, output, batch_size, dim, dummy);
            break;
        case 2:
            test_softmax_warp_primitives(input, output, batch_size, dim, dummy);
            break;
        case 3:
            test_softmax_warp_primitives_2d(input, output, batch_size, dim, dummy);
            break;
        case 4:
            test_softmax_warp_multi_row(input, output, batch_size, dim, dummy);
            break;
        default:
            printf("Invalid softmax kernel number: %d\n", kernel_num);
            break;
    }
}
