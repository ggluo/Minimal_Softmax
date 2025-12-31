#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <utils.cuh>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// 复制向量到矩阵的每一行
void copy_vector_to_matrix(float *matrix, const float *vector, int batch_size, int dim) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            matrix[i * dim + j] = vector[j];
        }
    }
}

// 从文件读取数据
int read_from_file(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &data[i]) != 1) {
            printf("Error: Failed to read data from file %s at position %d\n", filename, i);
            fclose(file);
            return 0;
        }
    }
    
    fclose(file);
    return 1;
}

// 保存数据到文件
int save_to_file(const char *filename, const float *data, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return 0;
    }
    
    for (int i = 0; i < size; i++) {
        fprintf(file, "%.8f\n", data[i]);
    }
    
    fclose(file);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <kernel_num>\n", argv[0]);
        printf("  kernel_num: 0-4 for softmax kernels\n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 4)
    {
        printf("Please enter a valid kernel number (0-4).\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Testing softmax kernel %d\n", kernel_num);
    };

    // 使用与 Python 版本相同的参数
    int batch_size = 2048*16;      // 较小的 batch_size 便于验证
    int dim = 128;           // 与 softmax.py 中的 N=256 一致
    
    // 从文件读取 Python 输入向量
    float *cpu_vector = (float *)malloc(dim * sizeof(float));
    if (!read_from_file("python_input.txt", cpu_vector, dim)) {
        printf("Fatal: Cannot read python_input.txt\n");
        free(cpu_vector);
        exit(EXIT_FAILURE);
    }
    
    int size = batch_size * dim;
    float *cpu_input = (float *)malloc(size * sizeof(float));
    float *cpu_output = (float *)malloc(size * sizeof(float));
    
    // 将向量复制到矩阵的每一行
    copy_vector_to_matrix(cpu_input, cpu_vector, batch_size, dim);
    
    // 初始化输出为0
    for (int i = 0; i < size; i++) {
        cpu_output[i] = 0.0f;
    }

    // GPU 内存分配
    float *gpu_input = NULL, *gpu_output = NULL;
    cudaCheck(cudaMalloc((void **)&gpu_input, size * sizeof(float)));
    cudaCheck(cudaMalloc((void **)&gpu_output, size * sizeof(float)));

    // 复制数据到 GPU
    cudaCheck(cudaMemcpy(gpu_input, cpu_input, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu_output, cpu_output, size * sizeof(float), cudaMemcpyHostToDevice));

    // 性能测试
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    
    int repeat = 100;  // 增加重复次数以获得更稳定的性能测量
    
    // Warmup
    test_softmax_kernel(gpu_input, gpu_output, batch_size, dim, 0, true);
    
    cudaEventRecord(beg);
    for (int i = 0; i < repeat; i++)
    {
        test_softmax_kernel(gpu_input, gpu_output, batch_size, dim, kernel_num, false);
    }
    cudaEventRecord(end);
    
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    
    // 复制结果回 CPU
    cudaMemcpy(cpu_output, gpu_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 保存 CUDA 输出到文件（只保存第一行，因为所有行都相同）
    char output_filename[64];
    snprintf(output_filename, sizeof(output_filename), "kernel%d_output.txt", kernel_num);
    if (save_to_file(output_filename, cpu_output, dim)) {
        printf("Saved CUDA output to %s (first row of %d x %d matrix)\n", output_filename, batch_size, dim);
    } else {
        printf("Warning: Failed to save CUDA output to file\n");
    }
    
    // 性能结果
    printf("\nPerformance results:\n");
    printf("  Batch size: %d, Dimension: %d\n", batch_size, dim);
    printf("  Repeat count: %d， Total time: %.5f ms, avg time/repeat: %.5f ms\n", repeat, elapsed_time, elapsed_time / repeat);
    printf("  Throughput: %.2f elements/ms\n", 
           (size * repeat) / elapsed_time);

    // 清理
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    free(cpu_vector);
    free(cpu_input);
    free(cpu_output);

    return 0;
}
