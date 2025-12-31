# Minimal Softmax CUDA Implementation

A minimal CUDA implementation of softmax with multiple kernel variants for educational purposes.

## Overview

This project implements and compares different softmax kernels in CUDA, ranging from basic to optimized versions. It includes a Python reference implementation for validation and a comprehensive testing framework.

## Features

- **5 CUDA Softmax Kernels**:
  - Kernel 0: Basic implementation (naive)
  - Kernel 1: Optimized with shared memory
  - Kernel 2: Warp-level primitives
  - Kernel 3: 2D block with warp primitives
  - Kernel 4: Multi-row per warp with template parameters

- **Python Reference Implementation**:
  - Kernel 1: Naive softmax (numerically unstable)
  - Kernel 2: Safe softmax (3-pass algorithm)
  - Kernel 3: Online softmax (simulating FlashAttention tiling)

- **Testing Framework**:
  - Automated build and test script (`run.sh`)
  - File-based comparison between CUDA and Python outputs
  - Performance measurement with CUDA events

## Project Structure

```
Minimal_Softmax/
├── src/
│   ├── kernel/
│   │   ├── common.cuh          # Warp reduction utilities
│   │   ├── kernel_0.cuh        # Basic softmax kernel
│   │   ├── kernel_1.cuh        # Shared memory optimized kernel
│   │   ├── kernel_2.cuh        # Warp primitive kernel
│   │   ├── kernel_3.cuh        # 2D block kernel
│   │   └── kernel_4.cuh        # Multi-row per warp kernel
│   ├── kernel.cuh              # Kernel declarations
│   ├── utils.cuh               # Utility function declarations
│   └── utils.cu                # Utility implementations
├── softmax.cu                  # Main CUDA test program
├── softmax.py                  # Python reference implementation
├── compare.py                  # File comparison utility
├── run.sh                      # Automated test script
└── CMakeLists.txt              # Build configuration
```

## Requirements

- CUDA Toolkit (tested with CUDA 12.8)
- CMake 3.10+
- Python 3.6+ with PyTorch and NumPy
- GCC/G++ with C++11 support

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ggluo/Minimal_Softmax.git
   cd Minimal_Softmax
   ```

2. **Run the complete test suite**:
   ```bash
   ./run.sh
   ```

   This will:
   - Generate test data using Python
   - Build all CUDA kernels
   - Test each kernel (0-4)
   - Compare CUDA outputs with Python reference
   - Report performance metrics

3. **Test individual kernels**:
   ```bash
   # Build the project
   mkdir build && cd build
   cmake .. && make
   
   # Test specific kernel
   ./softmax 0  # Kernel 0
   ./softmax 1  # Kernel 1
   ./softmax 2  # Kernel 2
   ./softmax 3  # Kernel 3
   ./softmax 4  # Kernel 4
   ```

4. **Compare files manually**:
   ```bash
   python3 compare.py python_output_kernel2.txt kernel0_output.txt
   ```

## Kernel Details

### Kernel 0: Basic Softmax
- Simple grid-stride loop
- Each thread processes multiple elements
- No shared memory or warp optimizations

### Kernel 1: Shared Memory Optimized
- Uses dynamic shared memory for reduction
- Each block processes one row
- Better memory coalescing

### Kernel 2: Warp Primitives
- Leverages warp shuffle instructions
- Each warp processes one row
- Efficient warp-level reductions

### Kernel 3: 2D Block with Warp Primitives
- 2D block layout (32x4 threads)
- Each warp processes one row
- Better occupancy with multiple warps per block

### Kernel 4: Multi-row per Warp
- Template-based implementation
- Each warp processes multiple rows (ROWS_PER_WARP=4)
- Register caching for better performance
- Assumes input dimension is 128 (COLS_PER_THREAD=4)

## Validation

The project uses a rigorous validation approach:
1. Python generates reference outputs using three different algorithms
2. CUDA kernels save their outputs to text files
3. The `compare.py` script computes maximum absolute error
4. Tolerance threshold: 1e-5

All kernels are validated against Python's "Safe Softmax" (Kernel 2) implementation.

### Performance Comparison (on NVIDIA RTX 4500 Ada)

| Kernel | Description | Avg Time (ms) | Throughput (elements/ms) | Speedup vs Kernel 0 |
|--------|-------------|---------------|--------------------------|---------------------|
| 0 | Basic Softmax | 0.29562 | 14.2M | 1.0x (baseline) |
| 1 | Shared Memory Optimized | 0.09587 | 43.8M | 3.1x |
| 2 | Warp Primitives | 0.03016 | 139.1M | 9.8x |
| 3 | 2D Block with Warp Primitives | 0.01770 | 236.9M | 16.7x |
| 4 | Multi-row per Warp | 0.01669 | 251.2M | 17.7x |

**Key Observations:**
1. **Significant optimization impact**: Kernel 4 is 17.7x faster than the baseline Kernel 0
2. **Warp-level optimizations**: Kernels 2-4 show dramatic improvements by leveraging warp shuffle instructions
3. **Memory hierarchy utilization**: Kernel 1 improves over baseline by using shared memory
4. **Best performance**: Kernel 4 achieves the highest throughput (251M elements/ms) with multi-row per warp processing

**Test Configuration:**
- GPU: NVIDIA RTX 4500 Ada (24GB VRAM)
- CUDA Version: 12.8
- Batch Size: 32768
- Dimension: 128
- Repetitions: 100

## Notes

- **Kernel 4 Assumption**: Kernel 1 and 4 are designed for dimension up to 512. The test configuration uses dim=128 to ensure all kernels work correctly.
- **Numerical Stability**: The Python implementation includes both numerically unstable (naive) and stable (safe) versions for comparison.

## License

This project is intended for educational purposes. Feel free to use and modify the code as needed.

## Acknowledgments

- Inspired by CUDA programming best practices and optimization techniques
- Online softmax implementation simulates FlashAttention's tiling approach
