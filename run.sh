#!/bin/bash

echo "=== Softmax Kernel Testing Script ==="
echo ""

# 步骤 1: 运行 Python 参考实现
echo "Step 1: Running Python reference implementation..."
if ! python3 softmax.py; then
    echo "Warning: Python script had issues, but continuing..."
fi
echo ""

# 步骤 2: 构建 CUDA 程序
echo "Step 2: Building CUDA program..."
mkdir -p build
cd build
if ! cmake .. > cmake.log 2>&1; then
    echo "CMake failed. Check build/cmake.log for details."
    exit 1
fi
if ! make -j4 > make.log 2>&1; then
    echo "Make failed. Check build/make.log for details."
    exit 1
fi
cd ..
echo "Build completed."
echo ""

# 步骤 3: 测试每个 kernel 并比较结果
echo "Step 3: Testing each CUDA kernel (0-4) and comparing with Python reference..."
echo ""


# 测试每个 kernel
all_passed=true
for((i=0;i<=4;i++))
do
    echo "--- Testing Kernel ${i} ---"
    
    # 运行 CUDA kernel
    ./softmax ${i}
    
    # 使用 compare.py 比较结果
    echo ""
    echo "Comparing CUDA kernel ${i} with Python kernel 2 (Safe Softmax)..."
    if python3 compare.py python_output_kernel2.txt kernel${i}_output.txt; then
        echo "✅ Kernel ${i}: PASS"
        # 保存输出文件到结果目录
    else
        echo "❌ Kernel ${i}: FAIL"
        all_passed=false
        # 保存输出文件到结果目录
    fi
    
    echo ""
done

# 总结
echo "=== Test Summary ==="
echo ""
if $all_passed; then
    echo "✅ All 5 kernels passed validation!"
else
    echo "❌ Some kernels failed validation."
    echo "   Check the output files"
fi

echo ""
echo "Files generated:"
echo "  - python_input.txt: Python generated input vector"
echo "  - python_output_kernel1.txt: Python naive softmax output"
echo "  - python_output_kernel2.txt: Python safe softmax output (reference)"
echo "  - python_output_kernel3.txt: Python online softmax output"
echo "  - kernel0_output.txt to kernel4_output.txt: CUDA kernel outputs"
echo ""
echo "To manually compare any two files:"
echo "  python3 compare.py <file1> <file2>"