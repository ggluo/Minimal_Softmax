import numpy as np
import sys

# 设置随机种子以便复现
np.random.seed(42)

# 准备数据：使用 linspace 而不是随机数，以便与 CUDA 版本一致
N = 128
x = np.linspace(0.0, 5.0, N)  # 从 -5 到 5 的 256 个点，与 CUDA 版本一致

def kernel_1_naive_softmax(x):
    """
    基础版 Softmax: exp(x) / sum(exp(x))
    注意：这个版本可能数值不稳定（上溢）
    """
    x_exp = np.exp(x)
    partition = np.sum(x_exp)
    return x_exp / partition

def kernel_2_safe_softmax(x):
    """
    安全版 Softmax: 3-pass 算法
    1. Pass 1: 找全局最大值 m
    2. Pass 2: 计算 exp(x-m) 和 sum
    3. Pass 3: 归一化
    """
    # Pass 1: 读取数据，计算最大值
    m = np.max(x)
    
    # Pass 2: 读取数据，计算指数和累加
    # 减去 m 保证指数最大为 e^0 = 1，避免上溢
    x_exp = np.exp(x - m)
    l = np.sum(x_exp)
    
    # Pass 3: 读取数据，计算最终结果
    out = x_exp / l
    return out

def kernel_3_online_softmax(x, block_size=32):
    """
    Online Softmax: 模拟 FlashAttention 的分块处理逻辑
    核心：Rescaling (重缩放)
    """
    n = len(x)
    # 初始化全局状态
    m_global = float('-inf') # 全局最大值
    l_global = 0.0           # 全局分母 (denominator)
    
    # 用于存放未归一化的分子（即 exp(x - m_global)），注意这里的 m_global 是动态变化的
    output_buffer = np.zeros_like(x)
    
    # 分块循环 (Tiling)
    for i in range(0, n, block_size):
        # 1. 加载当前块
        x_block = x[i : i + block_size]
        
        # 2. 计算当前块的局部统计量
        m_local = np.max(x_block)
        exp_local = np.exp(x_block - m_local)
        l_local = np.sum(exp_local)
        
        # 3. 更新全局最大值
        m_prev = m_global
        m_global = max(m_prev, m_local)
        
        # 4. 计算重缩放因子
        scale_prev = np.exp(m_prev - m_global) if m_prev != float('-inf') else 0.0
        scale_curr = np.exp(m_local - m_global)
        
        # 5. 更新全局分母 l
        l_global = l_global * scale_prev + l_local * scale_curr
        
        # 6. 重缩放之前的 Output Buffer
        if i > 0:
            output_buffer[:i] *= scale_prev
            
        # 7. 写入当前块到 Buffer
        output_buffer[i : i + block_size] = exp_local * scale_curr
        
    # 8. 最终统一除以全局分母
    return output_buffer / l_global

def main():
    print("=== Python Softmax Reference Implementation ===")
    print(f"Input vector size: {N}")
    print(f"Input range: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print("")
    
    # 测试 Kernel 1 (Naive)
    print("Testing Kernel 1 (Naive Softmax)...")
    try:
        res_naive = kernel_1_naive_softmax(x)
        has_nan_inf = np.isnan(res_naive).any() or np.isinf(res_naive).any()
        print(f"  Contains NaN/Inf: {has_nan_inf}")
        if has_nan_inf:
            print("  ⚠️  Warning: Naive softmax may have numerical issues!")
    except Exception as e:
        print(f"  Error: {e}")
        res_naive = np.full_like(x, float('nan'))
    
    # 测试 Kernel 2 (Safe)
    print("Testing Kernel 2 (Safe Softmax)...")
    res_safe = kernel_2_safe_softmax(x)
    print(f"  Max value: {np.max(res_safe):.8f}")
    print(f"  Min value: {np.min(res_safe):.8f}")
    
    # 测试 Kernel 3 (Online)
    print("Testing Kernel 3 (Online Softmax)...")
    res_online = kernel_3_online_softmax(x, block_size=32)
    print(f"  Max value: {np.max(res_online):.8f}")
    print(f"  Min value: {np.min(res_online):.8f}")
    
    # 误差分析
    print("\n=== Error Analysis ===")
    diff_safe_online = np.max(np.abs(res_safe - res_online))
    print(f"Max difference between Safe and Online: {diff_safe_online:.2e}")
    
    if diff_safe_online < 1e-6:
        print("✅ Safe and Online softmax results match within tolerance!")
    else:
        print("❌ Safe and Online softmax results differ!")
    
    # 保存结果到文件
    print("\n=== Saving Results ===")
    
    # 保存输入
    np.savetxt('python_input.txt', x, fmt='%.8f')
    print(f"Saved input to 'python_input.txt'")
    
    # 保存三种输出
    np.savetxt('python_output_kernel1.txt', res_naive, fmt='%.8f')
    print(f"Saved Kernel 1 (Naive) output to 'python_output_kernel1.txt'")
    
    np.savetxt('python_output_kernel2.txt', res_safe, fmt='%.8f')
    print(f"Saved Kernel 2 (Safe) output to 'python_output_kernel2.txt'")
    
    np.savetxt('python_output_kernel3.txt', res_online, fmt='%.8f')
    print(f"Saved Kernel 3 (Online) output to 'python_output_kernel3.txt'")
    
    # 打印前几个值用于验证
    print("\n=== First 5 Values ===")
    print("Index | Input      | Kernel1    | Kernel2    | Kernel3    ")
    print("------|------------|------------|------------|------------")
    for i in range(min(5, N)):
        print(f"{i:5d} | {x[i]:10.6f} | {res_naive[i]:10.6f} | {res_safe[i]:10.6f} | {res_online[i]:10.6f}")
    
    # 验证 softmax 性质：所有元素之和应为 1
    print("\n=== Softmax Property Check ===")
    print("(Sum of all elements should be 1)")
    sum_naive = np.sum(res_naive).item()
    sum_safe = np.sum(res_safe).item()
    sum_online = np.sum(res_online).item()
    
    print(f"Kernel 1 (Naive) sum: {sum_naive:.8f} {'✓' if abs(sum_naive - 1.0) < 1e-6 else '✗'}")
    print(f"Kernel 2 (Safe) sum:  {sum_safe:.8f} {'✓' if abs(sum_safe - 1.0) < 1e-6 else '✗'}")
    print(f"Kernel 3 (Online) sum: {sum_online:.8f} {'✓' if abs(sum_online - 1.0) < 1e-6 else '✗'}")

if __name__ == "__main__":
    main()
