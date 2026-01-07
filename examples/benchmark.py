"""
MacroTorch Benchmark Script

Compares performance of MacroTorch CUDA kernels against SciPy (CPU) and optionally PyTorch (GPU).

Run in Google Colab:
    !git clone https://github.com/ggSohamgg/macrotorch.git
    %cd macrotorch
    !pip install -e .[benchmark]
    !python examples/benchmark.py
"""

import numpy as np
import time
from scipy.signal import correlate2d
from numba import cuda

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def scipy_conv2d(A, K, padding=0):
    if padding > 0:
        A_padded = np.pad(A, padding, mode='constant', constant_values=0)
    else:
        A_padded = A
    return correlate2d(A_padded, K, mode='valid')


def benchmark_forward(H=512, W=512, Kh=5, Kw=5, padding=2, dtype='float32'):
    """Benchmark forward convolution."""
    from macrotorch import conv2d_forward
    
    np_dtype = np.float32 if dtype == 'float32' else np.float16
    
    print(f"\n{'='*75}")
    print(f"FORWARD PASS | Input: ({H}, {W}) | Kernel: ({Kh}, {Kw}) | Padding: {padding} | {dtype.upper()}")
    print(f"{'='*75}")
    
    np.random.seed(42)
    A = np.random.randn(H, W).astype(np_dtype)
    K = np.random.randn(Kh, Kw).astype(np_dtype)
    
    start = time.perf_counter()
    scipy_out = scipy_conv2d(A.astype(np.float32), K.astype(np.float32), padding)
    scipy_time = (time.perf_counter() - start) * 1000
    print(f"SciPy (CPU):      {scipy_time:.4f} ms")
    
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype == 'float32' else torch.float16
        t_A = torch.from_numpy(A).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        
        _ = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        pt_out = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - start) * 1000
        print(f"PyTorch (GPU):    {pt_time:.4f} ms")
    
    _ = conv2d_forward(A, K, padding=padding)
    
    start = time.perf_counter()
    mt_out = conv2d_forward(A, K, padding=padding)
    mt_time = (time.perf_counter() - start) * 1000
    print(f"MacroTorch (GPU): {mt_time:.4f} ms")
    
    mt_error = np.abs(mt_out - scipy_out).max()
    print(f"\nMax Error (vs SciPy): {mt_error:.2e}")
    print(f"Speedup vs SciPy: {scipy_time/mt_time:.2f}x")
    if TORCH_AVAILABLE:
        print(f"Speedup vs PyTorch: {pt_time/mt_time:.2f}x")


def benchmark_input_backward(H=512, W=512, Kh=5, Kw=5, padding=2, dtype='float32'):
    """Benchmark input gradient computation."""
    from macrotorch import conv2d_forward, conv2d_input_backward
    
    np_dtype = np.float32 if dtype == 'float32' else np.float16
    
    print(f"\n{'='*75}")
    print(f"INPUT BACKWARD | Input: ({H}, {W}) | Kernel: ({Kh}, {Kw}) | Padding: {padding} | {dtype.upper()}")
    print(f"{'='*75}")
    
    np.random.seed(42)
    A = np.random.randn(H, W).astype(np_dtype)
    K = np.random.randn(Kh, Kw).astype(np_dtype)
    
    output = conv2d_forward(A, K, padding=padding)
    grad_out = np.random.randn(*output.shape).astype(np_dtype)
    
    _ = conv2d_input_backward(grad_out, K, padding=padding)
    
    start = time.perf_counter()
    grad_input = conv2d_input_backward(grad_out, K, padding=padding)
    mt_time = (time.perf_counter() - start) * 1000
    print(f"MacroTorch (GPU): {mt_time:.4f} ms")
    print(f"Gradient shape: {grad_input.shape}")


def benchmark_bias_backward(N=32, C=128, H=64, W=64, dtype='float32'):
    """Benchmark bias gradient computation."""
    from macrotorch import conv2d_bias_backward
    
    np_dtype = np.float32 if dtype == 'float32' else np.float16
    
    print(f"\n{'='*75}")
    print(f"BIAS BACKWARD | Input: ({N}, {C}, {H}, {W}) | {dtype.upper()}")
    print(f"{'='*75}")
    
    np.random.seed(42)
    grad_out = np.random.randn(N, C, H, W).astype(np_dtype)
    
    start = time.perf_counter()
    scipy_result = np.sum(grad_out.astype(np.float32), axis=(0, 2, 3))
    scipy_time = (time.perf_counter() - start) * 1000
    print(f"NumPy (CPU):      {scipy_time:.4f} ms")
    
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        
        _ = t_grad.sum(dim=(0, 2, 3))
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        pt_result = t_grad.sum(dim=(0, 2, 3))
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - start) * 1000
        print(f"PyTorch (GPU):    {pt_time:.4f} ms")
    
    d_input = cuda.to_device(grad_out)
    d_output = cuda.device_array(C, dtype=np.float32)
    
    _ = conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
    
    start = time.perf_counter()
    conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
    cuda.synchronize()
    mt_time = (time.perf_counter() - start) * 1000
    
    mt_result = d_output.copy_to_host()
    print(f"MacroTorch (GPU): {mt_time:.4f} ms")
    
    mt_error = np.abs(mt_result - scipy_result).max()
    print(f"\nMax Error (vs NumPy): {mt_error:.2e}")
    print(f"Speedup vs NumPy: {scipy_time/mt_time:.2f}x")
    if TORCH_AVAILABLE:
        print(f"Speedup vs PyTorch: {pt_time/mt_time:.2f}x")


def main():
    print("\n" + "="*75)
    print("MacroTorch Benchmark Suite")
    print("="*75)
    
    if TORCH_AVAILABLE:
        print("PyTorch: Available (will compare against)")
    else:
        print("PyTorch: Not installed (install with: pip install macrotorch[benchmark])")
    
    print("\n--- Forward Pass Benchmarks ---")
    benchmark_forward(H=512, W=512, Kh=5, Kw=5, padding=2, dtype='float32')
    benchmark_forward(H=512, W=512, Kh=5, Kw=5, padding=2, dtype='float16')
    
    print("\n--- Input Backward Benchmarks ---")
    benchmark_input_backward(H=512, W=512, Kh=5, Kw=5, padding=2, dtype='float32')
    
    print("\n--- Bias Backward Benchmarks ---")
    benchmark_bias_backward(N=32, C=128, H=64, W=64, dtype='float32')
    benchmark_bias_backward(N=32, C=128, H=64, W=64, dtype='float16')
    
    print("\n" + "="*75)
    print("Benchmark Complete!")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
