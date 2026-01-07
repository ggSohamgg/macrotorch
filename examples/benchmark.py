"""
MacroTorch Comprehensive Benchmark Suite

Benchmarks all MacroTorch CUDA kernels against CPU baselines and optionally PyTorch.

Usage:
    pip install -e .[benchmark]
    python examples/benchmark.py
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

from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward, Conv2d


def scipy_conv2d(A, K, padding=0):
    if padding > 0:
        A_padded = np.pad(A, padding, mode='constant', constant_values=0)
    else:
        A_padded = A
    return correlate2d(A_padded, K, mode='valid')


def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def print_table_header():
    print(f"\n{'-'*80}")
    print(f"{'Implementation':<20} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"{'-'*80}")


def print_row(name, time_ms, speedup, error):
    print(f"{name:<20} | {time_ms:<12.4f} | {speedup:<10} | {error:<12}")


def benchmark_forward_single(H, W, Kh, Kw, padding, dtype_name, num_runs=5):
    """Benchmark a single forward configuration."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Input: ({H}x{W}) | Kernel: ({Kh}x{Kw}) | Padding: {padding} | {dtype_name.upper()}")
    print_table_header()
    
    np.random.seed(42)
    A = np.random.randn(H, W).astype(np_dtype)
    K = np.random.randn(Kh, Kw).astype(np_dtype)
    
    # SciPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        scipy_out = scipy_conv2d(A.astype(np.float32), K.astype(np.float32), padding)
        times.append((time.perf_counter() - start) * 1000)
    scipy_time = np.median(times)
    print_row("SciPy (CPU)", scipy_time, "1.00x", "Ground Truth")
    
    # PyTorch (GPU)
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_A = torch.from_numpy(A).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        
        for _ in range(3):
            _ = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_out = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_error = np.abs(pt_out.squeeze().cpu().numpy().astype(np.float32) - scipy_out).max()
        print_row("PyTorch (GPU)", pt_time, f"{scipy_time/pt_time:.2f}x", f"{pt_error:.2e}")
    
    # MacroTorch (GPU)
    for _ in range(3):
        _ = conv2d_forward(A, K, padding=padding)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        mt_out = conv2d_forward(A, K, padding=padding)
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_error = np.abs(mt_out - scipy_out).max()
    print_row("MacroTorch (GPU)", mt_time, f"{scipy_time/mt_time:.2f}x", f"{mt_error:.2e}")
    
    print(f"{'-'*80}")
    return {'scipy': scipy_time, 'mt': mt_time}


def benchmark_forward_suite():
    """Run forward pass benchmarks across multiple configurations."""
    print_header("FORWARD PASS BENCHMARKS")
    
    configs = [
        # (H, W, Kh, Kw, padding)
        (256, 256, 3, 3, 1),
        (256, 256, 5, 5, 2),
        (256, 256, 11, 11, 5),
        (512, 512, 3, 3, 1),
        (512, 512, 5, 5, 2),
        (512, 512, 11, 11, 5),
        (512, 512, 31, 31, 15),
        (1024, 1024, 5, 5, 2),
    ]
    
    print("\n--- FP32 ---")
    for H, W, Kh, Kw, padding in configs:
        benchmark_forward_single(H, W, Kh, Kw, padding, 'float32')
    
    print("\n--- FP16 ---")
    for H, W, Kh, Kw, padding in configs[:4]:
        benchmark_forward_single(H, W, Kh, Kw, padding, 'float16')


def benchmark_input_backward_single(H, W, Kh, Kw, padding, dtype_name, num_runs=5):
    """Benchmark a single input backward configuration."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Input: ({H}x{W}) | Kernel: ({Kh}x{Kw}) | Padding: {padding} | {dtype_name.upper()}")
    print_table_header()
    
    np.random.seed(42)
    A = np.random.randn(H, W).astype(np_dtype)
    K = np.random.randn(Kh, Kw).astype(np_dtype)
    
    output = conv2d_forward(A, K, padding=padding)
    grad_out = np.random.randn(*output.shape).astype(np_dtype)
    
    # Warmup
    for _ in range(3):
        _ = conv2d_input_backward(grad_out, K, padding=padding)
    
    # MacroTorch
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        grad_input = conv2d_input_backward(grad_out, K, padding=padding)
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    print_row("MacroTorch (GPU)", mt_time, "-", f"Shape: {grad_input.shape}")
    print(f"{'-'*80}")


def benchmark_input_backward_suite():
    """Run input backward benchmarks."""
    print_header("INPUT BACKWARD BENCHMARKS")
    
    configs = [
        (256, 256, 5, 5, 2),
        (512, 512, 5, 5, 2),
        (512, 512, 11, 11, 5),
        (1024, 1024, 5, 5, 2),
    ]
    
    for H, W, Kh, Kw, padding in configs:
        benchmark_input_backward_single(H, W, Kh, Kw, padding, 'float32')


def benchmark_bias_backward_single(N, C, H, W, dtype_name, num_runs=5):
    """Benchmark a single bias backward configuration."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Input: ({N}, {C}, {H}, {W}) | {dtype_name.upper()}")
    print_table_header()
    
    np.random.seed(42)
    grad_out = np.random.randn(N, C, H, W).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        numpy_result = np.sum(grad_out.astype(np.float32), axis=(0, 2, 3))
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.median(times)
    print_row("NumPy (CPU)", numpy_time, "1.00x", "Ground Truth")
    
    # PyTorch (GPU)
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        
        for _ in range(3):
            _ = t_grad.sum(dim=(0, 2, 3))
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_result = t_grad.sum(dim=(0, 2, 3))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
        print_row("PyTorch (GPU)", pt_time, f"{numpy_time/pt_time:.2f}x", f"{pt_error:.2e}")
    
    # MacroTorch (GPU) - Pre-allocated for accurate timing
    d_input = cuda.to_device(grad_out)
    d_output = cuda.device_array(C, dtype=np.float32)
    
    for _ in range(3):
        _ = conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
        cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_result = d_output.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result).max()
    print_row("MacroTorch (GPU)", mt_time, f"{numpy_time/mt_time:.2f}x", f"{mt_error:.2e}")
    
    print(f"{'-'*80}")


def benchmark_bias_backward_suite():
    """Run bias backward benchmarks."""
    print_header("BIAS BACKWARD BENCHMARKS")
    
    configs = [
        # (N, C, H, W)
        (8, 64, 28, 28),
        (16, 128, 32, 32),
        (32, 128, 64, 64),
        (32, 256, 64, 64),
        (64, 512, 32, 32),
    ]
    
    print("\n--- FP32 ---")
    for N, C, H, W in configs:
        benchmark_bias_backward_single(N, C, H, W, 'float32')
    
    print("\n--- FP16 ---")
    for N, C, H, W in configs[:3]:
        benchmark_bias_backward_single(N, C, H, W, 'float16')


def benchmark_layer_api():
    """Benchmark the Conv2d layer class."""
    print_header("CONV2D LAYER API BENCHMARKS")
    
    print("\n  Testing Conv2d layer with learnable weights")
    np.random.seed(42)
    
    conv = Conv2d(1, 1, kernel_size=5, padding=2, bias=True)
    x = np.random.randn(256, 256).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        _ = conv(x)
    
    # Forward
    times = []
    for _ in range(10):
        start = time.perf_counter()
        output = conv(x)
        times.append((time.perf_counter() - start) * 1000)
    
    print(f"\n  Layer: {conv}")
    print(f"  Input: {x.shape} | Output: {output.shape}")
    print(f"  Forward Time: {np.median(times):.4f} ms (median of 10 runs)")
    
    # Backward
    grad_out = np.ones_like(output)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        grad_input = conv.backward(grad_out)
        times.append((time.perf_counter() - start) * 1000)
    
    print(f"  Backward Time: {np.median(times):.4f} ms (median of 10 runs)")
    print(f"  Gradient Shape: {grad_input.shape}")


def main():
    print("\n" + "="*80)
    print(" MacroTorch Comprehensive Benchmark Suite")
    print("="*80)
    
    if TORCH_AVAILABLE:
        print("\n  PyTorch: AVAILABLE (will compare against)")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("\n  PyTorch: NOT INSTALLED")
        print("  Install with: pip install macrotorch[benchmark]")
    
    benchmark_forward_suite()
    benchmark_input_backward_suite()
    benchmark_bias_backward_suite()
    benchmark_layer_api()
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
