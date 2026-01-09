"""
MacroTorch Benchmark Suite

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

from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward, conv2d_weight_backward, Conv2d


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


def benchmark_forward(dtype_name='float32', num_runs=10):
    """Benchmark forward convolution."""
    H, W = 512, 512
    Kh, Kw = 5, 5
    padding = 2
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
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
    scipy_std = np.std(times)
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_A = torch.from_numpy(A).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        
        for _ in range(5):
            _ = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_out = torch.nn.functional.conv2d(t_A, t_K, padding=padding)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_error = np.abs(pt_out.squeeze().cpu().numpy().astype(np.float32) - scipy_out).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_forward(A, K, padding=padding)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        mt_out = conv2d_forward(A, K, padding=padding)
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_error = np.abs(mt_out - scipy_out).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'SciPy (CPU)':<18} | {scipy_time:<12.4f} | {scipy_std:<10.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{scipy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{scipy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_input_backward(dtype_name='float32', num_runs=10):
    """Benchmark input gradient computation."""
    H, W = 512, 512
    Kh, Kw = 5, 5
    padding = 2
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    A = np.random.randn(H, W).astype(np_dtype)
    K = np.random.randn(Kh, Kw).astype(np_dtype)
    K_flipped = K[::-1, ::-1].copy()
    
    output = conv2d_forward(A, K, padding=padding)
    grad_out = np.random.randn(*output.shape).astype(np_dtype)
    
    # SciPy (CPU) - Ground Truth (full convolution with flipped kernel)
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        scipy_result = correlate2d(grad_out.astype(np.float32), K_flipped.astype(np.float32), mode='full')
        # Crop to input size
        start_h = (scipy_result.shape[0] - H) // 2
        start_w = (scipy_result.shape[1] - W) // 2
        scipy_result = scipy_result[start_h:start_h+H, start_w:start_w+W]
        times.append((time.perf_counter() - start) * 1000)
    scipy_time = np.median(times)
    scipy_std = np.std(times)
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().unsqueeze(0).unsqueeze(0).to(pt_dtype)
        
        for _ in range(5):
            _ = torch.nn.functional.conv_transpose2d(t_grad, t_K, padding=padding)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_result = torch.nn.functional.conv_transpose2d(t_grad, t_K, padding=padding)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_out_np = pt_result.squeeze().cpu().numpy().astype(np.float32)
        pt_error = np.abs(pt_out_np - scipy_result).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_input_backward(grad_out, K, padding=padding)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        grad_input = conv2d_input_backward(grad_out, K, padding=padding)
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_error = np.abs(grad_input - scipy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'SciPy (CPU)':<18} | {scipy_time:<12.4f} | {scipy_std:<10.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{scipy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{scipy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_bias_backward(dtype_name='float32', num_runs=10):
    """Benchmark bias gradient computation."""
    N, C, H, W = 32, 128, 64, 64
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Channels:     {C}")
    print(f"    Spatial:      {H} x {W}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    grad_out = np.random.randn(N, C, H, W).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        numpy_result = np.sum(grad_out.astype(np.float32), axis=(0, 2, 3))
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.median(times)
    numpy_std = np.std(times)
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = t_grad.sum(dim=(0, 2, 3))
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_result = t_grad.sum(dim=(0, 2, 3))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU) - Pre-allocated
    d_input = cuda.to_device(grad_out)
    d_output = cuda.device_array(C, dtype=np.float32)
    
    for _ in range(5):
        _ = conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
        cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_result = d_output.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {numpy_std:<10.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_weight_backward(N, H, W, Kh, Kw, padding, dtype_name='float32', use_scipy=True, num_runs=10):
    """Benchmark weight gradient computation."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Input:        {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    
    # Create input and grad_out
    A = np.random.randn(N, H, W).astype(np_dtype)
    H_out = H - Kh + 1 + 2 * padding
    W_out = W - Kw + 1 + 2 * padding
    grad_out = np.random.randn(N, H_out, W_out).astype(np_dtype)
    
    # SciPy (CPU) - Ground Truth (only for small inputs)
    scipy_time, scipy_std = None, None
    if use_scipy:
        # Compute weight gradient using correlation
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            scipy_result = np.zeros((Kh, Kw), dtype=np.float32)
            for n in range(N):
                for i in range(H_out):
                    for j in range(W_out):
                        in_i = i - padding
                        in_j = j - padding
                        for u in range(Kh):
                            for v in range(Kw):
                                ii = in_i + u
                                jj = in_j + v
                                if 0 <= ii < H and 0 <= jj < W:
                                    scipy_result[u, v] += grad_out[n, i, j] * A[n, ii, jj]
            times.append((time.perf_counter() - start) * 1000)
        scipy_time = np.median(times)
        scipy_std = np.std(times)
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_input = torch.from_numpy(A).cuda().unsqueeze(1).to(pt_dtype)  # (N, 1, H, W)
        t_grad_out = torch.from_numpy(grad_out).cuda().unsqueeze(1).to(pt_dtype)  # (N, 1, H_out, W_out)
        
        weight_shape = (1, 1, Kh, Kw)
        
        for _ in range(5):
            _ = torch.nn.grad.conv2d_weight(t_input, weight_shape, t_grad_out, padding=padding)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            pt_result = torch.nn.grad.conv2d_weight(t_input, weight_shape, t_grad_out, padding=padding)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        pt_time = np.median(times)
        pt_std = np.std(times)
        
        pt_out_np = pt_result.squeeze().cpu().numpy().astype(np.float32)
        if use_scipy:
            pt_error = np.abs(pt_out_np - scipy_result).max()
    
    # MacroTorch (GPU) - Pre-allocate memory for fair comparison
    if A.ndim == 2:
        A_reshaped = A.reshape(1, *A.shape)
        grad_out_reshaped = grad_out.reshape(1, *grad_out.shape)
    else:
        A_reshaped = A
        grad_out_reshaped = grad_out

    d_A = cuda.to_device(A_reshaped)
    d_grad_out = cuda.to_device(grad_out_reshaped)
    d_grad_W = cuda.device_array((Kh, Kw), dtype=np.float32)

    # Warmup
    for _ in range(5):
        cuda.to_device(np.zeros((Kh, Kw), dtype=np.float32), to=d_grad_W)
        conv2d_weight_backward(None, None, padding=padding, 
                               d_grad_out=d_grad_out, d_A=d_A, d_grad_W=d_grad_W)

    times = []
    for _ in range(num_runs):
        cuda.to_device(np.zeros((Kh, Kw), dtype=np.float32), to=d_grad_W)
        start = time.perf_counter()
        conv2d_weight_backward(None, None, padding=padding,
                               d_grad_out=d_grad_out, d_A=d_A, d_grad_W=d_grad_W)
        cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    
    mt_result = d_grad_W.copy_to_host()
    
    # Compute error
    mt_error = None
    if use_scipy:
        mt_error = np.abs(mt_result - scipy_result).max()
    elif TORCH_AVAILABLE and pt_out_np is not None:
        # Compare against PyTorch when SciPy not available
        mt_error = np.abs(mt_result - pt_out_np).max()
    
    # Results
    print(f"\n  Results:")
    if use_scipy:
        print(f"  {'-'*74}")
        print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
        print(f"  {'-'*74}")
        print(f"  {'SciPy (CPU)':<18} | {scipy_time:<12.4f} | {scipy_std:<10.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{scipy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{scipy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
        print(f"  {'-'*74}")
    else:
        # No SciPy - show error vs PyTorch if available
        if TORCH_AVAILABLE and mt_error is not None:
            print(f"  {'-'*74}")
            print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Error vs PT':<12}")
            print(f"  {'-'*74}")
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {'Reference':<12}")
            print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{mt_error:.2e}':<12}")
            print(f"  {'-'*74}")
        else:
            print(f"  {'-'*50}")
            print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10}")
            print(f"  {'-'*50}")
            if TORCH_AVAILABLE:
                print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f}")
            print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f}")
            print(f"  {'-'*50}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def main():
    print("\n" + "="*80)
    print(" MacroTorch Benchmark Suite")
    print("="*80)
    
    if TORCH_AVAILABLE:
        print(f"\n  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Device:     {torch.cuda.get_device_name(0)}")
    else:
        print("\n  PyTorch: NOT INSTALLED (install with: pip install macrotorch[benchmark])")
    
    # Forward Pass
    print_header("FORWARD PASS - FP32")
    benchmark_forward(dtype_name='float32')
    
    print_header("FORWARD PASS - FP16")
    benchmark_forward(dtype_name='float16')
    
    # Input Backward
    print_header("INPUT BACKWARD - FP32")
    benchmark_input_backward(dtype_name='float32')
    
    print_header("INPUT BACKWARD - FP16")
    benchmark_input_backward(dtype_name='float16')
    
    # Bias Backward
    print_header("BIAS BACKWARD - FP32")
    benchmark_bias_backward(dtype_name='float32')
    
    print_header("BIAS BACKWARD - FP16")
    benchmark_bias_backward(dtype_name='float16')
    
    # Weight Backward - Small (with SciPy validation)
    print_header("WEIGHT BACKWARD (SMALL) - FP32")
    benchmark_weight_backward(N=8, H=64, W=64, Kh=5, Kw=5, padding=2, dtype_name='float32', use_scipy=True)
    
    print_header("WEIGHT BACKWARD (SMALL) - FP16")
    benchmark_weight_backward(N=8, H=64, W=64, Kh=5, Kw=5, padding=2, dtype_name='float16', use_scipy=True)
    
    # Weight Backward - Large (no SciPy, too slow)
    print_header("WEIGHT BACKWARD (LARGE) - FP32")
    benchmark_weight_backward(N=128, H=256, W=256, Kh=5, Kw=5, padding=2, dtype_name='float32', use_scipy=False)
    
    print_header("WEIGHT BACKWARD (LARGE) - FP16")
    benchmark_weight_backward(N=128, H=256, W=256, Kh=5, Kw=5, padding=2, dtype_name='float16', use_scipy=False)
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
