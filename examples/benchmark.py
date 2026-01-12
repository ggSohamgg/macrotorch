"""
MacroTorch Multi-Channel Benchmark Suite

Benchmarks all MacroTorch 4D CUDA kernels against CPU baselines and optionally PyTorch.

Usage:
    pip install -e .[benchmark]
    python examples/benchmark.py
"""

import numpy as np
import time
from numba import cuda
import math

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward, conv2d_weight_backward, Conv2d, relu, relu_backward, maxpool2d_forward
from macrotorch.kernels import WEIGHT_KERNEL, RELU_FORWARD, MAXPOOL2D_FORWARD


def numpy_conv2d_4d(A, K, padding=0, bias=None):
    """Slow NumPy implementation of 4D convolution for ground truth."""
    N, Cin, H, W = A.shape
    Cout, Cin_K, Kh, Kw = K.shape
    assert Cin == Cin_K
    
    if padding > 0:
        A_padded = np.pad(A, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        A_padded = A
        
    out_h = H - Kh + 1 + 2 * padding
    out_w = W - Kw + 1 + 2 * padding
    out = np.zeros((N, Cout, out_h, out_w), dtype=np.float32)
    
    for n in range(N):
        for c_out in range(Cout):
            for c_in in range(Cin):
                for i in range(out_h):
                    for j in range(out_w):
                        out[n, c_out, i, j] += np.sum(
                            A_padded[n, c_in, i:i+Kh, j:j+Kw] * K[c_out, c_in]
                        )
            if bias is not None:
                out[n, c_out] += bias[c_out]
    return out


def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def benchmark_forward(dtype_name='float32', num_runs=10):
    """Benchmark forward convolution (4D)."""
    # Small dimensions for CPU ground truth to be fast
    N, C, H, W = 2, 4, 32, 32
    Cout = 8
    Kh, Kw = 3, 3
    padding = 1
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch Size:   {N}")
    print(f"    In Channels:  {C}")
    print(f"    Out Channels: {Cout}")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    A = np.random.randn(N, C, H, W).astype(np_dtype)
    K = np.random.randn(Cout, C, Kh, Kw).astype(np_dtype)
    bias = np.random.randn(Cout).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    start = time.perf_counter()
    numpy_out = numpy_conv2d_4d(A.astype(np.float32), K.astype(np.float32), padding, bias.astype(np.float32))
    numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_A = torch.from_numpy(A).cuda().to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().to(pt_dtype)
        t_bias = torch.from_numpy(bias).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = torch.nn.functional.conv2d(t_A, t_K, t_bias, padding=padding)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_out = torch.nn.functional.conv2d(t_A, t_K, t_bias, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_error = np.abs(pt_out.cpu().numpy().astype(np.float32) - numpy_out).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_forward(A, K, padding=padding, bias=bias)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            mt_out = conv2d_forward(A, K, padding=padding, bias=bias)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            mt_out = conv2d_forward(A, K, padding=padding, bias=bias)
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_error = np.abs(mt_out - numpy_out).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'N/A':<10} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")


def benchmark_input_backward(dtype_name='float32', num_runs=10):
    """Benchmark input gradient computation (4D)."""
    N, C, H, W = 2, 4, 32, 32
    Cout = 8
    Kh, Kw = 3, 3
    padding = 1
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch Size:   {N}")
    print(f"    In Channels:  {C}")
    print(f"    Out Channels: {Cout}")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    A = np.random.randn(N, C, H, W).astype(np_dtype)
    K = np.random.randn(Cout, C, Kh, Kw).astype(np_dtype)
    
    output = conv2d_forward(A, K, padding=padding)
    grad_out = np.random.randn(*output.shape).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    def numpy_input_backward(grad_out, K, padding):
        N, Cout, out_h, out_w = grad_out.shape
        Cout_K, Cin, Kh, Kw = K.shape
        H_in = out_h + Kh - 1 - 2 * padding
        W_in = out_w + Kw - 1 - 2 * padding
        grad_A = np.zeros((N, Cin, H_in, W_in), dtype=np.float32)
        
        for n in range(N):
            for c_out in range(Cout):
                for c_in in range(Cin):
                    for i in range(out_h):
                        for j in range(out_w):
                            for u in range(Kh):
                                for v in range(Kw):
                                    in_r = i + u - padding
                                    in_c = j + v - padding
                                    if 0 <= in_r < H_in and 0 <= in_c < W_in:
                                        grad_A[n, c_in, in_r, in_c] += grad_out[n, c_out, i, j] * K[c_out, c_in, u, v]
        return grad_A

    start = time.perf_counter()
    numpy_result = numpy_input_backward(grad_out.astype(np.float32), K.astype(np.float32), padding)
    numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        t_K = torch.from_numpy(K).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = torch.nn.grad.conv2d_input((N, C, H, W), t_K, t_grad, padding=padding)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.nn.grad.conv2d_input((N, C, H, W), t_K, t_grad, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_input_backward(grad_out, K, padding=padding)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            mt_grad_in = conv2d_input_backward(grad_out, K, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            mt_grad_in = conv2d_input_backward(grad_out, K, padding=padding)
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_error = np.abs(mt_grad_in - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'N/A':<10} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")


def benchmark_bias_backward(dtype_name='float32', num_runs=10):
    """Benchmark bias gradient computation (4D)."""
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
    start = time.perf_counter()
    numpy_result = np.sum(grad_out.astype(np.float32), axis=(0, 2, 3))
    numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_grad = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = t_grad.sum(dim=(0, 2, 3))
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = t_grad.sum(dim=(0, 2, 3))
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_bias_backward(grad_out)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            mt_result = conv2d_bias_backward(grad_out)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            mt_result = conv2d_bias_backward(grad_out)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    mt_error = np.abs(mt_result - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'N/A':<10} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")


def benchmark_weight_backward(N, C, Cout, H, W, Kh, Kw, padding, dtype_name='float32', use_numpy=True, num_runs=10):
    """Benchmark weight gradient computation (4D)."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Channels:     {C} -> {Cout}")
    print(f"    Input:        {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    A = np.random.randn(N, C, H, W).astype(np_dtype)
    H_out = H - Kh + 1 + 2 * padding
    W_out = W - Kw + 1 + 2 * padding
    grad_out = np.random.randn(N, Cout, H_out, W_out).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    numpy_time = None
    numpy_result = None
    if use_numpy:
        def numpy_weight_backward(grad_out, A, Kh, Kw, padding):
            N, Cout, H_out, W_out = grad_out.shape
            _, Cin, H_in, W_in = A.shape
            grad_W = np.zeros((Cout, Cin, Kh, Kw), dtype=np.float32)
            A_padded = np.pad(A, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
            for n in range(N):
                for co in range(Cout):
                    for ci in range(Cin):
                        for u in range(Kh):
                            for v in range(Kw):
                                grad_W[co, ci, u, v] += np.sum(
                                    grad_out[n, co] * A_padded[n, ci, u:u+H_out, v:v+W_out]
                                )
            return grad_W

        start = time.perf_counter()
        numpy_result = numpy_weight_backward(grad_out.astype(np.float32), A.astype(np.float32), Kh, Kw, padding)
        numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    if TORCH_AVAILABLE:
        t_A = torch.from_numpy(A).cuda().to(torch.float32)
        t_grad_out = torch.from_numpy(grad_out).cuda().to(torch.float32)
        
        for _ in range(5):
            _ = torch.nn.grad.conv2d_weight(t_A, (Cout, C, Kh, Kw), t_grad_out, padding=padding)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.nn.grad.conv2d_weight(t_A, (Cout, C, Kh, Kw), t_grad_out, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_std = np.std(times)
        
        if use_numpy:
            pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_weight_backward(grad_out, A, padding=padding)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            mt_result = conv2d_weight_backward(grad_out, A, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            mt_result = conv2d_weight_backward(grad_out, A, padding=padding)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    
    if use_numpy:
        mt_error = np.abs(mt_result - numpy_result).max()
    elif TORCH_AVAILABLE:
        mt_error = np.abs(mt_result - pt_result.cpu().numpy()).max()
    
    # Results
    print(f"\n  Results:")
    if use_numpy:
        print(f"  {'-'*74}")
        print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
        print(f"  {'-'*74}")
        print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth'}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}'}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}'}")
        print(f"  {'-'*74}")
    else:
        print(f"  {'-'*50}")
        print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10}")
        print(f"  {'-'*50}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f}")
        print(f"  {'-'*50}")


def benchmark_relu_forward(size=(32, 64, 128, 128), dtype_name='float32', num_runs=10):
    """Benchmark ReLU forward pass (4D)."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    print(f"\n  Configuration: Size={size}, Precision={dtype_name.upper()}, Runs={num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(*size).astype(np_dtype)
    start = time.perf_counter()
    numpy_result = np.maximum(0, x)
    numpy_time = (time.perf_counter() - start) * 1000
    
    if TORCH_AVAILABLE:
        t_x = torch.from_numpy(x).cuda().to(torch.float32 if dtype_name=='float32' else torch.float16)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(num_runs):
            start_event.record()
            _ = torch.relu(t_x)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        
    for _ in range(5): _ = relu(x)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_runs):
        start_event.record()
        mt_out = relu(x)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    mt_time = np.median(times)
    mt_error = np.abs(mt_out - numpy_result).max()
    
    print(f"\n  Results:")
    print(f"  {'-'*60}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error':<12}")
    print(f"  {'-'*60}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth'}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {'~0'}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}'}")


def benchmark_relu_backward(size=(32, 64, 128, 128), dtype_name='float32', num_runs=10):
    """Benchmark ReLU backward pass (4D)."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    print(f"\n  Configuration: Size={size}, Precision={dtype_name.upper()}, Runs={num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(*size).astype(np_dtype)
    grad_out = np.random.randn(*size).astype(np_dtype)
    start = time.perf_counter()
    numpy_result = grad_out * (x > 0).astype(np.float32)
    numpy_time = (time.perf_counter() - start) * 1000
    
    if TORCH_AVAILABLE:
        t_x = torch.from_numpy(x).cuda().requires_grad_(True)
        t_grad_out = torch.from_numpy(grad_out).cuda()
        t_out = torch.relu(t_x)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(num_runs):
            start_event.record()
            t_out.backward(t_grad_out, retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            t_x.grad = None
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        
    for _ in range(5): _ = relu_backward(x, grad_out)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_runs):
        start_event.record()
        mt_grad_in = relu_backward(x, grad_out)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    mt_time = np.median(times)
    mt_error = np.abs(mt_grad_in - numpy_result).max()
    
    print(f"\n  Results:")
    print(f"  {'-'*60}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error':<12}")
    print(f"  {'-'*60}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth'}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {'~0'}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}'}")


def benchmark_maxpool2d(size=(512, 512), pool_size=2, dtype_name='float32', num_runs=10):
    """Benchmark MaxPool2D forward (2D Fallback)."""
    h, w = size
    print(f"\n  Configuration: HW={h}x{w}, Pool={pool_size}, Dtype={dtype_name.upper()}")
    x_2d = np.random.randn(h, w).astype(np.float32)
    start = time.perf_counter()
    def numpy_maxpool(x, p):
        oh, ow = x.shape[0]//p, x.shape[1]//p
        out = np.zeros((oh, ow), dtype=x.dtype)
        for i in range(oh):
            for j in range(ow):
                out[i,j] = x[i*p:(i+1)*p, j*p:(j+1)*p].max()
        return out
    numpy_res = numpy_maxpool(x_2d, pool_size)
    numpy_time = (time.perf_counter()-start)*1000
    for _ in range(5): _, _ = maxpool2d_forward(x_2d, pool_size)
    start = time.perf_counter()
    mt_out, _ = maxpool2d_forward(x_2d, pool_size)
    mt_time = (time.perf_counter()-start)*1000
    mt_error = np.abs(mt_out - numpy_res).max()
    print(f"\n  Results:")
    print(f"  {'-'*60}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error'}")
    print(f"  {'-'*60}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth'}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}'}")


def main():
    print("\n" + "="*80)
    print(" MacroTorch Multi-Channel Benchmark Suite")
    print("="*80)
    
    if TORCH_AVAILABLE:
        print(f"\n  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Device:     {torch.cuda.get_device_name(0)}")
    
    print_header("FORWARD PASS (4D)")
    benchmark_forward(dtype_name='float32')
    benchmark_forward(dtype_name='float16')
    
    print_header("INPUT BACKWARD (4D)")
    benchmark_input_backward(dtype_name='float32')
    benchmark_input_backward(dtype_name='float16')
    
    print_header("BIAS BACKWARD (4D)")
    benchmark_bias_backward(dtype_name='float32')
    benchmark_bias_backward(dtype_name='float16')
    
    print_header("WEIGHT BACKWARD (SMALL) (4D)")
    benchmark_weight_backward(N=2, C=4, Cout=8, H=32, W=32, Kh=3, Kw=3, padding=1, dtype_name='float32', use_numpy=True)
    benchmark_weight_backward(N=2, C=4, Cout=8, H=32, W=32, Kh=3, Kw=3, padding=1, dtype_name='float16', use_numpy=True)
    
    print_header("WEIGHT BACKWARD (LARGE) (4D)")
    benchmark_weight_backward(N=8, C=32, Cout=64, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float32', use_numpy=False)
    benchmark_weight_backward(N=8, C=32, Cout=64, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float16', use_numpy=False)
    
    print_header("RELU FORWARD (4D)")
    benchmark_relu_forward()
    
    print_header("RELU BACKWARD (4D)")
    benchmark_relu_backward()
    
    print_header("MAXPOOL2D (2D fallback)")
    benchmark_maxpool2d()
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
