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

from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward, conv2d_weight_backward, Conv2d, relu, relu_backward, maxpool2d_forward, softmax_forward, softmax_backward, matmul
from macrotorch.kernels import WEIGHT_KERNEL, RELU_FORWARD, MAXPOOL2D_FORWARD, SOFTMAX_FORWARD, SOFTMAX_BACKWARD, matmul_tiled
import math


def scipy_conv2d(A, K, padding=0, bias=None):
    """Slow SciPy implementation of 4D convolution for ground truth."""
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
    
    # Loop over batch and channels
    for n in range(N):
        for c_out in range(Cout):
            for c_in in range(Cin):
                # correlate2d 'valid' is equivalent to conv2d without flipping if we flip K manually 
                # but standard definition: conv(A, K) = corr(A, flip(K))
                # Here we use correlate2d with un-flipped K to mean correlation, or flipped if convolution.
                # PyTorch conv2d is correlation. So we use correlate2d directly with K.
                out[n, c_out] += correlate2d(A_padded[n, c_in], K[c_out, c_in], mode='valid')
            if bias is not None:
                out[n, c_out] += bias[c_out]
    return out


def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def benchmark_forward(dtype_name='float32', num_runs=10):
    """Benchmark forward convolution (4D)."""
    # Reduced sizes for CPU ground truth feasibility
    N, C, H, W = 2, 4, 64, 64
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
    
    # SciPy (CPU) - Ground Truth
    times = []
    # Run fewer times for CPU
    for _ in range(3):
        start = time.perf_counter()
        scipy_out = scipy_conv2d(A.astype(np.float32), K.astype(np.float32), padding, bias.astype(np.float32))
        times.append((time.perf_counter() - start) * 1000)
    scipy_time = np.median(times)
    scipy_std = np.std(times)
    
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
        pt_error = np.abs(pt_out.cpu().numpy().astype(np.float32) - scipy_out).max()
    
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
    """Benchmark input gradient computation (4D)."""
    N, C, H, W = 2, 4, 64, 64
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
    
    # SciPy (CPU) - Ground Truth
    def scipy_input_backward(grad_out, K, padding):
        N, Cout, out_h, out_w = grad_out.shape
        Cout_K, Cin, Kh, Kw = K.shape
        H_in = out_h + Kh - 1 - 2 * padding
        W_in = out_w + Kw - 1 - 2 * padding
        grad_A = np.zeros((N, Cin, H_in, W_in), dtype=np.float32)
        
        # Backward is convolution of grad_out with rotated K (transposed convolution)
        # Using full correlation/convolution
        for n in range(N):
            for c_in in range(Cin):
                for c_out in range(Cout):
                    # Rotate the kernel 180 degrees
                    k_rot = K[c_out, c_in, ::-1, ::-1]
                    # Full convolution
                    grad_slice = correlate2d(grad_out[n, c_out], k_rot, mode='full')
                    
                    # Crop to input size
                    start_h = (grad_slice.shape[0] - H_in) // 2
                    start_w = (grad_slice.shape[1] - W_in) // 2
                    grad_A[n, c_in] += grad_slice[start_h:start_h+H_in, start_w:start_w+W_in]
        return grad_A

    times = []
    # Run fewer times for CPU
    for _ in range(3):
        start = time.perf_counter()
        scipy_result = scipy_input_backward(grad_out.astype(np.float32), K.astype(np.float32), padding)
        times.append((time.perf_counter() - start) * 1000)
    scipy_time = np.median(times)
    scipy_std = np.std(times)
    
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
        pt_out_np = pt_result.cpu().numpy().astype(np.float32)
        pt_error = np.abs(pt_out_np - scipy_result).max()
    
    # MacroTorch (GPU)
    for _ in range(5):
        _ = conv2d_input_backward(grad_out, K, padding=padding)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            grad_input = conv2d_input_backward(grad_out, K, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
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
    
    # MacroTorch (GPU) - Pre-allocated
    d_input = cuda.to_device(grad_out)
    d_output = cuda.device_array(C, dtype=np.float32)
    
    for _ in range(5):
        _ = conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            conv2d_bias_backward(None, d_grad_out=d_input, d_grad_bias=d_output)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
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


def benchmark_weight_backward(N, C, Cout, H, W, Kh, Kw, padding, dtype_name='float32', use_scipy=True, num_runs=10):
    """Benchmark weight gradient computation."""
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
    
    # NumPy/SciPy (CPU) - Ground Truth (single run)
    numpy_time = None
    numpy_result = None
    if use_scipy:
        def numpy_weight_backward(grad_out, A, Kh, Kw, padding):
            grad_W = np.zeros((Cout, C, Kh, Kw), dtype=np.float32)
            A_padded = np.pad(A, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
            
            for n in range(N):
                for co in range(Cout):
                    for ci in range(C):
                        grad_W[co, ci] += correlate2d(A_padded[n, ci], grad_out[n, co], mode='valid')
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
        
        if use_scipy:
            pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU)
    # Direct kernel launch if A is 4D... wait, previous code had shape checks.
    # Current code is purely 4D.
    d_A = cuda.to_device(A)
    d_grad_out = cuda.to_device(grad_out)
    
    # Grid configuration
    threads = (16, 16)
    blocks = (
        math.ceil(W_out / 16),
        math.ceil(H_out / 16),
        Cout * C
    )

    # Warmup
    zeros_host = np.zeros((Cout, C, Kh, Kw), dtype=np.float32)
    d_grad_W = cuda.to_device(zeros_host)
    
    for _ in range(5):
        d_grad_W.copy_to_device(zeros_host)
        WEIGHT_KERNEL[blocks, threads](d_A, d_grad_out, padding, d_grad_W)
    cuda.synchronize()

    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            d_grad_W.copy_to_device(zeros_host)
            cuda.synchronize() # Ensure zeroing is complete before start (though copy is sync usually on same stream)
            start_event.record()
            WEIGHT_KERNEL[blocks, threads](d_A, d_grad_out, padding, d_grad_W)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            d_grad_W.copy_to_device(zeros_host)
            cuda.synchronize()
            start = time.perf_counter()
            WEIGHT_KERNEL[blocks, threads](d_A, d_grad_out, padding, d_grad_W)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    mt_std = np.std(times)
    
    mt_result = d_grad_W.copy_to_host()
    
    mt_error = None
    if use_scipy and numpy_result is not None:
        mt_error = np.abs(mt_result - numpy_result).max()
    elif TORCH_AVAILABLE and pt_result is not None:
        mt_error = np.abs(mt_result - pt_result.cpu().numpy()).max()
    
    # Results
    print(f"\n  Results:")
    if use_scipy and numpy_time is not None:
        print(f"  {'-'*74}")
        print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
        print(f"  {'-'*74}")
        print(f"  {'SciPy (CPU)':<18} | {numpy_time:<12.2f} | {'1.00x':<10} | {'Ground Truth':<12}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
        print(f"  {'-'*74}")
    else:
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


def benchmark_weight_backward_2d_legacy(N, H, W, Kh, Kw, padding, dtype_name='float32', use_scipy=True, num_runs=10):
    """Benchmark the 2D-only weight backward kernel (N, H, W) input."""
    from macrotorch.kernels import WEIGHT_KERNEL_2D_LEGACY
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Input:        {H} x {W}")
    print(f"    Kernel Size:  {Kh} x {Kw}")
    print(f"    Padding:      {padding}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    # 3D Tensors: (N, H, W)
    A = np.random.randn(N, H, W).astype(np_dtype)
    H_out = H - Kh + 1 + 2 * padding
    W_out = W - Kw + 1 + 2 * padding
    grad_out = np.random.randn(N, H_out, W_out).astype(np_dtype)
    
    # SciPy (CPU) - Ground Truth
    numpy_time = None
    numpy_result = None
    if use_scipy:
        start = time.perf_counter()
        grad_W = np.zeros((Kh, Kw), dtype=np.float32)
        A_padded = np.pad(A, ((0,0), (padding, padding), (padding, padding)), mode='constant')
        
        for n in range(N):
            grad_W += correlate2d(A_padded[n].astype(np.float32), grad_out[n].astype(np.float32), mode='valid')
        numpy_result = grad_W
        numpy_time = (time.perf_counter() - start) * 1000

    # PyTorch (GPU)
    pt_time, pt_std, pt_error = None, None, None
    pt_result = None
    if TORCH_AVAILABLE:
        # PyTorch needs 4D tensors: (N, 1, H, W) to simulate 2D batch
        t_A = torch.from_numpy(A).unsqueeze(1).cuda().to(torch.float32)
        t_grad_out = torch.from_numpy(grad_out).unsqueeze(1).cuda().to(torch.float32)
        
        # We need weight grad wrt kernel (1, 1, Kh, Kw)
        # But we sum over batch. To match conv2d_weight behavior for single channel in/out:
        # PyTorch conv2d_weight expects input (N, Cin, H, W) and returns (Cout, Cin, Kh, Kw)
        # Here Cin=1, Cout=1.
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_out = torch.nn.grad.conv2d_weight(t_A, (1, 1, Kh, Kw), t_grad_out, padding=padding)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_std = np.std(times)
        pt_result = pt_out.squeeze().cpu().numpy()
        
        if use_scipy:
            pt_error = np.abs(pt_result - numpy_result).max()
    
    # MacroTorch (GPU)
    d_A = cuda.to_device(A)
    d_grad_out = cuda.to_device(grad_out)
    
    # 2D Kernel Weights: (Kh, Kw)
    zeros_host = np.zeros((Kh, Kw), dtype=np.float32)
    d_grad_W = cuda.to_device(zeros_host)
    
    # Grid Config
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = Kh * Kw
    blocks = (blocks_x, blocks_y, blocks_z)
    
    # Warmup
    for _ in range(5):
        d_grad_W.copy_to_device(zeros_host)
        WEIGHT_KERNEL_2D_LEGACY[blocks, threads](d_A, d_grad_out, padding, d_grad_W)
    cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        d_grad_W.copy_to_device(zeros_host)
        cuda.synchronize()
        start = time.perf_counter()
        WEIGHT_KERNEL_2D_LEGACY[blocks, threads](d_A, d_grad_out, padding, d_grad_W)
        cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mt_time = np.median(times)
    mt_std = np.std(times)
    
    mt_result = d_grad_W.copy_to_host()
    mt_error = None
    if use_scipy and numpy_result is not None:
        mt_error = np.abs(mt_result - numpy_result).max()
    elif TORCH_AVAILABLE and pt_result is not None:
        mt_error = np.abs(mt_result - pt_result.cpu().numpy()).max()
    
    print(f"\n  Results:")
    if numpy_time is not None:
        print(f"  {'-'*74}")
        print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error':<12}")
        print(f"  {'-'*74}")
        print(f"  {'SciPy (CPU)':<18} | {numpy_time:<12.2f} | {'1.00x':<10} | {'Ground Truth':<12}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
        print(f"  {'MacroTorch (2D)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
        print(f"  {'-'*74}")
    else:
        print(f"  {'-'*50}")
        print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10}")
        print(f"  {'-'*50}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {pt_std:<10.4f}")
        print(f"  {'MacroTorch (2D)':<18} | {mt_time:<12.4f} | {mt_std:<10.4f}")
        print(f"  {'-'*50}")


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
    benchmark_weight_backward(N=2, C=4, Cout=8, H=32, W=32, Kh=3, Kw=3, padding=1, dtype_name='float32', use_scipy=True)
    
    print_header("WEIGHT BACKWARD (SMALL) - FP16")
    benchmark_weight_backward(N=2, C=4, Cout=8, H=32, W=32, Kh=3, Kw=3, padding=1, dtype_name='float16', use_scipy=True)
    
    # Weight Backward - Large (with SciPy validation)
    print_header("WEIGHT BACKWARD (LARGE) - FP32")
    benchmark_weight_backward(N=8, C=32, Cout=64, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float32', use_scipy=True)
    
    print_header("WEIGHT BACKWARD (LARGE) - FP16")
    benchmark_weight_backward(N=8, C=32, Cout=64, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float16', use_scipy=True)

    # Weight Backward - 2D Legacy
    print_header("WEIGHT BACKWARD (2D LEGACY) - FP32")
    benchmark_weight_backward_2d_legacy(N=8, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float32', use_scipy=True)
    
    print_header("WEIGHT BACKWARD (2D LEGACY) - FP16")
    benchmark_weight_backward_2d_legacy(N=8, H=128, W=128, Kh=3, Kw=3, padding=1, dtype_name='float16', use_scipy=True)
    
    # ReLU Benchmarks
    print_header("RELU FORWARD - FP32")
    benchmark_relu_forward(size=(1024, 1024), dtype_name='float32')
    
    print_header("RELU BACKWARD - FP32")
    benchmark_relu_backward(size=(1024, 1024), dtype_name='float32')
    
    # MaxPool2D Benchmarks (Small - with NumPy)
    print_header("MAXPOOL2D FORWARD (SMALL) - FP32")
    benchmark_maxpool2d_forward(N=2, C=4, H=64, W=64, pool_size=2, dtype_name='float32', use_scipy=True)
    
    # MaxPool2D Benchmarks (Large - PyTorch only)
    print_header("MAXPOOL2D FORWARD (LARGE) - FP32")
    benchmark_maxpool2d_forward(N=8, C=64, H=128, W=128, pool_size=2, dtype_name='float32', use_scipy=False)
    
    print_header("MAXPOOL2D BACKWARD (LARGE) - FP32")
    benchmark_maxpool2d_backward(N=8, C=64, H=128, W=128, pool_size=2, dtype_name='float32')
    
    # Softmax Benchmarks
    print_header("SOFTMAX FORWARD (SMALL) - FP32")
    benchmark_softmax_forward(N=8, C=10, H=1, W=1, dtype_name='float32')
    
    print_header("SOFTMAX FORWARD (LARGE) - FP32")
    benchmark_softmax_forward(N=8, C=10, H=28, W=28, dtype_name='float32')
    
    print_header("SOFTMAX BACKWARD (SMALL) - FP32")
    benchmark_softmax_backward(N=8, C=10, H=1, W=1, dtype_name='float32')
    
    print_header("SOFTMAX BACKWARD (LARGE) - FP32")
    benchmark_softmax_backward(N=8, C=10, H=28, W=28, dtype_name='float32')
    
    # MatMul Benchmarks
    print_header("MATMUL (SMALL) - FP32")
    benchmark_matmul(M=256, K=256, N=256, dtype_name='float32')
    
    print_header("MATMUL (SMALL) - FP16")
    benchmark_matmul(M=256, K=256, N=256, dtype_name='float16')
    
    print_header("MATMUL (LARGE) - FP32")
    benchmark_matmul(M=1024, K=1024, N=1024, dtype_name='float32')
    
    print_header("MATMUL (LARGE) - FP16")
    benchmark_matmul(M=1024, K=1024, N=1024, dtype_name='float16')
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE")
    print("="*80 + "\n")

def benchmark_relu_forward(size=(1024, 1024), dtype_name='float32', num_runs=10):
    """Benchmark ReLU forward pass."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Size:         {size[0]} x {size[1]}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(*size).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        numpy_result = np.maximum(0, x)
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.median(times)
    
    # PyTorch (GPU)
    pt_time = None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_x = torch.from_numpy(x).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = torch.relu(t_x)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.relu(t_x)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
    
    # MacroTorch (GPU)
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(x.shape, dtype=x.dtype)
    
    threads = 256
    blocks = math.ceil(x.size / threads)
    
    for _ in range(5):
        RELU_FORWARD[blocks, threads](d_x, d_out)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            RELU_FORWARD[blocks, threads](d_x, d_out)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            RELU_FORWARD[blocks, threads](d_x, d_out)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_result = d_out.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*60}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error':<12}")
    print(f"  {'-'*60}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {'~0':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*60}")
    
    if TORCH_AVAILABLE:
        speedup = pt_time / mt_time
        if speedup > 1:
            print(f"\n  MacroTorch vs PyTorch: {speedup:.2f}x faster")
        else:
            print(f"\n  MacroTorch vs PyTorch: {1/speedup:.2f}x slower")


def benchmark_relu_backward(size=(1024, 1024), dtype_name='float32', num_runs=10):
    """Benchmark ReLU backward pass."""
    from macrotorch.kernels import RELU_BACKWARD
    
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Size:         {size[0]} x {size[1]}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(*size).astype(np_dtype)
    grad_out = np.ones_like(x)
    
    # NumPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        numpy_result = grad_out * (x > 0).astype(np_dtype)
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.median(times)
    
    # PyTorch (GPU)
    pt_time = None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_x = torch.from_numpy(x).cuda().to(pt_dtype).requires_grad_(True)
        t_out = torch.relu(t_x)
        t_grad_out = torch.ones_like(t_out)
        
        for _ in range(5):
            t_out.backward(t_grad_out, retain_graph=True)
            t_x.grad = None
        torch.cuda.synchronize()
        
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
    
    # MacroTorch (GPU)
    d_x = cuda.to_device(x)
    d_grad_out = cuda.to_device(grad_out)
    d_grad_in = cuda.device_array(x.shape, dtype=np.float32)
    
    threads = 256
    blocks = math.ceil(x.size / threads)
    
    for _ in range(5):
        RELU_BACKWARD[blocks, threads](d_x, d_grad_out, d_grad_in)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            RELU_BACKWARD[blocks, threads](d_x, d_grad_out, d_grad_in)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            RELU_BACKWARD[blocks, threads](d_x, d_grad_out, d_grad_in)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_result = d_grad_in.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result.astype(np.float32)).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*60}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Error':<12}")
    print(f"  {'-'*60}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {'~0':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*60}")
    
    if TORCH_AVAILABLE:
        speedup = pt_time / mt_time
        if speedup > 1:
            print(f"\n  MacroTorch vs PyTorch: {speedup:.2f}x faster")
        else:
            print(f"\n  MacroTorch vs PyTorch: {1/speedup:.2f}x slower")


def benchmark_maxpool2d_forward(N, C, H, W, pool_size=2, dtype_name='float32', use_scipy=True, num_runs=10):
    """Benchmark MaxPool2D forward pass (4D)."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Channels:     {C}")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Pool Size:    {pool_size}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(N, C, H, W).astype(np_dtype)
    H_out = H // pool_size
    W_out = W // pool_size
    
    # NumPy (CPU) - Ground Truth
    numpy_time = None
    numpy_result = None
    if use_scipy:
        def numpy_maxpool4d(x, pool_size):
            N, C, H, W = x.shape
            H_out = H // pool_size
            W_out = W // pool_size
            out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
            for n in range(N):
                for c in range(C):
                    for i in range(H_out):
                        for j in range(W_out):
                            out[n, c, i, j] = x[n, c, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size].max()
            return out
        
        start = time.perf_counter()
        numpy_result = numpy_maxpool4d(x.astype(np.float32), pool_size)
        numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_error = None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_x = torch.from_numpy(x).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = torch.nn.functional.max_pool2d(t_x, kernel_size=pool_size)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.nn.functional.max_pool2d(t_x, kernel_size=pool_size)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        
        if use_scipy:
            pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result).max()
    
    # MacroTorch (GPU) - Pre-allocated arrays
    from macrotorch.kernels import MAXPOOL2D_FORWARD
    
    d_x = cuda.to_device(x)
    d_out = cuda.device_array((N, C, H_out, W_out), dtype=np_dtype)
    d_indices = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    # Warmup
    for _ in range(5):
        MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_out = d_out.copy_to_host()
    
    mt_error = None
    if use_scipy and numpy_result is not None:
        mt_error = np.abs(mt_out - numpy_result).max()
    elif TORCH_AVAILABLE:
        mt_error = np.abs(mt_out - pt_result.cpu().numpy()).max()
    
    # Results
    print(f"\n  Results:")
    if use_scipy and numpy_time is not None:
        print(f"  {'-'*74}")
        print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
        print(f"  {'-'*74}")
        print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.2f} | {'1.00x':<10} | {'Ground Truth':<12}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
        print(f"  {'-'*74}")
    else:
        print(f"  {'-'*60}")
        print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Error vs PT':<12}")
        print(f"  {'-'*60}")
        if TORCH_AVAILABLE:
            print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {'Reference':<12}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{mt_error:.2e}':<12}")
        print(f"  {'-'*60}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_maxpool2d_backward(N, C, H, W, pool_size=2, dtype_name='float32', num_runs=10):
    """Benchmark MaxPool2D backward pass (4D)."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Channels:     {C}")
    print(f"    Input Size:   {H} x {W}")
    print(f"    Pool Size:    {pool_size}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(N, C, H, W).astype(np_dtype)
    H_out = H // pool_size
    W_out = W // pool_size
    grad_out = np.random.randn(N, C, H_out, W_out).astype(np_dtype)
    
    # PyTorch (GPU)
    pt_time = None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_x = torch.from_numpy(x).cuda().to(pt_dtype).requires_grad_(True)
        t_out = torch.nn.functional.max_pool2d(t_x, kernel_size=pool_size)
        t_grad_out = torch.from_numpy(grad_out).cuda().to(pt_dtype)
        
        for _ in range(5):
            t_out.backward(t_grad_out, retain_graph=True)
            t_x.grad = None
        torch.cuda.synchronize()
        
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
        pt_grad = t_x.grad
    
    # MacroTorch (GPU) - Pre-allocated arrays
    from macrotorch.kernels import MAXPOOL2D_FORWARD, MAXPOOL2D_BACKWARD
    
    d_x = cuda.to_device(x)
    d_out = cuda.device_array((N, C, H_out, W_out), dtype=np_dtype)
    d_indices = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)
    d_grad_out = cuda.to_device(grad_out)
    d_grad_in = cuda.device_array((N, C, H, W), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    # Forward to get indices
    MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
    cuda.synchronize()
    
    # Warmup backward
    for _ in range(5):
        d_grad_in[:] = 0.0
        MAXPOOL2D_BACKWARD[blocks, threads](d_grad_out, d_indices, d_grad_in, pool_size)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            d_grad_in[:] = 0.0
            start_event.record()
            MAXPOOL2D_BACKWARD[blocks, threads](d_grad_out, d_indices, d_grad_in, pool_size)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            d_grad_in[:] = 0.0
            start = time.perf_counter()
            MAXPOOL2D_BACKWARD[blocks, threads](d_grad_out, d_indices, d_grad_in, pool_size)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    # Get result and compute error vs PyTorch
    mt_grad = d_grad_in.copy_to_host()
    mt_error = None
    if TORCH_AVAILABLE:
        # Run one more backward to get the gradient for comparison
        t_x.grad = None
        t_out.backward(t_grad_out, retain_graph=True)
        pt_grad = t_x.grad.cpu().numpy()
        mt_error = np.abs(mt_grad - pt_grad).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*70}")
    print(f"  {'Implementation':<18} | {'Median (ms)':<12} | {'Std (ms)':<10} | {'Max Error':<12}")
    print(f"  {'-'*70}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {np.std(times):<10.4f} | {'Reference':<12}")
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {np.std(times):<10.4f} | {f'{mt_error:.2e}':<12}")
    else:
        print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {np.std(times):<10.4f} | {'N/A':<12}")
    print(f"  {'-'*70}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_softmax_forward(N, C, H, W, dtype_name='float32', num_runs=10):
    """Benchmark Softmax forward pass (4D)."""
    np_dtype = np.float32
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Classes:      {C}")
    print(f"    Spatial:      {H} x {W}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(N, C, H, W).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    def numpy_softmax(x):
        x_max = x.max(axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    start = time.perf_counter()
    numpy_result = numpy_softmax(x)
    numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_error = None, None
    if TORCH_AVAILABLE:
        t_x = torch.from_numpy(x).cuda()
        
        for _ in range(5):
            _ = torch.nn.functional.softmax(t_x, dim=1)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.nn.functional.softmax(t_x, dim=1)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_error = np.abs(pt_result.cpu().numpy() - numpy_result).max()
    
    # MacroTorch (GPU) - Pre-allocated
    d_x = cuda.to_device(x)
    d_out = cuda.device_array((N, C, H, W), dtype=np_dtype)
    
    threads = (16, 16)
    blocks_x = math.ceil(W / 16)
    blocks_y = math.ceil(H / 16)
    blocks_z = N
    blocks = (blocks_x, blocks_y, blocks_z)
    
    for _ in range(5):
        SOFTMAX_FORWARD[blocks, threads](d_x, d_out)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            SOFTMAX_FORWARD[blocks, threads](d_x, d_out)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            SOFTMAX_FORWARD[blocks, threads](d_x, d_out)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_result = d_out.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_softmax_backward(N, C, H, W, dtype_name='float32', num_runs=10):
    """Benchmark Softmax backward pass (4D)."""
    np_dtype = np.float32
    
    print(f"\n  Configuration:")
    print(f"    Batch:        {N}")
    print(f"    Classes:      {C}")
    print(f"    Spatial:      {H} x {W}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    x = np.random.randn(N, C, H, W).astype(np_dtype)
    grad_out = np.random.randn(N, C, H, W).astype(np_dtype)
    
    # Compute softmax forward for probabilities
    def numpy_softmax(x):
        x_max = x.max(axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    probs = numpy_softmax(x)
    
    # NumPy (CPU) - Ground Truth
    def numpy_softmax_backward(grad_out, probs):
        # Gradient of softmax: grad_logits = probs * (grad_out - sum(grad_out * probs))
        sum_grad = (grad_out * probs).sum(axis=1, keepdims=True)
        return probs * (grad_out - sum_grad)
    
    start = time.perf_counter()
    numpy_result = numpy_softmax_backward(grad_out, probs)
    numpy_time = (time.perf_counter() - start) * 1000
    
    # PyTorch (GPU)
    pt_time, pt_error = None, None
    if TORCH_AVAILABLE:
        t_x = torch.from_numpy(x).cuda().requires_grad_(True)
        t_probs = torch.nn.functional.softmax(t_x, dim=1)
        t_grad_out = torch.from_numpy(grad_out).cuda()
        
        for _ in range(5):
            t_probs.backward(t_grad_out, retain_graph=True)
            t_x.grad = None
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            t_probs.backward(t_grad_out, retain_graph=True)
            end_event.record()
            torch.cuda.synchronize()
            t_x.grad = None
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        
        # Get PyTorch result for error comparison
        t_probs.backward(t_grad_out, retain_graph=True)
        pt_result = t_x.grad.cpu().numpy()
        pt_error = np.abs(pt_result - numpy_result).max()
        t_x.grad = None
    
    # MacroTorch (GPU) - Pre-allocated
    d_grad_out = cuda.to_device(grad_out)
    d_probs = cuda.to_device(probs)
    d_grad_logits = cuda.device_array((N, C, H, W), dtype=np_dtype)
    
    threads = (16, 16)
    blocks_x = math.ceil(W / 16)
    blocks_y = math.ceil(H / 16)
    blocks_z = N
    blocks = (blocks_x, blocks_y, blocks_z)
    
    for _ in range(5):
        SOFTMAX_BACKWARD[blocks, threads](d_grad_out, d_probs, d_grad_logits)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            SOFTMAX_BACKWARD[blocks, threads](d_grad_out, d_probs, d_grad_logits)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            SOFTMAX_BACKWARD[blocks, threads](d_grad_out, d_probs, d_grad_logits)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_result = d_grad_logits.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        print(f"\n  MacroTorch vs PyTorch: {pt_time/mt_time:.2f}x {'faster' if pt_time > mt_time else 'slower'}")


def benchmark_matmul(M, K, N, dtype_name='float32', num_runs=10):
    """Benchmark tiled matrix multiplication."""
    np_dtype = np.float32 if dtype_name == 'float32' else np.float16
    
    print(f"\n  Configuration:")
    print(f"    Matrix A:     {M} x {K}")
    print(f"    Matrix B:     {K} x {N}")
    print(f"    Output C:     {M} x {N}")
    print(f"    Precision:    {dtype_name.upper()}")
    print(f"    Runs:         {num_runs}")
    
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np_dtype)
    B = np.random.randn(K, N).astype(np_dtype)
    
    # NumPy (CPU) - Ground Truth
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        numpy_result = A @ B
        times.append((time.perf_counter() - start) * 1000)
    numpy_time = np.median(times)
    
    # PyTorch (GPU)
    pt_time, pt_error = None, None
    if TORCH_AVAILABLE:
        pt_dtype = torch.float32 if dtype_name == 'float32' else torch.float16
        t_A = torch.from_numpy(A).cuda().to(pt_dtype)
        t_B = torch.from_numpy(B).cuda().to(pt_dtype)
        
        for _ in range(5):
            _ = torch.matmul(t_A, t_B)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            pt_result = torch.matmul(t_A, t_B)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        pt_time = np.median(times)
        pt_error = np.abs(pt_result.cpu().numpy().astype(np.float32) - numpy_result.astype(np.float32)).max()
    
    # MacroTorch (GPU) - Pre-allocated
    d_A = cuda.to_device(A.astype(np.float32))
    d_B = cuda.to_device(B.astype(np.float32))
    d_C = cuda.device_array((M, N), dtype=np.float32)
    
    TILE_M, TILE_N = 16, 16
    threads = (TILE_N, TILE_M)
    blocks_x = math.ceil(N / TILE_N)
    blocks_y = math.ceil(M / TILE_M)
    blocks = (blocks_x, blocks_y)
    
    # Warmup
    for _ in range(5):
        matmul_tiled[blocks, threads](d_A, d_B, d_C)
    cuda.synchronize()
    
    if TORCH_AVAILABLE:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_event.record()
            matmul_tiled[blocks, threads](d_A, d_B, d_C)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            matmul_tiled[blocks, threads](d_A, d_B, d_C)
            cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mt_time = np.median(times)
    
    mt_result = d_C.copy_to_host()
    mt_error = np.abs(mt_result - numpy_result.astype(np.float32)).max()
    
    # Results
    print(f"\n  Results:")
    print(f"  {'-'*74}")
    print(f"  {'Implementation':<18} | {'Time (ms)':<12} | {'Speedup':<10} | {'Max Error':<12}")
    print(f"  {'-'*74}")
    print(f"  {'NumPy (CPU)':<18} | {numpy_time:<12.4f} | {'1.00x':<10} | {'Ground Truth':<12}")
    if TORCH_AVAILABLE:
        print(f"  {'PyTorch (GPU)':<18} | {pt_time:<12.4f} | {f'{numpy_time/pt_time:.2f}x':<10} | {f'{pt_error:.2e}':<12}")
    print(f"  {'MacroTorch (GPU)':<18} | {mt_time:<12.4f} | {f'{numpy_time/mt_time:.2f}x':<10} | {f'{mt_error:.2e}':<12}")
    print(f"  {'-'*74}")
    
    if TORCH_AVAILABLE:
        speedup = pt_time / mt_time
        if speedup > 1:
            print(f"\n  MacroTorch vs PyTorch: {speedup:.2f}x faster")
        else:
            print(f"\n  MacroTorch vs PyTorch: {1/speedup:.2f}x slower")


if __name__ == "__main__":
    main()