import numpy as np
import math
from numba import cuda
from .kernels import KERNELS , BACKWARD_KERNELS, TIERS


def forward(A , K , padding=0, dtype = 'auto' , verbose = False , d_A = None , d_K = None , d_out = None):
    assert A.ndim == 2 , f"A must be 2D, got shape {A.shape}"
    assert K.ndim == 2 , f"K must be 2D, got shape {K.shape}"
    
    H , W = A.shape
    Kh , Kw = K.shape
    
    # Validation strictly for validity:
    # If padding is effectively increasing image size, we check "virtual" size
    # But usually kernels require kernel to be smaller than the effective image region.
    # For now, simplistic validation.
    
    if dtype == "auto":
        dtype = 'fp16' if A.dtype == np.float16 else 'fp32'
    
    max_k = max(Kh , Kw)
    
    if max_k <= 7:
        tier = 'tiny'
    elif max_k <= 15:
        tier = 'small'
    elif max_k <= 63:
        tier = 'medium'
    elif max_k <= 93:
        tier = 'large'
    else:
        tier = 'xlarge'
    
    config = TIERS[tier]
    block_size = config['block_size']
    use_shared = config['use_shared']
    
    if verbose:
        memory_type = "Shared Memory" if use_shared else "Direct Global"
        shared_info = f"({config['shared_size']}×{config['shared_size']})" if use_shared else ""
        print(f"Algorithm: {tier.upper()} ({dtype.upper()}) - {memory_type} {shared_info}")
        print(f"Kernel: {Kh}×{Kw}, Block: {block_size}×{block_size}")
    
    # Update Output calculation for padding
    # H_out = H_in + 2*padding - dilation *(kernel_size - 1) - 1 + 1... assuming stride 1, dilation 1
    # Standard formula: out_dim = (in_dim + 2*pad - kernel) + 1
    out_h = H - Kh + 1 + (2 * padding)
    out_w = W - Kw + 1 + (2 * padding)
    
    if d_A is None:
        d_A = cuda.to_device(A)
    if d_K is None:
        d_K = cuda.to_device(K)
    if d_out is None:
        out = np.zeros((out_h , out_w) , dtype = np.float32)
        d_out = cuda.to_device(out)
    
    threads_per_block = (block_size , block_size)
    blocks_y = math.ceil(out_h / block_size)  
    blocks_x = math.ceil(out_w / block_size)  
    blocks_per_grid = (blocks_x , blocks_y)  
    
    kernel = KERNELS[(tier , dtype)]
    # Pass padding to kernel
    kernel[blocks_per_grid , threads_per_block](d_A , d_K , d_out, padding)
    
    cuda.synchronize()
    return d_out.copy_to_host()


def backward(grad_out, K, padding=0, dtype='auto', verbose=False):
    """
    Computes gradient w.r.t Input (grad_A)
    This corresponds to a 'Full' convolution of grad_out with rotated K.
    """
    assert grad_out.ndim == 2
    assert K.ndim == 2
    
    H_out, W_out = grad_out.shape
    Kh, Kw = K.shape
    
    if dtype == "auto":
        dtype = 'fp16' if grad_out.dtype == np.float16 else 'fp32'
        
    # Determine input shape from output shape (inverse of forward)
    # H_in = H_out + Kh - 1 - 2*padding
    H_in = H_out + Kh - 1 - (2 * padding)
    W_in = W_out + Kw - 1 - (2 * padding)
    
    max_k = max(Kh, Kw)
    
    # Select Tier
    if max_k <= 7:
        tier = 'tiny'
    elif max_k <= 15:
        tier = 'small'
    elif max_k <= 63:
        tier = 'medium'
    elif max_k <= 93:
        tier = 'large'
    else:
        tier = 'xlarge'
    
    config = TIERS[tier]
    block_size = config['block_size']
    
    if verbose:
        print(f"BWD Algorithm: {tier.upper()} ({dtype.upper()})")

    # Allocate Grad_A
    grad_A = np.zeros((H_in, W_in), dtype=np.float32)
    
    d_grad_out = cuda.to_device(grad_out)
    d_K = cuda.to_device(K)
    d_grad_A = cuda.to_device(grad_A)
    
    threads_per_block = (block_size, block_size)
    blocks_y = math.ceil(H_in / block_size)
    blocks_x = math.ceil(W_in / block_size)
    blocks_per_grid = (blocks_x, blocks_y)
    
    kernel = BACKWARD_KERNELS[(tier, dtype)]
    
    # Note: Backward input doesn't use padding arg inside kernel the same way forward does, 
    # as the logic is effectively a full convolution. 
    # Current implementation handles validation logic implicitly.
    kernel[blocks_per_grid, threads_per_block](d_grad_out, d_K, d_grad_A)
    
    cuda.synchronize()
    return d_grad_A.copy_to_host()
