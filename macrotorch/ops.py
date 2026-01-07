import numpy as np
import math
from numba import cuda
from .kernels import KERNELS , BACKWARD_KERNELS, BIAS_KERNEL, TIERS


def forward(A , K , padding=0, bias=None, dtype='auto' , verbose=False , d_A=None , d_K=None , d_out=None):
    """
    2D Convolution Forward Pass
    
    Computes the 2D convolution of input A with kernel K using custom CUDA kernels.
    
    Parameters
    ----------
    A : numpy.ndarray
        Input image/feature map of shape (H, W). Supports float32 or float16 dtype.
    K : numpy.ndarray
        Convolution kernel/filter of shape (Kh, Kw). Must have same dtype as A.
    padding : int, optional (default=0)
        Number of pixels to pad on all sides of the input.
    bias : float or None, optional (default=None)
        Scalar bias value to add to each output element.
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
    verbose : bool, optional (default=False)
        If True, prints kernel selection and execution details.
    d_A, d_K, d_out : cuda device arrays, optional
        Pre-allocated GPU memory for benchmarking.
    
    Returns
    -------
    numpy.ndarray
        Convolution output of shape (out_h, out_w) in float32.
    """
    assert A.ndim == 2 , f"A must be 2D, got shape {A.shape}"
    assert K.ndim == 2 , f"K must be 2D, got shape {K.shape}"
    
    H , W = A.shape
    Kh , Kw = K.shape
    
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
    
    out_h = H - Kh + 1 + (2 * padding)
    out_w = W - Kw + 1 + (2 * padding)
    
    if d_A is None:
        d_A = cuda.to_device(A)
    if d_K is None:
        d_K = cuda.to_device(K)
    if d_out is None:
        out = np.zeros((out_h , out_w) , dtype=np.float32)
        d_out = cuda.to_device(out)
    
    threads_per_block = (block_size , block_size)
    blocks_y = math.ceil(out_h / block_size)  
    blocks_x = math.ceil(out_w / block_size)  
    blocks_per_grid = (blocks_x , blocks_y)  
    
    kernel = KERNELS[(tier , dtype)]
    bias_val = float(bias) if bias is not None else 0.0
    kernel[blocks_per_grid , threads_per_block](d_A , d_K , d_out, padding, bias_val)
    
    cuda.synchronize()
    return d_out.copy_to_host()


def input_backward(grad_out, K, padding=0, dtype='auto', verbose=False):
    """
    2D Convolution Backward Pass (Input Gradient)
    
    Computes the gradient of the loss with respect to the input (∂L/∂A).
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient flowing back from the next layer, shape (H_out, W_out).
    K : numpy.ndarray
        Convolution kernel/filter of shape (Kh, Kw). Same kernel used in forward pass.
    padding : int, optional (default=0)
        Padding value used in the forward pass.
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
    verbose : bool, optional (default=False)
        If True, prints kernel selection and execution details.
    
    Returns
    -------
    numpy.ndarray
        Gradient with respect to input (grad_A) of shape (H_in, W_in) in float32.
    """
    assert grad_out.ndim == 2
    assert K.ndim == 2
    
    H_out, W_out = grad_out.shape
    Kh, Kw = K.shape
    
    if dtype == "auto":
        dtype = 'fp16' if grad_out.dtype == np.float16 else 'fp32'
        
    H_in = H_out + Kh - 1 - (2 * padding)
    W_in = W_out + Kw - 1 - (2 * padding)
    
    max_k = max(Kh, Kw)
    
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

    grad_A = np.zeros((H_in, W_in), dtype=np.float32)
    
    d_grad_out = cuda.to_device(grad_out)
    d_K = cuda.to_device(K)
    d_grad_A = cuda.to_device(grad_A)
    
    threads_per_block = (block_size, block_size)
    blocks_y = math.ceil(H_in / block_size)
    blocks_x = math.ceil(W_in / block_size)
    blocks_per_grid = (blocks_x, blocks_y)
    
    kernel = BACKWARD_KERNELS[(tier, dtype)]
    kernel[blocks_per_grid, threads_per_block](d_grad_out, d_K, padding, d_grad_A)
    
    cuda.synchronize()
    return d_grad_A.copy_to_host()


def bias_backward(grad_out, dtype='auto', verbose=False, d_grad_out=None, d_grad_bias=None):
    """
    2D Convolution Backward Pass (Bias Gradient)
    
    Computes the gradient of the loss with respect to the bias (∂L/∂b).
    For batched 4D input (N, C, H, W), sums gradients across N, H, W dimensions.
    
    Parameters
    ----------
    grad_out : numpy.ndarray or cuda.devicearray
        Gradient flowing back from the next layer, shape (N, C, H, W).
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
    verbose : bool, optional (default=False)
        If True, prints execution details.
    d_grad_out : cuda.devicearray, optional
        Pre-allocated input on GPU for benchmarking.
    d_grad_bias : cuda.devicearray, optional
        Pre-allocated output on GPU for benchmarking.
    
    Returns
    -------
    numpy.ndarray or cuda.devicearray
        Bias gradient of shape (C,) in float32.
    """
    if d_grad_out is None:
        assert grad_out is not None, "Either grad_out or d_grad_out must be provided"
        if hasattr(grad_out, '__cuda_array_interface__'):
            d_grad_out = grad_out
        else:
            d_grad_out = cuda.to_device(grad_out)
    
    N, C, H, W = d_grad_out.shape
    
    if verbose:
        print(f"Bias Backward: Shape: {d_grad_out.shape}")
    
    return_host = False
    if d_grad_bias is None:
        d_grad_bias = cuda.device_array(C, dtype=np.float32)
        cuda.to_device(np.zeros(C, dtype=np.float32), to=d_grad_bias)
        return_host = True
    else:
        cuda.to_device(np.zeros(C, dtype=np.float32), to=d_grad_bias)
    
    threads_per_block = (32, 8, 1)
    blocks_per_grid = (
        math.ceil(W / 32),
        math.ceil(H / 8),
        C
    )
    
    BIAS_KERNEL[blocks_per_grid, threads_per_block](d_grad_out, d_grad_bias)
    cuda.synchronize()
    
    if return_host:
        return d_grad_bias.copy_to_host()
    else:
        return d_grad_bias
