import numpy as np
import math
from numba import cuda
from .kernels import KERNELS , BACKWARD_KERNELS, BIAS_KERNEL, TIERS


def forward(A , K , padding=0, bias=None, dtype = 'auto' , verbose = False , d_A = None , d_K = None , d_out = None):
    """
    2D Convolution Forward Pass
    
    Computes the 2D convolution of input A with kernel K using custom CUDA kernels.
    
    Parameters
    ----------
    A : numpy.ndarray
        Input image/feature map of shape (H, W).
        Supports float32 or float16 dtype.
    
    K : numpy.ndarray
        Convolution kernel/filter of shape (Kh, Kw).
        Must have same dtype as A.
    
    padding : int, optional (default=0)
        Number of pixels to pad on all sides of the input.
        Output size = (H - Kh + 1 + 2*padding, W - Kw + 1 + 2*padding)
    
    bias : float or None, optional (default=None)
        Scalar bias value to add to each output element.
        If None, no bias is added.
    
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
        'auto' automatically detects from input dtype.
    
    verbose : bool, optional (default=False)
        If True, prints kernel selection and execution details.
    
    d_A, d_K, d_out : cuda device arrays, optional
        Pre-allocated GPU memory (advanced usage).
        If None, memory is automatically allocated.
    
    Returns
    -------
    numpy.ndarray
        Convolution output of shape (out_h, out_w) in float32.
    
    Examples
    --------
    >>> import numpy as np
    >>> from macrotorch import conv2d_forward
    >>> 
    >>> # Basic usage
    >>> img = np.random.randn(256, 256).astype(np.float32)
    >>> kernel = np.random.randn(5, 5).astype(np.float32)
    >>> output = conv2d_forward(img, kernel)
    >>> 
    >>> # With padding and bias
    >>> output = conv2d_forward(img, kernel, padding=2, bias=0.1)
    >>> 
    >>> # FP16 mode
    >>> img_fp16 = img.astype(np.float16)
    >>> kernel_fp16 = kernel.astype(np.float16)
    >>> output = conv2d_forward(img_fp16, kernel_fp16)
    """
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
    # Pass padding and bias to kernel
    kernel[blocks_per_grid , threads_per_block](d_A , d_K , d_out, padding, bias)
    
    cuda.synchronize()
    return d_out.copy_to_host()


def input_backward(grad_out, K, padding=0, dtype='auto', verbose=False):
    """
    2D Convolution Backward Pass (Input Gradient)
    
    Computes the gradient of the loss with respect to the input (∂L/∂A).
    This is equivalent to a full convolution of grad_out with the kernel.
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient flowing back from the next layer, shape (H_out, W_out).
        This is the output of the forward pass.
        Supports float32 or float16 dtype.
    
    K : numpy.ndarray
        Convolution kernel/filter of shape (Kh, Kw).
        Same kernel used in the forward pass.
        Must have same dtype as grad_out.
    
    padding : int, optional (default=0)
        Padding value used in the forward pass.
        Must match the padding used during forward propagation.
    
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
        'auto' automatically detects from grad_out dtype.
    
    verbose : bool, optional (default=False)
        If True, prints kernel selection and execution details.
    
    Returns
    -------
    numpy.ndarray
        Gradient with respect to input (grad_A) of shape (H_in, W_in) in float32.
        Shape: (H_out + Kh - 1 - 2*padding, W_out + Kw - 1 - 2*padding)
    
    Examples
    --------
    >>> import numpy as np
    >>> from macrotorch import conv2d_forward, conv2d_input_backward
    >>> 
    >>> # Forward pass
    >>> img = np.random.randn(256, 256).astype(np.float32)
    >>> kernel = np.random.randn(5, 5).astype(np.float32)
    >>> output = conv2d_forward(img, kernel, padding=2)
    >>> 
    >>> # Backward pass (assume we have grad_out from loss)
    >>> grad_out = np.random.randn(*output.shape).astype(np.float32)
    >>> grad_input = conv2d_input_backward(grad_out, kernel, padding=2)
    >>> 
    >>> # FP16 mode
    >>> grad_out_fp16 = grad_out.astype(np.float16)
    >>> kernel_fp16 = kernel.astype(np.float16)
    >>> grad_input = conv2d_input_backward(grad_out_fp16, kernel_fp16, padding=2)
    
    Notes
    -----
    This function computes only the gradient with respect to the INPUT.
    For weight gradients (∂L/∂K) or bias gradients (∂L/∂b), use separate functions.
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
    
    # Pass padding to backward kernel
    kernel[blocks_per_grid, threads_per_block](d_grad_out, d_K, padding, d_grad_A)
    
    cuda.synchronize()
    return d_grad_A.copy_to_host()


def bias_backward(grad_out, dtype='auto', verbose=False):
    """
    2D Convolution Backward Pass (Bias Gradient)
    
    Computes the gradient of the loss with respect to the bias (∂L/∂b).
    For batched 4D input (N, C, H, W), sums gradients across N, H, W dimensions.
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient flowing back from the next layer, shape (N, C, H, W).
        4D batched tensor: (Batch, Channels, Height, Width).
        Supports float32 or float16 dtype.
    
    dtype : str, optional (default='auto')
        Precision mode: 'fp32', 'fp16', or 'auto'.
        'auto' automatically detects from grad_out dtype.
    
    verbose : bool, optional (default=False)
        If True, prints execution details.
    
    Returns
    -------
    numpy.ndarray
        Bias gradient of shape (C,) in float32.
        One gradient value per channel.
    
    Examples
    --------
    >>> import numpy as np
    >>> from macrotorch import bias_backward
    >>> 
    >>> # Batched gradient (N=8, C=64, H=28, W=28)
    >>> grad_out = np.random.randn(8, 64, 28, 28).astype(np.float32)
    >>> grad_bias = bias_backward(grad_out)
    >>> print(grad_bias.shape)  # (64,)
    >>> 
    >>> # FP16 mode
    >>> grad_out_fp16 = grad_out.astype(np.float16)
    >>> grad_bias = bias_backward(grad_out_fp16)
    
    Notes
    -----
    This function expects 4D batched input (N, C, H, W).
    For 2D single-channel convolution, use a simple np.sum(grad_out) instead.
    """
    assert grad_out.ndim == 4, f"Expected 4D input (N, C, H, W), got shape {grad_out.shape}"
    
    N, C, H, W = grad_out.shape
    
    if dtype == "auto":
        dtype = 'fp16' if grad_out.dtype == np.float16 else 'fp32'
    
    if verbose:
        print(f"Bias Backward: {dtype.upper()} | Shape: {grad_out.shape}")
    
    # Allocate output (one gradient per channel)
    grad_bias = np.zeros(C, dtype=np.float32)
    
    d_grad_out = cuda.to_device(grad_out) 
    d_grad_bias = cuda.to_device(grad_bias)
    
    # 3D grid: (W, H, C)
    # 32x8x1 = 256 threads per block
    threads_per_block = (32, 8, 1)  # (x=W, y=H, z=C)
    blocks_per_grid = (
        math.ceil(W / 32),  # x covers Width
        math.ceil(H / 8),   # y covers Height
        C                   # z covers Channels (one z-layer per channel)
    )
    
    # Use single kernel (handles both FP16/FP32 via input conversion)
    BIAS_KERNEL[blocks_per_grid, threads_per_block](d_grad_out, d_grad_bias)
    
    cuda.synchronize()
    return d_grad_bias.copy_to_host()
