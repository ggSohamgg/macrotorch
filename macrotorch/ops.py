import numpy as np
import math
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernels import KERNELS , BACKWARD_KERNELS, BIAS_KERNEL, WEIGHT_KERNEL, TIERS, RELU_FORWARD, RELU_BACKWARD, MAXPOOL2D_FORWARD, MAXPOOL2D_BACKWARD, SOFTMAX_FORWARD, SOFTMAX_BACKWARD, matmul_tiled, cross_entropy_loss_kernel, cross_entropy_backward_kernel, BIAS_ADD_2D, SGD_UPDATE, IM2COL, COL2IM, SUM_AXIS0, TRANSPOSE_2D, PERMUTE4D_NHWC_NCHW, PERMUTE4D_NCHW_NHWC, ZERO_FILL


def is_device_array(x):
    """Check if x is a CUDA device array"""
    return isinstance(x, DeviceNDArray)


def to_device(x):
    """Transfer array to GPU if not already there"""
    if is_device_array(x):
        return x
    return cuda.to_device(x)


def conv2d_forward(A, K, padding=0, stride=1, bias=None, return_device=False):
    """
    2D Convolution Forward Pass using im2col + GEMM (Fully GPU-native)
    
    Parameters
    ----------
    A : numpy.ndarray or DeviceNDArray
        Input tensor of shape (N, C_in, H, W)
    K : numpy.ndarray or DeviceNDArray
        Kernel tensor of shape (C_out, C_in, Kh, Kw)
    padding : int
        Zero-padding added to both sides of input
    stride : int
        Stride of the convolution (default: 1)
    bias : numpy.ndarray or None
        Bias tensor of shape (C_out,)
    return_device : bool
        If True, return device array (stays on GPU)
    """
    if is_device_array(A):
        N, Cin, H, W = A.shape
        d_A = A
    else:
        N, Cin, H, W = A.shape
        d_A = cuda.to_device(A.astype(np.float32))
    
    if is_device_array(K):
        Cout, Cin_K, Kh, Kw = K.shape
        d_K = K
    else:
        Cout, Cin_K, Kh, Kw = K.shape
        d_K = cuda.to_device(K.astype(np.float32))
    
    assert Cin == Cin_K, f"Input channels {Cin} != kernel channels {Cin_K}"
    
    out_h = (H + 2 * padding - Kh) // stride + 1
    out_w = (W + 2 * padding - Kw) // stride + 1
    
    # im2col: (N * out_h * out_w, Cin * Kh * Kw)
    col_rows = N * out_h * out_w
    col_cols = Cin * Kh * Kw
    d_col = cuda.device_array((col_rows, col_cols), dtype=np.float32)
    
    # Launch im2col kernel
    total_elements = col_rows * col_cols
    threads = 256
    blocks = (total_elements + threads - 1) // threads
    IM2COL[blocks, threads](d_A, d_col, N, Cin, H, W, Kh, Kw, padding, stride, out_h, out_w)
    cuda.synchronize()
    
    # Reshape kernel on GPU: (Cout, Cin*Kh*Kw)
    d_K_reshaped = d_K.reshape((Cout, Cin * Kh * Kw))
    
    # GPU Transpose: K_reshaped.T -> (Cin*Kh*Kw, Cout)
    d_K_T = cuda.device_array((col_cols, Cout), dtype=np.float32)
    TILE = 16
    blocks_t = ((Cout + TILE - 1) // TILE, (col_cols + TILE - 1) // TILE)
    TRANSPOSE_2D[blocks_t, (TILE, TILE)](d_K_reshaped, d_K_T, Cout, col_cols)
    cuda.synchronize()
    
    # GEMM: col @ K.T -> (N*out_h*out_w, Cout)
    d_out_2d = cuda.device_array((col_rows, Cout), dtype=np.float32)
    
    M, K_dim = col_rows, col_cols
    N_dim = Cout
    
    threads_per_block = (TILE, TILE)
    blocks_x = (N_dim + TILE - 1) // TILE
    blocks_y = (M + TILE - 1) // TILE
    
    matmul_tiled[(blocks_y, blocks_x), threads_per_block](d_col, d_K_T, d_out_2d)
    cuda.synchronize()
    
    # Add bias if provided (GPU)
    if bias is not None:
        d_bias = cuda.to_device(bias.astype(np.float32)) if not is_device_array(bias) else bias
        threads_bias = (16, 16)
        blocks_bias = ((Cout + 15) // 16, (col_rows + 15) // 16)
        BIAS_ADD_2D[blocks_bias, threads_bias](d_out_2d, d_bias, col_rows, Cout)
        cuda.synchronize()
    
    # Reshape to 4D NHWC on GPU, then permute to NCHW
    # d_out_2d is (N*out_h*out_w, Cout) - interpret as (N, out_h, out_w, Cout) NHWC
    d_out_nhwc = d_out_2d.reshape((N, out_h, out_w, Cout))
    
    # Permute NHWC -> NCHW on GPU
    d_out = cuda.device_array((N, Cout, out_h, out_w), dtype=np.float32)
    total = N * Cout * out_h * out_w
    threads_p = 256
    blocks_p = (total + threads_p - 1) // threads_p
    PERMUTE4D_NHWC_NCHW[blocks_p, threads_p](d_out_nhwc, d_out, N, out_h, out_w, Cout)
    cuda.synchronize()
    
    if return_device:
        return d_out
    return d_out.copy_to_host()


# Keep old forward as alias for backward compatibility
def forward(A, K, padding=0, bias=None, dtype='auto', verbose=False, d_A=None, d_K=None, d_out=None, return_device=False):
    """Legacy wrapper - calls conv2d_forward"""
    return conv2d_forward(A, K, padding=padding, stride=1, bias=bias, return_device=return_device)


def input_backward(grad_out, K, padding=0, stride=1, dtype='auto', verbose=False, return_device=False):
    """
    2D Convolution Backward Pass (Input Gradient) using col2im + GEMM (Fully GPU-native)
    
    dX = col2im(dY_col @ W_reshaped)
    """
    if is_device_array(grad_out):
        N, Cout, out_h, out_w = grad_out.shape
        d_grad_out = grad_out
    else:
        N, Cout, out_h, out_w = grad_out.shape
        d_grad_out = cuda.to_device(grad_out.astype(np.float32))
    
    if is_device_array(K):
        Cout_K, Cin, Kh, Kw = K.shape
        d_K = K
    else:
        Cout_K, Cin, Kh, Kw = K.shape
        d_K = cuda.to_device(K.astype(np.float32))
    
    assert Cout == Cout_K
    
    # Compute input dimensions
    H_in = (out_h - 1) * stride - 2 * padding + Kh
    W_in = (out_w - 1) * stride - 2 * padding + Kw
    
    TILE = 16
    
    # Permute grad_out from NCHW to NHWC on GPU
    d_grad_out_nhwc = cuda.device_array((N, out_h, out_w, Cout), dtype=np.float32)
    total_p = N * Cout * out_h * out_w
    threads_p = 256
    blocks_p = (total_p + threads_p - 1) // threads_p
    PERMUTE4D_NCHW_NHWC[blocks_p, threads_p](d_grad_out, d_grad_out_nhwc, N, Cout, out_h, out_w)
    cuda.synchronize()
    
    # Reshape to 2D: (N*out_h*out_w, Cout)
    d_grad_out_2d = d_grad_out_nhwc.reshape((N * out_h * out_w, Cout))
    
    # Reshape kernel on GPU: (Cout, Cin*Kh*Kw)
    d_K_reshaped = d_K.reshape((Cout, Cin * Kh * Kw))
    
    # GEMM: dY @ W -> (N*out_h*out_w, Cin*Kh*Kw)
    M = N * out_h * out_w
    K_dim = Cout
    N_dim = Cin * Kh * Kw
    
    d_col = cuda.device_array((M, N_dim), dtype=np.float32)
    
    threads_per_block = (TILE, TILE)
    blocks_x = (N_dim + TILE - 1) // TILE
    blocks_y = (M + TILE - 1) // TILE
    
    matmul_tiled[(blocks_y, blocks_x), threads_per_block](d_grad_out_2d, d_K_reshaped, d_col)
    cuda.synchronize()
    
    # col2im: convert columns back to image
    d_grad_A = cuda.device_array((N, Cin, H_in, W_in), dtype=np.float32)
    
    # Zero initialize on GPU
    total_zero = N * Cin * H_in * W_in
    threads_z = 256
    blocks_z = (total_zero + threads_z - 1) // threads_z
    ZERO_FILL[blocks_z, threads_z](d_grad_A, total_zero)
    cuda.synchronize()
    
    total_elements = N * out_h * out_w * Cin * Kh * Kw
    threads = 256
    blocks = (total_elements + threads - 1) // threads
    COL2IM[blocks, threads](d_col, d_grad_A, N, Cin, H_in, W_in, Kh, Kw, padding, stride, out_h, out_w)
    cuda.synchronize()
    
    if return_device:
        return d_grad_A
    return d_grad_A.copy_to_host()


def weight_backward(grad_out, A, padding=0, stride=1, Kh=None, Kw=None, dtype='auto', verbose=False, d_grad_out=None, d_A=None, d_grad_W=None, return_device=False):
    """
    2D Convolution Backward Pass (Weight Gradient) using im2col + GEMM (Fully GPU-native)
    
    dW = dY.T @ col
    """
    if d_grad_out is None:
        if is_device_array(grad_out):
            d_grad_out = grad_out
        else:
            d_grad_out = cuda.to_device(grad_out.astype(np.float32))
    
    if d_A is None:
        if is_device_array(A):
            d_A = A
        else:
            d_A = cuda.to_device(A.astype(np.float32))
    
    N, Cout, out_h, out_w = d_grad_out.shape
    _, Cin, H_in, W_in = d_A.shape
    
    # Compute kernel size if not provided
    if Kh is None:
        Kh = H_in + 2 * padding - (out_h - 1) * stride
    if Kw is None:
        Kw = W_in + 2 * padding - (out_w - 1) * stride
    
    TILE = 16
    
    # im2col on input A: (N*out_h*out_w, Cin*Kh*Kw)
    col_rows = N * out_h * out_w
    col_cols = Cin * Kh * Kw
    d_col = cuda.device_array((col_rows, col_cols), dtype=np.float32)
    
    total_elements = col_rows * col_cols
    threads = 256
    blocks = (total_elements + threads - 1) // threads
    IM2COL[blocks, threads](d_A, d_col, N, Cin, H_in, W_in, Kh, Kw, padding, stride, out_h, out_w)
    cuda.synchronize()
    
    # Permute grad_out from NCHW to NHWC on GPU
    d_grad_out_nhwc = cuda.device_array((N, out_h, out_w, Cout), dtype=np.float32)
    total_p = N * Cout * out_h * out_w
    threads_p = 256
    blocks_p = (total_p + threads_p - 1) // threads_p
    PERMUTE4D_NCHW_NHWC[blocks_p, threads_p](d_grad_out, d_grad_out_nhwc, N, Cout, out_h, out_w)
    cuda.synchronize()
    
    # Reshape to 2D: (N*out_h*out_w, Cout)
    d_grad_out_2d = d_grad_out_nhwc.reshape((col_rows, Cout))
    
    # GPU Transpose: (N*out_h*out_w, Cout).T -> (Cout, N*out_h*out_w)
    d_grad_out_T = cuda.device_array((Cout, col_rows), dtype=np.float32)
    blocks_t = ((col_rows + TILE - 1) // TILE, (Cout + TILE - 1) // TILE)
    TRANSPOSE_2D[blocks_t, (TILE, TILE)](d_grad_out_2d, d_grad_out_T, col_rows, Cout)
    cuda.synchronize()
    
    # GEMM: dY.T @ col = (Cout, col_rows) @ (col_rows, col_cols) = (Cout, Cin*Kh*Kw)
    d_grad_W_2d = cuda.device_array((Cout, col_cols), dtype=np.float32)
    
    M = Cout
    K_dim = col_rows
    N_dim = col_cols
    
    threads_per_block = (TILE, TILE)
    blocks_x = (N_dim + TILE - 1) // TILE
    blocks_y = (M + TILE - 1) // TILE
    
    matmul_tiled[(blocks_y, blocks_x), threads_per_block](d_grad_out_T, d_col, d_grad_W_2d)
    cuda.synchronize()
    
    # Reshape to (Cout, Cin, Kh, Kw) on GPU
    d_grad_W = d_grad_W_2d.reshape((Cout, Cin, Kh, Kw))
    
    if return_device:
        return d_grad_W
    return d_grad_W.copy_to_host()


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


def relu(x, d_x=None, d_out=None, return_device=False):
    """
    ReLU Forward Pass
    
    Computes element-wise max(0, x) using CUDA.
    
    Parameters
    ----------
    x : numpy.ndarray or DeviceNDArray
        Input array of any shape
    d_x : numba.cuda.DeviceNDArray, optional
        Pre-allocated device input array
    d_out : numba.cuda.DeviceNDArray, optional
        Pre-allocated device output array
    return_device : bool
        If True, return device array. Default: False
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        ReLU output with same shape as input
    """
    if d_x is None:
        d_x = to_device(x)
    
    if d_out is None:
        d_out = cuda.device_array(d_x.shape, dtype=np.float32)
    
    # 1D grid for element-wise operation
    threads_per_block = 256
    blocks_per_grid = math.ceil(d_x.size / threads_per_block)
    
    RELU_FORWARD[blocks_per_grid, threads_per_block](d_x, d_out)
    cuda.synchronize()
    
    if return_device:
        return d_out
    return d_out.copy_to_host()


def relu_backward(x, grad_out, d_x=None, d_grad_out=None, d_grad_in=None, return_device=False):
    """
    ReLU Backward Pass
    
    Computes gradient through ReLU: grad_in = grad_out * (x > 0)
    
    Parameters
    ----------
    x : numpy.ndarray or DeviceNDArray
        Original input to ReLU (before ReLU was applied)
    grad_out : numpy.ndarray or DeviceNDArray
        Gradient from downstream layer
    d_x : numba.cuda.DeviceNDArray, optional
        Pre-allocated device input array
    d_grad_out : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient output array
    d_grad_in : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient input array
    return_device : bool
        If True, return device array. Default: False
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        Gradient with respect to input
    """
    if d_x is None:
        d_x = to_device(x)
    
    if d_grad_out is None:
        d_grad_out = to_device(grad_out)
    
    if d_grad_in is None:
        d_grad_in = cuda.device_array(d_x.shape, dtype=np.float32)
    
    # 1D grid for element-wise operation
    threads_per_block = 256
    blocks_per_grid = math.ceil(d_x.size / threads_per_block)
    
    RELU_BACKWARD[blocks_per_grid, threads_per_block](d_x, d_grad_out, d_grad_in)
    cuda.synchronize()
    
    if return_device:
        return d_grad_in
    return d_grad_in.copy_to_host()


def softmax_forward(x, d_x=None, d_out=None, return_device=False):
    """
    Softmax Forward Pass (4D Multi-Channel)
    
    Computes softmax along channel dimension for 4D input (N, C, H, W).
    
    Parameters
    ----------
    x : numpy.ndarray or DeviceNDArray
        Input logits of shape (N, C, H, W)
    return_device : bool
        If True, return device array. Default: False
        
    Returns
    -------
    numpy.ndarray or DeviceNDArray
        Softmax probabilities of shape (N, C, H, W)
    """
    if is_device_array(x):
        N, C, H, W = x.shape
        d_x = x
    else:
        N, C, H, W = x.shape
        if d_x is None:
            d_x = cuda.to_device(x.astype(np.float32))
    
    if d_out is None:
        d_out = cuda.device_array((N, C, H, W), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W / 16)
    blocks_y = math.ceil(H / 16)
    blocks_z = N
    blocks = (blocks_x, blocks_y, blocks_z)
    
    SOFTMAX_FORWARD[blocks, threads](d_x, d_out)
    cuda.synchronize()
    
    if return_device:
        return d_out
    return d_out.copy_to_host()

def softmax_backward(grad_out, probs, d_grad_out=None, d_probs=None, d_grad_logits=None):
    """
    Softmax Backward Pass (4D Multi-Channel)
    
    Computes gradient w.r.t. logits for softmax along channel dimension.
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient from next layer of shape (N, C, H, W)
    probs : numpy.ndarray
        Softmax probabilities from forward pass of shape (N, C, H, W)
    d_grad_out : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient output array
    d_probs : numba.cuda.DeviceNDArray, optional
        Pre-allocated device probabilities array
    d_grad_logits : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient logits array
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        Gradient w.r.t. input logits of shape (N, C, H, W)
    """
    N, C, H, W = grad_out.shape
    
    if d_grad_out is None:
        d_grad_out = cuda.to_device(grad_out.astype(np.float32))
        return_host = True
    else:
        return_host = False
    
    if d_probs is None:
        d_probs = cuda.to_device(probs.astype(np.float32))
    
    if d_grad_logits is None:
        d_grad_logits = cuda.device_array((N, C, H, W), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W / 16)
    blocks_y = math.ceil(H / 16)
    blocks_z = N
    blocks = (blocks_x, blocks_y, blocks_z)
    
    SOFTMAX_BACKWARD[blocks, threads](d_grad_out, d_probs, d_grad_logits)
    cuda.synchronize()
    
    if return_host:
        return d_grad_logits.copy_to_host()
    else:
        return d_grad_logits


def maxpool2d_forward(x, pool_size=2, d_x=None, d_out=None, d_indices=None, return_device=False):
    """
    MaxPool2D Forward Pass (4D Multi-Channel)
    
    Computes max pooling over 4D input (N, C, H, W).
    
    Parameters
    ----------
    x : numpy.ndarray or DeviceNDArray
        Input array of shape (N, C, H, W)
    pool_size : int
        Size of pooling window (default: 2)
    return_device : bool
        If True, return device arrays. Default: False
        
    Returns
    -------
    tuple(array, array)
        (output, indices) - pooled output (N, C, H_out, W_out) and max indices for backward pass
    """
    if is_device_array(x):
        N, C, H, W = x.shape
        d_x = x
    else:
        N, C, H, W = x.shape
        if d_x is None:
            d_x = cuda.to_device(x)
    
    H_out = H // pool_size
    W_out = W // pool_size
    
    if d_out is None:
        d_out = cuda.device_array((N, C, H_out, W_out), dtype=np.float32)
    
    if d_indices is None:
        d_indices = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
    cuda.synchronize()
    
    if return_device:
        return d_out, d_indices
    return d_out.copy_to_host(), d_indices.copy_to_host()


def maxpool2d_backward(grad_out, indices, input_shape, pool_size=2, d_grad_out=None, d_indices=None, d_grad_in=None, return_device=False):
    """
    MaxPool2D Backward Pass (4D Multi-Channel)
    
    Computes gradient w.r.t. input for max pooling.
    
    Parameters
    ----------
    grad_out : numpy.ndarray or DeviceNDArray
        Gradient from next layer of shape (N, C, H_out, W_out)
    indices : numpy.ndarray or DeviceNDArray
        Indices from forward pass of shape (N, C, H_out, W_out)
    input_shape : tuple
        Shape of original input (N, C, H, W)
    pool_size : int
        Size of pooling window (default: 2)
    return_device : bool
        If True, return device array. Default: False
        
    Returns
    -------
    numpy.ndarray or DeviceNDArray
        Gradient w.r.t. input of shape (N, C, H, W)
    """
    if is_device_array(grad_out):
        N, C, H_out, W_out = grad_out.shape
        d_grad_out = grad_out
    else:
        N, C, H_out, W_out = grad_out.shape
        if d_grad_out is None:
            d_grad_out = cuda.to_device(grad_out)
    
    if d_indices is None:
        d_indices = to_device(indices)
    
    if d_grad_in is None:
        d_grad_in = cuda.device_array(input_shape, dtype=np.float32)
        # Zero initialize
        cuda.to_device(np.zeros(input_shape, dtype=np.float32), to=d_grad_in)
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    MAXPOOL2D_BACKWARD[blocks, threads](d_grad_out, d_indices, d_grad_in, pool_size)
    cuda.synchronize()
    
    if return_device:
        return d_grad_in
    return d_grad_in.copy_to_host()


def matmul(A, B, d_A=None, d_B=None, d_C=None, return_device=False):
    """
    Tiled Matrix Multiplication using CUDA
    
    Performs C = A @ B using a tiled CUDA kernel with shared memory.
    Uses 16x16 tiles for efficient GPU computation.
    Supports both FP32 and FP16 inputs (uses FP32 accumulation for accuracy).
    
    Parameters
    ----------
    A : numpy.ndarray or DeviceNDArray
        Input matrix of shape (M, K), supports float32 or float16
    B : numpy.ndarray or DeviceNDArray
        Input matrix of shape (K, N), supports float32 or float16
    d_A : numba.cuda.DeviceNDArray, optional
        Pre-allocated device array for A
    d_B : numba.cuda.DeviceNDArray, optional
        Pre-allocated device array for B
    d_C : numba.cuda.DeviceNDArray, optional
        Pre-allocated device output array
    return_device : bool
        If True, return device array. Default: False
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        Result matrix C of shape (M, N) in float32
    """
    if d_A is None:
        d_A = to_device(A)
    if d_B is None:
        d_B = to_device(B)
    
    M, K = d_A.shape
    _, N = d_B.shape
    
    if d_C is None:
        d_C = cuda.device_array((M, N), dtype=np.float32)
    
    # Tile configuration matches the kernel
    TILE_M = 16
    TILE_N = 16
    
    threads_per_block = (TILE_N, TILE_M)
    blocks_x = math.ceil(N / TILE_N)
    blocks_y = math.ceil(M / TILE_M)
    blocks_per_grid = (blocks_x, blocks_y)
    
    matmul_tiled[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    
    if return_device:
        return d_C
    return d_C.copy_to_host()


def linear(x, weight, bias=None, return_device=False):
    """
    Linear Layer Forward Pass
    
    Computes Y = X @ W + b using CUDA matmul kernel.
    
    Parameters
    ----------
    x : numpy.ndarray or DeviceNDArray
        Input tensor of shape (B, in_features)
    weight : numpy.ndarray or DeviceNDArray
        Weight matrix of shape (in_features, out_features)
    bias : numpy.ndarray, optional
        Bias vector of shape (out_features,)
    return_device : bool
        If True, return device array. Default: False
    
    Returns
    -------
    numpy.ndarray or DeviceNDArray
        Output tensor of shape (B, out_features)
    """
    # Always compute matmul on GPU
    d_out = matmul(x, weight, return_device=True)
    
    if bias is not None:
        # Add bias on GPU using kernel
        B = d_out.shape[0]
        C = d_out.shape[1]
        d_bias = to_device(bias.astype(np.float32))
        d_result = cuda.device_array((B, C), dtype=np.float32)
        
        threads = (16, 16)
        blocks = (math.ceil(B / 16), math.ceil(C / 16))
        BIAS_ADD_2D[blocks, threads](d_out, d_bias, d_result)
        cuda.synchronize()
        
        if return_device:
            return d_result
        return d_result.copy_to_host()
    
    if return_device:
        return d_out
    return d_out.copy_to_host()


def linear_backward(grad_out, x, weight, return_device=False):
    """
    Linear Layer Backward Pass
    
    Computes gradients for linear layer.
    
    Parameters
    ----------
    grad_out : numpy.ndarray or DeviceNDArray
        Gradient from next layer of shape (B, out_features)
    x : numpy.ndarray or DeviceNDArray
        Input from forward pass of shape (B, in_features)
    weight : numpy.ndarray or DeviceNDArray
        Weight matrix of shape (in_features, out_features)
    return_device : bool
        If True, return device arrays. Default: False
    
    Returns
    -------
    tuple (dX, dW, db)
        dX : Gradient w.r.t. input of shape (B, in_features)
        dW : Gradient w.r.t. weight of shape (in_features, out_features)
        db : Gradient w.r.t. bias of shape (out_features,)
    """
    # Handle device arrays for transpose
    if is_device_array(x):
        x_host = x.copy_to_host()
        x_T = x_host.T
    else:
        x_T = x.T
    
    if is_device_array(weight):
        w_host = weight.copy_to_host()
        w_T = w_host.T
    else:
        w_T = weight.T
    
    if is_device_array(grad_out):
        grad_out_host = grad_out.copy_to_host()
    else:
        grad_out_host = grad_out
    
    # dW = X.T @ dY
    dW = matmul(x_T, grad_out_host, return_device=return_device)
    
    # dX = dY @ W.T
    dX = matmul(grad_out_host, w_T, return_device=return_device)
    
    # db = sum(dY, axis=0)
    db = np.sum(grad_out_host, axis=0)
    
    return dX, dW, db


def cross_entropy_loss(probs, targets, d_probs=None, d_targets=None, d_loss=None):
    """
    Cross-Entropy Loss Forward Pass
    
    Computes the cross-entropy loss for multi-class classification.
    
    Parameters
    ----------
    probs : numpy.ndarray or DeviceNDArray
        Softmax probabilities of shape (B, C) where B is batch size, C is num classes
    targets : numpy.ndarray
        Target class indices of shape (B,), integer values in [0, C-1]
    
    Returns
    -------
    float
        Mean cross-entropy loss over the batch
    """
    if is_device_array(probs):
        B, C = probs.shape
        d_probs = probs
    else:
        B, C = probs.shape
        d_probs = cuda.to_device(probs.astype(np.float32))
    
    if d_targets is None:
        d_targets = cuda.to_device(targets.astype(np.int32))
    
    if d_loss is None:
        d_loss = cuda.device_array(B, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = math.ceil(B / threads_per_block)
    
    cross_entropy_loss_kernel[blocks_per_grid, threads_per_block](d_probs, d_targets, d_loss, B, C)
    cuda.synchronize()
    
    # Always return scalar loss
    loss = d_loss.copy_to_host()
    return np.mean(loss)


def cross_entropy_backward(probs, targets, d_probs=None, d_targets=None, d_grad=None, return_device=False):
    """
    Cross-Entropy Loss Backward Pass
    
    Computes gradient of cross-entropy loss w.r.t. logits (after softmax).
    
    Parameters
    ----------
    probs : numpy.ndarray or DeviceNDArray
        Softmax probabilities of shape (B, C)
    targets : numpy.ndarray
        Target class indices of shape (B,)
    return_device : bool
        If True, return device array. Default: False
    
    Returns
    -------
    numpy.ndarray or DeviceNDArray
        Gradient w.r.t. softmax input of shape (B, C)
    """
    if is_device_array(probs):
        B, C = probs.shape
        d_probs = probs
    else:
        B, C = probs.shape
        d_probs = cuda.to_device(probs.astype(np.float32))
    
    if d_targets is None:
        d_targets = cuda.to_device(targets.astype(np.int32))
    
    if d_grad is None:
        d_grad = cuda.device_array((B, C), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(B / 16)
    blocks_y = math.ceil(C / 16)
    blocks = (blocks_x, blocks_y)
    
    cross_entropy_backward_kernel[blocks, threads](d_probs, d_targets, d_grad, B, C)
    cuda.synchronize()
    
    if return_device:
        return d_grad
    return d_grad.copy_to_host()


def flatten(x):
    """
    Flatten a tensor from shape (N, C, H, W) to (N, C*H*W)
    
    Parameters
    ----------
    x : numpy.ndarray
        Input tensor of shape (N, C, H, W) or any shape
    
    Returns
    -------
    numpy.ndarray
        Flattened tensor of shape (N, -1)
    """
    if x.ndim == 4:
        N, C, H, W = x.shape
        return x.reshape(N, C * H * W)
    elif x.ndim == 2:
        return x  # Already flat
    else:
        return x.reshape(x.shape[0], -1)


def flatten_backward(grad, original_shape):
    """
    Reshape gradient back to original shape (unflatten)
    
    Parameters
    ----------
    grad : numpy.ndarray
        Gradient of shape (N, C*H*W)
    original_shape : tuple
        Original shape (N, C, H, W)
    
    Returns
    -------
    numpy.ndarray
        Gradient reshaped to original_shape
    """
    return grad.reshape(original_shape)


class SGD:
    """
    Stochastic Gradient Descent Optimizer (without momentum)
    
    Parameters
    ----------
    params : list of numpy.ndarray
        List of parameters to optimize
    lr : float
        Learning rate (default: 0.01)
    
    Example
    -------
    >>> optimizer = SGD([W1, b1, W2, b2], lr=0.01)
    >>> # After computing gradients
    >>> optimizer.step([dW1, db1, dW2, db2])
    """
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self, grads):
        """
        Update parameters using gradients
        
        Parameters
        ----------
        grads : list of numpy.ndarray
            List of gradients corresponding to each parameter
        """
        for param, grad in zip(self.params, grads):
            param -= self.lr * grad
    
    def zero_grad(self):
        """Placeholder for compatibility - gradients are computed fresh each iteration"""
        pass


def sgd_update_gpu(d_weights, d_grads, lr):
    """
    GPU-based SGD update: weights -= lr * grads
    
    Updates weights in-place on GPU without any CPU transfer.
    
    Parameters
    ----------
    d_weights : DeviceNDArray
        Weights on GPU (updated in-place)
    d_grads : DeviceNDArray
        Gradients on GPU
    lr : float
        Learning rate
    """
    n = d_weights.size
    threads = 256
    blocks = (n + threads - 1) // threads
    SGD_UPDATE[blocks, threads](d_weights, d_grads, lr, n)


class SGD_GPU:
    """
    GPU-based Stochastic Gradient Descent Optimizer
    
    Keeps all weights on GPU and updates in-place without CPU transfer.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameter names to DeviceNDArray weights
    lr : float
        Learning rate (default: 0.01)
    
    Example
    -------
    >>> # Initialize weights on GPU
    >>> d_W1 = mt.to_device(W1)
    >>> d_b1 = mt.to_device(b1)
    >>> optimizer = SGD_GPU({'W1': d_W1, 'b1': d_b1}, lr=0.01)
    >>> # After computing gradients (as device arrays)
    >>> optimizer.step({'W1': d_grad_W1, 'b1': d_grad_b1})
    """
    def __init__(self, params, lr=0.01):
        self.params = params  # dict of name -> DeviceNDArray
        self.lr = lr
    
    def step(self, grads):
        """
        Update parameters using gradients (all on GPU)
        
        Parameters
        ----------
        grads : dict
            Dictionary of parameter names to gradient DeviceNDArrays
        """
        for name, d_param in self.params.items():
            if name in grads:
                d_grad = grads[name]
                # Ensure gradient is on GPU
                if not is_device_array(d_grad):
                    d_grad = to_device(d_grad)
                sgd_update_gpu(d_param, d_grad, self.lr)
        cuda.synchronize()
    
    def get_params(self):
        """Return parameters as CPU numpy arrays"""
        return {name: d_param.copy_to_host() for name, d_param in self.params.items()}