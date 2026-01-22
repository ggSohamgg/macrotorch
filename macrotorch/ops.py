import numpy as np
import math
from numba import cuda
from .kernels import KERNELS , BACKWARD_KERNELS, BIAS_KERNEL, WEIGHT_KERNEL, TIERS, RELU_FORWARD, RELU_BACKWARD, MAXPOOL2D_FORWARD, MAXPOOL2D_BACKWARD, SOFTMAX_FORWARD, SOFTMAX_BACKWARD, matmul_tiled, cross_entropy_loss_kernel, cross_entropy_backward_kernel


def forward(A , K , padding=0, bias=None, dtype='auto' , verbose=False , d_A=None , d_K=None , d_out=None):
    """
    2D Convolution Forward Pass (Multi-Channel Batched)
    
    Parameters
    ----------
    A : numpy.ndarray
        Input tensor of shape (N, C_in, H, W).
    K : numpy.ndarray
        Kernel tensor of shape (C_out, C_in, Kh, Kw).
    padding : int
        Zero-padding added to both sides of the input.
    bias : numpy.ndarray or None
        Bias tensor of shape (C_out,).
    """
    assert A.ndim == 4 , f"A must be 4D (N, C, H, W), got shape {A.shape}"
    assert K.ndim == 4 , f"K must be 4D (Cout, Cin, Kh, Kw), got shape {K.shape}"
    
    N, Cin, H, W = A.shape
    Cout, Cin_K, Kh, Kw = K.shape
    assert Cin == Cin_K, f"Input channels {Cin} must match kernel in_channels {Cin_K}"

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
        print(f"Algorithm: {tier.upper()} ({dtype.upper()}) - {memory_type}")
    
    out_h = H - Kh + 1 + (2 * padding)
    out_w = W - Kw + 1 + (2 * padding)
    
    if d_A is None:
        d_A = cuda.to_device(A)
    if d_K is None:
        d_K = cuda.to_device(K)
    
    if bias is None:
        bias = np.zeros(Cout, dtype=A.dtype)
    d_bias = cuda.to_device(bias)

    if d_out is None:
        out = np.zeros((N, Cout, out_h , out_w) , dtype=np.float32)
        d_out = cuda.to_device(out)
    
    threads_per_block = (block_size , block_size)
    blocks_y = math.ceil(out_h / block_size)  
    blocks_x = math.ceil(out_w / block_size)  
    blocks_z = N * Cout
    blocks_per_grid = (blocks_x , blocks_y, blocks_z)  
    
    kernel = KERNELS[(tier , dtype)]
    kernel[blocks_per_grid , threads_per_block](d_A , d_K , d_out, padding, d_bias)
    
    cuda.synchronize()
    return d_out.copy_to_host()


def input_backward(grad_out, K, padding=0, dtype='auto', verbose=False):
    """
    2D Convolution Backward Pass (Input Gradient - Multi-Channel Batched)
    """
    assert grad_out.ndim == 4
    assert K.ndim == 4
    
    N, Cout, out_h, out_w = grad_out.shape
    Cout_K, Cin, Kh, Kw = K.shape
    assert Cout == Cout_K
    
    if dtype == "auto":
        dtype = 'fp16' if grad_out.dtype == np.float16 else 'fp32'
        
    H_in = out_h + Kh - 1 - (2 * padding)
    W_in = out_w + Kw - 1 - (2 * padding)
    
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
    
    grad_A = np.zeros((N, Cin, H_in, W_in), dtype=np.float32)
    
    d_grad_out = cuda.to_device(grad_out)
    d_K = cuda.to_device(K)
    d_grad_A = cuda.to_device(grad_A)
    
    threads_per_block = (block_size, block_size)
    blocks_y = math.ceil(H_in / block_size)
    blocks_x = math.ceil(W_in / block_size)
    blocks_z = N * Cin
    blocks_per_grid = (blocks_x, blocks_y, blocks_z)
    
    kernel = BACKWARD_KERNELS[(tier, dtype)]
    kernel[blocks_per_grid, threads_per_block](d_grad_out, d_K, padding, d_grad_A)
    
    cuda.synchronize()
    return d_grad_A.copy_to_host()


def weight_backward(grad_out, A, padding=0, dtype='auto', verbose=False, d_grad_out=None, d_A=None, d_grad_W=None):
    """
    2D Convolution Backward Pass (Weight Gradient - Multi-Channel Batched)
    """
    if d_grad_out is None:
        assert grad_out.ndim == 4
        assert A.ndim == 4
        
        if dtype == 'auto':
            dtype = 'fp16' if grad_out.dtype == np.float16 else 'fp32'
            
        d_grad_out = cuda.to_device(grad_out)
        d_A = cuda.to_device(A)

    N, Cout, H_out, W_out = d_grad_out.shape
    _, Cin, H_in, W_in = d_A.shape
    
    Kh = H_in + 2 * padding - H_out + 1
    Kw = W_in + 2 * padding - W_out + 1
    
    if d_grad_W is None:
        grad_W = np.zeros((Cout, Cin, Kh, Kw), dtype=np.float32)
        d_grad_W = cuda.to_device(grad_W)
        return_host = True
    else:
        cuda.to_device(np.zeros((Cout, Cin, Kh, Kw), dtype=np.float32), to=d_grad_W) # zero init
        return_host = False
        
    # Grid Configuration
    TILE_H, TILE_W = 16, 16
    
    threads_per_block = (TILE_W, TILE_H)
    blocks_x = math.ceil(W_out / TILE_W)
    blocks_y = math.ceil(H_out / TILE_H)
    blocks_z = Cout * Cin
    blocks_per_grid = (blocks_x, blocks_y, blocks_z)
    
    WEIGHT_KERNEL[blocks_per_grid, threads_per_block](d_A, d_grad_out, padding, d_grad_W)
    
    cuda.synchronize()
    
    if return_host:
        return d_grad_W.copy_to_host()
    else:
        return d_grad_W


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


def relu(x, d_x=None, d_out=None):
    """
    ReLU Forward Pass
    
    Computes element-wise max(0, x) using CUDA.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array of any shape
    d_x : numba.cuda.DeviceNDArray, optional
        Pre-allocated device input array
    d_out : numba.cuda.DeviceNDArray, optional
        Pre-allocated device output array
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        ReLU output with same shape as input
    """
    if d_x is None:
        d_x = cuda.to_device(x)
        return_host = True
    else:
        return_host = False
    
    if d_out is None:
        d_out = cuda.device_array(d_x.shape, dtype=d_x.dtype)
    
    # 1D grid for element-wise operation
    threads_per_block = 256
    blocks_per_grid = math.ceil(d_x.size / threads_per_block)
    
    RELU_FORWARD[blocks_per_grid, threads_per_block](d_x, d_out)
    cuda.synchronize()
    
    if return_host:
        return d_out.copy_to_host()
    else:
        return d_out


def relu_backward(x, grad_out, d_x=None, d_grad_out=None, d_grad_in=None):
    """
    ReLU Backward Pass
    
    Computes gradient through ReLU: grad_in = grad_out * (x > 0)
    
    Parameters
    ----------
    x : numpy.ndarray
        Original input to ReLU (before ReLU was applied)
    grad_out : numpy.ndarray
        Gradient from downstream layer
    d_x : numba.cuda.DeviceNDArray, optional
        Pre-allocated device input array
    d_grad_out : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient output array
    d_grad_in : numba.cuda.DeviceNDArray, optional
        Pre-allocated device gradient input array
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        Gradient with respect to input
    """
    if d_x is None:
        d_x = cuda.to_device(x)
        return_host = True
    else:
        return_host = False
    
    if d_grad_out is None:
        d_grad_out = cuda.to_device(grad_out)
    
    if d_grad_in is None:
        d_grad_in = cuda.device_array(d_x.shape, dtype=np.float32)
    
    # 1D grid for element-wise operation
    threads_per_block = 256
    blocks_per_grid = math.ceil(d_x.size / threads_per_block)
    
    RELU_BACKWARD[blocks_per_grid, threads_per_block](d_x, d_grad_out, d_grad_in)
    cuda.synchronize()
    
    if return_host:
        return d_grad_in.copy_to_host()
    else:
        return d_grad_in


def softmax_forward(x, d_x=None, d_out=None):
    """
    Softmax Forward Pass (4D Multi-Channel)
    
    Computes softmax along channel dimension for 4D input (N, C, H, W).
    
    Parameters
    ----------
    x : numpy.ndarray
        Input logits of shape (N, C, H, W)
        
    Returns
    -------
    numpy.ndarray
        Softmax probabilities of shape (N, C, H, W)
    """
    N, C, H, W = x.shape
    
    if d_x is None:
        d_x = cuda.to_device(x.astype(np.float32))
        return_host = True
    else:
        return_host = False
    
    if d_out is None:
        d_out = cuda.device_array((N, C, H, W), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W / 16)
    blocks_y = math.ceil(H / 16)
    blocks_z = N
    blocks = (blocks_x, blocks_y, blocks_z)
    
    SOFTMAX_FORWARD[blocks, threads](d_x, d_out)
    cuda.synchronize()
    
    if return_host:
        return d_out.copy_to_host()
    else:
        return d_out

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


def maxpool2d_forward(x, pool_size=2, d_x=None, d_out=None, d_indices=None):
    """
    MaxPool2D Forward Pass (4D Multi-Channel)
    
    Computes max pooling over 4D input (N, C, H, W).
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array of shape (N, C, H, W)
    pool_size : int
        Size of pooling window (default: 2)
        
    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        (output, indices) - pooled output (N, C, H_out, W_out) and max indices for backward pass
    """
    N, C, H, W = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    
    if d_x is None:
        d_x = cuda.to_device(x)
        return_host = True
    else:
        return_host = False
    
    if d_out is None:
        d_out = cuda.device_array((N, C, H_out, W_out), dtype=x.dtype)
    
    if d_indices is None:
        d_indices = cuda.device_array((N, C, H_out, W_out), dtype=np.int32)
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    MAXPOOL2D_FORWARD[blocks, threads](d_x, d_out, d_indices, pool_size)
    cuda.synchronize()
    
    if return_host:
        return d_out.copy_to_host(), d_indices.copy_to_host()
    else:
        return d_out, d_indices


def maxpool2d_backward(grad_out, indices, input_shape, pool_size=2, d_grad_out=None, d_indices=None, d_grad_in=None):
    """
    MaxPool2D Backward Pass (4D Multi-Channel)
    
    Computes gradient w.r.t. input for max pooling.
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient from next layer of shape (N, C, H_out, W_out)
    indices : numpy.ndarray
        Indices from forward pass of shape (N, C, H_out, W_out)
    input_shape : tuple
        Shape of original input (N, C, H, W)
    pool_size : int
        Size of pooling window (default: 2)
        
    Returns
    -------
    numpy.ndarray
        Gradient w.r.t. input of shape (N, C, H, W)
    """
    N, C, H_out, W_out = grad_out.shape
    
    if d_grad_out is None:
        d_grad_out = cuda.to_device(grad_out)
        return_host = True
    else:
        return_host = False
    
    if d_indices is None:
        d_indices = cuda.to_device(indices)
    
    if d_grad_in is None:
        d_grad_in = cuda.device_array(input_shape, dtype=np.float32)
        d_grad_in[:] = 0.0
    
    threads = (16, 16)
    blocks_x = math.ceil(W_out / 16)
    blocks_y = math.ceil(H_out / 16)
    blocks_z = N * C
    blocks = (blocks_x, blocks_y, blocks_z)
    
    MAXPOOL2D_BACKWARD[blocks, threads](d_grad_out, d_indices, d_grad_in, pool_size)
    cuda.synchronize()
    
    if return_host:
        return d_grad_in.copy_to_host()
    else:
        return d_grad_in


def matmul(A, B, d_A=None, d_B=None, d_C=None):
    """
    Tiled Matrix Multiplication using CUDA
    
    Performs C = A @ B using a tiled CUDA kernel with shared memory.
    Uses 16x16 tiles for efficient GPU computation.
    Supports both FP32 and FP16 inputs (uses FP32 accumulation for accuracy).
    
    Parameters
    ----------
    A : numpy.ndarray
        Input matrix of shape (M, K), supports float32 or float16
    B : numpy.ndarray
        Input matrix of shape (K, N), supports float32 or float16
    d_A : numba.cuda.DeviceNDArray, optional
        Pre-allocated device array for A
    d_B : numba.cuda.DeviceNDArray, optional
        Pre-allocated device array for B
    d_C : numba.cuda.DeviceNDArray, optional
        Pre-allocated device output array
        
    Returns
    -------
    numpy.ndarray or numba.cuda.DeviceNDArray
        Result matrix C of shape (M, N) in float32
    """
    if d_A is None:
        assert A.ndim == 2, f"A must be 2D, got shape {A.shape}"
        assert B.ndim == 2, f"B must be 2D, got shape {B.shape}"
        assert A.shape[1] == B.shape[0], f"Inner dimensions must match: A.shape[1]={A.shape[1]} != B.shape[0]={B.shape[0]}"
        
        # Transfer to GPU preserving dtype - kernel handles float32 conversion
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        return_host = True
    else:
        return_host = False
    
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
    
    if return_host:
        return d_C.copy_to_host()
    else:
        return d_C


def linear(x, weight, bias=None):
    """
    Linear Layer Forward Pass
    
    Computes Y = X @ W + b using CUDA matmul kernel.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input tensor of shape (B, in_features)
    weight : numpy.ndarray
        Weight matrix of shape (in_features, out_features)
    bias : numpy.ndarray, optional
        Bias vector of shape (out_features,)
    
    Returns
    -------
    numpy.ndarray
        Output tensor of shape (B, out_features)
    """
    out = matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def linear_backward(grad_out, x, weight):
    """
    Linear Layer Backward Pass
    
    Computes gradients for linear layer.
    
    Parameters
    ----------
    grad_out : numpy.ndarray
        Gradient from next layer of shape (B, out_features)
    x : numpy.ndarray
        Input from forward pass of shape (B, in_features)
    weight : numpy.ndarray
        Weight matrix of shape (in_features, out_features)
    
    Returns
    -------
    tuple (dX, dW, db)
        dX : Gradient w.r.t. input of shape (B, in_features)
        dW : Gradient w.r.t. weight of shape (in_features, out_features)
        db : Gradient w.r.t. bias of shape (out_features,)
    """
    # dW = X.T @ dY
    dW = matmul(x.T, grad_out)
    
    # dX = dY @ W.T
    dX = matmul(grad_out, weight.T)
    
    # db = sum(dY, axis=0)
    db = np.sum(grad_out, axis=0)
    
    return dX, dW, db


def cross_entropy_loss(probs, targets, d_probs=None, d_targets=None, d_loss=None):
    """
    Cross-Entropy Loss Forward Pass
    
    Computes the cross-entropy loss for multi-class classification.
    
    Parameters
    ----------
    probs : numpy.ndarray
        Softmax probabilities of shape (B, C) where B is batch size, C is num classes
    targets : numpy.ndarray
        Target class indices of shape (B,), integer values in [0, C-1]
    
    Returns
    -------
    float
        Mean cross-entropy loss over the batch
    """
    B, C = probs.shape
    
    if d_probs is None:
        d_probs = cuda.to_device(probs.astype(np.float32))
        d_targets = cuda.to_device(targets.astype(np.int32))
        return_host = True
    else:
        return_host = False
    
    if d_loss is None:
        d_loss = cuda.device_array(B, dtype=np.float32)
    
    threads_per_block = 256
    blocks_per_grid = math.ceil(B / threads_per_block)
    
    cross_entropy_loss_kernel[blocks_per_grid, threads_per_block](d_probs, d_targets, d_loss, B, C)
    cuda.synchronize()
    
    if return_host:
        loss = d_loss.copy_to_host()
        return np.mean(loss)
    else:
        return d_loss


def cross_entropy_backward(probs, targets, d_probs=None, d_targets=None, d_grad=None):
    """
    Cross-Entropy Loss Backward Pass
    
    Computes gradient of cross-entropy loss w.r.t. logits (after softmax).
    
    Parameters
    ----------
    probs : numpy.ndarray
        Softmax probabilities of shape (B, C)
    targets : numpy.ndarray
        Target class indices of shape (B,)
    
    Returns
    -------
    numpy.ndarray
        Gradient w.r.t. softmax input of shape (B, C)
    """
    B, C = probs.shape
    
    if d_probs is None:
        d_probs = cuda.to_device(probs.astype(np.float32))
        d_targets = cuda.to_device(targets.astype(np.int32))
        return_host = True
    else:
        return_host = False
    
    if d_grad is None:
        d_grad = cuda.device_array((B, C), dtype=np.float32)
    
    threads = (16, 16)
    blocks_x = math.ceil(B / 16)
    blocks_y = math.ceil(C / 16)
    blocks = (blocks_x, blocks_y)
    
    cross_entropy_backward_kernel[blocks, threads](d_probs, d_targets, d_grad, B, C)
    cuda.synchronize()
    
    if return_host:
        return d_grad.copy_to_host()
    else:
        return d_grad


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