from numba import cuda , float32 , int32
import numpy as np
import math


def make_conv2d_kernel(shared_size , dtype):
    bytes_needed = shared_size * shared_size * (2 if dtype == np.float16 else 4)
    assert bytes_needed <= 49152 , \
        f"Shared memory {bytes_needed} bytes exceeds 48KB limit!"
    
    @cuda.jit
    def conv2d_kernel(A , K , out, padding , bias):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bz = cuda.blockIdx.z

        BW, BH = cuda.blockDim.x, cuda.blockDim.y
        
        N, Cin, H, W = A.shape
        Cout, _, Kh, Kw = K.shape
        _, _, out_h, out_w = out.shape

        n = bz // Cout
        c_out = bz % Cout

        i = by * BH + ty  
        j = bx * BW + tx  

        sh = cuda.shared.array((shared_size , shared_size) , dtype = dtype)
        sh_h = BH + Kh - 1
        sh_w = BW + Kw - 1
        
        base_i = by * BH - padding
        base_j = bx * BW - padding

        s = float32(0.0)
        
        for c_in in range(Cin):
            for ii in range(ty , sh_h , BH):  
                for jj in range(tx , sh_w , BW):  
                    global_i = base_i + ii
                    global_j = base_j + jj
                    if 0 <= global_i < H and 0 <= global_j < W:
                        sh[ii , jj] = A[n, c_in, global_i , global_j]
                    else:
                        sh[ii , jj] = dtype(0.0)
            
            cuda.syncthreads()

            if i < out_h and j < out_w:
                for u in range(Kh):
                    for v in range(Kw):
                        s += float32(sh[ty + u , tx + v]) * float32(K[c_out, c_in, u , v])  
            
            cuda.syncthreads()

        if i < out_h and j < out_w:
            s += float32(bias[c_out])
            out[n, c_out, i , j] = s
    
    return conv2d_kernel


def make_conv2d_direct(dtype):
    @cuda.jit
    def conv2d_direct(A , K , out, padding , bias):
        i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  
        bz = cuda.blockIdx.z

        N, Cin, H, W = A.shape
        Cout, _, Kh, Kw = K.shape
        _, _, out_h, out_w = out.shape

        n = bz // Cout
        c_out = bz % Cout

        if i < out_h and j < out_w:
            s = float32(0.0)
            for c_in in range(Cin):
                for u in range(Kh):
                    for v in range(Kw):
                        in_row = i - padding + u
                        in_col = j - padding + v
                        if 0 <= in_row < H and 0 <= in_col < W:
                            s += float32(A[n, c_in, in_row , in_col]) * float32(K[c_out, c_in, u , v])
            s += float32(bias[c_out])
            out[n, c_out, i , j] = s
    
    return conv2d_direct


def make_conv2d_backward_shared(shared_size, dtype):
    @cuda.jit
    def conv2d_backward_input_shared(grad_out, K, padding, grad_A):
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        bz = cuda.blockIdx.z
        BW, BH = cuda.blockDim.x, cuda.blockDim.y
        
        N, Cout, out_h, out_w = grad_out.shape
        _, Cin, Kh, Kw = K.shape
        _, _, H, W = grad_A.shape

        n = bz // Cin
        c_in = bz % Cin

        i = by * BH + ty
        j = bx * BW + tx
        
        sh = cuda.shared.array((shared_size, shared_size), dtype=dtype)
        
        base_i = by * BH - (Kh - 1) + padding
        base_j = bx * BW - (Kw - 1) + padding
        
        sh_h = BH + Kh - 1
        sh_w = BW + Kw - 1
        
        s = float32(0.0)

        for c_out in range(Cout):
            for ii in range(ty, sh_h, BH):
                for jj in range(tx, sh_w, BW):
                    gr, gc = base_i + ii, base_j + jj
                    if 0 <= gr < out_h and 0 <= gc < out_w:
                        sh[ii, jj] = grad_out[n, c_out, gr, gc]
                    else:
                        sh[ii, jj] = dtype(0.0)
            
            cuda.syncthreads()
            
            if i < H and j < W:
                for u in range(Kh):
                    for v in range(Kw):
                        sh_val = float32(sh[ty + (Kh - 1 - u), tx + (Kw - 1 - v)])
                        k_val = float32(K[c_out, c_in, u, v])
                        s += sh_val * k_val
            
            cuda.syncthreads()

        if i < H and j < W:
            grad_A[n, c_in, i, j] = s

    return conv2d_backward_input_shared


def make_conv2d_backward_global(dtype):
    @cuda.jit
    def conv2d_backward_input_global(grad_out, K, padding, grad_A):
        i, j = cuda.grid(2)
        bz = cuda.blockIdx.z
        
        N, Cout, out_h, out_w = grad_out.shape
        _, Cin, Kh, Kw = K.shape
        _, _, H, W = grad_A.shape

        n = bz // Cin
        c_in = bz % Cin

        if i < H and j < W:
            s = float32(0.0)
            for c_out in range(Cout):
                for u in range(Kh):
                    for v in range(Kw):
                        out_r = i + padding - u
                        out_c = j + padding - v
                        if 0 <= out_r < out_h and 0 <= out_c < out_w:
                           s += float32(grad_out[n, c_out, out_r, out_c]) * float32(K[c_out, c_in, u, v])
            grad_A[n, c_in, i, j] = s

    return conv2d_backward_input_global


@cuda.jit
def conv2d_backward_bias(grad_out, grad_bias):
    w, h, c = cuda.grid(3)
    N, C, H, W = grad_out.shape
    s_block_sum = cuda.shared.array(1, dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    if tx == 0 and ty == 0 and tz == 0:
        s_block_sum[0] = float32(0.0)
    
    cuda.syncthreads()

    if c < C and h < H and w < W:
        thread_sum = float32(0.0)
        for n in range(N):
            thread_sum += float32(grad_out[n, c, h, w])
        cuda.atomic.add(s_block_sum, 0, thread_sum)

    cuda.syncthreads()
    
    if tx == 0 and ty == 0 and tz == 0 and c < C:
        cuda.atomic.add(grad_bias, c, s_block_sum[0])

@cuda.jit
def conv2d_backward_weight_shared(input, grad_out, padding, grad_W):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z

    TILE_H = 16
    TILE_W = 16
    LINEAR_TID = ty * TILE_W + tx

    i = by * TILE_H + ty
    j = bx * TILE_W + tx

    Cout, Cin, Kh, Kw = grad_W.shape
    
    c_out = bz // Cin
    c_in = bz % Cin

    N, _, H_out, W_out = grad_out.shape
    _, _, H_in, W_in = input.shape
    
    s_partial = cuda.shared.array(256, dtype=float32)

    for u in range(Kh):
        for v in range(Kw):
            s = float32(0.0)
            
            if i < H_out and j < W_out:
                in_row = i + u - padding
                in_col = j + v - padding
                
                if 0 <= in_row < H_in and 0 <= in_col < W_in:
                    for n in range(N):
                        s += float32(grad_out[n, c_out, i, j]) * float32(input[n, c_in, in_row, in_col])
            
            s_partial[LINEAR_TID] = s
            cuda.syncthreads()
            
            stride = 128
            while stride > 0:
                if LINEAR_TID < stride:
                    s_partial[LINEAR_TID] += s_partial[LINEAR_TID + stride]
                cuda.syncthreads()
                stride //= 2
            
            if LINEAR_TID == 0:
                cuda.atomic.add(grad_W, (c_out, c_in, u, v), s_partial[0])
            
            cuda.syncthreads()

@cuda.jit
def conv2d_backward_weight_shared_2dchannel(input, grad_out, padding, grad_W):
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z

    TILE_H = 16
    TILE_W = 16
    LINEAR_TID = ty * TILE_W + tx

    i = by * TILE_H + ty
    j = bx * TILE_W + tx

    Kh, Kw = grad_W.shape
    u = bz // Kw
    v = bz % Kw

    N, H_out, W_out = grad_out.shape
    _, H_in, W_in = input.shape
    
    s_partial = cuda.shared.array(256, dtype=float32)
    s = float32(0.0)
    
    if i < H_out and j < W_out:
        in_row = i + u - padding
        in_col = j + v - padding
        
        if 0 <= in_row < H_in and 0 <= in_col < W_in:
            for n in range(N):
                s += float32(grad_out[n, i, j]) * float32(input[n, in_row, in_col])
    
    s_partial[LINEAR_TID] = s
    cuda.syncthreads()
    
    stride = 128
    while stride > 0:
        if LINEAR_TID < stride:
            s_partial[LINEAR_TID] += s_partial[LINEAR_TID + stride]
        cuda.syncthreads()
        stride //= 2
    
    if LINEAR_TID == 0:
        cuda.atomic.add(grad_W, (u, v), s_partial[0])

@cuda.jit
def relu_forward(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out.flat[i] = max(x.flat[i], 0.0)

@cuda.jit
def relu_backward(x, grad_out, grad_in):
    i = cuda.grid(1)
    if i < x.size:
        grad_in.flat[i] = grad_out.flat[i] if x.flat[i] > 0 else 0.0

@cuda.jit
def maxpool2d_forward(x, out, indices, pool_size):
    i, j = cuda.grid(2)
    bz = cuda.blockIdx.z
    
    N, C, H_in, W_in = x.shape
    _, _, H_out, W_out = out.shape
    
    n = bz // C
    c = bz % C
    
    if n < N and c < C and i < H_out and j < W_out:
        base_i = i * pool_size
        base_j = j * pool_size
        max_val = x[n, c, base_i, base_j]
        max_idx = int32(0)
        
        for u in range(pool_size):
            for v in range(pool_size):
                row = base_i + u
                col = base_j + v
                if row < H_in and col < W_in:
                    temp = x[n, c, row, col]
                    if temp > max_val:
                        max_val = temp
                        max_idx = int32(u * pool_size + v)
        
        out[n, c, i, j] = max_val
        indices[n, c, i, j] = max_idx

@cuda.jit
def maxpool2d_backward(grad_out, indices, grad_in, pool_size):
    i, j = cuda.grid(2)
    bz = cuda.blockIdx.z
    N, C, H_out, W_out = grad_out.shape
    
    n = bz // C
    c = bz % C

    if n < N and c < C and i < H_out and j < W_out:
        max_idx = indices[n, c, i, j]
        u = max_idx // pool_size
        v = max_idx % pool_size
        in_i = i * pool_size + u
        in_j = j * pool_size + v
        grad_in[n, c, in_i, in_j] = grad_out[n, c, i, j]

@cuda.jit 
def softmax_forward(logits, out):
    i, j = cuda.grid(2)
    n = cuda.blockIdx.z
    N, C, H, W = logits.shape
    
    if n < N and i < H and j < W:
        max_val = logits[n, 0, i, j]
        for c in range(1, C):
            max_val = max(max_val, logits[n, c, i, j])
        exp_sum = 0.0
        for c in range(C):
            exp_val = math.exp(logits[n, c, i, j] - max_val)
            out[n, c, i, j] = exp_val
            exp_sum += exp_val
        for c in range(C):
            out[n, c, i, j] /= exp_sum

@cuda.jit 
def softmax_backward(grad_out, probs, grad_logits):
    i, j = cuda.grid(2)
    N, C, H, W = grad_out.shape
    n = cuda.blockIdx.z
    
    if n < N and i < H and j < W:
        sum_grad = 0.0
        for c in range(C):
            sum_grad += grad_out[n, c, i, j] * probs[n, c, i, j]
        
        for c in range(C):
            grad_logits[n, c, i, j] = probs[n, c, i, j] * (grad_out[n, c, i, j] - sum_grad)

@cuda.jit
def matmul_tiled(A , B , C):
  M , N , K = A.shape[0] , B.shape[1] , A.shape[1]
  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y
  bx = cuda.blockIdx.x
  by = cuda.blockIdx.y

  TILE_M = 16   
  TILE_N = 16   
  TILE_K = 16

  row = by * TILE_M + ty
  col = bx * TILE_N + tx
  
  s_A = cuda.shared.array((16 , 16) , dtype = float32)
  s_B = cuda.shared.array((16 , 16) , dtype = float32)
  num_tiles = (K + TILE_K - 1) // TILE_K
  
  acc = float32(0.0)

  for t in range(num_tiles):
    k_base = t * TILE_K
    if row < M and (k_base + tx) < K:
      s_A[ty , tx] = float32(A[row , k_base + tx])
    else:
      s_A[ty , tx] = float32(0.0)

    if (k_base + ty) < K and col < N:
      s_B[ty , tx] = float32(B[k_base + ty , col])
    else:
      s_B[ty , tx] = float32(0.0)
  
    cuda.syncthreads()

    for k in range(TILE_K):
      acc += s_A[ty , k] * s_B[k , tx]
    cuda.syncthreads()

  if row < M and col < N:
    C[row , col] = acc

@cuda.jit
def cross_entropy_loss_kernel(probs, targets, loss_out, B, C):
    i = cuda.grid(1)
    
    if i < B:
        target_class = targets[i]
        prob_target = probs[i, target_class]
        loss_out[i] = -math.log(prob_target + 1e-8)


@cuda.jit
def cross_entropy_backward_kernel(probs, targets, grad, B, C):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    c = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if i < B and c < C:
        target_class = targets[i]
        if c == target_class:
            grad[i, c] = (probs[i, c] - 1.0) / B
        else:
            grad[i, c] = probs[i, c] / B


TIERS = {
    'tiny':   {'shared_size': 32  , 'block_size': 16 , 'use_shared': True}  ,
    'small':  {'shared_size': 48  , 'block_size': 16 , 'use_shared': True}  ,
    'medium': {'shared_size': 80  , 'block_size': 16 , 'use_shared': True}  ,
    'large':  {'shared_size': 110 , 'block_size': 16 , 'use_shared': True}  ,
    'xlarge': {'shared_size': 0   , 'block_size': 16 , 'use_shared': False}
}

KERNELS = {}
BACKWARD_KERNELS = {}

for tier_name , config in TIERS.items():
    if config['use_shared']:
        shared_size = config['shared_size']
        for dtype_name in ['fp16' , 'fp32']:
            dtype_type = np.float16 if dtype_name == 'fp16' else np.float32
            key = (tier_name , dtype_name)
            KERNELS[key] = make_conv2d_kernel(shared_size , dtype_type)
            BACKWARD_KERNELS[key] = make_conv2d_backward_shared(shared_size, dtype_type)

for dtype_name in ['fp16' , 'fp32']:
    dtype_type = np.float16 if dtype_name == 'fp16' else np.float32
    KERNELS[('xlarge' , dtype_name)] = make_conv2d_direct(dtype_type)
    BACKWARD_KERNELS[('xlarge', dtype_name)] = make_conv2d_backward_global(dtype_type)

@cuda.jit
def bias_add_2d(x, bias, out):
    i, j = cuda.grid(2)
    B, C = x.shape
    if i < B and j < C:
        out[i, j] = x[i, j] + bias[j]

@cuda.jit
def sgd_update_kernel(weights, grads, lr, n):
    i = cuda.grid(1)
    if i < n:
        weights.flat[i] -= lr * grads.flat[i]

BIAS_KERNEL = conv2d_backward_bias
WEIGHT_KERNEL = conv2d_backward_weight_shared
RELU_FORWARD = relu_forward
RELU_BACKWARD = relu_backward
MAXPOOL2D_FORWARD = maxpool2d_forward
MAXPOOL2D_BACKWARD = maxpool2d_backward
WEIGHT_KERNEL_2D_LEGACY = conv2d_backward_weight_shared_2dchannel
SOFTMAX_FORWARD = softmax_forward
SOFTMAX_BACKWARD = softmax_backward
BIAS_ADD_2D = bias_add_2d
SGD_UPDATE = sgd_update_kernel


