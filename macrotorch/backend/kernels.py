from numba import cuda , float32
import numpy as np
import math


def make_conv2d_kernel(shared_size , dtype):
    bytes_needed = shared_size * shared_size * (2 if dtype == np.float16 else 4)
    assert bytes_needed <= 49152 , \
        f"Shared memory {bytes_needed} bytes exceeds 48KB limit!"
    
    @cuda.jit
    def conv2d_kernel(A , K , out, padding):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y

        BW = cuda.blockDim.x
        BH = cuda.blockDim.y

        i = by * BH + ty  
        j = bx * BW + tx  

        H , W = A.shape
        Kh , Kw = K.shape
        out_h , out_w = out.shape

        sh = cuda.shared.array((shared_size , shared_size) , dtype = dtype)
        sh_h = BH + Kh - 1
        sh_w = BW + Kw - 1
        
        base_i = by * BH - padding
        base_j = bx * BW - padding

        for ii in range(ty , sh_h , BH):  
            for jj in range(tx , sh_w , BW):  
                global_i = base_i + ii
                global_j = base_j + jj
                if 0 <= global_i < H and 0 <= global_j < W:
                    sh[ii , jj] = A[global_i , global_j]
                else:
                    sh[ii , jj] = dtype(0.0)
        
        cuda.syncthreads()

        if i < out_h and j < out_w:
            s = float32(0.0)
            for u in range(Kh):
                for v in range(Kw):
                    s += float32(sh[ty + u , tx + v]) * float32(K[u , v])  
            out[i , j] = s
    
    return conv2d_kernel


def make_conv2d_direct(dtype):
    @cuda.jit
    def conv2d_direct(A , K , out, padding):
        i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  
        
        out_h , out_w = out.shape
        Kh , Kw = K.shape
        H, W = A.shape

        if i < out_h and j < out_w:
            s = float32(0.0)
            for u in range(Kh):
                for v in range(Kw):
                    in_row = i - padding + u
                    in_col = j - padding + v
                    if 0 <= in_row < H and 0 <= in_col < W:
                        s += float32(A[in_row , in_col]) * float32(K[u , v])
            out[i , j] = s
    
    return conv2d_direct


TIERS = {
    'tiny':   {'shared_size': 32  , 'block_size': 16 , 'use_shared': True}  ,
    'small':  {'shared_size': 48  , 'block_size': 16 , 'use_shared': True}  ,
    'medium': {'shared_size': 80  , 'block_size': 16 , 'use_shared': True}  ,
    'large':  {'shared_size': 110 , 'block_size': 16 , 'use_shared': True}  ,
    'xlarge': {'shared_size': 0   , 'block_size': 16 , 'use_shared': False}
}

KERNELS = {}

for tier_name , config in TIERS.items():
    if config['use_shared']:
        shared_size = config['shared_size']
        for dtype_name in ['fp16' , 'fp32']:
            dtype_type = np.float16 if dtype_name == 'fp16' else np.float32
            key = (tier_name , dtype_name)
            KERNELS[key] = make_conv2d_kernel(shared_size , dtype_type)

for dtype_name in ['fp16' , 'fp32']:
    dtype_type = np.float16 if dtype_name == 'fp16' else np.float32
    KERNELS[('xlarge' , dtype_name)] = make_conv2d_direct(dtype_type)
