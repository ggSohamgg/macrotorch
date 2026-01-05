import numpy as np
import math
from numba import cuda
from .kernels import KERNELS , TIERS


def forward(A , K , dtype = 'auto' , verbose = False , d_A = None , d_K = None , d_out = None):
    assert A.ndim == 2 , f"A must be 2D, got shape {A.shape}"
    assert K.ndim == 2 , f"K must be 2D, got shape {K.shape}"
    
    H , W = A.shape
    Kh , Kw = K.shape
    
    assert H >= Kh , f"Kernel height {Kh} exceeds image height {H}"
    assert W >= Kw , f"Kernel width {Kw} exceeds image width {W}"
    
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
    
    out_h , out_w = H - Kh + 1 , W - Kw + 1
    
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
    kernel[blocks_per_grid , threads_per_block](d_A , d_K , d_out)
    
    cuda.synchronize()
    return d_out.copy_to_host()


def backward():
    pass
