# @title GPU-Only Benchmark: MacroTorch (tqdm, no PyTorch)
"""
MNIST Benchmark - GPU-ONLY
- MacroTorch (Numba CUDA kernels)
- Batch sizes: 32, 64, 128, 256, 512, 1024
- 1 Epoch each (configurable)
- Uses tqdm progress bars
"""

# ============================================================================
# CELL 1: Install
# ============================================================================
# print("=" * 70)
# print("Installing MacroTorch")
# print("=" * 70)
# !pip install git+https://github.com/ggSohamgg/macrotorch.git --upgrade -q
# print("Installed!")


# ============================================================================
# CELL 2: Imports
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

import numpy as np
import time
from urllib.request import urlopen
import gzip
import struct
from numba import cuda
from tqdm import tqdm  # tqdm for notebooks/scripts

import macrotorch as mt

print(f"\n  • MacroTorch: {mt.__version__}")
print(f"  • GPU: {cuda.get_current_device().name.decode()}")


# ============================================================================
# CELL 3: Load MNIST
# ============================================================================
def load_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def read_images(url):
        with urlopen(url) as f:
            data = gzip.decompress(f.read())
        _, num, rows, cols = struct.unpack('>IIII', data[:16])
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num, 1, rows, cols)

    def read_labels(url):
        with urlopen(url) as f:
            data = gzip.decompress(f.read())
        return np.frombuffer(data[8:], dtype=np.uint8)

    X_train = read_images(base_url + 'train-images-idx3-ubyte.gz').astype(np.float32) / 255.0
    y_train = read_labels(base_url + 'train-labels-idx1-ubyte.gz').astype(np.int64)
    X_test  = read_images(base_url + 't10k-images-idx3-ubyte.gz').astype(np.float32) / 255.0
    y_test  = read_labels(base_url + 't10k-labels-idx1-ubyte.gz').astype(np.int64)
    return X_train, y_train, X_test, y_test

print("\nLoading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================================
# CELL 4: Weight Initialization
# ============================================================================
def he_init(shape):
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

def get_weights(seed=42):
    # USER-EDIT: change seed for different init
    np.random.seed(seed)
    return {
        'W_conv1': he_init((16, 1, 3, 3)),
        'b_conv1': np.zeros(16, dtype=np.float32),

        'W_conv2': he_init((32, 16, 3, 3)),
        'b_conv2': np.zeros(32, dtype=np.float32),

        'W_fc1': he_init((1568, 128)),
        'b_fc1': np.zeros(128, dtype=np.float32),

        'W_fc2': he_init((128, 10)),
        'b_fc2': np.zeros(10, dtype=np.float32),
    }


# ============================================================================
# CELL 5: Model Summary (no PyTorch)
# ============================================================================
def print_model_summary():
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)

    print("\nMNISTNet(")
    print("  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (pool): MaxPool2d(kernel_size=2, stride=2)")
    print("  (fc1): Linear(in_features=1568, out_features=128, bias=True)")
    print("  (fc2): Linear(in_features=128, out_features=10, bias=True)")
    print("  (relu): ReLU()")
    print(")")

    print("\n" + "─" * 80)
    print("Layer (type:depth-idx)                   Output Shape              Param #")
    print("=" * 80)

    batch_size = 64  # USER-EDIT: just for display

    layers = [
        ("Conv2d: 1-1",   f"[{batch_size}, 16, 28, 28]", 16*1*3*3 + 16),
        ("ReLU: 1-2",     f"[{batch_size}, 16, 28, 28]", 0),
        ("MaxPool: 1-3",  f"[{batch_size}, 16, 14, 14]", 0),
        ("Conv2d: 1-4",   f"[{batch_size}, 32, 14, 14]", 32*16*3*3 + 32),
        ("ReLU: 1-5",     f"[{batch_size}, 32, 14, 14]", 0),
        ("MaxPool: 1-6",  f"[{batch_size}, 32, 7, 7]",   0),
        ("Linear: 1-7",   f"[{batch_size}, 128]",        1568*128 + 128),
        ("ReLU: 1-8",     f"[{batch_size}, 128]",        0),
        ("Linear: 1-9",   f"[{batch_size}, 10]",         128*10 + 10),
    ]

    total_params = 0
    for layer_name, output_shape, params in layers:
        total_params += params
        print(f"{layer_name:<40} {output_shape:<25} {params:>10,}")

    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print("─" * 80)

print_model_summary()


# ============================================================================
# CELL 6: MacroTorch GPU Training (tqdm)
# ============================================================================
def train_macrotorch_gpu_epoch(X_train, y_train, batch_size, lr=0.01, seed=42):
    """
    MacroTorch training for 1 epoch.

    USER-EDIT knobs:
    - lr
    - seed
    - (below) where CPU transfers happen (linear bias + softmax path)
    """
    np.random.seed(seed)
    w = get_weights(seed=seed)

    # Upload all weights to GPU ONCE
    d_W_conv1 = mt.to_device(w['W_conv1'])
    d_W_conv2 = mt.to_device(w['W_conv2'])
    d_W_fc1   = mt.to_device(w['W_fc1'])
    d_W_fc2   = mt.to_device(w['W_fc2'])

    d_b_conv1 = mt.to_device(w['b_conv1'])
    d_b_conv2 = mt.to_device(w['b_conv2'])
    d_b_fc1   = mt.to_device(w['b_fc1'])
    d_b_fc2   = mt.to_device(w['b_fc2'])

    n_batches = len(X_train) // batch_size
    perm = np.random.permutation(len(X_train))

    # Warmup (keeps compilation out of timing)
    dummy = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    _ = mt.conv2d_forward(mt.to_device(dummy), d_W_conv1, padding=1, bias=d_b_conv1, return_device=True)
    cuda.synchronize()

    correct = 0
    start = time.time()

    pbar = tqdm(range(n_batches), desc=f"MacroTorch epoch (bs={batch_size})", leave=False)
    for i in pbar:
        idx = perm[i*batch_size:(i+1)*batch_size]
        X_batch = X_train[idx]
        y_batch = y_train[idx].astype(np.int32)

        # TRANSFER: input batch to GPU
        d_X = mt.to_device(X_batch)

        # ===== FORWARD (GPU) =====
        d_z1 = mt.conv2d_forward(d_X,  d_W_conv1, padding=1, bias=d_b_conv1, return_device=True)
        d_a1 = mt.relu(d_z1, return_device=True)
        d_p1, d_idx1 = mt.maxpool2d_forward(d_a1, pool_size=2, return_device=True)

        d_z2 = mt.conv2d_forward(d_p1, d_W_conv2, padding=1, bias=d_b_conv2, return_device=True)
        d_a2 = mt.relu(d_z2, return_device=True)
        d_p2, d_idx2 = mt.maxpool2d_forward(d_a2, pool_size=2, return_device=True)

        # Flatten on GPU (view)
        d_flat = d_p2.reshape((batch_size, 1568))

        # NOTE: mt.linear currently adds bias via a kernel, but expects bias on CPU in your code path.
        # USER-EDIT: if your mt.linear supports device bias, remove these copy_to_host() calls.
        b_fc1_cpu = d_b_fc1.copy_to_host()
        d_z3 = mt.linear(d_flat, d_W_fc1, b_fc1_cpu, return_device=True)
        d_a3 = mt.relu(d_z3, return_device=True)

        b_fc2_cpu = d_b_fc2.copy_to_host()
        d_z4 = mt.linear(d_a3, d_W_fc2, b_fc2_cpu, return_device=True)

        # Softmax API is 4D; reshape via CPU for now
        # USER-EDIT: if you add a 2D softmax kernel, remove this transfer.
        z4 = d_z4.copy_to_host()
        z4_4d = z4.reshape(batch_size, 10, 1, 1)
        d_probs = mt.softmax_forward(z4_4d, return_device=True)

        probs = d_probs.copy_to_host().reshape(batch_size, 10)
        correct += (probs.argmax(1) == y_batch).sum()

        # ===== BACKWARD =====
        d_dz4 = mt.cross_entropy_backward(probs, y_batch, return_device=True)

        da3, dW_fc2, db_fc2 = mt.linear_backward(d_dz4, d_a3, d_W_fc2, return_device=False)
        d_da3 = mt.to_device(da3)

        d_dz3 = mt.relu_backward(d_z3, d_da3, return_device=True)

        dflat, dW_fc1, db_fc1 = mt.linear_backward(d_dz3, d_flat, d_W_fc1, return_device=False)

        dp2 = mt.flatten_backward(dflat, d_p2.shape)
        d_dp2 = mt.to_device(dp2)

        d_da2 = mt.maxpool2d_backward(d_dp2, d_idx2, d_a2.shape, pool_size=2, return_device=True)
        d_dz2 = mt.relu_backward(d_z2, d_da2, return_device=True)

        # Conv2 grads
        dW_conv2 = mt.conv2d_weight_backward(
            d_dz2, d_p1, padding=1, Kh=3, Kw=3,
            d_grad_out=d_dz2, d_A=d_p1, return_device=False
        )
        db_conv2 = mt.conv2d_bias_backward(d_dz2, d_grad_out=d_dz2)

        dp1 = mt.conv2d_input_backward(d_dz2, d_W_conv2, padding=1, return_device=False)
        d_dp1 = mt.to_device(dp1)

        d_da1 = mt.maxpool2d_backward(d_dp1, d_idx1, d_a1.shape, pool_size=2, return_device=True)
        d_dz1 = mt.relu_backward(d_z1, d_da1, return_device=True)

        # Conv1 grads
        dW_conv1 = mt.conv2d_weight_backward(
            d_dz1, d_X, padding=1, Kh=3, Kw=3,
            d_grad_out=d_dz1, d_A=d_X, return_device=False
        )
        db_conv1 = mt.conv2d_bias_backward(d_dz1, d_grad_out=d_dz1)

        # ===== SGD updates (GPU in-place) =====
        mt.sgd_update_gpu(d_W_conv1, mt.to_device(dW_conv1), lr)
        mt.sgd_update_gpu(d_W_conv2, mt.to_device(dW_conv2), lr)
        mt.sgd_update_gpu(d_W_fc1,   mt.to_device(dW_fc1),   lr)
        mt.sgd_update_gpu(d_W_fc2,   mt.to_device(dW_fc2),   lr)

        mt.sgd_update_gpu(d_b_conv1, mt.to_device(db_conv1), lr)
        mt.sgd_update_gpu(d_b_conv2, mt.to_device(db_conv2), lr)
        mt.sgd_update_gpu(d_b_fc1,   mt.to_device(db_fc1),   lr)
        mt.sgd_update_gpu(d_b_fc2,   mt.to_device(db_fc2),   lr)

        # Update progress bar every 10 iterations for smoothness
        if (i + 1) % 10 == 0:
            pbar.set_postfix(acc=f"{100.0 * correct / ((i+1)*batch_size):.1f}%")

    cuda.synchronize()
    elapsed = time.time() - start
    acc = 100.0 * correct / (n_batches * batch_size)
    return elapsed, acc


# ============================================================================
# CELL 7: Run Benchmarks (MacroTorch only)
# ============================================================================
print("\n" + "=" * 70)
print("GPU Benchmark: MacroTorch (1 Epoch)")
print("=" * 70)

# USER-EDIT: change these
batch_sizes = [32, 64, 128, 256, 512, 1024]
lr = 0.01
seed = 42

results = []

for bs in batch_sizes:
    t_mt, acc_mt = train_macrotorch_gpu_epoch(X_train, y_train, bs, lr=lr, seed=seed)
    print(f"  MacroTorch GPU: {t_mt:6.2f}s | Acc: {acc_mt:5.1f}%")

    results.append({
        'batch': bs,
        'macrotorch_time': t_mt,
        'macrotorch_acc': acc_mt,
    })


# ============================================================================
# CELL 8: Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("Performance Summary (MacroTorch only)")
print("=" * 70)

print("\n┌─────────┬─────────────────┬──────────────┐")
print("│  Batch  │ MacroTorch (s)  │   Acc (%)    │")
print("├─────────┼─────────────────┼──────────────┤")

for r in results:
    print(f"│  {r['batch']:5}  │      {r['macrotorch_time']:6.2f}     │    {r['macrotorch_acc']:6.2f}   │")

print("└─────────┴─────────────────┴──────────────┘")

avg_time = float(np.mean([r['macrotorch_time'] for r in results]))
print(f"\nAverage epoch time (over batch_sizes): {avg_time:.2f}s")


# ============================================================================
# CELL 9: System Info
# ============================================================================
print("\n" + "=" * 70)
print("System Information")
print("=" * 70)

dev = cuda.get_current_device()
free_b, total_b = cuda.current_context().get_memory_info()  # (free, total) bytes [web:22]

print(f"GPU: {dev.name.decode()}")
print(f"GPU Memory (total): {total_b / 1e9:.1f} GB")
print(f"MacroTorch: {mt.__version__}")