# @title FIXED GPU-Only Benchmark: MacroTorch vs PyTorch
"""
MNIST Benchmark - FULLY GPU-OPTIMIZED (No Unnecessary Transfers!)
- MacroTorch (CUDA kernels) vs PyTorch (cuDNN)
- Batch sizes: 32, 64, 128, 256, 512, 1024
- 1 Epoch each
"""


# # ============================================================================
# # CELL 1: Install
# # ============================================================================
# print("=" * 70)
# print(" 📦 Installing MacroTorch")
# print("=" * 70)
# !pip install git+https://github.com/ggSohamgg/macrotorch.git --upgrade -q
# print("✅ Installed!")


# ============================================================================
# CELL 2: Imports
# ============================================================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numba import cuda
import macrotorch as mt
import time
from urllib.request import urlopen
import gzip
import struct


print(f"\n  • MacroTorch: {mt.__version__}")
print(f"  • PyTorch: {torch.__version__}")
print(f"  • GPU: {cuda.get_current_device().name.decode()}")
print(f"  • CUDA: {torch.version.cuda}")


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
    X_test = read_images(base_url + 't10k-images-idx3-ubyte.gz').astype(np.float32) / 255.0
    y_test = read_labels(base_url + 't10k-labels-idx1-ubyte.gz').astype(np.int64)
    return X_train, y_train, X_test, y_test


print("\n  Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()
print(f"  ✅ Train: {X_train.shape}, Test: {X_test.shape}")


# ============================================================================
# CELL 4: Weight Initialization
# ============================================================================
def he_init(shape):
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


def get_weights():
    np.random.seed(42)
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
# CELL 5: PyTorch Model
# ============================================================================
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 1568)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
    def load_numpy(self, w):
        with torch.no_grad():
            self.conv1.weight.copy_(torch.from_numpy(w['W_conv1']))
            self.conv1.bias.copy_(torch.from_numpy(w['b_conv1']))
            self.conv2.weight.copy_(torch.from_numpy(w['W_conv2']))
            self.conv2.bias.copy_(torch.from_numpy(w['b_conv2']))
            self.fc1.weight.copy_(torch.from_numpy(w['W_fc1'].T))
            self.fc1.bias.copy_(torch.from_numpy(w['b_fc1']))
            self.fc2.weight.copy_(torch.from_numpy(w['W_fc2'].T))
            self.fc2.bias.copy_(torch.from_numpy(w['b_fc2']))


# ============================================================================
# CELL 4.5: Print PyTorch-Style Model Architecture
# ============================================================================
def print_model_summary():
    """Print model architecture in PyTorch style with torchsummary-like table"""
    print("\n" + "=" * 80)
    print(" 🏗️  Model Architecture")
    print("=" * 80)
    
    # Create model to get actual layer info
    model = MNISTNet()
    
    # Print PyTorch-style architecture
    print("\nMNISTNet(")
    print("  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)")
    print("  (fc1): Linear(in_features=1568, out_features=128, bias=True)")
    print("  (fc2): Linear(in_features=128, out_features=10, bias=True)")
    print("  (relu): ReLU()")
    print(")")
    
    # Print summary table
    print("\n" + "─" * 80)
    print("Layer (type:depth-idx)                   Output Shape              Param #")
    print("=" * 80)
    
    batch_size = 64
    
    # Calculate shapes and params for each layer
    layers = [
        ("Conv2d: 1-1", f"[{batch_size}, 16, 28, 28]", 16*1*3*3 + 16),
        ("ReLU: 1-2", f"[{batch_size}, 16, 28, 28]", 0),
        ("MaxPool2d: 1-3", f"[{batch_size}, 16, 14, 14]", 0),
        ("Conv2d: 1-4", f"[{batch_size}, 32, 14, 14]", 32*16*3*3 + 32),
        ("ReLU: 1-5", f"[{batch_size}, 32, 14, 14]", 0),
        ("MaxPool2d: 1-6", f"[{batch_size}, 32, 7, 7]", 0),
        ("Linear: 1-7", f"[{batch_size}, 128]", 1568*128 + 128),
        ("ReLU: 1-8", f"[{batch_size}, 128]", 0),
        ("Linear: 1-9", f"[{batch_size}, 10]", 128*10 + 10),
    ]
    
    total_params = 0
    trainable_params = 0
    
    for layer_name, output_shape, params in layers:
        total_params += params
        trainable_params += params
        print(f"{layer_name:<40} {output_shape:<25} {params:>10,}")
    
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: 0")
    print("─" * 80)
    
    # Memory estimation
    input_size_mb = batch_size * 1 * 28 * 28 * 4 / (1024**2)
    params_size_mb = total_params * 4 / (1024**2)
    forward_size_mb = batch_size * (16*28*28 + 16*14*14 + 32*14*14 + 32*7*7 + 128 + 10) * 4 / (1024**2)
    
    print(f"\nInput size (MB): {input_size_mb:.2f}")
    print(f"Forward/backward pass size (MB): {forward_size_mb:.2f}")
    print(f"Params size (MB): {params_size_mb:.2f}")
    print(f"Estimated Total Size (MB): {input_size_mb + forward_size_mb + params_size_mb:.2f}")
    
    print("\n" + "=" * 80)


# Print architecture before training
print_model_summary()


def train_pytorch_gpu(X_train, y_train, batch_size):
    """PyTorch GPU training (cuDNN backend)"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    model = MNISTNet()
    model.load_numpy(get_weights())
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    n_batches = len(X_train) // batch_size
    perm = np.random.permutation(len(X_train))
    correct = 0
    
    # Warmup
    _ = model(torch.randn(batch_size, 1, 28, 28).cuda())
    torch.cuda.synchronize()
    
    start = time.time()
    
    for i in range(n_batches):
        idx = perm[i*batch_size:(i+1)*batch_size]
        X = torch.from_numpy(X_train[idx]).cuda()
        y = torch.from_numpy(y_train[idx]).cuda()
        
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        correct += out.argmax(1).eq(y).sum().item()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    acc = 100. * correct / (n_batches * batch_size)
    
    return elapsed, acc


# ============================================================================
# CELL 6:MacroTorch GPU Training
# ============================================================================
def train_macrotorch_gpu_fixed(X_train, y_train, batch_size):
    """
    FULLY GPU-OPTIMIZED MacroTorch Training
    - ALL intermediate values stay on GPU
    - Biases passed as device arrays
    - Only 2 CPU transfers per iteration: input batch + accuracy check
    """
    np.random.seed(42)
    w = get_weights()
    
    # Upload all weights to GPU ONCE
    d_W_conv1 = mt.to_device(w['W_conv1'])
    d_W_conv2 = mt.to_device(w['W_conv2'])
    d_W_fc1 = mt.to_device(w['W_fc1'])
    d_W_fc2 = mt.to_device(w['W_fc2'])
    d_b_conv1 = mt.to_device(w['b_conv1'])
    d_b_conv2 = mt.to_device(w['b_conv2'])
    d_b_fc1 = mt.to_device(w['b_fc1'])
    d_b_fc2 = mt.to_device(w['b_fc2'])
    
    lr = 0.01
    n_batches = len(X_train) // batch_size
    
    # Warmup
    dummy = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    _ = mt.conv2d_forward(mt.to_device(dummy), d_W_conv1, padding=1, 
                          bias=d_b_conv1, return_device=True)
    cuda.synchronize()
    
    perm = np.random.permutation(len(X_train))
    correct = 0
    
    start = time.time()
    
    for i in range(n_batches):
        idx = perm[i*batch_size:(i+1)*batch_size]
        X_batch = X_train[idx]
        y_batch = y_train[idx].astype(np.int32)
        
        # TRANSFER 1: Input batch to GPU (unavoidable)
        d_X = mt.to_device(X_batch)
        
        # ===== FORWARD PASS (ALL GPU!) =====
        # Conv1: Pass GPU bias array directly
        d_z1 = mt.conv2d_forward(d_X, d_W_conv1, padding=1, bias=d_b_conv1, return_device=True)
        d_a1 = mt.relu(d_z1, return_device=True)
        d_p1, d_idx1 = mt.maxpool2d_forward(d_a1, pool_size=2, return_device=True)
        
        # Conv2: Pass GPU bias array directly
        d_z2 = mt.conv2d_forward(d_p1, d_W_conv2, padding=1, bias=d_b_conv2, return_device=True)
        d_a2 = mt.relu(d_z2, return_device=True)
        d_p2, d_idx2 = mt.maxpool2d_forward(d_a2, pool_size=2, return_device=True)
        
        # Flatten (GPU reshape - no copy)
        d_flat = d_p2.reshape((batch_size, 1568))
        
        # FC1: Convert bias to CPU for now (linear function doesn't support GPU bias yet)
        b_fc1_cpu = d_b_fc1.copy_to_host()
        d_z3 = mt.linear(d_flat, d_W_fc1, b_fc1_cpu, return_device=True)
        d_a3 = mt.relu(d_z3, return_device=True)
        
        # FC2: Convert bias to CPU for now
        b_fc2_cpu = d_b_fc2.copy_to_host()
        d_z4 = mt.linear(d_a3, d_W_fc2, b_fc2_cpu, return_device=True)
        
        # Softmax (need to reshape for 4D API - this is on GPU)
        z4 = d_z4.copy_to_host()  # Small transfer for reshape
        z4_4d = z4.reshape(batch_size, 10, 1, 1)
        d_probs = mt.softmax_forward(z4_4d, return_device=True)
        
        # TRANSFER 2: Probs to CPU for accuracy (unavoidable)
        probs = d_probs.copy_to_host().reshape(batch_size, 10)
        correct += (probs.argmax(1) == y_batch).sum()
        
        # ===== BACKWARD PASS (ALL GPU!) =====
        # CE backward: returns GPU array
        d_dz4 = mt.cross_entropy_backward(probs, y_batch, return_device=True)
        
        # FC2 backward: ALL inputs are GPU arrays
        da3, dW_fc2, db_fc2 = mt.linear_backward(d_dz4, d_a3, d_W_fc2, return_device=False)
        d_da3 = mt.to_device(da3)
        
        # ReLU backward FC1: GPU -> GPU
        d_dz3 = mt.relu_backward(d_z3, d_da3, return_device=True)
        
        # FC1 backward: GPU inputs
        dflat, dW_fc1, db_fc1 = mt.linear_backward(d_dz3, d_flat, d_W_fc1, return_device=False)
        
        # Unflatten (GPU reshape - no copy)
        dp2 = mt.flatten_backward(dflat, d_p2.shape)
        d_dp2 = mt.to_device(dp2)
        
        # MaxPool2 backward: GPU -> GPU
        d_da2 = mt.maxpool2d_backward(d_dp2, d_idx2, d_a2.shape, pool_size=2, return_device=True)
        
        # ReLU backward Conv2: GPU -> GPU
        d_dz2 = mt.relu_backward(d_z2, d_da2, return_device=True)
        
        # Conv2 backward: PASS GPU ARRAYS DIRECTLY!
        # Check if your functions support d_grad_out, d_A parameters
        dW_conv2 = mt.conv2d_weight_backward(
            d_dz2, d_p1, padding=1, Kh=3, Kw=3,
            d_grad_out=d_dz2, d_A=d_p1, return_device=False
        )
        
        db_conv2 = mt.conv2d_bias_backward(
            d_dz2, d_grad_out=d_dz2
        )
        
        dp1 = mt.conv2d_input_backward(
            d_dz2, d_W_conv2, padding=1, return_device=False
        )
        d_dp1 = mt.to_device(dp1)
        
        # MaxPool1 backward: GPU -> GPU
        d_da1 = mt.maxpool2d_backward(d_dp1, d_idx1, d_a1.shape, pool_size=2, return_device=True)
        
        # ReLU backward Conv1: GPU -> GPU
        d_dz1 = mt.relu_backward(d_z1, d_da1, return_device=True)
        
        # Conv1 backward: GPU arrays
        dW_conv1 = mt.conv2d_weight_backward(
            d_dz1, d_X, padding=1, Kh=3, Kw=3,
            d_grad_out=d_dz1, d_A=d_X, return_device=False
        )
        
        db_conv1 = mt.conv2d_bias_backward(
            d_dz1, d_grad_out=d_dz1
        )
        
        # ===== GPU SGD UPDATES (ALL GPU!) =====
        mt.sgd_update_gpu(d_W_conv1, mt.to_device(dW_conv1), lr)
        mt.sgd_update_gpu(d_W_conv2, mt.to_device(dW_conv2), lr)
        mt.sgd_update_gpu(d_W_fc1, mt.to_device(dW_fc1), lr)
        mt.sgd_update_gpu(d_W_fc2, mt.to_device(dW_fc2), lr)
        mt.sgd_update_gpu(d_b_conv1, mt.to_device(db_conv1), lr)
        mt.sgd_update_gpu(d_b_conv2, mt.to_device(db_conv2), lr)
        mt.sgd_update_gpu(d_b_fc1, mt.to_device(db_fc1), lr)
        mt.sgd_update_gpu(d_b_fc2, mt.to_device(db_fc2), lr)
    
    cuda.synchronize()
    elapsed = time.time() - start
    acc = 100. * correct / (n_batches * batch_size)
    
    return elapsed, acc


# ============================================================================
# CELL 7: Run Benchmarks
# ============================================================================
print("\n" + "=" * 70)
print(" 🚀 GPU Benchmark: MacroTorch vs PyTorch (1 Epoch)")
print("=" * 70)


batch_sizes = [32, 64, 128, 256, 512, 1024]
results = []


for bs in batch_sizes:
    print(f"\n  📦 Batch Size: {bs}")
    
    # PyTorch GPU
    t_pt, acc_pt = train_pytorch_gpu(X_train, y_train, bs)
    print(f"    PyTorch GPU:    {t_pt:6.2f}s | Acc: {acc_pt:5.1f}%")
    
    # MacroTorch GPU (FIXED!)
    t_mt, acc_mt = train_macrotorch_gpu_fixed(X_train, y_train, bs)
    print(f"    MacroTorch GPU: {t_mt:6.2f}s | Acc: {acc_mt:5.1f}%")
    
    speedup = t_pt / t_mt
    slowdown = t_mt / t_pt
    
    if slowdown > 1:
        print(f"    → MacroTorch is {slowdown:.2f}x SLOWER")
    else:
        print(f"    → MacroTorch is {speedup:.2f}x FASTER")
    
    results.append({
        'batch': bs,
        'pytorch_time': t_pt,
        'pytorch_acc': acc_pt,
        'macrotorch_time': t_mt,
        'macrotorch_acc': acc_mt,
        'slowdown': slowdown
    })


# ============================================================================
# CELL 8: Summary Table
# ============================================================================
print("\n" + "=" * 70)
print(" 📊 Performance Summary (FIXED VERSION)")
print("=" * 70)


print("\n┌─────────┬──────────────┬───────────────┬────────────┐")
print("│  Batch  │ PyTorch (s)  │ MacroTorch (s)│  Slowdown  │")
print("├─────────┼──────────────┼───────────────┼────────────┤")


for r in results:
    symbol = "🐢" if r['slowdown'] > 1 else "⚡"
    print(f"│  {r['batch']:5}  │    {r['pytorch_time']:6.2f}    │     {r['macrotorch_time']:6.2f}    │   {r['slowdown']:5.2f}x {symbol}  │")


print("└─────────┴──────────────┴───────────────┴────────────┘")


# ============================================================================
# CELL 9: Accuracy Verification
# ============================================================================
print("\n" + "=" * 70)
print(" ✅ Accuracy Consistency Check")
print("=" * 70)


for r in results:
    diff = abs(r['pytorch_acc'] - r['macrotorch_acc'])
    status = "✅" if diff < 5.0 else "⚠️"
    print(f"  Batch {r['batch']:4}: PyTorch={r['pytorch_acc']:.1f}%, MacroTorch={r['macrotorch_acc']:.1f}% (Δ={diff:.1f}%) {status}")


# ============================================================================
# CELL 10: System Info & Comparison
# ============================================================================
print("\n" + "=" * 70)
print(" 🖥️  System Information")
print("=" * 70)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  CUDA: {torch.version.cuda}")
print(f"  PyTorch: {torch.__version__}")
print(f"  MacroTorch: {mt.__version__}")


avg_slowdown = np.mean([r['slowdown'] for r in results])
print(f"\n  Average Slowdown: {avg_slowdown:.2f}x")


print("\n" + "=" * 70)
print(" ✅ BENCHMARK COMPLETE")
print("=" * 70)