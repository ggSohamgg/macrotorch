# MacroTorch

Custom CUDA kernels for 4D multi-channel convolution with PyTorch-style API. Built using NumPy for seamless data handling and Numba for optimized CUDA performance. Supports optimized forward/backward passes with shared memory tiling, FP16/FP32 support, and competitive performance against PyTorch.

## Project Structure

```
macrotorch/
├── pyproject.toml       # pip install config
├── README.md
├── BENCHMARKS.md        # Detailed benchmarks
├── macrotorch/
│   ├── __init__.py      # Exports
│   ├── kernels.py       # CUDA kernel definitions
│   ├── ops.py           # Dispatch functions
│   └── layers.py        # Conv2d layer class
└── examples/
    └── benchmark.py     # Benchmark script
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ggSohamgg/macrotorch.git
cd macrotorch
pip install -e .
```

### For Benchmarking (with PyTorch comparison)

```bash
pip install -e .[benchmark]
```

### Basic Usage

```python
import numpy as np
from macrotorch import conv2d_forward

# Input: (Batch, Channels, Height, Width)
img = np.random.randn(8, 3, 224, 224).astype(np.float32)
# Weight: (Cout, Cin, Kh, Kw)
weight = np.random.randn(16, 3, 3, 3).astype(np.float32)
bias = np.random.randn(16).astype(np.float32)

# Forward pass
output = conv2d_forward(img, weight, padding=1, bias=bias)
print(output.shape) # (8, 16, 224, 224)
```

### Using Conv2d Layer (OO Style)

```python
from macrotorch import Conv2d

# Create layer with learnable weights
# Conv2d(in_channels, out_channels, kernel_size, padding, bias)
conv = Conv2d(3, 16, kernel_size=3, padding=1, bias=True)

# Forward pass
x = np.random.randn(8, 3, 224, 224).astype(np.float32)
output = conv(x)

# Backward pass
grad_out = np.random.randn(*output.shape).astype(np.float32)
grad_input = conv.backward(grad_out)

# Access weights and gradients
print(conv.weight.shape)      # (16, 3, 3, 3)
print(conv.grad_weight.shape) # (16, 3, 3, 3)
```

## API Reference

### `conv2d_forward(A, K, padding=0, bias=None, dtype='auto', verbose=False)`
Performs 4D batched multi-channel convolution.

**Parameters:**
- `A` (ndarray): Input tensor (N, Cin, H, W)
- `K` (ndarray): Kernel tensor (Cout, Cin, Kh, Kw)
- `padding` (int): Zero-padding (default: 0)
- `bias` (ndarray): Bias tensor (Cout,) (default: None → zeros)
- `dtype` (str): 'fp32', 'fp16', or 'auto'

**Returns:** Output array (N, Cout, H_out, W_out) in float32

### `conv2d_input_backward(grad_out, K, padding=0, dtype='auto', verbose=False)`
Computes gradient w.r.t. input (∂L/∂A).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (N, Cout, H_out, W_out)
- `K` (ndarray): Kernel from forward pass (Cout, Cin, Kh, Kw)

**Returns:** Input gradient (N, Cin, H_in, W_in) in float32

### `conv2d_weight_backward(grad_out, A, padding=0, dtype='auto', verbose=False)`
Computes gradient w.r.t. weights (∂L/∂W).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (N, Cout, H_out, W_out)
- `A` (ndarray): Original input (N, Cin, H_in, W_in)

**Returns:** Weight gradient (Cout, Cin, Kh, Kw) in float32

### `Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=True, dtype='fp32')`
Layer class with learnable weights. Handles storing gradients internally.

---

## 📊 Performance Benchmarks (Tesla T4)

MacroTorch achieves competitive performance, with **custom kernels that beat PyTorch** in specific operations:

| Kernel | Configuration | vs PyTorch |
| :--- | :--- | :---: |
| **Weight Backward (2D Legacy)** | 8×128×128, 3×3, FP32 | **🏆 5.3x faster** |
| **Weight Backward (2D Legacy)** | 8×128×128, 3×3, FP16 | **🏆 3.5x faster** |
| **ReLU Backward** | 1024×1024, FP32 | **🏆 1.8x faster** |

### Additional Highlights
- **10-17x speedup** over CPU for forward/backward passes
- **31-66x speedup** over CPU for bias gradient computation
- **Better FP16 precision** than PyTorch in bias gradient (2.75e-04 vs 2.27e-01)

**[View Detailed Benchmarks](BENCHMARKS.md)**

## Running Benchmarks

```bash
python examples/benchmark.py
```

## 📝 Acknowledgments

- **Core CUDA Kernels** (`macrotorch/kernels.py`): Handwritten by [@ggSohamgg](https://github.com/ggSohamgg) — the shared memory tiling, kernel optimizations, and all CUDA implementations are original work.
- **Benchmarks & Utilities** (`examples/benchmark.py`, `ops.py`, etc.): AI-assisted development with help from LLMs for boilerplate, documentation, and testing infrastructure.

*Credit where credit's due!* 🤝
