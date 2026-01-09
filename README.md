# MacroTorch

Custom CUDA kernels for 2D convolution with PyTorch-style API. Built using NumPy for seamless data handling and Numba for optimized CUDA performance. Supports optimized forward/backward passes with shared memory tiling, FP16/FP32 support, and 5-100x CPU speedup.

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
├── tests/
│   └── test_conv.py     # Unit tests
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
from macrotorch import Conv2d

# Create input and kernel
img = np.random.randn(256, 256).astype(np.float32)
kernel = np.random.randn(5, 5).astype(np.float32)

# Forward pass
output = Conv2d.forward(img, kernel, padding=2, bias=0.1)

# Backward pass (input gradient)
grad_out = np.random.randn(*output.shape).astype(np.float32)
grad_input = Conv2d.input_backward(grad_out, kernel, padding=2)

# Backward pass (bias gradient) - requires 4D input (N, C, H, W)
grad_out_4d = np.random.randn(8, 64, 28, 28).astype(np.float32)
grad_bias = Conv2d.bias_backward(grad_out_4d)
```

### Using Conv2d Layer (OO Style)

```python
from macrotorch import Conv2d

# Create layer with learnable weights
conv = Conv2d(1, 1, kernel_size=5, padding=2, bias=True)

# Forward pass
x = np.random.randn(28, 28).astype(np.float32)
output = conv(x)

# Backward pass
grad_input = conv.backward(grad_out)

# Access weights
print(conv.weight.shape)  # (5, 5)
print(conv.bias.shape)    # (1,)
```

### Functional API

```python
from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward

# Forward
output = conv2d_forward(img, kernel, padding=2, bias=0.1)

# Backward
grad_input = conv2d_input_backward(grad_out, kernel, padding=2)
grad_bias = conv2d_bias_backward(grad_out_4d)
```

### FP16 Support

```python
img_fp16 = img.astype(np.float16)
kernel_fp16 = kernel.astype(np.float16)
output = conv2d_forward(img_fp16, kernel_fp16, padding=2)
```

## API Reference

### `conv2d_forward(A, K, padding=0, bias=None, dtype='auto', verbose=False)`
Performs 2D convolution.

**Parameters:**
- `A` (ndarray): Input image (H, W), float32 or float16
- `K` (ndarray): Kernel (Kh, Kw), same dtype as A
- `padding` (int): Padding pixels (default: 0)
- `bias` (float): Scalar bias to add (default: None → 0.0)
- `dtype` (str): 'fp32', 'fp16', or 'auto' (default: 'auto')
- `verbose` (bool): Print kernel selection info (default: False)

**Returns:** Output array (H_out, W_out) in float32

### `conv2d_input_backward(grad_out, K, padding=0, dtype='auto', verbose=False)`
Computes gradient w.r.t. input (∂L/∂A).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (H_out, W_out)
- `K` (ndarray): Kernel from forward pass (Kh, Kw)
- `padding` (int): Must match forward pass padding
- `dtype` (str): 'fp32', 'fp16', or 'auto'
- `verbose` (bool): Print execution info

**Returns:** Input gradient (H_in, W_in) in float32

### `conv2d_bias_backward(grad_out, dtype='auto', verbose=False)`
Computes gradient w.r.t. bias (∂L/∂b).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (N, C, H, W) - **4D batched**
- `dtype` (str): 'fp32', 'fp16', or 'auto'
- `verbose` (bool): Print execution info

**Returns:** Bias gradient (C,) in float32

### `Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=True, dtype='fp32')`
Layer class with learnable weights.

**Methods:**
- `forward(x)` / `__call__(x)`: Forward pass
- `backward(grad_out)`: Backward pass
- `parameters()`: Returns list of weights
- `zero_grad()`: Resets gradients

## 📊 Performance Benchmarks

- **Forward Pass:** Up to **188x faster** than SciPy (CPU)
- **Input Backward:** **11x faster** than SciPy (CPU)
- **Weight Backward:** **1.5-2.6x FASTER than PyTorch!**

**[View Detailed Benchmarks](BENCHMARKS.md)** - Full FP16/FP32 results on Tesla T4 GPU.

> **Hardware Note:** All benchmarks and tests were performed on an **NVIDIA Tesla T4 GPU**.

## Running Tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

## Running Benchmarks

```bash
pip install -e .[benchmark]
python examples/benchmark.py
```

## Key Features
- **NumPy-Native:** Built to work directly with NumPy arrays both as input and output.
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large)
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow
- **Full Backward Pass:** Input gradient and bias gradient computation
- **Layer API:** Object-oriented Conv2d with learnable weights
- **Functional API:** Direct function calls for full control
- **Optional PyTorch Benchmarking:** Compare against PyTorch (install with `[benchmark]`)
