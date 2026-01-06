# MacroTorch

Custom CUDA kernels for 2D convolution with PyTorch-style API. Optimized forward/backward passes with shared memory tiling. FP16/FP32 support, 5-100x CPU speedup.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/ggSohamgg/macrotorch.git
cd macrotorch
pip install -e .
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

### Alternative Import Style

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
# Convert to FP16
img_fp16 = img.astype(np.float16)
kernel_fp16 = kernel.astype(np.float16)

# Automatic dtype detection
output = Conv2d.forward(img_fp16, kernel_fp16, padding=2)
```

## 📖 API Reference

### `Conv2d.forward(A, K, padding=0, bias=None, dtype='auto', verbose=False)`
Performs 2D convolution.

**Parameters:**
- `A` (ndarray): Input image (H, W), float32 or float16
- `K` (ndarray): Kernel (Kh, Kw), same dtype as A
- `padding` (int): Padding pixels (default: 0)
- `bias` (float): Scalar bias to add (default: None)
- `dtype` (str): 'fp32', 'fp16', or 'auto' (default: 'auto')
- `verbose` (bool): Print kernel selection info (default: False)

**Returns:** Output array (H_out, W_out) in float32

### `Conv2d.input_backward(grad_out, K, padding=0, dtype='auto', verbose=False)`
Computes gradient w.r.t. input (∂L/∂A).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (H_out, W_out)
- `K` (ndarray): Kernel from forward pass (Kh, Kw)
- `padding` (int): Must match forward pass padding
- `dtype` (str): 'fp32', 'fp16', or 'auto'
- `verbose` (bool): Print execution info

**Returns:** Input gradient (H_in, W_in) in float32

### `Conv2d.bias_backward(grad_out, dtype='auto', verbose=False)`
Computes gradient w.r.t. bias (∂L/∂b).

**Parameters:**
- `grad_out` (ndarray): Gradient from next layer (N, C, H, W) - **4D batched**
- `dtype` (str): 'fp32', 'fp16', or 'auto'
- `verbose` (bool): Print execution info

**Returns:** Bias gradient (C,) in float32

## 📊 Performance Benchmarks

### Forward Pass - FP32 (512×512 Input)

| Kernel Size | SciPy (CPU) | MacroTorch (GPU) | PyTorch (GPU) | Speedup vs SciPy |
| :---: | :---: | :---: | :---: | :---: |
| **3×3** | 2.96 ms | **1.54 ms** | 0.04 ms | **1.92x** |
| **11×11** | 86.78 ms | **2.55 ms** | 0.24 ms | **34.03x** |
| **31×31** | 536.36 ms | **5.59 ms** | 1.50 ms | **95.95x** |
| **63×63** | 2769.66 ms | **14.66 ms** | 3.01 ms | **188.91x** |

### Input Gradient Backward - FP32 (512×512, 5×5 Kernel)

| Implementation | Time | Speedup (vs SciPy) | Max Error (vs SciPy) |
| :---: | :---: | :---: | :---: |
| SciPy (CPU) | 23.03 ms | 1.00x | Ground Truth |
| PyTorch (GPU) | 0.44 ms | **51.96x** | `3.08e+01` |
| **MacroTorch (GPU)** | **1.99 ms** | **11.60x** | `3.08e+01` |

> **Note:** MacroTorch matches PyTorch accuracy exactly (0.00e+00 error in FP32).

📈 **[View Detailed Benchmarks](BENCHMARKS.md)** - Includes FP16 results and bias gradient benchmarks.

## 🛠️ Key Features
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large).
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow/underflow.
- **Match PyTorch Accuracy:** Zero MAE vs PyTorch in FP32 mode.
- **Full Backward Pass:** Input gradient and bias gradient computation.
- **PyTorch-style API:** Clean interface with `Conv2d.forward()`, `Conv2d.input_backward()`, `Conv2d.bias_backward()`.
