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
    ├── benchmark.py              # Benchmark script
    └── train_mnist_comparison.py # MNIST training & comparison
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ggSohamgg/macrotorch.git
cd macrotorch
pip install -e .
```

### For Benchmarking (with PyTorch comparison)

To run the examples in `examples/`, you need `torch` and `scipy` installed.
You can install these dependencies using the `benchmark` extra:

```bash
pip install -e .[benchmark]
```

### Running Benchmarks

```bash
python examples/benchmark.py
```

### Running MNIST Training Comparison (MacroTorch vs PyTorch)

This script trains a CNN on MNIST using both frameworks and compares training time and accuracy.

```bash
python examples/train_mnist_comparison.py
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
## 📊 Performance Benchmarks (Tesla T4)

MacroTorch achieves competitive performance, with **custom kernels that beat PyTorch** in specific operations:

| Kernel | Configuration | vs PyTorch |
| :--- | :--- | :---: |
| **Weight Backward (2D Legacy)** | 8×128×128, 3×3, FP32 | **4.1x faster** |
| **Weight Backward (2D Legacy)** | 8×128×128, 3×3, FP16 | **3.5x faster** |
| **MaxPool2D Backward** | 8×64×128×128, pool=2 | **2.6x faster** |
| **Softmax Backward** | 8×10×1×1, FP32 | **2.7x faster** |
| **Cross-Entropy Backward** | 256×1000, FP32 | **2.2x faster** |
| **ReLU Backward** | 1024×1024, FP32 | **1.7x faster** |

### Additional Highlights
- **10-17x speedup** over CPU for forward/backward passes
- **31-66x speedup** over CPU for bias gradient computation
- **Better FP16 precision** than PyTorch in bias gradient (2.75e-04 vs 2.27e-01)

**[View Detailed Benchmarks](BENCHMARKS.md)**


## 📝 Acknowledgments

- **Core CUDA Kernels** (`macrotorch/kernels.py`): Handwritten by [@ggSohamgg](https://github.com/ggSohamgg) — the shared memory tiling, kernel optimizations, and all CUDA implementations are original work.
- **Benchmarks & Utilities** (`examples/benchmark.py`, `ops.py`, etc.): AI-assisted development with help from LLMs for boilerplate, documentation, and testing infrastructure.

*Credit where credit's due!* 🤝
