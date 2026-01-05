# MacroTorch

**MacroTorch** is a high-performance deep learning library built from scratch using **Numba CUDA**. It features low-level kernel optimizations like shared memory tiling and coalesced memory access to achieve significant speedups over CPU-based implementations.

## Conv2D Forward Benchmarks
*NVIDIA T4 GPU vs SciPy (CPU) vs PyTorch (GPU)*

| Kernel Size | Image Size | SciPy (ms) | MacroTorch (ms) | Speedup vs CPU | PyTorch (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3×3 (Tiny)** | 256×256 | 6.29 | 1.57 | **3.99x** | 0.07 |
| **11×11 (Small)** | 512×512 | 93.56 | 2.55 | **36.71x** | 0.25 |
| **31×31 (Medium)** | 512×512 | 589.36 | 5.67 | **104.01x** | 1.51 |
| **63×63 (Large)** | 512×512 | 2201.27 | 14.75 | **149.28x** | 2.89 |

### 🛠️ Key Features
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large).
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow/underflow while saving 50% memory bandwidth.
- **Match PyTorch Accuracy:** Zero MAE (Mean Absolute Error) vs PyTorch in FP32 mode.


