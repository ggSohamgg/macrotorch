# MacroTorch

A lightweight PyTorch-like deep learning library built from scratch with custom CUDA kernels.

## 📊 Benchmark & Accuracy Analysis

Performance comparison on a **512×512 Input Image** (except 3×3 on 256×256) across varying kernel sizes.
Error is calculated against **SciPy (Ground Truth)** and verified against **PyTorch**.

| Kernel Size | Pure NumPy (CPU) | SciPy (CPU) | Custom CUDA (GPU) | PyTorch (GPU) | CUDA Speedup (vs NumPy) | Max Abs Error (vs SciPy) | Verified vs PyTorch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3×3** | 3.64 ms | 3.02 ms | **1.20 ms** | 0.02 ms | **3.02x** | `7.15e-07` | ✅ Exact Match |
| **11×11** | 35.23 ms | 89.13 ms | **2.64 ms** | 0.24 ms | **13.33x** | `2.29e-05` | ✅ Exact Match |
| **31×31** | 127.01 ms | 543.55 ms | **5.78 ms** | 1.50 ms | **21.98x** | `5.19e-04` | ✅ Exact Match |
| **63×63** | 280.89 ms | 1908.45 ms | **14.98 ms** | 3.39 ms | **18.76x** | `4.46e-03` | ✅ Exact Match |

> **Note:** Error increases slightly with kernel size due to floating-point accumulation, but remains within acceptable GPU tolerance. Matches PyTorch output exactly.

### 🛠️ Key Features
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large).
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow/underflow while saving 50% memory bandwidth.
- **Match PyTorch Accuracy:** Zero MAE (Mean Absolute Error) vs PyTorch in FP32 mode.
