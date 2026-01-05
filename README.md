# MacroTorch

A lightweight PyTorch-like deep learning library built from scratch with custom CUDA kernels.

## 📊 Benchmark & Accuracy Analysis

Performance comparison on a **512×512 Input Image** (except 3×3 on 256×256).

### Phase 1: FP32 Precision Benchmarks
| Kernel Size | Pure NumPy (CPU) | SciPy (CPU) | MacroTorch | PyTorch (GPU) | CUDA Speedup | Max Abs Error (vs SciPy) | Max Abs Error (SciPy vs PyTorch) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3×3** | 3.39 ms | 2.96 ms | **1.54 ms** | 0.04 ms | **2.20x** | `7.15e-07` | `7.15e-07` |
| **11×11** | 55.42 ms | 86.78 ms | **2.55 ms** | 0.24 ms | **21.77x** | `2.29e-05` | `2.29e-05` |
| **31×31** | 137.17 ms | 536.36 ms | **5.59 ms** | 1.50 ms | **24.53x** | `5.19e-04` | `5.19e-04` |
| **63×63** | 280.49 ms | 2769.66 ms | **14.66 ms** | 3.01 ms | **19.14x** | `4.46e-03` | `4.46e-03` |

### Phase 2: FP16 Precision Benchmarks
| Kernel Size | Pure NumPy (CPU) | SciPy (CPU) | MacroTorch | PyTorch (GPU) | CUDA Speedup | Max Abs Error (vs SciPy) | Max Abs Error (SciPy vs PyTorch) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3×3** | 2.15 ms | 3.02 ms | **1.12 ms** | 0.02 ms | **1.91x** | `0.00e+00` | `9.77e-04` |
| **11×11** | 35.77 ms | 86.57 ms | **2.16 ms** | 0.15 ms | **16.56x** | `2.29e-05` | `1.56e-02` |
| **31×31** | 139.17 ms | 546.75 ms | **4.75 ms** | 1.41 ms | **29.28x** | `5.19e-04` | `1.25e-01` |
| **63×63** | 316.62 ms | 1876.52 ms | **15.06 ms** | 2.74 ms | **21.03x** | `4.46e-03` | `5.01e-01` |

> **Note:** FP32 achieves exact match with PyTorch. FP16 shows expected deviation due to lower precision accumulation in large kernels.

### Phase 3: Backward Pass (Input Gradient) Benchmark

Performance comparison for `conv2d_backward` on **1024×1024 Input** with **5×5 Kernel** (Output: 1020×1020).

| Implementation | Time | Max Abs Error (vs SciPy) |
| :---: | :---: | :---: |
| **SciPy (CPU)** | 154.51 ms | Ground Truth |
| **PyTorch (GPU)** | 0.23 ms | `3.81e-06` |
| **MacroTorch (Global)** | 146.76 ms | `3.81e-06` |
| **MacroTorch (Shared)** | 0.36 ms | `3.81e-06` |

> **Note:** Shared memory implementation achieves near-PyTorch performance with identical accuracy.

### 🛠️ Key Features
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large).
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow/underflow while saving 50% memory bandwidth.
- **Match PyTorch Accuracy:** Zero MAE (Mean Absolute Error) vs PyTorch in FP32 mode.
- **Backward Pass Support:** Full gradient computation for input with both shared and global memory implementations.
