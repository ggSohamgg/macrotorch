# MacroTorch

A lightweight PyTorch-like deep learning library built from scratch with custom CUDA kernels.

## 📊 Benchmark & Accuracy Analysis

Performance comparison on a **512×512 Input Image** (except 3×3 on 256×256).

### Phase 1: FP32 Precision Benchmarks
| Kernel Size | Pure NumPy (CPU) | SciPy (CPU) | MacroTorch | PyTorch (GPU) | CUDA Speedup | Max Abs Error (vs SciPy) | Max Abs Error (vs PyTorch) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3×3** | 3.48 ms | 5.01 ms | **2.00 ms** | 0.06 ms | **1.74x** | `7.15e-07` | `0.00e+00` |
| **11×11** | 46.21 ms | 91.36 ms | **2.63 ms** | 0.25 ms | **17.59x** | `2.29e-05` | `0.00e+00` |
| **31×31** | 131.67 ms | 534.35 ms | **6.07 ms** | 1.51 ms | **21.69x** | `5.19e-04` | `0.00e+00` |
| **63×63** | 285.23 ms | 1933.72 ms | **14.94 ms** | 4.31 ms | **19.09x** | `4.46e-03` | `0.00e+00` |

### Phase 2: FP16 Precision Benchmarks
| Kernel Size | Pure NumPy (CPU) | SciPy (CPU) | MacroTorch | PyTorch (GPU) | CUDA Speedup | Max Abs Error (vs SciPy) | Max Abs Error (vs PyTorch) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3×3** | 2.20 ms | 3.07 ms | **1.26 ms** | 0.04 ms | **1.75x** | `0.00e+00` | `9.77e-04` |
| **11×11** | 45.45 ms | 87.74 ms | **2.21 ms** | 0.11 ms | **20.59x** | `2.29e-05` | `1.56e-02` |
| **31×31** | 126.54 ms | 626.36 ms | **6.29 ms** | 1.41 ms | **20.13x** | `5.19e-04` | `1.25e-01` |
| **63×63** | 566.94 ms | 2305.84 ms | **15.45 ms** | 3.97 ms | **36.71x** | `4.46e-03` | `5.00e-01` |

> **Note:** FP32 achieves exact match with PyTorch. FP16 shows expected deviation due to lower precision accumulation in large kernels.

### 🛠️ Key Features
- **Shared Memory Tiling:** Optimized kernels for different kernel sizes (Tiny to Large).
- **FP32 Accumulation:** FP16 kernels use FP32 accumulators to prevent overflow/underflow while saving 50% memory bandwidth.
- **Match PyTorch Accuracy:** Zero MAE (Mean Absolute Error) vs PyTorch in FP32 mode.
