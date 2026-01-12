# MacroTorch Benchmarks

Comprehensive performance benchmarks for MacroTorch custom CUDA kernels.

All benchmarks use **torch.cuda.Event** for precise GPU kernel timing.
All tests and benchmarks were performed on an **NVIDIA Tesla T4 GPU**.

---

## Forward Pass

**Configuration**: 512×512 Input, 5×5 Kernel, Padding=2

| Precision | SciPy (CPU) | PyTorch (GPU) | MacroTorch (GPU) | MT vs CPU |
| :---: | :---: | :---: | :---: | :---: |
| **FP32** | 26.24 ms | 0.09 ms | 2.23 ms | **11.8x faster** |
| **FP16** | 27.38 ms | 0.09 ms | 2.76 ms | **9.9x faster** |

> MacroTorch achieves exact accuracy match with PyTorch (3.81e-06 error).

---

## Input Gradient Backward Pass

**Configuration**: 512×512 Input, 5×5 Kernel, Padding=2

| Precision | SciPy (CPU) | PyTorch (GPU) | MacroTorch (GPU) | MT vs CPU |
| :---: | :---: | :---: | :---: | :---: |
| **FP32** | 50.24 ms | 0.13 ms | 2.67 ms | **18.8x faster** |
| **FP16** | 51.15 ms | 0.12 ms | 2.22 ms | **23.1x faster** |

> MacroTorch matches PyTorch accuracy exactly (5.72e-06 error in FP32).

---

## Bias Gradient Backward Pass

**Configuration**: Batch=32, Channels=128, Spatial=64×64

| Precision | NumPy (CPU) | PyTorch (GPU) | MacroTorch (GPU) | MT vs CPU |
| :---: | :---: | :---: | :---: | :---: |
| **FP32** | 30.38 ms | 0.34 ms | 0.97 ms | **31.4x faster** |
| **FP16** | 65.00 ms | 0.31 ms | 0.95 ms | **68.1x faster** |

> MacroTorch achieves better FP16 accuracy (3.05e-04) than PyTorch (2.27e-01).

---

## Weight Gradient Backward Pass

**torch.cuda.Event** profiling for precise GPU kernel timing.

| Config | Precision | PyTorch (ms) | MacroTorch (ms) | Speedup | Max Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** (8×64×64) | FP32 | 0.27 | **0.12** | **2.28x faster** | `1.02e-03` |
| **Small** (8×64×64) | FP16 | 0.27 | **0.17** | **1.64x faster** | `9.31e-04` |
| **Large** (128×256×256) | FP32 | 11.82 | **10.12** | **1.17x faster** | `1.81e-02` |
| **Large** (128×256×256) | FP16 | 11.61 | **4.13** | **2.81x faster** | `1.51e-02` |

> **MacroTorch is 1.2-2.8x FASTER than PyTorch** for weight gradient computation!

---

## ReLU Activation

**Configuration**: 1024×1024 array, FP32

| Pass | NumPy (CPU) | PyTorch (GPU) | MacroTorch (GPU) | MT vs PT | Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward** | 2.40 ms | 0.05 ms | 0.15 ms | 2.70x slower | `0.00e+00` |
| **Backward** | 1.83 ms | 0.21 ms | **0.13 ms** | **1.70x faster** | `0.00e+00` |

> MacroTorch ReLU backward is **1.7x faster than PyTorch**, while forward is optimized for simplicity. Both achieve exact accuracy.

---

## MaxPool2D

**Configuration**: 512×512 input, pool_size=2, FP32

| Operation | NumPy (CPU) | PyTorch (GPU) | MacroTorch (GPU) | MT vs PT | Error |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Forward** | 151.53 ms | 0.03 ms | 0.08 ms | 3.14x slower | `0.00e+00` |

> MacroTorch achieves **1811x speedup** over NumPy CPU with exact accuracy.

---

## Summary

MacroTorch demonstrates:
- ✅ **10-23x speedup** over CPU for forward/backward passes
- ✅ **31-68x speedup** over CPU for bias gradient computation
- ✅ **1.2-2.8x faster than PyTorch** for weight gradient backward
- ✅ **1.7x faster than PyTorch** for ReLU backward
- ✅ **Excellent accuracy** with max error ~1e-02 to 1e-04
- ✅ **Better FP16 precision** than PyTorch in some operations

### Note on PyTorch Comparison
PyTorch uses highly optimized cuDNN for forward and input backward passes, which is ~20-30x faster than MacroTorch for those operations. However, MacroTorch's custom kernels **beat PyTorch** for weight gradient computation and ReLU backward.
