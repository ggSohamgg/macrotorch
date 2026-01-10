"""
Unit tests for MacroTorch convolution operations.

Run with: pytest tests/test_conv.py -v
Requires: pip install macrotorch[dev]
"""

import numpy as np
import pytest
from scipy.signal import correlate2d


def scipy_conv2d(A, K, padding=0):
    """Ground truth convolution using SciPy."""
    if padding > 0:
        A_padded = np.pad(A, padding, mode='constant', constant_values=0)
    else:
        A_padded = A
    return correlate2d(A_padded, K, mode='valid')


class TestConv2dForwardCPU:
    """CPU-based correctness tests (validation only, uses SciPy as ground truth)."""
    
    def test_forward_basic(self):
        """Test basic convolution matches SciPy."""
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(3, 3).astype(np.float32)
        
        expected = scipy_conv2d(A, K, padding=0)
        assert expected.shape == (62, 62)
    
    def test_forward_with_padding(self):
        """Test padded convolution matches expected output size."""
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(3, 3).astype(np.float32)
        
        expected = scipy_conv2d(A, K, padding=1)
        assert expected.shape == (64, 64)
    
    def test_output_shape_formula(self):
        """Test output shape formula: out = (in + 2*pad - kernel) + 1."""
        H, W = 100, 100
        Kh, Kw = 5, 5
        padding = 2
        
        out_h = H - Kh + 1 + (2 * padding)
        out_w = W - Kw + 1 + (2 * padding)
        
        assert out_h == 100
        assert out_w == 100


@pytest.mark.gpu
class TestConv2dForwardGPU:
    """GPU-based tests (requires CUDA)."""
    
    def test_forward_small_kernel(self):
        """Test small kernel convolution on GPU."""
        from macrotorch import conv2d_forward
        
        np.random.seed(42)
        A = np.random.randn(256, 256).astype(np.float32)
        K = np.random.randn(3, 3).astype(np.float32)
        
        output = conv2d_forward(A, K, padding=0)
        expected = scipy_conv2d(A, K, padding=0)
        
        assert output.shape == expected.shape
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_forward_medium_kernel(self):
        """Test medium kernel convolution on GPU."""
        from macrotorch import conv2d_forward
        
        np.random.seed(42)
        A = np.random.randn(256, 256).astype(np.float32)
        K = np.random.randn(11, 11).astype(np.float32)
        
        output = conv2d_forward(A, K, padding=5)
        expected = scipy_conv2d(A, K, padding=5)
        
        assert output.shape == expected.shape
        np.testing.assert_allclose(output, expected, rtol=1e-3, atol=1e-3)
    
    def test_forward_with_bias(self):
        """Test convolution with bias addition."""
        from macrotorch import conv2d_forward
        
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(3, 3).astype(np.float32)
        bias = 0.5
        
        output = conv2d_forward(A, K, padding=1, bias=bias)
        expected = scipy_conv2d(A, K, padding=1) + bias
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_forward_fp16(self):
        """Test FP16 convolution on GPU."""
        from macrotorch import conv2d_forward
        
        np.random.seed(42)
        A = np.random.randn(256, 256).astype(np.float16)
        K = np.random.randn(3, 3).astype(np.float16)
        
        output = conv2d_forward(A, K, padding=0)
        expected = scipy_conv2d(A.astype(np.float32), K.astype(np.float32), padding=0)
        
        assert output.shape == expected.shape
        np.testing.assert_allclose(output, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
class TestConv2dBackwardGPU:
    """GPU-based backward pass tests."""
    
    def test_input_backward_shape(self):
        """Test input gradient has correct shape."""
        from macrotorch import conv2d_forward, conv2d_input_backward
        
        np.random.seed(42)
        A = np.random.randn(64, 64).astype(np.float32)
        K = np.random.randn(5, 5).astype(np.float32)
        padding = 2
        
        output = conv2d_forward(A, K, padding=padding)
        grad_out = np.ones_like(output)
        grad_input = conv2d_input_backward(grad_out, K, padding=padding)
        
        assert grad_input.shape == A.shape
    
    def test_bias_backward_shape(self):
        """Test bias gradient has correct shape."""
        from macrotorch import conv2d_bias_backward
        
        np.random.seed(42)
        grad_out = np.random.randn(8, 64, 28, 28).astype(np.float32)
        
        grad_bias = conv2d_bias_backward(grad_out)
        
        assert grad_bias.shape == (64,)
    
    def test_bias_backward_value(self):
        """Test bias gradient equals sum over N, H, W dimensions."""
        from macrotorch import conv2d_bias_backward
        
        np.random.seed(42)
        grad_out = np.random.randn(4, 16, 8, 8).astype(np.float32)
        
        grad_bias = conv2d_bias_backward(grad_out)
        expected = np.sum(grad_out, axis=(0, 2, 3))
        
        np.testing.assert_allclose(grad_bias, expected, rtol=1e-3, atol=1e-3)
    
    def test_weight_backward_shape(self):
        """Test weight gradient has correct shape."""
        from macrotorch import conv2d_weight_backward
        
        np.random.seed(42)
        N, H, W = 8, 32, 32
        Kh, Kw = 5, 5
        padding = 2
        H_out = H - Kh + 1 + 2 * padding
        W_out = W - Kw + 1 + 2 * padding
        
        A = np.random.randn(N, H, W).astype(np.float32)
        grad_out = np.random.randn(N, H_out, W_out).astype(np.float32)
        
        grad_W = conv2d_weight_backward(grad_out, A, padding=padding)
        
        assert grad_W.shape == (Kh, Kw)
    
    def test_weight_backward_value(self):
        """Test weight gradient matches NumPy ground truth."""
        from macrotorch import conv2d_weight_backward
        
        np.random.seed(42)
        N, H, W = 4, 16, 16
        Kh, Kw = 3, 3
        padding = 1
        H_out = H - Kh + 1 + 2 * padding
        W_out = W - Kw + 1 + 2 * padding
        
        A = np.random.randn(N, H, W).astype(np.float32)
        grad_out = np.random.randn(N, H_out, W_out).astype(np.float32)
        
        # MacroTorch result
        grad_W = conv2d_weight_backward(grad_out, A, padding=padding)
        
        # NumPy ground truth
        expected = np.zeros((Kh, Kw), dtype=np.float32)
        for u in range(Kh):
            for v in range(Kw):
                for n in range(N):
                    for i in range(H_out):
                        for j in range(W_out):
                            in_row = i - padding + u
                            in_col = j - padding + v
                            if 0 <= in_row < H and 0 <= in_col < W:
                                expected[u, v] += grad_out[n, i, j] * A[n, in_row, in_col]
        
        np.testing.assert_allclose(grad_W, expected, rtol=1e-3, atol=1e-3)
    
    def test_weight_backward_2d_input(self):
        """Test weight backward with 2D input (auto-reshape to 3D)."""
        from macrotorch import conv2d_weight_backward
        
        np.random.seed(42)
        H, W = 16, 16
        Kh, Kw = 3, 3
        padding = 1
        
        A = np.random.randn(H, W).astype(np.float32)
        grad_out = np.random.randn(H, W).astype(np.float32)
        
        grad_W = conv2d_weight_backward(grad_out, A, padding=padding)
        
        assert grad_W.shape == (Kh, Kw)


@pytest.mark.gpu
class TestConv2dLayer:
    """Tests for Conv2d layer class."""
    
    def test_layer_init(self):
        """Test layer initialization."""
        from macrotorch import Conv2d
        
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        
        assert conv.weight.shape == (3, 3)
        assert conv.bias.shape == (1,)
        assert conv.padding == 1
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        from macrotorch import Conv2d
        
        np.random.seed(42)
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        x = np.random.randn(28, 28).astype(np.float32)
        
        output = conv(x)
        
        assert output.shape == (28, 28)
    
    def test_layer_backward(self):
        """Test layer backward pass."""
        from macrotorch import Conv2d
        
        np.random.seed(42)
        conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        x = np.random.randn(28, 28).astype(np.float32)
        
        output = conv(x)
        grad_out = np.ones_like(output)
        grad_input = conv.backward(grad_out)
        
        assert grad_input.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
