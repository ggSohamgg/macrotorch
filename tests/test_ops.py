import numpy as np
import pytest
import math
from numba import cuda

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from macrotorch import conv2d_forward, conv2d_input_backward, conv2d_bias_backward, conv2d_weight_backward, relu, relu_backward

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available for verification")
class TestFunctionalOps:
    
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("padding", [0, 1])
    def test_conv2d_forward(self, dtype, padding):
        N, C, H, W = 2, 3, 32, 32
        Cout = 4
        Kh, Kw = 3, 3
        
        A = np.random.randn(N, C, H, W).astype(dtype)
        K = np.random.randn(Cout, C, Kh, Kw).astype(dtype)
        bias = np.random.randn(Cout).astype(dtype)
        
        # MacroTorch
        mt_out = conv2d_forward(A, K, padding=padding, bias=bias)
        
        # PyTorch
        t_A = torch.from_numpy(A).cuda()
        t_K = torch.from_numpy(K).cuda()
        t_bias = torch.from_numpy(bias).cuda()
        
        pt_out = torch.nn.functional.conv2d(t_A, t_K, t_bias, padding=padding)
        pt_out_np = pt_out.cpu().numpy()
        
        tolerance = 1e-3 if dtype == np.float16 else 1e-5
        np.testing.assert_allclose(mt_out, pt_out_np, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_conv2d_input_backward(self, dtype):
        N, C, H, W = 2, 3, 32, 32
        Cout = 4
        Kh, Kw = 3, 3
        padding = 1
        
        A = np.random.randn(N, C, H, W).astype(dtype)
        K = np.random.randn(Cout, C, Kh, Kw).astype(dtype)
        
        mt_out = conv2d_forward(A, K, padding=padding)
        grad_out = np.random.randn(*mt_out.shape).astype(dtype)
        
        # MacroTorch
        mt_grad_in = conv2d_input_backward(grad_out, K, padding=padding)
        
        # PyTorch
        t_grad = torch.from_numpy(grad_out).cuda()
        t_K = torch.from_numpy(K).cuda()
        
        pt_grad_in = torch.nn.grad.conv2d_input((N, C, H, W), t_K, t_grad, padding=padding)
        pt_grad_in_np = pt_grad_in.cpu().numpy()
        
        tolerance = 1e-2 if dtype == np.float16 else 1e-5
        np.testing.assert_allclose(mt_grad_in, pt_grad_in_np, atol=tolerance, rtol=tolerance)

    @pytest.mark.parametrize("dtype", [np.float32])
    def test_conv2d_weight_backward(self, dtype):
        N, C, H, W = 2, 3, 16, 16
        Cout = 4
        Kh, Kw = 3, 3
        padding = 1
        
        A = np.random.randn(N, C, H, W).astype(dtype)
        
        H_out = H - Kh + 1 + 2*padding
        W_out = W - Kw + 1 + 2*padding
        grad_out = np.random.randn(N, Cout, H_out, W_out).astype(dtype)
        
        # MacroTorch
        mt_grad_W = conv2d_weight_backward(grad_out, A, padding=padding)
        
        # PyTorch
        t_A = torch.from_numpy(A).cuda()
        t_grad = torch.from_numpy(grad_out).cuda()
        
        pt_grad_W = torch.nn.grad.conv2d_weight(t_A, (Cout, C, Kh, Kw), t_grad, padding=padding)
        pt_grad_W_np = pt_grad_W.cpu().numpy()
        
        tolerance = 1e-4
        np.testing.assert_allclose(mt_grad_W, pt_grad_W_np, atol=tolerance, rtol=tolerance)

    def test_conv2d_bias_backward(self):
        N, C, H, W = 4, 8, 16, 16
        grad_out = np.random.randn(N, C, H, W).astype(np.float32)
        
        mt_grad_bias = conv2d_bias_backward(grad_out)
        
        t_grad = torch.from_numpy(grad_out).cuda()
        pt_grad_bias = t_grad.sum(dim=(0, 2, 3))
        pt_grad_bias_np = pt_grad_bias.cpu().numpy()
        
        np.testing.assert_allclose(mt_grad_bias, pt_grad_bias_np, atol=1e-4)

    def test_relu_ops(self):
        x = np.random.randn(2, 4, 8, 8).astype(np.float32)
        mt_out = relu(x)
        np.testing.assert_allclose(mt_out, np.maximum(0, x))
        
        grad_out = np.random.randn(2, 4, 8, 8).astype(np.float32)
        mt_grad_in = relu_backward(x, grad_out)
        np.testing.assert_allclose(mt_grad_in, grad_out * (x > 0))
