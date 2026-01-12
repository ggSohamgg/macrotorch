import numpy as np
import pytest
from macrotorch import Conv2d

class TestLayers:
    
    def test_conv2d_init(self):
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        assert conv.weight.shape == (8, 3, 3, 3)
        assert conv.bias.shape == (8,)
        assert conv.grad_weight.shape == (8, 3, 3, 3)
        assert conv.grad_bias.shape == (8,)

    def test_conv2d_forward_shape(self):
        conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        x = np.random.randn(2, 3, 32, 32).astype(np.float32)
        out = conv.forward(x)
        assert out.shape == (2, 4, 32, 32)

    def test_conv2d_backward_gradients(self):
        conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        x = np.random.randn(2, 3, 16, 16).astype(np.float32)
        
        # Forward
        out = conv.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float32)
        
        # Backward
        grad_in = conv.backward(x, grad_out)
        
        # Check shapes
        assert grad_in.shape == (2, 3, 16, 16)
        assert conv.grad_weight.shape == (4, 3, 3, 3)
        assert conv.grad_bias.shape == (4,)
        
        # Check that gradients are not all zero
        assert np.any(conv.grad_weight != 0)
        assert np.any(conv.grad_bias != 0)
        assert np.any(grad_in != 0)

    def test_conv2d_no_bias(self):
        conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False)
        assert conv.bias is None
        assert conv.grad_bias is None
        
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        out = conv.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float32)
        conv.backward(x, grad_out)
        
        assert conv.grad_weight is not None
