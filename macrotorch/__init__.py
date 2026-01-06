from .backend.dispatch import forward as conv2d_forward
from .backend.dispatch import input_backward as conv2d_input_backward
from .backend.dispatch import bias_backward as conv2d_bias_backward


class Conv2d:
    """
    2D Convolution Operations (PyTorch-style API)
    
    Usage:
        from macrotorch import Conv2d
        
        # Forward
        output = Conv2d.forward(input, kernel, padding=2, bias=0.1)
        
        # Backward
        grad_input = Conv2d.input_backward(grad_out, kernel, padding=2)
        grad_bias = Conv2d.bias_backward(grad_out)
    """
    forward = staticmethod(conv2d_forward)
    input_backward = staticmethod(conv2d_input_backward)
    bias_backward = staticmethod(conv2d_bias_backward)


# Also keep individual imports for flexibility
__all__ = [
    'Conv2d',
    'conv2d_forward',
    'conv2d_input_backward', 
    'conv2d_bias_backward'
]
