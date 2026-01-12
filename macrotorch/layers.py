import numpy as np
from .ops import forward, input_backward, bias_backward


class Conv2d:
    """
    2D Convolution Layer with learnable weights and bias (Multi-Channel Batched).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolution kernel (Kh, Kw).
    padding : int, optional (default=0)
        Padding applied to input.
    bias : bool, optional (default=True)
        If True, adds a learnable bias to the output.
    dtype : str, optional (default='fp32')
        Precision mode: 'fp32' or 'fp16'.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, dtype='fp32'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dtype = dtype
        self.use_bias = bias
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        np_dtype = np.float16 if dtype == 'fp16' else np.float32
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = (np.random.randn(out_channels, in_channels, *kernel_size) * scale).astype(np_dtype)
        
        if bias:
            self.bias = np.zeros(out_channels, dtype=np_dtype)
        else:
            self.bias = None
        
        # Initialize gradients to zeros of correct shape
        self.grad_weight = np.zeros(self.weight.shape, dtype=np.float32)
        if bias:
            self.grad_bias = np.zeros(out_channels, dtype=np.float32)
        else:
            self.grad_bias = None
            
        self._last_input = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """
        Forward pass of the convolution.
        
        Parameters
        ----------
        x : numpy.ndarray
            Input of shape (N, Cin, H, W).
        
        Returns
        -------
        numpy.ndarray
            Output of shape (N, Cout, H_out, W_out).
        """
        self._last_input = x
        return forward(x, self.weight, padding=self.padding, bias=self.bias, dtype=self.dtype)
    
    def backward(self, grad_out, x=None):
        """
        Backward pass of the convolution.
        
        Parameters
        ----------
        grad_out : numpy.ndarray
            Gradient from next layer of shape (N, Cout, H_out, W_out).
        
        Returns
        -------
        numpy.ndarray
            Gradient with respect to input of shape (N, Cin, H_in, W_in).
        """
        grad_input = input_backward(grad_out, self.weight, padding=self.padding, dtype=self.dtype)
        
        # Compute gradients for parameters
        input_for_grad = x if x is not None else self._last_input
        self.grad_weight = weight_backward(grad_out, input_for_grad, padding=self.padding, dtype=self.dtype)
        if self.use_bias:
            self.grad_bias = bias_backward(grad_out, dtype=self.dtype)
            
        return grad_input
    
    def parameters(self):
        """Returns list of learnable parameters."""
        if self.use_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def zero_grad(self):
        """Resets gradients to zeros."""
        if self.grad_weight is not None:
            self.grad_weight.fill(0.0)
        if self.grad_bias is not None:
            self.grad_bias.fill(0.0)
    
    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, padding={self.padding}, bias={self.use_bias})")
