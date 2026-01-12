import numpy as np
from .ops import forward, input_backward, bias_backward


class Conv2d:
    """
    2D Convolution Layer with learnable weights and bias.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (currently supports 1).
    out_channels : int
        Number of output channels (currently supports 1).
    kernel_size : int or tuple
        Size of the convolution kernel (Kh, Kw).
    padding : int, optional (default=0)
        Padding applied to input.
    bias : bool, optional (default=True)
        If True, adds a learnable bias to the output.
    dtype : str, optional (default='fp32')
        Precision mode: 'fp32' or 'fp16'.
    
    Attributes
    ----------
    weight : numpy.ndarray
        Learnable kernel weights of shape (Kh, Kw).
    bias : numpy.ndarray or None
        Learnable bias of shape (1,) or None if bias=False.
    
    Examples
    --------
    >>> import numpy as np
    >>> from macrotorch import Conv2d
    >>> 
    >>> conv = Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
    >>> x = np.random.randn(28, 28).astype(np.float32)
    >>> output = conv(x)
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
        self.weight = (np.random.randn(*kernel_size) * scale).astype(np_dtype)
        
        if bias:
            self.bias = np.zeros(out_channels, dtype=np_dtype)
        else:
            self.bias = None
        
        self.grad_weight = None
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
            Input of shape (H, W) or (N, H, W) for batched input.
        
        Returns
        -------
        numpy.ndarray
            Output of shape (H_out, W_out) or (N, H_out, W_out).
        """
        self._last_input = x
        bias_val = float(self.bias[0]) if self.use_bias else None
        return forward(x, self.weight, padding=self.padding, bias=bias_val, dtype=self.dtype)
    
    def backward(self, grad_out):
        """
        Backward pass of the convolution.
        
        Parameters
        ----------
        grad_out : numpy.ndarray
            Gradient from next layer of shape (H_out, W_out) or (N, H_out, W_out).
        
        Returns
        -------
        numpy.ndarray
            Gradient with respect to input of shape (H_in, W_in) or (N, H_in, W_in).
        """
        grad_input = input_backward(grad_out, self.weight, padding=self.padding, dtype=self.dtype)
        return grad_input
    
    def parameters(self):
        """Returns list of learnable parameters."""
        if self.use_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def zero_grad(self):
        """Resets gradients to None."""
        self.grad_weight = None
        self.grad_bias = None
    
    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, padding={self.padding}, bias={self.use_bias})")
