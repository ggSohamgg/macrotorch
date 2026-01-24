from .ops import conv2d_forward
from .ops import input_backward as conv2d_input_backward
from .ops import bias_backward as conv2d_bias_backward
from .ops import weight_backward as conv2d_weight_backward
from .ops import relu
from .ops import relu_backward
from .ops import maxpool2d_forward
from .ops import maxpool2d_backward
from .ops import softmax_forward
from .ops import softmax_backward
from .ops import matmul
from .ops import linear
from .ops import linear_backward
from .ops import cross_entropy_loss
from .ops import cross_entropy_backward
from .ops import flatten
from .ops import flatten_backward
from .ops import SGD
from .ops import SGD_GPU
from .ops import sgd_update_gpu
from .ops import is_device_array
from .ops import to_device
from .layers import Conv2d

__version__ = "1.0.0"

__all__ = [
    'Conv2d',
    'conv2d_forward',
    'conv2d_input_backward',
    'conv2d_bias_backward',
    'conv2d_weight_backward',
    'relu',
    'relu_backward',
    'maxpool2d_forward',
    'maxpool2d_backward',
    'softmax_forward',
    'softmax_backward',
    'matmul',
    'linear',
    'linear_backward',
    'cross_entropy_loss',
    'cross_entropy_backward',
    'flatten',
    'flatten_backward',
    'SGD',
    'SGD_GPU',
    'sgd_update_gpu',
    'is_device_array',
    'to_device',
]

