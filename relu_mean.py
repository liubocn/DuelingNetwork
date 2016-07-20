"""
Defalt is Breakout
If you want to implement other game, change the number 4 placed before 'num_of_Actions'
"""

import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU_MEAN(function.Function):

    """after Rectified Linear Unit, minus mean."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.cpu_y = utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype))
        self.cpu_y -= self.cpu_y.sum(axis=1, keepdims=True) / 4. # num_of_Actions
        return self.cpu_y,

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gpu_y = cudnn.activation_forward(x[0], _mode)
            gpu_y -= gpu_y.sum(axis=1, keepdims=True) / 4. # num_of_Actions
            self.y = gpu_y
        else:
            gpu_y = cuda.cupy.maximum(x[0], 0)
            gpu_y_sum = gpu_y.sum(axis=1, keepdims=True) / 4. # num_of_Actions
            gpu_y -= gpu_y_sum
        return gpu_y,

    def backward_cpu(self, x, gy):
        cpu_gy = utils.force_array(gy[0] * (x[0] > 0) * (1 - 1 / 4.)) # 4 is num_of_Actions
        return cpu_gy,

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gx = cudnn.activation_backward(x[0], self.y, gy[0], _mode)
            gx = gx * (1 - 1 / 4. ) # 4 is num_of_Actions
        else:
            gx = cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = x > 0 ? gy * (1 - 1 / 4.) : (T)0',
                'relu_mean_bwd')(x[0], gy[0]) # 4 is num_of_Actions
        return gx,


def relu_Mean(x, use_cudnn=True):
    """after Rectified Linear Unit function :math:`f(x)=\\max(0, x)`, minus mean.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ReLU_MEAN(use_cudnn)(x)
