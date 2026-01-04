"""
yPyTorch - 简易版 PyTorch 实现
用于学习深度学习框架的核心原理
"""

__version__ = "0.1.0"

from .core.tensor import Tensor
from .core import dtype

# 便捷函数
def tensor(data, dtype=None, requires_grad=False, device=None):
    """创建张量"""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, device=device)

def zeros(shape, dtype=None, requires_grad=False):
    """创建全零张量"""
    import numpy as np
    data = np.zeros(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def ones(shape, dtype=None, requires_grad=False):
    """创建全一张量"""
    import numpy as np
    data = np.ones(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def randn(*shape, dtype=None, requires_grad=False):
    """创建随机张量（标准正态分布）"""
    import numpy as np
    data = np.random.randn(*shape).astype(dtype if dtype else np.float32)
    return Tensor(data, requires_grad=requires_grad)

# 导入常用操作
from .ops import math

# 导入神经网络模块
from . import nn

# 导入优化器模块
from . import optim

__all__ = [
    'Tensor',
    'tensor',
    'zeros',
    'ones',
    'randn',
    'dtype',
    'nn',
    'optim',
]

