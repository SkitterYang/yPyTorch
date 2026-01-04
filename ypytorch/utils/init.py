"""参数初始化函数"""

import numpy as np
from ..core.tensor import Tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0):
    """
    Xavier 均匀初始化
    
    Args:
        tensor: 要初始化的张量
        gain: 增益因子
    """
    if len(tensor.shape) < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions")
    
    fan_in = tensor.shape[-1]
    fan_out = tensor.shape[-2] if len(tensor.shape) > 1 else tensor.shape[0]
    
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    
    # 生成均匀分布的随机数
    uniform_data = np.random.uniform(-limit, limit, tensor.shape).astype(np.float32)
    tensor._storage._data = uniform_data


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
    """
    正态分布初始化
    
    Args:
        tensor: 要初始化的张量
        mean: 均值
        std: 标准差
    """
    normal_data = np.random.normal(mean, std, tensor.shape).astype(np.float32)
    tensor._storage._data = normal_data


def zeros_(tensor: Tensor):
    """
    零初始化
    
    Args:
        tensor: 要初始化的张量
    """
    tensor._storage._data = np.zeros(tensor.shape, dtype=np.float32)


def ones_(tensor: Tensor):
    """
    全一初始化
    
    Args:
        tensor: 要初始化的张量
    """
    tensor._storage._data = np.ones(tensor.shape, dtype=np.float32)

