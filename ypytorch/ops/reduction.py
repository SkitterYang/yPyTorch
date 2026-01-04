"""归约操作"""

import numpy as np
from ..core.tensor import Tensor
from typing import Optional, Union
from ..autograd.functions import Sum


def sum(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """
    求和
    
    Args:
        x: 输入张量
        dim: 求和的维度，None 表示所有维度
        keepdim: 是否保持维度
        
    Returns:
        求和后的张量
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.sum(x.data, axis=dim, keepdims=keepdim)
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data, requires_grad=False)
    
    return Sum.apply(x, dim, keepdim)


def mean(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """
    求均值
    
    使用 sum / n 来实现，以支持自动求导
    """
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.mean(x.data, axis=dim, keepdims=keepdim)
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data, requires_grad=False)
    
    # 使用 sum / n 来实现 mean，以支持自动求导
    sum_result = sum(x, dim=dim, keepdim=keepdim)
    
    # 计算元素数量
    if dim is None:
        # 所有维度
        n = x.size
    else:
        # 指定维度
        if isinstance(dim, int):
            dim = (dim,)
        n = 1
        for d in dim:
            n *= x.shape[d]
    
    # mean = sum / n
    result = sum_result / n
    return result


def max(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """求最大值"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    result_data = np.max(x.data, axis=dim, keepdims=keepdim)
    return Tensor(result_data, requires_grad=x.requires_grad)


def min(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """求最小值"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    result_data = np.min(x.data, axis=dim, keepdims=keepdim)
    return Tensor(result_data, requires_grad=x.requires_grad)


def std(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """求标准差"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    result_data = np.std(x.data, axis=dim, keepdims=keepdim)
    return Tensor(result_data, requires_grad=x.requires_grad)


def var(x: Tensor, dim: Optional[Union[int, tuple]] = None, keepdim: bool = False) -> Tensor:
    """求方差"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    result_data = np.var(x.data, axis=dim, keepdims=keepdim)
    return Tensor(result_data, requires_grad=x.requires_grad)

