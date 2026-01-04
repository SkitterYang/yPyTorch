"""数学运算操作"""

import numpy as np
from ..core.tensor import Tensor
from typing import Union
from ..autograd.functions import Add, Sub, Mul, Div, MatMul, Neg, Pow, Exp, Log, Max


def _to_tensor(x):
    """将输入转换为 Tensor"""
    if not isinstance(x, Tensor):
        return Tensor(x)
    return x


def add(x: Union[Tensor, np.ndarray, list, float], 
        y: Union[Tensor, np.ndarray, list, float]) -> Tensor:
    """加法运算"""
    x = _to_tensor(x)
    y = _to_tensor(y)
    
    # 如果不需要梯度，直接计算
    if not (x.requires_grad or y.requires_grad):
        result_data = np.add(x.data, y.data)
        return Tensor(result_data, requires_grad=False)
    
    # 使用 Function 支持自动求导
    return Add.apply(x, y)


def sub(x: Union[Tensor, np.ndarray, list, float],
        y: Union[Tensor, np.ndarray, list, float]) -> Tensor:
    """减法运算"""
    x = _to_tensor(x)
    y = _to_tensor(y)
    
    if not (x.requires_grad or y.requires_grad):
        result_data = np.subtract(x.data, y.data)
        return Tensor(result_data, requires_grad=False)
    
    return Sub.apply(x, y)


def mul(x: Union[Tensor, np.ndarray, list, float],
        y: Union[Tensor, np.ndarray, list, float]) -> Tensor:
    """乘法运算（逐元素）"""
    x = _to_tensor(x)
    y = _to_tensor(y)
    
    if not (x.requires_grad or y.requires_grad):
        result_data = np.multiply(x.data, y.data)
        return Tensor(result_data, requires_grad=False)
    
    return Mul.apply(x, y)


def div(x: Union[Tensor, np.ndarray, list, float],
        y: Union[Tensor, np.ndarray, list, float]) -> Tensor:
    """除法运算（逐元素）"""
    x = _to_tensor(x)
    y = _to_tensor(y)
    
    if not (x.requires_grad or y.requires_grad):
        result_data = np.divide(x.data, y.data)
        return Tensor(result_data, requires_grad=False)
    
    return Div.apply(x, y)


def matmul(x: Tensor, y: Tensor) -> Tensor:
    """矩阵乘法"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    if not isinstance(y, Tensor):
        y = Tensor(y)
    
    if not (x.requires_grad or y.requires_grad):
        result_data = np.matmul(x.data, y.data)
        return Tensor(result_data, requires_grad=False)
    
    return MatMul.apply(x, y)


def neg(x: Tensor) -> Tensor:
    """取负"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.negative(x.data)
        return Tensor(result_data, requires_grad=False)
    
    return Neg.apply(x)


def pow(x: Tensor, power: Union[int, float]) -> Tensor:
    """幂运算"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.power(x.data, power)
        return Tensor(result_data, requires_grad=False)
    
    return Pow.apply(x, power)


def exp(x: Tensor) -> Tensor:
    """指数运算"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.exp(x.data)
        return Tensor(result_data, requires_grad=False)
    
    return Exp.apply(x)


def log(x: Tensor) -> Tensor:
    """自然对数"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    if not x.requires_grad:
        result_data = np.log(x.data)
        return Tensor(result_data, requires_grad=False)
    
    return Log.apply(x)


def sqrt(x: Tensor) -> Tensor:
    """平方根"""
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    result_data = np.sqrt(x.data)
    return Tensor(result_data, requires_grad=x.requires_grad)

