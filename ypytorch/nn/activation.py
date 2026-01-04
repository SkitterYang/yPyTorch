"""激活函数实现"""

from .module import Module
from ..core.tensor import Tensor
import ypytorch as ypt
import numpy as np


class ReLU(Module):
    """
    ReLU 激活函数: f(x) = max(0, x)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # ReLU: max(0, x)
        # 使用 Max Function 支持梯度
        from ..autograd.functions import Max
        
        zero = ypt.tensor(0.0)
        
        # 如果不需要梯度，直接计算
        if not x.requires_grad:
            result_data = np.maximum(x.data, 0.0)
            return Tensor(result_data, requires_grad=False)
        
        # 使用 Max Function
        return Max.apply(x, zero)
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid 激活函数: f(x) = 1 / (1 + exp(-x))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # Sigmoid: 1 / (1 + exp(-x))
        neg_x = -x
        exp_neg_x = ypt.ops.math.exp(neg_x)
        one = ypt.tensor(1.0)
        denominator = one + exp_neg_x
        return one / denominator
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """
    Tanh 激活函数: f(x) = tanh(x)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # 或者使用 numpy 的 tanh
        exp_x = ypt.ops.math.exp(x)
        exp_neg_x = ypt.ops.math.exp(-x)
        numerator = exp_x - exp_neg_x
        denominator = exp_x + exp_neg_x
        return numerator / denominator
    
    def __repr__(self) -> str:
        return "Tanh()"

