"""线性层实现"""

import numpy as np
from .module import Module
from ..core.tensor import Tensor
from ..utils.init import xavier_uniform_, zeros_, ones_
import ypytorch as ypt


class Linear(Module):
    """
    线性层（全连接层）
    
    实现: y = xW^T + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        初始化线性层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # 初始化权重
        weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32))
        xavier_uniform_(weight)
        self.register_parameter('weight', weight)
        
        # 初始化偏置
        if bias:
            bias_tensor = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            self.register_parameter('bias', bias_tensor)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (..., in_features)
            
        Returns:
            输出张量，形状为 (..., out_features)
        """
        # y = x @ W^T + b
        output = ypt.ops.math.matmul(x, self.weight.transpose(0, 1))
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

