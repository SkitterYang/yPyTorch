"""损失函数实现"""

from .module import Module
from ..core.tensor import Tensor
import ypytorch as ypt
import numpy as np


class MSELoss(Module):
    """
    均方误差损失函数
    
    L = mean((y_pred - y_true)^2)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        计算损失
        
        Args:
            y_pred: 预测值
            y_true: 真实值
            
        Returns:
            损失值（标量）
        """
        diff = y_pred - y_true
        squared = diff ** 2
        return squared.mean()
    
    def __repr__(self) -> str:
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """
    交叉熵损失函数（简化版）
    
    用于多分类问题
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        计算交叉熵损失
        
        Args:
            y_pred: 预测值，形状为 (N, C) 或 (C,)，其中 C 是类别数
            y_true: 真实标签，形状为 (N,) 或标量，值为类别索引
            
        Returns:
            损失值（标量）
        """
        # 简化实现：使用 softmax + log + 负对数似然
        # 首先应用 softmax
        exp_pred = ypt.ops.math.exp(y_pred)
        sum_exp = exp_pred.sum(dim=-1, keepdim=True)
        softmax = exp_pred / sum_exp
        
        # 计算 log softmax
        log_softmax = ypt.ops.math.log(softmax)
        
        # 获取真实类别的 log 概率
        if len(y_true.shape) == 0:
            # 标量标签
            target_log_prob = log_softmax[int(y_true.item())]
        else:
            # 批量标签
            # 简化：只处理第一个样本
            target_log_prob = log_softmax[0, int(y_true.item())]
        
        # 返回负对数似然
        return -target_log_prob
    
    def __repr__(self) -> str:
        return "CrossEntropyLoss()"

