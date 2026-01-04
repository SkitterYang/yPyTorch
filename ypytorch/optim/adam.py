"""Adam 优化器实现"""

from .optimizer import Optimizer
from ..core.tensor import Tensor
from typing import Iterator
import numpy as np


class Adam(Optimizer):
    """
    Adam 优化器
    
    Adam (Adaptive Moment Estimation) 是一种自适应学习率的优化算法
    """
    
    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        初始化 Adam 优化器
        
        Args:
            params: 参数迭代器
            lr: 学习率（默认 0.001）
            betas: 动量衰减系数 (beta1, beta2)（默认 (0.9, 0.999)）
            eps: 数值稳定性常数（默认 1e-8）
            weight_decay: 权重衰减系数（默认 0.0）
        """
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
        self.step_count = 0
    
    def step(self):
        """执行一步优化"""
        self.step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # 权重衰减
                if weight_decay != 0:
                    grad = grad + weight_decay * param
                
                param_id = id(param)
                
                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'exp_avg': Tensor(grad.data * 0.0),  # 一阶矩估计
                        'exp_avg_sq': Tensor(grad.data * 0.0)  # 二阶矩估计
                    }
                
                state = self.state[param_id]
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # 更新一阶矩估计: m = beta1 * m + (1 - beta1) * grad
                exp_avg._storage._data = beta1 * exp_avg.data + (1 - beta1) * grad.data
                
                # 更新二阶矩估计: v = beta2 * v + (1 - beta2) * grad^2
                exp_avg_sq._storage._data = beta2 * exp_avg_sq.data + (1 - beta2) * grad.data ** 2
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算修正后的学习率
                step_size = lr / bias_correction1
                
                # 计算修正后的梯度估计
                denom = np.sqrt(exp_avg_sq.data / bias_correction2) + eps
                
                # 更新参数: param = param - step_size * m / sqrt(v)
                param._storage._data = param.data - step_size * (exp_avg.data / denom)

