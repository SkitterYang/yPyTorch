"""SGD 优化器实现"""

from .optimizer import Optimizer
from ..core.tensor import Tensor
from typing import Iterator, Optional


class SGD(Optimizer):
    """
    随机梯度下降（SGD）优化器
    
    更新规则: param = param - lr * grad
    带动量: param = param - lr * (grad + momentum * velocity)
    """
    
    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        初始化 SGD 优化器
        
        Args:
            params: 参数迭代器
            lr: 学习率
            momentum: 动量系数（0 表示不使用动量）
            weight_decay: 权重衰减系数（L2 正则化）
        """
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self):
        """执行一步优化"""
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # 权重衰减（L2 正则化）
                if weight_decay != 0:
                    grad = grad + weight_decay * param
                
                # 动量
                if momentum != 0:
                    # 获取或初始化速度
                    param_id = id(param)
                    if param_id not in self.state:
                        self.state[param_id] = {}
                        self.state[param_id]['velocity'] = Tensor(
                            grad.data * 0.0  # 创建与梯度相同形状的零张量
                        )
                    
                    velocity = self.state[param_id]['velocity']
                    # 更新速度: v = momentum * v + grad
                    velocity._storage._data = momentum * velocity.data + grad.data
                    # 更新参数: param = param - lr * v
                    param._storage._data = param.data - lr * velocity.data
                else:
                    # 不使用动量: param = param - lr * grad
                    param._storage._data = param.data - lr * grad.data

