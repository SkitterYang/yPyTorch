"""优化器基类"""

from typing import Iterator, List
from ..core.tensor import Tensor


class Optimizer:
    """
    所有优化器的基类
    """
    
    def __init__(self, params: Iterator[Tensor], defaults: dict):
        """
        初始化优化器
        
        Args:
            params: 参数迭代器
            defaults: 默认参数字典
        """
        # 将参数转换为列表
        if isinstance(params, Tensor):
            params = [params]
        
        self.param_groups = []
        param_group = {
            'params': list(params),
            **defaults
        }
        self.param_groups.append(param_group)
        
        # 状态字典，用于存储优化器的状态（如动量）
        self.state = {}
    
    def zero_grad(self):
        """将所有参数的梯度清零"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad = None
    
    def step(self):
        """
        执行一步优化，更新参数
        
        子类必须实现这个方法
        """
        raise NotImplementedError
    
    def state_dict(self):
        """
        返回优化器的状态字典
        
        Returns:
            包含优化器状态的字典
        """
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }
    
    def load_state_dict(self, state_dict):
        """
        加载优化器状态
        
        Args:
            state_dict: 状态字典
        """
        self.state = state_dict.get('state', {})
        self.param_groups = state_dict.get('param_groups', [])

