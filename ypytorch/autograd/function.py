"""Function 基类 - 计算图节点"""

from typing import List, Optional, Tuple
from ..core.tensor import Tensor


class Function:
    """
    Function 基类，表示计算图中的一个节点
    
    每个操作（如加法、乘法）都会创建一个 Function 实例，
    用于记录前向传播和反向传播的信息
    """
    
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        前向传播
        
        Args:
            ctx: 上下文对象，用于存储反向传播需要的信息
            *args: 输入参数
            
        Returns:
            输出张量
        """
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            ctx: 上下文对象，包含前向传播时保存的信息
            grad_output: 上游梯度
            
        Returns:
            输入参数的梯度元组
        """
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        应用函数，创建 Function 实例并执行前向传播
        
        Args:
            *args: 输入参数
            
        Returns:
            输出张量
        """
        # 创建上下文对象
        ctx = Context()
        
        # 检查是否需要梯度
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad 
            for arg in args
        )
        
        # 执行前向传播（forward 方法中会保存需要的张量）
        output = cls.forward(ctx, *args, **kwargs)
        
        # 如果不需要梯度，直接返回
        if not requires_grad:
            return output
        
        # 设置梯度函数
        if isinstance(output, Tensor):
            output._grad_fn = cls
            output._ctx = ctx
            output._is_leaf = False
            output.requires_grad = True
        
        return output


class Context:
    """
    上下文对象，用于在前向传播和反向传播之间传递信息
    """
    
    def __init__(self):
        self.saved_tensors: List[Tensor] = []
        self.saved_values = {}
    
    def save_for_backward(self, *tensors: Tensor):
        """保存张量用于反向传播"""
        self.saved_tensors.extend(tensors)
    
    def save(self, key: str, value):
        """保存任意值用于反向传播"""
        self.saved_values[key] = value

