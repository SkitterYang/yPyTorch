"""常用操作的梯度函数实现"""

import numpy as np
from .function import Function
from ..core.tensor import Tensor


class Add(Function):
    """加法操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        # 确保 x 和 y 是 Tensor
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        result = Tensor(x.data + y.data)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 加法的梯度：对两个输入都是 grad_output
        # 需要处理广播
        
        x, y = ctx.saved_tensors
        
        # 处理广播：如果形状不同，需要 sum 到原始形状
        grad_x = grad_output
        grad_y = grad_output
        
        # 如果 x 的形状与 grad_output 不同，需要 sum
        if x.shape != grad_output.shape:
            # 找到需要 sum 的维度
            axes = tuple(range(len(grad_output.shape) - len(x.shape)))
            grad_x = grad_output.sum(axis=axes)
            # 如果还有维度不匹配，reshape
            while grad_x.shape != x.shape:
                grad_x = grad_x.sum(axis=0, keepdims=True)
                if grad_x.shape == x.shape:
                    break
        
        # 如果 y 的形状与 grad_output 不同，需要 sum
        if y.shape != grad_output.shape:
            axes = tuple(range(len(grad_output.shape) - len(y.shape)))
            grad_y = grad_output.sum(axis=axes)
            while grad_y.shape != y.shape:
                grad_y = grad_y.sum(axis=0, keepdims=True)
                if grad_y.shape == y.shape:
                    break
        
        return grad_x, grad_y


class Mul(Function):
    """乘法操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        ctx.save_for_backward(x, y)
        result = Tensor(x.data * y.data)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 乘法的梯度：
        # d(x*y)/dx = y * grad_output
        # d(x*y)/dy = x * grad_output
        
        x, y = ctx.saved_tensors
        
        grad_x = Tensor(y.data * grad_output.data)
        grad_y = Tensor(x.data * grad_output.data)
        
        # 处理广播（简化版）
        if grad_x.shape != x.shape:
            if x.shape == ():
                grad_x = grad_x.sum()
            else:
                # 简化处理：如果维度不匹配，尝试 sum
                while len(grad_x.shape) > len(x.shape):
                    grad_x = grad_x.sum(axis=0)
        
        if grad_y.shape != y.shape:
            if y.shape == ():
                grad_y = grad_y.sum()
            else:
                while len(grad_y.shape) > len(y.shape):
                    grad_y = grad_y.sum(axis=0)
        
        return grad_x, grad_y


class Sub(Function):
    """减法操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        result = Tensor(x.data - y.data)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 减法的梯度：
        # d(x-y)/dx = grad_output
        # d(x-y)/dy = -grad_output
        
        return grad_output, -grad_output


class Div(Function):
    """除法操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        ctx.save_for_backward(x, y)
        result = Tensor(x.data / y.data)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 除法的梯度：
        # d(x/y)/dx = (1/y) * grad_output
        # d(x/y)/dy = (-x/y^2) * grad_output
        
        x, y = ctx.saved_tensors
        
        grad_x = Tensor((1.0 / y.data) * grad_output.data)
        grad_y = Tensor((-x.data / (y.data ** 2)) * grad_output.data)
        
        return grad_x, grad_y


class MatMul(Function):
    """矩阵乘法操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        ctx.save_for_backward(x, y)
        result = Tensor(np.matmul(x.data, y.data))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 矩阵乘法的梯度：
        # d(x@y)/dx = grad_output @ y.T
        # d(x@y)/dy = x.T @ grad_output
        
        x, y = ctx.saved_tensors
        
        grad_x = Tensor(np.matmul(grad_output.data, y.data.T))
        grad_y = Tensor(np.matmul(x.data.T, grad_output.data))
        
        return grad_x, grad_y


class Sum(Function):
    """求和操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        ctx.save_for_backward(x)
        ctx.save('dim', dim)
        ctx.save('keepdim', keepdim)
        ctx.save('original_shape', x.shape)
        result = Tensor(np.sum(x.data, axis=dim, keepdims=keepdim))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # 求和的梯度：将 grad_output 广播回原始形状
        
        x = ctx.saved_tensors[0]
        dim = ctx.saved_values.get('dim')
        keepdim = ctx.saved_values.get('keepdim', False)
        original_shape = ctx.saved_values.get('original_shape')
        
        # 如果 keepdim=False，需要恢复维度
        if not keepdim and dim is not None:
            # 在指定维度上添加维度
            grad_output = Tensor(np.expand_dims(grad_output.data, axis=dim))
        
        # 广播到原始形状
        grad = Tensor(np.broadcast_to(grad_output.data, original_shape))
        
        return grad


class Neg(Function):
    """取负操作的梯度函数"""
    
    @staticmethod
    def forward(ctx, x):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        result = Tensor(-x.data)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        return -grad_output


class Pow(Function):
    """幂运算的梯度函数"""
    
    @staticmethod
    def forward(ctx, x, power):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        ctx.save_for_backward(x)
        ctx.save('power', power)
        result = Tensor(np.power(x.data, power))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # d(x^p)/dx = p * x^(p-1) * grad_output
        
        x = ctx.saved_tensors[0]
        power = ctx.saved_values.get('power')
        
        grad = Tensor(power * np.power(x.data, power - 1) * grad_output.data)
        return grad


class Exp(Function):
    """指数运算的梯度函数"""
    
    @staticmethod
    def forward(ctx, x):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        ctx.save_for_backward(x)
        result = Tensor(np.exp(x.data))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # d(exp(x))/dx = exp(x) * grad_output
        
        x = ctx.saved_tensors[0]
        grad = Tensor(np.exp(x.data) * grad_output.data)
        return grad


class Log(Function):
    """对数运算的梯度函数"""
    
    @staticmethod
    def forward(ctx, x):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        ctx.save_for_backward(x)
        result = Tensor(np.log(x.data))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # d(log(x))/dx = (1/x) * grad_output
        
        x = ctx.saved_tensors[0]
        grad = Tensor((1.0 / x.data) * grad_output.data)
        return grad


class Max(Function):
    """最大值操作的梯度函数（简化版，用于 ReLU）"""
    
    @staticmethod
    def forward(ctx, x, y):
        """前向传播"""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        ctx.save_for_backward(x, y)
        result = Tensor(np.maximum(x.data, y.data))
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        # max(x, y) 的梯度：
        # 如果 x > y，梯度给 x；否则给 y
        
        x, y = ctx.saved_tensors
        mask = x.data > y.data
        grad_x = Tensor(mask.astype(np.float32) * grad_output.data)
        grad_y = Tensor((1 - mask.astype(np.float32)) * grad_output.data)
        
        return grad_x, grad_y
