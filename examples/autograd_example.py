"""自动求导示例"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt
import numpy as np


def example_basic_autograd():
    """示例：基础自动求导"""
    print("=" * 50)
    print("示例 1: 基础自动求导")
    print("=" * 50)
    
    # 创建需要梯度的张量
    x = ypt.tensor([2.0, 3.0], requires_grad=True)
    y = ypt.tensor([4.0, 5.0], requires_grad=True)
    
    # 前向传播
    z = x * y
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = x * y = {z}")
    
    # 反向传播
    loss = z.sum()
    loss.backward()
    
    print(f"\n反向传播后:")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    print(f"\n验证:")
    print(f"dz/dx = y = {y.data}")
    print(f"dz/dy = x = {x.data}")
    print()


def example_chain_rule():
    """示例：链式法则"""
    print("=" * 50)
    print("示例 2: 链式法则")
    print("=" * 50)
    
    x = ypt.tensor(2.0, requires_grad=True)
    
    # 复合函数: y = x^2, z = exp(y), w = log(z)
    y = x ** 2
    z = ypt.ops.math.exp(y)
    w = ypt.ops.math.log(z)
    
    print(f"x = {x.item()}")
    print(f"y = x^2 = {y.item()}")
    print(f"z = exp(y) = {z.item()}")
    print(f"w = log(z) = {w.item()}")
    
    # 反向传播
    w.backward()
    
    print(f"\n反向传播后:")
    print(f"x.grad = {x.grad.item()}")
    print(f"\n验证 (dw/dx = 2x):")
    print(f"理论值: 2 * {x.item()} = {2 * x.item()}")
    print()


def example_linear_regression():
    """示例：线性回归的梯度"""
    print("=" * 50)
    print("示例 3: 线性回归梯度")
    print("=" * 50)
    
    # 简单的线性模型: y = w * x + b
    w = ypt.tensor(2.0, requires_grad=True)
    b = ypt.tensor(1.0, requires_grad=True)
    x = ypt.tensor(3.0)
    
    # 前向传播
    y_pred = w * x + b
    y_true = ypt.tensor(10.0)
    
    # 损失函数: L = (y_pred - y_true)^2
    loss = (y_pred - y_true) ** 2
    
    print(f"w = {w.item()}, b = {b.item()}, x = {x.item()}")
    print(f"y_pred = {y_pred.item()}")
    print(f"y_true = {y_true.item()}")
    print(f"loss = {loss.item()}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n反向传播后:")
    print(f"dL/dw = {w.grad.item()}")
    print(f"dL/db = {b.grad.item()}")
    
    print(f"\n验证:")
    print(f"dL/dw = 2 * (w*x + b - y_true) * x")
    print(f"      = 2 * ({w.item()}*{x.item()} + {b.item()} - {y_true.item()}) * {x.item()}")
    expected_w_grad = 2 * (w.item() * x.item() + b.item() - y_true.item()) * x.item()
    print(f"      = {expected_w_grad}")
    print()


def example_matrix_multiplication():
    """示例：矩阵乘法的梯度"""
    print("=" * 50)
    print("示例 4: 矩阵乘法梯度")
    print("=" * 50)
    
    A = ypt.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = ypt.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    # 矩阵乘法
    C = A @ B
    
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"C = A @ B = \n{C}")
    
    # 对结果求和并反向传播
    loss = C.sum()
    loss.backward()
    
    print(f"\n反向传播后:")
    print(f"A.grad = \n{A.grad}")
    print(f"B.grad = \n{B.grad}")
    print()


if __name__ == '__main__':
    example_basic_autograd()
    example_chain_rule()
    example_linear_regression()
    example_matrix_multiplication()
    
    print("=" * 50)
    print("所有自动求导示例运行完成！")
    print("=" * 50)

