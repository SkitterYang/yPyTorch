"""自动求导测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt


def test_basic_backward():
    """测试基础反向传播"""
    x = ypt.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    
    z.backward()
    
    assert x.grad is not None
    assert np.allclose(x.grad.data, [2.0, 2.0])


def test_add_backward():
    """测试加法反向传播"""
    x = ypt.tensor([1.0, 2.0], requires_grad=True)
    y = ypt.tensor([3.0, 4.0], requires_grad=True)
    
    z = x + y
    loss = z.sum()
    loss.backward()
    
    assert x.grad is not None
    assert y.grad is not None
    assert np.allclose(x.grad.data, [1.0, 1.0])
    assert np.allclose(y.grad.data, [1.0, 1.0])


def test_mul_backward():
    """测试乘法反向传播"""
    x = ypt.tensor([2.0, 3.0], requires_grad=True)
    y = ypt.tensor([4.0, 5.0], requires_grad=True)
    
    z = x * y
    loss = z.sum()
    loss.backward()
    
    assert x.grad is not None
    assert y.grad is not None
    # dz/dx = y
    assert np.allclose(x.grad.data, [4.0, 5.0])
    # dz/dy = x
    assert np.allclose(y.grad.data, [2.0, 3.0])


def test_chain_rule():
    """测试链式法则"""
    x = ypt.tensor(2.0, requires_grad=True)
    y = x ** 2
    z = ypt.ops.math.exp(y)
    w = ypt.ops.math.log(z)
    
    w.backward()
    
    # w = log(exp(x^2)) = x^2
    # dw/dx = 2x = 4
    assert abs(x.grad.item() - 4.0) < 1e-6


def test_sum_backward():
    """测试求和反向传播"""
    x = ypt.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum()
    
    y.backward()
    
    # 所有元素的梯度都应该是 1
    assert np.allclose(x.grad.data, np.ones((2, 2)))


def test_matmul_backward():
    """测试矩阵乘法反向传播"""
    A = ypt.tensor([[1.0, 2.0]], requires_grad=True)
    B = ypt.tensor([[3.0], [4.0]], requires_grad=True)
    
    C = A @ B
    loss = C.sum()
    loss.backward()
    
    # C = A @ B = [[1*3 + 2*4]] = [[11]]
    # dC/dA = B^T = [[3, 4]]
    # dC/dB = A^T = [[1], [2]]
    assert np.allclose(A.grad.data, [[3.0, 4.0]])
    assert np.allclose(B.grad.data, [[1.0], [2.0]])


def test_no_grad():
    """测试不需要梯度的情况"""
    x = ypt.tensor([1.0, 2.0], requires_grad=False)
    y = x * 2
    
    assert not y.requires_grad


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

