"""神经网络模块测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt


def test_linear_layer():
    """测试线性层"""
    linear = ypt.nn.Linear(in_features=4, out_features=2)
    
    assert linear.in_features == 4
    assert linear.out_features == 2
    assert linear.weight.shape == (2, 4)
    assert linear.bias.shape == (2,)
    
    # 前向传播
    x = ypt.randn(3, 4)
    y = linear(x)
    
    assert y.shape == (3, 2)


def test_relu():
    """测试 ReLU 激活函数"""
    relu = ypt.nn.ReLU()
    x = ypt.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = relu(x)
    
    # ReLU 应该将负数变为 0
    assert y.data[0] == 0.0
    assert y.data[1] == 0.0
    assert y.data[2] == 0.0
    assert y.data[3] > 0
    assert y.data[4] > 0


def test_sigmoid():
    """测试 Sigmoid 激活函数"""
    sigmoid = ypt.nn.Sigmoid()
    x = ypt.tensor(0.0)
    y = sigmoid(x)
    
    # Sigmoid(0) = 0.5
    assert abs(y.item() - 0.5) < 0.1


def test_mse_loss():
    """测试 MSE 损失"""
    mse = ypt.nn.MSELoss()
    y_pred = ypt.tensor([1.0, 2.0, 3.0])
    y_true = ypt.tensor([1.0, 2.0, 3.0])
    
    loss = mse(y_pred, y_true)
    assert abs(loss.item() - 0.0) < 1e-6


def test_module_parameters():
    """测试模块参数"""
    linear = ypt.nn.Linear(3, 2)
    
    params = list(linear.parameters())
    assert len(params) == 2  # weight 和 bias
    
    # 检查参数形状
    assert params[0].shape == (2, 3)  # weight
    assert params[1].shape == (2,)  # bias


def test_module_zero_grad():
    """测试梯度清零"""
    linear = ypt.nn.Linear(2, 1)
    
    # 进行一次前向和反向传播
    x = ypt.tensor([[1.0, 2.0]])
    y = linear(x)
    loss = y.sum()
    loss.backward()
    
    # 检查梯度存在
    assert linear.weight.grad is not None
    
    # 清零梯度
    linear.zero_grad()
    
    # 梯度应该被清零（设为 None）
    assert linear.weight.grad is None
    assert linear.bias.grad is None


def test_simple_network():
    """测试简单网络"""
    class Net(ypt.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = ypt.nn.Linear(2, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    net = Net()
    x = ypt.tensor([[1.0, 2.0]])
    y = net(x)
    
    assert y.shape == (1, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

