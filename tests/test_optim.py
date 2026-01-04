"""优化器测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt


def test_sgd_step():
    """测试 SGD 优化器"""
    # 创建参数
    param = ypt.tensor([1.0, 2.0], requires_grad=True)
    
    # 创建优化器
    optimizer = ypt.optim.SGD([param], lr=0.1)
    
    # 模拟梯度
    param.grad = ypt.tensor([0.5, 0.5])
    
    # 执行一步优化
    optimizer.step()
    
    # 检查参数是否更新: param = param - lr * grad
    # 应该变成 [1.0 - 0.1*0.5, 2.0 - 0.1*0.5] = [0.95, 1.95]
    assert abs(param.data[0] - 0.95) < 1e-6
    assert abs(param.data[1] - 1.95) < 1e-6


def test_sgd_zero_grad():
    """测试 SGD 梯度清零"""
    param = ypt.tensor([1.0, 2.0], requires_grad=True)
    optimizer = ypt.optim.SGD([param], lr=0.1)
    
    param.grad = ypt.tensor([0.5, 0.5])
    optimizer.zero_grad()
    
    assert param.grad is None


def test_sgd_momentum():
    """测试 SGD 动量"""
    param = ypt.tensor([1.0], requires_grad=True)
    optimizer = ypt.optim.SGD([param], lr=0.1, momentum=0.9)
    
    # 第一次更新
    param.grad = ypt.tensor([1.0])
    optimizer.step()
    
    # 第二次更新
    param.grad = ypt.tensor([1.0])
    optimizer.step()
    
    # 由于动量，第二次更新应该更大
    # 简化验证：参数应该被更新
    assert param.data[0] < 1.0


def test_adam_step():
    """测试 Adam 优化器"""
    param = ypt.tensor([1.0, 2.0], requires_grad=True)
    optimizer = ypt.optim.Adam([param], lr=0.01)
    
    # 模拟梯度
    param.grad = ypt.tensor([0.5, 0.5])
    
    # 执行一步优化
    optimizer.step()
    
    # 检查参数是否更新（应该变小）
    assert param.data[0] < 1.0
    assert param.data[1] < 2.0


def test_adam_zero_grad():
    """测试 Adam 梯度清零"""
    param = ypt.tensor([1.0, 2.0], requires_grad=True)
    optimizer = ypt.optim.Adam([param], lr=0.01)
    
    param.grad = ypt.tensor([0.5, 0.5])
    optimizer.zero_grad()
    
    assert param.grad is None


def test_optimizer_with_model():
    """测试优化器与模型一起使用"""
    model = ypt.nn.Linear(2, 1)
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.1)
    
    # 前向传播
    x = ypt.tensor([[1.0, 2.0]])
    y = model(x)
    loss = y.sum()
    
    # 反向传播
    loss.backward()
    
    # 保存原始权重
    old_weight = model.weight.data.copy()
    
    # 更新参数
    optimizer.step()
    
    # 检查权重是否更新
    assert not np.allclose(model.weight.data, old_weight)


def test_optimizer_parameter_groups():
    """测试优化器参数组"""
    param1 = ypt.tensor([1.0], requires_grad=True)
    param2 = ypt.tensor([2.0], requires_grad=True)
    
    optimizer = ypt.optim.SGD([param1, param2], lr=0.1)
    
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]['params']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

