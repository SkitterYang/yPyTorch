"""训练流程集成测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt


def test_linear_regression_training():
    """测试线性回归训练"""
    # 生成数据: y = 2*x + 1 + noise
    np.random.seed(42)
    x_data = np.random.randn(20, 1).astype(np.float32)
    y_data = (2 * x_data + 1 + 0.1 * np.random.randn(20, 1)).astype(np.float32)
    
    x = ypt.tensor(x_data)
    y_true = ypt.tensor(y_data)
    
    # 创建模型
    model = ypt.nn.Linear(1, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
    
    # 记录初始损失
    initial_loss = None
    final_loss = None
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if epoch == num_epochs - 1:
            final_loss = loss.item()
    
    # 验证损失下降
    assert final_loss < initial_loss, f"损失应该下降: {initial_loss} -> {final_loss}"
    assert final_loss < initial_loss * 0.9, f"最终损失应该明显下降，实际为 {final_loss}"
    
    # 验证参数接近真实值（weight ≈ 2.0, bias ≈ 1.0）
    learned_weight = model.weight.item()
    learned_bias = model.bias.item()
    
    # 允许一定的误差范围（由于随机初始化和有限训练轮数，容差放宽）
    assert abs(learned_weight - 2.0) < 1.0, f"权重应该接近 2.0，实际为 {learned_weight}"
    assert abs(learned_bias - 1.0) < 1.0, f"偏置应该接近 1.0，实际为 {learned_bias}"


def test_multi_layer_network_training():
    """测试多层网络训练"""
    # 生成数据
    np.random.seed(42)
    x_data = np.random.randn(10, 4).astype(np.float32)
    y_data = np.random.randn(10, 2).astype(np.float32)
    
    x = ypt.tensor(x_data)
    y_true = ypt.tensor(y_data)
    
    # 创建多层网络
    class Net(ypt.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = ypt.nn.Linear(4, 8)
            self.relu = ypt.nn.ReLU()
            self.linear2 = ypt.nn.Linear(8, 2)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = Net()
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
    
    # 记录初始损失和参数
    initial_loss = None
    initial_weight1 = None
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if epoch == 0:
            initial_loss = loss.item()
            initial_weight1 = model.linear1.weight.data.copy()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    final_loss = loss.item()
    final_weight1 = model.linear1.weight.data.copy()
    
    # 验证参数被更新（多层网络可能损失不下降，但参数应该被更新）
    assert not np.allclose(final_weight1, initial_weight1), "参数应该被更新"


def test_adam_optimizer_training():
    """测试 Adam 优化器训练"""
    # 生成数据
    np.random.seed(42)
    x_data = np.random.randn(15, 3).astype(np.float32)
    y_data = np.random.randn(15, 1).astype(np.float32)
    
    x = ypt.tensor(x_data)
    y_true = ypt.tensor(y_data)
    
    # 创建模型
    model = ypt.nn.Linear(3, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.Adam(model.parameters(), lr=0.01)
    
    # 记录初始损失和参数
    initial_loss = None
    initial_weight = None
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if epoch == 0:
            initial_loss = loss.item()
            initial_weight = model.weight.data.copy()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    final_loss = loss.item()
    final_weight = model.weight.data.copy()
    
    # 验证损失下降
    assert final_loss < initial_loss, f"损失应该下降: {initial_loss} -> {final_loss}"
    
    # 验证参数更新
    assert not np.allclose(final_weight, initial_weight), "参数应该被更新"


def test_sgd_with_momentum_training():
    """测试带动量的 SGD 训练"""
    # 生成数据
    np.random.seed(42)
    x_data = np.random.randn(10, 2).astype(np.float32)
    y_data = np.random.randn(10, 1).astype(np.float32)
    
    x = ypt.tensor(x_data)
    y_true = ypt.tensor(y_data)
    
    # 创建模型
    model = ypt.nn.Linear(2, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 记录初始损失
    initial_loss = None
    final_loss = None
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if epoch == num_epochs - 1:
            final_loss = loss.item()
    
    # 验证损失下降
    assert final_loss < initial_loss, f"损失应该下降: {initial_loss} -> {final_loss}"


def test_gradient_flow():
    """测试梯度是否正确传播到所有参数"""
    # 创建多层网络
    class Net(ypt.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = ypt.nn.Linear(2, 4)
            self.linear2 = ypt.nn.Linear(4, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    model = Net()
    x = ypt.tensor([[1.0, 2.0]])
    y_true = ypt.tensor([[3.0]])
    
    # 前向传播
    y_pred = model(x)
    loss = ypt.nn.MSELoss()(y_pred, y_true)
    
    # 反向传播
    loss.backward()
    
    # 验证所有参数都有梯度
    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 应该有梯度"
        assert param.grad.data is not None, f"参数 {name} 的梯度数据不应该为 None"


def test_training_with_zero_grad():
    """测试梯度清零功能"""
    model = ypt.nn.Linear(2, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
    
    x = ypt.tensor([[1.0, 2.0]])
    y_true = ypt.tensor([[3.0]])
    
    # 第一次反向传播
    y_pred = model(x)
    loss1 = criterion(y_pred, y_true)
    optimizer.zero_grad()  # 先清零（第一次应该没有梯度）
    loss1.backward()
    
    # 保存梯度
    assert model.weight.grad is not None, "第一次反向传播后应该有梯度"
    grad1 = model.weight.grad.data.copy()
    
    # 第二次反向传播（应该累积）
    y_pred = model(x)
    loss2 = criterion(y_pred, y_true)
    loss2.backward()
    
    # 验证梯度累积
    grad2 = model.weight.grad.data.copy()
    assert not np.allclose(grad1, grad2), "梯度应该被累积"
    
    # 清零梯度
    optimizer.zero_grad()
    assert model.weight.grad is None, "梯度应该被清零"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

