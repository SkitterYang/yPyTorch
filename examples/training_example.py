"""完整训练流程示例"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt
import numpy as np


def example_sgd_training():
    """示例：使用 SGD 训练简单模型"""
    print("=" * 50)
    print("示例 1: SGD 训练")
    print("=" * 50)
    
    # 创建简单的线性模型
    model = ypt.nn.Linear(2, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
    
    # 准备数据: y = 2*x1 + 3*x2 + 1
    x = ypt.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    y_true = ypt.tensor([
        [9.0],   # 2*1 + 3*2 + 1 = 9
        [14.0],  # 2*2 + 3*3 + 1 = 14
        [19.0],  # 2*3 + 3*4 + 1 = 19
        [24.0]   # 2*4 + 3*5 + 1 = 24
    ])
    
    print("初始参数:")
    print(f"weight: {model.weight.data}")
    print(f"bias: {model.bias.data}")
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print("\n训练后的参数:")
    print(f"weight: {model.weight.data}")
    print(f"bias: {model.bias.data}")
    print()


def example_adam_training():
    """示例：使用 Adam 训练模型"""
    print("=" * 50)
    print("示例 2: Adam 训练")
    print("=" * 50)
    
    # 创建模型
    model = ypt.nn.Linear(3, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.Adam(model.parameters(), lr=0.01)
    
    # 准备数据
    x = ypt.randn(10, 3)
    y_true = ypt.randn(10, 1)
    
    print("训练中...")
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    print()


def example_multi_layer_training():
    """示例：多层网络训练"""
    print("=" * 50)
    print("示例 3: 多层网络训练")
    print("=" * 50)
    
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
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 准备数据
    x = ypt.randn(5, 4)
    y_true = ypt.randn(5, 2)
    
    print("训练多层网络...")
    num_epochs = 30
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    print()


def example_linear_regression():
    """示例：线性回归完整示例"""
    print("=" * 50)
    print("示例 4: 线性回归")
    print("=" * 50)
    
    # 生成数据: y = 2x + 1 + noise
    np.random.seed(42)
    x_data = np.random.randn(100, 1).astype(np.float32)
    y_data = (2 * x_data + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    x = ypt.tensor(x_data)
    y_true = ypt.tensor(y_data)
    
    # 创建模型
    model = ypt.nn.Linear(1, 1)
    criterion = ypt.nn.MSELoss()
    optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
    
    print("训练线性回归模型...")
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print(f"\n学习到的参数:")
    print(f"weight (应该接近 2.0): {model.weight.item():.4f}")
    print(f"bias (应该接近 1.0): {model.bias.item():.4f}")
    print()


if __name__ == '__main__':
    example_sgd_training()
    example_adam_training()
    example_multi_layer_training()
    example_linear_regression()
    
    print("=" * 50)
    print("所有训练示例运行完成！")
    print("=" * 50)

