"""神经网络模块示例"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt
import numpy as np


def example_linear_layer():
    """示例：线性层"""
    print("=" * 50)
    print("示例 1: 线性层")
    print("=" * 50)
    
    # 创建线性层
    linear = ypt.nn.Linear(in_features=4, out_features=2)
    
    print(f"线性层: {linear}")
    print(f"权重形状: {linear.weight.shape}")
    print(f"偏置形状: {linear.bias.shape if linear.bias is not None else None}")
    
    # 前向传播
    x = ypt.randn(3, 4)  # 批量大小为 3
    y = linear(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出:\n{y}")
    print()


def example_activation():
    """示例：激活函数"""
    print("=" * 50)
    print("示例 2: 激活函数")
    print("=" * 50)
    
    x = ypt.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"输入: {x}")
    
    # ReLU
    relu = ypt.nn.ReLU()
    y_relu = relu(x)
    print(f"\nReLU 输出: {y_relu}")
    
    # Sigmoid
    sigmoid = ypt.nn.Sigmoid()
    y_sigmoid = sigmoid(x)
    print(f"Sigmoid 输出: {y_sigmoid}")
    
    # Tanh
    tanh = ypt.nn.Tanh()
    y_tanh = tanh(x)
    print(f"Tanh 输出: {y_tanh}")
    print()


def example_loss_function():
    """示例：损失函数"""
    print("=" * 50)
    print("示例 3: 损失函数")
    print("=" * 50)
    
    # MSE Loss
    mse_loss = ypt.nn.MSELoss()
    y_pred = ypt.tensor([1.0, 2.0, 3.0])
    y_true = ypt.tensor([1.5, 2.5, 2.5])
    
    loss = mse_loss(y_pred, y_true)
    print(f"预测值: {y_pred}")
    print(f"真实值: {y_true}")
    print(f"MSE Loss: {loss.item()}")
    print()


def example_simple_network():
    """示例：简单神经网络"""
    print("=" * 50)
    print("示例 4: 简单神经网络")
    print("=" * 50)
    
    # 创建一个简单的两层网络
    class SimpleNet(ypt.nn.Module):
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
    
    # 创建网络
    net = SimpleNet()
    print(f"网络: {net}")
    
    # 查看参数
    print("\n参数:")
    for name, param in net.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # 前向传播
    x = ypt.randn(2, 4)  # 批量大小为 2
    y = net(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出:\n{y}")
    print()


def example_training_step():
    """示例：训练步骤"""
    print("=" * 50)
    print("示例 5: 训练步骤（带梯度）")
    print("=" * 50)
    
    # 创建简单的线性模型
    model = ypt.nn.Linear(2, 1)
    criterion = ypt.nn.MSELoss()
    
    # 准备数据
    x = ypt.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_true = ypt.tensor([[3.0], [5.0], [7.0]])  # y = x[0] + x[1]
    
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    
    print(f"输入:\n{x}")
    print(f"预测值:\n{y_pred}")
    print(f"真实值:\n{y_true}")
    print(f"初始损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n梯度:")
    print(f"weight.grad:\n{model.weight.grad}")
    print(f"bias.grad: {model.bias.grad}")
    print()


if __name__ == '__main__':
    example_linear_layer()
    example_activation()
    example_loss_function()
    example_simple_network()
    example_training_step()
    
    print("=" * 50)
    print("所有神经网络示例运行完成！")
    print("=" * 50)

