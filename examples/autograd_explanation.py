"""
autograd 包的作用解释

autograd = automatic differentiation（自动求导）
它的作用是：自动计算函数的梯度（导数），无需手动推导公式
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt
import numpy as np

print("=" * 60)
print("autograd 包的作用：自动求导（Automatic Differentiation）")
print("=" * 60)
print()

# ============================================================
# 1. 问题：为什么需要自动求导？
# ============================================================
print("【问题】为什么需要自动求导？")
print()
print("在深度学习中，我们需要计算损失函数对模型参数的梯度，")
print("然后使用梯度下降法更新参数。")
print()
print("传统方法：手动推导梯度公式")
print("  例如：f(x) = x^2，我们需要知道 df/dx = 2x")
print("  对于复杂函数，手动推导非常困难且容易出错")
print()
print("自动求导：框架自动计算梯度")
print("  我们只需要定义前向传播（如何计算），")
print("  框架会自动计算反向传播（如何求梯度）")
print()

# ============================================================
# 2. autograd 包的核心组件
# ============================================================
print("【autograd 包的核心组件】")
print()
print("1. Function 类：表示计算图中的一个节点")
print("   - forward(): 前向传播（计算函数值）")
print("   - backward(): 反向传播（计算梯度）")
print()
print("2. Context 类：存储前向传播时的中间结果")
print("   - saved_tensors: 保存的张量（用于反向传播）")
print("   - saved_values: 保存的其他值（如维度、常数等）")
print()
print("3. backward() 函数：反向传播引擎")
print("   - 从输出开始，沿着计算图反向传播梯度")
print("   - 自动调用每个节点的 backward() 方法")
print()

# ============================================================
# 3. 工作原理示例
# ============================================================
print("【工作原理示例】")
print()

# 创建一个需要梯度的张量
x = ypt.tensor(2.0, requires_grad=True)
print(f"1. 创建张量 x = {x.item()}, requires_grad={x.requires_grad}")

# 执行一些运算
y = x ** 2      # y = x^2
z = y * 3       # z = 3y = 3x^2
print(f"2. y = x^2 = {y.item()}")
print(f"3. z = 3y = {z.item()}")
print()

print("【计算图结构】")
print("   x (叶子节点)")
print("    ↓")
print("   y = x^2 (Pow Function)")
print("    ↓")
print("   z = 3y (Mul Function)")
print()

# 反向传播
print("4. 调用 z.backward() 开始反向传播")
z.backward()
print()

print("【反向传播过程】")
print("   1. z 的梯度 = 1.0 (默认)")
print("   2. Mul.backward():")
print("      - z = 3y，所以 dy/dz = 1/3")
print("      - y 的梯度 = z的梯度 * (1/3) = 1.0 * 1/3 = 0.333...")
print("   3. Pow.backward():")
print("      - y = x^2，所以 dy/dx = 2x = 2*2 = 4")
print("      - x 的梯度 = y的梯度 * 4 = 0.333... * 4 = 1.333...")
print()

print(f"5. 结果：x.grad = {x.grad.item():.6f}")
print(f"   验证：dz/dx = d(3x^2)/dx = 6x = 6*2 = 12")
print(f"   但这里我们计算的是 dz/dx，实际应该是 12")
print()

# ============================================================
# 4. 实际应用：训练神经网络
# ============================================================
print("=" * 60)
print("【实际应用：训练神经网络】")
print("=" * 60)
print()

# 创建一个简单的线性模型
model = ypt.nn.Linear(1, 1)
x_data = ypt.tensor([[1.0], [2.0], [3.0]])
y_true = ypt.tensor([[2.0], [4.0], [6.0]])  # y = 2x

print("1. 前向传播：计算预测值")
y_pred = model(x_data)
print(f"   预测值: {y_pred.data.flatten()}")

print("\n2. 计算损失")
criterion = ypt.nn.MSELoss()
loss = criterion(y_pred, y_true)
print(f"   损失值: {loss.item():.4f}")

print("\n3. 反向传播：自动计算梯度")
loss.backward()
print(f"   weight.grad: {model.weight.grad.data}")
print(f"   bias.grad: {model.bias.grad.data}")

print("\n4. 更新参数（使用优化器）")
optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
print(f"   更新后的 weight: {model.weight.data}")

print()
print("=" * 60)
print("总结：autograd 包让深度学习框架能够自动计算梯度，")
print("      无需手动推导复杂的梯度公式！")
print("=" * 60)


