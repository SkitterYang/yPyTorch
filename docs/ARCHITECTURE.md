# yPyTorch 架构设计

## 项目概述

yPyTorch 是一个简易版的 PyTorch 实现，用于学习深度学习框架的核心原理。本项目将逐步实现 PyTorch 的核心功能，包括张量操作、自动求导、神经网络模块等。

## 核心模块架构

```
ypytorch/
├── core/              # 核心功能
│   ├── tensor.py     # Tensor 类实现
│   ├── storage.py    # 底层存储
│   └── dtype.py      # 数据类型定义
├── autograd/         # 自动求导系统
│   ├── variable.py   # Variable 类（带梯度）
│   ├── function.py   # Function 基类
│   └── engine.py     # 反向传播引擎
├── nn/               # 神经网络模块
│   ├── module.py     # Module 基类
│   ├── linear.py     # 线性层
│   ├── activation.py # 激活函数
│   └── loss.py       # 损失函数
├── optim/            # 优化器
│   ├── optimizer.py  # 优化器基类
│   ├── sgd.py        # SGD 优化器
│   └── adam.py       # Adam 优化器
├── ops/              # 操作符实现
│   ├── math.py       # 数学运算
│   ├── reduction.py  # 归约操作
│   └── elementwise.py # 逐元素操作
└── utils/            # 工具函数
    ├── init.py       # 初始化函数
    └── device.py     # 设备管理
```

## 实现阶段

### 阶段 1: 基础 Tensor（核心数据结构）
- [ ] 实现基础的 Tensor 类
- [ ] 支持基本的数据类型（float32, int32）
- [ ] 实现 shape 和 strides
- [ ] 基础索引和切片

### 阶段 2: 基础操作（Ops）
- [ ] 实现基础数学运算（add, mul, sub, div）
- [ ] 实现矩阵乘法（matmul）
- [ ] 实现转置（transpose）
- [ ] 实现 reshape 和 view

### 阶段 3: 自动求导（Autograd）
- [ ] 实现计算图（Computation Graph）
- [ ] 实现前向传播
- [ ] 实现反向传播
- [ ] 实现梯度累积

### 阶段 4: 神经网络模块（NN）
- [ ] 实现 Module 基类
- [ ] 实现 Linear 层
- [ ] 实现激活函数（ReLU, Sigmoid, Tanh）
- [ ] 实现损失函数（MSE, CrossEntropy）

### 阶段 5: 优化器（Optim）
- [ ] 实现优化器基类
- [ ] 实现 SGD
- [ ] 实现 Adam

### 阶段 6: 完整训练流程
- [ ] 实现完整的训练循环
- [ ] 实现模型保存和加载
- [ ] 添加示例代码

## 技术栈

- **语言**: Python 3.8+
- **依赖**: NumPy（初期使用，后续可替换为纯 Python）
- **测试**: pytest
- **文档**: Markdown

## 设计原则

1. **简洁优先**: 代码清晰易懂，便于学习
2. **逐步实现**: 从简单到复杂，逐步添加功能
3. **API 兼容**: 尽量与 PyTorch API 保持一致
4. **文档完善**: 每个模块都有清晰的文档和示例

