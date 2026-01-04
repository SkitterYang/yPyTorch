# yPyTorch 开发路线图

## 当前状态

项目初始化阶段，正在搭建基础框架。

## 详细计划

### Phase 1: 基础 Tensor 实现（Week 1-2）

**目标**: 实现一个可用的 Tensor 类，支持基本操作

**任务清单**:
- [ ] 设计 Tensor 数据结构
- [ ] 实现底层存储（使用 NumPy 作为后端）
- [ ] 实现 shape 和 dtype 管理
- [ ] 实现基础索引和切片
- [ ] 编写单元测试

**关键文件**:
- `ypytorch/core/tensor.py`
- `ypytorch/core/storage.py`
- `ypytorch/core/dtype.py`

**示例代码**:
```python
import ypytorch as ypt

x = ypt.Tensor([1, 2, 3, 4])
print(x.shape)  # (4,)
print(x.dtype)  # float32
```

### Phase 2: 基础操作实现（Week 3-4）

**目标**: 实现常用的张量操作

**任务清单**:
- [ ] 实现元素级运算（+, -, *, /）
- [ ] 实现矩阵乘法
- [ ] 实现转置和 reshape
- [ ] 实现广播机制
- [ ] 实现归约操作（sum, mean, max, min）

**关键文件**:
- `ypytorch/ops/math.py`
- `ypytorch/ops/reduction.py`
- `ypytorch/ops/elementwise.py`

### Phase 3: 自动求导系统（Week 5-7）

**目标**: 实现自动求导，支持反向传播

**任务清单**:
- [ ] 设计计算图结构
- [ ] 实现 Function 基类
- [ ] 实现常用操作的梯度函数
- [ ] 实现反向传播引擎
- [ ] 实现梯度清零和累积

**关键文件**:
- `ypytorch/autograd/variable.py`
- `ypytorch/autograd/function.py`
- `ypytorch/autograd/engine.py`

**示例代码**:
```python
x = ypt.Tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()
print(x.grad)  # [2.0, 2.0]
```

### Phase 4: 神经网络模块（Week 8-10）

**目标**: 实现基础的神经网络组件

**任务清单**:
- [ ] 实现 Module 基类
- [ ] 实现 Linear 层
- [ ] 实现激活函数
- [ ] 实现损失函数
- [ ] 实现参数初始化

**关键文件**:
- `ypytorch/nn/module.py`
- `ypytorch/nn/linear.py`
- `ypytorch/nn/activation.py`
- `ypytorch/nn/loss.py`

### Phase 5: 优化器（Week 11-12）

**目标**: 实现常用的优化算法

**任务清单**:
- [ ] 实现优化器基类
- [ ] 实现 SGD
- [ ] 实现 Adam
- [ ] 实现学习率调度

**关键文件**:
- `ypytorch/optim/optimizer.py`
- `ypytorch/optim/sgd.py`
- `ypytorch/optim/adam.py`

### Phase 6: 完整示例（Week 13-14）

**目标**: 实现一个完整的训练示例

**任务清单**:
- [ ] 实现简单的线性回归示例
- [ ] 实现简单的分类任务（如 MNIST）
- [ ] 编写使用文档
- [ ] 性能优化

## 学习资源

- PyTorch 官方文档
- 《深度学习框架 PyTorch 入门与实践》
- PyTorch 源码阅读

## 贡献指南

欢迎提交 Issue 和 Pull Request！

