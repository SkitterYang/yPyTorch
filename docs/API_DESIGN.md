# yPyTorch API 设计

## 设计原则

1. **与 PyTorch 兼容**: 尽量保持与 PyTorch 相同的 API
2. **简洁明了**: 避免过度设计，保持代码可读性
3. **渐进式**: 先实现核心功能，再逐步扩展

## 核心 API

### Tensor 类

```python
class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, device=None):
        """
        创建张量
        
        Args:
            data: 数据（list, numpy array, 或其他 tensor）
            dtype: 数据类型（float32, int32 等）
            requires_grad: 是否需要梯度
            device: 设备（cpu, cuda，初期只支持 cpu）
        """
        pass
    
    @property
    def shape(self):
        """返回张量的形状"""
        pass
    
    @property
    def dtype(self):
        """返回数据类型"""
        pass
    
    @property
    def grad(self):
        """返回梯度"""
        pass
    
    def backward(self, gradient=None):
        """反向传播"""
        pass
    
    def item(self):
        """返回标量值（仅当 tensor 只有一个元素时）"""
        pass
    
    def numpy(self):
        """转换为 numpy array"""
        pass
```

### 基础操作

```python
# 创建张量
ypt.zeros(shape)
ypt.ones(shape)
ypt.randn(shape)
ypt.tensor(data)

# 数学运算
ypt.add(x, y)  # 或 x + y
ypt.sub(x, y)  # 或 x - y
ypt.mul(x, y)  # 或 x * y
ypt.div(x, y)  # 或 x / y
ypt.matmul(x, y)  # 或 x @ y

# 归约操作
ypt.sum(x, dim=None)
ypt.mean(x, dim=None)
ypt.max(x, dim=None)
ypt.min(x, dim=None)

# 形状操作
ypt.reshape(x, shape)
ypt.transpose(x, dim0, dim1)
ypt.view(x, shape)
```

### 自动求导

```python
# 创建需要梯度的张量
x = ypt.tensor([1.0, 2.0], requires_grad=True)

# 前向传播
y = x * 2
z = y.sum()

# 反向传播
z.backward()

# 访问梯度
print(x.grad)
```

### 神经网络模块

```python
# Module 基类
class Module:
    def forward(self, x):
        """前向传播"""
        pass
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        """返回所有参数"""
        pass

# Linear 层
linear = ypt.nn.Linear(in_features=784, out_features=128)
x = ypt.randn(32, 784)
y = linear(x)  # shape: (32, 128)

# 激活函数
relu = ypt.nn.ReLU()
x = relu(x)

# 损失函数
criterion = ypt.nn.MSELoss()
loss = criterion(pred, target)
```

### 优化器

```python
# SGD
optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(num_epochs):
    # 前向传播
    output = model(input)
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    optimizer.zero_grad()
```

## 与 PyTorch 的差异

1. **简化实现**: 初期只支持 CPU，不支持 CUDA
2. **有限操作**: 只实现最常用的操作
3. **性能**: 不追求极致性能，以可读性为主
4. **功能**: 不实现分布式训练、JIT 编译等高级功能

## 扩展计划

- [ ] 支持更多数据类型
- [ ] 支持更多操作（卷积、池化等）
- [ ] 支持批量归一化
- [ ] 支持 Dropout
- [ ] 支持模型保存和加载

