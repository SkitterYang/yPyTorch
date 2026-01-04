# yPyTorch

简易版 PyTorch 实现，用于学习深度学习框架的核心原理。

## 项目状态

✅ **Phase 1 完成**: 基础 Tensor 实现
- [x] 核心 Tensor 类
- [x] 基础数据类型支持
- [x] 底层存储实现
- [x] 基础数学运算
- [x] 归约操作
- [x] 形状操作（reshape, transpose）
- [x] 索引和切片

✅ **Phase 2 完成**: 自动求导系统
- [x] Function 基类（计算图节点）
- [x] 反向传播引擎
- [x] 常用操作的梯度函数（add, mul, sub, div, matmul, sum, exp, log, pow 等）
- [x] 链式法则支持
- [x] 梯度累积

✅ **Phase 3 完成**: 神经网络模块
- [x] Module 基类
- [x] Linear 层（全连接层）
- [x] 激活函数（ReLU, Sigmoid, Tanh）
- [x] 损失函数（MSE, CrossEntropy）
- [x] 参数初始化（Xavier, Normal）
- [x] 参数管理和梯度清零

✅ **Phase 4 完成**: 优化器
- [x] Optimizer 基类
- [x] SGD 优化器（支持动量和权重衰减）
- [x] Adam 优化器（自适应学习率）
- [x] 状态管理和参数更新

🚧 **进行中**: Phase 5 - 完整训练示例

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/yPyTorch.git
cd yPyTorch

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
import ypytorch as ypt

# 创建张量
x = ypt.tensor([1.0, 2.0, 3.0])
y = ypt.tensor([4.0, 5.0, 6.0])

# 张量运算
z = x + y
print(z)  # Tensor([5.0, 7.0, 9.0])

# 矩阵运算
a = ypt.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ypt.tensor([[5.0, 6.0], [7.0, 8.0]])
c = a @ b
print(c)

# 归约操作
t = ypt.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.sum())  # 10.0
print(t.mean())  # 2.5

# 自动求导
x = ypt.tensor([2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()
print(x.grad)  # [2.0, 2.0]

# 完整训练流程
model = ypt.nn.Linear(2, 1)
criterion = ypt.nn.MSELoss()
optimizer = ypt.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 运行示例

```bash
python examples/basic_usage.py
```

### 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
ypytorch/
├── core/              # 核心功能
│   ├── tensor.py     # Tensor 类
│   ├── storage.py    # 底层存储
│   └── dtype.py      # 数据类型
├── ops/              # 操作符
│   ├── math.py       # 数学运算
│   └── reduction.py  # 归约操作
├── autograd/         # 自动求导（开发中）
├── nn/               # 神经网络（计划中）
└── optim/            # 优化器（计划中）

docs/                 # 文档
examples/             # 示例代码
tests/                # 测试文件
```

## 文档

详细文档请查看 [docs/](./docs/) 目录：

- [架构设计](./docs/ARCHITECTURE.md) - 项目整体架构
- [开发路线图](./docs/ROADMAP.md) - 详细的开发计划
- [API 设计](./docs/API_DESIGN.md) - API 设计规范

## 开发计划

- [x] Phase 1: 基础 Tensor 实现
- [x] Phase 2: 自动求导系统
- [x] Phase 3: 神经网络模块
- [x] Phase 4: 优化器
- [x] Phase 5: 完整训练示例

## 学习目标

通过实现 yPyTorch，你将学习到：

1. **张量的底层实现** - 理解多维数组的存储和操作
2. **自动求导原理** - 理解反向传播和计算图
3. **神经网络构建** - 理解层、激活函数、损失函数
4. **优化算法** - 理解 SGD、Adam 等优化器

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
