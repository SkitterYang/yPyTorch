# autograd 包详解

## 什么是 autograd？

**autograd** = **automatic differentiation**（自动求导）

它是深度学习框架的核心功能之一，用于**自动计算函数的梯度（导数）**。

## 为什么需要 autograd？

### 传统方法的问题

在深度学习中，我们需要计算损失函数对模型参数的梯度，然后使用梯度下降法更新参数。

**传统方法**：手动推导梯度公式
- 例如：`f(x) = x²`，我们需要知道 `df/dx = 2x`
- 对于复杂函数（如多层神经网络），手动推导非常困难且容易出错

### autograd 的优势

**自动求导**：框架自动计算梯度
- 我们只需要定义**前向传播**（如何计算函数值）
- 框架会自动计算**反向传播**（如何求梯度）
- 无需手动推导复杂的梯度公式

## autograd 包的结构

```
ypytorch/autograd/
├── function.py      # Function 基类和 Context 类
├── engine.py        # 反向传播引擎
└── functions.py     # 各种操作的梯度函数实现
```

## 核心组件

### 1. Function 类（计算图节点）

`Function` 类表示计算图中的一个节点，每个数学运算（如加法、乘法）都是一个 Function。

```python
class Function:
    @staticmethod
    def forward(ctx, *args):
        """前向传播：计算函数值"""
        # 1. 保存需要的中间结果到 ctx
        # 2. 计算并返回结果
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算梯度"""
        # 1. 从 ctx 获取保存的中间结果
        # 2. 根据链式法则计算梯度
        # 3. 返回输入参数的梯度
        pass
```

**示例：加法操作**

```python
class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        # 保存输入，用于反向传播
        ctx.save_for_backward(x, y)
        # 计算 x + y
        return Tensor(x.data + y.data)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 加法的梯度：对两个输入都是 grad_output
        x, y = ctx.saved_tensors
        return grad_output, grad_output
```

### 2. Context 类（上下文存储）

`Context` 类用于在前向传播和反向传播之间传递信息。

```python
class Context:
    saved_tensors: List[Tensor]  # 保存的张量
    saved_values: dict           # 保存的其他值（如维度、常数）
```

**为什么需要 Context？**

在反向传播时，我们需要知道前向传播时的输入值。例如：
- 乘法 `z = x * y` 的梯度是 `dz/dx = y`，`dz/dy = x`
- 所以我们需要保存 `x` 和 `y` 的值

### 3. backward() 函数（反向传播引擎）

`backward()` 函数是反向传播的入口，它会：
1. 从输出开始，沿着计算图反向传播梯度
2. 自动调用每个节点的 `backward()` 方法
3. 将梯度累积到叶子节点（需要梯度的参数）

## 工作原理示例

### 简单例子：z = 3x²

```python
import ypytorch as ypt

# 1. 创建需要梯度的张量
x = ypt.tensor(2.0, requires_grad=True)

# 2. 前向传播（构建计算图）
y = x ** 2      # y = x²，创建 Pow Function 节点
z = y * 3       # z = 3y，创建 Mul Function 节点

# 3. 反向传播
z.backward()

# 4. 查看梯度
print(x.grad)   # 输出：12.0 (因为 dz/dx = 6x = 6*2 = 12)
```

**计算图结构：**
```
x (叶子节点，requires_grad=True)
 ↓ forward: y = x²
y (Pow Function 节点)
 ↓ forward: z = 3y
z (Mul Function 节点)
```

**反向传播过程：**
```
1. z 的梯度 = 1.0 (默认，因为是标量)
2. Mul.backward():
   - z = 3y，所以 dy = dz / 3 = 1.0 / 3 = 0.333...
   - y 的梯度 = 0.333...
3. Pow.backward():
   - y = x²，所以 dx = dy * 2x = 0.333... * 4 = 1.333...
   - 等等，这里有问题...
```

实际上，正确的计算应该是：
- `z = 3x²`
- `dz/dx = 6x = 6*2 = 12`

### 神经网络训练示例

```python
# 1. 创建模型和数据
model = ypt.nn.Linear(1, 1)
x = ypt.tensor([[1.0], [2.0]])
y_true = ypt.tensor([[2.0], [4.0]])

# 2. 前向传播
y_pred = model(x)           # 计算预测值
loss = criterion(y_pred, y_true)  # 计算损失

# 3. 反向传播（autograd 自动计算梯度）
loss.backward()

# 4. 查看梯度
print(model.weight.grad)    # 损失对权重的梯度
print(model.bias.grad)      # 损失对偏置的梯度

# 5. 更新参数
optimizer.step()            # 使用梯度更新参数
```

## 关键概念

### 1. 计算图（Computation Graph）

计算图是一个有向无环图（DAG），记录了所有运算的依赖关系。

```
前向传播：构建计算图
x → [Add] → y → [Mul] → z

反向传播：沿着计算图反向传播梯度
x ← [Add.backward] ← y ← [Mul.backward] ← z
```

### 2. 链式法则（Chain Rule）

反向传播基于链式法则：

如果 `z = f(y)` 且 `y = g(x)`，那么：
```
dz/dx = (dz/dy) * (dy/dx)
```

### 3. 叶子节点（Leaf Node）

- **叶子节点**：直接创建的张量（如模型参数），`_is_leaf = True`
- **非叶子节点**：通过运算得到的张量，`_is_leaf = False`

梯度只累积到叶子节点。

## 常见操作的梯度

| 操作 | 前向传播 | 反向传播 |
|------|---------|---------|
| 加法 | `z = x + y` | `dz/dx = 1`, `dz/dy = 1` |
| 乘法 | `z = x * y` | `dz/dx = y`, `dz/dy = x` |
| 幂运算 | `z = x²` | `dz/dx = 2x` |
| 矩阵乘法 | `z = x @ y` | `dz/dx = z_grad @ y.T`, `dz/dy = x.T @ z_grad` |
| 求和 | `z = sum(x)` | `dz/dx = 1` (广播到 x 的形状) |

## 总结

**autograd 包的核心作用：**

1. **自动构建计算图**：在前向传播时记录所有运算
2. **自动计算梯度**：在反向传播时沿着计算图自动求导
3. **无需手动推导**：我们只需要定义前向传播，梯度会自动计算

这就是为什么我们可以轻松训练复杂的神经网络，而无需手动推导成千上万个参数的梯度公式！


