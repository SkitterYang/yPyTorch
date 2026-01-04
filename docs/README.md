# yPyTorch 文档

欢迎来到 yPyTorch 文档！这是一个简易版的 PyTorch 实现，用于学习深度学习框架的核心原理。

## 文档导航

- [架构设计](./ARCHITECTURE.md) - 项目整体架构和模块划分
- [开发路线图](./ROADMAP.md) - 详细的开发计划和阶段划分
- [API 设计](./API_DESIGN.md) - API 设计规范和与 PyTorch 的对比

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/yPyTorch.git
cd yPyTorch

# 安装依赖
pip install -r requirements.txt
```

### 第一个示例

```python
import ypytorch as ypt

# 创建张量
x = ypt.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # [2.0, 2.0, 2.0]
```

## 项目目标

1. **学习**: 深入理解深度学习框架的工作原理
2. **实践**: 通过实现来掌握核心概念
3. **简化**: 保持代码简洁，便于理解和修改

## 贡献

欢迎贡献代码、文档或提出建议！

## 许可证

本项目遵循 MIT 许可证。

