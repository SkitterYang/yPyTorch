"""yPyTorch 基础使用示例"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt
import numpy as np


def example_tensor_creation():
    """示例：创建张量"""
    print("=" * 50)
    print("示例 1: 创建张量")
    print("=" * 50)
    
    # 从列表创建
    t1 = ypt.tensor([1, 2, 3, 4])
    print(f"从列表创建: {t1}")
    print(f"形状: {t1.shape}, 数据类型: {t1.dtype}")
    
    # 从 numpy array 创建
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t2 = ypt.tensor(arr)
    print(f"\n从 numpy array 创建:\n{t2}")
    
    # 使用便捷函数
    zeros = ypt.zeros((2, 3))
    ones = ypt.ones((3, 2))
    randn = ypt.randn(2, 2)
    
    print(f"\nzeros(2, 3):\n{zeros}")
    print(f"\nones(3, 2):\n{ones}")
    print(f"\nrandn(2, 2):\n{randn}")
    print()


def example_tensor_operations():
    """示例：张量运算"""
    print("=" * 50)
    print("示例 2: 张量运算")
    print("=" * 50)
    
    a = ypt.tensor([1.0, 2.0, 3.0])
    b = ypt.tensor([4.0, 5.0, 6.0])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # 加法
    c = a + b
    print(f"\na + b = {c}")
    
    # 乘法
    d = a * b
    print(f"a * b = {d}")
    
    # 减法
    e = b - a
    print(f"b - a = {e}")
    
    # 除法
    f = b / a
    print(f"b / a = {f}")
    
    # 矩阵乘法
    x = ypt.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = ypt.tensor([[5.0, 6.0], [7.0, 8.0]])
    z = x @ y
    print(f"\n矩阵乘法:")
    print(f"x = \n{x}")
    print(f"y = \n{y}")
    print(f"x @ y = \n{z}")
    print()


def example_tensor_reduction():
    """示例：归约操作"""
    print("=" * 50)
    print("示例 3: 归约操作")
    print("=" * 50)
    
    t = ypt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"原始张量:\n{t}")
    
    # 求和
    total = t.sum()
    print(f"\n总和: {total.item()}")
    
    # 求均值
    avg = t.mean()
    print(f"均值: {avg.item()}")
    
    # 沿维度求和
    sum_dim0 = t.sum(dim=0)
    print(f"沿维度 0 求和: {sum_dim0}")
    
    sum_dim1 = t.sum(dim=1)
    print(f"沿维度 1 求和: {sum_dim1}")
    print()


def example_tensor_reshape():
    """示例：形状操作"""
    print("=" * 50)
    print("示例 4: 形状操作")
    print("=" * 50)
    
    t = ypt.tensor([1, 2, 3, 4, 5, 6])
    print(f"原始形状: {t.shape}")
    print(f"数据: {t}")
    
    # reshape
    t2 = t.reshape(2, 3)
    print(f"\nreshape(2, 3):\n{t2}")
    
    # transpose
    t3 = t2.transpose(0, 1)
    print(f"\ntranspose(0, 1):\n{t3}")
    print()


def example_tensor_indexing():
    """示例：索引和切片"""
    print("=" * 50)
    print("示例 5: 索引和切片")
    print("=" * 50)
    
    t = ypt.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始张量:\n{t}")
    
    # 索引访问
    print(f"\nt[0, 0] = {t[0, 0].item()}")
    print(f"t[1, 2] = {t[1, 2].item()}")
    
    # 切片
    row = t[0, :]
    print(f"\nt[0, :] = {row}")
    
    col = t[:, 1]
    print(f"t[:, 1] = {col}")
    print()


if __name__ == '__main__':
    example_tensor_creation()
    example_tensor_operations()
    example_tensor_reduction()
    example_tensor_reshape()
    example_tensor_indexing()
    
    print("=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)

