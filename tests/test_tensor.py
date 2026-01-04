"""Tensor 基础测试"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ypytorch as ypt


def test_tensor_creation():
    """测试张量创建"""
    # 从列表创建
    t1 = ypt.tensor([1, 2, 3, 4])
    assert t1.shape == (4,)
    assert t1.dtype == np.float32
    
    # 从 numpy array 创建
    arr = np.array([1.0, 2.0, 3.0])
    t2 = ypt.tensor(arr)
    assert t2.shape == (3,)
    
    # 创建多维张量
    t3 = ypt.tensor([[1, 2], [3, 4]])
    assert t3.shape == (2, 2)


def test_tensor_operations():
    """测试张量运算"""
    a = ypt.tensor([1.0, 2.0, 3.0])
    b = ypt.tensor([4.0, 5.0, 6.0])
    
    # 加法
    c = a + b
    assert np.allclose(c.data, [5.0, 7.0, 9.0])
    
    # 乘法
    d = a * b
    assert np.allclose(d.data, [4.0, 10.0, 18.0])
    
    # 减法
    e = b - a
    assert np.allclose(e.data, [3.0, 3.0, 3.0])
    
    # 除法
    f = b / a
    assert np.allclose(f.data, [4.0, 2.5, 2.0])


def test_tensor_creation_functions():
    """测试张量创建函数"""
    # zeros
    z = ypt.zeros((3, 4))
    assert z.shape == (3, 4)
    assert np.allclose(z.data, 0.0)
    
    # ones
    o = ypt.ones((2, 3))
    assert o.shape == (2, 3)
    assert np.allclose(o.data, 1.0)
    
    # randn
    r = ypt.randn(3, 4)
    assert r.shape == (3, 4)


def test_tensor_reduction():
    """测试归约操作"""
    t = ypt.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # sum
    s = t.sum()
    assert s.item() == 10.0
    
    # mean
    m = t.mean()
    assert m.item() == 2.5
    
    # sum along dimension
    s0 = t.sum(dim=0)
    assert np.allclose(s0.data, [4.0, 6.0])


def test_tensor_reshape():
    """测试 reshape"""
    t = ypt.tensor([1, 2, 3, 4, 5, 6])
    t2 = t.reshape(2, 3)
    assert t2.shape == (2, 3)


def test_tensor_indexing():
    """测试索引"""
    t = ypt.tensor([[1, 2, 3], [4, 5, 6]])
    
    # 索引访问
    assert t[0, 0].item() == 1
    assert t[1, 2].item() == 6
    
    # 切片
    t_slice = t[0, :]
    assert np.allclose(t_slice.data, [1, 2, 3])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

