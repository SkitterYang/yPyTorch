"""底层存储实现"""

import numpy as np
from typing import Union, List, Tuple

class Storage:
    """
    底层存储类，使用 NumPy 数组作为后端
    """
    
    def __init__(self, data: Union[np.ndarray, List, Tuple, int, float, np.generic]):
        """
        初始化存储
        
        Args:
            data: 数据，可以是 numpy array、list、tuple、标量或 numpy 标量类型
        """
        if isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, np.generic):
            # 处理 numpy 标量类型（如 np.float32, np.int32 等）
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=np.float32)
        elif isinstance(data, (int, float)):
            # 标量应该保持为0维数组（形状为 ()）
            self._data = np.array(data, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @property
    def data(self) -> np.ndarray:
        """返回底层 numpy 数组"""
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """返回形状"""
        return self._data.shape
    
    @property
    def dtype(self):
        """返回数据类型"""
        return self._data.dtype
    
    @property
    def size(self) -> int:
        """返回元素总数"""
        return self._data.size
    
    def __getitem__(self, key):
        """索引访问"""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """索引赋值"""
        self._data[key] = value
    
    def __len__(self) -> int:
        """返回第一维的长度"""
        return len(self._data)
    
    def reshape(self, *shape):
        """重塑形状"""
        return Storage(self._data.reshape(*shape))
    
    def copy(self):
        """深拷贝"""
        return Storage(self._data.copy())
    
    def numpy(self) -> np.ndarray:
        """转换为 numpy array"""
        return self._data.copy()

