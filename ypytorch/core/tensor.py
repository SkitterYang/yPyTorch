"""核心 Tensor 类实现"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
from .storage import Storage
from .dtype import get_dtype, dtype_to_string


class Tensor:
    """
    张量类，是 yPyTorch 的核心数据结构
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, Tuple, int, float, 'Tensor'],
        dtype=None,
        requires_grad: bool = False,
        device: Optional[str] = None
    ):
        """
        初始化张量
        
        Args:
            data: 数据，可以是 numpy array、list、tuple、标量或其他 Tensor
            dtype: 数据类型，默认为 float32
            requires_grad: 是否需要梯度，用于自动求导
            device: 设备（'cpu' 或 'cuda'），目前只支持 'cpu'
        """
        # 处理设备（目前只支持 CPU）
        if device is None:
            device = 'cpu'
        if device != 'cpu':
            raise NotImplementedError(f"Device {device} is not supported yet")
        self.device = device
        
        # 处理数据类型
        if isinstance(data, Tensor):
            # 如果输入是 Tensor，复制其数据
            self._storage = Storage(data._storage.data.copy())
            dtype = dtype or data.dtype
        else:
            # 创建新的存储
            self._storage = Storage(data)
            if dtype is not None:
                self._storage._data = self._storage._data.astype(get_dtype(dtype))
        
        # 设置属性
        self.requires_grad = requires_grad
        self._grad = None
        
        # 用于自动求导
        self._grad_fn = None
        self._ctx = None
        self._is_leaf = True
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """返回张量的形状"""
        return self._storage.shape
    
    @property
    def dtype(self):
        """返回数据类型"""
        return self._storage.dtype
    
    @property
    def grad(self) -> Optional['Tensor']:
        """返回梯度"""
        return self._grad
    
    @grad.setter
    def grad(self, value: Optional['Tensor']):
        """设置梯度"""
        self._grad = value
    
    @property
    def data(self) -> np.ndarray:
        """返回底层数据（numpy array）"""
        return self._storage.data
    
    def backward(self, gradient: Optional['Tensor'] = None):
        """
        反向传播
        
        Args:
            gradient: 上游梯度，如果是标量张量则默认为 1.0
        """
        from ..autograd.engine import backward
        backward(self, gradient)
    
    def item(self) -> Union[int, float]:
        """
        返回标量值（仅当张量只有一个元素时）
        
        Returns:
            标量值
        """
        if self.size != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return self._storage.data.item()
    
    @property
    def size(self) -> int:
        """返回元素总数"""
        return self._storage.size
    
    def numpy(self) -> np.ndarray:
        """转换为 numpy array"""
        return self._storage.numpy()
    
    def __getitem__(self, key):
        """索引访问"""
        return Tensor(self._storage[key], requires_grad=self.requires_grad)
    
    def __setitem__(self, key, value):
        """索引赋值"""
        if isinstance(value, Tensor):
            self._storage[key] = value._storage.data
        else:
            self._storage[key] = value
    
    def __len__(self) -> int:
        """返回第一维的长度"""
        return len(self._storage)
    
    def __repr__(self) -> str:
        """字符串表示"""
        dtype_str = dtype_to_string(self.dtype)
        grad_str = ", grad_fn=<...>" if self._grad_fn is not None else ""
        return f"Tensor({self._storage.data}, dtype={dtype_str}, requires_grad={self.requires_grad}{grad_str})"
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.__repr__()
    
    # 数学运算（将在 ops 模块中实现）
    def __add__(self, other):
        from ..ops import math
        return math.add(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        from ..ops import math
        return math.sub(self, other)
    
    def __rsub__(self, other):
        from ..ops import math
        return math.sub(other, self)
    
    def __mul__(self, other):
        from ..ops import math
        return math.mul(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        from ..ops import math
        return math.div(self, other)
    
    def __rtruediv__(self, other):
        from ..ops import math
        return math.div(other, self)
    
    def __matmul__(self, other):
        from ..ops import math
        return math.matmul(self, other)
    
    def __neg__(self):
        from ..ops import math
        return math.neg(self)
    
    def __pow__(self, power):
        from ..ops import math
        return math.pow(self, power)
    
    def reshape(self, *shape):
        """重塑形状"""
        return Tensor(self._storage.data.reshape(*shape), requires_grad=self.requires_grad)
    
    def transpose(self, dim0: int = 0, dim1: int = 1):
        """转置"""
        axes = list(range(len(self.shape)))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(self._storage.data.transpose(axes), requires_grad=self.requires_grad)
    
    def sum(self, dim=None, keepdim=False):
        """求和"""
        from ..ops import reduction
        return reduction.sum(self, dim=dim, keepdim=keepdim)
    
    def mean(self, dim=None, keepdim=False):
        """求均值"""
        from ..ops import reduction
        return reduction.mean(self, dim=dim, keepdim=keepdim)
    
    def max(self, dim=None, keepdim=False):
        """求最大值"""
        from ..ops import reduction
        return reduction.max(self, dim=dim, keepdim=keepdim)
    
    def min(self, dim=None, keepdim=False):
        """求最小值"""
        from ..ops import reduction
        return reduction.min(self, dim=dim, keepdim=keepdim)

