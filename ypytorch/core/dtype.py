"""数据类型定义"""

import numpy as np

# 数据类型映射
DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool_,
}

# 反向映射
NP_TO_DTYPE = {v: k for k, v in DTYPE_MAP.items()}

def get_dtype(dtype):
    """
    获取数据类型
    
    Args:
        dtype: 可以是字符串 ('float32') 或 numpy dtype
        
    Returns:
        numpy dtype
    """
    if dtype is None:
        return np.float32
    
    if isinstance(dtype, str):
        return DTYPE_MAP.get(dtype, np.float32)
    
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return dtype
    
    return dtype

def dtype_to_string(dtype):
    """将 numpy dtype 转换为字符串"""
    if dtype is None:
        return 'float32'
    return NP_TO_DTYPE.get(dtype, str(dtype))

# 常用数据类型
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
bool_ = np.bool_

__all__ = [
    'get_dtype',
    'dtype_to_string',
    'float32',
    'float64',
    'int32',
    'int64',
    'bool_',
]

