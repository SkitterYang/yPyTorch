"""自动求导模块"""

from .function import Function
from .engine import backward

__all__ = ['Function', 'backward']

