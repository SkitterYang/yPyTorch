"""Module 基类"""

from typing import Iterator, Dict
from ..core.tensor import Tensor
import ypytorch as ypt


class Module:
    """
    所有神经网络模块的基类
    
    每个模块都应该继承这个类，并实现 forward 方法
    """
    
    def __init__(self):
        """初始化模块"""
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            输出张量
        """
        raise NotImplementedError(f"Module {self.__class__.__name__} must implement forward method")
    
    def __call__(self, x: Tensor) -> Tensor:
        """使模块可以像函数一样调用"""
        return self.forward(x)
    
    def parameters(self) -> Iterator[Tensor]:
        """
        返回所有需要训练的参数
        
        Returns:
            参数迭代器
        """
        for name, param in self._parameters.items():
            if param is not None:
                yield param
        
        # 递归获取子模块的参数
        for module in self._modules.values():
            if module is not None:
                yield from module.parameters()
    
    def named_parameters(self) -> Iterator[tuple]:
        """
        返回所有参数及其名称
        
        Returns:
            (name, param) 迭代器
        """
        for name, param in self._parameters.items():
            if param is not None:
                yield name, param
        
        # 递归获取子模块的参数
        for module_name, module in self._modules.items():
            if module is not None:
                for name, param in module.named_parameters():
                    yield f"{module_name}.{name}", param
    
    def register_parameter(self, name: str, param: Tensor):
        """
        注册一个参数
        
        Args:
            name: 参数名称
            param: 参数张量
        """
        self._parameters[name] = param
    
    def add_module(self, name: str, module: 'Module'):
        """
        添加一个子模块
        
        Args:
            name: 模块名称
            module: 模块实例
        """
        self._modules[name] = module
    
    def zero_grad(self):
        """将所有参数的梯度清零"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None
    
    def train(self, mode: bool = True):
        """
        设置训练模式
        
        Args:
            mode: True 表示训练模式，False 表示评估模式
        """
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
    
    def eval(self):
        """设置为评估模式"""
        self.train(False)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}()"

