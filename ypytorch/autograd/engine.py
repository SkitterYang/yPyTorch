"""反向传播引擎"""

from typing import List, Set
from ..core.tensor import Tensor
from .function import Function


def backward(tensor: Tensor, gradient: Tensor = None):
    """
    反向传播引擎
    
    Args:
        tensor: 需要反向传播的张量
        gradient: 上游梯度，如果是标量张量则默认为 1.0
    """
    if not tensor.requires_grad:
        raise RuntimeError("Cannot call backward on a tensor that doesn't require grad")
    
    # 如果是标量，默认梯度为 1.0
    if gradient is None:
        if tensor.shape != ():
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        import numpy as np
        gradient = Tensor(np.array(1.0))
    
    # 初始化梯度
    if tensor._grad is None:
        tensor._grad = gradient
    else:
        # 梯度累积
        tensor._grad = tensor._grad + gradient
    
    # 如果没有梯度函数，说明是叶子节点，停止
    if tensor._grad_fn is None:
        return
    
    # 执行反向传播
    _backward_impl(tensor, gradient)


def _backward_impl(tensor: Tensor, grad_output: Tensor):
    """
    反向传播实现
    
    Args:
        tensor: 当前张量
        grad_output: 上游梯度
    """
    # 获取梯度函数和上下文
    grad_fn = tensor._grad_fn
    ctx = tensor._ctx
    
    if grad_fn is None or ctx is None:
        return
    
    # 调用 backward 方法
    try:
        grad_inputs = grad_fn.backward(ctx, grad_output)
    except NotImplementedError:
        # 如果没有实现 backward，跳过
        return
    
    # 处理梯度输入
    if grad_inputs is None:
        return
    
    # 确保 grad_inputs 是元组或列表
    if not isinstance(grad_inputs, (tuple, list)):
        grad_inputs = (grad_inputs,)
    
    # 获取保存的输入张量
    saved_tensors = ctx.saved_tensors
    
    # 将梯度传播到输入
    for i, grad_input in enumerate(grad_inputs):
        if grad_input is None:
            continue
        
        if i < len(saved_tensors):
            input_tensor = saved_tensors[i]
            
            # 如果输入需要梯度
            if input_tensor.requires_grad:
                # 初始化或累积梯度
                if input_tensor._grad is None:
                    input_tensor._grad = grad_input
                else:
                    input_tensor._grad = input_tensor._grad + grad_input
                
                # 递归反向传播（使用累积后的梯度）
                if input_tensor._grad_fn is not None and input_tensor._is_leaf == False:
                    _backward_impl(input_tensor, input_tensor._grad)

