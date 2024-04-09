from planckgrad.tensor import Tensor
import numpy as np
from planckgrad.nn.parameter import Parameter
from planckgrad.tensor import Dependency

def sigmoid(x):
    data = 1 / (1 + np.exp(-x.data))
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: Tensor) -> Tensor:
            return Tensor(grad.data * data * (1 - data))

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def tanh(x):
    data = np.tanh(x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: Tensor) -> Tensor:
            return Tensor(grad.data * (1 - data * data))

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def relu(x):
    data = np.maximum(0, x.data)
    requires_grad = x.requires_grad

    if requires_grad:
        def grad_fn(grad: Tensor) -> Tensor:
            grad.data[grad.data<=0] = 0
            grad.data[grad.data>0] = 1
            return grad * Tensor(grad.data)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)