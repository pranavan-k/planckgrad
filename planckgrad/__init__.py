from planckgrad.tensor import Tensor
import numpy as np
from planckgrad.tensor import Dependency
import planckgrad.optim as optim


def mean(tensor: Tensor) -> float:
    data = np.mean(tensor.data)
    requires_grad = tensor.requires_grad

    depends_on = []

    if tensor.requires_grad:
        def grad_fn(grad):
            data = (grad.data.size + 1) / 2
            return Tensor(data)

        depends_on.append(Dependency(tensor, grad_fn))


    return Tensor(data, requires_grad, dependencies=depends_on)

def sum(tensor: Tensor) -> float:
    return np.sum(tensor.data)