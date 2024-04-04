from planckgrad.tensor import Tensor
import numpy as np


def mean(tensor: Tensor) -> float:
    return np.mean(tensor.data)

def sum(tensor: Tensor) -> float:
    return np.sum(tensor.data)