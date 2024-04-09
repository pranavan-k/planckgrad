import numpy as np
import planckgrad as pl
from planckgrad.tensor import Tensor
import inspect
from planckgrad.nn.parameter import Parameter
import planckgrad.nn.functional

class Module:
    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module): # allows for nested modules
                yield from value.parameters()
    
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

# layers

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(in_features, out_features)
        self.bias = Parameter(out_features) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if self.bias is None:
            return x @ self.weights

        return x @ self.weights + self.bias

# activation layers

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x: Tensor) -> Tensor:
        data = 1 / (1 + np.exp(-x.data))
        return Parameter(None, data=data)

# loss functions

class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
    
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_pred.shape[0]
        self.item()

        return self

    def item(self):
        if self.reduction == "mean":
            self.result = pl.mean((self.y_pred - self.y_true) ** 2)
            return self.result
        elif self.reduction == "sum":
            self.result = pl.sum((self.y_pred - self.y_true) ** 2)
            return self.result()
    
    def backward(self):
        self.result.backward()