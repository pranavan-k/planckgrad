import numpy as np
import planckgrad as pl
from planckgrad.tensor import Tensor

class Linear:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.rand(in_features, out_features)
        self.bias = Tensor.randn(out_features) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if self.bias is None:
            return x @ self.weights

        return x @ self.weights + self.bias


class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
    
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_pred.shape[0]

        return self

    def item(self):
        if self.reduction == "mean":
            return pl.mean((self.y_pred - self.y_true) ** 2)
        elif self.reduction == "sum":
            return pl.sum((self.y_pred - self.y_true) ** 2)