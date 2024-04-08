import planckgrad as pl
from planckgrad.tensor import Tensor
import planckgrad.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from matplotlib import cm
from planckgrad.optim import SGD

x = Tensor([[4.0]])

parameteres = [
    Tensor([[2.0]], requires_grad=True),
    Tensor([[1.0]], requires_grad=True)
]

y_pred = x @ parameteres[0] + parameteres[1]
y_true = Tensor([[2.0]], requires_grad=True)

loss_fn = nn.MSELoss()

loss = loss_fn(y_pred, y_true)
optim = SGD(lr=0.01)

loss.backward()
print(y_pred, parameteres[0], parameteres[1])
optim.step(parameteres)

print(y_pred, parameteres[0], parameteres[1])

# torchx = torch.tensor([4.0])

# torch_weight = torch.tensor([2.0], requires_grad=True)
# torch_bias = torch.tensor([1.0], requires_grad=True)

# torch_pred = torchx @ torch_weight + torch_bias
# torch_true = torch.tensor([2.0], requires_grad=True)

# torch_loss = torch.mean((torch_pred - torch_true) ** 2)
# torch_loss.backward()

# print(torch_pred.grad, torch_weight.grad, torch_bias.grad)