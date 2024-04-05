import planckgrad as pl
import torch
import numpy as np

t1 = torch.rand(1, 2, requires_grad=True)
t2 = torch.rand(2, 1, requires_grad=True)

t3 = t1 @ t2
print(t3)