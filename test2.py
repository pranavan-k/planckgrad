import planckgrad as pl
from planckgrad.tensor import Tensor
import planckgrad.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from matplotlib import cm

def fun(t, f): return (np.cos(f + 2 * t) + 1) / 2
(theta, phi) = np.meshgrid(np.linspace(0, 2 * np.pi, 41),
                           np.linspace(0, 2 * np.pi, 41))

X, y = make_regression(n_samples=1000, n_features=2, noise=20, random_state=42)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

dplot = ax.plot_surface(X[:, 0], X[:, 0], y.reshape(-1, 1))
ax.set(xlabel='x', 
       ylabel='y', 
       zlabel='z')

fc1 = nn.Linear(2, 1)

lr = 0.01
epochs = 35
loss_fn = nn.MSELoss()

for e in range(epochs):
    epoch_loss = 0
    for i, feature in enumerate(X):
        feature = Tensor.from_numpy(feature).reshape(1, 2)
        true = Tensor.from_numpy(y[i]).reshape(1, 1)

        out = fc1(feature)
        loss = loss_fn(out, true)
        
        epoch_loss += loss.item()

        # backprop
        fc1.weights = fc1.weights - lr * feature.T * 2*(out - true)
        fc1.bias = fc1.bias - lr * 2*(out - true)
    
    print(f"epoch: {e}, loss: {epoch_loss / len(X)}")

predictions = fc1(Tensor(X))
dplot = ax.plot_surface(X[:, 0], X[:, 0], predictions.reshape(-1, 1).data)

plt.show()