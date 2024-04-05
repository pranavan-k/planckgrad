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

X, y = make_regression(n_samples=1000, n_features=2, noise=0, random_state=42)

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')

dplot = ax.plot_surface(X[:, 0], X[:, 0], y.reshape(-1, 1))
ax.set(xlabel='x', 
       ylabel='y', 
       zlabel='z')

fc1 = nn.Linear(2, 2, bias=False)
fc2 = nn.Linear(2, 1)

lr = 1e-4
epochs = 110
loss_fn = nn.MSELoss()

def train(epochs, lr, loss_fn):
    for e in range(epochs):
        epoch_loss = 0
        for i, feature in enumerate(X):
            feature = Tensor.from_numpy(feature).reshape(1, 2)
            true = Tensor.from_numpy(y[i]).reshape(1, 1)

            out1 = fc1(feature)
            out2 = fc2(out1)
            loss = loss_fn(out2, true)
            
            epoch_loss += loss.item()

            # backprop
            fc2.weights = fc2.weights - lr * out1.T * (out2 - true)
            fc2.bias = fc2.bias - lr * 2*(out2 - true)
            fc1.weights = fc1.weights - lr * fc1.weights.T @ out1.T * (out2 - true)
    
        if e % 9 == 0:
            print(f"epoch: {e}, loss: {epoch_loss / len(X)}")

train(epochs, lr, loss_fn)

predictions = fc2(fc1(Tensor(X)))
dplot = ax.plot_surface(X[:, 0], X[:, 0], predictions.reshape(-1, 1).data)

plt.show()