# define a model with different layers
# define a forward functin
# define a backward functin

"""
First test with autograd engine.
"""

import planckgrad as pl
import planckgrad.nn as nn
from planckgrad import Tensor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import planckgrad.nn.functional as F
import numpy as np

X, y = make_regression(n_samples=1000, n_features=2, noise=0, random_state=42)

fig = plt.figure(num=1, clear=True)

ax = plt.figure().add_subplot(projection='3d')
surf = ax.plot_surface(X[:, 0], X[:, 1], y.reshape(-1, 1))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Model()


# train loop

# hyper parameters
epochs = 50
lr = 0.0001
loss_fn = nn.MSELoss()

optim = pl.optim.SGD(model.parameters(), lr)

for e in range(epochs):
    epoch_loss = 0
    for index, feature in enumerate(X):
        optim.zero_grad()
        feature = Tensor.from_numpy(feature).reshape(-1, 2)

        y_true = Tensor.from_numpy(y[index]).reshape(1, 1)

        out = model(feature)
        loss = loss_fn(out, y_true)
        epoch_loss += loss.item()

        loss.backward()
        optim.step()
    
    if e % 5 == 0:
        print(f"epoch {e}, loss: {epoch_loss.data / len(X)}")

predictions = []

for feature in X:
    feature = Tensor.from_numpy(feature).reshape(-1, 2)
    pred = model(feature)
    predictions.append(pred.data.flatten())

surf2 = ax.plot_surface(X[:, 0], X[:, 1], predictions)
plt.show()