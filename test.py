import planckgrad.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100).reshape(-1, 1)
weight = np.array([0.78812])
bias = np.array([0.18812])
y = x * weight + bias

plt.plot(x, y)

fc1 = nn.Linear(1, 1)
loss_fn = nn.MSELoss()

lr = 0.0001
# fc1.weights = fc1.weights - lr * x.T * 2*(out - y)

# out = fc1(x)
# loss = loss_fn(out, y)
# print(fc1.weights, "error", loss.item())

for e in range(15):
    epoch_loss = 0
    for i, feature in enumerate(x):
        out = fc1(feature)

        loss = loss_fn(out, y[i])
        epoch_loss += loss.item()

        fc1.weights = fc1.weights - lr * feature.T * 2*(out - y[i])
        fc1.bias = fc1.bias - lr * 2*(out - y[i])

    if e % 1 == 0:
        print(f"epoch: {e}, loss: {epoch_loss / x.shape[0]}")

predictions = x * fc1.weights + fc1.bias

print(fc1.weights, weight)

plt.plot(x, predictions)
plt.show()