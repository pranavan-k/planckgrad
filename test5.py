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

X, y = make_regression(n_samples=1000, n_features=1, noise=1, random_state=42)

plt.scatter(X, y, c="b")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

model = Model()


# train loop

# hyper parameters
epochs = 50
lr = 0.001
loss_fn = nn.MSELoss()

optim = pl.optim.SGD(model.parameters(), lr)

for e in range(epochs):
    epoch_loss = 0
    for index, feature in enumerate(X):
        optim.zero_grad()
        feature = Tensor.from_numpy(feature).reshape(-1, 1)

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
    feature = Tensor.from_numpy(feature).reshape(-1, 1)
    pred = model(feature)
    predictions.append(pred.data.flatten())

plt.scatter(X, predictions, c="r")
plt.show()