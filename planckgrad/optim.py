class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.learning_rate = lr
    
    def step(self):
        for parameter in self.parameters:
            parameter -= self.learning_rate * parameter.grad
    
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()