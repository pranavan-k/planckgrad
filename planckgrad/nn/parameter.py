from planckgrad.tensor import Tensor
import numpy as np

class Parameter(Tensor):
    def __init__(self, *shape, data=None, require_grad=True):
        if data == None:
            data = np.random.rand(*shape)
            
        super().__init__(data, require_grad)