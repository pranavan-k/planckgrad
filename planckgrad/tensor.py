import numpy as np

class Tensor:
    def __init__(self, array):
        self.data = np.array(array)
        self.grad = None
    
    @property
    def dtype(self): return self.data.dtype
    
    @property
    def shape(self): return self.data.shape
    
    def __repr__(self) -> str:
        return f"tensor({self.data})"
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data + other)
        else:
            raise TypeError("Unsupported operand type(s) for +")

    def __radd__(self, other):
      return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data - other)
        else:
            raise TypeError("Unsupported operand type(s) for +")

    def __rsub__(self, other):
      return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data * other)
        else:
            raise TypeError("Unsupported operand type(s) for +")

    def __rmul__(self, other):
      return self.__mul__(other)
    
    def __div__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data / other)
        else:
            raise TypeError("Unsupported operand type(s) for +")

    def __rdiv__(self, other):
      return self.__div__(other)
    
    def __pow__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data ** other.data)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data ** other)
        else:
            raise TypeError("Unsupported operand type(s) for +")

    def __rpow__(self, other):
      return self.__pow__(other)
    
    def __matmul__(self, other):
        data = self.data @ other.data
        return Tensor(data)
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __getitem__(self, key):
        return self.data[key]
    
    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape))
    
    def backward():
        pass
    
    @property
    def T(self): return Tensor(self.data.T)
    
    @staticmethod
    def from_numpy(array):
        return Tensor(array.data)
    
    @staticmethod
    def rand(*shape):
        data = np.random.rand(*shape)
        return Tensor(data)
    
    @staticmethod
    def zeros(*shape):
        data = np.zeros(shape)
        return Tensor(data)
    
    @staticmethod
    def ones(*shape):
        data = np.ones(shape)
        return Tensor(data)
    
    @staticmethod
    def linspace(start, stop, num):
        data = np.linspace(start, stop, num)
        return Tensor(data)
    
    @staticmethod
    def arange(start, stop, step):
        data = np.arange(start, stop, step)
        return Tensor(data)
    
    @staticmethod
    def randn(*shape):
        return Tensor(np.random.randn(*shape))