import numpy as np

class Dependency:
    def __init__(self, t, grad_fn):
        self.tensor = t
        self.grad_fn = grad_fn

class Tensor:
    def __init__(self, array, requires_grad=False, dependencies=[]):
        self.data = np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.dependency = dependencies
        self.ndim = self.data.ndim
        
        if self.requires_grad:
            self.zero_grad()
    
    @property
    def dtype(self): return self.data.dtype
    
    @property
    def shape(self): return self.data.shape

    @property
    def T(self): return Tensor(self.data.T)
    
    def __repr__(self) -> str:
        return f"tensor({self.data})"
    

    # operations

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def add(self, other):
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependency_list = []

        if self.requires_grad:
            def grad_fn1(grad):
                ndim_added = grad.ndim - self.data.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                return grad

            dependency_list.append(Dependency(self, grad_fn1))
        
        if other.requires_grad:
            def grad_fn2(grad: Tensor): 
                ndim_added = grad.ndim - other.data.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                return grad

            dependency_list.append(Dependency(other, grad_fn2))

        return Tensor(data, requires_grad, dependency_list)
    
    def mul(self, other):
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependency_list = []

        if self.requires_grad:
            def grad_fn1(grad):
                grad = Tensor(grad.data * other.data)

                ndim_added = grad.ndim - self.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                # Sum across broadcasted (but non-added dims)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad.data = grad.data.sum(axis=i, keepdims=True)

                return grad

            dependency_list.append(Dependency(self, grad_fn1))
        
        if other.requires_grad:
            def grad_fn2(grad): 
                grad = Tensor(grad.data * self.data)

                ndim_added = grad.ndim - other.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)
                
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad.data = grad.data.sum(axis=i, keepdims=True)

                return grad

            dependency_list.append(Dependency(other, grad_fn2))

        return Tensor(data, requires_grad, dependency_list)

    def dot(self, other):
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependency_list = []

        if self.requires_grad:
            def grad_fn1(grad):
                return Tensor(grad.data @ other.data.T)

            dependency_list.append(Dependency(self, grad_fn1))
        
        if other.requires_grad:
            def grad_fn2(grad): 
                return Tensor(self.data.T @ grad.data)

            dependency_list.append(Dependency(other, grad_fn2))

        return Tensor(data, requires_grad, dependency_list)
    
    def pow(self, other):
        data = self.data ** other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependency_list = []

        if self.requires_grad:
            def grad_fn1(grad):
                grad = Tensor(other.data * self.data**(other.data - 1))

                ndim_added = grad.ndim - self.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                # Sum across broadcasted (but non-added dims)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad.data = grad.data.sum(axis=i, keepdims=True)

                return grad

            dependency_list.append(Dependency(self, grad_fn1))
        
        if other.requires_grad:
            def grad_fn2(grad): 
                grad = Tensor(self.data ** other.data * np.log(self.data))

                ndim_added = grad.ndim - other.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                return grad

            dependency_list.append(Dependency(other, grad_fn2))

        return Tensor(data, requires_grad, dependency_list)

    def div(self, other):
        data = self.data / other.data
        requires_grad = self.requires_grad or other.requires_grad

        dependency_list = []

        if self.requires_grad:
            def grad_fn1(grad):
                grad = Tensor(grad.data / other.data)

                ndim_added = grad.ndim - self.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)

                # Sum across broadcasted (but non-added dims)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad.data = grad.data.sum(axis=i, keepdims=True)

                return grad

            dependency_list.append(Dependency(self, grad_fn1))
        
        if other.requires_grad:
            def grad_fn2(grad):
                grad = Tensor(-self.data / other.data ** 2)

                ndim_added = grad.ndim - other.ndim

                for _ in range(ndim_added):
                    grad.data = grad.data.sum(axis=0)
                
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad.data = grad.data.sum(axis=i, keepdims=True)

                return grad

            dependency_list.append(Dependency(other, grad_fn2))

        return Tensor(data, requires_grad, dependency_list)
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return self.add(other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.add(Tensor(other))
        # unknown operand
        raise TypeError("Unsupported operand type(s) for +")


    def __radd__(self, other):
      return self.__add__(other)
    
    def neg(self):
        data = -self.data
        requires_grad = self.requires_grad
        if requires_grad:
            depends_on = [Dependency(self, lambda x: -x)]
        else:
            depends_on = []

        return Tensor(data, requires_grad, depends_on)
    
    def __neg__(self):
        return self.neg()
    
    def __isub__(self, other):
        self.data = self.data - other.data
        return self
    
    def __iadd__(self, other):
        self.data = self.data + other.data
        return self
    
    def __imul__(self, other):
        self.data = self.data * other.data
        return self

    def __idiv__(self, other):
        self.data = self.data / other.data
        return self
    
    def __sub__(self, other):
        return self.add(-other)

    def __rsub__(self, other):
      return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self.mul(other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.mul(Tensor(other))
        # unknown operand
        raise TypeError("Unsupported operand type(s) for *")

    def __rmul__(self, other):
      return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self.div(other)
        elif isinstance(other, int) or isinstance(other, float):
            return Tensor(self.data / other)
        else:
            raise TypeError("Unsupported operand type(s) for /")

    def __rtruediv__(self, other):
      return self.__div__(other)
    
    def __pow__(self, other):
        if isinstance(other, Tensor):
            return self.pow(other)
        elif isinstance(other, int) or isinstance(other, float):
            return self.pow(Tensor(other))
        else:
            raise TypeError("Unsupported operand type(s) for **")

    def __rpow__(self, other):
      return self.__pow__(other)
    
    def __matmul__(self, other):
        return self.dot(other)
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __getitem__(self, key):
        return self.data[key]
    
    def reshape(self, *shape):
        data = self.data
        data = data.reshape(*shape)
        return Tensor(data)

    def sum(self):
        """
        Takes a tensor and returns the 0-tensor
        that's the sum of all its elements.
        """
        data = self.data.sum()
        requires_grad = self.requires_grad

        if requires_grad:
            def grad_fn(grad):
                """
                grad is necessarily a 0-tensor, so each input element
                contributes that much
                """
                return grad * Tensor(np.ones_like(self.data))

            depends_on = [Dependency(self, grad_fn)]

        else:
            depends_on = []

        return Tensor(data, requires_grad, depends_on)

    # initialization methods

    @staticmethod
    def from_numpy(array):
        return Tensor(array.data)
    
    @staticmethod
    def rand(*shape, requires_grad=False):
        data = np.random.rand(*shape)
        return Tensor(data, requires_grad)
    
    @staticmethod
    def zeros(*shape):
        data = np.zeros(*shape)
        return Tensor(data)
    
    @staticmethod
    def ones(*shape):
        data = np.ones(*shape)
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
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*shape), requires_grad)
    

    # compute gradient on Tensor

    def backward(self, grad=None):
        if grad is None:
            if self.shape == () or self.shape == (1,):
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-zero tensors")
        
        self.grad.data = self.grad.data + grad.data

        for depen in self.dependency:
            backward_grad = depen.grad_fn(grad)
            depen.tensor.backward(backward_grad)