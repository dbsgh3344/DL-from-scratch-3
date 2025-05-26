from dezero.core import Parameter
import weakref
import numpy as np
from dezero import functions as F

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()


class Linear(Layer):
    def __init__(self,out_size, nobias=False, dtype=np.float32,in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self.__init__W()



        # I, O = in_size, out_size
        # w_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
        # self.W = Parameter(w_data, name='W')
        
        
        if nobias:
            self.b = None
        else:
            b_data = np.zeros(out_size, dtype=dtype)
            self.b = Parameter(b_data, name='b')

    def __init__W(self):
        I, O = self.in_size, self.out_size
        w_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = w_data

    
    def forward(self, x):        
        if self.W.data is None:
            self.in_size = x.shape[1]
            self.__init__W()

        y = F.linear_simple(x, self.W, self.b)        
        return y