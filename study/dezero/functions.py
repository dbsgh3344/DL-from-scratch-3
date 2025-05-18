import numpy as np
import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero.core import Function, Variable,as_variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = -sin(x) * gy
        return gx
    
def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self,x):
        # y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def backward(self, gy):        
        y = self.outputs[0]() 
        gx = (1 - y ** 2) * gy
        return gx

def tanh(x):
    return Tanh()(x)
        

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        x = self.inputs        
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):        
        return transpose(gy)
        
def transpose(x):
    return Transpose()(x)



class Sum(Function):
    def forward(self, x):
        y = np.sum(x)
        return y

    def backward(self, gy):
        # gx = np.ones_like(self.inputs[0].data) * gy
        gx = sum([i for i in self.inputs]) * gy
        return gx

def sum(x):
    return Sum()(x)


if __name__ == "__main__":
    # Test the Sin and Cos functions
    x = Variable(np.array(1))
    y = sin(x)
    print("sin =", y.data)
    
    y.backward()
    print("Gradient of sin =", x.grad.data)
    
    x.clear_grad()
    
    y = cos(x)
    print("cos =", y.data)
    
    y.backward()
    print("Gradient of cos =", x.grad.data)