from typing import Optional
import numpy as np
import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero.core import Function, Variable,as_variable
from dezero import utils


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
    def __init__(self,axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis,keepdims=self.keepdims)
        return y

    def backward(self, gy):        
        # gx = sum([i for i in self.inputs]) * gy
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x,axis,keepdims=False):
    return Sum(axis,keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        x.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo:
    def __init__(self,shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, axis=self.shape, keepdims=True)
        return y

    def backward(self, gy):        
        gx = broadcast_to(gy, self.x_shape)
        return gx    

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
        



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