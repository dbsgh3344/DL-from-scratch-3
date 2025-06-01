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
        # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self,shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):        
        gx = broadcast_to(gy, self.x_shape)
        return gx    

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
        

class Matmul(Function):
    def forward(self,x,W):
        y = x.dot(W)
        return y
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return Matmul()(x, W)


class Exp(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        t = (2. / len(diff))        
        gx0 = gy * diff * t
        # gx0 = gy * diff * ( /)
        gx1 = -gx0        
        return gx0, gx1

def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)


    
def linear_simple(x,W,b=None):
    t= matmul(x,W)
    if b is None:        
        return t

    y = t + b
    t.data = None
    return y

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = clip(p, 1e-15 ,1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N),t.data]
    y = -1 * sum(tlog_p) / N
    return y


class GetItem(Function):
    def __init__(self,slices):
        self.slices = slices

    
    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices,x.shape)

        return f(gy)
        
def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self,slies,in_shape):
        self.slices = slies
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)
        

class Log(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)        
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        # xp = cuda.get_array_module(x)
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

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