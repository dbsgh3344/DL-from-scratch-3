from base_class import Function, Variable
import numpy as np


class Square(Function) :
    def forward(self,x) :
        return x**2
    def backward(self, gy:np.ndarray):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
        
    
class Exp(Function) :
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

class Add(Function) :
    def forward(self,x0,x1) :
        return x0 + x1

    def backward(self,gy) :
        return gy,gy


def square(x) :
    f = Square()
    return f(x)

def exp(x) :
    f = Exp
    return f(x)

def add(x0,x1) :
    f = Add()
    return f(x0,x1)