from base_class import Function, Variable
import numpy as np


class Square(Function) :
    def forward(self,x) :
        return x**2
    def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = 2*x*gy
        return gx
        
    
class Exp(Function) :
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx


def square(x) :
    f = Square()
    return f(x)

def exp(x) :
    f = Exp
    return f(x)