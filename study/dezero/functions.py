import numpy as np
import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero.core import Function, Variable

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