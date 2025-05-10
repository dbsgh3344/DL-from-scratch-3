import numpy as np
if '__file__' in globals():
    import sys, os    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Function, Variable
import math

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)


def custom_sin(x,threshold=1e-4):
    y = 0    

    for i in range(100000):
        term = ((-1)**i) * (x**(2*i+1)) / math.factorial(2*i+1)
        y += term

        if abs(term.data) < threshold:
            break
    
    return y
    
        


if __name__ == '__main__':
    
    # Sin function test
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()
    print(y)  # Expected output: 0.7071067811865475
    print(x.grad)  # Expected output: 0.7071067811865475


    # custom sin test
    x = Variable(np.array(np.pi / 4))
    y = custom_sin(x)
    y.backward()
    print(y.data)
    print(x.grad)
    