if '__file__' in globals():
    import sys, os    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
from dezero import Variable

def f(x):
    return x**4 - 2*x**2


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward()
    print(x.grad)  # 8.0