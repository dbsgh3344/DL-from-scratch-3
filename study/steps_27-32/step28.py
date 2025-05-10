import numpy as np
if '__file__' in globals():
    import sys, os    
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Function, Variable



def rosenbrock(x0, x1):
    a = 1
    b = 100
    y = b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2
    return y

def g_d(x0,x1):
    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0,x1)
        y = rosenbrock(x0,x1)

        x0.clear_grad()
        x1.clear_grad()
        y.backward()
        
        
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad





if __name__ == '__main__':
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    y = rosenbrock(x0, x1)
    y.backward()
    print(y)  # Expected output: 0.0
    print(x0.grad)  # Expected output: 0.0
    print(x1.grad)  # Expected output: 0.0


    # Gradient descent
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    g_d(x0,x1)