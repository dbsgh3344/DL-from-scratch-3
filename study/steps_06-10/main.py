import numpy as np
import sys
import os
# print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"study"))
from functions import Square,Exp
from base_class import Variable

A = Square()
B = Exp()
C = Square()

# 순전파
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(y.grad, b.grad , a.grad,x.grad)

