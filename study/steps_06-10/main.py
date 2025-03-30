import numpy as np
import sys
import os
# print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"study"))
from functions import Square,Exp,square,exp
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

assert y.creator == C 
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# print(y.creator.backward(y.grad))
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)

# 역전파 자동화
y.grad= np.array(1.0)
y.backward_recursion()
print(x.grad)

# x = Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# c = square(b)


# print(np.ones_like(np.array(1.0)))
