import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(),"study"))
from base_class import Variable,Config,Square,Exp,Add,add,mul,square,exp,neg,sub
import contextlib

@contextlib.contextmanager
def use_config(name,value) :
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try :
        yield
    finally:
        setattr(Config,name,old_value)


def no_grad(value) :
    return use_config('enable_backprop',value)
        
    
    


if __name__ == "__main__" :
    x0 = Variable(2.0)
    # print(x0)
    x1 = Variable(np.array(2.0))

    t = add(x0,x1)
    y = add(x0,t)
    y.backward()

    print(y.grad,t.grad)
    print(x0.grad,x1.grad)


    # step 17 mode 전환
    Config.enable_backprop = True    
    x = Variable(np.ones((100,100,100)))
    y = square((square(square(x))))
    y.backward()
    # print(x.grad,y)

    Config.enable_backprop = False    
    x = Variable(np.ones((100,100,100)))
    y = square((square(square(x))))
    print(x.grad)
    print(x.shape)
    

    # step 18 context 사용
    # with use_config('enable_backprop',False) :
    #     x= Variable(np.array(2.0))
    #     y = square(x)
    #     print(y.data)

    with no_grad(False):
        x= Variable(np.array(2.0))
        y = square(x)

    
    # step 19 매직 메소드 및 프로퍼티 
    x= Variable(np.array([2.0]))
    print(x.shape)
    print(x.dtype)
    print(x.size)
    print(x.ndim)
    print(x)


    # step 20 mul 구현
    with no_grad(True):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        c = Variable(np.array(4.0))


        y = add(mul(a,b),c)
        print("step 20 mul forward : ", y.data)
        y.backward()
        print("step 20 mul backward : ", y,a.grad,b.grad,c.grad)

    # step 20 mul 매직 메소드 구현
    with no_grad(True):
        a = Variable(np.array(2.0))
        b = Variable(np.array(3.0))
        c = Variable(np.array(4.0))


        y = a * b + c
        print("step 20 mul forward : ", y.data)
        y.backward()
        print("step 20 mul backward : ", y,a.grad,b.grad,c.grad)


    # step 21 np.array와 사용
    x = Variable(np.array([2.0]))
    y = x + np.array(3.0)
    print(y)

    # step 21 float와 사용
    y = x + 2.0
    y2 = 4.0 + x
    y3 = 4.0 * x + 2.0
    print(y,y2,y3)

    # step 21 좌항 np.array 케이스
    y = np.array([3.0]) + x
    print(y)


    # step 22 neg 구현
    x = Variable(np.array(2.0))
    y = neg(x)
    print(y.data)

    # step 22 뺄셈
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = sub(x0,x1)
    print(y.data)

    y = 3.0 - x0
    print(y.data)

    # step 22 pow
    y = x0 ** 2
    print(y)