import numpy as np
import sys
import os
# print(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"study"))
from functions import Square,Exp,square,exp,Add,add
from base_class import Variable



if __name__ == "__main__" : 
    # xs = [Variable(np.array(2)), Variable(np.array(3))]
    # f= Add()
    # ys = f(xs)
    # y = ys[0]
    # print(y.data)
    
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0,x1)
    print(y.data)

    # x2 + y2 예제
    z = add(square(x0),square(x1))
    z.backward()
    print(z.data)
    print(x0.grad)
    print(x1.grad)

    # 동일 변수 반복 사용
    x = Variable(np.array(2.0))
    y = add(x,x)
    print(y.data)
    y.backward()
    print(x.grad)

    # 계산 반복 사용
    y = add(add(x,x),x)
    y.backward()
    print(x.grad)

    # clear
    x.clear_grad()
    y = add(add(x,x),x)
    y.backward()
    print(x.grad)

    # 올바른 역전파 순서 테스트
    x.clear_grad()
    a = square(x)
    y = add(square(a),square(a))
    y.backward()
    print(y.data)
    print(x.grad)

