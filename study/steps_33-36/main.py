# if '__file__' in globals():
import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
# from dezero import Variable
from dezero import Variable
from dezero import functions as F
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2*x**2


def newton_method(x, itr=10):
    for i in range(itr):
        y = f(x)
        x.clear_grad()
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gxx = x.grad
        x.data -= gx.data / gxx.data
        yield x.data    
    
def sin_hd(x, order):
    for i in range(order):
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)
        print(x.grad)
        
def step34_sin_graph(x):
    y = F.sin(x)
    # y.backward(create_graph=True)
    # print(x.grad)
    # print(y)
    logs = [y.data]    
    
    # for i in range(1, 10):
    #     logs.append(x.grad.data)
    #     gx = x.grad
    #     x.clear_grad()
    #     gx.backward(create_graph=True)
        
    for i in range(3):
        y.backward(create_graph=True)
        logs.append(x.grad.data)
        y = x.grad
        x.clear_grad()

    print(len(logs))
    labels = ["y=sin(x)","y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        # print(f"y{i} = {v}")
        plt.plot(x.data, v, label=labels[i])
    plt.legend()
    plt.show()



if __name__ == '__main__':
    
    # step 33 2차 미분 테스트
    x = Variable(np.array(2.0))
    y = f(x)
    print(y)
    y.backward(create_graph=True)
    print(x.grad)  # 8.0
    gx = x.grad
    x.clear_grad()
    gx.backward()
    print(x.grad)

    # step 33 뉴턴 방법을 활용한 최적화
    itr = 10
    x = Variable(np.array(2.0))
    result = newton_method(x, itr=itr)
    for i in result:
        print(i)
    


    # step 34 sin 함수의 미분
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    print(x.grad, np.cos(1))
    sin_hd(x, order=3)

    
    # step 34 sin graph
    # x = Variable(np.arange(0, 2*np.pi, 0.1))
    x = Variable(np.linspace(-7, 7, 200))
    step34_sin_graph(x)

