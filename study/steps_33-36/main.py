# if '__file__' in globals():
import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
# from dezero import Variable
from dezero import Variable
from dezero import functions as F
from dezero.utils import plot_dot_graph_using_lib
import matplotlib.pyplot as plt
from memory_profiler import profile

class IterTest:
    def __init__(self, x, limit=10):
        self.x = x
        self.itr = 0
        self.limit = limit

    def __iter__(self):
        return self

    def __next__(self):
        cnt = 0
        if self.itr < self.limit:
            y = f(x)
            x.clear_grad()
            y.backward(create_graph=True)
            gx = x.grad
            x.clear_grad()
            gx.backward()
            gxx = x.grad
            self.itr += 1
            
            return gx.data / gxx.data
        else:   
            raise StopIteration




def f(x):
    return x**4 - 2*x**2

@profile
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

def newton_method3(x, itr=10):
    it = IterTest(x, itr)
    return it

@profile
def newton_method2(x, itr=10):
    r = []
    for i in range(itr):
        y = f(x)
        x.clear_grad()
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gxx = x.grad
        tmp = gx.data / gxx.data
        r.append(tmp)
        # print(x.data)    
    return r

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

def step35_tanh_graph(x:Variable, iters):
    y = F.tanh(x)
    x.name = "x"
    
    for i in range(iters):
        y.backward(create_graph=True)
        y.name = "y"
        y = x.grad
        x.clear_grad()

    y.name = "gx" + str(iters + 1)
    
    plot_dot_graph_using_lib(y, verbose=False, to_file="./tanh_graph.png")


@profile
def iter_test(x,iter):
    iter = newton_method(x, itr=iter)
    for i in iter:
        pass

@profile
def no_iter(x,iter):
    result = newton_method2(x, itr=iter)
    for i in result:
        pass


if __name__ == '__main__':
    
    # step 33 2차 미분 테스트
    x = Variable(np.array(2.0))
    y = f(x)
    print(y)
    y.backward(create_graph=True)
    print(x.grad)
    gx = x.grad
    x.clear_grad()
    gx.backward()
    print(x.grad)

    # step 33 뉴턴 방법을 활용한 최적화
    itr = 10000
    x = Variable(np.array(2.0))
    # result = newton_method(x, itr=itr)
    # for i in result:
    #     print(i)
    iter_test(x, itr)
    no_iter(x, itr)



    # # step 34 sin 함수의 미분
    # x = Variable(np.array(1.0))
    # y = F.sin(x)
    # y.backward(create_graph=True)

    # print(x.grad, np.cos(1))
    # sin_hd(x, order=3)

    
    # # step 34 sin graph
    # # x = Variable(np.arange(0, 2*np.pi, 0.1))
    # x = Variable(np.linspace(-7, 7, 200))
    # # step34_sin_graph(x)


    # # step 35 tanh 시각화
    # x = Variable(np.array(0.5))
    # step35_tanh_graph(x, iters=3)


    # # step 36 double backprop 구현
    # x = Variable(np.array(2.0))
    # y = x**2
    # y.backward(create_graph=True)
    # z = x.grad**3 + y
    # x.clear_grad()
    # z.backward(create_graph=True)
    # print(x.grad)

    
    # x = Variable(np.array([1.0,2.0]))

    # def f(x):
    #      t = x**2
    #      y = np.sum(t)
    #      return y

    # print(f(x))




    


