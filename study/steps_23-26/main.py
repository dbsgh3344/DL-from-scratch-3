if '__file__' in globals():
    import sys, os
    # sys.path.append(os.path.join(os.getcwd(),"study"))
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    

from dezero import Variable, Function, _dot_var, _dot_func, get_dot_graph, plot_dot_graph, plot_dot_graph_using_lib
import numpy as np

def sphere(x,y):
    return x**2 + y**2

def matyas(x,y):
    return 0.26*(x**2 + y**2) - 0.48*x*y

def gold_stein(x,y):
    a = 1 + ((x + y + 1)**2)*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    b = 30 + ((2*x - 3*y)**2)*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return a*b


if __name__ == '__main__':
    # step 23 패키지 테스트
    x = Variable(np.array(1.0))
    y = (x + 3) ** 3
    # y = Variable(np.array(2.0))
    # z = x + 2.0
    print("step 23: ",y)
    

    # # step 24 sphere 함수 테스트
    # x = Variable(np.array(3.0))
    # y = Variable(np.array(4.0))
    # z = sphere(x,y)
    # z.backward()    
    # print("step 24 ", z)
    # print("x.grad: ", x.grad)
    # print("y.grad: ", y.grad)

    # # step 24 matyas 함수 테스트
    # x = Variable(np.array(3.0))
    # y = Variable(np.array(4.0))
    # z = matyas(x,y)
    # z.backward()
    # print("step 24 matyas ", z)
    # print("x.grad: ", x.grad)
    # print("y.grad: ", y.grad)

    # # step 25 gold_stein 함수 테스트
    # x = Variable(np.array(1.0))
    # y = Variable(np.array(1.0))
    # z = gold_stein(x,y)
    # z.backward()
    # print("step 25 ", z)
    # print("x.grad: ", x.grad)
    # print("y.grad: ", y.grad)


    # # step 26 _dot_var 테스트
    # x = Variable(np.random.rand(2 , 3))
    # x.name = "x"
    # print(_dot_var(x))
    # print(_dot_var(x, verbose=True))

    # # step 26 _dot_func 테스트
    # x0 = Variable(np.array(2.0))
    # x1 = Variable(np.array(3.0))
    # y = x0 + x1
    # txt = _dot_func(y.creator)
    # print(txt)

    # # step 26 gold_stein 시각화
    # x = Variable(np.array(1.0))
    # y = Variable(np.array(1.0))
    # z = gold_stein(x,y)
    # z.backward()
    # x.name = "x"
    # y.name = "y"
    # z.name = "z"
    # print(os.getcwd())
    # # plot_dot_graph(z, verbose=False, to_file="./study/steps_23-26/gold_stein.png")
    # plot_dot_graph_using_lib(z, verbose=False, to_file="./study/steps_23-26/gold_stein_lib.png")
