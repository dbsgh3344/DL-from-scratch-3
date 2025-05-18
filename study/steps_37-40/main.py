import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
# from dezero import Variable
from dezero import Variable
from dezero import functions as F





if __name__ == "__main__":
    # 37.1 텐서 계산
    np_array = np.array([[1,2,3],[3,4,5]])
    x = Variable(np_array)
    y= F.sin(x)
    print(y)


    # 37.2 
    np_array = np.array([[1,2,3],[3,4,5]])
    np_array2 = np.array([[1,2,3],[3,4,5]])
    x = Variable(np_array)
    c = Variable(np_array2)
    t = x + c
    print(t)
    y = F.sum(t)
    # y = F.sin(t)
    print(y)
    # print(np.sum(np.array([[1,2,3],[3,4,5]]) * 2))
    y.backward(retain_grad=True)
    print(y.grad)
    print(t.grad)
    print(c.grad)
    print(x.grad)


    # 38.1
    # print(np.transpose(np.array([[1,2,3],[3,4,5]])))
    # print(np.reshape(np.array([[1,2,3],[3,4,5]]), (3,2)))
    x = Variable(np.array([[1,2,3],[3,4,5]])) # (2,3)
    y= F.reshape(x, (6,))
    print(y)
    y.backward(retain_grad=True)
    print(y.grad)
    print(x.grad)

    
    # 38.2 reshape 수정
    x = Variable(np.random.rand(2,2,3))
    print(x)
    y = x.reshape((2,6))
    print(y)  
    y = x.reshape(6,2)
    print(y)

    # 38.3 transpose 구현
    x = Variable(np.array([[1,2,3],[3,4,5]])) # (2,3)
    y = F.transpose(x)
    print(y) # (3,2)
    y.backward(retain_grad=True)
    print(y.grad) # (3,2)
    print(x.grad) # (2,3)

    x.clear_grad()
    y = x.transpose()
    print(y) # (3,2)
    y= x.T
    print(y) # (3,2)x

    # 38.4 transpose
    x = Variable(np.random.rand(1,2,3,4))
    x.T
    print(x.T.shape) # (4,3,2,1)
    
    