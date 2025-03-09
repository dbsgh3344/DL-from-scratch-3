import numpy as np
from base_class import Variable,Function
from functions import Square,Exp
from numerical_diff_impl import NumericalDiff

def f(x) :
    sq1 = Square()
    e = Exp()
    # sq2 = Square()
    return sq1(e(sq1(x)))


if __name__ == '__main__':    
    nu = NumericalDiff() 
    # x= Variable(np.array([[3.0,2.0,4.0],[1.0,2.0,3.0]]))
    x = Variable(np.array(3.0))
    
    
    # print(f(f(e_v)).data)
    # a=np.array([[1,2,3],[3,4,4]])
    # print(a,a.ndim,a.shape)

    # r = forward_diff(f,x)
    # print(r)
    sq = Square()
    r = nu.centered_diff(sq,x)
    print(r)

    
