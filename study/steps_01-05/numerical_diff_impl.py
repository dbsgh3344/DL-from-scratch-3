import numpy as np
from base_class import Variable,Function
from functions import Square,Exp


class NumericalDiff :
    def __init__(self) :
        self.eps = 1e-4


    def centered_diff(self,f:Function,x:Variable) :
        h = self.eps 
        x0 = Variable(x.data + h)
        x1 = Variable(x.data - h)
        y0 = f(x0)
        y1 = f(x1)
        return (y0.data - y1.data) / (2*h) 

    def forward_diff(self,f:Function,x:Variable) :
        h = self.eps
        x0 = Variable(x.data + h)
        x1 = Variable(x.data)
        y0 = f(x0)
        y1 = f(x1)
        return (y0.data - y1.data) / h