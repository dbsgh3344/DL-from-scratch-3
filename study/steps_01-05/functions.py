from base_class import Function, Variable
import numpy as np


class Square(Function) :
    def forward(self,x) :
        return x**2
    
class Exp(Function) :
    def forward(self, x):
        return np.exp(x)