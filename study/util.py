import numpy as np
import sys,os
sys.path.append(os.path.join(os.getcwd(),"study"))
# from functions import Square,Exp,Add,Mul,Neg



def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(data):
    from base_class import Variable
    if isinstance(data, Variable):
        return data
    return Variable(data)
