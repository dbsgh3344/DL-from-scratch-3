import numpy as np
gpu_enable = True
try:
    import torch
    tc = torch
except ImportError:
    gpu_enable = False

from dezero import Variable



def get_array_module(x):
    
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = 
    return xp