import weakref
import numpy as np
from heapq import heappop,heappush
import contextlib
# import sys, os    
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from dezero.core_simple import Function


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(data):    
    if isinstance(data, Variable):
        return data
    return Variable(data)




@contextlib.contextmanager
def use_config(name,value) :
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try :
        yield
    finally:
        setattr(Config,name,old_value)


def no_grad(value) :
    return use_config('enable_backprop',value)


def square(x) :
    f = Square()
    return f(x)

def exp(x) :
    f = Exp
    return f(x)

def add(x0,x1) :
    f = Add()    
    x1 = as_array(x1)
    return f(x0,x1)

def mul(x0,x1) :
    f = Mul()        
    return f(x0,x1)

def neg(x):
    return Neg()(x)

def sub(x0,x1):
    return Sub()(x0,x1)

def rsub(x0,x1):
    x1 = as_array(x1)
    return Sub()(x1,x0)

def div(x0,x1):
    return Div()(x0,x1)

def rdiv(x0,x1):
    x1 = as_array(x1)
    return Div(x1,x0)




class Config:
    enable_backprop = True

class Variable:         
    # __array__priority__ = 200
    def __init__(self, data:np.ndarray,name:str = None): 
        if data is None:
            if not isinstance(data, np.ndarray) :
                raise TypeError('{} is not ndarray type'.format(type(data)))
        # super()
        self.data = data
        self.name = name
        self.grad = None
        self.creator:Function = None
        self.generation = 0        

    # def __new__(cls, input_array, creator=None):
    #     obj = np.asarray(input_array).view(cls)
    #     print(obj)
    #     obj.creator = creator
    #     obj.grad = None
    #     obj.generation = 0
    #     return obj
    
    def clear_grad(self):
        self.grad = None


    def set_creator(self,func) :
        self.creator = func
        self.generation  = func.generation + 1
    
    def type(self):
        return self.data.type

    def backward_recursion(self) :
        f = self.creator
        x = None        
        print(id(self))
        if f is not None :
            x = f.input
            x.grad = f.backward(self.grad)            
            x.backward_recursion()

    def backward(self,retain_grad=False, create_graph=False) :
                        
        if self.grad is None :            
            self.grad = Variable(np.ones_like(self.data))
        
        funcs = []        
        heappush(funcs, self.creator)        

        while funcs :            
            func = heappop(funcs)            
            gys = [output().grad for output in func.outputs]
            with use_config('enable_backprop',create_graph):
                gxs = func.backward(*gys)                
                if not isinstance(gxs, tuple) :
                    gxs = (gxs,)

                    
                for x, gx in zip(func.inputs, gxs) :
                    if x.grad is None :
                        x.grad = gx
                    else :
                        x.grad = x.grad + gx
                
                    if x.creator:
                        # funcs.append(x.creator)
                        # add_func(x.creator)
                        heappush(funcs, x.creator)
                        funcs = list(set(funcs))
            
            if not retain_grad:
                for y in func.outputs:
                    y().grad = None
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n","\n" + ' ' * 9)        

        return f"Var({p})"
    

class Function: 
    def __call__(self, *inputs): 
        # if not isinstance(inputs, Variable) :
        #     raise TypeError('{} is not Variable'.format(type(inputs)))        
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([i.generation for i in inputs])
            for output in outputs:
                output.set_creator(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]        
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

    # def as_array(self,x):
    #     if np.isscalar(x):
    #         return np.array(x)
    #     return x
    

    def __lt__(self,item):
        return -self.generation < -item.generation

class Square(Function) :
    def forward(self,x) :
        return x**2
    def backward(self, gy:np.ndarray):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
        
    
class Exp(Function) :
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

class Add(Function) :
    def forward(self,x0,x1) :
        return x0 + x1
    def backward(self,gy) :
        return gy,gy
    
class Mul(Function) :
    def forward(self, x0,x1):
        return x0 * x1    
    def backward(self, gy):
        x0,x1 = self.inputs
        return x1*gy,x0*gy

class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self,x0,x1):
        return x0 - x1
    def backward(self, gy):        
        return gy, -gy

class Div(Function):
    def forward(self, x0,x1):
        return x0/x1
    def backward(self, gy):
        x0,x1 = self.inputs
        return (1/x1) * gy, (-x0/x1**2) * gy

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = x ** (c - 1) * c *gy
        return gx

def pow(x,c) :
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow