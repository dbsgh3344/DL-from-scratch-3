import numpy as np

class Variable: 
    def __init__(self, data:np.ndarray): 
        self.data = data
        self.grad = None
    
    def type(self):
        return self.data.type

class Function: 
    def __call__(self, input:Variable): 
        if not isinstance(input, Variable) :
            raise TypeError('{} is not Variable'.format(type(input)))
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()





if __name__ == '__main__':    
    x= Variable(np.array([[3.0,2.0,4.0],[1.0,2.0,3.0]]))
    print(type(x))
    print(x.data.ndim)    
    # f = Function()
    # y = f(x)
    # print(y.data)
