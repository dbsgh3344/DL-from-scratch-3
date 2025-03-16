import numpy as np

class Variable: 
    def __init__(self, data:np.ndarray): 
        if not data :
            if not isinstance(data, np.ndarray) :
                raise TypeError('{} is not ndarray type'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self,func) :
        self.creator = func
    
    def type(self):
        return self.data.type

    def backward_recursion(self) :
        f = self.creator
        x = None        
        if f is not None :
            x = f.input
            x.grad = f.backward(self.grad)            
            x.backward()

    def backward(self) :
        if not self.grad :
            self.grad = np.ones_like(self.data)


        funcs = [self.creator]        
        while funcs :
            func = funcs.pop()                      
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)

            if x.creator:
                funcs.append(x.creator)




class Function: 
    def __call__(self, input:Variable): 
        if not isinstance(input, Variable) :
            raise TypeError('{} is not Variable'.format(type(input)))
        x = input.data
        y = self.forward(x)
        output = Variable(self.as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy) :
        raise NotImplementedError()

    def as_array(self,x) :
        if np.isscalar(x) :
            return np.array(x)
        return x




if __name__ == '__main__':    
    x= Variable(np.array([[3.0,2.0,4.0],[1.0,2.0,3.0]]))
    print(type(x))
    print(x.data.ndim)    
    # f = Function()
    # y = f(x)
    # print(y.data)
