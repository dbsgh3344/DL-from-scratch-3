import numpy as np

class Variable: 
    def __init__(self, data:np.ndarray): 
        if not data :
            if not isinstance(data, np.ndarray) :
                raise TypeError('{} is not ndarray type'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
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

    def backward(self) :
        def add_func(f:Function) :
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

                
        
        if not self.grad :
            self.grad = np.ones_like(self.data)
        # funcs = [self.creator]        
        funcs = []
        seen_set = set()

        add_func(self.creator)



        while funcs :
            print(id(self))
            func = funcs.pop()                      
            # x, y = func.input, func.output
            # x.grad = func.backward(y.grad)
            gys = [output.grad for output in func.outputs]
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
                    add_func(x.creator)




class Function: 
    def __call__(self, *inputs): 
        # if not isinstance(inputs, Variable) :
        #     raise TypeError('{} is not Variable'.format(type(inputs)))
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(self.as_array(y)) for y in ys]
        self.generation = max([i.generation for i in inputs])
        for output in outputs :
            output.set_creator(self)
        

        self.inputs = inputs
        self.outputs = outputs
        # return outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys) :
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
