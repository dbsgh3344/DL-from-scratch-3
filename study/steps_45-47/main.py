import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
from dezero import Variable
from dezero import functions as F
from dezero import layers as L
from dezero.layers import Layer
from dezero.models import Model,MLP
from dezero import optimizers



class TwoLayerNet(Layer):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid_simple(y)
        y = self.l2(y)
        return y


class TwoLayerNet2(Model):
    def __init__(self,hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, inputs):
        y = F.sigmoid_simple(self.l1(inputs))
        y = self.l2(y)
        return y



def step_45():
    def predict(model,x):
        y = model.l1(x)
        y = F.sigmoid_simple(y)
        y = model.l2(y)
        return y

    model = Layer()
    model.l1 = L.Linear(5)
    model.l2 = L.Linear(3)
    

    for p in model.params():
        print(p)

    
    model.clear_grads()

def step_45_2():
    x = Variable(np.random.randn(5, 10),name="x")
    model = TwoLayerNet2(100, 10)
    model.plot(x, to_file='model.png')
    



def step_45_3():
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) * np.random.randn(100, 1)
    
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)


    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)
        
        model.clear_grads()
        loss.backward()
        
        for param in model.params():
            param.data -= lr * param.grad.data
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.data}")


def step_45_4():
    # model = MLP((10, 10, 10))
    model = MLP([10, 10, 10,10], activation=F.sigmoid_simple)
    print(model.__dict__)



def step_46_2():
    np.random.seed(0)
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) * np.random.randn(100, 1)
    lr = 0.2
    max_iter = 10000
    hidden_size = 10
    
    model = MLP((hidden_size, 1))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)

    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)
        
        model.clear_grads()
        loss.backward()
        
        optimizer.update()
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.data}")

def step_47_1():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = np.array([0, 0, 1])


def softmax1d(x):
    x = F.as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)    
    return y / sum_y

def step_47_2():
    model = MLP((10,3))
    x = Variable(np.array([[1, 2, 3]]))
    print(x.shape)
    y = model(x)
    p = softmax1d(y)
    print(y,p)

def step_47_3():
    model = MLP((10,3))
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2],[2.1, 0.3]])
    print(x.shape)
    t = np.array([2,0,1,0])
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)


if __name__ == "__main__":
    step_45()
    # step_45_2()
    # step_45_3()
    # step_45_4()
    step_46_2()
    # step_47_2()
    # step_47_3()
    




