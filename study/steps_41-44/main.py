import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
from dezero import Variable
from dezero import functions as F
from dezero import layers as L



def step_41():
    # 41.1
    x = Variable(np.array([[1,2,3],[3,4,5]])) # (2,3)
    W = Variable(np.array([[1,2],[3,4],[5,6]])) # (3,2)
    y = F.matmul(x, W) # (2,2)
    print(y) # (2,2)
    y.backward(retain_grad=True)
    print(y.grad) # (2,2)
    print(x.grad) # (2,3)
    print(W.grad) # (3,2)
    

def step_42():
    def predict(x,W,b):
        y = F.matmul(x, W) + b
        return y

    np.random.seed(0)
    x = np.random.randn(100,1)
    y = 5 + 2* x + np.random.randn(100,1)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1,1)))
    b = Variable(np.zeros(1))

    lr = 0.01
    iters = 150

    for i in range(iters):
        y_pred = predict(x,W,b)
        loss = mean_squared_error(y, y_pred)

        W.clear_grad()
        b.clear_grad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.data}, W: {W.data}, b: {b.data}")
    

def mean_squared_error(y_true, y_pred):
    diff = y_true - y_pred
    return F.sum(diff ** 2) / len(diff)


def step_43():
    def predict(x):
        y = F.linear_simple(x, W1, b1)
        y = F.sigmoid_simple(y)
        y = F.linear_simple(y, W2, b2)
        return y


    np.random.seed(0)
    x = np.random.rand(100,1)
    y = np.sin(2*np.pi*x) + np.random.rand(100,1)


    I,H,O = 1,10,1
    W1 = Variable(0.01 * np.random.randn(I,H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H,O))
    b2 = Variable(np.zeros(O))

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)
        
        W1.clear_grad()
        b1.clear_grad()
        W2.clear_grad()
        b2.clear_grad()
        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        if i % 1000 == 0:
            print(loss)

def step_44():
    # step 44
    def predict(x):
        y = l1(x)
        y = F.sigmoid_simple(y)
        y = l2(y)
        return y

    np.random.seed(0)
    x = np.random.rand(100,1)
    y = np.sin(2*np.pi*x) + np.random.rand(100,1)

    l1 = L.Linear(10)
    l2 = L.Linear(1)

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)
        
        l1.clear_grads()
        l2.clear_grads()        
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():                
                p.data -= lr * p.grad.data
        
        if i % 1000 == 0:
            print(loss)




if __name__ == "__main__":
    # numpy dot 
    a = np.array([[1,2],[3,4]])
    b = np.array([
        [1,2],
        [3,4]
        ]
        )
    print(np.dot(a,b))


    # matmul
    # step_41()

    # step 42 linear regression
    step_42()

    # 신경망 구현
    # step_43()
    # step_44()
