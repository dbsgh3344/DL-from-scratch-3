import sys, os    
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np 
from dezero import datasets
import math
from dezero.models import Model,MLP
from dezero import optimizers
from dezero import functions as F




def step_48_2():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    x, t = datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)
    print(model)
    print("x.shape", x.shape, "t.shape", t.shape)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)
    print("max_iter", max_iter)

    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)        
        sum_loss = 0

        for i in range(max_iter):

            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            y = model(batch_x)
            loss = F.softmax_cross_entropy_simple(y, batch_t)
            model.clear_grads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print("epoch %d , loss %.2f" % (epoch + 1, avg_loss))
        
        if avg_loss in [np.nan,"nan"]:
            print("loss is nan, break")
            print(loss)
            break


def step_49_3():
    train_set = datasets.Spiral()
    batch_index = [0,1,2]
    batch = [train_set[i] for i in batch_index]
    x = np.array([ b[0] for b in batch ])
    t = np.array([ b[1] for b in batch ])
    print(x.shape, t.shape)
    print(x[0], t[0])   


def step_49_4():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = datasets.Spiral()
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)
    print(model)
    

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)
    print("max_iter", max_iter)

    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)        
        sum_loss = 0

        for i in range(max_iter):

            batch_index = index[i * batch_size:(i + 1) * batch_size]
            # batch_x = x[batch_index]
            # batch_t = t[batch_index]
            batch = [train_set[i] for i in batch_index]            
            batch_x = np.array([ b[0] for b in batch ])
            batch_t = np.array([ b[1] for b in batch ])


            y = model(batch_x)
            loss = F.softmax_cross_entropy_simple(y, batch_t)
            model.clear_grads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print("epoch %d , loss %.2f" % (epoch + 1, avg_loss))
        




if __name__ == "__main__" :
    # x ,t = datasets.get_spiral(train=True)
    # print(x.shape, t.shape)
    # print(x[0], t[0])

    # step_48_2()
    # step_49_3()
    step_49_4()
    
