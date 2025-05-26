# DeZero Step 41 ~ 44 정리

## Step 41




## Step 42
42 단계에선 선형 회귀를 구현합니다.

### 선형 회귀 이론


### 선형 회귀 구현
선형 회귀 모델의 식에 값을 대입하여 구현합니다.  
오차함수로 mse를 사용하기 위해 mse도 구현합니다.  

$y=Wx+b$

```python
np.random.seed(0)
x = np.random.rand(10,1)
y = 5 + 2* x + np.random.rand(10,1)
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



def predict(x,W,b):
        y = F.matmul(x, W) + b
        return y

def mean_squared_error(y_true, y_pred):
    diff = y_true - y_pred
    return F.sum(diff ** 2) / len(diff)

```




## Step 43
43단계에선 신경망 구현을 수행해봅니다.  

### linear 함수
이전 단계에서 사용한 선형 회귀 식에서 메모리를 절약하기 위해 linear함수를 구현합니다. 

```python

```



## Step 44
44 단계에선 파라미터를 관리하는 Parameter와 Layer class를 구현합니다.  


### Parameter class
Parameter클래스는 Variable 클래스를 상속받아서 그대로 사용합니다.  
하지만 isinstance를 통해 클래스를 구별 할 수 있습니다.  


```python
class Parameter(Variable):
    pass
```

### Layer class
Layer 클래를 구현합니다. layer클래스는 매개변수 값을 저장하고 변환하기 위해 사용합니다.  

```python
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()
```


### Linear class
선형 변환을 사용하기 위해 Linear클래스를 구현합니다.  



```python
class Linear(Layer):
    def __init__(self,out_size, nobias=False, dtype=np.float32,in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self.__init__W()



        # I, O = in_size, out_size
        # w_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
        # self.W = Parameter(w_data, name='W')
        
        
        if nobias:
            self.b = None
        else:
            b_data = np.zeros(out_size, dtype=dtype)
            self.b = Parameter(b_data, name='b')

    def __init__W(self):
        I, O = self.in_size, self.out_size
        w_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = w_data

    
    def forward(self, x):        
        if self.W.data is None:
            self.in_size = x.shape[1]
            self.__init__W()

        y = F.linear_simple(x, self.W, self.b)        
        return y
```

