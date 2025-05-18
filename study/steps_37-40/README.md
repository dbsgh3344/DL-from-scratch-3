# DeZero Step 37 ~ 40 정리

## Step 37
37단계에서는 기존의 스칼라 계산이 아닌 텐서 계산을 사용합니다.

### 텐서 사용 시 역전파
텐서 연산을 순전파, 역전파로 수행합니다.  
이때 사용한 변수의 shape은 역전파 후에도 동일함을 확인 할 수 있습니다.  

```python
# 순전파
np_array = np.array([[1,2,3],[3,4,5]])
np_array2 = np.array([[1,2,3],[3,4,5]])
x = Variable(np_array)
c = Variable(np_array2)
t = x + c
y= F.sum(t)

# 역전파
y.backward(retain_grad=True)


```


## Step 38
Transpose와 reshape 함수를 구현합니다

### reshape 함수 구현
reshape은 형상을 변환해주는 함수입니다.  
형상을 변환할 뿐 연산이 발생하진 않지만 역전파 시 형상을 다시 입력과 동일하게 변환합니다.  


```python
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        x = self.inputs        
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

```

Variable 함수에도 reshape을 사용 할 수 있도록 구현합니다.  
파라미터를 packing하여 인수가 풀어서 들어올 떄도 tuple 형식으로 반환 될 수 있도록 합니다.  

```python
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        x = self.inputs        
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
```


### Transpose 구현
행렬 전치를 구현합니다.  


```python
class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):        
        return transpose(gy)
        
def transpose(x):
    return Transpose()(x)
```

### Transpose 보충
넘파이의 transpose 함수는 축의 순서를 변경 할 수 있습니다.  
dezero의 transpose도 축의 순서를 변경 할 수 있도록 수정해줍니다.

```python
def reshape(self, *shape):
        # if self.shape == shape:
        #     return self
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
```



## Step 39
39단계에선 텐서 연산이 가능한 sum 함수를 구현합니다.  
역전파 단계에서 입력값의 형상을 복원하기 위해 broadcast_to 함수를 사용합니다.  
sum 의 기준을 축에 따라 변경하는 axis, 기존 차원을 유지하는 keepdims도 구현합니다.  

```python
class Sum(Function):
    def __init__(self,axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis,keepdims=self.keepdims)
        return y

    def backward(self, gy):        
        # gx = sum([i for i in self.inputs]) * gy
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x,axis,keepdims=False):
    return Sum(axis,keepdims)(x)

```

## Step 40
40단계에선 broadcast를 구현합니다.  


### 40.2 broadcast, sum_to 구현
브로드캐스트는 현재 형상을 복사하여 형상을 늘릴 때 사용합니다.  



```python
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        x.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
```


```python
class SumTo:
    def __init__(self,shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, axis=self.shape, keepdims=True)
        return y

    def backward(self, gy):        
        gx = broadcast_to(gy, self.x_shape)
        return gx    

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
```




```python
```
