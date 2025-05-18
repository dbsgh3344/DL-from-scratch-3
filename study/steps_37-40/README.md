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




```python
```
