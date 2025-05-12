# DeZero Step 33 ~ 36 정리

## Step 33
33단계에선 2차 미분 테스트 및 뉴턴 방법을 통한 최적화를 진행합니다.

### 2차 미분 계산
2차 미분이 정상적으로 동작하는지 확인해봅니다.   
주의 할 것은 각 차수 미분 후 해당 기울기 값을 초기화해줘야 누적 계산이 발생하지 않습니다.

$y = x^4 - 2x^2$

```python
x = Variable(np.array(2.0))
y = f(x)
print(y)
y.backward(create_graph=True)
print(x.grad)  # 8.0
gx = x.grad
x.clear_grad()
gx.backward()
print(x.grad)
```


### 뉴턴 방법으로 최적화
2차 미분이 가능해졌으니 뉴턴 방법으로 최적화를 진행해봅니다.  
훨씬 빠른 속도로 최솟값에 도달 할 수 있습니다.  

$x \gets x - \frac{f'(x)}{f''(x)}$

```python
def newton_method(x, itr=10):
    for i in range(itr):
        y = f(x)
        x.clear_grad()
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gxx = x.grad
        x.data -= gx.data / gxx.data
        yield x.data    

itr = 10
x = Variable(np.array(2.0))
result = newton_method(x, itr=itr)
for i in result:
    print(i)

#1.4545454545454546
#1.1510467893775467
#1.0253259289766978
#1.0009084519430513
#1.0000012353089454
#1.000000000002289
#1.0
#1.0
#1.0
#1.0

```


## Step 34
34 단계에선 sin함수의 고차 미분을 구현합니다.

### sin,cos 함수 구현
backward 메서드를 구현 할 땐 모두 dezero의 함수를 사용합니다.  
그래서 sin 함수의 미분에서 cos 함수를 사용하는데 이 cos 함수도 직접 구현한 cos 함수를 사용합니다.  
```python
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = cos(x) * gy
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = -sin(x) * gy
        return gx
    
def cos(x):
    return Cos()(x)

```

### sin 함수 고차 미분
반복문을 사용해 sin 함수의 고차 미분을 구현합니다.  

```python
x = Variable(np.array(1.0))
y = F.sin(x)
y.backward(create_graph=True)
order = 3
for i in range(order):
    gx = x.grad
    x.clear_grad()
    gx.backward(create_graph=True)
    print(x.grad)

# Var(-0.8414709848078965)
# Var(-0.5403023058681398)
# Var(0.8414709848078965)

```


## Step 35
35단계에선 tanh 함수를 구현합니다.
### tanh 함수 구현
미분 공식에 따라 tanh 함수의 backward를 구현 할 수 있습니다.

```python
class Tanh(Function):
    def forward(self,x):        
        return np.tanh(x)

    def backward(self, gy):        
        y = self.outputs[0]() 
        gx = (1 - y ** 2) * gy
        return gx

def tanh(x):
    return Tanh()(x)
```


## Step 36
36 단계에선 double backprop을 테스트합니다.  

### double backprop의 용도
dezero가 고차 미분을 수행 할 수 있게 되면서 미분이 포함된 식을 다시 미분하는 방법도 가능해졌습니다.  
따라서 다음과 같은 식도 해결이 가능합니다.  
$y = x^2$  
$z = (\frac{dy}{dx})^3 + y  $    

문제: `x = 2.0` 에서 z의 미분  


```python
x = Variable(np.array(2.0))
y = x**2
y.backward(create_graph=True)
z = x.grad**3 + y
x.clear_grad()
z.backward(create_graph=True)
print(x.grad)
# Var(100.0)

```


