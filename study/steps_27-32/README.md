# DeZero Step 27 ~ 32 정리

## Step 27
테일러 급수를 계산해보는 단계입니다.

### sin 함수 구현
sin함수의 계산과 미분 값을 구현합니다.

```python
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    return Sin()(x)
```

### 테일러 급수 구현
테일러 급수를 사용해서 sin함수를 구현합니다.  
threshold 값이 작아질수록 더 정교하게 sin함수를 근사 할 수 있습니다.  


```python
def custom_sin(x,threshold=1e-4):
    y = 0    

    for i in range(100000):
        term = ((-1)**i) * (x**(2*i+1)) / math.factorial(2*i+1)
        y += term

        if abs(term.data) < threshold:
            break
    
    return y
```



## Step 28
### 로젠브록 함수 
로젠브록 함수는 함수 최적화 문제에 자주 사용되기에 이를 통해 최솟값을 구하는 예제를 진행합니다.  


```python
def rosenbrock(x0, x1):
    a = 1
    b = 100
    y = b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2
    return y

```



### 경사하강법 구현
경사하강법은 원하는 파라미터 값을 찾기 위해 반복해서 기울기를 구해서 최소,최대값에 도달하는 방법입니다.  
이를 로젠브록 함수에 적용해봅니다.  
반복횟수가 높아질수록 정확한 파라미터 값을 산출 할 수 있습니다.   
하지만 많은 반복에도 파라미터 산출이 어려울 수 있습니다.

```python
def g_d(x0,x1):
    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0,x1)
        y = rosenbrock(x0,x1)

        x0.clear_grad()
        x1.clear_grad()
        y.backward()
        
        
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

```



## Step 29



```python
```