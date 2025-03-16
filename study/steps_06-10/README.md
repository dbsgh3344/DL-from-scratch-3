# DeZero Step 6 ~ 10 정리

## Step 6
### 미분 값 추가
`Variable` 클래스에 미분 값도 저장할 수 있도록 변수를 추가합니다.

```python
class Variable :
    def __init__(self, data) :
    ...
    self.grad = None
```

### 역전파 구현
`Function` 클래스에 역전파 메소드 인터페이스를 추가합니다.  
그리고 `Function` 클래스를 상속받았던 함수들 내에서 역전파 메소드를 구현합니다.  
함수의 미분 값에 이전 함수의 역전파 결과 값을 곱해서 출력합니다. (chain rule)
```python
class Function :
    ...
    def backward(self) :
        raise NotImplementedError()


# Square
def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = 2*x*gy
        return gx

# Exp
def backward(self, gy:np.ndarray):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx

```


## Step 7
현재까지 구현한 수동으로 조합한 역전파를 자동으로 역전파를 실행하도록 수정합니다.
### 변수와 함수 연결
변수 내에 해당 변수를 가지고 실행할 함수를 저장합니다.  
```python
# Variable
...
def set_creator(self,func) :
    self.creator = func # 변수 내에 함수 저장

# Function
def __call_(self,input) :
    ...
    output.set_creator(self)    # 변수에 해당 함수 저장
    self.output = output    # 함수 내에 출력 저장

```


### 역전파 자동화
저장된 값들을 사용해서 역전파를 구현할 수 있습니다.  
이 과정은 역전파가 끝날 때까지 동일하게 반복되므로 이를 재귀를 통해 구현합니다.  

***base_class.py***
```python
y.grad = np.array(1.0)

C = y.creator   # 변수를 사용한 함수를 가져옴
b = C.input     # 그 함수의 입력으로 받았던 값을 가져옴
b.grad = C.backward(y.grad) # 함수의 입력 값의 기울기에 함수의 역전파 값을 대입함

# 재귀 구현
## Variable
def backward(self) :
    f = self.creator
    x = None        
    if f is not None :
        x = f.input
        x.grad = f.backward(self.grad)            
        x.backward()    # 함수의 입력으로 받았던 변수의 역전파를 호출 하며 재귀 구현

```

## Step 8
### 역전파 반복문 구현
기존 재귀 방식이 아닌 반복문을 통해 역전파를 자동화합니다.  
재귀 방식은 계속 이전 작업을 메모리에 들고 다니면서 다음 작업을 수행합니다.  
메모리를 매번 풀어주는 반복문으로 구현하는게 더 효율적입니다. 

***base_class.py***
```python
# Variable
def backward(self) :
    funcs = [self.creator]
    while funcs :
        func = funcs.pop()
        x, y = func.input, func.output
        x.grad = func.backward(y.grad)

        if not x.creator:
            funcs.append(x.creator)
```

## Step 9
### 파이썬 함수 이용
기존엔 매번 작성한 함수를 인스턴스를 생성 후 변수를 전달하는 방식으로 진행했는데  
이를 함수로 구현하여 인스턴스 없이 변수를 전달하게끔 변경합니다.

***fuctions.py***

```python

def square(x) :
    f = Square()
    return f(x)

def exp(x) :
    f = Exp
    return f(x)

```

### 역전파 간소화
기존엔 y의 기울기를 매번 직접 전달했다면  
자동으로 초기화 시켜주도록 변경합니다.

***base_class.py***

```python
# Variable backward
if not self.grad :
    self.grad = np.ones_like(self.data)
```


### type 제한
Variable 클래스가 ndarray 타입만 취급 할 수 있도록 필터링을 추가합니다.

***base_class.py***
```python
# Variable init
if not data :
    if not isinstance(data, np.ndarray) :
        raise TypeError('{} is not ndarray type'.format(type(data)))

```

numpy의 특성 상 0차원 ndarray가 특정 연산을 거치면 스칼라를 출력하는 경우가 있습니다.  
Variable 클래스는 ndarray 타입만 취급하기 때문에 이렇게 스칼라를 반환 할 경우 0차원 ndarray타입으로 변경해줘야 합니다.  
numpy의 np.isscalar 메소드를 통해 function클래스의 output이 스칼라일 경우 ndarray타입으로 변경합니다.

***base_class.py***
```python
# Function
def as_array(self,x) :
    if np.isscalar(x) :
        return np.array(x)
    return x

#Function.__call__
...
output = Variable(self.as_array(y))

```


