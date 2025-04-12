# DeZero Step 17 ~ 22 정리

## Step 17
python의 메모리 관리 두 가지 방식에 대해 설명합니다.
### 참조 카운트
객체의 카운트가 0에서 시작해서 참조 카운팅을 증가시킵니다.  
만약 참조 카운트가 0일 경우 메모리에서 삭제됩니다.  
참조 카운트 증가 조건
- 대입 연사자를 사용할 떄
- 함수에 인수로 전달할 때
- 컨테이너 타입 객체에 추가할 떄 (list, tuple .. )
```python
a = obj() # a:1
b = obj() # b:1
c = obj() # c:1

a.b = b # b:2
b.c = c # c:2
a = b = c = None # a:0 , b:1, c:1
```
b, c의 카운트가 1이지만  
b를 참조하던 a가 삭제 됐으므로 결국 0이 됨  
C를 참조하던 b가 삭제 됐으므로 결국 0이 됨  


#### 참조 카운트 예외 케이스
```python
a = obj() # a:1
b = obj() # b:1
c = obj() # c:1

a.b = b # b:2
b.c = c # c:2
c.a = a # b:2
a = b = c = None # a:1 , b:1, c:1
```
<br>
<img src="/Users/song-yunho/code/DL-from-scratch-3/study/steps_17-22/imgs/그림 17-2.png" width=40%>
<br>
객체에 None을 대입해도 순환 참조 현상에 의해 메모리가 삭제되지 않습니다.  

### Garbage Collection
Python에서 GC는 보통 메모리가 부족해질 시 자동으로 호출됩니다.  
GC는 이러한 순환 참조를 삭제 할 수 있습니다.  

```python
import gc 
gc.collect() # GC 수동 호출
```

### 약한 참조 반영
python에서 weakref 모듈을 사용하여 참조 카운팅을 하지 않을 수 있습니다.  
참조 카운팅 하지 않으면 사용된 객체는 삭제되어 메모리를 유지 할 수 있습니다.  
이를 dezero 코드에서 순환 참조가 발생하는 구간에 반영합니다.  
<br>  
<img src="/Users/song-yunho/code/DL-from-scratch-3/study/steps_17-22/imgs/그림 17-3.png" width=40%>
<br>
<!-- 순환 참조 발셍 구간 : Function의 output을 Variable로 받고 Variable backward에서  -->

```python
class Function: 
    def __call__(self,*inputs)
    ...
    self.outputs = [weakref.ref(output) for output in outputs]

class Variable:
    def backward(self) :
        ...
        gys = [output().grad for output in func.outputs]
```


## Step 18
메모리 절약을 위해 변수 등의 할당 된 메모리를 제거합니다.

### 사용 안하는 미분 값 삭제
미분이 필요하지 않은 연산에선 미분을 저장하지 않을 수 있도록 파라미터를 추가합니다.  
해당 파라미터가 True 일 땐 미분 값을 저장하지 않습니다.  

```python
def backward(self,retain_grad = False) :
    ...
    if not retain_grad:
        for y in func.outputs:
            y().grad = None

```

### Config를 통한 모드 전환
신경망에선 학습 시엔 역전파 계산이 필요하지만 추론 시엔 사용되지 않습니다.  
이처럼 역전파를 사용하지 않을 경우 역전파 계산 값을 저장하지 않도록 config를 추가합니다.  

```python
class Config
    enable_backprop = True

class Function:
    ...
    if Config.enable_backprop :
        self.generation = max([i.generation for i in inputs])
        for output in outputs :
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]        
```

### contexlib을 활용한 모드 전환
contextlib를 사용하여 context구문에 있을 때만 특정 config 값에 의한 실행이 되도록 추가합니다.  

```python
import contextlib
@contextlib.contextmanager
def use_config(name,value) :
    old_value = getattr(Config,name)    # Config class의 name 변수를 가져온다
    setattr(Config,name,value)  # Config class의 name 변수의 값을 value로 변경한다.
    try :
        yield   # 이 함수가 사용 된 context의 프로세스를 처리
    finally:
        setattr(Config,name,old_value)  # Config class의 name 변수의 값을 old_value로 변경한다.


def no_grad() :
    return use_config('enable_backprop',False)

```

## Step 19
Step 19에선 Variable 클래스의 사용성을 개선합니다.  
`numpy.ndarray`의 shape, ndim 등 데이터의 형상 정보를 볼 수 있는 인스턴스 변수들을 추가합니다.  

```python
class Variable:
    ...

    @property
    def shape(self) :
        return self.data.shape
    
    @property
    def ndim(self) :
        return self.data.ndim
    
    @property
    def size(self) :
        return self.data.size
    
    @property
    def dtype(self) :
        return self.data.dtype

    def __len__(self) :
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n","\n" + ' ' * 9)        

        return f"Var({p})"

```

## Step 20
step 20에서는 연산자 곱셈,덧셈 오버라이딩을 구현합니다.  


### Mul class 구현
곱셈 클래스를 구현합니다.  

```python
class Mul(Function) :
    def forward(self, x0,x1):
        return x0 * x1
    
    def backward(self, gy):
        x0,x1 = self.inputs[0].data , self.inputs[1].data
        return x1*gy,x0*gy

def mul(x0,x1) :
    f = Mul()
    return f(x0,x1)

```

### 연산자 오버라이딩
기존에 연산자를 사용하기 위해선 함수를 호출하고 인자를 넘기는 방식으로 코딩했습니다.  
연산자를 오버라이딩해서 numpy가 사용하는 방식처럼 간단하게 변경합니다.  

```python
y = add(mul(a,b),c) # 기존 연산자 사용 방식
y= a * b +c # 변경 할 방식

```
```python
class Variable:
    def __mul__(self,other):
            from functions import mul
            return mul(self,other)

    def __add__(self,other):
        from functions import add
        return add(self,other)

```


## Step 21
step 21에서는 Variable 클래스를 다른 타입의 인스턴스와 함께 사용하기 위한 구현을 추가합니다.

### np.array와 연산
Variable 클래스를 np.array와 연산하기 위해 np.array를 Variable 클래스로 변환해줍니다.  

```python
def as_variable(data):
    from base_class import Variable
    if isinstance(data, Variable):
        return data
    return Variable(data)

inputs = [util.as_variable(x) for x in inputs]

```

### float, int와 연산
Variable 클래스를 float, int와 연산하기 위해 float,int를 np.array로 변환해줍니다.  
np.array로 변환되면 앞선 as_variable에 의해 Variable 클래스로 변환됩니다.  

```python
def add(x0,x1) :
    f = Add()    
    x1 = util.as_array(x1)
    return f(x0,x1)

```
### 이슈 케이스 2 가지
1. 첫 번째 인수가 float or int 일 경우 rmul,radd가 구현되어 있지 않기 때문에 에러 발생  
- right 메소드 구현하기
2. np.array가 우항일 때 np.array의 연산자 메소드를 사용하게 됨.  
- 연산자 우선 순위 변경하기
```python
# case 1
def __rmul__(self,other):
    from functions import mul
    return mul(self,other)


def __radd__(self,other):
    from functions import add
    return add(self,other)

# case 2
class Variable:         
    __array__priority__ = 200
    ...

```
## Step 22
### 부호 변환 연산자 추가
부호를 변환해주는 연산자를 추가합니다.  

```python
class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self, gy):
        return -gy
...
def neg(x):
    return Neg()(x)
...
Variable.__neg__ = neg

```

### 빼기 연산자 추가
뺄셈은 오른쪽 항 클래스에서 호출 될 수도 있으니 rsub도 추가합니다.  
```python
class Sub(Function):
    def forward(self,x0,x1):
        return x0 - x1
    def backward(self, gy):        
        return gy, -gy
...
def sub(x):
    return Sub()(x)
def rsub(x0,x1):
    x1 = util.as_array(x1)
    return Sub()(x1,x0)
...
Variable.__sub__ = sub
Variable.__rsub__ = rsub
```

### 나누기 연산자 추가

```python
class Div(Function):
    def forward(self, x0,x1):
        return x0/x1
    def backward(self, gy):
        x0,x1 = self.inputs[0].data,self.inputs[1].data
        return (1/x1) * gy, (-x0/x1**2) * gy

...
def rdiv(x0,x1):
    x1 = util.as_array(x1)
    return Div(x1,x0)

```



### 거듭제곱 연산자 추가
거듭제곱 연산자는 초기화 클래스에 지수 값을 제공하는 방식으로 작성합니다.  
```python
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0].data
        return gy * x ** (self.c-1) * self.c
...
def pow(x,c) :
    return Pow(c)(x)

```
