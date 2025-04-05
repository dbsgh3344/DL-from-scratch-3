# DeZero Step 11 ~ 16 정리

## Step 11
### 가변 길이 입출력 표현
현재 function 클래스는 하나의 인수를 받고 하나의 값만 출력하고 있습니다.  
여러 개의 값을 받고 출력하도록 수정합니다.
```python
def __call__(self, inputs:list[Variable]):
    ...
    xs = [x.data for x in inputs]
    ys = self.forward(xs)
    outputs = [Variable(self.as_array(y)) for y in ys]
    for output in outputs :
            output.set_creator(self)             

```

### Add 클래스 구현
2개의 값을 들고 있는 리스트를 인수로 받아서 더한 값을 튜플로 반환합니다.  

```python
class Add(Function) :
    def forward(self,xs) :
        x0,x1 = xs
        y = x0 + x1
        return (y,)
```
```python
xs = [Variable(np.array(2)), Variable(np.array(3))]
f= Add()
ys = f(xs)
y = ys[0]
print(y.data)

```

## Step 12
step 12 에선 Add 클래스를 사용,구현 입장에서 쉽게 사용하도록 수정합니다.  
### 함수 사용 개선
함수에 값을 전달 할 때 매번 리스트를 만들어서 전달하지 않아도 되도록 수정합니다.  
함수의 출력 값이 1개 일 경우 리스트, 튜플이 아닌 Variable변수로 받도록 수정합니다.
```python
def __call__(self, *inputs:list[Variable]): 
    ...
    return outputs if len(outputs) > 1 else outputs[0]

```

### 함수 구현 개선
리스트 언팩을 사용하여 forward 함수 구현 쪽에서 받는 인수 개수를 지정합니다.  
결과 값은 tuple로 변환합니다.  

```python
ys = self.forward(*xs)
if not isinstance(ys, tuple) :
    ys = (ys,)
```
```python
class Add(Function) :
    def forward(self,x0,x1) :
        return x0 + x1
```

## Step 13
step 13에선 역전파에서 가변 길이 인수를 받도록 수정합니다.  
### Add 클래스 역전파
Add 클래스는 순전파와 반대로 입력이 1개 출력이 2개 입니다.  
```python
class Add(Function) :
    ...

    def backward(self,gy) :
        return gy,gy

```
### Variable 클래스 수정
출력이 여러 개가 되는 케이스를 커버하기 위해서 Variable 클래스를 수정합니다.  
해당 변수를 사용한 함수에서 여러 개의 출력 값을 꺼내서 역전파시킵니다.  
역전파 결과 값을 받아서 다음 변수의 기울기에 대입합니다.  
```python
class Variable : 
    ...
    def backward(self) :
        ...
        gys = [output.grad for output in func.outputs]
        gxs = func.backward(*gys)
        if not isinstance(gxs, tuple) :
            gxs = (gxs,)
        for x, gx in zip(func.inputs, gxs) :
            x.grad = gx
        
            if x.creator:
                funcs.append(x.creator)

```

### Square 클래스 수정
Square클래스도 가변 길이 인수에 맞게 수정합니다.  
```python
def backward(self, gy:np.ndarray):
    x = self.inputs[0].data
    gx = 2*x*gy
    return gx
```

## Step 14
step 14에서는 같은 변수 인스턴스를 사용 할 경우 문제가 되는 부분을 수정합니다.

### 동일 변수 사용시 이슈
동일한 변수를 그대로 사용하여 add해봤을 떄 역전파에서 문제가 발생합니다.  

```python
x = Variable(np.array(2.0))
y = add(x,x)
print(y.data)
# 4
y.backward()
print(x.grad)
# 1

```
동일한 인스턴스에 grad 인스턴스 변수 값을 덮어씌우면서 발생하는 문제입니다.

### 이슈 해결
Variable 클래스에서 grad 변수 대입 부분을 수정합니다.  
```python
class Variable :
    def backward(self) :
        ...
        if x.grad is None : 
            x.grad = gx
        else :
        x.grad = x.grad + gx
```

### 미분값 재설정
x.grad에 누적을 해줌으로서 해당 변수 값을 계속 재사용 할 시 계속 누적되는 이슈가 발생합니다.  
grad 초기화 메소드를 추가합니다.  

```python
class Variable :
    def clear_grad(self) :
        self.grad = None

```


## Step 15
현재 역전파 시 여러개의 출력 값을 가지는 케이스는 제대로 역전파를 계산하지 못하는 이슈가 존재합니다.  
해당 이슈는 역전파에서 사용 할 함수에 우선순위를 부여함으로서 해결 할 수 있습니다.

## Step 16

### generation 추가
Variable 클래스에 generation 인스턴스 변수를 추가하고 해당 클래스를 사용한 부모 함수의 generation 값에 +1을 해주면서 증가시킵니다.

```python
class Variable
    def __init__(self, data) :
        ...
        self.generation = 0

    def set_creator(self,func) :
        self.generation = func.generation + 1

```

Function 클래스에도 generation 인스턴스 변수를 추가하는데 여러 개의 인풋이 있었다면 그 중 generation 값이 큰 값을 대입합니다. 

```python
def __call__(self, *inputs):     
    self.generation = max([i.generation for i in inputs])

```


### 함수 sort
Variable 클래스에서 함수를 담은 list를 generation 값을 기준으로 sort합니다.

```python
def backward(self):
    def add_func(f:Function) :
        if f not in seen_set :
            funcs.append(f)
            seen_set.add(f)
            funcs.sort(key=lambda x: x.generation)

    funcs = []
    seen_set = set()
    add_func(self.creator)
    ...
    if x.creator:
        add_func(self.creator)

```
