# DeZero Step 41 ~ 44 정리

## Step 45
45 단계에서는 Layer클래스에서 다른 Layer클래스도 담을 수 있도록 확장하고 모델 클래스를 추가합니다.

### 45.1
Layer클래스에 Layer클래스도 받을 수 있도록 코드를 수정합니다.

```python
def __setattr__(self, name, value):
        if isinstance(value, (Parameter,Layer)):    # Layer 추가
            self._params.add(name)
        super().__setattr__(name, value)    
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):  # Layer 인스턴스이면 yield from으로 파라미터 넘기기 
                yield from obj.params()
            else:
                yield obj
```

### 45.2 
Model클래스를 추가합니다.

```python
class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph_using_lib(y, verbose=True, to_file=to_file)
```


### 45.4
MLP 클래스를 구현합니다. Layer 개수를 동적으로 생성 할 수 있도록 `setattr`을 사용합니다  



```python
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid_simple):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, output_size in enumerate(fc_output_sizes):
            layer = L.Linear(output_size)
            setattr(self, f'l{i}', layer)
            self.layers.append(layer)

    def forward(self, x):    
        for l in self.layers[:-1]:
            x = l(x)
            x = self.activation(x)
        return self.layers[-1](x)
```


## Step 46

### 46.1
Optimizer 클래스를 구현합니다.

```python
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]    # 파라미터 갱신. 
        
        for f in self.hooks:
            f(params)
        
        for param in params:
            self.update_one(param)
        

    def setup (self,target):
        self.target = target    # target = model
        return self # 자기자신을 리턴함으로서 변수에 대입하는 식으로도 사용가능
    

    def add_hook(self, f):
        self.hooks.append(f)


    def update_one(self,param):
        raise NotImplementedError("This method should be overridden by subclasses.")
```

### 46.2
SGD 클래스를 구현합니다
```python

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
```

### 46.4
Momentum 클래스를 구현합니다  
Momentum 수식  
$v \leftarrow av - \eta \frac{\delta L}{\delta W}$  
$W \leftarrow W + v $


```python
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

```

## Step 47

### 47.1


```python
```
