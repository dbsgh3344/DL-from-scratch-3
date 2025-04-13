# DeZero Step 23 ~ 26 정리

## Step 23
step 23에선 기존에 작성했던 코드들을 패키지화합니다.
### 


## Step 24
step 24에서는 다양한 최적화에서 사용하는 테스트 함수들을 dezero패키지로 구현해 봅니다.   

### Sphere 함수 구현
sphere 함수를 구현합니다.  

$z= x^2 + y^2$  
$\frac{dz}{dx} = 2x$  
$\frac{dz}{dy} = 2y$

```python
def sphere(x,y):
    return x**2 + y**2
```


### matyas 함수 구현

$z= 0.26(x^2 + y^2) - 0.48xy$  
$ \frac{dz}{dx} = 0.52x - 0.48y$  
$\frac{dz}{dy} = 0.52y - 0.48x$

```python
def matyas(x,y):
    return 0.26*(x**2 + y**2) - 0.48*x*y
```


### GoldStein-Price 함수 구현


```python
def gold_stein(x,y):
    a = 1 + ((x + y + 1)**2)*(19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    b = 30 + ((2*x - 3*y)**2)*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return a*b
```


## Step 25
step 25에서는 graphviz를 설치하고 테스트합니다.  

```bash
# for mac
brew install graphviz
```

```bash
digraph g {
    x
    y
}
```


```bash
# -T : 파일 형식 지정
# -o : output 파일명 지정
dot sample.dot -T png -o sample.png
```



## Step 26
step 26에서는 dezero계산을 계산그래프로 변환하는 작업을 수행합니다.  

### 변수 데이터 변환
변수 클래스에 담긴 데이터를 계산그래프에서 반영하기 위해 변환합니다.  

```python
def _dot_var(v, verbose=False):    
    dot_var = '{} [label="{}", color=black, style=filled, fillcolor=lightblue]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += f"{v.shape} {v.dtype}"
    return dot_var.format(id(v), name)
```

### 함수 데이터 변환

```python
def _dot_func(f, verbose=False):
    dot_func = '{} [label="{}", color=lightblue, style=filled, fillcolor=lightgreen, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt
```

### 이미지 변환
dot_func, dot_var을 이용해서 digraph g{} 사이에 텍스트를 추가하여 dot 언어로 변환합니다.  
파이썬에서 dot 언어를 command로 실행하여 그래프를 저장합니다.  

```python
def get_dot_graph(output, verbose=False):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output,verbose=verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func,verbose=verbose)
        for x in func.inputs:
            txt += _dot_var(x,verbose=verbose)
            
            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'
    

def plot_dot_graph(output, verbose=False, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose=verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'),'.dezero')
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd,shell=True)
```
### 이미지 변환 using lib
파이썬의 graphviz 라이브러리를 사용해서 똑같은 작업을 수행합니다.   

```bash
pip install graphviz
```

```python
def plot_dot_graph_using_lib(output, verbose=False, to_file='graph.png'):    
    dot_graph = get_dot_graph(output, verbose=verbose)
    extension = os.path.splitext(to_file)[1][1:]
    src = Source(dot_graph)
    src.format = extension    
    src.render(to_file, cleanup=True)
```
