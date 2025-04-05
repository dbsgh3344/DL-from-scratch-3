# DeZero Step 17 ~ 22 정리

## Step 17
python의 메모리 관리 두 가지 방식에 대해 설명합니다.
### 참조 카운트
객체의 카운트가 0에서 시작해서 참조 카운팅을 증가시킵니다.  
만약 참조 카운트가 0일 경우 메모리에서 삭제됩니다.  
참조 카운트 증가 조건
- 대입 연사자를 사용할 떄
- 함수에 인수로 전달할 때
- 컨테이너 타입 객체에 추가할 떄
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
<img src="https://github.com/user-attachments/assets/" width=40%>
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
