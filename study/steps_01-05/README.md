# DeZero Step 1 ~ 5 정리

## Step 1
### 변수 구현
입력받은 데이터를 변수에 할당하는 로직을 구현하는 단계입니다.  
데이터를 Variable Class에 할당하면서 데이터의 타입을 정의합니다. 

***(Variable class in base_class.py )***  
```python
data = 1
x = Variable(data)
print(type(x))
# <class '__main__.Variable'>

```

### numpy 이해
numpy는 다차원 배열을 표현할 수 있는 데이터 구조입니다.  
스칼라, 벡터, 행렬을 구현하기 위해 아래와 같이 numpy를 사용하며 shape을 통해 데이터의 형태를 확인 할 수 있습니다.

```python
# 0차원 (스칼라)
arr = np.array(5) , arr.shape
# >>> 5 , ()

# 1차원 (벡터)
arr = np.array([1,2]) , arr.shape
# >>>  [1,2], (2,)

# 2차원 (행렬)
arr = np.array([[1,2],[2,3]]) , arr.shape
# >>> [[1,2][2,3]] , (2,2)
```

## Step 2
### 함수 구현
여기서 함수를 구현할 때 `Variable` 인스턴스를 받아서 `Variable` 인스턴스로 출력해주도록 구현해야 합니다.  
이후에 `함수 연결`을 구현하기 위해 함수의 이러한 구조가 필요합니다.  

***(Fuction class in base_class.py )***  
```python
x = Variable(data)
f = Function()
y = f(x)
print(type(y))
# <class '__main__.Variable'>
print(y.data)   # Variable 인스턴스의 데이터 값은 data변수에 할당되어 있다.
```


## Step 3
### 함수 연결
함수의 출력을 바로 다른 함수의 입력으로 줄 수 있습니다.  

```python
s = Square()
e = Exp()
s2 = Square()

x = Variable(np.array(2.0))
y = s2(e(s(x)))
print(y.data)

```

## Step 4
### 수치 미분
수치 미분은 미세한 차이를 이용한 미분을 말하며 컴퓨터와 같이 극한을 표현하기 힘들 때 사용할 수 있습니다.  
수치 미분은 진정한 미분에 근사하지만 오차가 발생하는데, 이 오차를 줄이기 위해 중앙 차분을 사용합니다.  
중앙 차분은 기존 식에서 아래와 같이 변경됩니다.

$\frac{f(x + h) - f(x)}{h}$ -> $\frac{f(x + h) - f(x - h)}{2h}$

### 중앙 차분 구현
***(centered_diff in numerical_diff_impl.py)***
```python
f = Square() 
x = Variable(np.array(3.0))
nu = NumericDiff()
dy = nu.centered_diff(f,x)
print(dy)
# 6.000000000012662
```
### 수치 미분의 문제점
수치 미분은 미세하지만 오차가 발생하며 계산량이 많다는 단점이 있습니다.

## Step 5
### 연쇄 법칙
연쇄법칙에 따르면 합성함수의 미분은 각 함수를 미분해서 곱한 값과 같습니다.

<img src="https://github.com/user-attachments/assets/4690618b-548b-49d3-a9dd-88646673c39b" width=40%>

### 역전파
합성함수의 곱을 출력 방향에서부터 계산해 나갑니다.  

$\frac{dy}{dx} = ((\frac{dy}{dy}\frac{dy}{db})\frac{db}{da})\frac{da}{dx}$  

<!-- $\frac{dy}{dx}$ <- $A'(x)$ <- $\frac{dy}{dx}$   -->

<img src="https://github.com/user-attachments/assets/80868984-15be-4f4e-b3eb-8cfe695b2cc3" width=40%>  
<br>  

$\frac{dy}{db} = C'(b)$ , $\frac{db}{da} = B'(a)$ , $\frac{da}{dx} = A'(x)$ 이므로 아래 그림으로 표현될 수 있습니다. 

<br>
<img src="https://github.com/user-attachments/assets/ebb06ac6-6a84-4720-a8b7-db71f787816c" width=40%>
<br>

y의 미분 값이 모든 곳에서 전파되고 있음을 볼 수 있습니다.  
다만, 역전파를 계산하기 위해선 순전파 당시의 계산 값을 모두 들고 있어야만 합니다.   




![이미지 출처](https://github.com/WegraLee/deep-learning-from-scratch-3/tree/master)