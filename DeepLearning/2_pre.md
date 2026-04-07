---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: ml
    language: python
    name: python3
---

# 第2章 预备知识

参考资料：[动手学深度学习 v2](https://zh-v2.d2l.ai/chapter_preliminaries/index.html)

```python
import torch
import numpy as np
import pandas as pd
import os
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from torch.distributions import multinomial


print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 是否可用: {torch.cuda.is_available()}')
```

### 2.1 数据操作

深度学习中最基本的数据结构是 **张量**（tensor）——n 维数组。
与 NumPy 不同，PyTorch 张量支持 GPU 加速和自动微分。


**基础定义与维度变化**

```python
# 创建张量：使用 arange 生成从 0 到 11 的整数行向量
x = torch.arange(12)
print('x:', x)
print('形状 shape:', x.shape)
print('元素总数 numel:', x.numel())
```

```python
# 改变张量形状，不改变元素数量和值
X = x.reshape(3, 4)
print('reshape 后:\n', X)

# 用 -1 自动推断维度
X2 = x.reshape(-1, 4)   # 等价于 reshape(3, 4)
print('\n使用 -1 自动推断:\n', X2)
```

```python
# 全零、全一、随机张量
zeros = torch.zeros(2, 3, 4)
ones  = torch.ones(2, 3, 4)
rand  = torch.randn(3, 4)   # 标准正态分布

print('全零张量形状:', zeros.shape)
print('全一张量形状:', ones.shape)
print('随机张量:\n', rand)
```

```python
# 通过 Python 列表直接创建张量
t = torch.tensor([[2, 1, 4, 3],
                  [1, 2, 3, 4],
                  [4, 3, 2, 1]])
print('从列表创建:\n', t)
```

**基础运算符**

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2.0, 2, 2, 2])

print('x + y =', x + y)
print('x - y =', x - y)
print('x * y =', x * y)
print('x / y =', x / y)
print('x ** y =', x ** y)  # 幂运算
print('exp(x) =', torch.exp(x))
```

```python
# 张量拼接：沿不同维度
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3],
                  [1,   2, 3, 4],
                  [4,   3, 2, 1]])

print('沿行(dim=0)拼接:\n', torch.cat([X, Y], dim=0))
print('\n沿列(dim=1)拼接:\n', torch.cat([X, Y], dim=1))
```

**广播机制**：当两个张量形状不完全相同时，可以通过 **广播（broadcasting）** 机制自动扩展维度后再运算。

```python
a = torch.arange(3).reshape(3, 1)  # 形状 (3, 1)
b = torch.arange(2).reshape(1, 2)  # 形状 (1, 2)

print('a:\n', a)
print('b:\n', b)
print('\na + b (广播后形状 3×2):\n', a + b)
```

**索引和切片**

```python
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)

print('X:\n', X)
print('\n最后一行 X[-1]:', X[-1])
print('第1~2行 X[1:3]:\n', X[1:3])

# 写入操作
X[1, 2] = 9
print('\n修改 X[1,2]=9 后:\n', X)

# 批量写入
X[0:2, :] = 12
print('\n前两行全部改为 12:\n', X)
```

**in-place操作**：可以避免创建新的内存空间，提高效率。

```python
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.ones(3, 4)

# 普通赋值：会创建新对象
before = id(Y)
Y = Y + X
print('普通赋值后 id 改变:', id(Y) != before)  # True

# 就地操作：不创建新对象
Z = torch.zeros_like(Y)
before = id(Z)
Z[:] = X + Y
print('就地操作后 id 不变:', id(Z) == before)  # True

# 也可以使用 += 运算符
X += Y
print('使用 += 后 X:\n', X)
```

---
### 2.2 数据预处理

真实数据往往以 CSV 等格式存储，需要经过读取、清洗、转换才能输入模型。
本节使用 **pandas** 完成这一流程。

```python
# 创建示例数据集：房屋信息
os.makedirs('data', exist_ok=True)
data_file = 'data/house_tiny.csv'

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')   # 列名
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)
```

```python
# 分离输入特征与输出标签
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print('输入特征:\n', inputs)
print('\n输出标签:\n', outputs)
```

```python
# 处理缺失值
# 数值列：用均值填充
inputs = inputs.copy()
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())

# 类别列：独热编码（缺失值单独作为一类）
inputs = pd.get_dummies(inputs, dummy_na=True)
print('处理缺失值后:\n', inputs)
```

```python
# 转换为 PyTorch 张量
X = torch.tensor(inputs.to_numpy(dtype=float), dtype=torch.float32)
y = torch.tensor(outputs.to_numpy(dtype=float), dtype=torch.float32)
print('X (特征张量):\n', X)
print('y (标签张量):', y)
```

---
### 2.3 数学基础

深度学习归根到最后还是数学的知识，包括线性代数、微积分、概率论等。那么，这些数学基础理论所涉及到的公式和运算又在深度学习框架中是怎么使用和表示的，这是这部分内容的核心。


#### 2.3.1 线性代数
线性代数主要负责数据的表示、变换和计算

```python
# 标量定义和计算
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print('x + y =', x + y)
print('x * y =', x * y)
print('x / y =', x / y)
print('x ** y =', x ** y)

# 向量
x = torch.arange(4)
print('形状 x.shape:', x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print('矩阵 A:\n', A)
print('\n转置 A.T:\n', A.T)

# 张量
X = torch.arange(24).reshape(2, 3, 4)
print('三维张量 X:\n', X)
```

线性代数基本运算

```python
# 点积
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print('x:', x, '  y:', y)
print('点积 dot(x,y):', torch.dot(x, y))
```

```python
# 矩阵-向量积
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print('A 形状:', A.shape, '  x 形状:', x.shape)
print('矩阵-向量积 mv(A,x):', torch.mv(A, x))
```

```python
# 矩阵-矩阵乘法
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print('A 形状:', A.shape)
B = torch.ones(4, 3)
print('矩阵乘法 mm(A,B) 形状:', torch.mm(A, B).shape)
print(torch.mm(A, B))
```

**范数**用来衡量向量的「大小」，在深度学习中被广泛用于正则化和优化目标：

- **L2 范数（欧几里得范数）**：$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$
- **L1 范数**：$\|\mathbf{x}\|_1 = \sum_i |x_i|$
- **Frobenius 范数**（矩阵）：$\|\mathbf{A}\|_F = \sqrt{\sum_{ij} A_{ij}^2}$

```python
u = torch.tensor([3.0, -4.0])

print('L2 范数:', torch.norm(u))
print('L1 范数:', torch.abs(u).sum())

# 矩阵的 Frobenius 范数
M = torch.ones(4, 9)
print('Frobenius 范数:', torch.norm(M))
```

---
#### 2.3.2 微积分

优化深度学习模型参数的核心工具是**梯度（gradient）**，来自微积分中的导数概念。



函数 $f(x)$ 在点 $x$ 处的导数定义为：
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

常用微分规则：
- 常数：$\frac{d}{dx}C = 0$
- 幂次：$\frac{d}{dx}x^n = nx^{n-1}$
- 指数：$\frac{d}{dx}e^x = e^x$
- 对数：$\frac{d}{dx}\ln x = \frac{1}{x}$

```python
def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    """数值方法计算极限（近似导数）"""
    return (f(x + h) - f(x)) / h

# 在 x=1 处，f'(x) = 6x - 4 = 2
print(f'理论值 f\'(1) = {6*1 - 4}')
print('\nh 趋向 0 时的数值极限：')
h = 0.1
for i in range(5):
    print(f'  h={h:.5f},  数值极限 = {numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```


对多元函数 $f(x_1, x_2, \ldots, x_n)$，对 $x_i$ 的**偏导数**为：
$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i+h, \ldots) - f(\ldots, x_i, \ldots)}{h}$$

**链式法则**：对复合函数 $y = f(u)$，$u = g(x)$：
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**梯度**是所有偏导数组成的向量：
$$\nabla_\mathbf{x} f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top$$






深度学习框架通过**自动微分（autograd）**，根据设计好的模型自动计算导数。
PyTorch 通过构建**计算图**来追踪运算，并利用反向传播求梯度。


这里我们计算 $y = 2\mathbf{x}^\top\mathbf{x}$ 对向量 $\mathbf{x}$ 的梯度，
理论结果为 $\nabla y = 4\mathbf{x}$。

```python
x = torch.arange(4.0)
print('x:', x)

# 为 x 分配梯度存储空间
x.requires_grad_(True)
print('x.grad (初始):', x.grad)  # None

# 计算 y
y = 2 * torch.dot(x, x)
print('y = 2 * x^T x =', y)

# 反向传播
y.backward()
print('\n梯度 x.grad:', x.grad)
print('验证 grad == 4x:', x.grad == 4 * x)

# 注意：PyTorch 默认梯度累积，每次需清零
x.grad.zero_()
```

分离计算：有时需要将某些变量从计算图中分离，使其被当作常数处理。

```python
x.grad.zero_()

y = x * x
u = y.detach()  # 将 y 作为常数，不追踪梯度
z = u * x       # dz/dx = u（将 u 看作常数）

z.sum().backward()
print('z = u * x 的梯度 (u 为常数):', x.grad)
print('等于 u = x*x:', u)
print('验证:', x.grad == u)
```

自动微分的强大之处在于：即使函数包含 `if`/`while` 等控制流，也能正确计算梯度。

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# f 实际上是分段线性函数：d = k*a，所以 grad = k = d/a
print('a:', a)
print('d:', d)
print('梯度 a.grad:', a.grad)
print('验证 grad == d/a:', a.grad == d / a)
```

---
#### 2.3.3 概率

机器学习的核心是从数据中做出预测和决策，概率论为此提供了数学基础，这里核心要学会的东西是分布采样。

```python
# 模拟抛骰子：每个面概率均为 1/6
fair_probs = torch.ones(6) / 6

# 多项分布采样：投 1 次，结果为独热向量
sample = multinomial.Multinomial(1, fair_probs).sample()
print('单次投掷结果:', sample)

# 投 1000 次并统计频率
counts = multinomial.Multinomial(1000, fair_probs).sample()
print('\n1000次投掷计数:', counts)
print('频率:', counts / 1000)
```

```python
# 大数定律：投掷次数越多，频率越接近概率
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
```

期望与方差
- **期望**：$E[X] = \sum_x x P(X=x)$（连续：$E[X] = \int x p(x)\,dx$）
- **方差**：$\text{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2$
- **标准差**：$\sigma = \sqrt{\text{Var}[X]}$

```python
# 均匀骰子的期望与方差
faces = torch.arange(1, 7, dtype=torch.float32)
probs = torch.ones(6) / 6

expectation = (faces * probs).sum()
variance = ((faces - expectation)**2 * probs).sum()
std_dev = variance.sqrt()

print(f'期望 E[X] = {expectation:.4f}')      # 理论值 3.5
print(f'方差 Var[X] = {variance:.4f}')       # 理论值 35/12 ≈ 2.9167
print(f'标准差 σ = {std_dev:.4f}')           # 理论值 ≈ 1.7078

# 使用 torch 内置函数验证
samples = multinomial.Multinomial(10000, probs).sample()
# 还原为观测值
obs = torch.repeat_interleave(faces, samples.long())
print(f'\n样本均值（10000次）: {obs.mean():.4f}')
print(f'样本标准差（10000次）: {obs.std():.4f}')
```
