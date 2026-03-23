---
title: 核方法 (Kernel Method)
tags:
  - SVM
  - 核方法
  - 机器学习
aliases:
  - Kernel Method
  - 核技巧
---

# 核方法 (Kernel Method)

## 核心思想

### (1) 非线性带来高维转换

很多分类问题是非线性的，比如很著名的异或问题，这样传统的分类算法就失效了，这时该如何处理？

| 线性可分   | 允许部分错误 | 线性不可分             |
| -------- | ---------- | -------------------- |
| 感知机    | pocket算法  | 多层感知机（神经网络）    |
| 硬间隔SVM | 软间隔SVM   | ==非线性转换+SVM==     |

核方法的思想基础就是能否做一个显性的非线性转换处理（一般会转换到高维，因为高维是比低维更容易线性可分的）：

$$
x=(x_1,x_2,...,x_p)\xrightarrow{\phi}z=(z_1,z_2,...,z_q),\quad q\gg p
$$

![[MachineLearning/6_SVM/assets/5.png]]
### (2) 对偶表示带来内积

我们观察上面的 SVM 对偶问题：

$$
\max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i,\ s.t.\ \lambda_i\ge0
$$

其中是要求一个内积 $x_i^Tx_j$，而对于非线性可分是要先转换后再求内积 $\phi(x_i)^T\phi(x_j)$ 的，本身 $\phi(x_i)$ 是高维的就很难求了，而其内积可能就更加难求。那么我们就想有没有一种更好的方法把这个内积求出来呢？核方法就针对这个问题而产生，通过定义核函数，尝试直接获得这个内积：

$$
K(x,x')=\phi(x)^T\phi(x')
$$

## 核函数定义

**核函数**：$\forall x,x'\in\mathcal{X}$，函数 $k:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ 将两个输入映射为一个实数，称 $k(x,x')$ 为一个核函数。

**正定核函数**：

$$
\forall x,x'\in\mathcal{X},\exists \phi \in\mathcal{H}:x\rightarrow z\ s.t.\ k(x,x')=\phi(x)^T\phi(x')
$$

称 $k(x,x')$ 为一个正定核函数，其中 $\mathcal{H}$ 是 ==Hilbert 空间==【完备的（序列的极限操作后也在空间里），可能无限维的，赋予内积的线性空间】。如果去掉内积这个条件我们简单地称为核函数。

### 正定核的等价定义

正定核函数有下面的等价定义——如果核函数满足：

1. **对称性** $\Leftrightarrow$ $k(x,z)=k(z,x)$，显然满足内积的定义
2. **正定性** $\Leftrightarrow$ $\forall N,x_1,x_2,\cdots,x_N\in\mathcal{X}$，对应的 Gram Matrix $K=[k(x_i,x_j)]$ 是半正定的

那么这个核函数是正定核函数。

> [!abstract]- 证明：正定核的等价性
> 对称性是显然的，那么证明正定性就行。
>
> **要证**：$k(x,z)=\phi(x)^T\phi(z)\Leftrightarrow K$ Gram Matrix 半正定。
>
> **必要性** $\Rightarrow$：对于正定性：
>
> $$
> K=\begin{pmatrix}k(x_1,x_2)&\cdots&k(x_1,x_N)\\\vdots&\vdots&\vdots\\k(x_N,x_1)&\cdots&k(x_N,x_N)\end{pmatrix}
> $$
>
> 任意取 $\alpha\in\mathbb{R}^N$，即需要证明 $\alpha^TK\alpha\ge0$：
>
> $$
> \alpha^TK\alpha=\sum\limits_{i,j}\alpha_i\alpha_jK_{ij}=\sum\limits_{i,j}\alpha_i\phi^T(x_i)\phi(x_j)\alpha_j=\sum\limits_{i}\alpha_i\phi^T(x_i)\sum\limits_{j}\alpha_j\phi(x_j)
> $$
>
> 这个式子就是内积的形式，Hilbert 空间满足线性性，于是正定性得证。
>
> **充分性** $\Leftarrow$：对 $K$ 进行分解，对于对称矩阵 $K=V\Lambda V^T$，那么令 $\phi(x_i)=\sqrt{\lambda_i}V_i$，其中 $V_i$ 是特征向量，于是就构造了 $k(x,z)=\sqrt{\lambda_i\lambda_j}V_i^TV_j$。

## 常用核函数

- **线性核函数 (Linear Kernel)**
  - 公式：$K(x,y)=x^Ty$
  - 特点：适用于线性可分的数据，计算简单，就是最基础的。

- **多项式核函数 (Polynomial Kernel)**
  - 公式：$K(x, y) = (x^T y + c)^d$，其中 $c$ 是常数，$d$ 是多项式的次数。
  - 特点：能处理非线性关系，适合数据具有多项式特征的情况。

- **高斯核函数 (Gaussian Kernel / RBF Kernel)**
  - 公式：$K(x, y) = \exp(-\gamma \|x - y\|^2)$，也常写为 $K(x,y)=\exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$
  - 参数：$\gamma=\frac{1}{2\sigma^2}>0$ 控制高斯函数的"宽度"——$\gamma$ 越大，核函数越局部化（只有非常近的点才有高相似度），模型越复杂；$\gamma$ 越小，核函数越平滑，模型越简单。
  - 特点：非常灵活，能适应复杂的非线性关系，广泛应用于各种数据集。==高斯核对应的特征映射是无限维的==——通过泰勒展开可以看出 $\exp(-\gamma\|x-y\|^2)$ 等价于在一个无限维 Hilbert 空间中的内积，这意味着它理论上可以拟合任意复杂的决策边界。

- **Sigmoid 核函数 (Sigmoid Kernel)**
  - 公式：$K(x, y) = \tanh(\alpha x^T y + c)$，其中 $\alpha$ 和 $c$ 是参数。
  - 特点：类似于神经网络的激活函数，适用于特定非线性问题，但使用较少。

> [!info] 核函数的组合性质
> 核函数可以通过以下方式构造新的核函数：
> - 若 $k_1,k_2$ 是核函数，则 $k_1+k_2$ 也是核函数
> - 若 $k_1$ 是核函数，$c>0$ 为正常数，则 $ck_1$ 也是核函数
> - 若 $k_1,k_2$ 是核函数，则 $k_1\cdot k_2$（逐元素乘积）也是核函数

### 多项式核的例子

以二维输入 $x=(x_1,x_2)$ 和二次多项式核 $K(x,y)=(x^Ty+1)^2$ 为例，展开：

$$
K(x,y)=(x_1y_1+x_2y_2+1)^2=x_1^2y_1^2+x_2^2y_2^2+2x_1x_2y_1y_2+2x_1y_1+2x_2y_2+1
$$

对应的特征映射为：

$$
\phi(x)=(x_1^2,\ x_2^2,\ \sqrt{2}x_1x_2,\ \sqrt{2}x_1,\ \sqrt{2}x_2,\ 1)^T
$$

可以验证 $K(x,y)=\phi(x)^T\phi(y)$。原始 2 维输入被映射到了 6 维空间，核函数让我们==无需显式计算这个高维映射==就能直接得到内积结果。

## 核化 SVM

有了核函数，就可以将 SVM 的对偶问题中所有涉及内积的地方用核函数替换，这就是==核技巧 (Kernel Trick)==。

### 对偶问题的核化

回顾 [[SVM]] 中软间隔 SVM 的对偶问题：

$$
\max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i\quad s.t.\ 0\le\lambda_i\le C
$$

将内积 $x_i^Tx_j$ 替换为核函数 $K(x_i,x_j)$：

$$
\max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jK(x_i,x_j)+\sum\limits_{i=1}^N\lambda_i\quad s.t.\ 0\le\lambda_i\le C
$$

### 决策函数的核化

原始 SVM 决策函数为：

$$
f(x)=\text{sign}(w^Tx+b)=\text{sign}\left(\sum\limits_{i=1}^N\lambda_iy_ix_i^Tx+b\right)
$$

核化后变为：

$$
f(x)=\text{sign}\left(\sum\limits_{i=1}^N\lambda_iy_iK(x_i,x)+b\right)
$$

其中 $b$ 由支持向量（$0<\lambda_k<C$）计算：

$$
b=y_k-\sum\limits_{i=1}^N\lambda_iy_iK(x_i,x_k)
$$

> [!important] 核技巧的本质
> 我们从头到尾都不需要知道 $\phi(x)$ 的具体形式，只需要能够计算核函数 $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ 就够了。这使得我们可以在**隐式的高维（甚至无限维）空间**中进行线性分类，而计算代价仅取决于样本数 $N$，不取决于特征空间的维度。

## 核函数选择指南

| 核函数          | 适用场景                           | 优缺点                             |
| ------------- | -------------------------------- | --------------------------------- |
| **线性核**      | 特征维度高、样本量大（如文本分类）       | 计算快，但只能处理线性可分             |
| **多项式核**    | 特征之间存在交互关系                  | 可控复杂度（调 $d$），但高次计算量大    |
| **高斯核 (RBF)** | ==默认首选==，大多数非线性问题         | 灵活性强，但需调 $\gamma$，易过拟合   |
| **Sigmoid 核** | 特定场景（模拟神经网络）               | 不一定满足正定性，使用较少             |

> [!tip] 实践经验
> - **优先尝试 RBF 核**，它是最通用的选择
> - 如果特征维度 $p$ 远大于样本数 $N$（如基因数据、文本 TF-IDF），**线性核**通常就足够了，此时高维映射反而可能导致过拟合
> - 调参时，$C$ 和 $\gamma$（或其他核参数）需要**联合调参**，通常用网格搜索 + 交叉验证
> - 使用核函数前，建议对特征做**标准化**处理（均值为0，方差为1），因为核函数中的距离/内积对特征尺度敏感

## 总结

SVM 在实际应用中就可以用不同的核函数处理不同分类数据。核方法的核心流程：

1. **选择核函数** $K(x,x')$（根据数据特点和先验知识）
2. **构建核矩阵** $K_{ij}=K(x_i,x_j)$，替换对偶问题中的内积
3. **求解对偶问题**，得到 $\lambda_i$（使用 SMO 等算法）
4. **预测**：利用核化的决策函数 $f(x)=\text{sign}\left(\sum_i\lambda_iy_iK(x_i,x)+b\right)$
