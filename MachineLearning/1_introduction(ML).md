---
tags:
  - 机器学习
  - 入门
---
# 前言

在我的理解中，给出一组输入数据，给出或不给出输出数据，如何在给定这些数据的基础上得到一个函数，能够将输入数据拟合成输出数据或者预想的形式，**学习到这个函数的过程**就可以理解为机器学习。

这一部分就是概念最广的机器学习，参考了PRML和李航的统计机器学习的知识，不过多地引入深度学习等具体技术的分支。就今天来看可以说是上古时期的知识了，很多深度学习或者基于大模型的方法在很多任务上可以说吊打这些方法了，但怎么说呢，你有时候也不得不服古人的智慧，这部分内容的特点是很多模型都是基于统计学模型用最优化一些方法推导求解得到的，所以说它们的理论性与可解释性都是要强于深度学习的知识的，深度学习很多时候会依赖于一些不那么严谨的假设，而传统的机器学习表现出了其独特的白盒特性，同时在简单化轻量化的需求下，传统机器学习方法展现出了其独特的特性，因此很多方法和思想都可以整合到现在的一些工作中。

> [!info] 参考资料
> - 视频：[Bilibili - 机器学习](https://www.bilibili.com/video/BV1aE411o7qd/)
> - 笔记：[MachineLearningNotes - Intro_Math](https://github.com/tsyw/MachineLearningNotes/blob/master/1.Intro_Math.md)

---

# 两个派别

机器学习的目的是**学习描述数据的模型**，那么很自然的就可以将概率引入到机器学习中，因为可以认为 $X$ 是符合一种概率模型，这个模型可以参数化为 $\theta$。对于一组数据 $X$：

$$
X_{N\times p}=(x_{1},x_{2},\cdots,x_{N})^{T},\quad x_{i}=(x_{i1},x_{i2},\cdots,x_{ip})
$$

表示这组数据有 $N$ 个样本，每个样本都是 $p$ 维向量。其中每个观测都是由 $p(x|\theta)$ 生成的。那么怎么根据这组样本去求概率模型 $p(x|\theta)$，也就是去解 $\theta$，就是机器学习的主要任务。

而关于怎么求解，可以分为**频率派**和**贝叶斯派**两个主要观点：

## 频率派的观点

- $\theta$：未知的**常量**
- $X$：**随机变量**

有随机变量，估计概率模型的常量，就用概率论里很常用的**最大似然（MLE）**：

$$
\theta_{MLE}=\mathop{argmax}\limits _{\theta}\log p(X|\theta)\mathop{=}\limits _{iid}\mathop{argmax}\limits _{\theta}\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)
$$

> [!tip] 本质
> 频率派的本质是一个**优化问题**。

## 贝叶斯派的观点

- $\theta$：也是**随机变量**，服从概率分布 $\theta\sim p(\theta)$（先验）

先通过贝叶斯定理（后验 $\sim$ 似然 $\times$ 先验）将各概率联系起来：

$$
p(\theta|X)=\frac{p(X|\theta)\cdot p(\theta)}{p(X)}=\frac{p(X|\theta)\cdot p(\theta)}{\int\limits _{\theta}p(X|\theta)\cdot p(\theta)d\theta}
$$

我们要最大化这个参数后验，也就是取最能描述样本特点的 $\theta$，使用**后验概率最大 MAP**：

$$
\theta_{MAP}=\mathop{argmax}\limits _{\theta}p(\theta|X)=\mathop{argmax}\limits _{\theta}p(X|\theta)\cdot p(\theta)
$$

求解这个 $\theta$，也就得到了参数的后验概率 $p(\theta|X)$。

那么解了有什么用呢？可以求新数据属于原数据的概率，也就是做**贝叶斯预测**：

$$
p(x_{new}|X)=\int\limits _{\theta}p(x_{new}|\theta)\cdot p(\theta|X)d\theta
$$

> [!tip] 本质
> 贝叶斯派的关键是求后验，本质是**求积分**，延伸出概率图模型。

---

# 三范式

## 监督学习

- **定义**：使用带有标签的数据（输入-输出对）训练模型，目标是学习从输入到输出的映射。
- **特点**：
    - 数据有明确的输入和输出（例如，图片和对应的类别标签）。
    - 目标是最小化预测输出与实际标签之间的误差。
- **算法示例**：线性回归、逻辑回归、支持向量机、神经网络。

## 非监督学习

- **定义**：使用无标签的数据，模型尝试发现数据中的内在结构或模式。
- **特点**：
    - 没有预定义的输出标签，模型基于数据的分布或特征进行学习。
    - 目标是找到数据的隐藏模式或分组。
- **算法示例**：K 均值聚类、层次聚类、自编码器、PCA 降维。

## 强化学习

- **定义**：智能体（Agent）通过与环境交互，根据奖励信号学习最优策略。
- **特点**：
    - 没有明确的标签，而是通过试错获得延迟的奖励反馈。
    - 目标是最大化累计奖励。
- **算法示例**：Q-Learning、Policy Gradient、PPO。

---

# 三要素

## 模型

模型是机器学习系统用来表示输入数据和输出结果之间关系的数学函数或结构。简单来说，它是数据和预测之间的"映射"。

模型可以看作一个假设函数，用来描述输入特征 $X$ 与目标变量 $Y$ 之间的关系。例如，线性回归模型假设 $y = wX + b$。

## 策略

策略是指如何衡量模型的好坏，以及如何通过优化目标来调整模型参数。通常表现为**损失函数（Loss Function）**和优化目标。

例如线性回归最小二乘法给出的优化函数。

## 算法

算法是用于优化模型参数的具体计算方法，基于策略（损失函数）寻找最优解。简单来说，算法是"如何让模型变好"的具体步骤，这里可以具体去看优化问题部分[[Introduction(optimization)]]。

例如随机梯度下降法，就是有了策略的优化函数，具体如何求解。

---

# 两类模型

## 生成模型 (Generative Models)

生成模型的目标是学习数据的**联合概率分布** $P(X,Y)=P(X|Y)P(Y)$，进而借助贝叶斯得到条件概率 $P(Y|X)$，从而能够生成类似训练数据的新样本。

- **核心思想**：建模数据生成过程，学习数据的概率分布。
- **典型算法**：朴素贝叶斯 (Naive Bayes)、隐马尔可夫模型 (HMM) 等。
- **应用**：数据生成、数据补全、异常检测。

## 判别模型 (Discriminative Models)

判别模型的目标是学习给定输入 $X$ 的条件下，直接输出 $Y$ 的**条件概率分布** $P(Y|X)$。它专注于给定了 $X$，应该得到什么样的 $Y$。

- **典型算法**：逻辑回归、支持向量机等。

---

# 数学基础

## 概率论基础

### 均值

- **概率论（期望）**：$E[X] = \int x f(x) dx$
- **统计学（样本均值）**：$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

### 方差

$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

### 协方差

$$
\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
$$

### 边缘概率 (Marginal Probability)

描述单个随机变量的概率，忽略其他变量的影响。对于变量 $x$，边缘概率是：

$$
P(X = x) = \int P(X = x, Y = y) \, dy
$$

如果只有数据集，可以统计 $X$ 取某值的频率。

### 条件概率 (Conditional Probability)

在给定某个事件发生的情况下，另一个事件的概率。$P(X|Y)$ 表示在 $Y$ 发生时 $X$ 的概率：

$$
P(X | Y) = \frac{P(X, Y)}{P(Y)}
$$

如果有数据集，统计在取 $Y$ 值（$Y$ 发生时）$X$ 发生的频率。

### 联合概率 (Joint Probability)

多个随机变量同时发生的概率。$P(X,Y)$ 表示 $X$ 和 $Y$ 同时满足特定条件的概率。

- 如果变量**独立**：$P(X,Y)=P(X)\cdot P(Y)$
- 如果**不独立**：需通过联合概率分布表、公式或数据集统计
- 如果已知条件概率：$P(X,Y) = P(X|Y) \cdot P(Y)$

### 贝叶斯定理 (Bayes' Theorem)

由条件概率推导而来，是贝叶斯派的核心公式：

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

其中：

| 术语 | 含义 |
|------|------|
| $P(A)$ | **先验概率**：在观测数据之前对 $A$ 的信念 |
| $P(B\|A)$ | **似然**：在 $A$ 成立时观测到 $B$ 的概率 |
| $P(A\|B)$ | **后验概率**：观测到 $B$ 之后对 $A$ 的更新信念 |
| $P(B)$ | **证据/边缘似然**：归一化常数 |

### 常见概率分布

**离散分布：**

- **伯努利分布 (Bernoulli)**：单次试验，成功概率 $p$

$$
P(X=k) = p^k(1-p)^{1-k}, \quad k \in \{0, 1\}
$$

- **二项分布 (Binomial)**：$n$ 次独立伯努利试验中成功的次数

$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

- **多项分布 (Multinomial)**：将伯努利推广到 $K$ 个类别，每个类别的概率为 $p_k$，$n$ 次试验中各类别出现 $n_k$ 次的概率：

$$
P(n_1, \dots, n_K) = \frac{n!}{n_1! \cdots n_K!} \prod_{k=1}^{K} p_k^{n_k}
$$

**连续分布：**

- **高斯分布 (Gaussian / Normal)**：机器学习中最核心的分布

$$
\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

多维形式（$d$ 维）：

$$
\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中 $\boldsymbol{\mu}$ 为均值向量，$\boldsymbol{\Sigma}$ 为协方差矩阵。

- **均匀分布 (Uniform)**：在区间 $[a, b]$ 上等概率

$$
f(x) = \frac{1}{b-a}, \quad a \leq x \leq b
$$

- **Beta 分布**：定义在 $[0,1]$ 上，常用作伯努利/二项分布参数 $p$ 的先验（共轭先验）

$$
\text{Beta}(p|\alpha, \beta) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha, \beta)}
$$

### 大数定律与中心极限定理

- **大数定律**：当样本量 $n \to \infty$ 时，样本均值 $\bar{X}_n$ 依概率收敛到总体期望 $\mu$。这为用样本估计总体提供了理论基础。

$$
\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu
$$

- **中心极限定理 (CLT)**：无论总体分布如何，当样本量足够大时，样本均值的分布趋近于正态分布：

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

这解释了为什么高斯分布在机器学习中如此重要。

### 信息论基础

**信息熵 (Entropy)**：衡量随机变量的不确定性。越均匀（不确定性越大），熵越高：

$$
H(X) = -\sum_{i} P(x_i) \log P(x_i)
$$

连续情况称为**微分熵**：$H(X) = -\int f(x) \log f(x) \, dx$

**交叉熵 (Cross-Entropy)**：衡量用分布 $Q$ 去编码真实分布 $P$ 的平均编码长度，在分类任务中常用作损失函数：

$$
H(P, Q) = -\sum_{i} P(x_i) \log Q(x_i)
$$

**KL 散度 (Kullback-Leibler Divergence)**：衡量两个分布之间的"距离"（非对称）：

$$
D_{KL}(P \| Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)} = H(P, Q) - H(P)
$$

> [!note] 三者关系
> $D_{KL}(P \| Q) = H(P, Q) - H(P)$
> 最小化交叉熵等价于最小化 KL 散度（因为 $H(P)$ 是常数）。这就是为什么分类任务用交叉熵做损失函数。

### 常用不等式

- **Jensen 不等式**：若 $f$ 是凸函数，则 $f(E[X]) \leq E[f(X)]$。在推导 EM 算法、变分推断中非常关键。
- **柯西-施瓦茨不等式**：$|E[XY]|^2 \leq E[X^2] \cdot E[Y^2]$

## 矩阵论基础

### 特征值分解

对于一个 $n \times n$ 的方阵 $A$，如果存在标量 $\lambda$（特征值）和非零向量 $v$（特征向量），满足：

$$
A v = \lambda v
$$

$\lambda$ 是 $A$ 的特征值，$v$ 是对应的特征向量。进而可以实现特征值分解：

$$
A = V \Lambda V^{-1}
$$

其中 $V$ 为特征向量组成的矩阵，$\Lambda$ 为特征值组成的对角矩阵。

### 奇异值分解 (SVD)

特征值分解是对于方阵的，那么一般的对于一个 $m \times n$ 矩阵 $A$，SVD 分解形式为：

$$
A = U \Sigma V^T
$$

其中：

| 符号 | 维度 | 含义 |
|------|------|------|
| $U$ | $m \times m$ | 正交矩阵（$U^T U = I$），列向量是 $AA^T$ 的特征向量，称为**左奇异向量** |
| $\Sigma$ | $m \times n$ | "对角"矩阵，非负对角元素 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > 0$，称为**奇异值** |
| $V$ | $n \times n$ | 正交矩阵（$V^T V = I$），列向量是 $A^T A$ 的特征向量，称为**右奇异向量** |

### 正定矩阵

一个 $n \times n$ 的实对称矩阵 $A$（即 $A = A^T$）称为**正定矩阵**，如果对于任意非零向量 $x \in \mathbb{R}^n$，满足：

$$
x^T A x > 0
$$

如果 $x^T A x \geq 0$，则称为**半正定矩阵**。这里 $x^T A x$ 叫做**二次型**。

### 迹 (Trace) 与行列式 (Determinant)

**迹**：方阵对角线元素之和，常出现在矩阵求导和损失函数中：

$$
\text{tr}(A) = \sum_{i=1}^n a_{ii}
$$

常用性质：

- $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(AB) = \text{tr}(BA)$（循环置换不变性）
- $\text{tr}(A^T) = \text{tr}(A)$
- 标量的迹是自身：$a = \text{tr}(a)$，因此 $x^TAx = \text{tr}(x^TAx)$

**行列式**：衡量矩阵对空间的缩放因子，在多维高斯分布中出现：

$$
|\boldsymbol{\Sigma}|^{-1/2} \text{ 出现在高斯分布的归一化系数中}
$$

常用性质：

- $|AB| = |A| \cdot |B|$
- $|A^{-1}| = |A|^{-1}$
- $|A^T| = |A|$

### 矩阵求导

| 函数形式 | 求导结果 |
|----------|----------|
| $f = a^T x$ | $\frac{\partial f}{\partial x} = a$ |
| $f = x^T A x$ | $\frac{\partial f}{\partial x} = (A + A^T)x$（$A$ 对称时 $= 2Ax$） |
| $f = a^T X b$ | $\frac{\partial f}{\partial X} = ab^T$ |
| $f = \text{tr}(AB)$ | $\frac{\partial f}{\partial A} = B^T$ |
| $f = \text{tr}(A^T B)$ | $\frac{\partial f}{\partial A} = B$ |
| $f = \log\|X\|$ | $\frac{\partial f}{\partial X} = X^{-T}$ |

> [!tip] 布局约定
> 上表采用**分母布局**（denominator layout），即求导结果的维度与分母一致。部分教材使用分子布局，结果会转置，注意区分。

---

## 微积分基础

### 梯度 (Gradient)

标量函数 $f(\mathbf{x})$ 对向量 $\mathbf{x} = (x_1, \dots, x_n)^T$ 的梯度是一个向量，指向函数增长最快的方向：

$$
\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right)^T
$$

梯度下降法就是沿负梯度方向更新参数：$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$

### 雅可比矩阵 (Jacobian)

当函数是向量到向量的映射 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 时，其一阶偏导数构成雅可比矩阵：

$$
J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}_{m \times n}
$$

### 海森矩阵 (Hessian)

标量函数 $f(\mathbf{x})$ 的二阶偏导数构成的矩阵，用于描述函数的**曲率**：

$$
H = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}
$$

- $H$ 正定 → 该点是**严格局部极小值**
- $H$ 负定 → 该点是**严格局部极大值**
- $H$ 不定 → 该点是**鞍点**

### 链式法则 (Chain Rule)

复合函数求导的基础，也是**反向传播**算法的数学核心：

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
$$

多变量情况：若 $f$ 依赖于 $g_1, g_2, \dots, g_m$，每个 $g_i$ 又依赖于 $x$：

$$
\frac{\partial f}{\partial x} = \sum_{i=1}^{m} \frac{\partial f}{\partial g_i} \cdot \frac{\partial g_i}{\partial x}
$$

### 泰勒展开 (Taylor Expansion)

将函数在某点 $x_0$ 附近用多项式逼近，许多优化算法（牛顿法、拟牛顿法）的理论基础：

$$
f(x) = f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2}f''(x_0)(x - x_0)^2 + \cdots
$$

多元情况：

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)
$$

---

## 优化理论基础

### 凸函数与凸优化

**凸函数**定义：对于任意 $x_1, x_2$ 和 $\lambda \in [0, 1]$：

$$
f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)
$$

直观理解：函数曲线上任意两点连线都在曲线上方。

- 凸函数的局部最小值就是**全局最小值**，这使得优化问题变得容易
- 常见凸函数：$x^2$、$e^x$、$-\log x$、$\|x\|$

### 拉格朗日乘数法 (Lagrange Multipliers)

求解**等式约束**优化问题：

$$
\min f(x) \quad \text{s.t.} \quad g_i(x) = 0, \quad i = 1, \dots, m
$$

构造拉格朗日函数：

$$
\mathcal{L}(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x)
$$

令 $\nabla_x \mathcal{L} = 0$ 且 $\nabla_\lambda \mathcal{L} = 0$ 求解。SVM 的推导大量用到此方法。

### KKT 条件 (Karush-Kuhn-Tucker)

将拉格朗日乘数法推广到**不等式约束**：

$$
\min f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \quad h_j(x) = 0
$$

KKT 条件（最优解的必要条件）：

1. **平稳性**：$\nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) + \sum_j \lambda_j \nabla h_j(x^*) = 0$
2. **原始可行性**：$g_i(x^*) \leq 0$，$h_j(x^*) = 0$
3. **对偶可行性**：$\mu_i \geq 0$
4. **互补松弛**：$\mu_i g_i(x^*) = 0$

