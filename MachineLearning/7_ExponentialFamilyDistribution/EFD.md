---
title: 指数族分布
tags:
  - MachineLearning
  - 概率论
  - 指数族
aliases:
  - Exponential Family Distribution
  - EFD
---

# 指数族分布

## 定义

指数族是一类分布，包括**高斯分布、伯努利分布、二项分布、泊松分布、Beta 分布、Dirichlet 分布、Gamma 分布**等一系列分布。

指数族分布可以写为统一的形式：

$$
p(x|\eta)=h(x)\exp\left(\eta^T\phi(x)-A(\eta)\right)=\frac{1}{\exp(A(\eta))}h(x)\exp\left(\eta^T\phi(x)\right)
$$

其中各项的含义为：

| 符号 | 名称 | 说明 |
|------|------|------|
| $\eta$ | 自然参数（natural parameter） | 参数向量，又称规范参数 |
| $\phi(x)$ | 充分统计量（sufficient statistic） | 从数据中提取的统计量 |
| $A(\eta)$ | 对数配分函数（log-partition function） | 起归一化作用，保证概率积分为 1 |
| $h(x)$ | 底测度（base measure） | 与参数无关的标量函数 |

## 三大性质

### (1) 充分统计量

$\phi(x)$ 叫做充分统计量，包含样本集合所有的信息。有了这个量，样本就可以扔掉了。

> [!example] 高斯分布的充分统计量
> 高斯分布中的充分统计量为 $\phi(x)=\begin{pmatrix}x \\ x^2\end{pmatrix}$，包含了计算均值 $\mu$ 和方差 $\sigma^2$ 所需的全部信息。

充分统计量在**在线学习**中有应用：对于一个数据集，只需要记录样本的充分统计量即可，无需保留原始数据。

### (2) 共轭先验

$$
p(z|x) \propto p(x|z)\,p(z)
$$

给定一个似然 $p(x|z)$，如果它具有一个与其**共轭**的先验 $p(z)$，那么后验与先验有相同的分布形式，以此简化贝叶斯推断。

> [!example] 共轭先验的例子
> - **Beta-Bernoulli 共轭**：伯努利似然 + Beta 先验 → Beta 后验
> - **Gaussian-Gaussian 共轭**：高斯似然（已知方差） + 高斯先验 → 高斯后验
> - **Dirichlet-Multinomial 共轭**：多项式似然 + Dirichlet 先验 → Dirichlet 后验

### (3) 最大熵

最大熵原理是在满足所有已知约束条件（如均值、方差或其他统计量）的情况下，选择熵最大（最随机）的概率分布。给定一组随机的数据，通过最大熵原则推出来的无信息先验 $p(z)$ 就是符合指数族分布的。

> [!tip] 直观理解
> 在"只知道一些统计量"的条件下，指数族分布是"最不做额外假设"的分布。

## 引申方法

### (1) 广义线性模型（GLM）

广义线性模型由三部分组成：

1. **随机分量**：响应变量 $y$ 服从指数族分布，$y|x \sim \text{ExpFamily}$
2. **线性组合**：$\eta = w^T x$，称为线性预测器
3. **链接函数（link function）**：$g(\mathbb{E}[y]) = \eta$，将分布的均值与线性预测器关联

$$
y = f(w^Tx), \quad g(\mu) = w^Tx, \quad y|x \sim \text{ExpFamily}
$$

> [!example] GLM 的特例
> - 链接函数为恒等函数 → **线性回归**（高斯分布）
> - 链接函数为 logit 函数 → **逻辑回归**（伯努利分布）
> - 链接函数为 log 函数 → **泊松回归**（泊松分布）

### (2) 概率图模型（PGM）

概率图模型利用图结构表示多个随机变量之间的联合分布。指数族分布在概率图模型中扮演重要角色：

- **无向图模型（马尔可夫随机场）**：联合分布天然具有指数族的形式，即 $p(x) \propto \exp\left(\sum_c \psi_c(x_c)\right)$，其中 $\psi_c$ 为团势函数。
- **有向图模型（贝叶斯网络）**：各条件分布若为指数族，整体推断更高效，共轭性使后验计算简化。

### (3) 变分推断（Variational Inference）

变分推断将后验推断问题转化为优化问题。在**平均场变分推断**中，假设变分分布 $q(z)$ 可分解为各变量的乘积：

$$
q(z) = \prod_{i} q_i(z_i)
$$

当模型中的条件分布属于指数族时，每个最优变分因子 $q_i^*(z_i)$ 也属于指数族，其自然参数由其他变量的期望充分统计量决定，极大地简化了变分更新公式（CAVI 算法）。

---

## 一维高斯分布的指数族形式

一维高斯分布：

$$
p(x|\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

将这个式子改写为指数族形式：

$$
\begin{aligned}
&= \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x^2-2\mu x+\mu^2)\right) \\[6pt]
&= \exp\left(\log(2\pi\sigma^2)^{-1/2}\right)\exp\left(-\frac{1}{2\sigma^2}\begin{pmatrix}-2\mu & 1\end{pmatrix}\begin{pmatrix}x\\x^2\end{pmatrix}-\frac{\mu^2}{2\sigma^2}\right) \\[6pt]
&= \exp\left\{\begin{pmatrix}\frac{\mu}{\sigma^2} & -\frac{1}{2\sigma^2}\end{pmatrix}\begin{pmatrix}x\\x^2\end{pmatrix}-\left(\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log(2\pi\sigma^2)\right)\right\}
\end{aligned}
$$

> [!note] 注意
> 此处 $h(x)=1$，底测度为常数。

因此各参数对应为：

$$
\eta=\begin{pmatrix}\frac{\mu}{\sigma^2}\\[4pt]-\frac{1}{2\sigma^2}\end{pmatrix}=\begin{pmatrix}\eta_1\\\eta_2\end{pmatrix}, \qquad \phi(x)=\begin{pmatrix}x\\x^2\end{pmatrix}
$$

从 $\eta$ 反解原始参数：$\mu = -\dfrac{\eta_1}{2\eta_2}$，$\sigma^2 = -\dfrac{1}{2\eta_2}$

于是对数配分函数 $A(\eta)$：

$$
A(\eta) = \frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log(2\pi\sigma^2) = -\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\log\left(-\frac{\pi}{\eta_2}\right)
$$

---

## 对数配分函数与充分统计量的关系

对于指数族分布：

$$
p(x|\eta)=h(x)\exp\left(\eta^T\phi(x)-A(\eta)\right) = \frac{1}{\exp(A(\eta))}h(x)\exp\left(\eta^T\phi(x)\right)
$$

由归一化条件（概率密度积分为 1）得：

$$
\exp(A(\eta))=\int h(x)\exp\left(\eta^T\phi(x)\right)\,dx
$$

**两边对 $\eta$ 求导**（一阶导）：

$$
\begin{aligned}
\exp(A(\eta))\,A'(\eta) &= \int h(x)\exp\left(\eta^T\phi(x)\right)\phi(x)\,dx \\[6pt]
A'(\eta) &= \frac{\int h(x)\exp\left(\eta^T\phi(x)\right)\phi(x)\,dx}{\exp(A(\eta))} \\[6pt]
&= \int h(x)\exp\left(\eta^T\phi(x)-A(\eta)\right)\phi(x)\,dx \\[6pt]
&= \int p(x|\eta)\,\phi(x)\,dx
\end{aligned}
$$

> [!important] 一阶导 = 充分统计量的期望
> $$A'(\eta) = \mathbb{E}_{p(x|\eta)}[\phi(x)]$$

类似地，对 $\eta$ 再求一次导（二阶导）：

> [!important] 二阶导 = 充分统计量的方差
> $$A''(\eta) = \mathrm{Var}_{p(x|\eta)}[\phi(x)]$$

由于方差恒为正，$A''(\eta) > 0$，因此 $A(\eta)$ 一定是**凸函数**。

> [!summary] 总结
> - $A'(\eta) = \mathbb{E}[\phi(x)]$：对数配分函数的**一阶导**是充分统计量的**期望（均值）**
> - $A''(\eta) = \mathrm{Var}[\phi(x)]$：对数配分函数的**二阶导**是充分统计量的**方差**
> - $A(\eta)$ 是凸函数（由方差非负保证）

---

## 充分统计量与极大似然估计

从极大似然估计的角度来看参数 $\eta$ 怎么求。

对于独立同分布采样得到的数据集 $\mathcal{D}=\{x_1,x_2,\cdots,x_N\}$：

$$
\begin{aligned}
\eta_{\text{MLE}} &= \mathop{\arg\max}_\eta \sum_{i=1}^N \log p(x_i|\eta) \\[6pt]
&= \mathop{\arg\max}_\eta \sum_{i=1}^N \log\left(h(x_i)\exp\left(\eta^T\phi(x_i)-A(\eta)\right)\right) \\[6pt]
&= \mathop{\arg\max}_\eta \sum_{i=1}^N \left(\eta^T\phi(x_i)-A(\eta)\right)
\end{aligned}
$$

对 $\eta$ 求导令其为零：

$$
\sum_{i=1}^N \phi(x_i) - N\,A'(\eta) = 0
$$

> [!important] MLE 的充分统计量条件
> $$A'(\eta_{\text{MLE}}) = \frac{1}{N}\sum_{i=1}^N \phi(x_i)$$
> 即**对数配分函数在 MLE 处的一阶导 = 充分统计量的样本均值**。

由此可以看到，$A(\eta)$ 的函数形式是已知的，$A'(\eta)$ 也是已知的，右侧的样本均值可直接从数据计算，因此可以解出 $\eta_{\text{MLE}}$。又因为 $A(\eta)$ 是凸函数，所以该方程有唯一解。

---

## 最大熵原理与指数族

### 信息量与信息熵

**信息量**：$-\log p$

> [!note] 直觉
> 概率 $p$ 越大（事件越容易发生），信息量越少；小概率事件包含的信息量更大。

**信息熵**：

$$
H(p) = \mathbb{E}[-\log p] = -\int p(x)\log p(x)\,dx
$$

> [!tip] 熵的含义
> 熵是"不确定性"的度量。对于完全随机的变量（等可能分布），信息熵最大。

### 最大熵原则

最大熵原则主张：在既定事实（已知约束条件）下，选择熵最大的概率分布作为最合理的分布。

> [!abstract]- 离散情况下的最大熵（等可能性证明）
> 假设数据是离散分布的，$K$ 个特征的概率分别为 $p_k$，最大熵原理可以表述为：
>
> $$
> \max\{H(p)\} = \min\left\{\sum_{k=1}^K p_k\log p_k\right\} \quad \text{s.t.} \quad \sum_{k=1}^K p_k = 1
> $$
>
> 利用 Lagrange 乘子法：
>
> $$
> \mathcal{L}(p,\lambda) = \sum_{k=1}^K p_k\log p_k + \lambda\left(1-\sum_{k=1}^K p_k\right)
> $$
>
> 对 $p_k$ 求导令其为零，可得：
>
> $$
> p_1 = p_2 = \cdots = p_K = \frac{1}{K}
> $$
>
> 因此**等可能的情况下熵最大**（均匀分布）。

### 从最大熵推导指数族

在机器学习中，$N$ 个样本就是既定事实。我们要在这个既定事实下求最合理的概率分布。

**构造约束**：数据集 $\mathcal{D}$ 可以转换为经验分布 $\hat{p}(x=X)=\dfrac{\text{Count}(X)}{N}$（类似古典概率，样本中 $X$ 的频次除以样本总数）。

有了经验分布，所有的数字特征（期望、方差等）都可以求出。对任意函数 $f(x)$，其均值是可知的：$\mathbb{E}_{\hat{p}}[f(x)]=\Delta$。

于是最大熵模型为：

$$
\begin{aligned}
&\max\{H(p)\} = \min\left\{\sum_{k=1}^N p_k\log p_k\right\} \\[6pt]
&\text{s.t.} \quad \sum_{k=1}^N p_k = 1, \quad \mathbb{E}_p[f(x)] = \Delta
\end{aligned}
$$

构造 Lagrange 函数：

$$
\mathcal{L}(p,\lambda_0,\lambda) = \sum_{k=1}^N p_k\log p_k + \lambda_0\left(1-\sum_{k=1}^N p_k\right) + \lambda^T\left(\Delta - \mathbb{E}_p[f(x)]\right)
$$

对 $p(x)$ 求导（其中 $\mathbb{E}_p[f(x)]=\sum_{k=1}^N p(x_k)f(x_k)$）：

$$
\frac{\partial \mathcal{L}}{\partial p(x)} = \log p(x) + 1 - \lambda_0 - \lambda^T f(x) = 0
$$

> [!note] 向量求导说明
> 因为 $p(x)$ 是向量，求导后结果也是向量，向量的每个元素都为 0。由于数据集是任意的，求和中的每一项都为零。

解得：

$$
\boxed{p(x) = \exp\left(\lambda^T f(x) + \lambda_0 - 1\right)}
$$

> [!success] 结论
> 这就是指数族分布的形式！其中：
> - $\lambda$ 对应自然参数 $\eta$
> - $f(x)$ 对应充分统计量 $\phi(x)$
> - $\lambda_0 - 1$ 与对数配分函数 $A(\eta)$ 相关
>
> 这证明了：**在给定矩约束下，最大熵分布一定是指数族分布**。
