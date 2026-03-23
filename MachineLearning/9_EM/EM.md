---
title: EM算法
tags:
  - 机器学习
  - 概率模型
  - 参数估计
  - EM
aliases:
  - Expectation-Maximization
  - 期望最大化算法
created: 2026-03-15
---

# EM算法

## 概述

> [!abstract] 核心思想
> EM（Expectation-Maximization）算法是一种**迭代优化算法**，用于求解含有**隐变量**的概率模型的**极大似然估计（MLE）**。每次迭代分为两步：**E步**（求期望）和 **M步**（求最大化）。
>
> **注意：EM 是一种算法（类似于梯度下降），而不是一个模型。**

## 动机

对于一般的概率模型 $p(x|\theta)$，MLE 对参数的估计为：

$$
\theta_{MLE}=\mathop{argmax}\limits_\theta\log p(x|\theta)
$$

但是对于包含**隐变量** $z$ 的模型，对数似然变为：

$$
\log p(x|\theta)=\log\int_z p(x,z|\theta)dz
$$

由于对数内部存在积分（或求和），直接求解析解是十分困难的。EM 算法正是为了解决这一问题而提出的。

> [!tip] 直觉理解
> 想象你在一个有雾的山上找最高点（MLE），但你看不清地形（隐变量）。EM 的策略是：
> 1. **E步**：根据当前位置，猜测地形长什么样（估计隐变量的分布）
> 2. **M步**：在猜测的地形上，走到最高点（优化参数）
> 3. 重复以上步骤，直到收敛

## 算法步骤

EM 算法采用迭代方法，核心公式为：

$$
\theta^{t+1}=\mathop{argmax}\limits_{\theta}\int_z\log [p(x,z|\theta)]\cdot p(z|x,\theta^t)dz
$$

其中：
- $p(z|x,\theta^t)$：给定 $x$ 和上一时刻参数 $\theta^t$ 的**后验分布**
- $p(x,z|\theta)$：**完整数据**的联合概率分布

上式可以看成一个期望：

$$
\theta^{t+1}=\mathop{argmax}\limits_{\theta}\mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]
$$

这个公式包含了迭代的两步：

> [!info] E步 & M步
> 1. **E step（期望步）**：计算 $\log p(x,z|\theta)$ 在概率分布 $p(z|x,\theta^t)$ 下的期望，构造 $Q$ 函数
>
> $$Q(\theta,\theta^t)=\mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]=\int_z p(z|x,\theta^t)\log p(x,z|\theta)dz$$
>
> 2. **M step（最大化步）**：最大化 $Q$ 函数，求得下一步的参数
>
> $$\theta^{t+1}=\mathop{argmax}\limits_{\theta}Q(\theta,\theta^t)$$

### 算法伪代码

> [!example] EM 算法流程
> 1. 初始化参数 $\theta^0$
> 2. **重复**直至收敛：
>    - **E步**：计算 $Q(\theta, \theta^t) = \mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]$
>    - **M步**：$\theta^{t+1} = \mathop{argmax}\limits_{\theta} Q(\theta, \theta^t)$
>    - 检查收敛条件：$|\theta^{t+1} - \theta^t| < \epsilon$ 或 $|\log p(x|\theta^{t+1}) - \log p(x|\theta^t)| < \epsilon$
> 3. 返回 $\theta^*$

## 收敛性证明

> [!note] 定理
> EM 算法的每一步迭代都不会降低对数似然：$\log p(x|\theta^t)\le\log p(x|\theta^{t+1})$

> [!quote]- 证明（点击展开）
> **证明：**
>
> 由贝叶斯公式：$\log p(x|\theta)=\log p(z,x|\theta)-\log p(z|x,\theta)$
>
> 左右两边对 $p(z|x,\theta^t)$ 求期望：
>
> $$
> \text{Left}:\int_zp(z|x,\theta^t)\log p(x|\theta)dz=\log p(x|\theta) \int_zp(z|x,\theta^t)dz=\log p(x|\theta)
> $$
>
> $$
> \text{Right}:\underbrace{\int_zp(z|x,\theta^t)\log p(x,z|\theta)dz}_{Q(\theta,\theta^t)}-\underbrace{\int_zp(z|x,\theta^t)\log p(z|x,\theta)dz}_{H(\theta,\theta^t)}
> $$
>
> 所以：
>
> $$
> \log p(x|\theta)=Q(\theta,\theta^t)-H(\theta,\theta^t)
> $$
>
> **Part 1**：由于 $\theta^{t+1}=\mathop{argmax}\limits_{\theta}Q(\theta,\theta^t)$，所以 $Q(\theta^{t+1},\theta^t)\ge Q(\theta^t,\theta^t)$。
>
> **Part 2**：需证 $H(\theta^t,\theta^t)\ge H(\theta^{t+1},\theta^t)$：
>
> $$
> H(\theta^{t+1},\theta^t)-H(\theta^{t},\theta^t)=\int_zp(z|x,\theta^{t})\log\frac{p(z|x,\theta^{t+1})}{p(z|x,\theta^t)}=-\mathrm{KL}\big(p(z|x,\theta^t)\|p(z|x,\theta^{t+1})\big)\le0
> $$
>
> **综合**：$Q$ 增大，$H$ 减小，因此：
>
> $$
> \log p(x|\theta^{t+1})=Q(\theta^{t+1},\theta^t)-H(\theta^{t+1},\theta^t)\ge Q(\theta^t,\theta^t)-H(\theta^t,\theta^t)=\log p(x|\theta^t) \quad\blacksquare
> $$

> [!warning] 收敛性的局限
> EM 算法保证似然**单调不降**，但只能保证收敛到**局部最优**或鞍点，不保证全局最优。实际中常用**多次随机初始化**来缓解这个问题。

## 推导

### 推导一：ELBO 视角

对于 $\log p(x|\theta)$，引入任意分布 $q(z)$：

$$
\log p(x|\theta)=\log p(z,x|\theta)-\log p(z|x,\theta)=\log \frac{p(z,x|\theta)}{q(z)}-\log \frac{p(z|x,\theta)}{q(z)}
$$

分别对两边求期望 $\mathbb{E}_{q(z)}$：

$$
\text{Left}:\int_zq(z)\log p(x|\theta)dz=\log p(x|\theta)
$$

$$
\text{Right}:\underbrace{\int_zq(z)\log \frac{p(z,x|\theta)}{q(z)}dz}_{\text{ELBO}}+\underbrace{\mathrm{KL}(q(z)\|p(z|x,\theta))}_{\ge 0}
$$

因此：

$$
\log p(x|\theta) = \mathrm{ELBO} + \mathrm{KL}(q(z)\|p(z|x,\theta)) \ge \mathrm{ELBO}
$$

> [!important] 核心关系
> 对数似然 = ELBO + KL散度
>
> - ELBO（Evidence Lower Bound）是对数似然的**下界**
> - 等号成立条件：$q(z)=p(z|x,\theta)$，即 KL 散度为 0

EM 算法的目的是将 ELBO 最大化。在每一步 EM 中，令 $q(z) = p(z|x,\theta^t)$ 使得 ELBO 取到最紧的下界，然后优化 $\theta$：

$$
\hat{\theta}=\mathop{argmax}_{\theta}\mathrm{ELBO}=\mathop{argmax}_\theta\int_zq(z)\log\frac{p(x,z|\theta)}{q(z)}dz
$$

代入 $q(z)=p(z|x,\theta^t)$：

$$
\hat{\theta}=\mathop{argmax}_\theta\int_zp(z|x,\theta^t)\log\frac{p(x,z|\theta)}{p(z|x,\theta^t)}dz=\mathop{argmax}_\theta\int_z p(z|x,\theta^t)\log p(x,z|\theta)dz
$$

最后一步成立是因为 $p(z|x,\theta^t)$ 关于 $\theta$ 是常数，可以从 $\mathop{argmax}$ 中移除。这正是 EM 迭代的核心公式。

### 推导二：Jensen 不等式视角

> [!note] Jensen 不等式
> 对于凹函数 $f$（如 $\log$），有：$f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]$
>
> 即**期望的函数值 ≥ 函数值的期望**

$$
\log p(x|\theta)=\log\int_zp(x,z|\theta)dz=\log\int_z\frac{p(x,z|\theta)\cdot q(z)}{q(z)}dz=\log \mathbb{E}_{q(z)}\left[\frac{p(x,z|\theta)}{q(z)}\right]
$$

由 Jensen 不等式（$\log$ 是凹函数）：

$$
\log \mathbb{E}_{q(z)}\left[\frac{p(x,z|\theta)}{q(z)}\right]\ge \mathbb{E}_{q(z)}\left[\log\frac{p(x,z|\theta)}{q(z)}\right] = \mathrm{ELBO}
$$

等号成立条件：$\frac{p(x,z|\theta)}{q(z)}=C$（常数），即：

$$
q(z)=\frac{p(x,z|\theta)}{C}=\frac{p(x,z|\theta)}{p(x|\theta)}=p(z|x,\theta)
$$

与 ELBO 推导得出的结论完全一致。

## 广义 EM

### 隐变量生成模型

对于一组可观测样本 $X=\{x_i\}_{i=1}^N$，直接求解 $p(x|\theta)$ 可能非常困难。于是我们人为引入一组隐变量 $Z=\{z_i\}_{i=1}^N$，$Z$ 用于支撑生成 $X$。

> [!tip] 隐变量的本质
> 隐变量取决于观测数据——观测数据不同，隐变量也会不同。从这个角度看，隐变量可以看作一种"参数"。

### 广义 EM 的动机

在标准 EM 的 E步中，我们假定 $q(z)=p(z|x,\theta)$（令 KL=0），但这是有前提的——$p(z|x,\theta)$ 必须可以精确求解。

当**后验 $p(z|x,\theta)$ 无法精确求解**时，必须使用近似方法：
- **变分推断**（Variational Inference）→ [[MachineLearning/11_inference/VI|VBEM]]
- **MCMC 采样** → [[MachineLearning/11_inference/MCMC|MCEM]]

### 广义 EM 算法

广义 EM 的基本思路：**交替优化** $q$ 和 $\theta$。

> [!info] 广义 EM 步骤
> 1. **E step**（固定 $\theta$，优化 $q$）：
>
> $$\hat{q}^{t+1}(z)=\mathop{argmax}_q\int_zq(z)\log\frac{p(x,z|\theta^t)}{q(z)}dz$$
>
> 2. **M step**（固定 $q$，优化 $\theta$）：
>
> $$\hat{\theta}^{t+1}=\mathop{argmax}_\theta \int_zq^{t+1}(z)\log\frac{p(x,z|\theta)}{q^{t+1}(z)}dz$$

其中 ELBO 可以分解为：

$$
\mathrm{ELBO}=\int_zq(z)\log\frac{p(x,z|\theta)}{q(z)}dz=\mathbb{E}_{q(z)}[\log p(x,z|\theta)]+H(q)
$$

其中 $H(q)$ 是 $q(z)$ 的熵。

> [!note] 坐标上升法的视角
> EM 算法类似于**坐标上升法**：固定部分参数，优化其他参数，再交替迭代。从这个角度看，先做 E步还是先做 M步其实是无所谓的。

## 典型应用

> [!example] EM 算法的经典应用
> - **[[MachineLearning/10_GMM/GMM|高斯混合模型（GMM）]]**：隐变量为样本所属的高斯分量
> - 隐马尔可夫模型（**[[隐马尔可夫模型 HMM|HMM]]）**：隐变量为隐状态序列，E步对应 Baum-Welch 算法
> - **概率主成分分析（PPCA[[DimensionlityReduction]]）**：隐变量为低维潜在表示
> - **缺失数据处理**：将缺失值视为隐变量，用 EM 迭代补全

## 小结

| 特性 | 说明 |
|------|------|
| **类型** | 迭代优化算法 |
| **目标** | 含隐变量模型的极大似然估计 |
| **核心** | E步求期望，M步最大化 |
| **收敛** | 单调不降，但可能陷入局部最优 |
| **推广** | 广义 EM（VBEM / MCEM） |
| **关联** | 坐标上升法、变分推断、MCMC |

