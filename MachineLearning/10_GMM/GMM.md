---
title: 高斯混合模型
tags:
  - 机器学习
  - 概率模型
  - 聚类
  - GMM
  - EM
aliases:
  - Gaussian Mixture Model
  - GMM
created: 2026-03-15
---

# 高斯混合模型

## 概述

> [!abstract] 核心思想
> 高斯混合模型（Gaussian Mixture Model, GMM）是一种**概率生成模型**，假设数据由 $K$ 个高斯分布**混合**而成。每个样本以一定概率属于某个高斯分量，这一概率由**隐变量** $z$ 控制。
>
> GMM 是 [[MachineLearning/9_EM/EM|EM 算法]]最经典的应用之一，也是**软聚类**的代表方法。

## 基本思想

### 1. 几何角度：多个高斯分布的加权平均
![[MachineLearning/10_GMM/assets/1.png]]
GMM 的概率密度函数为多个高斯分布的**线性组合**：

$$
p(x)=\sum\limits_{k=1}^K\alpha_k\mathcal{N}(\mu_k,\Sigma_k)
$$

其中 $\alpha_k$ 为第 $k$ 个高斯分量的**混合系数（权重）**，满足 $\alpha_k \ge 0,\ \sum_{k=1}^K \alpha_k = 1$。

### 2. 混合模型角度：引入隐变量


![[MachineLearning/10_GMM/assets/2.png]]
对于一组可观测的样本 $X=\{x_i\}_{i=1}^N$，为了表示每个样本 $x$ 属于哪一个高斯分布，我们引入一组**隐变量** $Z=\{z_i\}_{i=1}^N$。

$z$ 是一个离散随机变量，对于每个 $x_i$，其对应的 $z_i$ 服从：

| $z$ | 1 | 2 | ... | $K$ |
| --- | --- | --- | --- | --- |
| $p(z)$ | $p_1$ | $p_2$ | ... | $p_K$ |

$$
p(z=k)=p_k,\quad \sum\limits_{k=1}^Kp(z=k)=1
$$

> [!tip] 隐变量的直觉
> 隐变量 $z_i$ 可以理解为样本 $x_i$ 的**"身份标签"**——它告诉我们 $x_i$ 是从第几个高斯分量中生成的。但在实际中，$z_i$ 是不可观测的，需要通过 EM 算法来推断。

### 3. 生成模型角度：概率图表示

作为一个生成式模型，GMM 通过隐变量 $z$ 的分布来生成样本。用**概率图**表示为：
![[MachineLearning/10_GMM/assets/3.png]]

其中，$\pi$ 控制 $z$ 的分布（混合系数），$\mu, \Sigma$ 控制每个高斯分量的形状。生成过程为：

> [!info] GMM 数据生成过程
> 对每个样本 $x_i$（$i = 1, \ldots, N$）：
> 1. **采样隐变量**：$z_i \sim \mathrm{Categorical}(p_1, p_2, \ldots, p_K)$（决定属于哪个高斯分量）
> 2. **采样观测数据**：$x_i \sim \mathcal{N}(\mu_{z_i}, \Sigma_{z_i})$（从对应的高斯分布中采样）

因此，$x$ 的边际分布为：

$$
p(x)=\sum\limits_zp(x,z)=\sum\limits_{k=1}^Kp(x,z=k)=\sum\limits_{k=1}^Kp(z=k)p(x|z=k)
$$

即：

$$
p(x)=\sum\limits_{k=1}^Kp_k\mathcal{N}(x|\mu_k,\Sigma_k)
$$

## 极大似然估计

对于 GMM 的概率密度函数：

$$
p(x)=\sum\limits_{k=1}^Kp_k\mathcal{N}(x|\mu_k,\Sigma_k)
$$

参数为 $\theta=\{p_1,\cdots,p_K,\ \mu_1,\cdots,\mu_K,\ \Sigma_1,\cdots,\Sigma_K\}$。通过极大似然估计求解 $\theta$：

$$
\theta_{MLE}=\mathop{argmax}\limits_{\theta}\log p(X)=\mathop{argmax}_{\theta}\sum\limits_{i=1}^N\log p(x_i)=\mathop{argmax}_\theta\sum\limits_{i=1}^N\log \sum\limits_{k=1}^Kp_k\mathcal{N}(x_i|\mu_k,\Sigma_k)
$$

> [!warning] 直接求解的困难
> 由于 $\log$ 内部存在 $\sum$（log-sum 结构），无法交换求和与对数，直接求导**无法得到解析解**。因此需要使用 [[MachineLearning/9_EM/EM|EM 算法]]进行迭代求解。

## EM 求解 GMM

[[MachineLearning/9_EM/EM|EM 算法]]的核心迭代公式为：

$$
\theta^{t+1}=\mathop{argmax}\limits_{\theta}\mathbb{E}_{z|x,\theta^t}[\log p(x,z|\theta)]
$$

其中期望部分也记为 $Q(\theta,\theta^t)$。

### E步：求期望

套用 GMM 的表达式，对数据集来说：

$$
Q(\theta,\theta^t)=\sum\limits_z\left[\log\prod\limits_{i=1}^Np(x_i,z_i|\theta)\right]\prod \limits_{i=1}^Np(z_i|x_i,\theta^t)
$$

$$
=\sum\limits_z\left[\sum\limits_{i=1}^N\log p(x_i,z_i|\theta)\right]\prod \limits_{i=1}^Np(z_i|x_i,\theta^t)
$$

> [!quote]- 化简过程（点击展开）
> 展开求和号，观察第 1 项：
>
> $$
> \sum\limits_z\log p(x_1,z_1|\theta)\prod\limits_{i=1}^Np(z_i|x_i,\theta^t)
> $$
>
> $$
> =\sum\limits_z\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^t)\prod\limits_{i=2}^Np(z_i|x_i,\theta^t)
> $$
>
> $$
> =\sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^t)\underbrace{\sum\limits_{z_2,\cdots,z_K}\prod\limits_{i=2}^Np(z_i|x_i,\theta^t)}_{=1}
> $$
>
> $$
> =\sum\limits_{z_1}\log p(x_1,z_1|\theta)\cdot p(z_1|x_1,\theta^t)
> $$
>
> **要点解释：**
> - **对 $z$ 求和的含义**：$z$ 不是一个数，而是 $N$ 个离散随机变量的集合，因此 $\sum_z$ 是对 $N$ 个离散分布的所有组合求和
> - **后半部分为何消失**：展开后是多个概率分布的边缘化求和，每个 $\sum_{z_i} p(z_i|x_i,\theta^t) = 1$，因此乘积也为 1

类似地，$Q$ 可以化简为：

$$
Q(\theta,\theta^t)=\sum\limits_{i=1}^N\sum\limits_{z_i}\log p(x_i,z_i|\theta)\cdot p(z_i|x_i,\theta^t)
$$

**代入具体分布。** 对于联合分布 $p(x,z|\theta)$：

$$
p(x,z|\theta)=p(z|\theta)p(x|z,\theta)=p_z\mathcal{N}(x|\mu_z,\Sigma_z)
$$

对于后验分布 $p(z|x,\theta^t)$（也称**响应度/责任度**）：

$$
p(z=k|x,\theta^t)=\frac{p(x,z=k|\theta^t)}{p(x|\theta^t)}=\frac{p_k^t\mathcal{N}(x|\mu_k^t,\Sigma_k^t)}{\sum\limits_{j=1}^Kp_j^t\mathcal{N}(x|\mu_j^t,\Sigma_j^t)}
$$

> [!note] 记号约定
> 为简洁，记响应度为：
>
> $$\gamma_{ik} \triangleq p(z_i=k|x_i,\theta^t)$$
>
> 它表示在当前参数 $\theta^t$ 下，第 $i$ 个样本属于第 $k$ 个高斯分量的**后验概率**。

代入 $Q$ 函数：

$$
Q=\sum\limits_{i=1}^N\sum\limits_{k=1}^K\gamma_{ik}\log\left[p_k\mathcal{N}(x_i|\mu_k,\Sigma_k)\right]
$$

$$
=\sum\limits_{k=1}^K\sum\limits_{i=1}^N\gamma_{ik}\left[\log p_k+\log \mathcal{N}(x_i|\mu_k,\Sigma_k)\right]
$$

### M步：求最大化

对 $Q$ 函数中的各参数分别求最优值。

#### 估计混合系数 $p_k$

$$
p_k^{t+1}=\mathop{argmax}\limits_{p_k}\sum\limits_{k=1}^K\sum\limits_{i=1}^N\gamma_{ik}\log p_k \quad s.t.\ \sum\limits_{k=1}^Kp_k=1
$$

引入 Lagrange 乘子：

$$
L(p_k,\lambda)=\sum\limits_{k=1}^K\sum\limits_{i=1}^N\gamma_{ik}\log p_k-\lambda\left(1-\sum\limits_{k=1}^Kp_k\right)
$$

> [!quote]- 求解过程（点击展开）
> 对 $p_k$ 求偏导：
>
> $$
> \frac{\partial L}{\partial p_k}=\sum\limits_{i=1}^N\frac{\gamma_{ik}}{p_k}+\lambda=0 \quad \Rightarrow \quad p_k = -\frac{1}{\lambda}\sum\limits_{i=1}^N\gamma_{ik}
> $$
>
> 对所有 $k$ 求和，利用约束 $\sum_k p_k = 1$：
>
> $$
> \sum\limits_k p_k = -\frac{1}{\lambda}\sum\limits_k\sum\limits_{i=1}^N\gamma_{ik}=1
> $$
>
> 由于 $\sum_k \gamma_{ik} = \sum_k p(z_i=k|x_i,\theta^t) = 1$，所以：
>
> $$
> \sum\limits_k\sum\limits_{i=1}^N\gamma_{ik} = N \quad \Rightarrow \quad \lambda = -N
> $$

$$
\boxed{p_k^{t+1}=\frac{1}{N}\sum\limits_{i=1}^N\gamma_{ik} = \frac{N_k}{N}}
$$

其中 $N_k = \sum_{i=1}^N \gamma_{ik}$，表示**第 $k$ 个分量的有效样本数**。

#### 估计均值 $\mu_k$

$\mu_k$ 无约束，对 $Q$ 中与 $\mu_k$ 相关的部分直接求导。展开高斯分布的对数：

$$
\log \mathcal{N}(x_i|\mu_k,\Sigma_k) = -\frac{1}{2}(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k) + \text{const}
$$

对 $\mu_k$ 求导并令其为零：

$$
\frac{\partial Q}{\partial \mu_k}=\sum\limits_{i=1}^N\gamma_{ik}\Sigma_k^{-1}(x_i-\mu_k)=0
$$

$$
\boxed{\mu_k^{t+1}=\frac{\sum\limits_{i=1}^N\gamma_{ik}\cdot x_i}{\sum\limits_{i=1}^N\gamma_{ik}}=\frac{1}{N_k}\sum\limits_{i=1}^N\gamma_{ik}\cdot x_i}
$$

> [!tip] 直觉理解
> $\mu_k$ 的更新公式是所有样本的**加权平均**，权重就是每个样本属于第 $k$ 个分量的后验概率 $\gamma_{ik}$。这与 K-Means 中的"质心更新"非常类似，但 K-Means 使用的是**硬分配**（0 或 1），GMM 使用的是**软分配**（概率值）。

#### 估计协方差矩阵 $\Sigma_k$

类似地，对 $\Sigma_k$ 求导（或等价地对 $\Sigma_k^{-1}$ 求导）并令其为零：

$$
\boxed{\Sigma_k^{t+1}=\frac{1}{N_k}\sum\limits_{i=1}^N\gamma_{ik}(x_i-\mu_k^{t+1})(x_i-\mu_k^{t+1})^T}
$$

这同样是一个**加权协方差**估计。

### 算法总结

> [!example] GMM-EM 算法流程
> **输入**：数据集 $X = \{x_i\}_{i=1}^N$，高斯分量数 $K$
>
> 1. **初始化**参数 $\theta^0 = \{p_k^0, \mu_k^0, \Sigma_k^0\}_{k=1}^K$
> 2. **重复**直至收敛：
>    - **E步**：计算响应度
>      $$\gamma_{ik} = \frac{p_k^t\mathcal{N}(x_i|\mu_k^t,\Sigma_k^t)}{\sum_{j=1}^K p_j^t\mathcal{N}(x_i|\mu_j^t,\Sigma_j^t)}$$
>    - **M步**：更新参数
>      $$N_k = \sum_{i=1}^N \gamma_{ik}$$
>      $$p_k^{t+1} = \frac{N_k}{N}, \quad \mu_k^{t+1} = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik} x_i, \quad \Sigma_k^{t+1} = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik}(x_i - \mu_k^{t+1})(x_i - \mu_k^{t+1})^T$$
>    - 检查收敛：$|\log p(X|\theta^{t+1}) - \log p(X|\theta^t)| < \epsilon$
> 3. **返回** $\theta^*$

## GMM 与 K-Means 的对比

| 特性 | **K-Means** | **GMM** |
|------|-------------|---------|
| **分配方式** | 硬分配（每个点归属一个簇） | 软分配（概率归属多个分量） |
| **簇的形状** | 球形（基于欧氏距离） | 椭球形（由协方差矩阵决定） |
| **优化方法** | 坐标下降 | EM 算法 |
| **输出** | 簇标签 | 后验概率分布 |
| **模型假设** | 非概率模型 | 概率生成模型 |
| **关系** | — | K-Means 是 GMM 的**特例**（$\Sigma_k = \sigma^2 I$，$\sigma^2 \to 0$） |

> [!important] K-Means 是 GMM 的特例
> 当所有高斯分量的协方差矩阵取为 $\Sigma_k = \sigma^2 I$，且令 $\sigma^2 \to 0$ 时，GMM 的软分配退化为硬分配，EM 算法退化为 K-Means 算法。

## 小结

| 特性 | 说明 |
|------|------|
| **类型** | 概率生成模型 / 软聚类 |
| **参数** | $\{p_k, \mu_k, \Sigma_k\}_{k=1}^K$ |
| **求解** | EM 算法（E步求响应度，M步更新参数） |
| **收敛** | 对数似然单调不降，可能陷入局部最优 |
| **优势** | 软分配、可建模椭球形簇、概率输出 |
| **局限** | 需指定 $K$、对初始化敏感、可能出现奇异解 |

