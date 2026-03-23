---
title: 线性回归
tags:
  - 机器学习
  - 回归
  - 线性模型
aliases:
  - Linear Regression
---

# 线性回归

假设数据集为：

$$
\mathcal{D}=\{(x_1, y_1),(x_2, y_2),\cdots,(x_N, y_N)\}
$$

记：

$$
X=(x_1,x_2,\cdots,x_N)^T, \quad Y=(y_1,y_2,\cdots,y_N)^T
$$

其中 $X$ 为 $N \times P$ 矩阵（$N$ 个样本，$P$ 维特征），$Y$ 为 $N \times 1$ 列向量。线性回归的假设函数为：

$$
f(w)=w^Tx
$$

## 最小二乘法

采用二范数定义的平方误差作为损失函数：

$$
L(w)=\sum_{i=1}^N \|w^Tx_i-y_i\|^2_2
$$

展开得：

$$
L(w)=w^TX^TXw-2w^TX^TY+Y^TY
$$

> [!note]- 展开过程
> ![[3.1.png]]

最小化损失函数，令导数为零求解 $\hat{w}$：

$$
\hat{w}=\mathop{argmin}\limits_w L(w) \longrightarrow \frac{\partial}{\partial w}L(w)=0
$$

$$
2X^TX\hat{w}-2X^TY=0 \longrightarrow \hat{w}=(X^TX)^{-1}X^TY=X^+Y
$$

其中 $(X^TX)^{-1}X^T$ 称为**伪逆**。对于满秩的 $X$ 可以直接求解；对于非满秩的情况，需要使用**奇异值分解（SVD）**：

$$
X = U\Sigma V^T
$$

于是：

$$
\hat{w} = V\Sigma^{-1}U^TY
$$

### 几何解释

最小二乘法在几何上的含义：假设样本张成一个 $p$ 维列空间（满秩情况下）：

$$
\text{Col}(X) = \text{span}(x_1, x_2, \cdots, x_p)
$$

模型预测值 $\hat{Y} = Xw$ 是 $X$ 列向量的线性组合。最小二乘法要求 $Y$ 与 $\hat{Y}$ 的距离最小，因此残差 $Y - \hat{Y}$ 应与列空间正交：

$$
X^T(Y - X\hat{w}) = 0 \longrightarrow X^TX\hat{w} = X^TY \longrightarrow \hat{w} = (X^TX)^{-1}X^TY
$$

> [!tip] 几何直觉
> 最小二乘解本质上是 $Y$ 在列空间 $\text{Col}(X)$ 上的==正交投影==，与代数推导完全一致。

## 噪声为高斯分布的 MLE

对于一维情况，记 $y=w^Tx+\epsilon$，其中 $\epsilon\sim\mathcal{N}(0,\sigma^2)$，则 $y\sim\mathcal{N}(w^Tx,\sigma^2)$。

将 $y$ 视为以 $w$ 为参数的概率模型，代入极大似然估计：

$$
L(w)=\log p(Y|X,w)=\log\prod_{i=1}^N p(y_i|x_i,w)
$$

$$
=\sum_{i=1}^N \log\left(\frac{1}{\sqrt{2\pi\sigma}} e^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}}\right) = \sum_{i=1}^N \left(\log\frac{1}{\sqrt{2\pi\sigma}}-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right)
$$

$$
\mathop{argmax}\limits_w L(w)=\mathop{argmin}\limits_w \sum_{i=1}^{N}(y_i-w^Tx_i)^2
$$

> [!important] MLE 与 LSE 的等价性
> 当噪声服从高斯分布时，**MLE 等价于最小二乘估计（LSE）**。

## 正则化

> [!warning] 存在问题
> 当 $X^TX$ 不可逆时（样本不足或特征维度过高导致不满秩），直接求解会失败，且容易造成**过拟合**。

**解决方案：**

1. **加数据** — 增加训练样本量
2. **特征选择** — 降低特征维度（如 PCA）
3. **正则化** — 对 $w$ 施加约束

正则化在损失函数上加入正则项（惩罚模型复杂度），两种常见框架：

$$
\text{L1:} \quad \mathop{argmin}\limits_w L(w)+\lambda\|w\|_1, \quad \lambda>0
$$

$$
\text{L2:} \quad \mathop{argmin}\limits_w L(w)+\lambda\|w\|^2_2, \quad \lambda>0
$$

### L1 — Lasso

L1 正则化可以产生**稀疏解**（部分权重恰好为零）。

- **导数角度**：L1 项在 $0$ 附近的左右导数都不为零，因此更容易取到零解
- **约束角度**：等价于约束优化问题：

$$
\mathop{argmin}\limits_w L(w) \quad \text{s.t.} \quad \|w\|_1 \leq C
$$

平方误差的等值线是椭球，与 $\|w\|_1 = C$（菱形）的切点更容易落在坐标轴上，从而产生稀疏解。

### L2 — Ridge

$$
\hat{w} = \mathop{argmin}\limits_w L(w) + \lambda w^Tw
$$

令导数为零：

$$
\frac{\partial}{\partial w}\left(w^TX^TXw - 2w^TX^TY + Y^TY + \lambda w^Tw\right) = 0
$$

$$
\frac{\partial}{\partial w}\left(w^T(X^TX+\lambda I)w - 2w^TX^TY + Y^TY\right) = 0
$$

$$
2(X^TX+\lambda I)\hat{w} - 2X^TY = 0 \longrightarrow \hat{w} = (X^TX + \lambda I)^{-1}X^TY
$$

> [!tip] L2 正则化的双重作用
> 加入 $\lambda I$ 后：
> 1. 使 $w$ 的各分量趋于较小值（防止过拟合）
> 2. 保证 $X^TX + \lambda I$ 正定可逆（解决不可逆问题）

### 贝叶斯角度的正则化（MAP）

从概率角度，$y\sim\mathcal{N}(w^Tx,\sigma^2)$ 是 $y$ 在 $w$ 已知下的似然。

#### 高斯先验 → L2 正则化

取先验 $w\sim\mathcal{N}(0,\sigma_0^2)$，由贝叶斯定理：

$$
p(w|y)=\frac{p(y|w)p(w)}{p(y)}
$$

$$
\hat{w}=\mathop{argmax}\limits_w p(w|Y) = \mathop{argmax}\limits_w p(Y|w)p(w) = \mathop{argmax}\limits_w \log p(Y|w)p(w)
$$

$$
=\mathop{argmax}\limits_w \left(\log p(Y|w)+\log p(w)\right) = \mathop{argmin}\limits_w \left[\sum_{i=1}^N(y_i-w^Tx_i)^2+\frac{\sigma^2}{\sigma_0^2}w^Tw\right]
$$

> [!note]- 推导说明
> ![[3.2.png]]
>
> 此处省略了 $X$；$p(Y)$ 与 $w$ 无关，利用了高斯 MLE 的结果。

这与 L2 Ridge 正则化形式完全一致，其中 $\lambda = \dfrac{\sigma^2}{\sigma_0^2}$。

#### Laplace 先验 → L1 正则化

取 **Laplace 先验** $p(w) \propto \exp\left(-\frac{\|w\|_1}{b}\right)$，MAP 估计得到：

$$
\hat{w}=\mathop{argmin}\limits_w\left[\sum_{i=1}^N(y_i-w^Tx_i)^2+\lambda\|w\|_1\right]
$$

这与 L1 Lasso 正则化一致。

> [!abstract] 概率视角总结
> | 方法 | 先验 | 等价形式 |
> |:---:|:---:|:---:|
> | **MLE**（无先验） | — | 最小二乘估计 |
> | **MAP** + 高斯先验 | $w \sim \mathcal{N}(0, \sigma_0^2)$ | L2 Ridge 正则化 |
> | **MAP** + Laplace 先验 | $w \sim \text{Laplace}(0, b)$ | L1 Lasso 正则化 |
>
> 以上 MAP 均为**点估计**方法。若需推断参数的完整后验分布，则需使用**贝叶斯线性回归**。

## 小结

线性回归虽是最简单的模型，但"麻雀虽小，五脏俱全"：

- ==噪声为高斯分布==时，MLE 等价于最小二乘误差
- 最小二乘 + ==L2 正则项== 等价于高斯先验下的 MAP 解
- 最小二乘 + ==L1 正则项== 等价于 Laplace 先验下的 MAP 解

### 从线性回归到其他模型

线性回归有三个基本特点：**1. 线性**、**2. 全局性**、**3. 数据未加工**。当这些特性被修改时，就引出不同的模型：

| 修改方向 | 描述 | 对应模型 |
|:---|:---|:---|
| **特征非线性** | 加入高次项，如 $wx^2$ 等多项式拟合 | 多项式回归 |
| **激活函数非线性** | 线性方程后加非线性激活函数 | 线性分类（感知机） |
| **系数非线性** | 同一特征多次变换，每次系数不同 | 多层感知机（深度前馈网络） |
| **局部性** | 不同区域引入不同线性/非线性 | 线性样条回归、决策树 |
| **数据预加工** | 先对高维数据降维处理 | PCA、流形学习 |
