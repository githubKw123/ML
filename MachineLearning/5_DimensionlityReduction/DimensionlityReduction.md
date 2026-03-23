---
title: 降维（Dimensionality Reduction）
tags:
  - 机器学习
  - 降维
  - PCA
  - 无监督学习
aliases:
  - Dimensionality Reduction
  - 主成分分析
---

# 降维

## 概述

在高维数据中，特征之间往往存在冗余和相关性。降维（Dimensionality Reduction）的目的是将数据从高维空间映射到低维空间，同时尽可能保留原始数据的关键信息。

> [!tip] 降维的意义
> 1. **缓解维度灾难**：高维空间中数据稀疏，模型需要指数级增长的样本量
> 2. **解决过拟合**：除了正则化和增加数据，降维是缓解过拟合的有效方法
> 3. **去除噪声**：低维投影可以过滤掉噪声维度
> 4. **可视化**：将高维数据降到 2D/3D 以便直观观察
> 5. **降低计算成本**：减少特征数量，加快训练和推理速度

降维的算法分为：

| 类别 | 方法 | 特点 |
|------|------|------|
| 直接降维 | 特征选择（Filter, Wrapper, Embedded） | 从原始特征中选择子集，不改变特征本身 |
| 线性降维 | PCA, MDS, LDA 等 | 通过线性变换将数据投影到低维子空间 |
| 非线性降维 | 流形学习（Isomap, LLE, t-SNE, UMAP 等） | 保留数据的非线性流形结构 |

为了方便，我们首先将均值和协方差矩阵（数据集）写成能用矩阵表示的形式：

$$
\overline{x}=\frac{1}{N}\sum\limits_{i=1}^Nx_i=\frac{1}{N}X^T1_N
$$

$$
S=\frac{1}{N}\sum\limits_{i=1}^N(x_i-\overline{x})(x_i-\overline{x})^T=\frac{1}{N}(x_1-\overline{x},x_2-\overline{x},\cdots,x_N-\overline{x})(x_1-\overline{x},x_2-\overline{x},\cdots,x_N-\overline{x})^T
$$

$$
=\frac{1}{N}(X^T-\frac{1}{N}X^T\mathbb{1}_{N}\mathbb{1}_{N}^T)(X^T-\frac{1}{N}X^T\mathbb{1}_{N}\mathbb{1}_{N}^T)^T=\frac{1}{N}X^T(I_N-\frac{1}{N}\mathbb{1}_{N}\mathbb{1}_{N})(I_N-\frac{1}{N}\mathbb{1}_{N}\mathbb{1}_{N})^TX
$$

$$
=\frac{1}{N}X^TH_NH_N^TX=\frac{1}{N}X^TH_NH_NX=\frac{1}{N}X^THX
$$




上面这个式子利用了中心矩阵 $H = I_N - \frac{1}{N}\mathbb{1}_N\mathbb{1}_N^T$ 的**对称性**（$H^T=H$）和**幂等性**（$H^2=H$），$H$ 也是一个投影矩阵。

## 线性降维-主成分分析 PCA

### 思想

**核心目的：原始特征空间的重构（相关变无关）**
![[5.1.png]]

（如图重构前可能两个特征成一个线性关系，重构后x无论怎么变化,y基本保持在一个范围内，就变成线性无关了，这时候X很能表现样本特点，我们就可以将二维特性降维到X一维上，这个X就是主成分）

**实现核心目的的方法：
1.最大投影方差
2.最小重构距离**

### 损失函数

**最大化投影方差：**
对于数据集来说，我们首先将其中心化$x_i-\overline{x}$，这样好计算方差。那么其在$u_1$方向上的投影就可以表示为$(x_i-\overline{x})^Tu_1$(详见LDA)，因为均值为0，那么投影的方差就可以直接表示为

$$
J=\frac{1}{N}\sum\limits_{i=1}^N((x_i-\overline{x})^Tu_1)^2=u_1^T \sum\limits_{i=1}^N\frac{1}{N}(x_i-\overline{x})^T(x_i-\overline{x})u_1=u_1^TSu_1
$$

其中

$$
s.t.\ u_1^Tu_1=1
$$

这里只是以二维为例，多维是一样的，就有多个重构方向$u$，降维的时候想降到几维就取几个重构方向即可。

**最小重构距离：**
原来的数据很有可能各个维度之间是相关的，于是我们希望找到一组 $p$ 个新的线性无关的单位基 $u_i$，降维就是取其中的 $q$ 个基。于是对于一个样本 $x_i$，经过这个坐标变换后(对应方向上的投影乘以单位基)：

$$
{x_i}=\sum\limits_{i=1}^p(x_i^Tu_i)u_i
$$

之后我们进行降维，取其中的 $q$ 个基，扔掉其他的，得到真正的

$$
\hat{x_i}=\sum\limits_{i=1}^q(x_i^Tu_i)u_i
$$

而最小重构代价，其实就是让扔掉的那一部分最小：

$$
J = \frac{1}{N}\sum\limits_{i=1}^N||x_i-\hat{x_i}||^2
$$

然后再考虑上去中心化：

$$
J=\frac{1}{N}\sum\limits_{i=1}^N\sum\limits_{j=q+1}^p((x_i-\overline{x})^Tu_j)^2=\sum\limits_{j=q+1}^pu_j^TSu_j\ ,\ s.t.\ u_j^Tu_j=1
$$

这跟上面那个理解推导出的是一样的，无非这种是多维，又因为重构后空间正交，所以一维跟多维也没啥大区别。

### 求解

这是一个带约束的优化并问题，直接进行拉格朗日乘子法，由于每个基都是线性无关的，于是每一个 $u_j$ 的求解可以分别进行，使用拉格朗日乘子法：

$$
\mathop{argmax}_{u_j}L(u_j,\lambda)=\mathop{argmax}_{u_j}u_j^TSu_j+\lambda(1-u_j^Tu_j)
$$

求导，导数等于零得：

$$
Su_j=\lambda u_j
$$

可见，我们需要的基就是协方差矩阵的特征向量。损失函数最大取在本特征值前 $q$ 个最大值。这样实际上对向量的协方差矩阵特征值分解得$S=GKG^T$，要降为$q$维,取对角矩阵前$q$个值，以及$G$对应的前$q$个向量就行。

> [!note] PCA 求解步骤总结
> 1. 对数据进行**中心化**：$x_i \leftarrow x_i - \overline{x}$
> 2. 计算**协方差矩阵**：$S = \frac{1}{N}X^THX$
> 3. 对 $S$ 进行**特征值分解**：$S = G\Lambda G^T$
> 4. 取前 $q$ 个最大特征值对应的特征向量组成投影矩阵 $W = (u_1, u_2, \cdots, u_q)$
> 5. **投影**：$Z = XW$，得到降维后的数据

### 主成分数量的选择

选择主成分数量 $q$ 的常用方法是基于**累计方差贡献率**（Cumulative Explained Variance Ratio）：

$$
\eta(q) = \frac{\sum_{j=1}^q \lambda_j}{\sum_{j=1}^p \lambda_j} \geq \text{threshold}
$$

通常选取阈值为 **95%** 或 **99%**，即保留能解释 95%/99% 方差的前 $q$ 个主成分。也可以通过**碎石图**（Scree Plot）观察特征值的"拐点"来决定。

### 数据的SVD

一般来说任何矩阵都可以进行奇异值分解，下面使用实际训练时常常使用的 SVD 直接求得这个$q$个特征向量。

对中心化后的数据集进行奇异值分解：

$$
HX=U\Sigma V^T,U^TU=E_N,V^TV=E_p,\Sigma:N\times p
$$

于是：

$$
S=\frac{1}{N}X^THX=\frac{1}{N}X^TH^THX=\frac{1}{N}V\Sigma^T\Sigma V^T=\frac{1}{N}V\Sigma^2 V^T
$$

这里我们发现自然而然的$S$完成了特征值分解，因此我们直接对中心化后的数据集进行 SVD，就可以得到特征值和特征向量 $V$

### PCOA

由上面的推导，我们也可以得到另一种方法 PCoA 主坐标分析，定义并进行特征值分解：

$$
T=HXX^TH=U\Sigma^2U^T
$$

这时我们发现$S$和$T$有相同的特征值,用S特征值分解完得到的是主成分,也就是投影方向，而T得到的就直接是主坐标，这是因为:

从S来看，主坐标是$HXV=U\Sigma V^TV=U\Sigma$

从T来看，$TU\Sigma=U\Sigma(\Sigma^T\Sigma)$，$U\Sigma$就是特征向量

于是可以直接得到坐标。这两种方法都可以得到主成分，但是由于方差矩阵是 $p\times p$ 的，而 $T$ 是 $N\times N$ 的，所以对样本量较少的时候可以采用 PCoA的方法。

### p-PCA

**模型**

对原数据 $x\in\mathbb{R}^p$（观测数据） ，降维后的数据为 $z\in\mathbb{R}^q,q<p$（隐变量）。降维通过一个矩阵变换（投影）进行，而这里其实表示了根据隐变量生成数据的过程：

$$
z\sim\mathcal{N}(\mathbb{O}_{q1},\mathbb{I}_{qq})\\x=Wz+\mu+\varepsilon\\\varepsilon\sim\mathcal{N}(0,\sigma^2\mathbb{I}_{pp})
$$

我们的目的就是，在已知隐变量分布$p(z)$，已知线性变化过程$p(x|z)$,求数据$p(x)$的分布，求后验$p(z|x)$,这几者关系如下![[5.2.png]]


**学习策略**

对于这个模型，我们可以使用期望-最大（EM）的算法进行学习，在进行推断的时候需要求得 $p(z|x)$，推断的求解过程和线性高斯模型类似，详见第二章最后。

首先求边缘分布和后验分布：

$$
\mathbb{E}[x]=\mathbb{E}[Wz+\mu+\varepsilon]=\mu
$$

$$
\text{Var}[x]=WW^T+\sigma^2\mathbb{I}_{pp}
$$

$$
\Longrightarrow p(x)=\mathcal{N}(\mu,\ WW^T+\sigma^2\mathbb{I}_{pp})
$$

由贝叶斯公式 $p(z|x)=\frac{p(x|z)p(z)}{p(x)}$，可得后验分布：

$$
p(z|x)=\mathcal{N}\Big(W^T(WW^T+\sigma^2\mathbb{I})^{-1}(x-\mu),\ \mathbb{I}-W^T(WW^T+\sigma^2\mathbb{I})^{-1}W\Big)
$$

**EM 算法求解**

记 $M = W^T W + \sigma^2 \mathbb{I}_q$，则后验分布的均值和方差可以简写为：

$$
\mathbb{E}[z|x] = M^{-1}W^T(x-\mu), \quad \text{Var}[z|x] = \sigma^2 M^{-1}
$$

- **E 步**：计算后验期望

$$
\mathbb{E}[z_i|x_i] = M^{-1}W^T(x_i - \mu)
$$

$$
\mathbb{E}[z_i z_i^T|x_i] = \text{Var}[z|x] + \mathbb{E}[z_i|x_i]\mathbb{E}[z_i|x_i]^T = \sigma^2 M^{-1} + \mathbb{E}[z_i|x_i]\mathbb{E}[z_i|x_i]^T
$$

- **M 步**：更新参数

$$
W_{new} = \left(\sum_{i=1}^N (x_i-\mu)\mathbb{E}[z_i|x_i]^T\right)\left(\sum_{i=1}^N \mathbb{E}[z_iz_i^T|x_i]\right)^{-1}
$$

$$
\sigma^2_{new} = \frac{1}{Np}\sum_{i=1}^N \left( \|x_i-\mu\|^2 - 2\mathbb{E}[z_i|x_i]^TW_{new}^T(x_i-\mu) + \text{tr}(\mathbb{E}[z_iz_i^T|x_i]W_{new}^TW_{new}) \right)
$$

> [!info] p-PCA 与 PCA 的关系
> 当 $\sigma^2 \to 0$ 时，p-PCA 的最大似然解退化为传统 PCA 的解。p-PCA 的优势在于：
> - 提供了一个**概率生成模型**，可以处理缺失数据
> - 可以通过 EM 算法高效求解，适合大规模数据
> - 自然地给出了数据的**似然函数**，可用于模型选择

## 非线性降维-核 PCA（Kernel PCA）

传统 PCA 只能发现数据中的**线性结构**。当数据分布在非线性流形上时，线性降维效果有限。核 PCA 利用**核技巧**（Kernel Trick）将数据隐式映射到高维特征空间，再在该空间中做 PCA。

### 基本思想

设非线性映射 $\phi: \mathbb{R}^p \to \mathcal{F}$，将原始数据映射到高维（甚至无穷维）特征空间。在特征空间中做 PCA 就是对协方差矩阵：

$$
C = \frac{1}{N}\sum_{i=1}^N \phi(x_i)\phi(x_i)^T
$$

进行特征值分解。但我们不需要显式计算 $\phi(x)$，只需通过**核函数** $k(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ 即可。

### 求解

定义核矩阵 $K_{ij} = k(x_i, x_j)$，中心化后的核矩阵为：

$$
\tilde{K} = HKH = (I - \frac{1}{N}\mathbb{1}\mathbb{1}^T)K(I - \frac{1}{N}\mathbb{1}\mathbb{1}^T)
$$

对 $\tilde{K}$ 进行特征值分解 $\tilde{K}\alpha = \lambda \alpha$，取前 $q$ 个最大特征值对应的特征向量 $\alpha_1, \cdots, \alpha_q$（归一化使得 $\lambda_k \alpha_k^T \alpha_k = 1$），则新样本 $x$ 在第 $k$ 个主成分上的投影为：

$$
y_k(x) = \sum_{i=1}^N \alpha_{ki} \cdot k(x_i, x)
$$

> [!example] 常用核函数
> - **多项式核**：$k(x,y) = (x^Ty + c)^d$
> - **高斯（RBF）核**：$k(x,y) = \exp(-\frac{\|x-y\|^2}{2\sigma^2})$
> - **Sigmoid 核**：$k(x,y) = \tanh(\alpha x^Ty + c)$

## PCA 算法对比总结

| 方法 | 核心思想 | 矩阵规模 | 适用场景 |
|------|----------|----------|----------|
| PCA (EVD) | 协方差矩阵特征值分解 | $p \times p$ | 特征维度 $p$ 较小 |
| PCA (SVD) | 数据矩阵奇异值分解 | $N \times p$ | 通用，数值更稳定 |
| PCoA | 内积矩阵 $T$ 特征值分解 | $N \times N$ | 样本量 $N \ll p$ |
| p-PCA | 概率生成模型 + EM | 迭代更新 | 缺失数据、模型选择 |
| Kernel PCA | 核技巧 + 特征空间 PCA | $N \times N$ | 非线性结构数据 |

> [!tip] 实践建议
> 1. **数据预处理**：PCA 前务必进行**标准化**（StandardScaler），否则方差大的特征会主导结果
> 2. **选择方法**：一般优先使用 SVD 实现（sklearn 默认），数值稳定且高效
> 3. **主成分数量**：先用碎石图和累计方差贡献率确定，通常保留 95% 以上方差
> 4. **非线性数据**：若线性 PCA 效果不好，考虑 Kernel PCA 或流形学习方法（t-SNE, UMAP）


