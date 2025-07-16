# 前言

在我的理解中，给出一组输入数据，给出或不给出输出数据，如何在给定这些数据的基础上得到一个函数，能够将输入数据拟合成输出数据或者预想的形式，学习到这个函数的过程就可以理解为机器学习

这一部分就是概念最广的机器学习，涉及到最简单的线性回归到复杂玻尔兹曼机，不过多地引入深度学习等具体技术的分支，参考了这个视频[https://www.bilibili.com/video/BV1aE411o7qd/?spm_id_from=333.337.search-card.all.click&vd_source=09c57c1cea91c317c3c9403c4fef990a](https://www.bilibili.com/video/BV1aE411o7qd/?spm_id_from=333.337.search-card.all.click&vd_source=09c57c1cea91c317c3c9403c4fef990a)

这个笔记很多部分也参考了[https://github.com/tsyw/MachineLearningNotes/blob/master/1.Intro_Math.md?plain=1](https://github.com/tsyw/MachineLearningNotes/blob/master/1.Intro_Math.md?plain=1)

# 两个派别

机器学习的目的是学习描述数据的模型，那么很自然的就可以将概率引入到机器学习中，因为可以认为$X$是符合一种概率模型，这个模型可以参数化为$\theta$,对于一组数据$X$:

$$
X_{N\times p}=(x_{1},x_{2},\cdots,x_{N})^{T},x_{i}=(x_{i1},x_{i2},\cdots,x_{ip})
$$

表示这组数据有$N$个样本，每个样本都是$p$维向量。其中每个观测都是由 $p(x|\theta)$ 生成的。那么怎么根据这组样本去求概率模型$p(x|\theta)$，也就是去解$\theta$就是机器学习的主要任务。

而关于怎么求解，可以分为**频率派**和**贝叶斯派**两个主要观点：

## 频率派的观点

$\theta$：未知的常量

$X$：随机变量

有随机变量，估计概率模型的常量，就用概率论里很常用的**最大似然（MLE）**

$$
\theta_{MLE}=\mathop{argmax}\limits _{\theta}\log p(X|\theta)\mathop{=}\limits _{iid}\mathop{argmax}\limits _{\theta}\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)
$$

**本质：是一个优化问题**

## 贝叶斯派的观点

$\theta$：也是随机变量，服从概率分布 $\theta\sim p(\theta)$（先验）

那么先通过贝叶斯定理（后验$\sim$似然$\times$先验）将各概率联系起来：

$$
p(\theta|X)=\frac{p(X|\theta)\cdot p(\theta)}{p(X)}=\frac{p(X|\theta)\cdot p(\theta)}{\int\limits _{\theta}p(X|\theta)\cdot p(\theta)d\theta}
$$

我们要最大化这个参数后验，也就是取最能描述样本特点的$\theta$，使用后验概率最大MAP：

$$
\theta_{MAP}=\mathop{argmax}\limits _{\theta}p(\theta|X)=\mathop{argmax}\limits _{\theta}p(X|\theta)\cdot p(\theta)
$$

求解这个$\theta$，也就得到了参数的后验概率$p(\theta|X)$。

那么解了有什么用呢，可以求新数据属于原数据的概率，也就是做预测贝叶斯预测：

$$
p(x_{new}|X)=\int\limits _{\theta}p(x_{new}|\theta)\cdot p(\theta|X)d\theta
$$

**本质：关键是求后验，本质是求积分，延伸出概率图模型**

# 

# 三范式

## 监督学习

* **定义**：使用带有标签的数据（输入-输出对）训练模型，目标是学习从输入到输出的映射。
* **特点**：
  
  * 数据有明确的输入和输出（例如，图片和对应的类别标签）。
  * 目标是最小化预测输出与实际标签之间的误差。
* **算法示例**：线性回归、逻辑回归、支持向量机、神经网络。

## 非监督学习

* **定义**：使用无标签的数据，模型尝试发现数据中的内在结构或模式。
* **特点**：
  
  * 没有预定义的输出标签，模型基于数据的分布或特征进行学习。
  * 目标是找到数据的隐藏模式或分组。
* **算法示例**：K均值聚类、层次聚类、自编码器、PCA降维。

## 强化学习

# 三要素

## 模型：

模型是机器学习系统用来表示输入数据和输出结果之间关系的数学函数或结构。简单来说，它是数据和预测之间的“映射”。

模型可以看作一个假设函数，用来描述输入特征$X$与目标变量$Y$之间的关系。例如，线性回归模型假设$y = wX + b$.

## 策略

策略是指如何衡量模型的好坏，以及如何通过优化目标来调整模型参数。通常表现为损失函数（Loss Function）和优化目标。

例如线性回归最小二乘法给出的优化函数

## 算法

算法是用于优化模型参数的具体计算方法，基于策略（损失函数）寻找最优解。简单来说，算法是“如何让模型变好”的具体步骤。

例如随机梯度下降法，就是有了策略的优化函数，具体如何求解。

## 两个模型

### **生成模型 (Generative Models)**

生成模型的目标是学习数据的联合概率分布$P(X,Y)=P(X|Y)P(Y)$，进而借助贝叶斯得到条件概率$P(Y|X)$，从而能够生成类似训练数据的新样本。

其**核心思想**是建模数据生成过程，学习数据的概率分布，如朴素贝叶斯 (Naive Bayes)、隐马尔可夫模型 (HMM)等方法，应用于数据生成、数据补全、异常检测等。

## 判别模型 (Discriminative Models)

判别模型的目标是学习给定输入$X$的条件下，直接输出 $Y$的条件概率分布$P(Y|X)$。它专注于给定了$X$，应该得到什么样的$Y$。

典型算法包括逻辑回归、支持向量机等

## 概率论基础知识

**均值**：$E[X] = \int x f(x) dx$(概率论，期望)，$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$（统计学）

**方差**：$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

**协方差**：$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$

**边缘概率 (Marginal Probability)**: 描述单个随机变量的概率，忽略其他变量的影响。对于变量$x$，边缘概率是$P(X = x) = \int P(X = x, Y = y)dy$。如果只有数据集，可以统计$X$取某值的频率
**条件概率 (Conditional Probability)**: 在给定某个事件发生的情况下，另一个事件的概率。$P(X∣Y)$表示在$Y$发生时$X$的概率。$P(X | Y) = \frac{P(X, Y)}{P(Y)}$,如果有数据集，统计在取$Y$值($Y$发生时) $X$发生的频率
**联合概率 (Joint Probability)**: 多个随机变量同时发生的概率。P(X,Y)表示X和Y同时满足特定条件的概率。如果变量独立，$P(X,Y)=P(X)\cdot P(Y)$,如果不独立，需通过联合概率分布表、公式或数据集统计。如果已知条件概率$P(X,Y) = P(X|Y) \cdot P(Y)$



## 矩阵论基础知识

**特征值分解**：对于一个$n \times n$的方阵$A$如果存在标量$\lambda$（特征值）和非零向量$v$（特征向量），满足$A v = \lambda v$,$\lambda$是$A$的特征值，$v$是对应的特征向量,进而可以实现特征值分解$A = V \Lambda V^{-1}$其中$V$为特征向量组成的矩阵，$\Lambda$为特征值组成的对角矩阵。
**奇异值分解**：特征值分解是对于方阵的，那么一般的对于一个$m \times n$矩阵$A$,SVD分解形式为：$A = U \Sigma V^T$,其中：

* $U$:$ m \times m$的正交矩阵（$U^T U = I $，列向量是 $A A^T$ 的特征向量，称为左奇异向量）。
* $\Sigma$: $m \times n$的“对角”矩阵，非负对角元素$\sigma\_1 \geq \sigma\_2 \geq \dots \geq \sigma\_r > 0$称为奇异值，其余元素为 0。
* $V$:$n \times n$的正交矩阵（$V^T V = I$，列向量是$A^T A$,的特征向量，称为右奇异向量）。
* 

**正定**：一个$n \times n$的实对称矩阵$A$(即$A = A^T$）称为​**正定矩阵**​，如果对于任意非零向量$x \in \mathbb{R}^n$，满足$x^T A x > 0$,如果$x^T A x \geq 0 $，则称为​**半正定矩阵**,这里$x^T A x$叫做二次型。

矩阵求导：

* 标量$f=a^Tx$,$\frac{\partial f}{\partial x} = a$
* 标量$f=x^TAx$,$\frac{\partial f}{\partial x} = 2Ax$
* 
* 




