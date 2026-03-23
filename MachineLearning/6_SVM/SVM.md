---
title: 支撑向量机（SVM）
tags:
  - 机器学习
  - 监督学习
  - 分类
  - SVM
aliases:
  - SVM
  - Support Vector Machine
created: 2026-03-15
---

# 支撑向量机

> [!abstract] 三大核心
> SVM 有三宝：**间隔、对偶、核技巧**

> [!info] 核心思想
> 对于数据空间里的一组数据，找到一个超平面，实现两类数据的分类，并使找到的这个超平面**最好**。
>
> 不同于感知机（只需找到任意一个可分超平面，且结果与初始解强相关），SVM 关注的是：在所有可行超平面中，**哪个最优**。

![[MachineLearning/6_SVM/assets/1.png]]

对于怎么定义超平面好不好，这里具体分为三种手段：

1. **Hard-margin SVM** — [[#硬间隔SVM（Hard-margin SVM）]]
2. **Soft-margin SVM** — [[#软间隔 （Soft-margin SVM）]]
3. **Kernel Method** → 详见 [[KernelMethod]]

---

## 硬间隔SVM（Hard-margin SVM）

### 模型

对于一组数据$\{x_i,y_i\}_{i=1}^N,x_i\in\mathbb{R}^p,y_i\in\{1,-1\}$,硬间隔的目的就是找到一个最大的间隔函数$margin(w,b)$实现对数据的分割

$$
max\ margin(w,b)\\
s.t.\ y_i(w^Tx+b)>0
$$

这里最大的间隔函数$margin(w,b)$定义为离直线$w^Tx+b$最近的那个点的距离，也就是

$$
\min_{x_i}\frac{|w^Tx_i+b|}{||w||}
$$

那么最大化间隔这个约束问题就可以表示为（约束为分类任务的要求）：

$$
arg\max_{w,b}[\min_{x_i}\frac{|w^Tx_i+b|}{||w||}]\\ s.t.\ y_i(w^Tx_i+b)>0
$$

因为$y_i$只能取1或-1,其相当于代替了绝对值的作用，那么原式就可以表示为：

$$
arg\max_{w,b}\min_{x_i}\frac{y_i(w^Tx_i+b)}{||w||}\\ s.t.\ y_i(w^Tx_i+b)>0
$$

对于这个约束 $y_i(w^Tx_i+b)>0$，不妨固定 $\min y_i(w^Tx_i+b)=1>0$

> [!tip]- 为什么可以固定为 1？
> - 对于 $y_i(w^Tx_i+b)$ 来说，它等于某个大于零的数 $r$
> - 对于超平面 $w^Tx_i+b$ 来说，$w^T$ 和 $b$ 是变量，所以超平面等于几无所谓，对应的 $w^T$ 和 $b$ 除以 $r$ 即可，因此可以做这样的简化
> - **几何意义**：去掉 $y_i=\pm 1$，$w^Tx_i+b=r$ 是离目标超平面最近的一个平行超平面，它约束的是那些离目标超平面最近的点——这些点后面称为**支持向量**

$$
arg\max_{w,b}\frac{1}{||w||}\\ s.t.\ y_i(w^Tx_i+b) \ge 1
$$

这就是一个包含 $N$ 个约束的凸优化问题，写成标准形式（优化问题里最大化$\frac{1}{||w||}$和最小化$\frac{1}{2}||w||^2$等价）：

$$
\min_{w,b}\frac{1}{2}w^Tw\\ s.t.\ y_i(w^Tx_i+b) \ge 1
$$

### 求解算法（对偶问题）

如果样本数量或维度非常高，直接求解困难甚至不可解，于是需要对这个问题进一步处理。
这里我们用拉格朗日对偶性分析这个问题，引入 Lagrange 函数：

$$
L(w,b,\lambda)=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b))
$$

我们有原问题就等价于：

$$
\min_{w,b}\max_{\lambda}L(w,b,\lambda_i)\\
 s.t.\ \lambda_i\ge0
$$

> [!question]- 为什么等价？
> 当约束满足时（$1-y_i(w^Tx_i+b)\le0$），最大化 $\lambda$ 只能取 0，目标退化为 $\frac{1}{2}w^Tw$；当约束不满足时，$\lambda$ 可以取无穷大，目标为 $+\infty$。因此外层最小化会自动避开不满足约束的区域。

我们交换最小和最大值的符号得到对偶问题：

$$
\max_{\lambda_i}\min_{w,b}L(w,b,\lambda_i)\ s.t.\ \lambda_i\ge0
$$

由于这个约束问题是个凸优化（目标函数是凸的，约束是仿射的），满足 **Slater 条件**，**强对偶性**成立，对偶问题和原问题等价，这里就可以直接写出KKT条件：

> [!note]- KKT 条件
> $$
> \frac{\partial L}{\partial w}=0,\quad\frac{\partial L}{\partial b}=0
> $$
> $$
> \lambda_k(1-y_k(w^Tx_k+b))=0 \quad (\text{互补松弛条件})
> $$
> $$
> \lambda_i\ge0,\quad 1-y_i(w^Tx_i+b)\le0
> $$

* $b$：$\frac{\partial}{\partial b}L=0\Rightarrow\sum\limits_{i=1}^N\lambda_iy_i=0$
* $w$：首先将 $b$ 代入：

  $$
  L(w,b,\lambda_i)=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i(1-y_iw^Tx_i-y_ib)=\frac{1}{2}w^Tw+\sum\limits_{i=1}^N\lambda_i-\sum\limits_{i=1}^N\lambda_iy_iw^Tx_i
  $$

  $$
  \frac{\partial}{\partial w}L=0\Rightarrow w=\sum\limits_{i=1}^N\lambda_iy_ix_i
  $$
* 将上面两个参数代入：

  $$
  L(w,b,\lambda_i)=-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i
  $$

根据KKT条件就得到了对应的最佳参数（$\lambda$求解下面那个优化问题就可以得到）：

$$
\hat{w}=\sum\limits_{i=1}^N\lambda_iy_ix_i \\
$$$$
\hat{b}=y_k-w^Tx_k=y_k-\sum\limits_{i=1}^N\lambda_iy_ix_i^Tx_k,  \exists k,\ 1-y_k(w^Tx_k+b)=0 \\

$$
$$
\max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i,\ s.t.\ \lambda_i\ge0
$$

![[MachineLearning/6_SVM/assets/2.png]]
### 支持向量的含义

$b$ 的求解来自互补松弛条件 $\lambda_k(1-y_k(w^Tx_k+b))=0$：

- 当点落在支持向量上（即 $1-y_k(w^Tx_k+b)=0$）时，$\lambda_k$ 才会有非零值
- 对于非支持向量，通过 $\lambda_k=0$ 将其排除
- $\lambda_k$ 起到了**筛选支持向量**的作用

实际求解时，得到 $\lambda_k$ 后会发现大部分样本对应的 $\lambda_k=0$，而 $\lambda_k\ne0$ 的那些就是支持向量上的点，对这些点求解即可得到最终的目标超平面。

> [!important] 支持向量的几何意义
> 支持向量就是距离分类超平面**最近**的那些样本点，它们"支撑"起了间隔边界。SVM 的决策边界**仅由支持向量决定**，与其他样本点无关——这也是 SVM ==稀疏性==的体现。

---

## 软间隔 （Soft-margin SVM）

Hard-margin 的 SVM 只对**线性可分**数据可解。如果数据不可分、或存在噪声，它会把噪声当成支持向量。于是我们引入软间隔。

> [!info] 核心思想
> 加一个损失函数，允许模型有一定的失误。

### 模型

我们的基本想法是在损失函数中加入错误分类的可能性。错误分类的个数可以写成：

$$
error=\sum\limits_{i=1}^N\mathbb{I}\{y_i(w^Tx_i+b)\lt1\}
$$

这个函数不连续，可以将其改写为

$$
y_i(w^Tx_i+b)\ge1 \  \ \ loss=0\\
y_i(w^Tx_i+b)<1 \  \ \ loss=1-y_i(w^Tx_i+b)
$$

也就是

$$
\max\{0,1-y_i(w^Tx_i+b)\}
$$

这个式子又叫做 **Hinge Loss（铰链损失函数）**。

![[MachineLearning/6_SVM/assets/4.png]]
> [!example]- 与其他损失函数的对比
> | 损失函数 | 公式 | 特点 |
> |---------|------|------|
> | **0-1 Loss** | $\mathbb{I}\{yf(x)<0\}$ | 不连续、不可导，无法优化 |
> | **Hinge Loss** | $\max\{0,1-yf(x)\}$ | 连续、凸函数，是 0-1 Loss 的上界 |
> | **Logistic Loss** | $\log(1+e^{-yf(x)})$ | 光滑近似，对应逻辑回归 |
> | **Exponential Loss** | $e^{-yf(x)}$ | 对应 AdaBoost |

将这个损失函数加入 Hard-margin SVM 中，于是：

$$
\min\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\max\{0,1-y_i(w^Tx_i+b)\}
$$

这个式子中，常数 $C$ 可以看作允许的错误水平（正则化参数）：

> [!tip] 超参数 $C$ 的作用
> - **$C$ 越大**：对误分类惩罚越大，间隔越小，模型越复杂（趋向硬间隔）
> - **$C$ 越小**：允许更多误分类，间隔越大，模型越简单（更强的正则化）

为了进一步消除 $\max$ 符号，对数据集中的每一个观测，我们令$\xi_i=\max\{0,1-y_i(w^Tx_i+b)\}$,则$\xi_i \ge 0$

对于$\xi$（松弛变量）的含义我们发现，其实就是我们把误差点距离放宽到$w^Tx+b=1-\xi_i$，让部分样本点可以违反，但是会对目标值产生影响，因此这部分约束变成 $y_i(w^Tx+b)\ge1-\xi_i$，进一步的化简：

$$
\min_{w,b,\xi}\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\xi_i\\ s.t.\ y_i(w^Tx_i+b)\ge1-\xi_i,\ \xi_i\ge0,\ i=1,2,\cdots,N
$$

> [!note] 松弛变量 $\xi_i$ 的几何解释
> - $\xi_i=0$：样本点在间隔边界上或正确一侧（分类正确且在 margin 外）
> - $0<\xi_i<1$：样本点在间隔内部但分类正确
> - $\xi_i=1$：样本点恰好在决策边界上
> - $\xi_i>1$：样本点被**错误分类**

### 求解算法（对偶问题）

求解思路和硬间隔一样，构造 Lagrange 函数：

$$
L(w,b,\xi,\lambda,\mu)=\frac{1}{2}w^Tw+C\sum\limits_{i=1}^N\xi_i+\sum\limits_{i=1}^N\lambda_i(1-\xi_i-y_i(w^Tx_i+b))-\sum\limits_{i=1}^N\mu_i\xi_i
$$

其中 $\lambda_i\ge0,\mu_i\ge0$ 分别是两组约束的 Lagrange 乘子。

对 $w,b,\xi$ 分别求偏导并令其为零：

$$
\frac{\partial L}{\partial w}=0\Rightarrow w=\sum\limits_{i=1}^N\lambda_iy_ix_i
$$

$$
\frac{\partial L}{\partial b}=0\Rightarrow\sum\limits_{i=1}^N\lambda_iy_i=0
$$

$$
\frac{\partial L}{\partial \xi_i}=0\Rightarrow C-\lambda_i-\mu_i=0\Rightarrow \lambda_i=C-\mu_i
$$

由于 $\mu_i\ge0$，所以 $\lambda_i\le C$。将以上结果代入得到对偶问题：

$$
\max_{\lambda}-\frac{1}{2}\sum\limits_{i=1}^N\sum\limits_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j+\sum\limits_{i=1}^N\lambda_i\\
s.t.\ 0\le\lambda_i\le C,\ \sum\limits_{i=1}^N\lambda_iy_i=0
$$

> [!important] 硬间隔 vs 软间隔对偶形式
> 两者**几乎一样**，唯一区别在于 $\lambda_i$ 多了上界 $C$：
> - 硬间隔：$\lambda_i\ge0$
> - 软间隔：$0\le\lambda_i\le C$

对应的 KKT 条件（互补松弛条件）：

$$
\lambda_i(1-\xi_i-y_i(w^Tx_i+b))=0\\
\mu_i\xi_i=(C-\lambda_i)\xi_i=0
$$

由此可以分析支持向量的分类：
- $\lambda_i=0$：非支持向量，样本在间隔外，分类正确
- $0<\lambda_i<C$：$\xi_i=0$，样本恰好在间隔边界上（**边界支持向量**，用于计算 $b$）
- $\lambda_i=C$：$\xi_i\ge0$，样本在间隔内部或被误分类

$b$ 的求解：选取边界支持向量（$0<\lambda_k<C$）：

$$
\hat{b}=y_k-\sum\limits_{i=1}^N\lambda_iy_ix_i^Tx_k
$$

实际中通常取所有边界支持向量的平均值以提高数值稳定性。

---

## SMO 算法

上面的对偶问题最终都归结为一个二次规划（QP）问题。当样本量 $N$ 很大时，通用的 QP 求解器效率低下。**序列最小优化（Sequential Minimal Optimization, SMO）** 算法是 SVM 最常用的高效求解方法。

**基本思路**：每次只选取**两个**变量 $\lambda_i,\lambda_j$ 进行优化（因为等式约束 $\sum\lambda_iy_i=0$ 的存在，至少要同时更新两个变量），固定其他所有变量，将问题化简为一个只有**两个变量**的 QP 子问题，该子问题可以**解析求解**。

**算法步骤**：
1. **选择变量**：选取违反 KKT 条件最严重的 $\lambda_i$，配对选取使目标函数下降最快的 $\lambda_j$
2. **解析更新**：固定其他变量，解析求解两个变量的子问题，并将结果裁剪到 $[0,C]$ 范围
3. **更新参数**：更新 $b$ 和误差缓存
4. **重复**直到所有变量满足 KKT 条件（在容差范围内）

> [!tip] SMO 的关键优势
> 每个子问题有**解析解**，不需要调用数值优化器，因此效率非常高。这是 `libsvm` 等工业级 SVM 库的核心算法。

---

## SVM 总结与对比

| 特性 | Hard-margin SVM | Soft-margin SVM |
|------|----------------|-----------------|
| **适用场景** | 数据线性可分 | 数据近似线性可分/有噪声 |
| **约束条件** | $y_i(w^Tx_i+b)\ge1$ | $y_i(w^Tx_i+b)\ge1-\xi_i$ |
| **对偶约束** | $\lambda_i\ge0$ | $0\le\lambda_i\le C$ |
| **超参数** | 无 | $C$（正则化参数） |
| **鲁棒性** | 对噪声敏感 | 对噪声有一定容忍度 |

> [!success] SVM 的优点
> - 最终决策函数仅由少数支持向量决定，具有==稀疏性==，计算复杂度取决于支持向量数目而非样本维度
> - 通过核技巧（→ [[KernelMethod]]）可以高效处理非线性分类
> - 有严格的数学理论支撑（结构风险最小化、VC 维理论）
> - 在中小规模数据上表现优异

> [!failure] SVM 的缺点
> - 大规模数据训练效率低（即使用 SMO，时间复杂度也在 $O(N^2)$ 到 $O(N^3)$ 之间）
> - 对超参数 $C$ 和核函数参数敏感，需要仔细调参
> - 原生只支持二分类，多分类需要额外策略（如 one-vs-one, one-vs-rest）
> - 对缺失数据敏感，不能直接处理
