# 优化问题

### 前言

对于优化问题，可能分为凸优化问题和非凸优化，对于凸优化可能有一定方法能得到最优解，但是很多非凸问题就没有这么简单。

### 模型

$$
\mathop{min}\limits _{x\in \mathbb{R^n}}f(x)
$$

**数值解**：通过数值方法（不断迭代）近似计算得到的解，结果通常是一系列离散的数值点

$$
\{x_k\}_{k=0}^\infty
$$

**单调性：**

$$
\{f(x_k)\}_{k=0}^\infty,f(x_{k+1})<f(x_{k-m})
$$

**策略：**
线搜索方法：先定方向，再定步长（步长更重要）$x_{k+1}=x_k+\alpha_kp_k$
信赖域方法：方向步长一起定$x_{k+1}=x_k+p_k$

