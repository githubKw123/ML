---
tags:
  - optimization
  - line-search
aliases:
  - 线搜索
  - 一维搜索
---

# 线搜索方法 (Line Search Method)

线搜索是迭代优化的基本框架：在每一步选择一个**下降方向** $p_k$，再沿该方向确定合适的**步长** $\alpha_k$，从而产生新的迭代点：

$$
x_{k+1} = x_k + \alpha_k p_k
$$

---

# 方向

**假设**
1. 函数单调下降，$\{f(x_k)\}_{k=0}^\infty,\; f(x_{k+1})<f(x_{k})$
2. $\alpha_k$ 足够小
3. $\|p_k\|=1$

因为 $x_{k+1}=x_k+\alpha_kp_k$，所以 $f(x_{k+1})=f(x_k+\alpha_kp_k)$，一阶泰勒展开：

$$
f(x_k+\alpha_kp_k) = f(x_{k})+\alpha_k\nabla f_k^Tp_k+O(\alpha_k^2)
$$

那么

$$
f(x_{k+1})-f(x_k) \approx \alpha_k\nabla f_k^Tp_k
$$

只考虑方向的话，让 $\nabla f_k^Tp_k<0$ 就行，即 $p_k$ 与梯度的夹角大于 90° 即可。

自然而然地，当 $p_k$ 取**负梯度方向**时：

$$
p_k = -\frac{\nabla f_k}{\|\nabla f_k\|}
$$

此时 $\nabla f_k^T p_k = -\|\nabla f_k\|$ 达到最小值，也就是**最速下降方向 (Steepest Descent Direction)**。

> [!note] 常见下降方向
> | 方法 | 方向 $p_k$ |
> |------|-----------|
> | 最速下降法 | $-\nabla f_k$ |
> | 牛顿法 | $-(\nabla^2 f_k)^{-1}\nabla f_k$ |
> | 拟牛顿法 | $-B_k^{-1}\nabla f_k$，$B_k$ 为 Hessian 近似 |
> | 共轭梯度法 | $-\nabla f_k + \beta_k p_{k-1}$ |

---

# 步长

## 精确搜索 (Exact Line Search)

方向确定后，将 $\alpha_k$ 视为自变量，求使目标函数最小的步长：

$$
\alpha_k = \mathop{\arg\min}_{\alpha>0}\; f(x_k+\alpha\, p_k)
$$

即求解一维优化问题。精确搜索在理论上最优，但实际中往往**计算代价过高**，因此更常用非精确搜索。

---

## 非精确搜索 (Inexact Line Search)

有时精确求解代价太大，那么 $\alpha_k$ 可以取一个满足一定条件的**近似值**来代替。

定义辅助函数：$\phi(\alpha) = f(x_k + \alpha\, p_k)$，则 $\phi(0) = f(x_k)$，$\phi'(0) = \nabla f_k^T p_k < 0$。

### 必要条件（充分下降条件）

$\alpha_k$ 最基本的要求是保证函数值下降：

$$
\phi(\alpha) < \phi(0) \quad\Longleftrightarrow\quad f(x_k+\alpha\, p_k) < f(x_k)
$$

但仅要求函数值下降还不够——步长可能过小导致收敛极慢，所以需要更强的条件。

---

### Armijo 条件（充分下降条件）

又称**第一 Wolfe 条件**，要求函数值充分下降：

$$
\phi(\alpha) \leq \phi(0) + c_1\,\alpha\,\nabla f_k^T p_k, \quad c_1 \in (0,1)
$$

几何含义：函数值必须落在切线 $\phi(0)+\alpha\,\nabla f_k^Tp_k$ 与水平线 $\phi(0)$ 之间的一条线以下。通常取 $c_1 = 10^{-4}$。

> [!warning] 仅用 Armijo 条件不够
> Armijo 条件只给了**上限**，$\alpha$ 可以任意小都满足，需要配合下面的条件给出**下限**。

---

### Goldstein 条件

在 Armijo 条件基础上，增加一个**下界**防止步长过小：

$$
\phi(0) + (1-c)\,\alpha\,\nabla f_k^T p_k \leq \phi(\alpha) \leq \phi(0) + c\,\alpha\,\nabla f_k^T p_k, \quad c \in (0, 0.5)
$$

几何含义：可接受的 $\alpha$ 使得 $\phi(\alpha)$ 夹在两条线之间：
- **上限线**：$l(\alpha)=\phi(0)+c\,\nabla f_k^Tp_k\,\alpha$（Armijo 线）
- **下限线**：$l(\alpha)=\phi(0)+(1-c)\,\nabla f_k^Tp_k\,\alpha$

![[Optimization/assets/1.png]]

> [!tip] Goldstein 条件的特点
> - 结构简单，计算代价低
> - 缺点：下界是线性的，可能排除掉函数的极小值点（不保证收敛到最优 $\alpha$）

---

### Wolfe 条件

Wolfe 条件用**曲率条件 (Curvature Condition)** 替代 Goldstein 的线性下界，效果更好：

**Wolfe 条件 = Armijo 条件 + 曲率条件**

$$
\begin{cases}
\phi(\alpha) \leq \phi(0) + c_1\,\alpha\,\nabla f_k^T p_k & \text{(Armijo / 充分下降)} \\[6pt]
\nabla f(x_k+\alpha\, p_k)^T p_k \geq c_2\,\nabla f_k^T p_k & \text{(曲率条件)}
\end{cases}
$$

其中 $0 < c_1 < c_2 < 1$，常用取值为 $c_1=10^{-4}$，$c_2=0.9$（拟牛顿法）或 $c_2=0.1$（共轭梯度法）。

曲率条件的含义：要求在 $\alpha$ 处的斜率 $\phi'(\alpha)$ 不能太负（即曲线已经足够"平坦"了），从而排除了过小的步长。

![[3.png]]

#### 强 Wolfe 条件 (Strong Wolfe Conditions)

将曲率条件改为取绝对值，使搜索点更接近驻点：

$$
\begin{cases}
\phi(\alpha) \leq \phi(0) + c_1\,\alpha\,\nabla f_k^T p_k \\[6pt]
|\nabla f(x_k+\alpha\, p_k)^T p_k| \leq c_2\,|\nabla f_k^T p_k|
\end{cases}
$$

强 Wolfe 条件排除了 $\phi'(\alpha)$ 为较大正值的点，使步长更精确。

---

## 回溯法 (Backtracking Line Search)

回溯法是最常用的非精确线搜索实现，只使用 **Armijo 条件**，通过不断缩减步长来找到满足条件的 $\alpha$：

> [!abstract] 算法流程
> **输入**：下降方向 $p_k$，初始步长 $\bar{\alpha}>0$，参数 $c\in(0,1)$，缩减因子 $\rho\in(0,1)$
>
> 1. 令 $\alpha \leftarrow \bar{\alpha}$
> 2. **while** $f(x_k+\alpha\, p_k) > f(x_k) + c\,\alpha\,\nabla f_k^Tp_k$：
>    - $\alpha \leftarrow \rho\,\alpha$ （缩减步长）
> 3. **return** $\alpha_k = \alpha$

常用参数：$\bar{\alpha}=1$（牛顿法），$c=10^{-4}$，$\rho\in[0.1,\,0.8]$。

回溯法简单高效，适合牛顿法和拟牛顿法（这些方法提供了天然的初始步长 $\bar{\alpha}=1$）。

---

## 总结对比

| 条件 | 公式 | 特点 |
|------|------|------|
| **精确搜索** | $\alpha_k=\arg\min_\alpha f(x_k+\alpha p_k)$ | 理论最优，计算代价高 |
| **Armijo** | $\phi(\alpha)\leq\phi(0)+c_1\alpha\phi'(0)$ | 仅上限，需配合其他条件 |
| **Goldstein** | 两条线性界夹逼 | 简单，但可能错过极小值 |
| **Wolfe** | Armijo + 曲率条件 | 最常用，理论性质好 |
| **强 Wolfe** | Armijo + 强曲率条件 | 更精确，共轭梯度法常用 |
| **回溯法** | Armijo + 逐步缩减 | 实现最简单，实践中高效 |

> [!important] 存在性定理
> 对于连续可微且下有界的函数 $f$，只要 $p_k$ 是下降方向，则满足 Wolfe 条件（或 Goldstein 条件）的步长 $\alpha_k$ **一定存在**。
