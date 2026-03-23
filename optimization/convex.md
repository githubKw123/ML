---
title: 凸优化
tags:
  - optimization
  - math
  - convex
aliases:
  - 凸函数
  - 凸优化问题
---

# 凸优化

本笔记系统介绍凸优化的核心概念：凸集、凸函数（及严格凸、强凸）、约束优化问题的拉格朗日对偶理论与 KKT 条件。更多优化基础见 [[Introduction(optimization)|优化问题]]。

---

## 凸集

### 定义

集合 $C \subseteq \mathbb{R}^n$ 是**凸集**（convex set），当且仅当对任意 $x, y \in C$ 及 $\theta \in [0, 1]$，有：

$$
\theta x + (1 - \theta) y \in C
$$

即集合中任意两点的连线段仍在集合内。

> [!example] 常见凸集
> - 超平面：$\{x \mid a^T x = b\}$
> - 半空间：$\{x \mid a^T x \leq b\}$
> - 球、椭球
> - 多面体（线性不等式的交集）
> - 半正定矩阵锥 $\mathbb{S}_+^n$

> [!tip] 凸集的保持运算
> 凸集经过**交集、仿射变换、透视函数、线性分式函数**等运算后仍为凸集。

---

## 凸函数

### 定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是**凸函数**（convex function），当且仅当 $\text{dom}\ f$ 为凸集，且对任意 $x, y \in \text{dom}\ f$，$\theta \in [0, 1]$，有：

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
$$

![[assets/4]]

几何意义：函数图像上任意两点之间的弦（线段）位于函数图像的上方。

### 判定条件

#### 限制到直线

$$
g(t) = f(x + tv)
$$

$f$ 为高维函数，$g$ 为沿任意方向 $v$ 切出的一维函数。$f$ 为凸函数**当且仅当**对所有 $x \in \text{dom}\ f$ 和 $v \in \mathbb{R}^n$，$g(t)$ 为凸函数。这将高维凸性判定化归为一维问题。

#### 一阶条件

若 $f$ 可微，则 $f$ 为凸函数**当且仅当** $\text{dom}\ f$ 为凸集，且：

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x), \quad \forall\, x, y \in \text{dom}\ f
$$

> [!note] 几何解释
> 凸函数的一阶泰勒近似是其**全局下界**。即函数的切线（切超平面）始终在函数图像下方。

#### 二阶条件

若 $f$ 二阶可微，则 $f$ 为凸函数**当且仅当** $\text{dom}\ f$ 为凸集，且：

$$
\nabla^2 f(x) \succeq 0, \quad \forall\, x \in \text{dom}\ f
$$

即 Hessian 矩阵处处**半正定**。

---

## 严格凸函数

### 定义

函数 $f$ 是**严格凸函数**（strictly convex function），若对任意 $x \neq y \in \text{dom}\ f$，$\theta \in (0, 1)$，不等号严格成立：

$$
f(\theta x + (1 - \theta) y) < \theta f(x) + (1 - \theta) f(y)
$$

> [!important] 严格凸的性质
> 严格凸函数若存在最小值，则**最小值点唯一**。这在机器学习中至关重要（如保证模型参数唯一解）。

### 判定条件

若 $f$ 二阶可微，Hessian 矩阵处处**正定**（$\nabla^2 f(x) \succ 0$）是严格凸的**充分条件**（但非必要条件，例如 $f(x) = x^4$ 在 $x=0$ 处 $f''(0) = 0$，但仍然严格凸）。

---

## 强凸函数

### 定义

函数 $f$ 是 **$m$-强凸函数**（$m$-strongly convex, $m > 0$），若对任意 $x, y \in \text{dom}\ f$，$\theta \in [0, 1]$，有：

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y) - \frac{m}{2} \theta(1 - \theta) \|x - y\|^2
$$

等价地，$f(x) - \dfrac{m}{2}\|x\|^2$ 是凸函数。

### 判定条件

若 $f$ 二阶可微，则 $f$ 为 $m$-强凸**当且仅当**：

$$
\nabla^2 f(x) \succeq mI, \quad \forall\, x \in \text{dom}\ f
$$

即 Hessian 矩阵的最小特征值不小于 $m$。

> [!tip] 三者关系
> $$\text{强凸} \implies \text{严格凸} \implies \text{凸}$$
> 反之均不成立。强凸不仅保证最小值点唯一，还能对收敛速率给出更强的保证——例如梯度下降法在强凸条件下具有**线性收敛**速率。

---

## 凸优化问题

**凸优化问题**是指目标函数为凸函数，且可行域为凸集的优化问题。详见 [[Introduction(optimization)|优化问题]]。

> [!important] 凸优化的核心优势
> 对于凸优化问题，**任何局部最优解都是全局最优解**。

---

## 约束优化与拉格朗日对偶

约束优化问题在支持向量机等模型中非常核心，需要用到拉格朗日对偶性进行求解。

### 原问题

一般的约束优化问题（原问题）可以写成：

$$
\begin{aligned}
\min_{x \in \mathbb{R}^p} \quad & f(x) \\
\text{s.t.} \quad & m_i(x) \leq 0, \quad i = 1, 2, \cdots, M \\
& n_j(x) = 0, \quad j = 1, 2, \cdots, N
\end{aligned}
$$

### Lagrange 函数

定义 Lagrange 函数：

$$
L(x, \lambda, \eta) = f(x) + \sum_{i=1}^{M} \lambda_i m_i(x) + \sum_{j=1}^{N} \eta_j n_j(x)
$$

其中 $\lambda_i \geq 0$ 为不等式约束的**拉格朗日乘子**（KKT 乘子），$\eta_j$ 为等式约束的拉格朗日乘子。

### 等价的无约束形式

原问题可以等价为：

$$
\min_{x \in \mathbb{R}^p} \max_{\lambda \geq 0,\, \eta} L(x, \lambda, \eta)
$$

> [!note] 为何等价？
> - 若 $x$ 满足所有约束（$m_i(x) \leq 0$, $n_j(x) = 0$），则最大化时 $\lambda_i = 0$ 可取最大值，此时 $\max L = f(x)$，直接等价于原问题。
> - 若 $x$ 不满足某个不等式约束（$m_i(x) > 0$），则令对应 $\lambda_i \to +\infty$，有 $\max L = +\infty$。
> - 外层取 $\min$ 时不会选取 $+\infty$ 的情况，因此自动排除了不可行点。

### 对偶问题

交换 $\min$ 和 $\max$ 的顺序，得到**对偶问题**：

$$
\max_{\lambda \geq 0,\, \eta} \min_{x \in \mathbb{R}^p} L(x, \lambda, \eta)
$$

定义**对偶函数**：

$$
g(\lambda, \eta) = \min_{x} L(x, \lambda, \eta)
$$

对偶问题即为关于 $\lambda, \eta$ 的最大化问题：$\max_{\lambda \geq 0,\, \eta} g(\lambda, \eta)$。

### 弱对偶与强对偶

> [!abstract] 弱对偶不等式
> 对偶问题的最优值 $d^*$ 总是不超过原问题的最优值 $p^*$：
> $$d^* = \max_{\lambda, \eta} \min_{x} L(x, \lambda, \eta) \leq \min_{x} \max_{\lambda, \eta} L(x, \lambda, \eta) = p^*$$
>
> **证明**：显然有 $\min_{x} L \leq L \leq \max_{\lambda, \eta} L$，因此 $\max_{\lambda, \eta} \min_{x} L \leq L$，且 $\min_{x} \max_{\lambda, \eta} L \geq L$，对所有 $x, \lambda, \eta$ 成立，故不等式成立。

差值 $p^* - d^*$ 称为**对偶间隙**（duality gap）：

| 对偶关系 | 条件 | 对偶间隙 |
|----------|------|---------|
| **弱对偶** | 始终成立 | $p^* - d^* \geq 0$ |
| **强对偶** | 需额外条件 | $p^* - d^* = 0$ |

### Slater 条件（强对偶的充分条件）

> [!tip] Slater 条件
> 对于**凸优化问题**，若满足 Slater 条件，则强对偶成立。
>
> 记问题的定义域为 $\mathcal{D} = \text{dom}\ f \cap \text{dom}\ m_i \cap \text{dom}\ n_j$。Slater 条件为：
>
> $$
> \exists\, \hat{x} \in \text{relint}\ \mathcal{D}, \quad \text{s.t.} \quad m_i(\hat{x}) < 0, \quad \forall\, i = 1, 2, \cdots, M
> $$
>
> 其中 $\text{relint}$ 表示**相对内部**（不包含边界的内部）。

关于 Slater 条件的说明：

1. 对于大多数凸优化问题，Slater 条件成立。
2. **松弛形式**：若 $M$ 个不等式约束中有 $K$ 个为仿射函数，则仅需其余 $M - K$ 个约束满足严格不等式即可。

---

## KKT 条件

上面介绍了原问题和对偶问题的关系，实际求解时使用 **KKT（Karush-Kuhn-Tucker）条件**。

> [!important] KKT 条件
> 对于凸优化问题，**KKT 条件与强对偶性等价**。最优解 $x^*$、$\lambda^*$、$\eta^*$ 必须满足以下四组条件：
>
> **1. 原始可行性**
> $$
> \begin{aligned}
> m_i(x^*) &\leq 0, \quad i = 1, \cdots, M \\
> n_j(x^*) &= 0, \quad j = 1, \cdots, N
> \end{aligned}
> $$
>
> **2. 对偶可行性**
> $$
> \lambda_i^* \geq 0, \quad i = 1, \cdots, M
> $$
>
> **3. 互补松弛条件**
> $$
> \lambda_i^* m_i(x^*) = 0, \quad \forall\, i = 1, \cdots, M
> $$
>
> **4. 梯度为零（驻点条件）**
> $$
> \frac{\partial L(x, \lambda^*, \eta^*)}{\partial x}\bigg|_{x = x^*} = 0
> $$

### 互补松弛条件的推导

设对偶问题的最优值为 $d^*$，原问题为 $p^*$，在强对偶条件下：

$$
\begin{aligned}
d^* &= \max_{\lambda, \eta} g(\lambda, \eta) = g(\lambda^*, \eta^*) = \min_{x} L(x, \lambda^*, \eta^*) \\
&\leq L(x^*, \lambda^*, \eta^*) = f(x^*) + \sum_{i=1}^{M} \lambda_i^* m_i(x^*) \\
&\leq f(x^*) = p^*
\end{aligned}
$$

强对偶要求 $d^* = p^*$，因此两个不等号必须同时取等：

- **第一个 $\leq$ 取等** $\Rightarrow$ $x^*$ 是 $L(x, \lambda^*, \eta^*)$ 关于 $x$ 的极值点 $\Rightarrow$ **梯度为零条件**
- **第二个 $\leq$ 取等** $\Rightarrow$ $\sum_{i=1}^{M} \lambda_i^* m_i(x^*) = 0$，又因 $\lambda_i^* \geq 0$，$m_i(x^*) \leq 0$，每一项非正，故每一项必须为零 $\Rightarrow$ **互补松弛条件**

> [!note] 互补松弛的直观理解
> 互补松弛条件 $\lambda_i^* m_i(x^*) = 0$ 意味着：
> - 若约束**不紧**（$m_i(x^*) < 0$），则对应乘子 $\lambda_i^* = 0$（该约束"不起作用"）
> - 若乘子**非零**（$\lambda_i^* > 0$），则约束必须**取等**（$m_i(x^*) = 0$，约束"激活"）

---

## 相关笔记

- [[Introduction(optimization)|优化问题]] — 优化理论总览
- [[optimization/LinearSearch|线搜索方法]] — 方向与步长选择
- [[optimization/Newton|牛顿法]] — 基于二阶信息的求解方法
