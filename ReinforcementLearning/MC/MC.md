---
title: 蒙特卡洛方法
tags:
  - reinforcement-learning
  - monte-carlo
  - model-free
  - policy-evaluation
  - policy-control
aliases:
  - MC
  - 蒙特卡洛
---

# 蒙特卡洛方法

本节介绍 **Model-Free** 下的蒙特卡洛（Monte Carlo, MC）方法。前置知识见 [[1.introduction（RL）]] 和 [[DP]]。

## 强化学习基本框架

**两个主体：** Agent、Environment
**一个框架：** MDP
**五大元素：** $S,A,R$（三集合）、$\pi,p(s',r|s,a)$（两分布）

**核心问题：** 如何找到最优策略？

1. 价值函数 $V_{\pi}, q_{\pi}$
2. 策略迭代：策略评估 + 策略改进
3. 价值迭代

> [!info] 从 Model-Based 到 Model-Free
> 上述方法（[[DP]]）的前提是动态特性 $p(s',r|s,a)$ 是**完全已知**的（Model-Based）。面对动态特性未知（Model-Free）的情况，就需要用到**蒙特卡洛方法**：通过大量采样轨迹来估计价值函数，无需环境模型。

---

## MC 策略评估

### 核心思想

考虑贝尔曼期望方程：

$$
V_{\pi}(s) = \mathbb E_{\pi}[G_t|S_t=s]=\sum_{a\in \mathcal A} \pi(a|s) \sum_{r,s'} p(s',r|s,a) [r + \gamma V_{\pi}(s')]
$$

等号右侧的求和需要已知 $p(s',r|s,a)$，MC 方法的策略是**绕过它**——直接利用等号左侧的**期望定义**，借助**大数定律**，通过多次采样轨迹，计算每个状态的平均回报来近似：

$$
V_{\pi}(s) \approx \frac{1}{N} \sum_{i=1}^N G_t^{(i)}
$$

### First-visit MC vs Every-visit MC

> [!note] 两种访问计数方式
> 在一条轨迹中，同一个状态 $s$ 可能被多次访问。对此有两种处理方式：
> - **First-visit MC**：仅统计状态 $s$ 在每条轨迹中**第一次**出现时的回报 $G_t$
> - **Every-visit MC**：统计状态 $s$ 在轨迹中**每一次**出现时的回报 $G_t$
>
> 两者在 $N \to \infty$ 时都收敛到 $V_{\pi}(s)$。First-visit MC 的样本间相互独立，理论分析更简洁；Every-visit MC 数据利用率更高。

### 算法步骤（First-visit MC）

> [!example]- 伪代码：First-visit MC 策略评估
> **输入**：策略 $\pi$，Episode 数量 $N$
> 1. 初始化 $V(s) = 0$，$Returns(s) = []$，$\forall s \in \mathcal{S}$
> 2. **For** 每个 Episode：
>    1. 按策略 $\pi$ 采样一条轨迹 $S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T$
>    2. 令 $G \leftarrow 0$
>    3. **从后向前**遍历 $t = T-1, T-2, \ldots, 0$：
>       - $G \leftarrow \gamma G + R_{t+1}$
>       - 若 $S_t$ 未在 $S_0, S_1, \ldots, S_{t-1}$ 中出现过：
>         - 将 $G$ 追加到 $Returns(S_t)$
>         - $V(S_t) \leftarrow \text{average}(Returns(S_t))$

### 增量更新形式

在实际实现中，无需保存所有历史回报。利用**增量均值公式**，可以在线更新：

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]
$$

其中 $\alpha = \frac{1}{N(S_t)}$ 时等价于精确均值；使用固定步长 $\alpha$ 则可追踪非平稳环境。

> [!tip] 与 TD 的联系
> 这个增量更新公式正是 [[TD]] 算法的出发点——TD 方法进一步将 $G_t$ 替换为单步自举估计 $R_{t+1} + \gamma V(S_{t+1})$，从而无需等到轨迹结束。

### 关于 $q_{\pi}$ 的估计

上述流程是针对 $V_{\pi}$ 的。对于 $q_{\pi}$ 同理，但存在两个额外问题：

1. **数据需求更大**：$q_{\pi}$ 需要采样 $(s,a)$ 对的数据，状态-动作空间比状态空间更大
2. **试探性出发假设**（Exploring Starts）：要求每种 $(s,a)$ 对都以非零概率作为 Episode 起点，以确保每个状态-动作对都有被采样到的可能

---

## MC 控制

类似于 [[DP]] 中的策略迭代，MC 控制同样包含两步：

1. **策略评估**：用 MC 方法评估 $q_{\pi}$
2. **策略改进**：$\pi'(s) = argmax_a q_{\pi}(s,a)$

> [!warning] MC 控制的两个条件
> 1. **无限幕**（Infinite Episodes）：理论上需要无限多的 Episode 才能精确评估 $q_{\pi}$
> 2. **试探性出发假设**：每种 $(s,a)$ 对都需以非零概率作为起点

### 条件的放松

**对于无限幕问题**：采用 **GPI**（广义策略迭代）的思想——不必等策略评估完全收敛，每一幕之后就进行一次策略改进（类似于价值迭代对策略迭代的简化）。

**对于试探性出发假设**：在实际中很难满足，有两种替代方案：

| 方法 | 核心思路 | 策略约束 |
|------|---------|---------|
| **同轨策略**（On-policy） | 行动策略 = 目标策略 | 均为软性策略 |
| **离轨策略**（Off-policy） | 行动策略 ≠ 目标策略 | 行动策略必须软性，目标策略任意 |

---

### 同轨策略方法（On-policy）

同轨策略的关键在于选择一种**软性策略**（soft policy），使得 $\pi(a|s) > 0, \forall s, a$，从而保证充分探索。

#### $\epsilon$-贪心策略

给定 $q_{\pi}(s,a)$，$\epsilon$-贪心策略定义为：

$$
\pi(a|s) = \begin{cases} 1 - \epsilon + \dfrac{\epsilon}{|\mathcal{A}(s)|}, & \text{if } a = argmax_{a'} q_{\pi}(s,a') \\ \dfrac{\epsilon}{|\mathcal{A}(s)|}, & \text{otherwise} \end{cases}
$$

其中 $\epsilon \in (0,1]$ 控制探索程度。当 $\epsilon = 0$ 退化为纯贪心策略。

#### $\epsilon$-贪心策略的改进证明

> [!abstract]- 证明：$\epsilon$-贪心策略改进定理
> 设 $\pi$ 为任意 $\epsilon$-soft 策略，$\pi'$ 为基于 $q_{\pi}$ 的 $\epsilon$-贪心策略。对 $\forall s \in \mathcal{S}$：
>
> $$
> \begin{aligned}
> q_{\pi}(s, \pi'(s)) &= \sum_a \pi'(a|s) \, q_{\pi}(s,a) \\
> &= \frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_{\pi}(s,a) + (1-\epsilon) \max_a q_{\pi}(s,a) \\
> &\ge \frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_{\pi}(s,a) + (1-\epsilon) \sum_a \frac{\pi(a|s) - \frac{\epsilon}{|\mathcal{A}(s)|}}{1-\epsilon} q_{\pi}(s,a) \\
> &= \sum_a \pi(a|s) \, q_{\pi}(s,a) = V_{\pi}(s)
> \end{aligned}
> $$
>
> 其中关键不等式利用了：$\max_a q_{\pi}(s,a) \ge \sum_a w(a) q_{\pi}(s,a)$，对任意权重分布 $w$。
>
> 由策略改进定理，得 $V_{\pi'}(s) \ge V_{\pi}(s)$，$\forall s$。$\blacksquare$

#### On-policy MC 控制算法

> [!example]- 伪代码：On-policy First-visit MC 控制
> **输入**：$\epsilon > 0$
> 1. 初始化 $Q(s,a)$ 任意值，$Returns(s,a) = []$，$\forall s,a$
> 2. $\pi \leftarrow$ 基于 $Q$ 的 $\epsilon$-贪心策略
> 3. **For** 每个 Episode：
>    1. 按策略 $\pi$ 生成轨迹 $S_0, A_0, R_1, \ldots, S_T$
>    2. 令 $G \leftarrow 0$
>    3. **从后向前**遍历 $t = T-1, \ldots, 0$：
>       - $G \leftarrow \gamma G + R_{t+1}$
>       - 若 $(S_t, A_t)$ 未在之前出现过：
>         - 将 $G$ 追加到 $Returns(S_t, A_t)$
>         - $Q(S_t, A_t) \leftarrow \text{average}(Returns(S_t, A_t))$
>         - 更新 $\pi(s)$ 为基于 $Q$ 的 $\epsilon$-贪心策略

> [!warning] On-policy 的局限
> $\epsilon$-贪心策略只能收敛到**$\epsilon$-soft 策略中的最优策略**，而非全局最优策略。这是因为策略始终保持 $\epsilon$ 概率的随机探索，无法完全贪心。

---

### 离轨策略方法（Off-policy）

离轨策略方法使用两个不同的策略：
- **目标策略** $\pi$（target policy）：我们想要学习/优化的策略
- **行动策略** $b$（behavior policy）：实际用于生成数据的策略

> [!important] 覆盖假设（Coverage）
> 要求 $\pi(a|s) > 0 \Rightarrow b(a|s) > 0$，即行动策略必须覆盖目标策略可能选择的所有动作。

#### 重要性采样（Importance Sampling）

由于数据由 $b$ 生成，而我们要估计 $\pi$ 下的期望，需要通过**重要性采样比**（Importance Sampling Ratio）进行修正。

对于从时刻 $t$ 开始的一条轨迹，其在两个策略下的概率之比为：

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

> [!note] 状态转移概率抵消
> 注意轨迹概率中包含的状态转移概率 $p(S_{k+1}|S_k, A_k)$ 在分子分母中完全抵消，因此 $\rho$ **仅依赖于两个策略和轨迹上的动作**，无需已知环境模型。

##### 推导过程

将 $V_{\pi}(s)$ 的期望展开为对轨迹的求和：

$$
V_{\pi}(s) = \mathbb E_{\pi}[G_t|S_t=s] = \sum G_t \prod_{k=t}^{T-1} \left[\pi(A_k|S_k) \, p(R_{k+1},S_{k+1}|S_k,A_k)\right]
$$

将行动策略 $b$ 引入（乘以 $\frac{b}{b}$）：

$$
= \sum \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)} \cdot G_t \cdot \prod_{k=t}^{T-1} \left[b(A_k|S_k) \, p(R_{k+1},S_{k+1}|S_k,A_k)\right]
$$

前面的连乘即为重要性采样比 $\rho_{t:T-1}$，后面的部分则是在行动策略 $b$ 下的轨迹概率。因此：

$$
V_{\pi}(s) = \mathbb E_{b}\left[\rho_{t:T-1} \, G_t \mid S_t = s\right]
$$

#### 两种重要性采样估计

##### 普通重要性采样（Ordinary Importance Sampling）

$$
V_{\pi}(s) \approx \frac{1}{N} \sum_{i=1}^{N} \rho_{t:T-1}^{(i)} \, G_t^{(i)}
$$

- 无偏估计
- 方差可能非常大（甚至无界）

##### 加权重要性采样（Weighted Importance Sampling）

$$
V_{\pi}(s) \approx \frac{\sum_{i=1}^{N} \rho_{t:T-1}^{(i)} \, G_t^{(i)}}{\sum_{i=1}^{N} \rho_{t:T-1}^{(i)}}
$$

- 有偏估计（偏差随 $N$ 增大渐近消失）
- 方差显著更小，实际中**更常用**

> [!tip] 直觉理解
> 加权重要性采样相当于对回报进行**归一化加权平均**，权重大的样本（与目标策略行为更一致的轨迹）贡献更大，从而降低了极端权重带来的方差问题。

#### 增量更新公式

采用加权重要性采样的增量更新形式：

$$
C(S_t) \leftarrow C(S_t) + w
$$

$$
V(S_t) \leftarrow V(S_t) + \frac{w}{C(S_t)} \left[ G_t - V(S_t) \right]
$$

其中 $w = \rho_{t:T-1}$ 为重要性采样权重，$C(S_t)$ 为累积权重。

#### Off-policy MC 控制算法

> [!example]- 伪代码：Off-policy MC 控制（加权重要性采样）
> **输入**：行动策略 $b$（软性策略）
> 1. 初始化 $Q(s,a)$ 任意值，$C(s,a) = 0$，$\forall s,a$
> 2. $\pi(s) \leftarrow \argmax_a Q(s,a)$（确定性贪心策略）
> 3. **For** 每个 Episode：
>    1. 按策略 $b$ 生成轨迹 $S_0, A_0, R_1, \ldots, S_T$
>    2. 令 $G \leftarrow 0$，$W \leftarrow 1$
>    3. **从后向前**遍历 $t = T-1, \ldots, 0$：
>       - $G \leftarrow \gamma G + R_{t+1}$
>       - $C(S_t, A_t) \leftarrow C(S_t, A_t) + W$
>       - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G - Q(S_t, A_t)]$
>       - $\pi(S_t) \leftarrow \argmax_a Q(S_t, a)$
>       - 若 $A_t \ne \pi(S_t)$，则 **break**（提前终止）
>       - $W \leftarrow W \cdot \frac{1}{b(A_t|S_t)}$

> [!warning] Off-policy 的挑战
> - 重要性采样比 $\rho$ 的**方差**随轨迹长度指数增长
> - 若 $\pi$ 和 $b$ 差异较大，收敛速度会很慢
> - 上述伪代码中的提前终止（当动作不匹配时 break）有助于减少方差，但也可能导致学习信号不足

---

## MC 方法总结

### MC vs DP

| 维度 | DP | MC |
|------|-----|-----|
| **环境模型** | 需要 $p(s',r\|s,a)$（Model-Based） | 不需要（Model-Free） |
| **更新方式** | 基于贝尔曼方程的自举（bootstrapping） | 基于完整轨迹的采样平均 |
| **偏差/方差** | 有自举偏差，方差较低 | 无偏，但方差较高 |
| **计算** | 每步需遍历所有状态 | 每步只更新轨迹经过的状态 |

### MC vs TD

| 维度 | MC | [[TD]] |
|------|-----|--------|
| **更新时机** | 需等到 Episode 结束 | 每一步即可更新 |
| **适用场景** | 仅适用于有终止的 Episode 任务 | 也适用于连续任务 |
| **目标值** | $G_t$（完整回报） | $R_{t+1} + \gamma V(S_{t+1})$（TD target） |
| **偏差/方差** | 无偏、高方差 | 有偏、低方差 |
| **自举** | 否 | 是 |

> [!abstract] 三种方法的统一视角
> DP、MC 和 TD 都可以纳入 **GPI**（广义策略迭代）框架中——交替进行策略评估与策略改进。它们的区别仅在于**策略评估的实现方式**：
> - **DP**：利用模型进行全宽度自举更新
> - **MC**：通过采样完整轨迹进行无偏估计
> - **TD**：通过采样 + 单步自举进行有偏但低方差的估计
