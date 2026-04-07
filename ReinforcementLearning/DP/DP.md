---
title: 动态规划求解MDP
tags:
  - reinforcement-learning
  - dynamic-programming
  - policy-iteration
  - value-iteration
aliases:
  - DP
  - 动态规划
---

# 动态规划

这一节主要是用**动态规划方法**求解 MDP 问题。前置知识见 [[1.introduction（RL）]]。

> [!info] 前提假设
> 动态规划方法要求 MDP 的动态特性 $p(s',r|s,a)$ 是**完全已知**的。若动态特性未知，则需要使用 [[MC]] 或 [[TD]] 等方法。

DP 求解 MDP 的核心思路是利用**贝尔曼方程的递推结构**，将求解最优策略的问题分解为子问题迭代求解。主要包括两种方式：

- **策略迭代**（Policy Iteration）：交替进行策略评估与策略改进
- **价值迭代**（Value Iteration）：将策略评估截断为一步，直接迭代价值函数

---

## 策略迭代

策略迭代包括以下两个交替进行的过程：

1. **策略评估**（Policy Evaluation）：给定策略 $\pi$，计算其价值函数 $V_{\pi}$
2. **策略改进**（Policy Improvement）：基于 $V_{\pi}$，构造更优的新策略 $\pi'$

$$
\pi_0 \xrightarrow{\text{评估}} V_{\pi_0} \xrightarrow{\text{改进}} \pi_1 \xrightarrow{\text{评估}} V_{\pi_1} \xrightarrow{\text{改进}} \cdots \xrightarrow{} \pi_* \xrightarrow{} V_*
$$

### 策略评估

已知 MDP 动态特性 $p(s',r|s,a)$，给定策略 $\pi$，求 $V_{\pi}$。记 $V_{\pi} = (V_{\pi}(s_1), V_{\pi}(s_2), \ldots, V_{\pi}(s_{|\mathcal{S}|}))^T$。

依照贝尔曼期望方程：

$$
V_{\pi}(s) = \sum_{a\in \mathcal A} \pi(a|s) \, Q_{\pi}(s,a) = \sum_{a\in \mathcal A} \pi(a|s) \sum_{r,s'} p(s',r|s,a) \left[ r + \gamma V_{\pi}(s') \right]
$$

#### 解析解（矩阵形式）

将上式写成矩阵形式 $V_{\pi} = R_{\pi} + \gamma P_{\pi} V_{\pi}$，可得解析解：

$$
V_{\pi} = (I - \gamma P_{\pi})^{-1} R_{\pi}
$$

> [!warning] 解析解的局限
> - 矩阵求逆的复杂度为 $O(|\mathcal{S}|^3)$，当状态空间较大时计算代价极高
> - 实际中一般不使用解析解，而是采用迭代方法

#### 迭代策略评估

构造一个序列 $\{V_k\}_{k=0}^{\infty}$，使其收敛到 $V_{\pi}$。

**迭代公式：**

$$
V_{k+1}(s) = \sum_{a\in \mathcal A} \pi(a|s) \sum_{r,s'} p(s',r|s,a) \left[ r + \gamma V_{k}(s') \right], \quad \forall s \in \mathcal{S}
$$

> [!tip] 收敛性保证
> 当 $\gamma < 1$ 时，迭代策略评估是一个**压缩映射**（Contraction Mapping）。由 Banach 不动点定理可知，无论初始值 $V_0$ 如何选取，序列 $\{V_k\}$ 都会收敛到唯一不动点 $V_{\pi}$。
>
> 收敛速率：$\|V_{k+1} - V_{\pi}\|_{\infty} \le \gamma \|V_k - V_{\pi}\|_{\infty}$，即每步误差至少缩小 $\gamma$ 倍。

> [!example]- 伪代码：迭代策略评估
> **输入**：策略 $\pi$，阈值 $\theta > 0$
> 1. 初始化 $V(s) = 0$，$\forall s \in \mathcal{S}$
> 2. **Repeat**：
>    - $\Delta \leftarrow 0$
>    - **For** 每个 $s \in \mathcal{S}$：
>      - $v \leftarrow V(s)$
>      - $V(s) \leftarrow \sum_{a} \pi(a|s) \sum_{r,s'} p(s',r|s,a)[r + \gamma V(s')]$
>      - $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
>    - **Until** $\Delta < \theta$
> 3. **输出** $V \approx V_{\pi}$

### 策略改进

给定当前策略 $\pi$，计算得到新策略 $\pi'$，使其优于 $\pi$。

#### 策略改进定理

> [!abstract] 策略改进定理（Policy Improvement Theorem）
> 给定两个确定性策略 $\pi$ 和 $\pi'$，如果对所有状态 $s \in \mathcal{S}$ 满足：
>
> $$q_{\pi}(s, \pi'(s)) \ge V_{\pi}(s)$$
>
> 那么策略 $\pi'$ 不劣于 $\pi$，即对所有状态 $s \in \mathcal{S}$：
>
> $$V_{\pi'}(s) \ge V_{\pi}(s)$$

#### 贪心策略改进

有了判断方法，如何构造 $\pi'$？使用**贪心策略**：

$$
\pi'(s) = \arg\max_{a} \, q_{\pi}(s, a), \quad \forall s \in \mathcal{S}
$$

> [!note]- 简单证明
> 对 $\forall s \in \mathcal{S}$：
>
> $$V_{\pi}(s) \le \max_a q_{\pi}(s,a) = q_{\pi}(s, \pi'(s))$$
>
> 由策略改进定理，得 $V_{\pi'}(s) \ge V_{\pi}(s)$。$\square$

> [!important] 最优性判断
> 当 $V_{\pi'}(s) = V_{\pi}(s)$，$\forall s \in \mathcal{S}$ 时，说明贪心策略已无法找到更优策略。此时贝尔曼最优方程成立，即已达到 $V_*$。

### 策略迭代算法流程

> [!example]- 伪代码：策略迭代
> 1. **初始化**：任意选取 $\pi$ 和 $V$
> 2. **策略评估**：根据当前 $\pi$，用迭代策略评估计算 $V_{\pi}$
> 3. **策略改进**：
>    - $\text{stable} \leftarrow \text{true}$
>    - **For** 每个 $s \in \mathcal{S}$：
>      - $a_{\text{old}} \leftarrow \pi(s)$
>      - $\pi(s) \leftarrow \arg\max_{a} \sum_{r,s'} p(s',r|s,a)[r + \gamma V(s')]$
>      - 若 $a_{\text{old}} \ne \pi(s)$，则 $\text{stable} \leftarrow \text{false}$
>    - 若 $\text{stable} = \text{true}$，输出 $V_*$ 和 $\pi_*$，**停止**
>    - 否则回到步骤 2

> [!tip] 收敛性
> 有限 MDP 只有有限个策略，每次策略改进都严格提升价值函数（除非已达最优），因此策略迭代在**有限步内**必定收敛到最优策略 $\pi_*$。

---

## 价值迭代

### 动机

策略迭代存在一个效率问题：策略评估本身就是一个迭代过程，这就产生了**迭代套迭代**的结构。考虑到中间的 $V_{\pi}$ 并不需要精确计算——我们只需要它能引导出更好的策略即可——因此可以对策略评估进行**截断**。

极端情况：策略评估**只迭代一步**，然后立即进行策略改进。

### 推导

策略评估走一步：

$$
q_{k+1}(s,a) = \sum_{r,s'} p(s',r|s,a) \left[ r + \gamma V_{k}(s') \right]
$$

结合贪心策略改进 $V_{k+1}(s) = \max_a \, q_{k+1}(s,a)$，得到：

$$
\boxed{V_{k+1}(s) = \max_a \sum_{r,s'} p(s',r|s,a) \left[ r + \gamma V_{k}(s') \right]}
$$

> [!success] 关键观察
> 这个迭代公式**与策略 $\pi$ 完全无关**！只需要不断迭代价值函数 $V$ 即可直接逼近最优价值函数 $V_*$。这就是**价值迭代**（Value Iteration）。

> [!note] 与贝尔曼最优方程的关系
> 价值迭代本质上就是将**贝尔曼最优方程** $V_*(s) = \max_a \sum_{r,s'} p(s',r|s,a)[r + \gamma V_*(s')]$ 转化为迭代更新规则。收敛后的不动点即为 $V_*$。

### 算法流程

> [!example]- 伪代码：价值迭代
> **输入**：阈值 $\theta > 0$
> 1. 初始化 $V(s) = 0$，$\forall s \in \mathcal{S}$
> 2. **Repeat**：
>    - $\Delta \leftarrow 0$
>    - **For** 每个 $s \in \mathcal{S}$：
>      - $v \leftarrow V(s)$
>      - $V(s) \leftarrow \max_a \sum_{r,s'} p(s',r|s,a)[r + \gamma V(s')]$
>      - $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
>    - **Until** $\Delta < \theta$
> 3. **输出策略**：$\pi(s) = \arg\max_a \sum_{r,s'} p(s',r|s,a)[r + \gamma V(s')]$

> [!tip] 收敛性
> 价值迭代同样是压缩映射，由 Banach 不动点定理保证收敛到 $V_*$。实际中价值迭代往往比策略迭代需要更多的迭代次数，但每次迭代的计算量更小（无需内层循环）。

---

## 异步动态规划

### 动机

无论是策略迭代还是价值迭代，每一步迭代都需要**遍历所有状态**进行更新（full sweep），当状态空间非常大时代价很高。异步动态规划（Asynchronous DP）放松了这个要求：每次迭代**只更新部分状态**，只要所有状态在迭代过程中都能被无限次更新到，就仍然能保证收敛。

### 常见变体

| 方法 | 策略 | 特点 |
|------|------|------|
| **原地 DP**（In-place DP） | 只维护一个价值数组，更新时直接覆盖 | 节省内存，收敛通常更快 |
| **优先级扫描**（Prioritized Sweeping） | 优先更新贝尔曼误差最大的状态 | 更高效地分配计算资源 |
| **实时 DP**（Real-time DP） | 只更新 Agent 实际访问到的状态 | 适用于在线场景，忽略不相关状态 |

> [!tip] 异步 DP 的意义
> 异步方法使得 DP 可以应用于更大规模的问题，同时为后续的 [[TD]] 等在线学习方法提供了理论基础——TD 方法可以看作是一种基于采样的异步 DP。

---

## 策略迭代 vs 价值迭代

| 维度 | 策略迭代 | 价值迭代 |
|------|---------|---------|
| **结构** | 外层：策略循环；内层：评估迭代 | 单层迭代 |
| **每步计算** | 需要完整策略评估（多次 sweep） | 仅一次 sweep |
| **迭代次数** | 通常较少（策略空间有限） | 通常较多 |
| **总体效率** | 状态空间小时更优 | 状态空间大时更优 |
| **统一视角** | 截断 $k=\infty$ 步评估 | 截断 $k=1$ 步评估 |

> [!abstract] 广义策略迭代（GPI）
> 策略迭代和价值迭代可以看作**广义策略迭代**（Generalized Policy Iteration）框架的两个极端。GPI 的核心思想是：策略评估和策略改进交替进行，无论评估截断到什么程度（$k=1$ 到 $k=\infty$），只要两个过程持续交互，最终都能收敛到最优策略。这一思想也贯穿于 [[MC]] 和 [[TD]] 方法中。

---

## 动态规划的局限性

> [!warning] DP 方法的适用边界
> - **维度灾难**（Curse of Dimensionality）：状态和动作空间增大时，计算量呈指数增长
> - **Model-Based**：要求完全已知 $p(s',r|s,a)$，现实问题中往往无法满足
>
> 当模型未知时，需要转向 Model-Free 方法：
> - [[MC|蒙特卡洛方法]]：通过完整的采样轨迹估计价值函数
> - [[TD|时序差分方法]]：结合自举（bootstrapping）与采样，无需等到轨迹结束
