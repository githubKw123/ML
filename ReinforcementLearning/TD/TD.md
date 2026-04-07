---
title: 时序差分方法
tags:
  - reinforcement-learning
  - temporal-difference
  - model-free
  - SARSA
  - Q-learning
aliases:
  - TD
  - 时序差分
---

# 时序差分方法

本节介绍 **Model-Free** 下的时序差分（Temporal-Difference, TD）方法。前置知识见 [[1.introduction（RL）]]、[[DP]] 和 [[MC]]。

> [!info] TD 的定位
> TD 方法结合了 [[MC]] 的**无需环境模型**（Model-Free）和 [[DP]] 的**自举思想**（Bootstrapping），是强化学习中最核心的方法之一。与 MC 不同，TD **无需等到轨迹结束**即可进行更新；与 DP 不同，TD **无需已知动态特性** $p(s',r|s,a)$。

---

## TD 策略评估

### 从 MC 到 TD

回顾 [[MC]] 方法的增量更新公式：

$$
V_{\pi}(s_t) \leftarrow V_{\pi}(s_t) + \alpha \left( G_t - V_{\pi}(s_t) \right)
$$

这里的 $G_t$ 是**完整回报**，需要从当前状态一直等到 Episode 结束才能计算，这带来了两个问题：

1. **时效性差**：必须等到轨迹终止后才能更新
2. **数据需求大**：需要完整的采样轨迹，有时代价很高
3. **不适用于连续任务**：没有终止状态的任务无法使用

### TD(0) 的核心思想

TD 方法的关键洞察：**不必等待真实的完整回报 $G_t$，可以用当前的估计值来替代。**

回忆回报的递推关系：

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

TD(0) 将 $G_{t+1}$ 替换为当前对 $V_{\pi}(S_{t+1})$ 的估计，得到 **TD 目标**（TD Target）：

$$
\underbrace{R_{t+1} + \gamma V_{\pi}(S_{t+1})}_{\text{TD Target}} \approx G_t
$$

### TD(0) 更新公式

$$
V_{\pi}(s_t) \leftarrow V_{\pi}(s_t) + \alpha \left( \underbrace{R_{t+1} + \gamma V_{\pi}(S_{t+1})}_{\text{TD Target}} - V_{\pi}(s_t) \right)
$$

$$
Q_{\pi}(s_t, a_t) \leftarrow Q_{\pi}(s_t, a_t) + \alpha \left( R_{t+1} + \gamma Q_{\pi}(S_{t+1}, A_{t+1}) - Q_{\pi}(s_t, a_t) \right)
$$

> [!important] TD 误差（TD Error）
> 定义 **TD 误差**为：
>
> $$\delta_t = R_{t+1} + \gamma V_{\pi}(S_{t+1}) - V_{\pi}(S_t)$$
>
> TD 误差衡量的是**当前估计**与**一步更优估计**之间的差距。直觉上，$\delta_t > 0$ 说明当前状态的价值被低估了，$\delta_t < 0$ 则说明被高估了。更新过程就是不断缩小这个误差。

> [!example]- 伪代码：TD(0) 策略评估
> **输入**：策略 $\pi$，步长 $\alpha > 0$
> 1. 初始化 $V(s)$ 任意值，$\forall s \in \mathcal{S}$（终止状态 $V(\text{terminal}) = 0$）
> 2. **For** 每个 Episode：
>    1. 初始化 $S_0$
>    2. **For** $t = 0, 1, 2, \ldots$（直到 $S_t$ 为终止状态）：
>       - 按策略 $\pi$ 选择动作 $A_t \sim \pi(\cdot | S_t)$
>       - 执行 $A_t$，观测 $R_{t+1}$ 和 $S_{t+1}$
>       - $V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$

### 自举（Bootstrapping）

TD 更新中用到了 $V_{\pi}(S_{t+1})$——这本身也是一个**估计值**，而非真实值。这种"用估计值来更新估计值"的做法称为**自举**（Bootstrapping）。

> [!note] 自举的利弊
> - **优势**：无需等到 Episode 结束，每一步即可更新；方差更低，因为更新只依赖一步的随机性
> - **代价**：引入了偏差（bias），因为 $V(S_{t+1})$ 本身是不准确的估计
> - **结论**：在实际中，TD 的低方差优势通常使其收敛速度快于 MC

### 收敛性

> [!tip] TD(0) 收敛性保证
> - 对于固定策略 $\pi$，当步长 $\alpha$ 满足 Robbins-Monro 条件（$\sum_t \alpha_t = \infty$，$\sum_t \alpha_t^2 < \infty$）时，TD(0) 以概率 1 收敛到 $V_{\pi}$
> - 使用固定小步长 $\alpha$ 时，TD(0) 收敛到 $V_{\pi}$ 的一个邻域内，适合非平稳环境
> - TD(0) 收敛到的解实际上是**最大似然 MDP 模型**的精确解，在某种意义上比 MC 的均值估计更优

---

## TD 控制

与 [[MC]] 类似，TD 控制也遵循 **GPI**（广义策略迭代）框架：交替进行策略评估（用 TD 方法）和策略改进。根据行动策略与目标策略的关系，分为以下几种算法。

### SARSA（同轨策略 TD 控制）

SARSA 是 **On-policy** 的 TD 控制方法，名字来源于更新所需的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**算法步骤：**

1. 观测当前状态和动作 $(S_t, A_t)$
2. 执行 $A_t$，观测奖励 $R_{t+1}$ 和下一状态 $S_{t+1}$
3. 按**当前策略**（必须是软性策略，如 $\epsilon$-贪心）采样下一动作 $A_{t+1} \sim \pi(S_{t+1})$
4. 更新 Q 值：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$

> [!example]- 伪代码：SARSA
> **输入**：步长 $\alpha > 0$，探索率 $\epsilon > 0$
> 1. 初始化 $Q(s, a)$ 任意值，$\forall s, a$（终止状态 $Q(\text{terminal}, \cdot) = 0$）
> 2. **For** 每个 Episode：
>    1. 初始化 $S_0$，按 $\epsilon$-贪心策略选择 $A_0$
>    2. **For** $t = 0, 1, 2, \ldots$（直到 $S_t$ 为终止状态）：
>       - 执行 $A_t$，观测 $R_{t+1}$, $S_{t+1}$
>       - 按 $\epsilon$-贪心策略选择 $A_{t+1}$
>       - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$
>       - $S_t \leftarrow S_{t+1}$，$A_t \leftarrow A_{t+1}$

> [!note] SARSA 的特点
> - **On-policy**：行动策略与目标策略相同，都是 $\epsilon$-贪心策略
> - **保守性**：由于下一步动作 $A_{t+1}$ 是从软性策略中采样的（包含随机探索），SARSA 会将探索带来的风险纳入价值估计，因此行为偏保守
> - **收敛性**：在适当条件下收敛到 $\epsilon$-贪心策略下的最优 Q 值

---

### 期望 SARSA

期望 SARSA 是 SARSA 的改进版本，关键区别在于：**不对下一步动作进行采样，而是直接计算 Q 值的期望。**

**算法步骤：**

1. 观测当前 $(S_t, A_t)$，执行后得到 $R_{t+1}$, $S_{t+1}$
2. **不采样** $A_{t+1}$，而是对所有可能动作计算期望
3. 更新 Q 值：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a} \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

> [!tip] 期望 SARSA 的优势
> - 通过取期望**消除了对 $A_{t+1}$ 采样的随机性**，因此方差更低
> - 在相同步长下，通常比 SARSA 收敛更快、更稳定
> - 计算量略大（需遍历所有动作），但在动作空间不大时可以接受
> - 当策略为贪心策略时，期望 SARSA 退化为 Q-learning

---

### Q-learning（离轨策略 TD 控制）

Q-learning 是 **Off-policy** 的 TD 控制方法，由 Watkins（1989）提出，是强化学习中最重要的突破之一。

**核心思想**：更新时直接使用下一状态的**最优动作**（贪心），而不是实际执行的动作。

**算法步骤：**

1. 观测当前 $(S_t, A_t)$，执行后得到 $R_{t+1}$, $S_{t+1}$
2. 选择下一步的**最优动作**（贪心策略）：$a^* = \arg\max_a Q(S_{t+1}, a)$
3. 更新 Q 值：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

> [!example]- 伪代码：Q-learning
> **输入**：步长 $\alpha > 0$，探索率 $\epsilon > 0$
> 1. 初始化 $Q(s, a)$ 任意值，$\forall s, a$（终止状态 $Q(\text{terminal}, \cdot) = 0$）
> 2. **For** 每个 Episode：
>    1. 初始化 $S_0$
>    2. **For** $t = 0, 1, 2, \ldots$（直到 $S_t$ 为终止状态）：
>       - 按 $\epsilon$-贪心策略选择 $A_t$（**行动策略**）
>       - 执行 $A_t$，观测 $R_{t+1}$, $S_{t+1}$
>       - $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$
>       - $S_t \leftarrow S_{t+1}$

> [!important] Q-learning 的 Off-policy 本质
> - **行动策略**（behavior policy）：$\epsilon$-贪心策略，负责探索
> - **目标策略**（target policy）：纯贪心策略（$\max$），负责学习最优 Q 值
> - 两个策略不同，因此是 Off-policy 的
> - 关键优势：无需重要性采样修正（因为 $\max$ 操作已隐式地选择了最优动作），比 [[MC#离轨策略方法（Off-policy）|MC Off-policy]] 简洁得多

---

### SARSA vs Q-learning

| 维度 | SARSA | Q-learning |
|------|-------|------------|
| **策略类型** | On-policy | Off-policy |
| **更新目标** | $R + \gamma Q(S', A')$，$A'$ 按当前策略采样 | $R + \gamma \max_a Q(S', a)$ |
| **行为风格** | **保守**：将探索风险纳入价值估计 | **激进**：直接追求最优价值 |
| **收敛目标** | $\epsilon$-贪心策略下的最优 Q | 全局最优 $Q_*$ |
| **安全性** | 更安全（避开高风险区域） | 可能经过高风险区域 |

![[assets/6]]

> [!note] 悬崖行走实验（Cliff Walking）
> 上图是经典的悬崖行走实验。在这个环境中：
> - SARSA 由于考虑了探索时可能掉下悬崖的风险，会学到一条**远离悬崖的安全路径**
> - Q-learning 直接学习最优策略，会找到**紧贴悬崖的最短路径**——虽然理论最优，但探索阶段会频繁掉下悬崖

---

## 多步 TD（TD(n)）

### 动机

TD(0) 只向前看**一步**就进行自举，而 MC 则等到**整条轨迹结束**。多步 TD 是两者之间的过渡——向前看 $n$ 步后再自举，提供了一个连续的偏差-方差权衡谱。

### $n$ 步回报

定义 $n$ 步回报为：

$$
G_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i R_{t+i+1} + \gamma^n V(S_{t+n})
$$

| $n$ 的取值 | 回报形式 | 对应方法 |
|-----------|---------|---------|
| $n = 1$ | $R_{t+1} + \gamma V(S_{t+1})$ | TD(0) |
| $n = 2$ | $R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$ | 2 步 TD |
| $n = k$ | $\sum_{i=0}^{k-1} \gamma^i R_{t+i+1} + \gamma^k V(S_{t+k})$ | $k$ 步 TD |
| $n = \infty$ | $\sum_{i=0}^{\infty} \gamma^i R_{t+i+1} = G_t$ | MC |

### $n$ 步 TD 更新

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t^{(n)} - V(S_t) \right]
$$

对于 Q 值的 $n$ 步更新：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \sum_{i=0}^{n-1} \gamma^i R_{t+i+1} + \gamma^n Q(S_{t+n}, A_{t+n}) - Q(S_t, A_t) \right]
$$

> [!tip] 偏差-方差权衡
> - **$n$ 小**（接近 TD(0)）：偏差大、方差小，更新频繁，适合状态空间大或需要快速学习的场景
> - **$n$ 大**（接近 MC）：偏差小、方差大，需要更多数据，适合奖励信号延迟较长的场景
> - 实际中 $n$ 是一个需要调节的超参数，常见选择为 $n = 3 \sim 10$

---

## TD($\lambda$)

### 动机

多步 TD 需要选择一个固定的 $n$，而 TD($\lambda$) 提供了一种更优雅的方式：**对所有 $n$ 步回报进行加权平均**。

### $\lambda$ 回报

$$
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}, \quad \lambda \in [0, 1]
$$

权重 $(1-\lambda)\lambda^{n-1}$ 构成几何分布，保证总权重为 1。

| $\lambda$ 取值 | 效果 |
|----------------|------|
| $\lambda = 0$ | 退化为 TD(0)，只看一步 |
| $\lambda = 1$ | 退化为 MC，完整回报 |
| $0 < \lambda < 1$ | 介于两者之间的平滑插值 |

### 资格迹（Eligibility Traces）

直接计算 $\lambda$ 回报需要等到轨迹结束（前向视角），这丧失了 TD 的在线优势。**资格迹**（Eligibility Traces）提供了等价的**后向视角**实现，使得 TD($\lambda$) 可以在线逐步更新。

定义每个状态的资格迹 $e_t(s)$：

$$
e_t(s) = \begin{cases} \gamma \lambda \, e_{t-1}(s) + 1, & \text{if } s = S_t \\ \gamma \lambda \, e_{t-1}(s), & \text{if } s \ne S_t \end{cases}
$$

更新规则：

$$
V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s), \quad \forall s
$$

其中 $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 为 TD 误差。

> [!note] 资格迹的直觉
> 资格迹记录了每个状态的"近因性"——最近访问过的状态资格高、远处的状态资格低。当 TD 误差 $\delta_t$ 产生时，所有有资格的状态都会按比例更新，离当前越近的状态更新幅度越大。

---

## 方法总结与对比

### TD vs MC vs DP

| 维度 | DP | MC | TD |
|------|-----|-----|-----|
| **环境模型** | 需要（Model-Based） | 不需要 | 不需要 |
| **自举** | ✓ | ✗ | ✓ |
| **采样** | ✗（全宽度更新） | ✓ | ✓ |
| **更新时机** | 每步遍历所有状态 | Episode 结束 | **每步即更新** |
| **适用任务** | 有限 MDP | Episode 任务 | Episode + 连续任务 |
| **偏差** | 有（自举偏差） | 无 | 有（自举偏差） |
| **方差** | 低 | 高 | 低 |
| **数据效率** | - | 低 | **高** |

### TD 控制算法对比

| 维度 | SARSA | 期望 SARSA | Q-learning |
|------|-------|-----------|------------|
| **策略类型** | On-policy | 可 On/Off | Off-policy |
| **更新目标** | $Q(S', A')$ | $\mathbb{E}[Q(S', \cdot)]$ | $\max_a Q(S', a)$ |
| **方差** | 较高 | **最低** | 中等 |
| **收敛目标** | $Q_{\epsilon\text{-greedy}}$ | 取决于策略 | $Q_*$ |
| **行为风格** | 保守 | 中等 | 激进 |

> [!abstract] 统一视角：GPI 框架
> DP、MC 和 TD 都可以纳入**广义策略迭代**（GPI）框架中——交替进行策略评估与策略改进。它们的区别仅在于**策略评估的实现方式**：
> - **DP**：利用模型进行全宽度自举更新
> - **MC**：通过采样完整轨迹进行无偏估计
> - **TD**：通过采样 + 单步（或多步）自举进行有偏但低方差的估计
>
> TD 方法因其**在线更新、低方差、不依赖模型**的特性，成为现代强化学习（如 DQN、Actor-Critic 等）的基础。详见 [[Policy Based|策略梯度方法]]。
