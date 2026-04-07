---
title: TD3（双延迟深度确定性策略梯度）
tags:
  - reinforcement-learning
  - actor-critic
  - TD3
  - deep-RL
  - off-policy
  - continuous-control
aliases:
  - Twin Delayed DDPG
  - Twin Delayed Deep Deterministic Policy Gradient
  - 双延迟深度确定性策略梯度
---

# TD3（Twin Delayed Deep Deterministic Policy Gradient）

TD3（Twin Delayed DDPG）是 Fujimoto 等人于 2018 年提出的确定性策略梯度算法，针对 DDPG 中 **Q 值过高估计（overestimation）** 问题，引入三项关键改进：**双 Q 网络（Clipped Double-Q）**、**延迟策略更新（Delayed Policy Update）** 和 **目标策略平滑（Target Policy Smoothing）**。TD3 是目前连续动作空间中最稳定、最常用的 off-policy 算法之一。前置知识见 [[PolicyBased]]（Actor-Critic）和 [[TD]]（时序差分学习）。

> [!info] 为什么需要 TD3？
> DDPG（Deep Deterministic Policy Gradient）将 DQN 的思想拓展到连续动作空间，但存在严重的**过高估计问题**：
>
> - **Q 值过高估计**：函数逼近误差在 Bellman 更新中被**正向累积放大**，导致 Critic 高估某些动作的价值
> - **策略利用高估**：确定性策略总是选择 Q 值最高的动作，而这些"高 Q 值"可能是估计误差而非真实优势
> - **恶性循环**：高估的 Q → 错误的策略更新 → 采集更差的数据 → 更大的高估，最终导致训练崩溃
>
> TD3 通过三项针对性的改进，系统性地解决了这些问题。

---

## 理论基础

### DDPG

DDPG 是 Actor-Critic 架构在连续动作空间的经典实现：

- **Actor**（确定性策略）：$\mu_\theta(s)$ 直接输出确定性动作
- **Critic**（Q 网络）：$Q_\phi(s, a)$ 估计状态-动作价值
- **目标网络**：$\mu_{\theta'}$、$Q_{\phi'}$ 通过软更新提供稳定的训练目标

**Critic 更新**（最小化 TD 误差）：

$$
L(\phi) = E\left[\left(Q_\phi(s, a) - y\right)^2\right], \quad y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

**Actor 更新**（确定性策略梯度定理）：

$$
\nabla_\theta J(\theta) = E_s\left[\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]
$$

> [!warning] DDPG 的核心缺陷
> 在目标值 $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$ 的计算中：
>
> 1. $Q_{\phi'}$ 本身存在函数逼近误差
> 2. $\mu_{\theta'}(s')$ 倾向于选择使 $Q_{\phi'}$ 最大的动作
> 3. 取最大值操作使正向误差更容易被选中（$E[\max(X_1, X_2)] \geq \max(E[X_1], E[X_2])$）
> 4. 这个正向偏差通过 Bellman 更新逐步累积，导致 Q 值严重膨胀

### 过高估计的数学分析

假设 Q 网络的估计 $Q_\phi(s,a) = Q^*(s,a) + \epsilon(s,a)$，其中 $\epsilon$ 是均值为零的随机误差。

对于策略 $\mu_\theta(s) = \arg\max_a Q_\phi(s,a)$：

$$
E\left[Q_\phi(s, \mu_\theta(s))\right] = E\left[\max_a Q_\phi(s,a)\right] \geq \max_a Q^*(s,a)
$$

即使误差是**无偏**的，通过 $\max$ 操作后仍会产生**正向偏差**。在多步 Bellman 传播下，这个偏差会被放大：

$$
Q_\phi(s,a) \approx r + \gamma \left(Q^*(s', a^*) + \epsilon_{\text{bias}}\right) + \gamma^2(\ldots)
$$

> [!note] 与 Double DQN 的联系
> 在离散动作空间中，Double DQN 通过**动作选择**和**动作评估**使用不同网络来解决过高估计。TD3 将类似思想拓展到连续动作空间，但采用了不同的策略：取两个 Q 网络的最小值。

---

## TD3 的三项核心改进

### 改进一：双 Q 网络（Clipped Double-Q Learning）

维护**两个独立的 Critic 网络** $Q_{\phi_1}$ 和 $Q_{\phi_2}$，计算目标值时取**两者的最小值**：

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}(s', \tilde{a}')
$$

两个 Critic 独立训练，各自最小化自己的 TD 误差：

$$
L(\phi_i) = E\left[\left(Q_{\phi_i}(s, a) - y\right)^2\right], \quad i = 1, 2
$$

> [!important] 为什么取最小值有效？
> - 两个独立网络的逼近误差分布不同
> - 对于同一个 $(s', a')$，两个网络不太可能在同一方向上同时产生大误差
> - $\min(Q_1, Q_2)$ 倾向于选择**更保守**的估计，从而抑制过高估计
> - 可能引入轻微的**低估偏差**，但低估比高估更安全——不会导致策略盲目追逐虚假的高价值

### 改进二：延迟策略更新（Delayed Policy Update）

**Critic 每更新 $d$ 次**，Actor 才更新一次（论文中 $d = 2$）。

$$
\text{Actor 更新条件：} \quad t \bmod d = 0
$$

直觉：

| 更新频率 | 问题 |
|---|---|
| Actor 与 Critic 同频更新 | Critic 还不准确时，Actor 就基于错误的 Q 值更新，导致策略偏移 |
| Actor 延迟更新 | 等 Critic 收敛到更准确的估计后再更新策略，减少策略更新中的误差传播 |

> [!tip] 延迟更新的附加好处
> - **目标网络也延迟更新**：只在 Actor 更新时才软更新目标网络，进一步稳定训练目标
> - **减少计算开销**：Actor 更新频率降低一半

### 改进三：目标策略平滑（Target Policy Smoothing）

在计算目标值时，对目标策略的动作**添加裁剪噪声**：

$$
\tilde{a}' = \mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

$$
\tilde{a}' = \text{clip}(\tilde{a}', a_{\text{low}}, a_{\text{high}})
$$

其中 $\sigma$ 为目标策略噪声标准差（通常 $\sigma = 0.2$），$c$ 为噪声裁剪范围（通常 $c = 0.5$）。

> [!important] 平滑的核心思想
> 这本质上是一种**正则化**手段：
>
> - 如果 Q 函数在某个动作附近有**尖锐的峰值**（由逼近误差导致的伪峰），添加噪声后这个峰值会被平滑掉
> - 相当于要求 Q 函数在动作空间中是**局部平滑**的，即相似动作应有相似的 Q 值
> - 类似于期望 SARSA 的思想：不只评估单个动作，而是评估目标动作**邻域**的平均价值
>
> $$E_\epsilon\left[Q(s', \mu_{\theta'}(s') + \epsilon)\right] \approx \text{对动作邻域的平均估计}$$

---

## 完整的 TD3 目标函数

### Critic 损失

两个 Q 网络共享同一个目标值 $y$，各自独立优化：

$$
L_{\text{critic}} = \frac{1}{|B|}\sum_{(s,a,r,s') \in B}\left[\left(Q_{\phi_1}(s,a) - y\right)^2 + \left(Q_{\phi_2}(s,a) - y\right)^2\right]
$$

其中：

$$
y = r + \gamma \min_{i=1,2} Q_{\phi'_i}\left(s',\; \text{clip}\left(\mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c),\; a_{\text{low}},\; a_{\text{high}}\right)\right)
$$

### Actor 损失

只使用 $Q_{\phi_1}$（避免双重优化的复杂性）：

$$
L_{\text{actor}} = -\frac{1}{|B|}\sum_{s \in B} Q_{\phi_1}(s, \mu_\theta(s))
$$

### 目标网络软更新

每 $d$ 步（Actor 更新时）执行：

$$
\phi'_i \leftarrow \tau \phi_i + (1 - \tau) \phi'_i, \quad i = 1, 2
$$

$$
\theta' \leftarrow \tau \theta + (1 - \tau) \theta'
$$

---

## TD3 算法流程

### 训练流程

```
┌───────────────────────────────────────────────────────────────────┐
│  初始化：Actor μ_θ, Critic Q_φ1, Q_φ2, 目标网络 θ'←θ, φ'←φ       │
│         经验回放缓冲区 D，初始随机探索填充                            │
│                                                                   │
│  For 每个时间步 t：                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  1. 选择动作（添加探索噪声）：                                 │ │
│  │     a = μ_θ(s) + ε,  ε ~ N(0, σ_explore²)                 │ │
│  │     a = clip(a, a_low, a_high)                              │ │
│  │                                                             │ │
│  │  2. 执行动作，存储转移 (s, a, r, s', done) 到 D              │ │
│  │                                                             │ │
│  │  3. 从 D 中随机采样 mini-batch                               │ │
│  │                                                             │ │
│  │  4. 计算目标值 y（双Q取最小 + 目标策略平滑）：                  │ │
│  │     ã' = μ_θ'(s') + clip(ε, -c, c),  ε ~ N(0, σ_target²)  │ │
│  │     y = r + γ · min(Q_φ1'(s', ã'), Q_φ2'(s', ã'))          │ │
│  │                                                             │ │
│  │  5. 更新两个 Critic（每步都更新）                              │ │
│  │     最小化 MSE(Q_φi(s,a), y),  i = 1, 2                    │ │
│  │                                                             │ │
│  │  6. 延迟更新（每 d 步执行一次）：                              │ │
│  │     ┌───────────────────────────────────────────────┐       │ │
│  │     │ a. 更新 Actor：最大化 Q_φ1(s, μ_θ(s))         │       │ │
│  │     │ b. 软更新目标网络：θ', φ1', φ2'               │       │ │
│  │     └───────────────────────────────────────────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

> [!example]- 伪代码：TD3
> **输入**：Critic 学习率 $\alpha_Q$，Actor 学习率 $\alpha_\mu$，目标平滑系数 $\tau$，策略延迟 $d$，目标噪声 $\sigma$，噪声裁剪 $c$
> 1. 初始化 Critic $Q_{\phi_1}, Q_{\phi_2}$，Actor $\mu_\theta$，目标网络 $\phi_1' \leftarrow \phi_1, \phi_2' \leftarrow \phi_2, \theta' \leftarrow \theta$
> 2. 初始化经验回放缓冲区 $\mathcal{D}$
> 3. **For** $t = 1, 2, \ldots$：
>    1. 观测状态 $s$，选择动作 $a = \mu_\theta(s) + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)$
>    2. 执行 $a$，观测 $r, s', d$，存入 $\mathcal{D}$
>    3. 从 $\mathcal{D}$ 采样 mini-batch $\{(s, a, r, s', d)\}$
>    4. $\tilde{a}' \leftarrow \mu_{\theta'}(s') + \text{clip}(\mathcal{N}(0, \sigma^2), -c, c)$
>    5. $y \leftarrow r + \gamma(1-d) \min_{i=1,2} Q_{\phi'_i}(s', \tilde{a}')$
>    6. 更新 $\phi_i$：$\phi_i \leftarrow \phi_i - \alpha_Q \nabla_{\phi_i} \frac{1}{|B|}\sum(Q_{\phi_i}(s,a) - y)^2$
>    7. **If** $t \bmod d = 0$：
>       - 更新 $\theta$：$\theta \leftarrow \theta + \alpha_\mu \nabla_\theta \frac{1}{|B|}\sum Q_{\phi_1}(s, \mu_\theta(s))$
>       - 软更新：$\phi'_i \leftarrow \tau\phi_i + (1-\tau)\phi'_i$，$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$

---

## 代码实现解析

本目录下的 `td3.py` 实现了 TD3 算法在 **Pendulum-v1** 环境上的训练。以下是核心实现要点与算法公式的对应关系。

### 网络架构

```
Actor (确定性策略网络)                  Critic (双Q网络，在同一个类中)
┌──────────────────────┐               ┌──────────────────────────┐
│ Input: state(3)      │               │ Input: [state(3), action(1)]│
│       ↓              │               │         ↓           ↓      │
│ Linear(3→400)        │               │    ┌─────────┐ ┌─────────┐│
│ ReLU                 │               │    │  Q1分支  │ │  Q2分支  ││
│       ↓              │               │    │ 4→400    │ │ 4→400   ││
│ Linear(400→300)      │               │    │ ReLU     │ │ ReLU    ││
│ ReLU                 │               │    │ 400→300  │ │ 400→300 ││
│       ↓              │               │    │ ReLU     │ │ ReLU    ││
│ Linear(300→1)        │               │    │ 300→1    │ │ 300→1   ││
│ tanh × max_action    │               │    │ → Q1(s,a)│ │→ Q2(s,a)││
│ → μ(s) ∈ [-2, 2]    │               │    └─────────┘ └─────────┘│
└──────────────────────┘               └──────────────────────────┘
```

> [!note] 双 Q 网络的实现方式
> 代码中两个 Q 网络封装在同一个 `Critic` 类中（`fc1-fc3` 为 Q1，`fc4-fc6` 为 Q2），共享一个优化器。`forward()` 同时返回 Q1 和 Q2，另有 `Q1()` 方法单独返回 Q1（用于 Actor 更新）。

### 关键代码对应

**目标值计算**（双 Q 取最小 + 目标策略平滑）：

$$
\tilde{a}' = \text{clip}\left(\mu_{\theta'}(s') + \text{clip}(\epsilon, -0.5, 0.5),\; -2,\; 2\right), \quad \epsilon \sim \mathcal{N}(0, 0.2^2)
$$

$$
y = r + (1-d) \cdot 0.99 \cdot \min\left(Q_{\phi'_1}(s', \tilde{a}'),\; Q_{\phi'_2}(s', \tilde{a}')\right)
$$

**Critic 损失**：

$$
L_{\text{critic}} = \text{MSE}(Q_{\phi_1}(s,a),\; y) + \text{MSE}(Q_{\phi_2}(s,a),\; y)
$$

**Actor 损失**（每 2 步更新一次）：

$$
L_{\text{actor}} = -\frac{1}{|B|}\sum Q_{\phi_1}(s,\; \mu_\theta(s))
$$

**软更新**（仅在 Actor 更新时执行）：

$$
\phi'_i \leftarrow 0.005 \cdot \phi_i + 0.995 \cdot \phi'_i, \quad \theta' \leftarrow 0.005 \cdot \theta + 0.995 \cdot \theta'
$$

---

## TD3 的实践技巧

### 探索策略

TD3 使用**独立的探索噪声**（非目标策略噪声）：

$$
a = \mu_\theta(s) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{explore}}^2)
$$

可以使用噪声衰减策略提升后期性能：

$$
\sigma_{\text{explore}}(t) = \max\left(\sigma_0 \cdot (1 - t/T), \sigma_{\min}\right)
$$

### 经验回放预填充

训练开始前使用随机策略预填充经验池，确保初期有足够的数据多样性：

```python
# 代码中的实现
for _ in range(1000):
    action = env.action_space.sample()  # 完全随机动作
    ...
    replay_buffer.add(state, action, reward, next_state, done)
```

### 常见 trick 汇总

| 技巧 | 说明 |
|---|---|
| **经验池预填充** | 用随机策略收集初始数据，避免 Critic 在空池上过拟合 |
| **探索噪声衰减** | 逐渐减小 $\sigma_{\text{explore}}$，从探索转向利用 |
| **奖励缩放/归一化** | 对奖励进行归一化，稳定 Q 值数量级 |
| **观测归一化** | 对输入状态使用 running mean/std 归一化 |
| **梯度裁剪** | 限制梯度范数，防止更新过大 |
| **网络宽度** | 使用较宽的网络（如 400-300），确保足够的表达能力 |
| **目标网络初始化** | 目标网络参数必须初始化为与主网络相同 |

---

