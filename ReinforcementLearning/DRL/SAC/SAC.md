---
title: SAC（柔性演员-评论家）
tags:
  - reinforcement-learning
  - actor-critic
  - SAC
  - deep-RL
  - off-policy
  - continuous-control
  - maximum-entropy
aliases:
  - Soft Actor-Critic
  - 柔性演员-评论家
  - 最大熵强化学习
---

# SAC（Soft Actor-Critic）

SAC（Soft Actor-Critic）是 Haarnoja 等人于 2018 年提出的**最大熵强化学习**算法，核心思想是在最大化累积奖励的同时**最大化策略熵**，从而实现更好的探索和更鲁棒的策略。SAC 结合了 off-policy 的数据效率、随机策略的探索能力，以及自动温度调节机制，是目前连续动作空间中**超参数最鲁棒、最广泛使用**的算法之一。前置知识见 [[PolicyBased]]（Actor-Critic、策略梯度）和 [[TD]]（时序差分学习）。

> [!info] 为什么需要 SAC？
> 在 SAC 之前的连续控制算法存在以下问题：
>
> - **DDPG**：确定性策略，探索能力弱，训练极不稳定，超参数敏感
> - **[[ReinforcementLearning/DRL/TD3/TD3|TD3]]**：解决了 DDPG 的过高估计问题，但仍使用确定性策略，探索依赖人工设计的噪声
> - **PPO**：稳定但数据效率低（on-policy），每批数据只用一次
>
> SAC 的突破在于：用**最大熵框架**将探索内化到目标函数中，不再需要外部探索噪声，同时保持 off-policy 的数据效率和训练稳定性。

---

## 理论基础

### 最大熵强化学习框架

标准 RL 的目标是最大化累积奖励：

$$
J(\pi) = \sum_{t=0}^{T} E_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t)\right]
$$

**最大熵 RL** 在此基础上增加策略熵正则项，形成**软目标**（Soft Objective）：

$$
J_{\text{soft}}(\pi) = \sum_{t=0}^{T} E_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

其中 $\mathcal{H}(\pi(\cdot|s)) = -E_{a \sim \pi}\left[\log \pi(a|s)\right]$ 是策略在状态 $s$ 下的熵，$\alpha > 0$ 是**温度参数**，控制熵正则化的强度。

> [!important] 最大熵目标的直觉
> 这个目标要求策略在**完成任务的前提下，尽可能保持随机**：
>
> - **高熵** = 策略输出接近均匀分布，探索充分
> - **低熵** = 策略集中在少数动作上，行为确定
>
> 最优策略会在"获得更多奖励"和"保持更多随机性"之间找到平衡。$\alpha$ 越大，越鼓励探索；$\alpha$ 越小，越接近标准 RL。

### 最大熵的好处

为什么要最大化熵？这不仅仅是为了"探索"：

| 好处 | 说明 |
|---|---|
| **更好的探索** | 高熵策略自然地覆盖更多状态-动作空间，避免陷入局部最优 |
| **多模态策略** | 允许学习多种解决方案，而非只记住一条最优轨迹 |
| **鲁棒性** | 对环境扰动更鲁棒——策略不会过度依赖某个特定动作序列 |
| **迁移学习** | 学到的策略更"通用"，可作为下游任务的良好初始化 |
| **组合性** | 多个最大熵策略可以通过简单的数学操作进行组合 |

### 软贝尔曼方程

在最大熵框架下，标准的贝尔曼方程被修改为**软贝尔曼方程**（Soft Bellman Equation）：

**软 Q 函数**：

$$
Q_{\text{soft}}(s_t, a_t) = r(s_t, a_t) + \gamma \, E_{s_{t+1}}\left[V_{\text{soft}}(s_{t+1})\right]
$$

**软价值函数**：

$$
V_{\text{soft}}(s_t) = E_{a_t \sim \pi}\left[Q_{\text{soft}}(s_t, a_t) - \alpha \log \pi(a_t|s_t)\right]
$$

将 $V$ 代入 $Q$ 的定义，得到**软 Bellman 备份**：

$$
Q_{\text{soft}}(s_t, a_t) = r(s_t, a_t) + \gamma \, E_{s_{t+1}, a_{t+1} \sim \pi}\left[Q_{\text{soft}}(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1}|s_{t+1})\right]
$$

> [!note] 与标准贝尔曼方程的区别
> 唯一的区别是在未来价值中**减去了 $\alpha \log \pi$**（即加上了熵奖励 $\alpha \mathcal{H}$）。这意味着：
>
> - 高确定性的动作（低熵）会被"惩罚"
> - 在 Q 值相近的动作中，策略会倾向于选择更分散的分布
> - 随着 $\alpha \to 0$，退化为标准贝尔曼方程

### 最优软策略

在最大熵框架下，最优策略具有**能量基分布**（energy-based）的解析形式：

$$
\pi^*(a|s) = \frac{\exp\left(\frac{1}{\alpha}Q_{\text{soft}}^*(s, a)\right)}{Z(s)}
$$

其中 $Z(s) = \int \exp\left(\frac{1}{\alpha}Q^*(s, a')\right) da'$ 是配分函数（归一化常数）。

等价地：

$$
\pi^*(a|s) \propto \exp\left(\frac{1}{\alpha}Q_{\text{soft}}^*(s, a)\right)
$$

> [!tip] 直觉理解
> 最优策略按 Q 值的**指数权重**分配概率：
>
> - Q 值高的动作获得指数级更大的概率
> - 但不像确定性策略那样只选最高 Q 的动作
> - $\alpha$ 越小，分布越尖锐（越接近确定性）；$\alpha$ 越大，分布越平坦（越接近均匀）

---

## SAC 的三个核心组件

### 组件一：Clipped Double-Q（双 Q 网络）

与 [[ReinforcementLearning/DRL/TD3/TD3|TD3]] 相同，SAC 使用两个独立的 Q 网络来抑制过高估计：

$$
y = r + \gamma\left(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a'|s')\right), \quad a' \sim \pi_\theta(\cdot|s')
$$

注意与 TD3 的关键区别：

| | TD3 | SAC |
|---|---|---|
| 目标值中的动作来源 | 目标策略 $\mu_{\theta'}(s')$ + 噪声 | 当前策略采样 $a' \sim \pi_\theta(\cdot\|s')$ |
| 熵项 | 无 | $-\alpha \log \pi_\theta(a'\|s')$ |
| 策略平滑 | 显式裁剪噪声 | 随机策略天然提供 |

### 组件二：随机策略与重参数化

SAC 使用**压缩高斯策略**（Squashed Gaussian Policy）：

**1. 网络输出均值和对数标准差**：

$$
\mu_\theta(s), \log\sigma_\theta(s) = f_\theta(s)
$$

**2. 重参数化采样**：

$$
\xi \sim \mathcal{N}(0, I), \quad u = \mu_\theta(s) + \sigma_\theta(s) \odot \xi
$$

**3. Tanh 压缩到有界动作空间**：

$$
a = a_{\max} \cdot \tanh(u)
$$

**4. 对数概率计算**（需要考虑 tanh 变换的雅可比行列式）：

$$
\log \pi_\theta(a|s) = \log \mathcal{N}(u; \mu, \sigma^2) - \sum_{i=1}^{d}\log\left(1 - \tanh^2(u_i)\right) - \log a_{\max}
$$

> [!important] 重参数化技巧（Reparameterization Trick）
> 为什么不直接从 $\pi_\theta$ 采样？因为**采样操作不可微**，梯度无法通过采样节点反向传播。
>
> 重参数化将随机性"外置"到噪声 $\xi$ 上：
>
> $$a = g_\theta(s, \xi) = a_{\max} \cdot \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \xi)$$
>
> 这样 $a$ 关于 $\theta$ 是可微的，梯度可以正常流过 $\mu_\theta$ 和 $\sigma_\theta$。

> [!warning] Tanh 对数概率的修正
> 直接用高斯分布的 $\log p$ 是错误的！$\tanh$ 变换改变了概率密度，需要**变量替换公式**（change of variables）：
>
> $$\log \pi(a|s) = \log p(u|s) - \log\left|\det\frac{\partial a}{\partial u}\right|$$
>
> 由于 $\tanh$ 是逐元素操作，雅可比行列式的对数简化为：$\sum_i \log(1 - \tanh^2(u_i))$。
> 代码中使用 `log(1 - y_t.pow(2) + 1e-6)` 实现，加 $10^{-6}$ 防止 $\tanh$ 在边界处导致 $\log(0)$。

### 组件三：自动温度调节（Automatic Temperature Tuning）

手动调节 $\alpha$ 非常困难——不同任务、不同训练阶段需要不同的 $\alpha$。SAC 将 $\alpha$ 也作为可优化变量，通过约束优化自动调节。

**目标**：找到 $\alpha$ 使得策略熵满足目标约束：

$$
\max_\pi \; E\left[\sum_t r(s_t, a_t)\right] \quad \text{s.t.} \quad E_{(s_t, a_t) \sim \rho_\pi}\left[-\log\pi(a_t|s_t)\right] \geq \mathcal{H}_{\text{target}}, \; \forall t
$$

转化为对偶问题后，$\alpha$ 的损失函数为：

$$
L(\alpha) = -\alpha \cdot E_{a_t \sim \pi_\theta}\left[\log\pi_\theta(a_t|s_t) + \mathcal{H}_{\text{target}}\right]
$$

在实践中，优化 $\log\alpha$ 而非 $\alpha$，确保 $\alpha > 0$：

$$
L(\log\alpha) = -\log\alpha \cdot E_{a_t \sim \pi_\theta}\left[\log\pi_\theta(a_t|s_t) + \mathcal{H}_{\text{target}}\right]
$$

**目标熵的启发式设定**：

$$
\mathcal{H}_{\text{target}} = -\dim(\mathcal{A})
$$

即负的动作空间维度。例如，动作维度为 1 时，$\mathcal{H}_{\text{target}} = -1$。

> [!tip] 自动温度调节的直觉
> - 当策略熵 $\mathcal{H}(\pi) < \mathcal{H}_{\text{target}}$（探索不足）时：
>   - $\log\pi + \mathcal{H}_{\text{target}} < 0$ → 梯度使 $\alpha$ 增大 → 更强的熵正则化 → 鼓励探索
> - 当策略熵 $\mathcal{H}(\pi) > \mathcal{H}_{\text{target}}$（探索过多）时：
>   - $\log\pi + \mathcal{H}_{\text{target}} > 0$ → 梯度使 $\alpha$ 减小 → 减弱熵正则化 → 允许更确定的行为
>
> 这形成了一个**自稳定的反馈机制**，免去了手动调参的负担。

---

## 完整的 SAC 目标函数

### Critic 损失

两个 Q 网络共享同一个软 Bellman 目标值 $y$，各自独立优化：

$$
L_{\text{critic}} = \frac{1}{|B|}\sum_{(s,a,r,s') \in B}\left[\left(Q_{\phi_1}(s,a) - y\right)^2 + \left(Q_{\phi_2}(s,a) - y\right)^2\right]
$$

其中：

$$
y = r + \gamma(1 - d)\left(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log\pi_\theta(a'|s')\right), \quad a' \sim \pi_\theta(\cdot|s')
$$

> [!note] SAC 没有 Actor 目标网络
> 与 TD3 不同，SAC **不需要 Actor 的目标网络**。目标值中的 $a'$ 从当前策略 $\pi_\theta$ 采样（而非目标策略），只有 Critic 使用目标网络 $Q_{\phi'}$。

### Actor 损失

通过重参数化采样，Actor 损失为：

$$
L_{\text{actor}} = \frac{1}{|B|}\sum_{s \in B}\left[\alpha \log\pi_\theta(\tilde{a}|s) - \min_{i=1,2} Q_{\phi_i}(s, \tilde{a})\right]
$$

其中 $\tilde{a} = a_{\max} \cdot \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \xi)$，$\xi \sim \mathcal{N}(0, I)$。

直觉：Actor 要找到使 $Q$ 值最大、同时熵尽可能高的策略。

### 温度损失

$$
L(\alpha) = -\alpha \cdot \frac{1}{|B|}\sum_{s \in B}\left[\log\pi_\theta(\tilde{a}|s) + \mathcal{H}_{\text{target}}\right]
$$

### 目标网络软更新

每步（不延迟）执行：

$$
\phi'_i \leftarrow \tau \phi_i + (1 - \tau) \phi'_i, \quad i = 1, 2
$$

---

## SAC 算法流程

### 训练流程

```
┌───────────────────────────────────────────────────────────────────┐
│  初始化：Actor π_θ, Critic Q_φ1, Q_φ2, 目标网络 φ'←φ              │
│         温度参数 log α, 目标熵 H_target = -dim(A)                  │
│         经验回放缓冲区 D                                           │
│                                                                   │
│  For 每个时间步 t：                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  1. 选择动作（策略自带随机性，无需额外噪声）：                   │ │
│  │     a ~ π_θ(·|s)                                            │ │
│  │                                                             │ │
│  │  2. 执行动作，存储转移 (s, a, r, s', done) 到 D              │ │
│  │                                                             │ │
│  │  3. 从 D 中随机采样 mini-batch                               │ │
│  │                                                             │ │
│  │  4. 更新 Critic（软 Bellman 目标 + 双Q取最小）：               │ │
│  │     a' ~ π_θ(·|s')                                          │ │
│  │     y = r + γ(1-d)[min(Q_φ1'(s',a'), Q_φ2'(s',a'))         │ │
│  │                     - α·log π_θ(a'|s')]                     │ │
│  │     最小化 MSE(Q_φi(s,a), y),  i = 1, 2                    │ │
│  │                                                             │ │
│  │  5. 更新 Actor（最大化 Q 值 + 熵）：                          │ │
│  │     ã ~ π_θ(·|s)  (重参数化)                                 │ │
│  │     最小化 α·log π_θ(ã|s) - min(Q_φ1(s,ã), Q_φ2(s,ã))     │ │
│  │                                                             │ │
│  │  6. 更新温度参数 α：                                          │ │
│  │     最小化 -α·(log π_θ(ã|s) + H_target)                     │ │
│  │                                                             │ │
│  │  7. 软更新目标 Critic 网络：                                   │ │
│  │     φ'_i ← τ·φ_i + (1-τ)·φ'_i                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

> [!example]- 伪代码：SAC
> **输入**：Critic 学习率 $\alpha_Q$，Actor 学习率 $\alpha_\pi$，温度学习率 $\alpha_T$，软更新系数 $\tau$，目标熵 $\bar{\mathcal{H}}$
> 1. 初始化 Critic $Q_{\phi_1}, Q_{\phi_2}$，Actor $\pi_\theta$，目标网络 $\phi'_1 \leftarrow \phi_1, \phi'_2 \leftarrow \phi_2$
> 2. 初始化温度 $\log\alpha$，经验回放缓冲区 $\mathcal{D}$
> 3. **For** $t = 1, 2, \ldots$：
>    1. 观测状态 $s$，采样动作 $a \sim \pi_\theta(\cdot|s)$
>    2. 执行 $a$，观测 $r, s', d$，存入 $\mathcal{D}$
>    3. 从 $\mathcal{D}$ 采样 mini-batch $\{(s, a, r, s', d)\}$
>    4. **Critic 更新**：
>       - $a' \sim \pi_\theta(\cdot|s')$
>       - $y \leftarrow r + \gamma(1-d)\left(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log\pi_\theta(a'|s')\right)$
>       - $\phi_i \leftarrow \phi_i - \alpha_Q \nabla_{\phi_i} \frac{1}{|B|}\sum(Q_{\phi_i}(s,a) - y)^2$
>    5. **Actor 更新**：
>       - $\tilde{a} \sim \pi_\theta(\cdot|s)$（重参数化）
>       - $\theta \leftarrow \theta - \alpha_\pi \nabla_\theta \frac{1}{|B|}\sum\left[\alpha\log\pi_\theta(\tilde{a}|s) - \min_{i} Q_{\phi_i}(s, \tilde{a})\right]$
>    6. **温度更新**：
>       - $\alpha \leftarrow \alpha - \alpha_T \nabla_\alpha \left(-\alpha \cdot \frac{1}{|B|}\sum\left[\log\pi_\theta(\tilde{a}|s) + \bar{\mathcal{H}}\right]\right)$
>    7. **软更新**：$\phi'_i \leftarrow \tau\phi_i + (1-\tau)\phi'_i$

### 与 TD3 流程的关键区别

| 步骤 | TD3 | SAC |
|---|---|---|
| 动作选择 | $\mu_\theta(s) + \epsilon_{\text{explore}}$ | $a \sim \pi_\theta(\cdot\|s)$ |
| 目标动作 | $\mu_{\theta'}(s') + \text{clip}(\epsilon)$ | $a' \sim \pi_\theta(\cdot\|s')$ |
| 目标值 | $\min Q'_i(s', \tilde{a}')$ | $\min Q'_i(s', a') - \alpha\log\pi(a'\|s')$ |
| Actor 更新 | 延迟（每 $d$ 步） | **每步**都更新 |
| 目标网络更新 | 延迟（每 $d$ 步） | **每步**都更新 |
| 额外优化变量 | 无 | 温度 $\alpha$ |

---

## 代码实现解析

本目录下的 `sac.py` 实现了 SAC 算法在 **Pendulum-v1** 环境上的训练。以下是核心实现要点与算法公式的对应关系。

### 网络架构

```
Actor (随机策略网络)                    Critic (双Q网络)
┌──────────────────────┐               ┌──────────────────────────┐
│ Input: state(3)      │               │ Input: [state(3), action(1)]│
│       ↓              │               │         ↓           ↓      │
│ Linear(3→256)        │               │    ┌─────────┐ ┌─────────┐│
│ ReLU                 │               │    │  Q1分支  │ │  Q2分支  ││
│       ↓              │               │    │ 4→256    │ │ 4→256   ││
│ Linear(256→256)      │               │    │ ReLU     │ │ ReLU    ││
│ ReLU                 │               │    │ 256→256  │ │ 256→256 ││
│       ↓              │               │    │ ReLU     │ │ ReLU    ││
│ ┌─────────┐          │               │    │ 256→1    │ │ 256→1   ││
│ │ μ分支    │ σ分支    │               │    │→ Q1(s,a) │ │→ Q2(s,a)││
│ │256→1    │ 256→1    │               │    └─────────┘ └─────────┘│
│ │→ μ(s)   │→ log σ(s)│               └──────────────────────────┘
│ └─────────┘          │
│ 重参数化: u = μ + σ·ξ │
│ a = 2·tanh(u)        │
└──────────────────────┘
```

> [!note] Actor 输出均值和标准差
> 与 TD3 的 Actor（仅输出确定性动作）不同，SAC 的 Actor 输出**两个头**：
> - `mean` 分支：输出高斯分布的均值 $\mu$
> - `log_std` 分支：输出对数标准差 $\log\sigma$，裁剪到 $[-20, 2]$ 防止数值问题
>
> 代码中使用**正交初始化**（`orthogonal_`），比默认初始化更有利于梯度流。

### 关键代码对应

**目标值计算**（软 Bellman + 双 Q 取最小）：

$$
a' \sim \pi_\theta(\cdot|s'), \quad y = r + (1-d) \cdot 0.99 \cdot \left(\min(Q_{\phi'_1}(s', a'), Q_{\phi'_2}(s', a')) - \alpha \log\pi_\theta(a'|s')\right)
$$

**Critic 损失**：

$$
L_{\text{critic}} = \text{MSE}(Q_{\phi_1}(s,a),\; y) + \text{MSE}(Q_{\phi_2}(s,a),\; y)
$$

**Actor 损失**（重参数化梯度）：

$$
L_{\text{actor}} = \frac{1}{|B|}\sum\left[\alpha \log\pi_\theta(\tilde{a}|s) - \min(Q_{\phi_1}(s, \tilde{a}), Q_{\phi_2}(s, \tilde{a}))\right]
$$

**温度参数损失**：

$$
L(\alpha) = -\log\alpha \cdot \frac{1}{|B|}\sum\left[\log\pi_\theta(\tilde{a}|s) + \mathcal{H}_{\text{target}}\right]
$$

**对数概率修正**（tanh 压缩）：

$$
\log\pi(a|s) = \log\mathcal{N}(u;\mu,\sigma^2) - \sum_i \log\left(a_{\max}(1 - \tanh^2(u_i)) + 10^{-6}\right)
$$

**软更新**（每步执行）：

$$
\phi'_i \leftarrow 0.005 \cdot \phi_i + 0.995 \cdot \phi'_i
$$

---

## SAC vs TD3 vs PPO 对比

| 特性 | SAC | [[ReinforcementLearning/DRL/TD3/TD3\|TD3]] | [[ReinforcementLearning/DRL/PPO/PPO\|PPO]] |
|---|---|---|---|
| 策略类型 | **随机**（最大熵） | 确定性 | 随机 |
| 数据利用 | **Off-policy** | Off-policy | On-policy |
| Q 网络数量 | 2（取最小值） | 2（取最小值） | 无（用价值网络） |
| 探索机制 | **策略熵**（内在） | 外部噪声 | 策略随机性 |
| 温度/正则化 | **自动 $\alpha$ 调节** | 无 | 裁剪 $\epsilon$ |
| 数据效率 | **高** | **高** | 低 |
| 训练稳定性 | **好** | 好 | **好** |
| 超参数敏感度 | **低** | 中 | 中 |
| 动作空间 | 连续 | 连续 | 连续 + 离散 |
| 适用场景 | 通用连续控制 | 不需太多探索的任务 | 通用（含离散） |

> [!tip] 如何选择？
> - **连续控制首选 SAC**：自动探索、超参数鲁棒、数据高效
> - **需要确定性策略时选 TD3**：如机器人控制中不希望有随机性
> - **离散动作或需要 on-policy 时选 PPO**：如游戏、NLP（RLHF）

---

## SAC 的实践技巧

### 自动温度 vs 固定温度

| 方式 | 优点 | 缺点 |
|---|---|---|
| **自动调节 $\alpha$**（推荐） | 无需手动调参，适应性强 | 额外的优化变量和学习率 |
| 固定 $\alpha$ | 简单，计算开销略低 | 需要仔细调参，不同阶段需求不同 |

### 对数标准差裁剪

```python
log_std = torch.clamp(log_std, -20, 2)
```

- $\log\sigma = -20$ → $\sigma \approx 2 \times 10^{-9}$（接近确定性）
- $\log\sigma = 2$ → $\sigma \approx 7.4$（非常随机）

不裁剪可能导致：标准差爆炸（数值溢出）或坍缩为零（探索停止）。

### 确定性评估

训练时使用随机策略探索，测试时使用**均值动作**（确定性评估）：

$$
a_{\text{test}} = a_{\max} \cdot \tanh(\mu_\theta(s))
$$

代码中通过 `deterministic=True` 参数控制。

### 常见 trick 汇总

| 技巧 | 说明 |
|---|---|
| **自动温度调节** | 使用可学习的 $\alpha$，免去手动调参 |
| **正交初始化** | 用正交矩阵初始化网络权重，改善梯度流 |
| **对数标准差裁剪** | 限制 $\log\sigma$ 范围，防止数值不稳定 |
| **经验池预填充** | 用随机策略收集初始数据 |
| **奖励缩放/归一化** | 对奖励进行归一化，稳定 Q 值数量级 |
| **观测归一化** | 对输入状态使用 running mean/std 归一化 |
| **优化 $\log\alpha$ 而非 $\alpha$** | 保证 $\alpha = e^{\log\alpha} > 0$，优化更稳定 |
| **tanh log_prob 修正** | 必须用变量替换公式修正概率密度 |

---

## 从能量模型视角理解 SAC

> [!abstract] 能量基模型（EBM）视角
> SAC 的最优策略 $\pi^*(a|s) \propto \exp(Q^*(s,a)/\alpha)$ 本质上是一个**玻尔兹曼分布**：
>
> - Q 函数扮演"负能量"的角色：$E(s,a) = -Q(s,a)$
> - 温度 $\alpha$ 控制分布的"锐度"
> - 高 Q 值（低能量）的动作被赋予更高概率
>
> 这与统计力学中的配分函数有深刻的数学联系，也解释了为什么 SAC 能自然地学到**多模态**的策略——能量景观中的多个低能量区域对应多种可行的解决方案。
>
> | 温度 $\alpha$ | 类比 | 策略行为 |
> |---|---|---|
> | $\alpha \to 0$ | 绝对零度 | 确定性策略，只选最优动作 |
> | $\alpha$ 适中 | 常温 | 按 Q 值加权采样，有探索 |
> | $\alpha \to \infty$ | 高温 | 接近均匀随机，纯探索 |

---
