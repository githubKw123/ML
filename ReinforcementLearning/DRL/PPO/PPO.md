---
title: PPO（近端策略优化）
tags:
  - reinforcement-learning
  - policy-gradient
  - actor-critic
  - PPO
  - deep-RL
  - on-policy
aliases:
  - Proximal Policy Optimization
  - 近端策略优化
---

# PPO（Proximal Policy Optimization）

PPO（Proximal Policy Optimization，近端策略优化）是 OpenAI 于 2017 年提出的策略梯度算法，核心思想是在保证策略单调改进的前提下，用**简单的裁剪机制**替代 TRPO 复杂的约束优化，从而实现高效、稳定的策略更新。前置知识见 [[PolicyBased]]（策略梯度定理、Actor-Critic、GAE）。

> [!info] 为什么需要 PPO？
> [[PolicyBased#Actor-Critic 方法|Actor-Critic]] 的策略梯度更新存在一个根本矛盾：
>
> - **步长太大**：策略突变，导致采样到的数据与新策略不匹配，性能崩溃（catastrophic collapse）
> - **步长太小**：收敛极慢，数据效率低
>
> TRPO 通过 KL 散度约束解决了这个问题，但其二阶优化（共轭梯度 + 线搜索）实现复杂、计算开销大。PPO 用**一阶方法**达到了与 TRPO 相当甚至更好的效果，是目前最主流的策略梯度算法。

---

## 理论基础

### 策略改进的单调性保证

为了理解 PPO 的设计动机，首先需要回顾策略改进的理论基础。

对于两个策略 $\pi$ 和 $\pi'$，新策略 $\pi'$ 的性能可以用旧策略 $\pi$ 的价值函数来表示：

$$
J(\pi') = J(\pi) + E_{s \sim d^{\pi'}, a \sim \pi'}\left[A_{\pi}(s,a)\right]
$$

其中 $d^{\pi'}$ 是新策略下的状态分布，$A_{\pi}(s,a) = Q_{\pi}(s,a) - V_{\pi}(s)$ 是旧策略下的[[PolicyBased#优势函数（Advantage Function）|优势函数]]。

> [!warning] 循环依赖问题
> 上式中新策略的性能依赖于 $d^{\pi'}$——即新策略下的状态分布。但我们还没有部署新策略，无法获得 $d^{\pi'}$ 的样本。这就构成了一个循环依赖：要评估新策略需要新策略的数据，但要收集新策略的数据需要先确定新策略。

### 替代目标函数

为了打破循环依赖，用**旧策略的状态分布** $d^{\pi}$ 近似 $d^{\pi'}$，得到替代目标（Surrogate Objective）：

$$
L^{CPI}(\theta) = E_{s \sim d^{\pi_{\theta_{\text{old}}}}, a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A_{\pi_{\theta_{\text{old}}}}(s,a)\right]
$$

记**重要性采样比率**为：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

则替代目标可以简写为：

$$
L^{CPI}(\theta) = E_t\left[r_t(\theta) \hat{A}_t\right]
$$

> [!note] 重要性采样比率的直觉
> - $r_t(\theta) = 1$：新旧策略对该动作的概率相同
> - $r_t(\theta) > 1$：新策略比旧策略更倾向选择该动作
> - $r_t(\theta) < 1$：新策略比旧策略更不倾向选择该动作
>
> 如果不加约束地最大化 $L^{CPI}$，可能导致 $r_t(\theta)$ 偏离 1 太远，使得近似失效、策略更新过大。

### 信任域方法（TRPO）

TRPO 的解决方案是加入**KL 散度约束**，将优化问题变为：

$$
\max_\theta \; E_t\left[r_t(\theta) \hat{A}_t\right] \quad \text{s.t.} \quad E_t\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta)\right] \leq \delta
$$

这保证了新旧策略足够接近，从而替代目标的近似是可靠的。但 TRPO 需要：

1. 计算 Fisher 信息矩阵（或其近似）
2. 共轭梯度法求解约束优化
3. 线搜索确保满足约束

这使得 TRPO **实现复杂、不兼容包含 Dropout 或参数共享的网络架构**。PPO 正是为了解决这些实际问题而诞生的。

---

## PPO-Clip（裁剪版本）

PPO 的核心版本，用**裁剪（Clipping）** 直接限制策略更新幅度，无需显式计算 KL 散度。

### 裁剪目标函数

$$
L^{CLIP}(\theta) = E_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $\epsilon$ 是裁剪超参数，通常取 $\epsilon = 0.2$。

> [!important] 裁剪机制的工作原理
> `min` 操作的两个项：
> - **第一项** $r_t(\theta) \hat{A}_t$：标准的替代目标
> - **第二项** $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$：将比率裁剪到 $[1-\epsilon, 1+\epsilon]$ 范围内
>
> 取两者的最小值，确保最终目标是替代目标的**下界**。

### 分情况分析

根据优势函数 $\hat{A}_t$ 的正负，裁剪的行为不同：

**情况 1：$\hat{A}_t > 0$**（动作好于平均）

此时策略梯度会增大 $r_t(\theta)$（增加该动作的概率），但裁剪限制其不超过 $1+\epsilon$：

$$
L^{CLIP} = \min\left(r_t(\theta), 1+\epsilon\right) \hat{A}_t
$$

| $r_t(\theta)$ 的范围 | 使用的项 | 效果 |
|---|---|---|
| $r_t \leq 1+\epsilon$ | $r_t \hat{A}_t$（未裁剪） | 正常更新，增加好动作概率 |
| $r_t > 1+\epsilon$ | $(1+\epsilon) \hat{A}_t$（裁剪） | 梯度为零，阻止概率进一步增大 |

**情况 2：$\hat{A}_t < 0$**（动作差于平均）

此时策略梯度会减小 $r_t(\theta)$（降低该动作的概率），但裁剪限制其不低于 $1-\epsilon$：

$$
L^{CLIP} = \max\left(r_t(\theta), 1-\epsilon\right) \hat{A}_t
$$

| $r_t(\theta)$ 的范围 | 使用的项 | 效果 |
|---|---|---|
| $r_t \geq 1-\epsilon$ | $r_t \hat{A}_t$（未裁剪） | 正常更新，降低坏动作概率 |
| $r_t < 1-\epsilon$ | $(1-\epsilon) \hat{A}_t$（裁剪） | 梯度为零，阻止概率进一步减小 |

> [!tip] 裁剪的核心思想
> 无论优势是正还是负，裁剪都在做同一件事：**当策略变化超出信任域 $[1-\epsilon, 1+\epsilon]$ 时，停止在当前方向上继续更新**。这相当于用一种"懒惰"但有效的方式替代了 TRPO 的硬约束。

---

## PPO-Penalty（惩罚版本）

PPO 的另一种变体，将 KL 散度约束转化为目标函数中的**自适应惩罚项**：

$$
L^{KPEN}(\theta) = E_t\left[r_t(\theta) \hat{A}_t - \beta \, D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta)\right]
$$

其中 $\beta$ 是惩罚系数，根据实际 KL 散度与目标 $d_{\text{targ}}$ 的偏离自适应调整：

$$
\beta \leftarrow
\begin{cases}
\beta / 2 & \text{if } D_{\text{KL}} < d_{\text{targ}} / 1.5 \quad \text{（KL 太小，放松约束）} \\
\beta \times 2 & \text{if } D_{\text{KL}} > d_{\text{targ}} \times 1.5 \quad \text{（KL 太大，收紧约束）} \\
\beta & \text{otherwise}
\end{cases}
$$

> [!note] Clip vs Penalty
> 实践中 **PPO-Clip 更为常用**，因为它不需要额外的超参数 $\beta$ 和 $d_{\text{targ}}$，实现更简单，效果也通常更好。OpenAI 原论文的实验也表明 PPO-Clip 在多数任务上优于 PPO-Penalty。

---

## 完整的 PPO 目标函数

实际实现中，PPO 的总损失函数通常由三部分组成：

$$
L(\theta) = E_t\left[L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 H[\pi_\theta](s_t)\right]
$$

| 项 | 含义 | 作用 |
|---|---|---|
| $L^{CLIP}_t(\theta)$ | 裁剪的策略目标 | 优化策略（最大化） |
| $L^{VF}_t(\theta) = (V_\theta(s_t) - V_t^{\text{targ}})^2$ | 价值函数损失 | 训练 Critic 准确估计价值 |
| $H[\pi_\theta](s_t)$ | 策略熵 | 鼓励探索，防止过早收敛 |
| $c_1, c_2$ | 损失系数 | 平衡三项的权重（通常 $c_1=0.5$, $c_2=0.01$） |

> [!note] 参数共享
> 当 Actor 和 Critic 共享网络参数时（如共享底层特征提取层），三项损失可以合并为一个目标函数一起训练。如果 Actor 和 Critic 是独立网络，则分别更新策略损失和价值损失。

---

## 广义优势估计（GAE）

PPO 通常使用 [[PolicyBased#GAE（Generalized Advantage Estimation）|GAE]] 来计算优势函数，平衡偏差与方差。

### 计算方式

定义单步 TD 误差：

$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

GAE 优势估计为：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}
$$

在实际实现中通常从后往前递推计算：

$$
\hat{A}_T = 0, \quad \hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}
$$

> [!tip] GAE 在 PPO 中的典型参数
> - $\gamma = 0.99$（折扣因子）
> - $\lambda = 0.95$（GAE 参数，平衡偏差与方差）
>
> $\lambda$ 越大越接近 MC（低偏差高方差），越小越接近 TD(0)（高偏差低方差）。

---

## PPO 算法流程

### 训练流程

```
┌─────────────────────────────────────────────────────────┐
│  1. 用当前策略 π_θ_old 在环境中收集 T 步经验              │
│     {s_t, a_t, r_t, s_{t+1}, log π(a_t|s_t)}           │
│                                                         │
│  2. 计算 GAE 优势估计 Â_t 和回报目标 V_t^targ            │
│                                                         │
│  3. 对收集的数据进行 K 轮(epoch) 小批量更新：               │
│     ┌─────────────────────────────────────────────┐     │
│     │  a. 从 buffer 中随机采样 mini-batch           │     │
│     │  b. 计算 r_t(θ) = π_θ(a_t|s_t)/π_θ_old     │     │
│     │  c. 计算裁剪目标 L^CLIP                      │     │
│     │  d. 计算价值损失 L^VF                         │     │
│     │  e. 计算熵奖励 H                              │     │
│     │  f. 梯度上升更新 θ                            │     │
│     └─────────────────────────────────────────────┘     │
│                                                         │
│  4. θ_old ← θ，回到步骤 1                               │
└─────────────────────────────────────────────────────────┘
```

> [!example]- 伪代码：PPO-Clip
> **输入**：初始策略参数 $\theta_0$，裁剪参数 $\epsilon$，GAE 参数 $\gamma, \lambda$
> 1. **For** $k = 0, 1, 2, \ldots$：
>    1. **收集数据**：用策略 $\pi_{\theta_k}$ 运行 $T$ 步，收集轨迹 $\{s_t, a_t, r_t, \log\pi_{\theta_k}(a_t|s_t)\}$
>    2. **计算优势**：用 GAE 计算 $\hat{A}_t$，并标准化：$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + \epsilon}$
>    3. **计算回报目标**：$V_t^{\text{targ}} = \hat{A}_t + V(s_t)$
>    4. **For** epoch $= 1, \ldots, K$：
>       - 将数据随机打乱，分为若干 mini-batch
>       - **For** 每个 mini-batch：
>         - $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}$
>         - $L^{CLIP} = \frac{1}{|B|}\sum_{t \in B}\min\left(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)$
>         - $L^{VF} = \frac{1}{|B|}\sum_{t \in B}\left(V_\theta(s_t) - V_t^{\text{targ}}\right)^2$
>         - $\theta \leftarrow \theta + \alpha \nabla_\theta \left(L^{CLIP} - c_1 L^{VF} + c_2 H\right)$

### 关键超参数

| 超参数 | 含义 | 常用值 | 备注 |
|---|---|---|---|
| $\epsilon$ | 裁剪范围 | $0.1 \sim 0.3$ | 最常用 $0.2$ |
| $\gamma$ | 折扣因子 | $0.99$ | 环境相关 |
| $\lambda$ | GAE 参数 | $0.95$ | 偏差-方差权衡 |
| $K$ | 更新轮数（epochs） | $3 \sim 30$ | 过大可能过拟合旧数据 |
| $T$ | 每次收集步数 | $128 \sim 2048$ | 取决于环境复杂度 |
| 学习率 | 参数更新步长 | $3\times10^{-4}$ | 可配合线性衰减 |
| mini-batch size | 小批量大小 | $32 \sim 512$ | 需整除 $T$ |
| $c_1$ (value coef) | 价值损失系数 | $0.5$ | — |
| $c_2$ (entropy coef) | 熵正则系数 | $0.01$ | 连续控制可更小 |
| max grad norm | 梯度裁剪范数 | $0.5$ | 防止梯度爆炸 |

---

## 代码实现解析

本目录下的 `PPO.py` 实现了 PPO-Clip 算法在 **Pendulum-v1** 环境上的训练。以下是核心实现要点与算法公式的对应关系。

### 网络架构

```
Actor (策略网络)                    Critic (价值网络)
┌──────────────────┐               ┌──────────────────┐
│ Input: state(3)  │               │ Input: state(3)  │
│       ↓          │               │       ↓          │
│ Linear(3→64)     │               │ Linear(3→64)     │
│ ReLU             │               │ ReLU             │
│       ↓          │               │       ↓          │
│ Linear(64→64)    │               │ Linear(64→64)    │
│ ReLU             │               │ ReLU             │
│       ↓          │               │       ↓          │
│ Linear(64→1)     │               │ Linear(64→1)     │
│ 2·tanh → [-2,2]  │               │ → V(s)           │
│ → μ(s)           │               └──────────────────┘
│ σ = 0.5 (固定)    │
│ π = N(μ, σ²)     │
└──────────────────┘
```

> [!note] 固定标准差的设计选择
> 代码中使用**固定标准差** $\sigma = 0.5$，而非让网络输出 $\sigma$。这是一种简化策略，避免 $\sigma$ 过早收缩到 0 导致探索不足。更高级的实现会让 $\sigma$ 也作为可学习参数。

### 关键代码对应

**GAE 计算**（对应 `compute_gae` 方法）：

$$
\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)
$$

$$
\hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1}
$$

**PPO-Clip 更新**（对应 `update` 方法）：

$$
r_t(\theta) = \exp\left(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t)\right)
$$

$$
L^{CLIP} = -\frac{1}{|B|}\sum_{t}\min\left(r_t \hat{A}_t,\; \text{clip}(r_t, 0.8, 1.2)\hat{A}_t\right) - 0.001 \cdot H
$$


---

## PPO 的实践技巧

### 优势标准化

将 GAE 优势估计进行标准化（零均值、单位方差），能显著提升训练稳定性：

$$
\hat{A}_t \leftarrow \frac{\hat{A}_t - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + \epsilon}
$$

> [!tip] 为什么有效？
> 标准化后，大约一半的优势为正（鼓励的动作），一半为负（抑制的动作），使得梯度更新方向更加平衡。未标准化时，优势可能全为正或量级差异大，导致训练不稳定。

### 学习率调度

常用线性衰减学习率，从初始值线性降低到 0：

$$
\text{lr}(t) = \text{lr}_0 \cdot \left(1 - \frac{t}{T_{\text{total}}}\right)
$$

### 常见 trick 汇总

| 技巧 | 说明 |
|---|---|
| **优势标准化** | 对每个 mini-batch 的优势进行标准化 |
| **正交初始化** | 用正交矩阵初始化网络权重，改善梯度流 |
| **价值函数裁剪** | 对 Critic 的更新也进行裁剪，防止价值函数剧变 |
| **奖励归一化** | 对奖励进行 running mean/std 归一化 |
| **观测归一化** | 对输入状态进行归一化，加速训练 |
| **梯度裁剪** | 限制梯度范数，防止梯度爆炸 |
| **学习率衰减** | 线性或余弦衰减学习率 |

---


