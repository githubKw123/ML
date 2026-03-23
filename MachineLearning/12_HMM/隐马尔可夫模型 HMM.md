---
title: 隐马尔可夫模型 (HMM)
tags:
  - 机器学习
  - 概率图模型
  - 动态模型
  - HMM
aliases:
  - HMM
  - Hidden Markov Model
---

# 隐马尔可夫模型 (HMM)

## 概述

隐马尔可夫模型是一种==概率图模型==，属于**动态模型**。

> [!info] 机器学习的两大流派
> - **频率派**：核心是==优化问题==
> - **贝叶斯派**：核心是==积分问题==（变分推断、MCMC 等）

概率图模型的基本分类：

| 类型 | 名称 | 示例 |
|------|------|------|
| 有向图 | 贝叶斯网络 | GMM |
| 无向图 | 马尔可夫随机场 | — |

当样本之间存在**时序关联**（不独立同分布）时，模型演变为**动态模型**——隐变量随时间变化，观测变量也随之变化。
![[MachineLearning/12_HMM/1.png]]

### 动态模型分类

根据状态变量的特点，动态模型可分为：

1. **HMM**：状态变量（隐变量）是==离散==的
2. **Kalman 滤波**：状态变量是==连续、线性==的
3. **粒子滤波**：状态变量是==连续、非线性==的

---

## 符号定义

> [!note] 模型参数
> 用参数 $\lambda = (\pi, A, B)$ 表示 HMM，其中：
> - $\pi$：初始状态概率分布
> - $A$：状态转移矩阵
> - $B$：发射矩阵（观测概率矩阵）

| 符号 | 含义 |
|------|------|
| $o_t$ | 观测变量 |
| $O$ | 观测序列 |
| $V = \{v_1, v_2, \cdots, v_M\}$ | 观测值域 |
| $i_t$ | 状态变量 |
| $I$ | 状态序列 |
| $Q = \{q_1, q_2, \cdots, q_N\}$ | 状态值域 |
| $A = (a_{ij} = p(i_{t+1}=q_j \mid i_t=q_i))$ | 状态转移矩阵 |
| $B = (b_j(k) = p(o_t=v_k \mid i_t=q_j))$ | 发射矩阵 |

---

## 两个基本假设

> [!important] HMM 的核心假设
>
> **1. 齐次 Markov 假设**（未来只依赖于当前）：
>
> $$p(i_{t+1}|i_t, i_{t-1}, \cdots, i_1, o_t, o_{t-1}, \cdots, o_1) = p(i_{t+1}|i_t)$$
>
> **2. 观测独立假设**：
>
> $$p(o_t|i_t, i_{t-1}, \cdots, i_1, o_{t-1}, \cdots, o_1) = p(o_t|i_t)$$

---

## 三大核心问题

| 问题 | 数学表述 | 算法 |
|------|----------|------|
| **Evaluation（评估）** | $p(O \mid \lambda)$ | Forward-Backward 算法 |
| **Learning（学习）** | $\lambda = \arg\max_\lambda p(O \mid \lambda)$ | EM 算法（Baum-Welch） |
| **Decoding（解码）** | $I = \arg\max_I p(I \mid O, \lambda)$ | Viterbi 算法 |

此外还有两个相关问题：
- **预测问题**：$p(i_{t+1} \mid o_1, o_2, \cdots, o_t)$
- **滤波问题**：$p(i_t \mid o_1, o_2, \cdots, o_t)$

---

## Evaluation（评估问题）

### 基本推导

$$p(O|\lambda) = \sum_I p(I, O|\lambda) = \sum_I p(O|I, \lambda) p(I|\lambda) \tag{3}$$

由齐次 Markov 假设：

$$p(I|\lambda) = \pi_1 \prod_{t=2}^{T} a_{i_{t-1}, i_t} \tag{6}$$

由观测独立假设：

$$p(O|I, \lambda) = \prod_{t=1}^{T} b_{i_t}(o_t) \tag{7}$$

合并得：

$$p(O|\lambda) = \sum_I \pi_{i_1} \prod_{t=2}^{T} a_{i_{t-1}, i_t} \prod_{t=1}^{T} b_{i_t}(o_t) \tag{8}$$

> [!warning] 复杂度问题
> 直接求和的复杂度为 $O(N^T)$，需要使用前向/后向算法优化。

### 前向算法 (Forward Algorithm)

定义前向变量：

$$\alpha_t(i) = p(o_1, o_2, \cdots, o_t, i_t = q_i | \lambda)$$

则：

$$p(O|\lambda) = \sum_{i=1}^{N} \alpha_T(i) \tag{9}$$

**递推公式**：

$$\alpha_{t+1}(j) = \sum_{i=1}^{N} b_j(o_t) a_{ij} \alpha_t(i) \tag{11}$$

### 后向算法 (Backward Algorithm)

定义后向变量：

$$\beta_t(i) = p(o_{t+1}, o_{t+1}, \cdots, o_T | i_t = i, \lambda)$$

用后向变量表示 $p(O|\lambda)$：

$$p(O|\lambda) = \sum_{i=1}^{N} b_i(o_1) \pi_i \beta_1(i) \tag{12}$$

**递推公式**：

$$\beta_t(i) = \sum_{j=1}^{N} b_j(o_{t+1}) a_{ij} \beta_{t+1}(j) \tag{13}$$

---

## Learning（学习问题）

### EM 算法（Baum-Welch 算法）

目标：最大似然估计

$$\lambda_{MLE} = \arg\max_\lambda p(O|\lambda) \tag{14}$$

采用 EM 算法迭代求解：

$$\theta^{t+1} = \arg\max_\theta \int_z \log p(X, Z|\theta) p(Z|X, \theta^t) dz \tag{15}$$

其中 $X$ 是观测变量，$Z$ 是隐变量序列。

#### 展开 Q 函数

$$\sum_I \log p(O, I|\lambda) p(O, I|\lambda^t) = \sum_I [\log \pi_{i_1} + \sum_{t=2}^{T} \log a_{i_{t-1}, i_t} + \sum_{t=1}^{T} \log b_{i_t}(o_t)] p(O, I|\lambda^t) \tag{17}$$

#### 求解 $\pi$ 的更新公式

通过 Lagrange 乘子法（约束 $\sum_i \pi_i = 1$），定义：

$$L(\pi, \eta) = \sum_{i=1}^{N} \log \pi_i \cdot p(O, i_1 = q_i | \lambda^t) + \eta(\sum_{i=1}^{N} \pi_i - 1) \tag{20}$$

求导并令其为零：

$$\frac{\partial L}{\partial \pi_i} = \frac{1}{\pi_i} p(O, i_1 = q_i | \lambda^t) + \eta = 0 \tag{21}$$

最终得到更新公式：

$$\pi_i^{t+1} = \frac{p(O, i_1 = q_i | \lambda^t)}{p(O | \lambda^t)} \tag{23}$$

---

## Decoding（解码问题）

### Viterbi 算法

目标：

$$I = \arg\max_I p(I|O, \lambda) \tag{24}$$

> [!tip] 核心思想
> 寻找概率最大的状态序列，本质是参数空间中的==最优路径==问题，采用**动态规划**求解。

定义：

$$\delta_t(j) = \max_{i_1, \cdots, i_{t-1}} p(o_1, \cdots, o_t, i_1, \cdots, i_{t-1}, i_t = q_j) \tag{25}$$

**递推公式**：

$$\delta_{t+1}(j) = \max_{1 \le i \le N} \delta_t(i) a_{ij} b_j(o_{t+1}) \tag{26}$$

**路径回溯**：

$$\psi_{t+1}(j) = \arg\max_{1 \le i \le N} \delta_t(i) a_{ij} \tag{27}$$

---

## 小结

> [!abstract] 总结
> HMM 是一种动态模型，是由混合树形模型和时序结合起来的模型（类似 **GMM + Time**）。

![[MachineLearning/12_HMM/2.png]]
### 推断任务总览

| 任务 | 表达式 | 类型 |
|------|--------|------|
| **译码 Decoding** | $p(z_1, z_2, \cdots, z_t \mid x_1, x_2, \cdots, x_t)$ | — |
| **似然概率** | $p(X \mid \theta)$ | — |
| **滤波 Filtering** | $p(z_t \mid x_1, \cdots, x_t)$ | Online |
| **平滑 Smoothing** | $p(z_t \mid x_1, \cdots, x_T)$ | Offline |
| **预测 Prediction** | $p(z_{t+1}, z_{t+2} \mid x_1, \cdots, x_t)$ | — |

### 滤波

$$p(z_t | x_{1:t}) = \frac{p(x_{1:t}, z_t)}{p(x_{1:t})} = C\alpha_t(z_t) \tag{28}$$

### 平滑（前向后向算法）

$$p(z_t | x_{1:T}) = \frac{\alpha_t(z_t) p(x_{t+1:T} | z_t)}{p(x_{1:T})} = C\alpha_t(z_t)\beta_t(z_t) \tag{30}$$

### 预测

$$p(z_{t+1}|x_{1:t}) = \sum_{z_t} p(z_{t+1}|z_t) p(z_t|x_{1:t}) \tag{31}$$

$$p(x_{t+1}|x_{1:t}) = \sum_{z_{t+1}} p(x_{t+1}|z_{t+1}) p(z_{t+1}|x_{1:t}) \tag{32}$$
