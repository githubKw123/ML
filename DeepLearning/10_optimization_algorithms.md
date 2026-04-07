---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: ml
    language: python
    name: python3
---

# 优化算法 (Optimization Algorithms)

> 参考：[动手学深度学习 v2 第11章](https://zh-v2.d2l.ai/chapter_optimization/index.html)



优化算法是实现深度学习模型训练的重要工具，理论部分的工作参考[最优化](../Optimization/introduction(Optimization).md)，这里主要解释现代深度学习喜欢使用的优化算法是怎么样的。

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


print(f"PyTorch 版本：{torch.__version__}")

```

<!-- #region -->
### 1. 优化与深度学习


| 概念 | 说明 |
|---|---|
| 风险（Risk） | 期望损失，目标是最小化它 |
| 经验风险（Empirical Risk） | 训练集上的平均损失，实际可计算 |
| 优化目标 | 最小化经验风险，以期减小风险 |

> 两者差异 = **泛化误差**。优化只管训练集，不保证泛化。

**三大挑战**

**① 局部极小值（Local Minima）**：非凸损失面存在大量局部最优解。

**② 鞍点（Saddle Points）**：梯度为零但非极值点。在高维空间中，鞍点比局部极小值更普遍。
- 海森矩阵特征值全正 → 局部极小值
- 海森矩阵特征值全负 → 局部极大值
- **混合** → 鞍点（高维中概率极大）

**③ 梯度消失（Vanishing Gradients）**：激活函数饱和区梯度趋近 0，优化停滞。

<!-- #endregion -->

### 2. 梯度下降

这是优化算法中最基础方法，详细见[线搜索](../Optimization/LinearSearch.md)。从泰勒展开出发：$f(\mathbf{x} + \boldsymbol{\epsilon}) \approx f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x})$，令 $\boldsymbol{\epsilon} = -\eta \nabla f(\mathbf{x})$ 可保证函数值下降：

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$

其中 $\eta > 0$ 为**学习率**。


<!-- #region -->
### 3. 随机梯度下降（SGD）


目标函数为均值：$f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{x})$

一般的梯度下降为**全批量梯度下降**：$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$，计算代价 $O(n)$

而**SGD**：随机抽取一个样本 $i$：
$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x})$$

这样引入了一定的随机性不让模型向着局部最低点一路狂奔

无偏性：$\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$

**学习率衰减策略**

随着迭代推进，噪声梯度的**方差**需要被压制：

| 策略 | 公式 |
|---|---|
| 分段常数 | $\eta(t) = \eta_i$（分段） |
| 指数衰减 | $\eta(t) = \eta_0 e^{-\lambda t}$ |
| 多项式衰减 | $\eta(t) = \eta_0 (\beta t+1)^{-\alpha}$ |

<!-- #endregion -->

**小批量随机梯度下降**

简单说就是大数据下SGD的升级，也是DL中常用的优化算法。从批量 $\mathcal{B}_t$（大小 $b$）计算梯度：

$$\mathbf{g}_t = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x})$$

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{g}_t$$

**方差缩减**：相比单样本 SGD，方差降低 $b^{-1/2}$ 倍。

> 批量越大 → 梯度方差越小 → 可使用更大学习率（线性缩放规则：$\eta \propto b$）。


```python
# ── PyTorch Mini-batch SGD 标准用法 ──
torch.manual_seed(42)
n, d = 1000, 5
X_data = torch.randn(n, d)
true_w = torch.randn(d)
y_data = X_data @ true_w + 0.1 * torch.randn(n)

dataset = torch.utils.data.TensorDataset(X_data, y_data)

net = nn.Sequential(nn.Linear(d, 1))
loss_fn = nn.MSELoss()

results = {}
for batch_size in [1, 32, 256, n]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Sequential(nn.Linear(d, 1))
    # 学习率按批量大小线性缩放
    lr = 0.01 * (batch_size / 32)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(15):
        epoch_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            l = loss_fn(model(Xb).squeeze(), yb)
            l.backward()
            optimizer.step()
            epoch_loss += l.item()
        losses.append(epoch_loss / len(loader))
    results[f'bs={batch_size}, lr={lr:.4f}'] = losses

plt.figure(figsize=(8, 4))
for label, loss_curve in results.items():
    plt.plot(loss_curve, label=label)
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('Mini-batch SGD Convergence'); plt.legend(); plt.tight_layout(); plt.show()

```

<!-- #region -->
### 4. 动量法（Momentum）


用**梯度的指数加权平均**（泄漏平均）替代瞬时梯度，抑制振荡方向、加速收敛方向：

$$\mathbf{v}_t \leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t}$$

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t$$

递归展开：$\mathbf{v}_t = \sum_{\tau=0}^{t-1} \beta^\tau \mathbf{g}_{t-\tau}$（指数加权历史梯度）

**有效学习率**：等比数列求和 $= \frac{\eta}{1-\beta}$；有效窗口约 $\frac{1}{1-\beta}$ 步。

**PyTorch API**

```python
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)
```

<!-- #endregion -->

```python
# ── 动量法 vs 梯度下降（病态曲率）──
def f2d(x1, x2):    return 0.1*x1**2 + 2*x2**2
def grad_f2d(x1, x2): return np.array([0.2*x1, 4*x2])

def momentum_2d(eta, beta, n_steps=30, x0=(5.0, -2.0)):
    x = np.array(x0, dtype=float)
    v = np.zeros(2)
    hist = [x.copy()]
    for _ in range(n_steps):
        g = grad_f2d(*x)
        v = beta * v + g
        x -= eta * v
        hist.append(x.copy())
    return np.array(hist)

x1_r = np.linspace(-6, 6, 200)
x2_r = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1_r, x2_r)
Z = f2d(X1, X2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs = [
    ('GD η=0.4',        lambda: momentum_2d(0.4, 0.0)),
    ('Momentum η=0.4 β=0.9', lambda: momentum_2d(0.4, 0.9)),
    ('Momentum η=0.6 β=0.5', lambda: momentum_2d(0.6, 0.5)),
]
for ax, (title, fn) in zip(axes, configs):
    hist = fn()
    ax.contour(X1, X2, Z, levels=25, cmap='Blues', alpha=0.6)
    ax.plot(hist[:, 0], hist[:, 1], 'r-o', markersize=4)
    ax.scatter(*hist[0],  c='green', s=100, zorder=5)
    ax.scatter(0, 0, c='gold', s=150, marker='*', zorder=5)
    ax.set_title(title); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
plt.suptitle('Momentum vs Gradient Descent (Stiff Cur)',  fontsize=13)
plt.tight_layout(); plt.show()

```

<!-- #region -->
### 5. AdaGrad

对**每个参数维度**维护历史梯度平方累计，为不频繁更新的参数提供更大的有效学习率（适合稀疏特征）：

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + \mathbf{g}_t^2 \quad \text{（逐元素平方累加）}$$

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

- $\mathbf{s}_0 = \mathbf{0}$，$\epsilon \approx 10^{-7}$（防零除）
- 有效学习率近似 $O(t^{-1/2})$，随时间单调递减

| 优点 | 缺点 |
|---|---|
| 自适应每维度学习率 | 学习率单调递减，可能过早停止 |
| 适合 NLP 稀疏特征 | 长期训练后学习率趋于零 |
| 无需手动调 LR 衰减 | 不适合非平稳（non-stationary）目标 |

**PyTorch API**

```python
optimizer = torch.optim.Adagrad(params, lr=0.1)
```

<!-- #endregion -->

<!-- #region -->
### 6. RMSProp


修复 AdaGrad 学习率单调递减的问题：用**指数加权移动平均（EMA）**替代无界累加：

$$\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$$

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

- $\gamma = 0.9$（默认），有效窗口 $\approx \frac{1}{1-\gamma} = 10$ 步
- $\epsilon \approx 10^{-6}$

**与 AdaGrad 对比**：

| | AdaGrad | RMSProp |
|---|---|---|
| 状态更新 | $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$（无界） | $\mathbf{s}_t = \gamma\mathbf{s}_{t-1} + (1-\gamma)\mathbf{g}_t^2$（有界） |
| LR 趋势 | 单调递减 | 受控（由 $\gamma$ 调节） |
| 适合场景 | 凸问题 | 深度学习（非凸） |

**PyTorch API**

```python
# PyTorch 中 gamma 参数名为 alpha
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.9)
```

<!-- #endregion -->

<!-- #region -->
### 7. Adadelta

AdaGrad 的改进变体，**无需显式设置学习率**，通过参数变化量的历史校准步长。维护两个状态变量：

**状态1 — 梯度平方的泄漏均值：**
$$\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1-\rho) \mathbf{g}_t^2$$

**重缩放梯度（利用参数变化量历史）：**
$$\mathbf{g}_t' = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

**参数更新：**
$$\mathbf{x}_t = \mathbf{x}_{t-1} - \mathbf{g}_t'$$

**状态2 — 参数变化量平方的泄漏均值：**
$$\Delta\mathbf{x}_t = \rho \Delta\mathbf{x}_{t-1} + (1-\rho)(\mathbf{g}_t')^2$$

- $\rho = 0.9$（默认）
- **自校准**：分子用参数变化历史，分母用梯度历史，量纲匹配

**PyTorch API**

```python
optimizer = torch.optim.Adadelta(params, rho=0.9)
```

<!-- #endregion -->

<!-- #region -->
### 8. Adam


Adam = **动量**（一阶矩）+ **自适应学习率**（二阶矩，类似 RMSProp）+ **偏差修正**（修正零初始化偏差），更新步骤为：

**① 一阶矩（动量）：**
$$\mathbf{v}_t \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

**② 二阶矩（梯度平方 EMA）：**
$$\mathbf{s}_t \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

**③ 偏差修正：**
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}$$

**④ 参数更新：**
$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \eta \frac{\hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$$

**默认超参数：** $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-6}$

**PyTorch API**

```python
optimizer = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-6)
```

<!-- #endregion -->

<!-- #region -->
### 9. 学习率调度器


| 策略 | 公式 | 特点 |
|---|---|---|
| 平方根衰减 | $\eta(t) = \eta_0 (t+1)^{-0.5}$ | 理论保证 |
| 指数衰减 | $\eta(t) = \eta_0 \alpha^t$ | 简单，衰减快 |
| 分段常数（多步） | 在里程碑处乘以 $\gamma$ | 实践效果好 |
| 余弦退火 | $\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2}(1+\cos\frac{\pi t}{T})$ | CV 常用 |
| 线性预热 | 初始 $k$ 步从 0 线性增加到 $\eta_0$ | 防止深层网络早期发散 |

**PyTorch 调度器 API**

```python
# 分段衰减（最常用）
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[15, 30], gamma=0.5)

# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-5)

# 指数衰减
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95)

# 每 epoch 结束调用
scheduler.step()
```

<!-- #endregion -->

<!-- #region -->
### 10总结

| 优化器 | 状态变量 | 更新规则（核心） | 特点 |
|---|---|---|---|
| GD / SGD | 无 | $\mathbf{x} \leftarrow \mathbf{x} - \eta \mathbf{g}$ | 最简单，需调 LR |
| Momentum | $\mathbf{v}$ | $\mathbf{v}=\beta\mathbf{v}+\mathbf{g}$；$\mathbf{x}-=\eta\mathbf{v}$ | 加速收敛，减弱振荡 |
| AdaGrad | $\mathbf{s}$ | $\mathbf{s}+=\mathbf{g}^2$；$\mathbf{x}-=\frac{\eta}{\sqrt{\mathbf{s}+\epsilon}}\mathbf{g}$ | 自适应 LR，LR 单调降 |
| RMSProp | $\mathbf{s}$ | $\mathbf{s}=\gamma\mathbf{s}+(1-\gamma)\mathbf{g}^2$；$\mathbf{x}-=\frac{\eta}{\sqrt{\mathbf{s}+\epsilon}}\mathbf{g}$ | 修复 AdaGrad |
| Adadelta | $\mathbf{s}, \Delta\mathbf{x}$ | 见第9节 | 无 LR 参数 |
| **Adam** | $\mathbf{v}, \mathbf{s}$ | 一阶矩+二阶矩+偏差修正 | **最广泛使用** |


- **首选 Adam**：收敛快，对超参不敏感，lr=1e-3/1e-4 通常可用
- **SGD + Momentum + LR调度**：精调时可超越 Adam（尤其视觉任务）
- **AdaGrad/RMSProp**：NLP/稀疏特征场景
- **学习率调度必不可少**：即使用 Adam 也应配合 warmup 或余弦退火

<!-- #endregion -->

```python
# ── PyTorch 优化器 API 速查 ──
print("=" * 55)
print("PyTorch 优化器 API 速查")
print("=" * 55)
apis = [
    ("SGD",       "torch.optim.SGD(params, lr=0.01)"),
    ("Momentum",  "torch.optim.SGD(params, lr=0.005, momentum=0.9)"),
    ("Adagrad",   "torch.optim.Adagrad(params, lr=0.1)"),
    ("RMSprop",   "torch.optim.RMSprop(params, lr=0.01, alpha=0.9)"),
    ("Adadelta",  "torch.optim.Adadelta(params, rho=0.9)"),
    ("Adam",      "torch.optim.Adam(params, lr=1e-3, betas=(0.9,0.999))"),
    ("MultiStep", "lr_scheduler.MultiStepLR(opt, milestones=[15,30], gamma=0.5)"),
    ("Cosine",    "lr_scheduler.CosineAnnealingLR(opt, T_max=50)"),
    ("Exponential","lr_scheduler.ExponentialLR(opt, gamma=0.95)"),
]
for name, api in apis:
    print(f"  {name:<12}  {api}")

```
