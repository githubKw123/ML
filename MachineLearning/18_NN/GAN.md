---
title: 生成对抗网络（GAN）
tags:
  - 深度学习
  - 生成模型
  - GAN
  - 对抗训练
aliases:
  - GAN
  - Generative Adversarial Network
  - 生成对抗网络
created: 2026-03-17
---

# 生成对抗网络（GAN）

## 1. 核心思想与直觉

> [!abstract] 核心思想
> GAN（Generative Adversarial Network）由 Ian Goodfellow 于 2014 年提出，是一种通过**对抗训练**来学习数据分布的生成模型框架。其核心思想是：让两个神经网络——**生成器 G** 和 **判别器 D** ——互相博弈，最终使生成器学会产生以假乱真的数据。

> [!tip] "伪造者 vs 警察" 类比
> 想象一个造假币的人（生成器 G）和一个验钞的警察（判别器 D）：
> - **造假者**不断尝试制造更逼真的假币
> - **警察**不断提升辨别真假币的能力
> - 随着博弈的进行，造假者的技术越来越高超，警察也越来越敏锐
> - **最终均衡**：造假者制造的假币与真币完全无法区分，警察对任何一张钞票的判断概率都是 $\frac{1}{2}$（即完全猜不出真假）

### 为什么 GAN 重要？

1. **隐式密度估计**：不需要显式定义数据分布的形式（不像 [[MachineLearning/18_NN/VAE|VAE]] 需要假设先验），直接通过对抗过程学习
2. **生成质量极高**：在图像生成领域，GAN 能产生极其逼真的样本
3. **框架通用性强**：GAN 的思想可以应用于图像、文本、音频等多种数据模态
4. **理论优美**：与博弈论、信息论有深刻联系

---

## 2. 网络架构

### 整体框架

```
噪声 z ~ p_z(z) ──→ [生成器 G] ──→ 生成样本 G(z) ──┐
                                                       ├──→ [判别器 D] ──→ D(·) ∈ (0,1)
真实数据 x ~ p_data(x) ──────────────────────────────┘
```

### 生成器 G（Generator）

| 属性 | 描述 |
|------|------|
| **角色** | "造假者"——从噪声中生成逼真的数据 |
| **输入** | 随机噪声向量 $z \sim p_z(z)$，通常 $z \sim \mathcal{N}(0, I)$ 或 $z \sim \text{Uniform}(-1, 1)$ |
| **输出** | 生成的假样本 $G(z)$，与真实数据具有相同的维度 |
| **目标** | 让生成的样本尽可能"骗过"判别器，即使 $D(G(z))$ 尽可能接近 1 |
| **参数** | $\theta_g$（通过反向传播更新） |

生成器定义了一个从隐空间到数据空间的映射：

$$
G: \mathbb{R}^{d_z} \rightarrow \mathbb{R}^{d_x}
$$

其中 $d_z$ 是噪声维度，$d_x$ 是数据维度。生成器在数据空间上隐式地定义了一个概率分布 $p_g$：如果 $z \sim p_z$，则 $G(z) \sim p_g$。

### 判别器 D（Discriminator）

| 属性 | 描述 |
|------|------|
| **角色** | "警察"——判断输入数据是真实的还是生成的 |
| **输入** | 一个样本 $x$（可能是真实数据，也可能是生成数据 $G(z)$）|
| **输出** | 一个标量 $D(x) \in (0, 1)$，表示输入为真实数据的概率 |
| **目标** | 正确区分真假：对真实数据输出高概率，对生成数据输出低概率 |
| **参数** | $\theta_d$（通过反向传播更新） |

判别器定义了一个从数据空间到概率的映射：

$$
D: \mathbb{R}^{d_x} \rightarrow (0, 1)
$$

---

## 3. 核心数学推导

### 3.1 Minimax 目标函数

GAN 的训练被表述为一个**极小极大（minimax）博弈**：

$$
\min_G \max_D V(D, G)
$$

其中价值函数 $V(D, G)$ 定义为：

$$
\boxed{V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]}
$$

> [!note] 直觉理解
> - **判别器**想要**最大化** $V$：
>   - 对真实数据 $x$，希望 $D(x) \to 1$，使 $\log D(x) \to 0$（最大值）
>   - 对生成数据 $G(z)$，希望 $D(G(z)) \to 0$，使 $\log(1 - D(G(z))) \to 0$（最大值）
> - **生成器**想要**最小化** $V$：
>   - 希望 $D(G(z)) \to 1$，使 $\log(1 - D(G(z))) \to -\infty$（最小值）
>   - 生成器不影响第一项 $\mathbb{E}[\log D(x)]$

将第二项中的 $z$ 替换为 $x = G(z)$，利用 $p_g$ 的定义，目标函数可以等价地写为：

$$
V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D(x))]
$$

进一步写成积分形式：

$$
V(D, G) = \int_x p_{\text{data}}(x) \log D(x) \, dx + \int_x p_g(x) \log(1 - D(x)) \, dx
$$

合并为一个积分：

$$
V(D, G) = \int_x \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx
$$

### 3.2 最优判别器 $D^*$（固定 G）

> [!important] 定理：最优判别器
> 给定固定的生成器 $G$，最优判别器为：
> $$D_G^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**推导过程：**

对于固定的 $G$，我们要对 $D$ 最大化 $V(D, G)$。对被积函数中的 $D(x)$ 逐点求最大值即可。

对于每一个固定的 $x$，令 $a = p_{\text{data}}(x)$，$b = p_g(x)$，$y = D(x) \in (0, 1)$，则需要最大化：

$$
f(y) = a \log y + b \log(1 - y)
$$

对 $y$ 求导并令其为零：

$$
f'(y) = \frac{a}{y} - \frac{b}{1 - y} = 0
$$

$$
\frac{a}{y} = \frac{b}{1 - y}
$$

$$
a(1 - y) = by
$$

$$
a = y(a + b)
$$

$$
y^* = \frac{a}{a + b}
$$

验证这是最大值（二阶导数检验）：

$$
f''(y) = -\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0
$$

因此 $f''(y^*) < 0$，确认 $y^*$ 是最大值点。

代回原变量，得到最优判别器：

$$
\boxed{D_G^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}}
$$

> [!tip] 直觉
> 最优判别器的输出就是**贝叶斯最优分类器**的后验概率：在等先验假设下，观测到 $x$ 后它来自真实数据的概率。

### 3.3 最优判别器下的目标函数与 JS 散度

> [!important] 定理：与 Jensen-Shannon 散度的关系
> 当判别器取最优 $D^*$ 时，GAN 的目标函数等价于：
> $$C(G) = V(D_G^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$
> 其中 $\text{JSD}$ 是 Jensen-Shannon 散度。

**推导过程：**

将 $D_G^*(x)$ 代入目标函数：

$$
C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]
$$

**第一步：引入常数 $-\log 4$**

注意到 $-\log 4 = -\log 2 - \log 2$，我们将其拆分并分别加减到两项中：

$$
C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]
$$

在每个对数项的分母上乘除 $2$：

$$
= \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{2 \cdot p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \cdot \frac{1}{2}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{2 \cdot p_g(x)}{p_{\text{data}}(x) + p_g(x)} \cdot \frac{1}{2}\right]
$$

$$
= \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{2 \cdot p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} - \log 2\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{2 \cdot p_g(x)}{p_{\text{data}}(x) + p_g(x)} - \log 2\right]
$$

由于 $\mathbb{E}_{x \sim p_{\text{data}}}[\log 2] = \log 2$ 且 $\mathbb{E}_{x \sim p_g}[\log 2] = \log 2$（常数的期望等于自身），得到：

$$
= -\log 2 - \log 2 + \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{2 \cdot p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{2 \cdot p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]
$$

$$
= -\log 4 + \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}}\right]
$$

**第二步：识别 KL 散度**

令 $m(x) = \frac{p_{\text{data}}(x) + p_g(x)}{2}$（两个分布的混合分布），则：

$$
C(G) = -\log 4 + \text{KL}\left(p_{\text{data}} \,\Big\|\, m\right) + \text{KL}\left(p_g \,\Big\|\, m\right)
$$

**第三步：识别 Jensen-Shannon 散度**

回忆 Jensen-Shannon 散度的定义：

$$
\text{JSD}(p \| q) = \frac{1}{2}\text{KL}\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2}\text{KL}\left(q \,\Big\|\, \frac{p+q}{2}\right)
$$

因此：

$$
\text{KL}(p_{\text{data}} \| m) + \text{KL}(p_g \| m) = 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)
$$

最终：

$$
\boxed{C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)}
$$

### 3.4 全局最优解证明：$p_g = p_{\text{data}}$

> [!important] 定理：全局最优
> $C(G)$ 的全局最小值当且仅当 $p_g = p_{\text{data}}$ 时取到，此时 $C(G) = -\log 4$。

**证明：**

**（1）JSD 的非负性**

Jensen-Shannon 散度满足：

$$
\text{JSD}(p \| q) \geq 0
$$

且等号成立当且仅当 $p = q$。

> [!note] 为什么 JSD $\geq 0$？
> 因为 $\text{JSD}$ 是两个 KL 散度的非负线性组合，而 KL 散度由 Gibbs 不等式保证非负：$\text{KL}(p\|q) \geq 0$，等号成立当且仅当 $p = q$。

**（2）最小值条件**

由于：

$$
C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g) \geq -\log 4
$$

等号成立当且仅当：

$$
\text{JSD}(p_{\text{data}} \| p_g) = 0 \iff p_g = p_{\text{data}}
$$

**（3）最优解下的判别器**

当 $p_g = p_{\text{data}}$ 时，最优判别器变为：

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{1}{2}
$$

这正好对应了我们的直觉：**当生成分布完全匹配真实分布时，判别器完全无法区分真假，对任何输入都输出 $\frac{1}{2}$**。

> [!summary] 推导总结
> 1. 固定 $G$，最优判别器 $D^*$ 是贝叶斯最优分类器
> 2. 代入 $D^*$ 后，目标函数变为 JS 散度（加常数）
> 3. JS 散度非负，等于零当且仅当两个分布相同
> 4. 因此全局最优为 $p_g = p_{\text{data}}$，此时 $D^* = \frac{1}{2}$

---

## 4. 训练算法

### 交替优化（Alternating Optimization）

GAN 的训练采用交替优化策略，在每次迭代中：

> [!abstract] GAN 训练算法
> **输入**：学习率 $\alpha$，判别器训练步数 $k$（通常 $k=1$）
>
> **for** 每个训练迭代 **do**：
>
> &emsp; **--- 第一步：训练判别器（重复 $k$ 次）---**
>
> &emsp; **for** $i = 1, \dots, k$ **do**：
>
> &emsp;&emsp; 1. 从噪声先验中采样 $m$ 个噪声样本：$\{z^{(1)}, \dots, z^{(m)}\} \sim p_z(z)$
>
> &emsp;&emsp; 2. 从真实数据中采样 $m$ 个样本：$\{x^{(1)}, \dots, x^{(m)}\} \sim p_{\text{data}}(x)$
>
> &emsp;&emsp; 3. 通过**梯度上升**更新判别器参数 $\theta_d$：
>
> $$\theta_d \leftarrow \theta_d + \alpha \cdot \nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^{m} \left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]$$
>
> &emsp; **end for**
>
> &emsp; **--- 第二步：训练生成器（1 次）---**
>
> &emsp; 1. 从噪声先验中采样 $m$ 个噪声样本：$\{z^{(1)}, \dots, z^{(m)}\} \sim p_z(z)$
>
> &emsp; 2. 通过**梯度下降**更新生成器参数 $\theta_g$：
>
> $$\theta_g \leftarrow \theta_g - \alpha \cdot \nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^{m} \log(1 - D(G(z^{(i)})))$$
>
> **end for**

> [!warning] 实践中的技巧
> 在训练早期，$G$ 很差，$D$ 能轻松区分真假，导致 $\log(1 - D(G(z)))$ 接近 $\log 1 = 0$，梯度消失。
>
> **解决方案**：生成器不最小化 $\log(1 - D(G(z)))$，而是**最大化** $\log D(G(z))$。
>
> 这不改变最优解，但在训练早期提供了更强的梯度信号（非饱和损失，Non-Saturating Loss）。

---

## 6. 重要 GAN 变体

### 6.1 DCGAN（Deep Convolutional GAN, 2015）

> [!info] 核心贡献
> **解决的问题**：原始 GAN 使用全连接网络，生成图像质量有限且训练不稳定。

**关键创新**：
- 用**卷积层**替代全连接层（生成器用转置卷积，判别器用步幅卷积）
- 去除池化层，使用步幅卷积进行上/下采样
- 使用**批归一化（Batch Normalization）**稳定训练
- 生成器使用 ReLU（最后一层 Tanh），判别器使用 LeakyReLU

**意义**：确立了 GAN 的卷积网络架构范式，是后续几乎所有图像 GAN 的基础。

### 6.2 WGAN（Wasserstein GAN, 2017）

> [!info] 核心贡献
> **解决的问题**：JS 散度在分布不重叠时梯度为零，导致训练不稳定和模式崩塌。

**关键创新**：用 **Wasserstein-1 距离**（Earth-Mover Distance）替代 JS 散度：

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]
$$

通过 Kantorovich-Rubinstein 对偶性，WGAN 的目标函数变为：

$$
\max_{D \in \text{1-Lipschitz}} \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

- 判别器改称为 **Critic**（输出不再需要限制在 $(0,1)$，去掉 sigmoid）
- 通过**权重裁剪（Weight Clipping）**或**梯度惩罚（WGAN-GP）**来强制 Lipschitz 约束
- 训练更稳定，损失函数值与生成质量正相关（可用于监控训练）

### 6.3 CGAN（Conditional GAN, 2014）

> [!info] 核心贡献
> **解决的问题**：原始 GAN 无法控制生成内容的类别或属性。

**关键创新**：在 G 和 D 中都加入**条件信息** $y$（如类别标签）：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]
$$

- 生成器输入变为 $(z, y)$，可以生成指定类别的样本
- 判别器输入变为 $(x, y)$，判断样本在给定条件下是否真实
- 典型应用：生成指定数字（MNIST）、文本到图像生成

### 6.4 StyleGAN（2018-2021）

> [!info] 核心贡献
> **解决的问题**：缺乏对生成图像不同层次特征（姿态、面部结构、纹理等）的精细控制。

**关键创新**：
- **Mapping Network**：将噪声 $z$ 映射到中间隐空间 $w = f(z)$，解耦不同的变异因素
- **Adaptive Instance Normalization (AdaIN)**：在生成器的每一层注入风格信息 $w$
- **风格混合（Style Mixing）**：在不同分辨率层使用不同的 $w$，分离控制粗粒度和细粒度特征
- **噪声注入**：在每一层添加随机噪声，控制随机性细节（如头发丝、皮肤纹理）
- StyleGAN2 引入了权重解调（Weight Demodulation）进一步提升质量
- StyleGAN3 解决了"纹理粘连"问题

**意义**：生成了史上最逼真的人脸图像，开创了可控高分辨率图像生成的时代。

### 6.5 CycleGAN（2017）

> [!info] 核心贡献
> **解决的问题**：图像到图像翻译通常需要**成对**训练数据（如照片-油画配对），但这种配对数据极难获取。

**关键创新**：引入**循环一致性损失（Cycle Consistency Loss）**，实现**无配对**的图像翻译：

$$
\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_X}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_Y}[\|G(F(y)) - y\|_1]
$$

- 使用两个生成器：$G: X \to Y$ 和 $F: Y \to X$
- **循环一致性**：$x \to G(x) \to F(G(x)) \approx x$（翻译过去再翻译回来应该得到原图）
- 典型应用：马 $\leftrightarrow$ 斑马、夏天 $\leftrightarrow$ 冬天、照片 $\leftrightarrow$ 油画

### 6.6 其他值得关注的变体

| 变体 | 年份 | 核心思想 |
|------|------|----------|
| **InfoGAN** | 2016 | 通过最大化互信息学习可解释的隐表示 |
| **Progressive GAN** | 2017 | 渐进式增长分辨率，训练更稳定 |
| **BigGAN** | 2018 | 大规模训练（大 batch size + 大模型），在 ImageNet 上生成高质量图像 |
| **Pix2Pix** | 2016 | 有配对数据的图像到图像翻译 |
| **StarGAN** | 2018 | 多域图像翻译（一个模型处理多种属性转换）|

---

## 参考文献

1. Goodfellow, I. et al. "Generative Adversarial Nets." NeurIPS, 2014.
2. Radford, A. et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." (DCGAN) ICLR, 2016.
3. Arjovsky, M. et al. "Wasserstein GAN." ICML, 2017.
4. Mirza, M. & Osindero, S. "Conditional Generative Adversarial Nets." arXiv, 2014.
5. Karras, T. et al. "A Style-Based Generator Architecture for Generative Adversarial Networks." (StyleGAN) CVPR, 2019.
6. Zhu, J. et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." (CycleGAN) ICCV, 2017.
7. Gulrajani, I. et al. "Improved Training of Wasserstein GANs." (WGAN-GP) NeurIPS, 2017.
