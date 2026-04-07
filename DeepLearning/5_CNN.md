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

# 第5章 卷积神经网络（CNN）

卷积神经网络是专为处理具有**空间结构数据**（图像、序列）而设计的网络架构。
本章从卷积操作的数学原理出发，逐步构建出第一个完整的 CNN——LeNet。

> 参考：[动手学深度学习 v2 · 第6章](https://zh-v2.d2l.ai/chapter_convolutional-neural-networks/index.html)

```python
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import time

print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
```

---
### 5.1 从全连接层到卷积

对于一张 $100\times100$ 的 RGB 图像，输入维度为 $30000$。
若第一个全连接隐藏层有 $1000$ 个单元，则需要 $3\times10^7$ 个参数，而实际图像往往是百万像素级别——参数量完全不可承受。那么怎么设计神经网络才能有效地处理图像数据？我们认为一个有效的设计应该在处理图像时应该满足以下两个特性：



**1. 平移不变性（Translation Invariance）**

无论猫出现在图像的左上角还是右下角，检测猫的特征应该产生相同的响应。
数学上，若权重不依赖于绝对坐标 $(i, j)$，只与偏移量 $(a, b)$ 有关：

$$[\mathbf{H}]_{i,j} = u + \sum_a \sum_b [\mathbf{V}]_{a,b} [\mathbf{X}]_{i+a, j+b}$$

这就是二维卷积的形式，参数从 $n^4$ 降到了卷积核大小 $k^2$。

**2. 局部性（Locality）**

早期特征（边缘、纹理）只与局部像素有关。限制偏移量 $|a|, |b| \leq \Delta$，只看半径 $\Delta$ 内的邻域：

$$[\mathbf{H}]_{i,j} = u + \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} [\mathbf{V}]_{a,b} [\mathbf{X}]_{i+a, j+b}$$

> 这两条性质将参数量从**数十亿**压缩到**几百个**，同时赋予网络空间泛化能力。


---
### 5.2 卷积层

基于此我们引入了卷积操作。而深度学习框架实现的实际上是**互相关（cross-correlation）**，而非严格的数学卷积（二者区别仅在于核是否翻转，学习出的核会自动适应，结果等价）。

![](assets\correlation.svg)

*对于输入 $\mathbf{X}$（大小 $n_h \times n_w$）和卷积核 $\mathbf{K}$（大小 $k_h \times k_w$）：

$$[\mathbf{Y}]_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} [\mathbf{X}]_{i+m,\, j+n} \cdot [\mathbf{K}]_{m,n}$$

输出尺寸：$(n_h - k_h + 1) \times (n_w - k_w + 1)$

**这就是卷积层**：对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。 所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。 就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。


**卷积层应用——边缘检测实验**

用一个精心设计的 $1\times2$ 核 $[1, -1]$ 检测图像的**垂直边缘**：
- 相邻像素相同 → 输出 0（无边缘）
- 白到黑 → 输出 +1
- 黑到白 → 输出 -1

```python
def corr2d(X, K):
    """二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 构造黑白竖条图像
X_edge = torch.ones(6, 8)
X_edge[:, 2:6] = 0   # 中间4列为黑色

# 垂直边缘检测核
K_v = torch.tensor([[1.0, -1.0]])
# 水平边缘检测核
K_h = torch.tensor([[1.0], [-1.0]])

Y_v = corr2d(X_edge, K_v)
Y_h = corr2d(X_edge, K_h)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].imshow(X_edge.numpy(), cmap='gray', vmin=0, vmax=1)
axes[0].set_title(' (6×8)', fontsize=11)
axes[0].set_xticks([]); axes[0].set_yticks([])

im1 = axes[1].imshow(Y_v.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('[1,-1] → (6×7)', fontsize=11)
axes[1].set_xticks([]); axes[1].set_yticks([])
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(Y_h.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
axes[2].set_title('[[1],[-1]] → (5×8)', fontsize=11)
axes[2].set_xticks([]); axes[2].set_yticks([])
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.suptitle('experiment', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
```

**学习卷积核**

上面的 $[1,-1]$ 核是人工设计的。实际中，卷积核通过**梯度下降从数据中自动学习**。
下面验证：用随机初始化的核，通过最小化预测误差，能否学到 $[1,-1]$？

```python
# 目标：从 X_edge 的输入-输出对中学习到核 [1, -1]
conv_learn = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X_in = X_edge.reshape(1, 1, 6, 8)         # (N, C, H, W)
Y_in = Y_v.reshape(1, 1, 6, 7)            # 目标输出

optimizer = torch.optim.SGD(conv_learn.parameters(), lr=0.01)
loss_fn   = nn.MSELoss()

loss_hist = []
for epoch in range(50):
    optimizer.zero_grad()
    loss = loss_fn(conv_learn(X_in), Y_in)
    loss.backward()
    optimizer.step()
    loss_hist.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch+1:2d},  loss: {loss.item():.6f},  '
              f'kernel: {conv_learn.weight.data.squeeze().tolist()}')

print(f'\n学到的核: {conv_learn.weight.data.squeeze().tolist()}')
print(f'目标核:   [1.0, -1.0]')
```

```python
plt.figure(figsize=(6, 3))
plt.plot(loss_hist)
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('learned kernel [1, -1]')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- **特征映射（Feature Map）**：卷积层的输出，表示输入在该层学到的特征的空间分布
- **感受野（Receptive Field）**：影响某个输出元素的所有输入区域。随着网络加深，深层神经元的感受野指数级增大，能「看到」更大的上下文


---
### 5.3 填充与步幅

**填充（Padding）**

**每次卷积后，输出的高/宽各减少 $k-1$。多层叠加后图像越来越小，边缘信息丢失。
**填充**在输入四周补零，恢复输出尺寸。

添加 $p_h$ 行、$p_w$ 列填充后，输出尺寸为：

$$\text{输出} = (n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)$$

令 $p_h = k_h - 1$、$p_w = k_w - 1$（通常 $p = \lfloor k/2 \rfloor$，要求 $k$ 为奇数），则**输出与输入同形**。

```python
def comp_conv2d(conv2d_layer, X):
    """包装成 (H, W) -> (H, W) 的形式方便观察形状"""
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d_layer(X)
    return Y.reshape(Y.shape[2:])

X_8 = torch.rand(8, 8)

print('='*55)
print(f'输入: 8×8')
print('='*55)

# 无填充
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=3, padding=0), X_8)
print(f'padding=0, kernel=3×3  →  输出: {tuple(Y.shape)}')

# padding=1 保持尺寸
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=3, padding=1), X_8)
print(f'padding=1, kernel=3×3  →  输出: {tuple(Y.shape)}  ← 输入输出同形')

# padding=2 配合 5×5 核
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=5, padding=2), X_8)
print(f'padding=2, kernel=5×5  →  输出: {tuple(Y.shape)}  ← 输入输出同形')

# 非对称填充
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1)), X_8)
print(f'padding=(0,1), kernel=3×5 → 输出: {tuple(Y.shape)}')
```

**步幅（Stride）**

卷积窗口默认每次滑动 1 个像素。设置步幅 $s > 1$ 可以**跳步滑动**，快速降采样：

$$\text{输出} = \left\lfloor \frac{n_h - k_h + p_h + s_h}{s_h} \right\rfloor \times \left\lfloor \frac{n_w - k_w + p_w + s_w}{s_w} \right\rfloor$$

步幅为 2 时，输出尺寸约减半；步幅为 2 且 padding = kernel//2 时，精确减半。

```python
print('输入: 8×8')
print('='*55)

# 步幅=2 降采样
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2), X_8)
print(f'stride=2, padding=1, kernel=3×3 → 输出: {tuple(Y.shape)}')

# 步幅=3
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=3, padding=0, stride=3), X_8)
print(f'stride=3, padding=0, kernel=3×3 → 输出: {tuple(Y.shape)}')

# 非对称步幅
Y = comp_conv2d(nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4)), X_8)
print(f'stride=(3,4), kernel=(3,5)      → 输出: {tuple(Y.shape)}')
```

```python
# 可视化填充和步幅对输出尺寸的影响
n, k = 8, 3
paddings = range(0, 4)
strides  = [1, 2, 3, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for s in strides:
    sizes = [max(0, (n - k + p + s) // s) for p in paddings]
    axes[0].plot(paddings, sizes, marker='o', label=f'stride={s}')
axes[0].set_xlabel('padding'); axes[0].set_ylabel('output size')
axes[0].set_title(f'input size={n}, kernel size={k}×{k}:padding effect')
axes[0].axhline(n, color='gray', ls='--', lw=1, label='input size')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

for p in paddings:
    sizes = [max(0, (n - k + p + s) // s) for s in strides]
    axes[1].plot(strides, sizes, marker='o', label=f'padding={p}')
axes[1].set_xlabel('stride'); axes[1].set_ylabel('output size')
axes[1].set_title(f'input size={n}, kernel size={k}×{k}:stride effect')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---
### 5.4 多输入/输出通道

**多输入通道**

![](assets\conv-multi-in.svg)

彩色图像有 RGB 三个通道。对 $c_i$ 个输入通道，卷积核扩展为 $c_i \times k_h \times k_w$，对每个通道分别做互相关后求和：

$$[\mathbf{Y}]_{i,j} = \sum_{c=1}^{c_i} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} [\mathbf{X}]_{c,\, i+m,\, j+n} \cdot [\mathbf{K}]_{c,m,n}$$


**多输出通道**

为学习多种特征（如水平边缘、垂直边缘、颜色斑点……），需要 $c_o$ 个不同的卷积核，每个生成一个输出通道。
完整的卷积核张量形状为 $c_o \times c_i \times k_h \times k_w$。

<!-- #region -->
**1×1 卷积**

$1\times1$ 卷积不捕捉空间特征，而是在**通道维度**上做全连接变换：

![](assets\conv-1x1.svg)


- 用途：通道数升维/降维（控制计算量）、跨通道特征融合
- 等价于：对每个空间位置 $(i,j)$ 独立地做一次 $c_i \to c_o$ 的线性变换
<!-- #endregion -->

---
### 6.5 池化层

池化层（Pooling）有两个目的：
1. **降低对位置的敏感性**：目标轻微移动不影响高层特征
2. **空间降采样**：减少后续层的计算量

池化层**没有可学习参数**，是固定的统计操作。可以对窗口内的元素取**最大值**或**平均值**：

```python
X_torch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
print('输入 (1,1,4,4):\n', X_torch.squeeze().numpy())

# 默认 stride = pool_size
print('\n最大池化 2×2（默认 stride=2）→',
      nn.MaxPool2d(2)(X_torch).shape)
print(nn.MaxPool2d(2)(X_torch).squeeze().numpy())

# 自定义 padding 和 stride
print('\n最大池化 3×3, padding=1, stride=2 →',
      nn.MaxPool2d(3, padding=1, stride=2)(X_torch).shape)

# 非正方形窗口
print('\n最大池化 (2,3), padding=(0,1), stride=(2,3) →',
      nn.MaxPool2d((2, 3), padding=(0, 1), stride=(2, 3))(X_torch).shape)
```

```python
# 多通道池化：对每个通道独立处理
X_mc = torch.cat([X_torch, X_torch + 1], dim=1)  # (1, 2, 4, 4)
print('多通道输入形状:', X_mc.shape)
out_mc = nn.MaxPool2d(3, padding=1, stride=2)(X_mc)
print('多通道池化输出形状:', out_mc.shape)   # 通道数不变
print('\n通道0 输出:\n', out_mc[0, 0].numpy())
print('\n通道1 输出:\n', out_mc[0, 1].numpy())
```

<!-- #region -->
---
### 6.6 卷积神经网络（LeNet）


LeNet-5（LeCun et al., 1998）是**第一个被大规模成功应用**的卷积神经网络，用于手写数字识别（MNIST）。它奠定了 CNN 的经典结构：**卷积 → 池化 → 卷积 → 池化 → 全连接**。


LeNet 由两部分组成：

**① 卷积编码器（提取空间特征）**

| 层 | 操作 | 输入尺寸 | 输出尺寸 |
|---|------|---------|--------|
| Conv1 | 6 个 5×5 核，padding=2，Sigmoid | 1×28×28 | 6×28×28 |
| Pool1 | 平均池化 2×2，stride=2 | 6×28×28 | 6×14×14 |
| Conv2 | 16 个 5×5 核，Sigmoid | 6×14×14 | 16×10×10 |
| Pool2 | 平均池化 2×2，stride=2 | 16×10×10 | 16×5×5 |

**② 全连接分类器**

| 层 | 操作 | 输入 | 输出 |
|---|------|------|------|
| Flatten | 展平 | 16×5×5 | 400 |
| FC1 | Linear + Sigmoid | 400 | 120 |
| FC2 | Linear + Sigmoid | 120 | 84 |
| FC3 | Linear | 84 | 10 |
<!-- #endregion -->

```python
lenet = nn.Sequential(
    # 卷积块1
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 卷积块2
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 全连接分类器
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

total_params = sum(p.numel() for p in lenet.parameters())
print(f'LeNet 总参数量: {total_params:,}')
```

```python
# 逐层查看输出形状
X_demo = torch.randn(1, 1, 28, 28)
layer_names = [
    'Conv1(6,5×5,p=2)', 'Sigmoid', 'AvgPool1(2×2,s=2)',
    'Conv2(16,5×5)',     'Sigmoid', 'AvgPool2(2×2,s=2)',
    'Flatten',
    'Linear(400→120)',  'Sigmoid',
    'Linear(120→84)',   'Sigmoid',
    'Linear(84→10)'
]

print(f'{"层名":<25} {"输出形状"}')
print('-' * 45)
print(f'{"输入":<25} {tuple(X_demo.shape)}')
for name, layer in zip(layer_names, lenet):
    X_demo = layer(X_demo)
    print(f'{name:<25} {tuple(X_demo.shape)}')
```

**在 Fashion-MNIST 上训练**

```python
# 加载 Fashion-MNIST 数据集
batch_size = 256
trans = transforms.ToTensor()

train_set = torchvision.datasets.FashionMNIST(
    root='../data', train=True,  transform=trans, download=True)
test_set  = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=True)

train_iter = data.DataLoader(train_set, batch_size, shuffle=True,  num_workers=0)
test_iter  = data.DataLoader(test_set,  batch_size, shuffle=False, num_workers=0)

print(f'训练集: {len(train_set)} 张  测试集: {len(test_set)} 张')
```

```python
def evaluate_accuracy_gpu(net, data_iter, device):
    """在 GPU 上评估准确率"""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total   += y.numel()
    return correct / total

def train_lenet(net, train_iter, test_iter, num_epochs, lr, device):
    net.to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    history   = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()
            total_loss += l.item() * y.numel()
            correct    += (y_hat.argmax(dim=1) == y).sum().item()
            total      += y.numel()

        tr_loss = total_loss / total
        tr_acc  = correct / total
        te_acc  = evaluate_accuracy_gpu(net, test_iter, device)
        elapsed = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        print(f'epoch {epoch+1:2d}  loss {tr_loss:.4f}  '
              f'train {tr_acc:.3f}  test {te_acc:.3f}  '
              f'{total/elapsed:.0f} ex/s')

    return history

lenet = lenet.to(device)
history = train_lenet(lenet, train_iter, test_iter,
                      num_epochs=10, lr=0.9, device=device)
```

```python
# 训练曲线
epochs = range(1, 11)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history['train_loss'], marker='o', color='steelblue')
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_acc'], marker='o', label='train')
axes[1].plot(epochs, history['test_acc'],  marker='s', label='test')
axes[1].set_title('Classification Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('LeNet in Fashion-MNIST', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
```

**卷积核可视化**

```python
# 可视化 Conv1 学到的 6 个卷积核
kernels = lenet[0].weight.data.cpu()
print(f'Conv1 卷积核形状: {kernels.shape}  (c_out=6, c_in=1, 5×5)')

fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
for i, ax in enumerate(axes):
    k = kernels[i, 0].numpy()
    im = ax.imshow(k, cmap='RdBu_r',
                   vmin=-abs(k).max(), vmax=abs(k).max())
    ax.set_title(f'kernel {i+1}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.08)

plt.suptitle('LeNet Conv1 Kernels', fontsize=12, y=1.05)
plt.tight_layout()
plt.show()
```

```python
# 可视化一张测试图片经过 Conv1+Pool1 后的 6 张特征图
lenet.eval()
X_sample, y_sample = next(iter(test_iter))
X_one = X_sample[0:1].to(device)    # 取第一张

labels_text = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

with torch.no_grad():
    # 前向传播到 Pool1 输出（前3层）
    feat = X_one
    for layer in list(lenet.children())[:3]:
        feat = layer(feat)

feat_maps = feat.squeeze().cpu()   # (6, 14, 14)

fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
axes[0].imshow(X_one.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title(f'Input\n({labels_text[y_sample[0]]})', fontsize=9)
axes[0].set_xticks([]); axes[0].set_yticks([])

for i in range(6):
    axes[i+1].imshow(feat_maps[i].numpy(), cmap='viridis')
    axes[i+1].set_title(f'feature map {i+1}', fontsize=9)
    axes[i+1].set_xticks([]); axes[i+1].set_yticks([])

plt.suptitle('input → Conv1+Pool1 feature maps\n6 (14,14)',
             fontsize=11, y=1.05)
plt.tight_layout()
plt.show()
```

**预测结果可视化**

```python
lenet.eval()
with torch.no_grad():
    preds = lenet(X_sample.to(device)).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 9, figsize=(15, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_sample[i].squeeze().numpy(), cmap='gray')
    pred_label = labels_text[preds[i]]
    true_label = labels_text[y_sample[i]]
    color = 'green' if preds[i] == y_sample[i] else 'red'
    ax.set_title(f'{pred_label}\n({true_label})', fontsize=7, color=color)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('LeNet Classification Results',
             fontsize=11, y=1.03)
plt.tight_layout()
plt.show()
```

<!-- #region -->


### LeNet 的设计哲学

```
卷积层（局部特征提取）→ 池化层（位置鲁棒）
    ↓  重复 N 次
空间分辨率↓  通道数↑  感受野↑
    ↓  Flatten
全连接层（全局分类决策）
```

这一 **「卷积编码器 + 全连接分类器」** 的结构被后来的 AlexNet、VGG、ResNet 等沿用至今。
<!-- #endregion -->
