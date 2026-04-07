---
title: Transformer 架构详解
tags:
  - LLM
  - Transformer
  - 深度学习
  - 注意力机制
created: 2026-04-07
---

# Transformer 架构详解

> [!info] 说明
> 本笔记以 **Decoder** 为例，梳理 Transformer 的核心流程。

![[LLM/assets/img.png]]

## 整体流程概览

1. **输入编码**：将输入文本（如"我是一个"）转换为向量嵌入格式（词嵌入 + 位置编码）。
2. **自注意力计算**：向量组成矩阵，输入自注意力机制（Q、K、V 均来自同一输入），捕捉不同位置间的语义关系。
3. **预测输出**：经过归一化 + 前馈网络，生成下一个字符的概率分布（如：人 0.8，狗 0.2）。
4. **损失计算**：将预测结果与训练数据的真实标签（如"是一个人"）对比，计算损失并反向传播更新权重。
5. **迭代优化**：模型不断更新参数，逐步提高预测精度。

---

## 1. Tokenization（分词）

将输入文本借助词表转化为对应的编码（Token），核心步骤：**拆分 + 转化**。

| 步骤 | 说明 | 示例 |
|------|------|------|
| **拆分** | 单词级、字符级、子词级（BPE 等） | "unhappiness" → "un" + "happiness" |
| **转化** | 根据词库映射为数字 ID（如 tiktoken） | ["我", "是"] → [1234, 5678] |

---

## 2. Word Embeddings（词嵌入）

Tokenization 之后，通过一个 **Token Embedding Look-up Table** 将每个 Token ID 映射为一个高维向量。

- **表的维度**：`词表大小 × 嵌入维度`（初始值随机，经训练后具有语义意义）
- **词表大小**：所有可能的 Token 种类数
- **嵌入维度**：每个词在不同语义维度上的表示

> [!tip] 操作
> Input Embedding = 在查找表中取出每个 Token 对应的行向量，拼接为矩阵。

---

## 3. Positional Encoding（位置编码）

Transformer 没有循环结构，因此需要显式注入位置信息。原论文使用正余弦函数，使每个位置的编码**唯一且平滑**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

最终输入 = **词嵌入矩阵 + 位置编码矩阵**

![[LLM/assets/3.png]]

---

## 4. Transformer Block

![[LLM/assets/6.png]]

### 4.1 多头自注意力（Multi-Head Self-Attention）

核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

> [!important] Transformer 学习的是什么？
> 学习的是**注意力权重矩阵** $W_Q$、$W_K$、$W_V$，即图中蓝色部分。

计算步骤详解：

| 步骤 | 操作 | 说明 |
|------|------|------|
| **① MatMul** | $QK^T$ | Q 与 K 的转置相乘，得到每个字与其他字的**相似度矩阵** |
| **② Scale** | $\div \sqrt{d_k}$ | 缩放因子，防止点积值过大导致 softmax 梯度消失 |
| **③ Mask** | 上三角置为 $-\infty$ | Decoder 中，当前位置只能关注**之前**的位置（因果掩码），mask 后 softmax 输出为 0 |
| **④ Softmax** | 归一化为概率 | 将相似度分数转换为注意力权重（概率分布） |
| **⑤ MatMul** | $\times V$ | 用注意力权重对 V 加权求和，得到上下文感知的表示 |
| **⑥ Concatenate** | 多头拼接 | 如 $d_{\text{model}}=64$ 分为 4 个头，每头处理 16 维，最后拼接回 64 维 |

### 4.2 Layer Normalization（层归一化）

对每个样本的特征维度进行归一化（均值为 0，方差为 1），**稳定训练过程**，加速收敛。

### 4.3 残差连接（Residual Connection）

$$
\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

将子层的输入直接加到输出上，**缓解梯度消失问题**，使深层网络更易训练。

### 4.4 Feed Forward Network（前馈网络）

两层全连接网络 + ReLU 激活，对每个位置独立处理：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

作用：为模型增加非线性变换能力。

### 4.5 Linear + Softmax（输出层）

- **Linear**：将 Transformer 输出映射为 `序列长度 × 词表大小` 的矩阵（logits）
- **Softmax**：将 logits 转换为概率分布，表示词表中每个词作为下一个 Token 的可能性
