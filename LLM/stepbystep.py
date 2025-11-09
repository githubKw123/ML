import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import tiktoken
import torch
import torch.nn as nn

# Hyperparameters
batch_size = 4
context_length = 16  # 一个句子的token数
d_model = 64  # 每个token的维度
num_layers = 8  # transformer模块数
num_heads = 4  # 注意力头数 # 我们的代码中通过 d_model / num_heads = 来获取 head_size
learning_rate = 1e-3  #
dropout = 0.1 #
max_iters = 5000  # 训练回合数
eval_interval = 50  # 验证间隔
eval_iters = 20  #
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Instead of using the cpu, we'll use the GPU if it's available.

TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# 下载文本数据
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 对整个文本token化
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text) # 全文有 77,919 个token
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device) # Convert tokens into a tensor
max_token_value = tokenized_text.max().item() # the maximum index value in our vocabulary

# print(encoding.encode('Chapter 1: Building Rapport and Capturing'))
# print(encoding.decode([26072, 220, 16, 25, 17283, 23097, 403, 323, 17013, 1711]))

# 划分数据集和验证集
split_idx = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# 准备训练数据
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])

# 定义embding的表
token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)

# 获得word embding
x = token_embedding_lookup_table(x_batch.data)
y = token_embedding_lookup_table(y_batch.data)

# 定义position embding表
position_encoding_lookup_table = torch.zeros(context_length, d_model) # initial with zeros with shape (context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1) #add batch to the first dimension

# position和word embding相加
input_embedding_x = x + position_encoding_lookup_table # [4, 16, 64] [batch_size, context_length, d_model]
input_embedding_y = y + position_encoding_lookup_table

# 这里就是最后的输入
X = input_embedding_x





query = key = value = X # [4, 16, 64] [batch_size, context_length, d_model]

# 这里要先乘一个可学习的参数
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

# 分头
Q = Wq(query) #[4, 16, 64]
Q = Q.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

K = Wk(key) #[4, 16, 64]
K = K.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

V = Wv(value) #[4, 16, 64]
V = V.view(batch_size, -1, num_heads, d_model // num_heads)  #[4, 16, 4, 16]

# 转换下维度形状
Q = Q.transpose(1, 2) # [4, 4, 16, 16]
K = K.transpose(1, 2) # [4, 4, 16, 16]
V = V.transpose(1, 2) # [4, 4, 16, 16]

# 计算 Q 和 K^T
attention_score = torch.matmul(Q, K.transpose(-2, -1))

# Scale 除以根下维度数
attention_score = attention_score / math.sqrt(d_model // num_heads)

# mask 掩码
attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]), diagonal=1).bool(), float('-inf')) #[4, 4, 16, 16] [batch_size, num_heads, context_length, context_length]

# Softmax
attention_score = torch.softmax(attention_score, dim=-1)

# 最终输出
A = torch.matmul(attention_score, V) # [4, 4, 16, 16] [batch_size, num_heads, context_length, head_size]

# 转换形状后多头拼接
A = A.transpose(1, 2) # [4, 16, 4, 16] [batch_size, context_length, num_heads, head_size]
A = A.reshape(batch_size, -1, d_model) # [4, 16, 64] [batch_size, context_length, d_model]

# 再乘一参数
Wo = nn.Linear(d_model, d_model)
output = Wo(A) # [4, 16, 64] [batch_size, context_length, d_model]

# 残差处理
output = output + X

# layer层
layer_norm = nn.LayerNorm(d_model)
output_layernorm = layer_norm(output)


# 线性网络层
output = nn.Linear(d_model, d_model * 4)(output_layernorm)
output = nn.ReLU()(output)
output = nn.Linear(d_model * 4, d_model)(output)
output = torch.dropout(output, p=dropout, train=True)

# 再过一遍
output = output + output_layernorm
# Add Layer Normalization
layer_norm = nn.LayerNorm(d_model)
output = layer_norm(output)

# 再过一线形层，将其映射到整个词库上
logits = nn.Linear(d_model, max_token_value+1)(output)
probabilities = torch.softmax(logits, dim=-1)

# 得到预测的字的索引
predicted_index = torch.argmax(logits[0,0]).item()