import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参
batch_size = 4
context_length = 16  # 一个句子的token数
d_model = 64  # 每个token的维度
num_layers = 8  # transformer模块数
num_heads = 4  # 注意力头数 # 我们的代码中通过 d_model / num_heads = 来获取 head_size
num_blocks = 8  # transformer模块数
learning_rate = 1e-3  #
dropout = 0.1 #
max_iters = 5000  # 训练回合数
eval_interval = 50  # 验证间隔
eval_iters = 20  #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)



# 用于前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# 注意力机制
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        # 输入先要乘的参数，这也是注意力层要学的东西
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps(current context_length), Channels(dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # 注意力机制计算
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # softmax
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        # 这里是聚合后再多过的一层线性层
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # 这里跟原文不太一样，这里是现在一般使用的思路，把层归一化提前
        # LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # 残差链接
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # 残差链接
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        # 设置 word embedding表
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # 这里跟原文不太一样，这里最后加了一个LayerNorm模块
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))

        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        """这里训练的时候两个值都会传入，推理的时候是只传idx的，这里的idx就是输入句子，targets"""
        B, T = idx.shape # Batch size, Time steps(current context_length)

        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # 获得句子对应的哪些position embedding
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # 这里的logits就是输出结果没过softmax时候的样子，因为后面cross_entropy会计算
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            # 这里不理解可以看一下cross_entropy的用法
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """给一段句子，让它给出它预测的下一个词"""
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 获得训练数据
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y



# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    # 下载数据集
    if not os.path.exists('data/sales_textbook.txt'):
        url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
        with open('data/sales_textbook.txt', 'w') as f:
            f.write(requests.get(url).text)

    with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 将数据集文本token化
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_text = encoding.encode(text)
    max_token_value = max(tokenized_text) + 1
    tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

    # 划分训练集和验证集
    split_idx = int(len(tokenized_text) * 0.9)
    train_data = tokenized_text[:split_idx]
    val_data = tokenized_text[split_idx:]


    # 初始化
    model = TransformerLanguageModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    tracked_losses = list()
    for step in range(max_iters):
        if step % eval_iters == 0 or step == max_iters - 1:
            losses = estimate_loss()
            tracked_losses.append(losses)
            print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                  round(losses['valid'].item(), 3))

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model state dictionary
    torch.save(model.state_dict(), 'model-ckpt.pt')

    # 验证
    model.eval()
    start = 'The salesperson'
    start_ids = encoding.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = model.generate(x, max_new_tokens=100)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')